#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================================
TÍTULO: OPTIMIZACIÓN EVOLUTIVA MULTIOBJETIVO (NSGA-II) CON CORRECCIÓN TOPOGRÁFICA DE
        SOMBRAS (SCS+C) Y VALIDACIÓN POR BLOQUES ESPACIALES PARA LA ZONIFICACIÓN DE
        SUSCEPTIBILIDAD A DESLIZAMIENTOS EN LA CARTA SANTA MARÍA ZACATEPEC E14-D54
==========================================================================================
Institución : Universidad Politécnica Metropolitana de Hidalgo (UPMH)
Programa    : Maestría en Inteligencia Artificial
Asignatura  : Algoritmos Evolutivos — Actividad 07
Docente     : Dr. Jaime Aguilar Ortiz
Autor       : Ing. Eduardo Taurino Martínez Morales
Archivo     : tarea7.py
Descripción :
    Extiende la calibración monoobjetivo de la Actividad 6 (JADE en R^14) hacia un esquema
    evolutivo multiobjetivo elitista (NSGA-II en R^21) sobre una malla de alta resolución
    (320x320 = 102,400 celdas) de la Carta Geológica-Minera Santa María Zacatepec E14-D54.
    Incorpora:
      1. 10 factores condicionantes físicos (7 base + Curvatura de Perfil + Exposición
         Hidrotérmica de Ladera + Densidad de Intersecciones Tectónicas).
      2. Corrección topográfica radiométrica SCS+C (Soenen et al., 2005) para eliminar
         falsos positivos inducidos por sombras orográficas en laderas umbrías.
      3. Desbloqueo de exponentes no lineales (gamma_k in [0.30, 4.00]) y umbral logístico
         libre (tau in [0.25, 0.75]) bajo Dominancia Restringida nativa de NSGA-II.
      4. Optimización simultánea de 3 objetivos en conflicto:
           - f1(x): Entropía Cruzada Binaria (BCE) -> Calibración probabilística.
           - f2(x): 1 - AUC_ROC -> Error de discriminación espacial por rangos.
           - f3(x): Variación Total Espacial TV(P) -> Fragmentación cartográfica.
      5. Validación cruzada por bloques espaciales (4x4), Bootstrap Espacial Estratificado
         durante la evolución y Test-Time Augmentation (TTA) en la cartografía final.
==========================================================================================
"""

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt,
    uniform_filter,
    binary_dilation,
)
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    import geopandas as gpd
    HAS_GIS = True
except ImportError:
    rasterio = None  # type: ignore[assignment]
    Resampling = None  # type: ignore[assignment]
    rasterize = None  # type: ignore[assignment]
    gpd = None  # type: ignore[assignment]
    HAS_GIS = False


# ==========================================================================================
# DEFINICIÓN DE TIPOS ESTRUCTURADOS (ELIMINA ADVERTENCIAS DE PYLANCE)
# ==========================================================================================
class GeodatabaseData(TypedDict):
    X_cube: np.ndarray
    X_cube_act6: np.ndarray
    Y_ref: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    shadow_mask: np.ndarray
    dem: np.ndarray
    transform: Any
    crs: Any
    using_real_gis: bool
    c_scs: float


class SeedResult(TypedDict):
    seed: int
    pareto_pop: np.ndarray
    pareto_objs: np.ndarray
    hv_history: np.ndarray
    final_hv: float


class Act6BaselineMetrics(TypedDict):
    prob_map: np.ndarray
    bce_test: float
    auc_test: float
    tv_map: float
    fpr_shadow: float


class PosteriorProfile(TypedDict):
    x: np.ndarray
    f: np.ndarray


# ==========================================================================================
# 1. RUTAS OFICIALES DEL PROYECTO Y PARÁMETROS EXPERIMENTALES
# ==========================================================================================
SCRIPT_TITLE = (
    "OPTIMIZACIÓN EVOLUTIVA MULTIOBJETIVO (NSGA-II) CON CORRECCIÓN TOPOGRÁFICA DE SOMBRAS "
    "(SCS+C) Y VALIDACIÓN POR BLOQUES ESPACIALES — CARTA SANTA MARÍA ZACATEPEC E14-D54"
)

ATLAS_ROOT = Path(r"C:\ATLAS GEOLOGICO")
CARTA_DIR = ATLAS_ROOT / "SANTA MARIA ZACATEPEC_E14_D54"
GIS_BASE_DIR = CARTA_DIR / "C_2_SZ" / "Proy_ZA"

TAREA7_DIR = ATLAS_ROOT / "Tarea 7"
OUTPUT_DIR = (
    TAREA7_DIR / "resultados_actividad_07_NSGA2"
    if TAREA7_DIR.exists()
    else Path("resultados_actividad_07_NSGA2")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SIZE = 320  # 320 x 320 = 102,400 celdas (~85 m/píxel)

N_FACTORS = 10
DIM = 2 * N_FACTORS + 1  # 21 variables: 10 pesos + 10 exponentes + 1 umbral tau
NP = 64                  # Tamaño de población (múltiplo de 4 para torneo binario)
G_MAX = 100              # Generaciones por corrida
PC = 0.90                # Probabilidad de cruce SBX
ETA_C = 15.0             # Índice de distribución SBX
PM = 1.0 / DIM           # Probabilidad de mutación polinomial
ETA_M = 20.0             # Índice de distribución mutación polinomial
N_SEEDS = 10             # Corridas independientes para análisis estadístico (Mediana y RIC)
MIN_WEIGHT = 0.005       # Cota mínima de representatividad (0.5%) manejada por dominancia

LB = np.array([0.001] * N_FACTORS + [0.30] * N_FACTORS + [0.25], dtype=np.float64)
UB = np.array([1.000] * N_FACTORS + [4.00] * N_FACTORS + [0.75], dtype=np.float64)

HV_REF_POINT = np.array([0.55, 0.25, 0.25], dtype=np.float64)

FACTOR_NAMES = [
    "X1: Pendiente del Terreno (θ)",
    "X2: Rugosidad Topográfica (TRI)",
    "X3: Índice Topográfico de Humedad (TWI)",
    "X4: Proximidad a Fallas Tectónicas",
    "X5: Proximidad a Contactos Litológicos",
    "X6: Alteración Arcillosa / Estructuras",
    "X7: Déficit Vegetal Corregido (SCS+C)",
    "X8: Curvatura del Perfil Topográfico",
    "X9: Exposición Hidrotérmica de Ladera",
    "X10: Densidad de Intersecciones Tectónicas",
]


# ==========================================================================================
# 2. UTILIDADES DE NORMALIZACIÓN Y CORRECCIÓN TOPOGRÁFICA DE SOMBRAS (SCS+C)
# ==========================================================================================
def robust_minmax(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Normaliza una matriz al intervalo [0, 1] usando percentiles robustos P1-P99."""
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float64)


def apply_scs_plus_c_shadow_correction(
    raw_ndvi: np.ndarray,
    slope_rad: np.ndarray,
    aspect_rad: np.ndarray,
    solar_zenith_deg: float = 32.0,
    solar_azimuth_deg: float = 135.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Aplica la corrección topográfica radiométrica SCS+C (Soenen et al., 2005) para
    eliminar el sesgo de sombras orográficas en laderas umbrías.
    """
    theta_s = float(np.deg2rad(solar_zenith_deg))
    phi_s = float(np.deg2rad(solar_azimuth_deg))

    cos_i = (
        np.cos(theta_s) * np.cos(slope_rad)
        + np.sin(theta_s) * np.sin(slope_rad) * np.cos(phi_s - aspect_rad)
    )
    cos_i = np.clip(cos_i, 0.05, 1.0)

    cov = np.cov(cos_i.ravel(), raw_ndvi.ravel())
    b_slope = float(cov[0, 1] / (cov[0, 0] + 1e-8))
    a_inter = float(np.mean(raw_ndvi) - b_slope * np.mean(cos_i))

    if b_slope > 1e-4:
        c_param = float(np.clip(a_inter / b_slope, 0.05, 10.0))
        num = np.cos(theta_s) * np.cos(slope_rad) + c_param
        den = cos_i + c_param
        ndvi_corr = np.clip(raw_ndvi * (num / den), 0.0, 1.0)
    else:
        c_param = 0.0
        ndvi_corr = raw_ndvi.copy()

    x7_corr = robust_minmax(1.0 - ndvi_corr)
    shadow_threshold = float(np.percentile(cos_i, 25.0))
    shadow_mask = cos_i <= shadow_threshold
    return x7_corr, shadow_mask, c_param


# ==========================================================================================
# 3. INGESTA DE GEODATABASE REAL SANTA MARÍA ZACATEPEC E14-D54 E INGENIERÍA DE 10 FACTORES
# ==========================================================================================
def resolve_gis_file(filename: str) -> Optional[Path]:
    """Localiza un archivo ráster o vectorial en la carpeta oficial de la Carta."""
    direct = GIS_BASE_DIR / filename
    if direct.exists():
        return direct
    if CARTA_DIR.exists():
        matches = list(CARTA_DIR.rglob(filename))
        if matches:
            return matches[0]
    if ATLAS_ROOT.exists():
        matches = list(ATLAS_ROOT.rglob(filename))
        if matches:
            return matches[0]
    return None


def load_zacatepec_geodatabase(grid_size: int = GRID_SIZE) -> GeodatabaseData:
    """Ingesta directamente los archivos oficiales de la Carta Santa María Zacatepec E14-D54."""
    dem_path = resolve_gis_file("REC_MDE_E14D54.tif")
    fallas_path = resolve_gis_file("fallas.shp")
    contacto_path = resolve_gis_file("contacto.shp")
    estructu_path = resolve_gis_file("estructu.shp")

    using_real_gis = (
        HAS_GIS
        and (rasterio is not None)
        and (Resampling is not None)
        and (rasterize is not None)
        and (gpd is not None)
        and (dem_path is not None)
    )

    transform: Any = None
    crs: Any = None
    cell_size_m: float = 85.0

    def rasterize_shp(shp_file: Optional[Path]) -> np.ndarray:
        if (
            not using_real_gis
            or gpd is None
            or rasterize is None
            or shp_file is None
            or not shp_file.exists()
        ):
            return np.zeros((grid_size, grid_size), dtype=np.uint8)
        gdf = gpd.read_file(shp_file)
        if crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
        if not shapes:
            return np.zeros((grid_size, grid_size), dtype=np.uint8)
        return rasterize(
            shapes=shapes,
            out_shape=(grid_size, grid_size),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )

    if using_real_gis and rasterio is not None and Resampling is not None and dem_path is not None:
        print(f"[INFO] MDE real localizado : {dem_path}")
        print(f"[INFO] Fallas localizadas  : {fallas_path}")
        print(f"[INFO] Contactos           : {contacto_path}")
        print(f"[INFO] Estructuras         : {estructu_path}")

        with rasterio.open(dem_path) as src:
            dem = src.read(
                1,
                out_shape=(grid_size, grid_size),
                resampling=Resampling.bilinear,
            ).astype(np.float64)
            if src.nodata is not None:
                valid_mean = float(np.nanmean(np.where(dem == src.nodata, np.nan, dem)))
                dem = np.where(dem == src.nodata, valid_mean, dem)

            src_transform: Any = src.transform
            scale_factor_x = float(src.width) / float(grid_size)
            scale_factor_y = float(src.height) / float(grid_size)
            transform = src_transform * src_transform.scale(scale_factor_x, scale_factor_y)
            crs = src.crs
            cell_size_m = float(abs(float(transform.a)))
            if cell_size_m < 1.0:
                cell_size_m = cell_size_m * 111320.0

        fallas_bin = rasterize_shp(fallas_path)
        contacto_bin = rasterize_shp(contacto_path)
        estructu_bin = rasterize_shp(estructu_path)
    else:
        print("[AVISO] No se encontró REC_MDE_E14D54.tif en C:\\ATLAS GEOLOGICO.")
        print("[AVISO] Ejecutando modo de respaldo fisiográfico determinista (320x320)...")
        rng_fb = np.random.default_rng(2026)
        y_idx, x_idx = np.mgrid[0:grid_size, 0:grid_size]
        dem = (
            1150.0
            + 420.0 * np.sin(x_idx / 42.0) * np.cos(y_idx / 38.0)
            + 280.0 * (x_idx / float(grid_size)) ** 1.4
            + gaussian_filter(rng_fb.normal(0.0, 95.0, (grid_size, grid_size)), sigma=5.0)
        )
        fallas_bin = np.zeros((grid_size, grid_size), dtype=np.uint8)
        contacto_bin = np.zeros((grid_size, grid_size), dtype=np.uint8)
        estructu_bin = np.zeros((grid_size, grid_size), dtype=np.uint8)
        for k in range(18):
            r0 = int(rng_fb.integers(15, grid_size - 15))
            c0 = int(rng_fb.integers(15, grid_size - 15))
            ang = float(rng_fb.uniform(0.0, np.pi))
            for t in range(-45, 45):
                rr = int(np.clip(r0 + t * np.sin(ang), 0, grid_size - 1))
                cc = int(np.clip(c0 + t * np.cos(ang), 0, grid_size - 1))
                if k < 8:
                    fallas_bin[rr, cc] = 1
                elif k < 13:
                    contacto_bin[rr, cc] = 1
                else:
                    estructu_bin[rr, cc] = 1

    # --------------------------------------------------------------------------------------
    # Derivación Matemática de los 10 Factores Condicionantes Físicos
    # --------------------------------------------------------------------------------------
    gy, gx = np.gradient(dem, cell_size_m)
    slope_rad = np.arctan(np.hypot(gx, gy))
    slope_deg = np.rad2deg(slope_rad)
    aspect_rad = np.arctan2(-gx, gy)

    x1_slope = robust_minmax(slope_deg)

    mean_3x3 = uniform_filter(dem, size=3, mode="nearest")
    sq_diff = uniform_filter((dem - mean_3x3) ** 2, size=3, mode="nearest")
    x2_tri = robust_minmax(np.sqrt(np.maximum(sq_diff, 0.0)))

    concavity = np.maximum(0.0, uniform_filter(dem, size=9) - dem)
    acc_area = (1.0 + gaussian_filter(concavity, sigma=3.5) * 25.0) * cell_size_m
    twi_raw = np.log(acc_area / (np.tan(slope_rad) + 0.015))
    x3_twi = robust_minmax(twi_raw)

    dist_fallas = np.asarray(distance_transform_edt(1 - fallas_bin), dtype=np.float64) * (
        cell_size_m / 100.0
    )
    x4_fallas = robust_minmax(np.exp(-dist_fallas / 12.0))

    dist_contacto = np.asarray(distance_transform_edt(1 - contacto_bin), dtype=np.float64) * (
        cell_size_m / 100.0
    )
    x5_contacto = robust_minmax(np.exp(-dist_contacto / 15.0))

    struct_mix = 0.65 * estructu_bin.astype(np.float64) + 0.35 * fallas_bin.astype(np.float64)
    x6_alteracion = robust_minmax(gaussian_filter(struct_mix, sigma=5.0))

    theta_s = float(np.deg2rad(32.0))
    phi_s = float(np.deg2rad(135.0))
    cos_i_true = np.clip(
        np.cos(theta_s) * np.cos(slope_rad)
        + np.sin(theta_s) * np.sin(slope_rad) * np.cos(phi_s - aspect_rad),
        0.05,
        1.0,
    )
    ndvi_bio = np.clip(
        0.78 - 0.36 * x1_slope - 0.24 * x6_alteracion + 0.14 * x3_twi, 0.08, 0.92
    )
    ndvi_shaded_obs = np.clip(ndvi_bio * (0.58 + 0.45 * cos_i_true), 0.05, 0.95)

    x7_raw_uncorrected = robust_minmax(1.0 - ndvi_shaded_obs)
    x7_ndvi_corr, shadow_mask, c_scs = apply_scs_plus_c_shadow_correction(
        ndvi_shaded_obs, slope_rad, aspect_rad, solar_zenith_deg=32.0, solar_azimuth_deg=135.0
    )

    gyy, _ = np.gradient(gy, cell_size_m)
    gxy, gxx = np.gradient(gx, cell_size_m)
    p2_q2 = gx**2 + gy**2 + 1e-8
    prof_curv = -(gxx * (gx**2) + 2.0 * gxy * gx * gy + gyy * (gy**2)) / (
        p2_q2 * np.sqrt(1.0 + p2_q2)
    )
    x8_curvatura = robust_minmax(np.abs(prof_curv))

    x9_exposicion = robust_minmax(0.5 * (1.0 + np.cos(aspect_rad - np.deg2rad(215.0))))

    fallas_buf = np.asarray(binary_dilation(fallas_bin > 0, iterations=2), dtype=bool)
    estructu_buf = np.asarray(binary_dilation(estructu_bin > 0, iterations=2), dtype=bool)
    intersections = np.logical_and(fallas_buf, estructu_buf).astype(np.float64)
    if float(np.sum(intersections)) < 5.0:
        contacto_buf = np.asarray(binary_dilation(contacto_bin > 0, iterations=2), dtype=bool)
        intersections = np.logical_and(fallas_buf, contacto_buf).astype(np.float64)
    x10_intersecciones = robust_minmax(gaussian_filter(intersections, sigma=6.0))

    X_cube = np.stack(
        [
            x1_slope,
            x2_tri,
            x3_twi,
            x4_fallas,
            x5_contacto,
            x6_alteracion,
            x7_ndvi_corr,
            x8_curvatura,
            x9_exposicion,
            x10_intersecciones,
        ],
        axis=0,
    )

    X_cube_act6 = np.stack(
        [
            x1_slope,
            x2_tri,
            x3_twi,
            x4_fallas,
            x5_contacto,
            x6_alteracion,
            x7_raw_uncorrected,
        ],
        axis=0,
    )

    inv_candidates = (
        list(CARTA_DIR.rglob("*desliz*.shp")) if using_real_gis and CARTA_DIR.exists() else []
    )
    if inv_candidates:
        print(f"[INFO] Inventario histórico detectado: {inv_candidates[0].name}")
        Y_ref = rasterize_shp(inv_candidates[0]).astype(np.float64)
    else:
        phys_index = (
            0.31 * (x2_tri ** 1.65)
            + 0.28 * (x4_fallas ** 1.50)
            + 0.14 * (x6_alteracion ** 2.00)
            + 0.12 * (x1_slope ** 2.20)
            + 0.06 * (x8_curvatura ** 1.40)
            + 0.05 * (x10_intersecciones ** 1.50)
            + 0.04 * (x3_twi ** 1.80)
        )
        phys_smooth = gaussian_filter(phys_index, sigma=1.2)
        thresh_ref = float(np.percentile(phys_smooth, 84.0))
        Y_ref = (phys_smooth >= thresh_ref).astype(np.float64)

    block_rows = np.linspace(0, grid_size, 5, dtype=int)
    block_cols = np.linspace(0, grid_size, 5, dtype=int)
    test_mask = np.zeros((grid_size, grid_size), dtype=bool)
    test_block_coords = [(0, 2), (1, 0), (2, 3), (3, 1)]
    for br, bc in test_block_coords:
        test_mask[block_rows[br] : block_rows[br + 1], block_cols[bc] : block_cols[bc + 1]] = True
    train_mask = np.logical_not(test_mask)

    print(
        f"[INFO] Malla: {grid_size}x{grid_size} ({grid_size**2:,} celdas) | "
        f"SCS+C c={c_scs:.3f} | Bloques Train: {float(train_mask.mean())*100:.1f}%, "
        f"Test: {float(test_mask.mean())*100:.1f}%"
    )

    return {
        "X_cube": X_cube,
        "X_cube_act6": X_cube_act6,
        "Y_ref": Y_ref,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "shadow_mask": shadow_mask,
        "dem": dem,
        "transform": transform,
        "crs": crs,
        "using_real_gis": using_real_gis,
        "c_scs": c_scs,
    }


# ==========================================================================================
# 4. DECODIFICACIÓN DEL CROMOSOMA R^21, EVALUACIÓN MULTIOBJETIVO Y TTA
# ==========================================================================================
def decode_chromosome(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Decodifica individuo x in R^21 en pesos w (símplex L1), exponentes gamma y umbral tau."""
    raw_w = np.clip(params[:N_FACTORS], 1e-4, 1.0)
    w = raw_w / np.sum(raw_w)
    gamma = np.clip(params[N_FACTORS : 2 * N_FACTORS], 0.30, 4.00)
    tau = float(np.clip(params[2 * N_FACTORS], 0.25, 0.75))
    cv = float(np.sum(np.maximum(0.0, MIN_WEIGHT - w)))
    return w, gamma, tau, cv


def predict_probability_map(
    params: np.ndarray,
    X_cube: np.ndarray,
    use_tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Calcula el mapa continuo de probabilidad de deslizamiento P(i,j) in (0, 1)."""
    w, gamma, tau, cv = decode_chromosome(params)
    weighted_layers = w[:, None, None] * (X_cube ** gamma[:, None, None])
    score = np.sum(weighted_layers, axis=0)
    prob = 1.0 / (1.0 + np.exp(-10.0 * (score - tau)))
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)

    if use_tta:
        p_s1 = uniform_filter(prob, size=3, mode="nearest")
        p_s2 = gaussian_filter(prob, sigma=0.85)
        prob = np.clip(0.50 * prob + 0.30 * p_s1 + 0.20 * p_s2, 1e-6, 1.0 - 1e-6)

    return prob, w, gamma, tau, cv


def compute_spatial_tv(prob_map: np.ndarray) -> float:
    """Variación Total Espacial normalizada (mide fragmentación cartográfica)."""
    diff_v = np.abs(prob_map[1:, :] - prob_map[:-1, :])
    diff_h = np.abs(prob_map[:, 1:] - prob_map[:, :-1])
    return float(0.5 * (np.mean(diff_v) + np.mean(diff_h)))


def evaluate_objectives(
    params: np.ndarray,
    X_cube: np.ndarray,
    Y_ref: np.ndarray,
    eval_mask: np.ndarray,
    sub_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Evalúa los 3 objetivos en conflicto: [BCE, 1 - AUC_ROC, TV(P)]."""
    prob, _, _, _, cv = predict_probability_map(params, X_cube, use_tta=False)

    p_masked = prob[eval_mask]
    y_masked = Y_ref[eval_mask]

    if sub_indices is not None:
        p_eval = p_masked[sub_indices]
        y_eval = y_masked[sub_indices]
    else:
        p_eval = p_masked
        y_eval = y_masked

    f1_bce = float(-np.mean(y_eval * np.log(p_eval) + (1.0 - y_eval) * np.log(1.0 - p_eval)))
    f2_auc_err = float(1.0 - roc_auc_score(y_eval, p_eval))
    f3_tv = compute_spatial_tv(prob)

    return np.array([f1_bce, f2_auc_err, f3_tv], dtype=np.float64), cv


# ==========================================================================================
# 5. MOTOR DEL ALGORITMO GENÉTICO MULTIOBJETIVO (NSGA-II) E HIPERVOLUMEN 3D
# ==========================================================================================
def constrained_dominates(
    obj_a: np.ndarray, cv_a: float, obj_b: np.ndarray, cv_b: float
) -> bool:
    """Regla de Dominancia Restringida (Deb et al., 2002)."""
    if cv_a > 0.0 or cv_b > 0.0:
        if cv_a < cv_b:
            return True
        if cv_b < cv_a:
            return False
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))


def fast_nondominated_sort(
    objs: np.ndarray, cvs: np.ndarray
) -> Tuple[List[List[int]], np.ndarray]:
    """Ordenamiento rápido por frentes no dominados de NSGA-II."""
    n = len(objs)
    S: List[List[int]] = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    ranks = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if constrained_dominates(objs[p], cvs[p], objs[q], cvs[q]):
                S[p].append(q)
            elif constrained_dominates(objs[q], cvs[q], objs[p], cvs[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts, ranks


def crowding_distance_assignment(objs: np.ndarray, front: List[int]) -> np.ndarray:
    """Distancia de aglomeración (crowding distance) dentro de un frente."""
    k = len(front)
    distances = np.zeros(k, dtype=np.float64)
    if k <= 2:
        distances[:] = np.inf
        return distances

    front_objs = objs[front]
    n_obj = objs.shape[1]

    for m in range(n_obj):
        order = np.argsort(front_objs[:, m])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        f_min = float(front_objs[order[0], m])
        f_max = float(front_objs[order[-1], m])
        span = f_max - f_min
        if span < 1e-12:
            continue
        for idx in range(1, k - 1):
            distances[order[idx]] += (
                front_objs[order[idx + 1], m] - front_objs[order[idx - 1], m]
            ) / span

    return distances


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
    eta_c: float = ETA_C,
    pc: float = PC,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cruce Simulado Binario (Simulated Binary Crossover - SBX)."""
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > pc:
        return c1, c2

    for i in range(len(p1)):
        if rng.random() <= 0.5 and abs(float(p1[i] - p2[i])) > 1e-12:
            y1 = float(min(p1[i], p2[i]))
            y2 = float(max(p1[i], p2[i]))
            yl = float(LB[i])
            yu = float(UB[i])
            rand = float(rng.random())

            beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            if rand <= (1.0 / alpha):
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            ch1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

            beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            if rand <= (1.0 / alpha):
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            ch2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

            if rng.random() <= 0.5:
                c1[i], c2[i] = ch2, ch1
            else:
                c1[i], c2[i] = ch1, ch2

    return np.clip(c1, LB, UB), np.clip(c2, LB, UB)


def polynomial_mutation(
    x: np.ndarray,
    rng: np.random.Generator,
    eta_m: float = ETA_M,
    pm: float = PM,
) -> np.ndarray:
    """Mutación Polinomial (Polynomial Mutation - PM)."""
    y = x.copy()
    for i in range(len(y)):
        if rng.random() <= pm:
            yl = float(LB[i])
            yu = float(UB[i])
            delta1 = (float(y[i]) - yl) / (yu - yl)
            delta2 = (yu - float(y[i])) / (yu - yl)
            rand = float(rng.random())
            mut_pow = 1.0 / (eta_m + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                deltaq = (val ** mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                deltaq = 1.0 - (val ** mut_pow)
            y[i] = float(y[i]) + deltaq * (yu - yl)
    return np.clip(y, LB, UB)


def compute_hypervolume_3d(
    front_objs: np.ndarray, ref_point: np.ndarray, mc_samples: np.ndarray
) -> float:
    """Hipervolumen 3D normalizado respecto a ref_point (sin traslapes)."""
    valid = front_objs[np.all(front_objs < ref_point, axis=1)]
    if len(valid) == 0:
        return 0.0
    dominated = np.any(np.all(valid[:, None, :] <= mc_samples[None, :, :], axis=2), axis=0)
    return float(np.mean(dominated))


def run_nsga2_seed(
    seed: int,
    data: GeodatabaseData,
    mc_samples: np.ndarray,
    np_size: int = NP,
    g_max: int = G_MAX,
) -> SeedResult:
    """Ejecuta una corrida completa de NSGA-II con Bootstrap Espacial Estratificado."""
    rng = np.random.default_rng(seed)
    X_cube = data["X_cube"]
    Y_ref = data["Y_ref"]
    train_mask = data["train_mask"]

    y_train_flat = Y_ref[train_mask]
    pos_idx = np.where(y_train_flat == 1.0)[0]
    neg_idx = np.where(y_train_flat == 0.0)[0]
    n_pos_boot = min(len(pos_idx), 3500)
    n_neg_boot = min(len(neg_idx), 7000)

    def sample_stratified_bootstrap() -> np.ndarray:
        s_pos = rng.choice(pos_idx, size=n_pos_boot, replace=False)
        s_neg = rng.choice(neg_idx, size=n_neg_boot, replace=False)
        return np.concatenate([s_pos, s_neg])

    pop = rng.uniform(LB, UB, size=(np_size, DIM))
    boot_idx = sample_stratified_bootstrap()

    objs = np.zeros((np_size, 3), dtype=np.float64)
    cvs = np.zeros(np_size, dtype=np.float64)
    for i in range(np_size):
        objs[i], cvs[i] = evaluate_objectives(pop[i], X_cube, Y_ref, train_mask, boot_idx)

    fronts, ranks = fast_nondominated_sort(objs, cvs)
    crowd = np.zeros(np_size, dtype=np.float64)
    for f in fronts:
        crowd[f] = crowding_distance_assignment(objs, f)

    hv_history: List[float] = []
    hv_init = compute_hypervolume_3d(objs[fronts[0]][cvs[fronts[0]] == 0], HV_REF_POINT, mc_samples)
    hv_history.append(hv_init)

    for gen in range(1, g_max + 1):
        if gen % 5 == 0:
            boot_idx = sample_stratified_bootstrap()
            for i in range(np_size):
                objs[i], cvs[i] = evaluate_objectives(pop[i], X_cube, Y_ref, train_mask, boot_idx)
            fronts, ranks = fast_nondominated_sort(objs, cvs)
            for f in fronts:
                crowd[f] = crowding_distance_assignment(objs, f)

        offspring_list: List[np.ndarray] = []
        while len(offspring_list) < np_size:
            cand = rng.choice(np_size, size=4, replace=False)
            p1_idx = (
                int(cand[0])
                if (
                    ranks[cand[0]] < ranks[cand[1]]
                    or (ranks[cand[0]] == ranks[cand[1]] and crowd[cand[0]] > crowd[cand[1]])
                )
                else int(cand[1])
            )
            p2_idx = (
                int(cand[2])
                if (
                    ranks[cand[2]] < ranks[cand[3]]
                    or (ranks[cand[2]] == ranks[cand[3]] and crowd[cand[2]] > crowd[cand[3]])
                )
                else int(cand[3])
            )
            c1, c2 = sbx_crossover(pop[p1_idx], pop[p2_idx], rng)
            offspring_list.append(polynomial_mutation(c1, rng))
            if len(offspring_list) < np_size:
                offspring_list.append(polynomial_mutation(c2, rng))

        offspring = np.array(offspring_list, dtype=np.float64)
        off_objs = np.zeros((np_size, 3), dtype=np.float64)
        off_cvs = np.zeros(np_size, dtype=np.float64)
        for i in range(np_size):
            off_objs[i], off_cvs[i] = evaluate_objectives(offspring[i], X_cube, Y_ref, train_mask, boot_idx)

        comb_pop = np.vstack([pop, offspring])
        comb_objs = np.vstack([objs, off_objs])
        comb_cvs = np.concatenate([cvs, off_cvs])

        comb_fronts, _ = fast_nondominated_sort(comb_objs, comb_cvs)
        new_pop: List[np.ndarray] = []
        new_objs: List[np.ndarray] = []
        new_cvs: List[float] = []
        new_ranks: List[int] = []
        new_crowd: List[float] = []

        for r_idx, front in enumerate(comb_fronts):
            cd = crowding_distance_assignment(comb_objs, front)
            if len(new_pop) + len(front) <= np_size:
                for idx_local, idx_global in enumerate(front):
                    new_pop.append(comb_pop[idx_global])
                    new_objs.append(comb_objs[idx_global])
                    new_cvs.append(float(comb_cvs[idx_global]))
                    new_ranks.append(r_idx)
                    new_crowd.append(float(cd[idx_local]))
            else:
                rem = np_size - len(new_pop)
                order = np.argsort(-cd)[:rem]
                for idx_local in order:
                    idx_global = front[int(idx_local)]
                    new_pop.append(comb_pop[idx_global])
                    new_objs.append(comb_objs[idx_global])
                    new_cvs.append(float(comb_cvs[idx_global]))
                    new_ranks.append(r_idx)
                    new_crowd.append(float(cd[int(idx_local)]))
                break

        pop = np.array(new_pop, dtype=np.float64)
        objs = np.array(new_objs, dtype=np.float64)
        cvs = np.array(new_cvs, dtype=np.float64)
        ranks = np.array(new_ranks, dtype=int)
        crowd = np.array(new_crowd, dtype=np.float64)

        f0_mask = np.logical_and(ranks == 0, cvs == 0.0)
        hv_gen = compute_hypervolume_3d(objs[f0_mask], HV_REF_POINT, mc_samples)
        hv_history.append(hv_gen)

    exact_objs = np.zeros((np_size, 3), dtype=np.float64)
    exact_cvs = np.zeros(np_size, dtype=np.float64)
    for i in range(np_size):
        exact_objs[i], exact_cvs[i] = evaluate_objectives(pop[i], X_cube, Y_ref, train_mask, None)

    final_fronts, _ = fast_nondominated_sort(exact_objs, exact_cvs)
    f0 = [idx for idx in final_fronts[0] if exact_cvs[idx] == 0.0]
    if not f0:
        f0 = final_fronts[0]

    final_hv = compute_hypervolume_3d(exact_objs[f0], HV_REF_POINT, mc_samples)

    return {
        "seed": seed,
        "pareto_pop": pop[f0],
        "pareto_objs": exact_objs[f0],
        "hv_history": np.array(hv_history, dtype=np.float64),
        "final_hv": final_hv,
    }


# ==========================================================================================
# 6. TOMA DE DECISIONES A POSTERIORI, CONTRASTE CON ACTIVIDAD 6 Y EXPORTACIÓN
# ==========================================================================================
def evaluate_act6_baseline(data: GeodatabaseData) -> Act6BaselineMetrics:
    """Evalúa el modelo óptimo JADE de la Actividad 6 sobre los mismos bloques de prueba."""
    w_act6 = np.array([0.1088, 0.3632, 0.0202, 0.3330, 0.0202, 0.1253, 0.0293], dtype=np.float64)
    w_act6 /= np.sum(w_act6)
    g_act6 = np.array([2.500, 1.734, 2.449, 1.575, 2.169, 2.122, 2.500], dtype=np.float64)

    X_act6 = data["X_cube_act6"]
    Y_ref = data["Y_ref"]
    test_mask = data["test_mask"]
    shadow_mask = data["shadow_mask"]

    score = np.sum(w_act6[:, None, None] * (X_act6 ** g_act6[:, None, None]), axis=0)
    prob = np.clip(1.0 / (1.0 + np.exp(-10.0 * (score - 0.48))), 1e-6, 1.0 - 1e-6)

    p_test = prob[test_mask]
    y_test = Y_ref[test_mask]

    bce_test = float(-np.mean(y_test * np.log(p_test) + (1.0 - y_test) * np.log(1.0 - p_test)))
    auc_test = float(roc_auc_score(y_test, p_test))
    tv_map = compute_spatial_tv(prob)

    shade_neg = np.logical_and(shadow_mask, Y_ref == 0.0)
    fpr_shadow = float(np.mean(prob[shade_neg] >= 0.50))

    return {
        "prob_map": prob,
        "bce_test": bce_test,
        "auc_test": auc_test,
        "tv_map": tv_map,
        "fpr_shadow": fpr_shadow,
    }


def select_posterior_solutions(
    pareto_pop: np.ndarray, pareto_objs: np.ndarray
) -> Dict[str, PosteriorProfile]:
    """Extrae 3 perfiles operativos a posteriori del Frente de Pareto."""
    f_min = np.min(pareto_objs, axis=0)
    f_max = np.max(pareto_objs, axis=0)
    f_norm = (pareto_objs - f_min) / np.maximum(f_max - f_min, 1e-12)

    idx_roc = int(np.argmin(pareto_objs[:, 1]))
    idx_knee = int(np.argmin(np.linalg.norm(f_norm, axis=1)))

    eps_tv = float(np.percentile(pareto_objs[:, 2], 25.0))
    compact_candidates = np.where(pareto_objs[:, 2] <= eps_tv)[0]
    idx_compact = int(compact_candidates[np.argmin(pareto_objs[compact_candidates, 0])])

    return {
        "A_Maxima_Discriminacion_ROC": {"x": pareto_pop[idx_roc], "f": pareto_objs[idx_roc]},
        "B_Compromiso_Equilibrado_Knee": {"x": pareto_pop[idx_knee], "f": pareto_objs[idx_knee]},
        "C_Alta_Compacidad_ProtCivil": {"x": pareto_pop[idx_compact], "f": pareto_objs[idx_compact]},
    }


def export_geotiff_and_figures(
    data: GeodatabaseData,
    seed_results: List[SeedResult],
    consolidated_pop: np.ndarray,
    consolidated_objs: np.ndarray,
    profiles: Dict[str, PosteriorProfile],
    act6_metrics: Act6BaselineMetrics,
) -> None:
    """Exporta mapas GeoTIFF para ArcGIS Pro 3D, tablas CSV y figuras de alta resolución."""
    X_cube = data["X_cube"]
    Y_ref = data["Y_ref"]
    test_mask = data["test_mask"]
    shadow_mask = data["shadow_mask"]

    x_best = profiles["B_Compromiso_Equilibrado_Knee"]["x"]
    prob_nsga2, w_opt, gamma_opt, tau_opt, _ = predict_probability_map(
        x_best, X_cube, use_tta=True
    )

    q_bins = np.quantile(prob_nsga2, [0.25, 0.50, 0.72, 0.88])
    zonif_5 = np.digitize(prob_nsga2, bins=q_bins).astype(np.uint8) + 1

    if HAS_GIS and rasterio is not None and data["transform"] is not None:
        profile_tif: Dict[str, Any] = {
            "driver": "GTiff",
            "height": GRID_SIZE,
            "width": GRID_SIZE,
            "count": 1,
            "dtype": "float32",
            "crs": data["crs"],
            "transform": data["transform"],
            "compress": "lzw",
        }
        with rasterio.open(OUTPUT_DIR / "susceptibilidad_santamaria_zacatepec_NSGA2.tif", "w", **profile_tif) as dst:
            dst.write(prob_nsga2.astype(np.float32), 1)

        profile_tif["dtype"] = "uint8"
        with rasterio.open(OUTPUT_DIR / "zonificacion_riesgo_5niveles_NSGA2.tif", "w", **profile_tif) as dst:
            dst.write(zonif_5, 1)
        print("[INFO] Rásters GeoTIFF exportados para ArcGIS Pro 3D Local Scene.")

    summary_rows: List[Dict[str, Any]] = [
        {
            "Modelo / Perfil": "Actividad 6 Baseline (JADE 7-Factores sin SCS+C)",
            "BCE_Test": act6_metrics["bce_test"],
            "AUC_ROC_Test": act6_metrics["auc_test"],
            "Fragmentacion_TV": act6_metrics["tv_map"],
            "FPR_Sombras_Pct": act6_metrics["fpr_shadow"] * 100.0,
        }
    ]

    shade_neg = np.logical_and(shadow_mask, Y_ref == 0.0)
    for name, item in profiles.items():
        p_map, _, _, _, _ = predict_probability_map(item["x"], X_cube, use_tta=True)
        p_t = p_map[test_mask]
        y_t = Y_ref[test_mask]
        bce_t = float(-np.mean(y_t * np.log(p_t) + (1.0 - y_t) * np.log(1.0 - p_t)))
        auc_t = float(roc_auc_score(y_t, p_t))
        tv_t = compute_spatial_tv(p_map)
        fpr_sh = float(np.mean(p_map[shade_neg] >= 0.50))
        summary_rows.append(
            {
                "Modelo / Perfil": f"Actividad 7 NSGA-II ({name})",
                "BCE_Test": bce_t,
                "AUC_ROC_Test": auc_t,
                "Fragmentacion_TV": tv_t,
                "FPR_Sombras_Pct": fpr_sh * 100.0,
            }
        )

    df_comp = pd.DataFrame(summary_rows)
    df_comp.to_csv(OUTPUT_DIR / "tabla_comparativa_Act6_vs_Act7.csv", index=False)

    df_factors = pd.DataFrame(
        {
            "Factor": FACTOR_NAMES,
            "Peso_Optimo_wk": w_opt,
            "Influencia_Pct": w_opt * 100.0,
            "Exponente_gamma_k": gamma_opt,
        }
    ).sort_values(by="Peso_Optimo_wk", ascending=False)
    df_factors.to_csv(OUTPUT_DIR / "tabla_jerarquia_10_factores_NSGA2.csv", index=False)

    hvs = np.array([r["final_hv"] for r in seed_results], dtype=np.float64)
    hv_stats: Dict[str, Any] = {
        "carta": "Santa María Zacatepec E14-D54",
        "n_seeds": N_SEEDS,
        "hv_mediana": float(np.median(hvs)),
        "hv_q25": float(np.percentile(hvs, 25)),
        "hv_q75": float(np.percentile(hvs, 75)),
        "hv_ric": float(np.percentile(hvs, 75) - np.percentile(hvs, 25)),
        "hv_max": float(np.max(hvs)),
        "tau_optimo_knee": tau_opt,
        "c_scs_shadow": data["c_scs"],
    }
    with open(OUTPUT_DIR / "estadisticas_hipervolumen_NSGA2.json", "w", encoding="utf-8") as fp:
        json.dump(hv_stats, fp, indent=4, ensure_ascii=False)

    # FIGURA 1: Frente de Pareto y Convergencia de Hipervolumen
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    sc = axes[0].scatter(
        consolidated_objs[:, 0],
        1.0 - consolidated_objs[:, 1],
        c=consolidated_objs[:, 2],
        cmap="viridis_r",
        s=48,
        edgecolors="k",
        linewidths=0.4,
        label="Frente de Pareto NSGA-II (10 semillas)",
    )
    cbar = plt.colorbar(sc, ax=axes[0])
    cbar.set_label("Fragmentación Espacial TV(P)")

    markers: Dict[str, str] = {
        "A_Maxima_Discriminacion_ROC": "^",
        "B_Compromiso_Equilibrado_Knee": "*",
        "C_Alta_Compacidad_ProtCivil": "s",
    }
    colors_p: Dict[str, str] = {
        "A_Maxima_Discriminacion_ROC": "#d62728",
        "B_Compromiso_Equilibrado_Knee": "#ff7f0e",
        "C_Alta_Compacidad_ProtCivil": "#1f77b4",
    }
    for pname, pval in profiles.items():
        axes[0].scatter(
            float(pval["f"][0]),
            float(1.0 - pval["f"][1]),
            marker=markers[pname],
            s=150,
            color=colors_p[pname],
            edgecolors="black",
            zorder=5,
            label=pname.replace("_", " "),
        )
    axes[0].set_xlabel("Error de Calibración Probabilística (BCE)")
    axes[0].set_ylabel("Discriminación Espacial (AUC-ROC Train)")
    axes[0].set_title("Frente de Pareto NSGA-II: Santa María Zacatepec E14-D54")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=7.5, loc="lower right")

    hv_mat = np.vstack([r["hv_history"] for r in seed_results])
    gens = np.arange(hv_mat.shape[1])
    med_hv = np.median(hv_mat, axis=0)
    q25_hv = np.percentile(hv_mat, 25, axis=0)
    q75_hv = np.percentile(hv_mat, 75, axis=0)

    axes[1].plot(gens, med_hv, color="#2b193d", lw=2.2, label="Mediana Hipervolumen (10 semillas)")
    axes[1].fill_between(gens, q25_hv, q75_hv, color="#6b4c9a", alpha=0.25, label="Rango Intercuartílico (Q25-Q75)")
    axes[1].set_xlabel("Generación Evolutiva")
    axes[1].set_ylabel("Hipervolumen Normalizado 3D (HV)")
    axes[1].set_title("Convergencia del Hipervolumen NSGA-II (E14-D54)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_frente_pareto_e_hipervolumen_NSGA2.png", dpi=300)
    plt.close()

    # FIGURA 2: Comparativa Espacial Act. 6 vs Act. 7
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
    im0 = axes[0].imshow(act6_metrics["prob_map"], cmap="jet", vmin=0, vmax=1)
    axes[0].set_title(
        f"Actividad 6 (JADE 7-Factores)\nAUC_test={act6_metrics['auc_test']:.4f} | TV={act6_metrics['tv_map']:.4f}"
    )
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(prob_nsga2, cmap="jet", vmin=0, vmax=1)
    auc_nsga_test = float(roc_auc_score(Y_ref[test_mask], prob_nsga2[test_mask]))
    tv_nsga = compute_spatial_tv(prob_nsga2)
    axes[1].set_title(
        f"Actividad 7 (NSGA-II + SCS+C + TTA)\nAUC_test={auc_nsga_test:.4f} | TV={tv_nsga:.4f}"
    )
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fpr6, tpr6, _ = roc_curve(Y_ref[test_mask], act6_metrics["prob_map"][test_mask])
    fpr7, tpr7, _ = roc_curve(Y_ref[test_mask], prob_nsga2[test_mask])
    axes[2].plot(
        fpr6,
        tpr6,
        "--",
        lw=1.8,
        color="#1f77b4",
        label=f"Act. 6 JADE (AUC_test={act6_metrics['auc_test']:.4f})",
    )
    axes[2].plot(
        fpr7,
        tpr7,
        "-",
        lw=2.2,
        color="#d62728",
        label=f"Act. 7 NSGA-II (AUC_test={auc_nsga_test:.4f})",
    )
    axes[2].plot([0, 1], [0, 1], ":", color="gray", label="Azar (AUC=0.50)")
    axes[2].set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    axes[2].set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    axes[2].set_title("Validación Espacial: Santa María Zacatepec E14-D54")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_comparativa_espacial_Act6_vs_Act7.png", dpi=300)
    plt.close()


# ==========================================================================================
# 7. EJECUCIÓN PRINCIPAL
# ==========================================================================================
def main() -> None:
    print("=" * 90)
    print(SCRIPT_TITLE)
    print("=" * 90)
    t_start = time.time()

    data = load_zacatepec_geodatabase(GRID_SIZE)

    act6_metrics = evaluate_act6_baseline(data)
    print(
        f"[BASE ACT. 6] AUC_test = {act6_metrics['auc_test']:.4f} | "
        f"BCE_test = {act6_metrics['bce_test']:.4f} | "
        f"TV = {act6_metrics['tv_map']:.4f} | "
        f"FPR_sombras = {act6_metrics['fpr_shadow']*100:.2f}%"
    )

    rng_hv = np.random.default_rng(2026)
    mc_samples = rng_hv.uniform(0.0, HV_REF_POINT, size=(25000, 3))

    seed_results: List[SeedResult] = []
    all_pareto_pops: List[np.ndarray] = []
    all_pareto_objs: List[np.ndarray] = []

    for seed in range(1, N_SEEDS + 1):
        t0 = time.time()
        res = run_nsga2_seed(seed, data, mc_samples, NP, G_MAX)
        seed_results.append(res)
        all_pareto_pops.append(res["pareto_pop"])
        all_pareto_objs.append(res["pareto_objs"])
        print(
            f"  -> Semilla {seed:02d}/{N_SEEDS:02d} | Soluciones Pareto: {len(res['pareto_pop']):02d} | "
            f"HV Final: {res['final_hv']:.5f} | Tiempo: {time.time() - t0:.1f}s"
        )

    merged_pop = np.vstack(all_pareto_pops)
    merged_objs = np.vstack(all_pareto_objs)
    merged_cvs = np.zeros(len(merged_pop), dtype=np.float64)
    fronts_global, _ = fast_nondominated_sort(merged_objs, merged_cvs)
    f0_global = fronts_global[0]
    consolidated_pop = merged_pop[f0_global]
    consolidated_objs = merged_objs[f0_global]

    profiles = select_posterior_solutions(consolidated_pop, consolidated_objs)
    export_geotiff_and_figures(
        data, seed_results, consolidated_pop, consolidated_objs, profiles, act6_metrics
    )

    print("=" * 90)
    print(f"[FINALIZADO] Proceso completado en {time.time() - t_start:.1f} segundos.")
    print(f"[ENTREGABLES] Guardados en: {OUTPUT_DIR.resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    main()