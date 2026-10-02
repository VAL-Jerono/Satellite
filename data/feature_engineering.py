"""
data/feature_engineering.py
──────────────────────────────
Remote Sensing Feature Engineering Module.
Calculates 8 key spectral indices and spatial texture metrics for pixel-level
and patch-level Sentinel-2 data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union

# Band index mapping for Sentinel-2 10-band array [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
BANDS = {
    "B2": 0,    # Blue (490 nm)
    "B3": 1,    # Green (560 nm)
    "B4": 2,    # Red (665 nm)
    "B5": 3,    # Red Edge 1 (705 nm)
    "B6": 4,    # Red Edge 2 (740 nm)
    "B7": 5,    # Red Edge 3 (783 nm)
    "B8": 6,    # NIR Broad (842 nm)
    "B8A": 7,   # NIR Narrow (865 nm)
    "B11": 8,   # SWIR 1 (1610 nm)
    "B12": 9,   # SWIR 2 (2190 nm)
}

INDEX_NAMES = ["NDVI", "NDWI", "MNDWI", "NDBI", "EVI", "SAVI", "BSI", "NDRE"]


def compute_spectral_indices(data: Union[np.ndarray, pd.DataFrame], eps: float = 1e-6) -> Dict[str, np.ndarray]:
    """
    Computes 8 key spectral indices from 10-band Sentinel-2 input.

    Input can be:
    - 1D/2D array or DataFrame of shape (N, 10) or dict of bands
    - 3D array of shape (10, H, W)
    """
    if isinstance(data, pd.DataFrame):
        b2  = data["B2"].values
        b3  = data["B3"].values
        b4  = data["B4"].values
        b5  = data["B5"].values
        b8  = data["B8"].values
        b11 = data["B11"].values
        b12 = data["B12"].values
    elif isinstance(data, np.ndarray) and data.ndim == 3:
        b2  = data[BANDS["B2"]]
        b3  = data[BANDS["B3"]]
        b4  = data[BANDS["B4"]]
        b5  = data[BANDS["B5"]]
        b8  = data[BANDS["B8"]]
        b11 = data[BANDS["B11"]]
        b12 = data[BANDS["B12"]]
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        b2  = data[:, BANDS["B2"]]
        b3  = data[:, BANDS["B3"]]
        b4  = data[:, BANDS["B4"]]
        b5  = data[:, BANDS["B5"]]
        b8  = data[:, BANDS["B8"]]
        b11 = data[:, BANDS["B11"]]
        b12 = data[:, BANDS["B12"]]
    else:
        raise ValueError(f"Unsupported input shape/type: {type(data)}")

    # 1. NDVI: Normalized Difference Vegetation Index
    ndvi = (b8 - b4) / (b8 + b4 + eps)

    # 2. NDWI: Normalized Difference Water Index (McFeeters)
    ndwi = (b3 - b8) / (b3 + b8 + eps)

    # 3. MNDWI: Modified NDWI (Xu)
    mndwi = (b3 - b11) / (b3 + b11 + eps)

    # 4. NDBI: Normalized Difference Built-Up Index
    ndbi = (b11 - b8) / (b11 + b8 + eps)

    # 5. EVI: Enhanced Vegetation Index
    evi_denom = b8 + 6.0 * b4 - 7.5 * b2 + 1.0
    evi = 2.5 * (b8 - b4) / (evi_denom + eps)
    evi = np.clip(evi, -2.0, 2.0)

    # 6. SAVI: Soil-Adjusted Vegetation Index (L=0.5)
    savi = ((b8 - b4) / (b8 + b4 + 0.5 + eps)) * 1.5

    # 7. BSI: Bare Soil Index
    bsi_num = (b11 + b4) - (b8 + b2)
    bsi_denom = (b11 + b4) + (b8 + b2) + eps
    bsi = bsi_num / bsi_denom

    # 8. NDRE: Red Edge NDVI
    ndre = (b8 - b5) / (b8 + b5 + eps)

    return {
        "NDVI": ndvi.astype(np.float32),
        "NDWI": ndwi.astype(np.float32),
        "MNDWI": mndwi.astype(np.float32),
        "NDBI": ndbi.astype(np.float32),
        "EVI": evi.astype(np.float32),
        "SAVI": savi.astype(np.float32),
        "BSI": bsi.astype(np.float32),
        "NDRE": ndre.astype(np.float32),
    }


def extract_patch_features(patch: np.ndarray) -> Dict[str, float]:
    """
    Extracts summary statistics (mean, std, min, max, spatial gradients) for a 3D patch (10, H, W).
    Used to build rich tabular features for EDA and GBDT models.
    """
    feats = {}
    band_names = list(BANDS.keys())

    # Band statistics
    for b_idx, b_name in enumerate(band_names):
        b_data = patch[b_idx]
        feats[f"{b_name}_mean"] = float(np.mean(b_data))
        feats[f"{b_name}_std"]  = float(np.std(b_data))
        feats[f"{b_name}_min"]  = float(np.min(b_data))
        feats[f"{b_name}_max"]  = float(np.max(b_data))

    # Spectral Index statistics
    indices = compute_spectral_indices(patch)
    for idx_name, idx_arr in indices.items():
        feats[f"{idx_name}_mean"] = float(np.mean(idx_arr))
        feats[f"{idx_name}_std"]  = float(np.std(idx_arr))
        feats[f"{idx_name}_p10"]  = float(np.percentile(idx_arr, 10))
        feats[f"{idx_name}_p90"]  = float(np.percentile(idx_arr, 90))

    # Spatial texture / gradient features (Sobel approximation)
    red_grad_x = np.abs(np.diff(patch[BANDS["B4"]], axis=1)).mean()
    red_grad_y = np.abs(np.diff(patch[BANDS["B4"]], axis=0)).mean()
    feats["spatial_texture_red"] = float(red_grad_x + red_grad_y)

    nir_grad_x = np.abs(np.diff(patch[BANDS["B8"]], axis=1)).mean()
    nir_grad_y = np.abs(np.diff(patch[BANDS["B8"]], axis=0)).mean()
    feats["spatial_texture_nir"] = float(nir_grad_x + nir_grad_y)

    return feats


def augment_dataframe_with_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds spectral indices to a point DataFrame containing raw band columns (B2..B12).
    """
    out_df = df.copy()
    indices = compute_spectral_indices(out_df)
    for col_name, col_vals in indices.items():
        out_df[col_name] = col_vals
    return out_df
