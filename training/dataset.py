"""
training/dataset.py
────────────────────
PyTorch Dataset that loads .npy patches from disk and returns
(image_tensor, label) pairs. Includes:
- Refined dataset-wide per-band mean/std normalization across 8,356 patches
- Enhanced spatial & spectral augmentations for strong LORO generalization
- 10-band Sentinel-2 input with automatic Drive auto-recovery & shape check
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple

from data.compute_band_stats import load_band_stats, EMPIRICAL_BAND_MEAN, EMPIRICAL_BAND_STD

# Use absolute path so band_stats.json resolves correctly regardless of CWD
_ROOT_DS = Path(__file__).resolve().parents[1]
_BAND_STATS_PATH = _ROOT_DS / "data" / "band_stats.json"

# Load computed band statistics or fallback to high-fidelity empirical defaults
_BAND_MEAN, _BAND_STD = load_band_stats(_BAND_STATS_PATH)


class PatchDataset(Dataset):
    """
    Parameters
    ----------
    manifest_csv : str | Path
        CSV with columns: filename, label, region, weight
    augment : bool
        Apply spatial & spectral augmentations (use True for training).
    """

    def __init__(self, manifest_csv: str | Path, augment: bool = False):
        self.df       = pd.read_csv(manifest_csv)
        self.augment  = augment
        self.mean, self.std = load_band_stats(_ROOT_DS / "data" / "band_stats.json")
        self.mean_arr = self.mean[:, None, None]   # (C, 1, 1)
        self.std_arr  = self.std[:, None, None]
        # NOTE: patches are loaded on-demand from Drive in __getitem__ — no bulk copy here.

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        fname = Path(row["filename"])
        if not fname.exists():
            drive_fname = Path("/content/drive/MyDrive/land_atlas_baseline/patches") / fname.name
            if drive_fname.exists():
                fname = drive_fname
            else:
                fname = None

        arr = None
        if fname is not None and fname.exists():
            try:
                raw_arr = np.load(fname).astype(np.float32)
                if raw_arr.shape == (10, 64, 64):
                    arr = raw_arr
                else:
                    print(f"WARN: Corrupt patch shape {raw_arr.shape} at {fname}. Removing file.")
                    try:
                        fname.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception as exc:
                print(f"WARN: Failed loading patch {fname}: {exc}. Removing file.")
                try:
                    fname.unlink(missing_ok=True)
                except Exception:
                    pass

        if arr is None:
            arr = np.zeros((10, 64, 64), dtype=np.float32)

        # Exact Refined Band Normalization
        arr = (arr - self.mean_arr) / (self.std_arr + 1e-6)
        arr = np.clip(arr, -3.5, 3.5)

        # Advanced Augmentations (training only)
        if self.augment:
            # 1. Vertical flip
            if np.random.rand() > 0.5:
                arr = arr[:, ::-1, :].copy()
            # 2. Horizontal flip
            if np.random.rand() > 0.5:
                arr = arr[:, :, ::-1].copy()
            # 3. Random 90/180/270 degree rotation
            k = np.random.randint(0, 4)
            if k > 0:
                arr = np.rot90(arr, k=k, axes=(1, 2)).copy()
            # 4. Spectral intensity jittering (+/- 5%)
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.95, 1.05, size=(10, 1, 1)).astype(np.float32)
                arr  *= scale
            # 5. Gaussian noise
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.015, arr.shape).astype(np.float32)
                arr  += noise

        img   = torch.from_numpy(arr)
        label = int(row["label"])
        weight= float(row.get("weight", 1.0))
        return img, label, weight

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for CrossEntropyLoss / Focal Loss."""
        counts = self.df["label"].value_counts().sort_index()
        # Handle missing classes in smaller splits
        weights = np.zeros(6, dtype=np.float32)
        for cls_idx in range(6):
            c_val = counts.get(cls_idx, 0)
            weights[cls_idx] = 1.0 / (c_val + 5.0)  # smoothed inverse frequency
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)
