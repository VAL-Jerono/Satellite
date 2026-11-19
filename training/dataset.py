"""
training/dataset.py
────────────────────
PyTorch Dataset that loads .npy patches from disk and returns
(image_tensor, label) pairs. Includes:
- Refined dataset-wide per-band mean/std normalization across 8,356 patches
- Enhanced spatial & spectral augmentations for strong LORO generalization
- 10-band Sentinel-2 input with automatic Drive auto-recovery & shape check
- Local /tmp cache: patches are pre-staged from Drive to /tmp on first access
  to avoid repeated slow FUSE reads during training epochs.
"""

from __future__ import annotations
import os
import shutil
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

# Local SSD/tmpfs cache directory for patches (survives within a Colab session)
_LOCAL_PATCH_CACHE = Path("/tmp/sat_patches")


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
        _LOCAL_PATCH_CACHE.mkdir(parents=True, exist_ok=True)
        # NOTE: patches are pre-staged to /tmp/sat_patches in __getitem__ on first access.
        # This converts slow Drive FUSE reads (per-epoch) into fast local reads after epoch 1.

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        fname = Path(row["filename"])

        # ── Local /tmp cache: convert slow Drive FUSE reads → fast local reads ──
        # On first access the patch is copied from Drive to /tmp/sat_patches/.
        # Subsequent epochs read from local SSD — 10–50× faster than Drive FUSE.
        cached = _LOCAL_PATCH_CACHE / fname.name
        if cached.exists():
            fname = cached
        elif not fname.exists():
            # Try Drive fallback path
            drive_fname = Path("/content/drive/MyDrive/land_atlas_baseline/patches") / fname.name
            if drive_fname.exists():
                # Stage to local cache for future epochs
                try:
                    shutil.copy2(drive_fname, cached)
                    fname = cached
                except Exception:
                    fname = drive_fname
            else:
                fname = None
        else:
            # File exists at manifest path — stage to cache
            try:
                shutil.copy2(fname, cached)
                fname = cached
            except Exception:
                pass  # use original fname

        arr = None
        if fname is not None and fname.exists():
            try:
                raw_arr = np.load(fname).astype(np.float32)
                if raw_arr.shape == (10, 64, 64):
                    arr = raw_arr
                else:
                    print(f"WARN: Corrupt patch shape {raw_arr.shape} at {fname}.")
                    try:
                        fname.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception as exc:
                print(f"WARN: Failed loading patch {fname}: {exc}.")

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


def warm_patch_cache(split_csvs: list, max_patches: int = 30000, verbose: bool = True) -> int:
    """
    Pre-stage patches from Google Drive FUSE → /tmp/sat_patches before training.

    Why: Reading 28k patches from Drive FUSE during the first epoch is slow (~1–2 h).
    This function copies patches to /tmp/ SSD in a single sequential scan BEFORE
    training starts, so all subsequent epoch reads are fast (local NVMe/tmpfs).

    Parameters
    ----------
    split_csvs : list of str/Path
        CSV files to stage patches for (e.g. fold_0_train.csv, fold_0_val.csv).
    max_patches : int
        Cap on total patches to stage (safety limit for disk space).
    verbose : bool
        Print progress every 1000 files.

    Returns
    -------
    int : number of patches newly staged (skips already-cached).
    """
    _LOCAL_PATCH_CACHE.mkdir(parents=True, exist_ok=True)
    drive_dir = Path("/content/drive/MyDrive/land_atlas_baseline/patches")
    staged, skipped, missing = 0, 0, 0

    all_filenames = []
    for csv_path in split_csvs:
        try:
            df = pd.read_csv(csv_path)
            if "filename" in df.columns:
                all_filenames.extend(df["filename"].tolist())
        except Exception:
            pass

    seen = set()
    for fn_str in all_filenames:
        if staged >= max_patches:
            break
        fn = Path(fn_str)
        name = fn.name
        if name in seen:
            continue
        seen.add(name)

        cached = _LOCAL_PATCH_CACHE / name
        if cached.exists():
            skipped += 1
            continue

        # Try manifest path first, then Drive fallback
        src = fn if fn.exists() else (drive_dir / name if drive_dir.exists() and (drive_dir / name).exists() else None)
        if src is None:
            missing += 1
            continue

        try:
            shutil.copy2(src, cached)
            staged += 1
        except Exception:
            missing += 1

        if verbose and (staged % 1000 == 0) and staged > 0:
            print(f"  [cache] staged {staged} patches so far…")

    if verbose:
        print(f"  [cache] Done — staged={staged}, already_cached={skipped}, missing={missing}")
        print(f"  [cache] /tmp/sat_patches now has {len(list(_LOCAL_PATCH_CACHE.glob('*.npy')))} patches")
    return staged
