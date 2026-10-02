"""
training/dataset.py
────────────────────
PyTorch Dataset that loads .npy patches from disk and returns
(image_tensor, label) pairs.  Includes:
- Band normalisation (per-band mean/std computed on training set)
- Augmentation (random flip, random 90° rotation, Gaussian noise) for train
- 10-band Sentinel-2 input (no RGB conversion needed – CNN adapts)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


# Per-band approximate statistics (Sentinel-2 SR, scale 0-1)
# Computed on a representative Kenya sample; refine after first full run.
_BAND_MEAN = np.array([
    0.0673, 0.0830, 0.0881, 0.1189,  # B2 B3 B4 B5
    0.1983, 0.2342, 0.2547, 0.2633,  # B6 B7 B8 B8A
    0.1686, 0.0954,                   # B11 B12
], dtype=np.float32)

_BAND_STD = np.array([
    0.0461, 0.0497, 0.0603, 0.0672,
    0.0888, 0.1003, 0.1073, 0.1098,
    0.0841, 0.0586,
], dtype=np.float32)


class PatchDataset(Dataset):
    """
    Parameters
    ----------
    manifest_csv : str | Path
        CSV with columns: filename, label, region, weight
    augment : bool
        Apply random flips and noise (use True for training).
    """

    def __init__(self, manifest_csv: str | Path, augment: bool = False):
        self.df       = pd.read_csv(manifest_csv)
        self.augment  = augment
        self.mean     = _BAND_MEAN[:, None, None]   # (C, 1, 1)
        self.std      = _BAND_STD[:, None, None]

        # Auto-sync patches from Drive if local files are missing after a Colab restart
        drive_dir = Path("/content/drive/MyDrive/land_atlas_baseline/patches")
        if drive_dir.exists() and len(self.df) > 0:
            first_path = Path(self.df.iloc[0]["filename"])
            if not first_path.exists():
                print(f"Local patches missing at {first_path.parent}. Restoring from Drive: {drive_dir}...")
                import shutil
                first_path.parent.mkdir(parents=True, exist_ok=True)
                for drive_file in drive_dir.glob("*.npy"):
                    dest = first_path.parent / drive_file.name
                    if not dest.exists():
                        shutil.copy(drive_file, dest)
                print("Restored patches from Drive.")

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

        # Normalise
        arr = (arr - self.mean) / (self.std + 1e-6)
        arr = np.clip(arr, -3.0, 3.0)

        # Augmentation (train only)
        if self.augment:
            if np.random.rand() > 0.5:
                arr = arr[:, ::-1, :].copy()          # vertical flip
            if np.random.rand() > 0.5:
                arr = arr[:, :, ::-1].copy()          # horizontal flip
            k = np.random.randint(0, 4)
            arr = np.rot90(arr, k=k, axes=(1, 2)).copy()  # 0/90/180/270°
            noise = np.random.normal(0, 0.02, arr.shape).astype(np.float32)
            arr  += noise

        img   = torch.from_numpy(arr)
        label = int(row["label"])
        weight= float(row.get("weight", 1.0))
        return img, label, weight

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss."""
        counts  = self.df["label"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(float)
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)
