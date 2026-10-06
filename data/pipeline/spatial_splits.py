"""
data/pipeline/spatial_splits.py
────────────────────────────────
Builds buffered spatial-block folds from the patch manifest.
Supports both county-level (LORO) and climate-zone-level (LOZO) splits.

Writes:
    data/splits/fold_{k}_train.csv
    data/splits/fold_{k}_val.csv
    data/splits/loro_{region}_train.csv   (leave-one-region-out, per county)
    data/splits/loro_{region}_val.csv
    data/splits/lozo_{zone}_train.csv     (leave-one-zone-out, per climate zone)
    data/splits/lozo_{zone}_val.csv
    data/splits/split_summary.json

Run from repo root:
    python -m data.pipeline.spatial_splits
"""

import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import GroupKFold

_ROOT = Path(__file__).resolve().parents[2]
_CFG  = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())

BLOCK_DEG   = _CFG["training"]["block_deg"]
N_FOLDS     = _CFG["training"]["n_folds"]
MANIFEST    = _ROOT / _CFG["paths"]["patches_dir"] / "manifest.csv"
MASTER_CSV  = _ROOT / _CFG["paths"]["raw_csv"]
SPLITS_DIR  = _ROOT / _CFG["paths"]["splits_dir"]


def _assign_blocks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bx"]    = np.floor(df["lon"] / BLOCK_DEG).astype(int)
    df["by"]    = np.floor(df["lat"] / BLOCK_DEG).astype(int)
    df["block"] = df["region"] + "_" + df["bx"].astype(str) + "_" + df["by"].astype(str)
    return df


def _buffer_filter(df: pd.DataFrame, train_idx, test_idx) -> np.ndarray:
    """Remove training points in blocks adjacent to any test block."""
    reg = df["region"].values
    bx  = df["bx"].values
    by  = df["by"].values
    test_set = set(zip(reg[test_idx], bx[test_idx], by[test_idx]))
    keep = [
        i for i in train_idx
        if not any(
            (reg[i], bx[i] + dx, by[i] + dy) in test_set
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        )
    ]
    return np.array(keep)


def build_splits(project: str = "propertysatellite"):
    """
    Build spatial-block CV + LORO + LOZO splits.

    Data source priority:
      1. manifest.csv  — when .npy patches are present locally (CNN workflow)
      2. master CSV    — always has lon/lat from EE pull; used for tabular ML
    """
    df = None

    # ── Try manifest.csv first (patch-level splits for CNN) ───────────────────
    manifest_path = MANIFEST
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        try:
            tmp = pd.read_csv(manifest_path)
            if len(tmp) > 0 and {"lon", "lat", "region", "label"}.issubset(tmp.columns):
                df = tmp
                print(f"  [SPLITS] Using patch manifest: {len(df)} patches")
        except Exception as exc:
            print(f"  [SPLITS] WARN: could not read manifest.csv ({exc}) — falling back to master CSV")

    # ── Fall back to master CSV (tabular ML splits) ───────────────────────────
    if df is None:
        drive_csv = Path("/content/drive/MyDrive/land_atlas_baseline") / MASTER_CSV.name
        csv_candidates = [MASTER_CSV, drive_csv]
        for candidate in csv_candidates:
            if candidate.exists() and candidate.stat().st_size > 0:
                try:
                    tmp = pd.read_csv(candidate)
                    if len(tmp) > 0 and {"lon", "lat", "region", "label"}.issubset(tmp.columns):
                        df = tmp
                        print(f"  [SPLITS] Using master CSV: {len(df)} rows from {candidate}")
                        break
                except Exception as exc:
                    print(f"  [SPLITS] WARN: could not read {candidate}: {exc}")

    if df is None:
        raise FileNotFoundError(
            "Cannot build splits: neither manifest.csv nor master CSV are available.\n"
            f"Expected master CSV at: {MASTER_CSV}\n"
            "Run the pipeline merge_csv stage first."
        )

    df = _assign_blocks(df)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {"n_total": len(df), "n_blocks": df["block"].nunique(), "folds": []}

    # ── Block CV folds ────────────────────────────────────────────────────────
    gkf = GroupKFold(n_splits=N_FOLDS)
    for k, (tr_raw, te) in enumerate(gkf.split(df, df["label"], groups=df["block"])):
        tr = _buffer_filter(df, tr_raw, te)
        df.iloc[tr].to_csv(SPLITS_DIR / f"fold_{k}_train.csv", index=False)
        df.iloc[te].to_csv(SPLITS_DIR / f"fold_{k}_val.csv",   index=False)
        info = dict(fold=k, train=int(len(tr)), val=int(len(te)),
                    train_before_buffer=int(len(tr_raw)))
        summary["folds"].append(info)
        print(f"Fold {k}: train {len(tr_raw)} → {len(tr)} (buffered) | val {len(te)}")

    # ── Leave-one-region-out (county-level) ──────────────────────────────────
    loro = []
    for region in df["region"].unique():
        te_idx = np.where(df["region"].values == region)[0]
        tr_idx = np.where(df["region"].values != region)[0]
        df.iloc[tr_idx].to_csv(SPLITS_DIR / f"loro_{region}_train.csv", index=False)
        df.iloc[te_idx].to_csv(SPLITS_DIR / f"loro_{region}_val.csv",   index=False)
        loro.append(dict(held_out=region, train=int(len(tr_idx)), val=int(len(te_idx))))
        print(f"LORO {region}: train {len(tr_idx)} | val {len(te_idx)}")
    summary["loro"] = loro

    # ── Leave-one-zone-out (climate-zone-level) ───────────────────────────────
    # Requires a 'climate_zone' column in the manifest (added by ee_pull v2)
    _RGN2 = yaml.safe_load((_ROOT / "configs" / "regions.yaml").read_text())
    zone_map = {k: v.get("climate_zone", "unknown") for k, v in _RGN2["regions"].items()}

    if "climate_zone" not in df.columns:
        df["climate_zone"] = df["region"].map(zone_map).fillna("unknown")

    lozo = []
    for zone in df["climate_zone"].unique():
        te_idx = np.where(df["climate_zone"].values == zone)[0]
        tr_idx = np.where(df["climate_zone"].values != zone)[0]
        df.iloc[tr_idx].to_csv(SPLITS_DIR / f"lozo_{zone}_train.csv", index=False)
        df.iloc[te_idx].to_csv(SPLITS_DIR / f"lozo_{zone}_val.csv",   index=False)
        lozo.append(dict(held_out_zone=zone, train=int(len(tr_idx)), val=int(len(te_idx))))
        print(f"LOZO {zone}: train {len(tr_idx)} | val {len(te_idx)}")
    summary["lozo"] = lozo

    # ── Save reference features for Evidently ────────────────────────────────
    ref_path = SPLITS_DIR / "reference_features.parquet"
    if summary["folds"]:
        first_fold_train_indices = list(range(summary["folds"][0]["train"]))
        df.iloc[first_fold_train_indices].to_parquet(ref_path, index=False)
    else:
        df.to_parquet(ref_path, index=False)

    with open(SPLITS_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSplits saved to {SPLITS_DIR}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="propertysatellite", help="GCP project for Earth Engine")
    args = parser.parse_args()
    build_splits(project=args.project)

