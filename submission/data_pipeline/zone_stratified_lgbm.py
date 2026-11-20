"""
data/pipeline/zone_stratified_lgbm.py
─────────────────────────────────────
Zone-Stratified LightGBM Training & Leave-One-Zone-Out (LOZO) Evaluation.

Implements Phase 3.1 & Phase 4 of the Kenya Land Readiness Atlas:
  1. Climate-Zone Mapping (Highland, Coastal, ASAL, Lake Basin, Rift Valley).
  2. ASAL-specific prevalence weighting and class oversampling (3x for rare built_up/water).
  3. Leave-One-Zone-Out (LOZO) cross-validation across all 5 agro-ecological zones.
  4. Exports zone models (`models/lgbm_zone_{zone}.pkl`) and MILP confidence parameters (`data/atlas/zone_confidence.json`).

Usage:
  python -m data.pipeline.zone_stratified_lgbm
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CFG = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
_RGN = yaml.safe_load((_ROOT / "configs" / "regions.yaml").read_text())

RAW_CSV = _ROOT / _CFG["paths"]["raw_csv"]
MODELS_DIR = _ROOT / "models"
ATLAS_DIR = _ROOT / "data" / "atlas"
SUMMARIES_DIR = _ROOT / "data" / "county_summaries"

LGB_PARAMS = {
    "n_estimators": 250,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

EXCLUDE_COLS = {
    "label", "region", "lon", "lat", "weight", "climate_zone",
    "bx", "by", "block", "filename", "sub_idx", "row_id",
}


def load_and_prepare_data(csv_path: Path | str) -> Tuple[pd.DataFrame, List[str]]:
    """Loads dataset and assigns climate zones if missing."""
    p = Path(csv_path)
    if not p.exists():
        # Fallback to Google Drive path
        drive_path = Path("/content/drive/MyDrive/land_atlas_baseline") / p.name
        if drive_path.exists():
            p = drive_path
        else:
            raise FileNotFoundError(f"Cannot find dataset at {p} or {drive_path}")

    df = pd.read_csv(p)

    # Assign climate zones from regions.yaml if missing
    zone_map = {k: v.get("climate_zone", "unknown") for k, v in _RGN["regions"].items()}
    if "climate_zone" not in df.columns and "region" in df.columns:
        df["climate_zone"] = df["region"].map(zone_map).fillna("unknown")

    # In-situ index generation if missing
    eps = 1e-6
    if "NDVI" not in df.columns and {"B8", "B4"}.issubset(df.columns):
        df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + eps)
    if "NDWI" not in df.columns and {"B3", "B8"}.issubset(df.columns):
        df["NDWI"] = (df["B3"] - df["B8"]) / (df["B3"] + df["B8"] + eps)
    if "NDBI" not in df.columns and {"B11", "B8"}.issubset(df.columns):
        df["NDBI"] = (df["B11"] - df["B8"]) / (df["B11"] + df["B8"] + eps)
    if "MNDWI" not in df.columns and {"B3", "B11"}.issubset(df.columns):
        df["MNDWI"] = (df["B3"] - df["B11"]) / (df["B3"] + df["B11"] + eps)

    # In-situ BSI, SAVI, EVI, NDRE
    if "BSI" not in df.columns and {"B11", "B4", "B8", "B2"}.issubset(df.columns):
        scale = 10000.0 if np.nanmedian(df["B8"]) > 10.0 else 1.0
        b2 = df["B2"].values / scale
        b4 = df["B4"].values / scale
        b8 = df["B8"].values / scale
        b11 = df["B11"].values / scale
        df["BSI"] = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)
        df["SAVI"] = ((b8 - b4) / (b8 + b4 + 0.5 + eps)) * 1.5

    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS and np.issubdtype(df[c].dtype, np.number)
    ]

    return df, sorted(feature_cols)


def run_lozo_evaluation(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Evaluates Leave-One-Zone-Out (LOZO) cross-validation across all climate zones.
    """
    zones = [z for z in df["climate_zone"].unique() if z != "unknown"]
    print(f"\n========================================================")
    print(f"📊 LEAVE-ONE-ZONE-OUT (LOZO) CROSS-VALIDATION")
    print(f"   Testing geographic generalisation across {len(zones)} climatic zones")
    print(f"   Features ({len(feature_cols)}): {feature_cols}")
    print(f"========================================================")

    results = []

    for zone in zones:
        val_mask = (df["climate_zone"] == zone).values
        train_mask = ~val_mask

        X_train = df.loc[train_mask, feature_cols].fillna(0).values
        y_train = df.loc[train_mask, "label"].values.astype(int)

        X_val = df.loc[val_mask, feature_cols].fillna(0).values
        y_val = df.loc[val_mask, "label"].values.astype(int)

        # Apply sample weighting / prevalence correction
        sample_weights = np.ones(len(y_train), dtype=np.float32)
        if "weight" in df.columns:
            raw_w = df.loc[train_mask, "weight"].values
            sample_weights = np.clip(raw_w / (raw_w.mean() + 1e-6), 0.1, 10.0)

        # Train model for this LOZO fold
        clf = lgb.LGBMClassifier(**LGB_PARAMS)
        clf.fit(X_train, y_train, sample_weight=sample_weights)

        preds = clf.predict(X_val)
        mf1 = f1_score(y_val, preds, average="macro", zero_division=0)
        acc = accuracy_score(y_val, preds)

        # Per-class F1 for error diagnosis
        class_f1s = f1_score(y_val, preds, average=None, zero_division=0)
        classes = ["built_up", "cropland", "trees", "grass_shrub", "water", "bare"]
        per_class_str = ", ".join(f"{c[:4]}:{f1:.2f}" for c, f1 in zip(classes, class_f1s))

        print(f"  LOZO Held-out: {zone:<15} | Macro-F1: {mf1:.4f} | Accuracy: {acc*100:.2f}% | [{per_class_str}]")

        results.append({
            "climate_zone": zone,
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "macro_f1": round(float(mf1), 4),
            "accuracy": round(float(acc), 4),
            "f1_built_up": round(float(class_f1s[0]) if len(class_f1s) > 0 else 0.0, 4),
            "f1_cropland": round(float(class_f1s[1]) if len(class_f1s) > 1 else 0.0, 4),
            "f1_trees": round(float(class_f1s[2]) if len(class_f1s) > 2 else 0.0, 4),
            "f1_grass_shrub": round(float(class_f1s[3]) if len(class_f1s) > 3 else 0.0, 4),
            "f1_water": round(float(class_f1s[4]) if len(class_f1s) > 4 else 0.0, 4),
            "f1_bare": round(float(class_f1s[5]) if len(class_f1s) > 5 else 0.0, 4),
        })

    lozo_df = pd.DataFrame(results)
    mean_f1 = lozo_df["macro_f1"].mean()
    print("--------------------------------------------------------")
    print(f"🌟 LOZO Mean Macro-F1 across Kenya: {mean_f1:.4f}")
    print("========================================================\n")

    return lozo_df


def train_zone_stratified_models(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, lgb.LGBMClassifier]:
    """
    Trains distinct specialized models for each climate zone.
    For the ASAL zone, applies 3x oversampling of rare built_up and water classes.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    zones = [z for z in df["climate_zone"].unique() if z != "unknown"]
    zone_models = {}

    print("========================================================")
    print("🌲 TRAINING ZONE-STRATIFIED PRODUCTION MODELS")
    print("========================================================")

    for zone in zones:
        zone_df = df[df["climate_zone"] == zone].copy()

        # ASAL special handling: 3x oversampling for rare built_up (0) and water (4)
        if zone == "asal_zone":
            rare_built = zone_df[zone_df["label"] == 0]
            rare_water = zone_df[zone_df["label"] == 4]
            if len(rare_built) > 0 and len(rare_water) > 0:
                print(f"  [ASAL Tuning] Applying 3x minority oversampling for built_up and water...")
                zone_df = pd.concat([zone_df, rare_built, rare_built, rare_water, rare_water], ignore_index=True)

        X_z = zone_df[feature_cols].fillna(0).values
        y_z = zone_df["label"].values.astype(int)

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_z, y_z)

        model_path = MODELS_DIR / f"lgbm_zone_{zone}.pkl"
        joblib.dump(model, model_path)
        zone_models[zone] = model
        print(f"  ✅ Saved Zone Model: {model_path.name:<25} ({len(X_z):,} samples trained)")

    return zone_models


def export_milp_confidence_weights(lozo_df: pd.DataFrame) -> Path:
    """
    Exports zone_confidence.json mapping each climate zone to its LOZO F1 score.
    Used by Phase 9 MILP housing allocation to weight land readiness constraints.
    """
    confidence_map = {
        row["climate_zone"]: round(float(row["macro_f1"]), 4)
        for _, row in lozo_df.iterrows()
    }

    out_path = ATLAS_DIR / "zone_confidence.json"
    with open(out_path, "w") as f:
        json.dump(confidence_map, f, indent=2)
    print(f"✅ Saved MILP zone confidence weights -> {out_path}")
    return out_path


def run_pipeline_zone_stratification(csv_path: Path | str | None = None):
    """Executes the full zone-stratified LightGBM pipeline."""
    target_csv = csv_path if csv_path else RAW_CSV
    df, feature_cols = load_and_prepare_data(target_csv)

    # 1. Run LOZO Cross-Validation
    lozo_df = run_lozo_evaluation(df, feature_cols)
    lozo_path = SUMMARIES_DIR / "lozo_results.csv"
    lozo_df.to_csv(lozo_path, index=False)
    print(f"✅ Saved LOZO results -> {lozo_path}")

    # 2. Train and export zone-stratified models
    train_zone_stratified_models(df, feature_cols)

    # 3. Export MILP zone confidence weights
    export_milp_confidence_weights(lozo_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Zone-Stratified LightGBM and evaluate LOZO")
    parser.add_argument("--csv", default=None, help="Path to input samples CSV")
    args = parser.parse_args()
    run_pipeline_zone_stratification(args.csv)

