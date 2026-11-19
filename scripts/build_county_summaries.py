"""
scripts/build_county_summaries.py
───────────────────────────────────
Generates per-county JSON summary files and all_counties.csv from the
enriched master CSV + zone-stratified LightGBM models.

Outputs (all git-ignored but needed locally for the API):
  data/county_summaries/all_counties.csv
  data/county_summaries/loro_results.csv   (already present — not touched)
  data/county_summaries/{county_name}.json  (one per county, used by /predict/county)
  data/splits/reference_features.parquet   (used by Evidently drift monitoring)

Runtime: ~15 seconds on CPU (27-feature tabular inference on 28k rows)
Run:  python3 scripts/build_county_summaries.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "data" / "county_summaries"
ATLAS_DIR = ROOT / "data" / "atlas"
SPLITS_DIR = ROOT / "data" / "splits"
RAW_CSV = ROOT / "data" / "raw" / "samples_2021_n400.csv"
RGN_CFG = yaml.safe_load((ROOT / "configs" / "regions.yaml").read_text())

OUT_DIR.mkdir(parents=True, exist_ok=True)
ATLAS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = RGN_CFG["classes"]  # ['built_up','cropland','trees','grass_shrub','water','bare']

EXCLUDE = {
    "label", "region", "lon", "lat", "weight", "climate_zone",
    "bx", "by", "block", "filename", "sub_idx", "row_id"
}

# ── 1. Load dataset ────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(RAW_CSV)
assert len(df) == 28121, f"Expected 28121 rows, got {len(df)}"

FEATURE_COLS = sorted([
    c for c in df.columns
    if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
])
print(f"✅ {len(df):,} rows × {len(FEATURE_COLS)} features")

# Map climate zones if missing
zone_map = {k: v.get("climate_zone", "unknown") for k, v in RGN_CFG["regions"].items()}
if "climate_zone" not in df.columns:
    df["climate_zone"] = df["region"].map(zone_map).fillna("unknown")

# ── 2. Load zone models ────────────────────────────────────────────────────────
print("\nLoading zone models...")
zone_models = {}
for pkl in sorted(MODELS_DIR.glob("lgbm_zone_*.pkl")):
    zone_name = pkl.stem.replace("lgbm_zone_", "")
    zone_models[zone_name] = joblib.load(pkl)
    print(f"  ✅ {pkl.name}")

# Load zone confidence weights
confidence_path = ATLAS_DIR / "zone_confidence.json"
zone_weights = {}
if confidence_path.exists():
    zone_weights = json.loads(confidence_path.read_text())
    print(f"  ✅ Loaded zone_confidence.json")

# ── 3. Run zone-routed inference on all 28,121 points ─────────────────────────
print("\nRunning zone-routed inference on all 28,121 points...")
all_probs = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
all_preds = np.zeros(len(df), dtype=int)

X_all = df[FEATURE_COLS].fillna(0).values.astype(np.float32)

for zone, model in zone_models.items():
    zone_mask = (df["climate_zone"] == zone).values
    if zone_mask.sum() == 0:
        continue
    X_zone = X_all[zone_mask]
    probs = model.predict_proba(X_zone).astype(np.float32)
    preds = model.predict(X_zone).astype(int)
    all_probs[zone_mask] = probs
    all_preds[zone_mask] = preds
    print(f"  → {zone}: {zone_mask.sum():,} points inferred")

# Handle any rows with unknown zone — use first available model as fallback
unknown_mask = (df["climate_zone"] == "unknown").values | \
    ~df["climate_zone"].isin(zone_models.keys()).values
if unknown_mask.sum() > 0:
    fallback_model = list(zone_models.values())[0]
    all_probs[unknown_mask] = fallback_model.predict_proba(X_all[unknown_mask]).astype(np.float32)
    all_preds[unknown_mask] = fallback_model.predict(X_all[unknown_mask]).astype(int)
    print(f"  → fallback: {unknown_mask.sum():,} points")

df["predicted_class"] = [CLASSES[p] for p in all_preds]
for i, cls in enumerate(CLASSES):
    df[f"prob_{cls}"] = all_probs[:, i]

print(f"✅ Inference complete")

# ── 4. Build per-county summaries ──────────────────────────────────────────────
print("\nBuilding county summaries...")
county_rows = []

for county, cdf in df.groupby("region"):
    zone = cdf["climate_zone"].iloc[0]
    total = len(cdf)

    # Land-cover share by area (% of classified points)
    cover_shares = (cdf["predicted_class"].value_counts() / total * 100).round(2).to_dict()
    cover_shares_full = {cls: round(cover_shares.get(cls, 0.0), 2) for cls in CLASSES}

    # True label distribution (from WorldCover ground truth)
    label_map = {i: cls for i, cls in enumerate(CLASSES)}
    true_shares = (cdf["label"].map(label_map).value_counts() / total * 100).round(2).to_dict()
    true_shares_full = {cls: round(true_shares.get(cls, 0.0), 2) for cls in CLASSES}

    # Settlement readiness score: settle preference = cropland + bare (brownfield) on flat land
    settle_score = round(
        cover_shares_full["bare"] * 0.6 + cover_shares_full["cropland"] * 0.3 +
        cover_shares_full["grass_shrub"] * 0.1, 2
    )
    # Farm suitability: existing cropland + flat grass/shrub
    farm_score = round(
        cover_shares_full["cropland"] * 0.7 + cover_shares_full["grass_shrub"] * 0.3, 2
    )
    # Protection priority: trees + water (ecosystem services)
    protect_score = round(
        cover_shares_full["trees"] * 0.6 + cover_shares_full["water"] * 0.4, 2
    )

    # Recommendation
    scores = {"settle": settle_score, "farm": farm_score, "protect": protect_score}
    recommendation = max(scores, key=scores.get)

    # Zone confidence weight
    weight = zone_weights.get(zone, 0.5)

    row = {
        "county": county,
        "climate_zone": zone,
        "n_samples": total,
        "zone_confidence": weight,
        "recommendation": recommendation,
        "settle_score": settle_score,
        "farm_score": farm_score,
        "protect_score": protect_score,
        **{f"pred_{cls}": cover_shares_full[cls] for cls in CLASSES},
        **{f"true_{cls}": true_shares_full[cls] for cls in CLASSES},
    }
    county_rows.append(row)

    # Per-county JSON for /predict/county/{name} endpoint
    county_json = {
        "county": county,
        "climate_zone": zone,
        "n_samples": total,
        "zone_confidence": weight,
        "recommendation": recommendation,
        "scores": scores,
        "predicted_cover_pct": cover_shares_full,
        "true_cover_pct": true_shares_full,
    }
    out_path = OUT_DIR / f"{county.lower()}.json"
    out_path.write_text(json.dumps(county_json, indent=2))

all_counties_df = pd.DataFrame(county_rows).sort_values("county")
all_counties_df.to_csv(OUT_DIR / "all_counties.csv", index=False)
print(f"✅ Saved all_counties.csv ({len(all_counties_df)} counties)")
print(f"✅ Saved {len(all_counties_df)} per-county JSON files")

# ── 5. Build reference parquet for Evidently drift monitoring ──────────────────
print("\nBuilding drift reference parquet...")
ref_df = df[FEATURE_COLS + ["label"]].copy()
ref_df.to_parquet(SPLITS_DIR / "reference_features.parquet", index=False)
print(f"✅ Saved reference_features.parquet ({len(ref_df):,} rows, {len(FEATURE_COLS)} features)")

# ── 6. Print summary table ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("COUNTY LAND-COVER ALLOCATION SUMMARY")
print("=" * 72)
print(f"{'County':<14} {'Zone':<16} {'Reco':<8} {'Settle':>7} {'Farm':>7} {'Protect':>8} {'Conf':>6}")
print("-" * 72)
for _, r in all_counties_df.iterrows():
    print(f"{r['county']:<14} {r['climate_zone']:<16} {r['recommendation']:<8} "
          f"{r['settle_score']:>7.1f} {r['farm_score']:>7.1f} "
          f"{r['protect_score']:>8.1f} {r['zone_confidence']:>6.3f}")
print("=" * 72)
print("\n✅ All county summaries built successfully!")

