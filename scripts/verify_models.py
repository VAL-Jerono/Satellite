"""
scripts/verify_models.py
─────────────────────────
Verify all zone LightGBM models load correctly and feature columns match.
Run: python3 scripts/verify_models.py
"""
import joblib
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {
    "label", "region", "lon", "lat", "weight", "climate_zone",
    "bx", "by", "block", "filename", "sub_idx", "row_id"
}

df = pd.read_csv(ROOT / "data/raw/samples_2021_n400.csv")
FEATURE_COLS = sorted([
    c for c in df.columns
    if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
])

print(f"✅ Dataset: {len(df):,} rows × {len(FEATURE_COLS)} features")
print(f"   Features: {FEATURE_COLS}\n")

all_ok = True
for pkl in sorted((ROOT / "models").glob("lgbm_zone_*.pkl")):
    m = joblib.load(pkl)
    n_ok = m.n_features_in_ == len(FEATURE_COLS)
    status = "✅" if n_ok else "❌"
    print(f"{status} {pkl.name}: classes={m.n_classes_}, n_features={m.n_features_in_}")
    if not n_ok:
        all_ok = False

print()
if all_ok:
    print("✅ All models verified — ready for deployment.")
else:
    print("❌ Feature mismatch — retrain before deploying.")
    sys.exit(1)

