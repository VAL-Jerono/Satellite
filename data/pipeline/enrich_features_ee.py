"""
data/pipeline/enrich_features_ee.py
───────────────────────────────────
Non-destructive Feature Enrichment via Google Earth Engine.

Extracts the missing 4 multi-temporal & hydrological features for the EXACT
28,121 coordinates in samples_2021_n400.csv WITHOUT re-sampling points or
invalidating existing .npy patch files.

Missing features added:
  1. ndvi_wet            - S2 median NDVI during long rains (March 1 - June 1, 2021)
  2. ndvi_dry            - S2 median NDVI during dry season (July 1 - October 1, 2021)
  3. ndvi_seasonal_swing - ndvi_wet - ndvi_dry (cropland vs forest vs bare discriminant)
  4. twi                 - Topographic Wetness Index (HydroSHEDS 15ACC + SRTM slope)

Also computes the 4 in-situ spectral indices from existing bands (B2..B12):
  5. BSI   - Bare Soil Index (disambiguates bare rock from built-up in ASAL)
  6. SAVI  - Soil-Adjusted Vegetation Index (sparse vegetation calibration)
  7. EVI   - Enhanced Vegetation Index
  8. NDRE  - Red-Edge NDVI (chlorophyll absorption)

Usage:
  python -m data.pipeline.enrich_features_ee --project propertysatellite
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CFG = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
_RGN = yaml.safe_load((_ROOT / "configs" / "regions.yaml").read_text())

YEAR = _CFG["training"]["year"]
CLOUD_THRESH = _CFG["training"]["cloud_thresh"]
MASTER_CSV = _ROOT / _CFG["paths"]["raw_csv"]
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def _build_ee_seasonal_image() -> "ee.Image":
    """Constructs the 4-band seasonal and hydrological image in GEE."""
    import ee

    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    def _masked_collection(start_date: str, end_date: str) -> "ee.ImageCollection":
        return (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .linkCollection(csp, ["cs_cdf"])
            .map(lambda im: im.updateMask(im.select("cs_cdf").gte(CLOUD_THRESH)))
        )

    # Seasonal collections (Kenya: wet = Mar-May, dry = Jul-Sep)
    s2_wet = _masked_collection(f"{YEAR}-03-01", f"{YEAR}-06-01")
    s2_dry = _masked_collection(f"{YEAR}-07-01", f"{YEAR}-10-01")
    s2_annual = _masked_collection(f"{YEAR}-01-01", f"{YEAR+1}-01-01")

    def _calc_ndvi(im):
        b8 = im.select("B8").divide(10000.0)
        b4 = im.select("B4").divide(10000.0)
        return b8.subtract(b4).divide(b8.add(b4).add(1e-6)).rename("NDVI")

    # Wet and dry NDVI medians
    ndvi_wet = s2_wet.map(_calc_ndvi).median().rename("ndvi_wet")
    ndvi_dry = s2_dry.map(_calc_ndvi).median().rename("ndvi_dry")

    # Cloud fallback to annual median NDVI
    annual_ndvi = s2_annual.map(_calc_ndvi).median()
    ndvi_wet = ndvi_wet.unmask(annual_ndvi)
    ndvi_dry = ndvi_dry.unmask(annual_ndvi)
    ndvi_swing = ndvi_wet.subtract(ndvi_dry).rename("ndvi_seasonal_swing")

    # Topographic Wetness Index (HydroSHEDS + SRTM)
    elev = ee.Image("USGS/SRTMGL1_003").rename("elev")
    slope_deg = ee.Terrain.slope(elev).rename("slope")
    slope_rad = slope_deg.multiply(3.14159265 / 180.0).add(0.001)
    flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").rename("flow_acc").toFloat()
    twi = flow_acc.add(1).log().divide(slope_rad).rename("twi")

    return ndvi_wet.addBands(ndvi_dry).addBands(ndvi_swing).addBands(twi)


def compute_spectral_indices_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Computes BSI, SAVI, EVI, and NDRE from existing band columns."""
    df = df.copy()
    eps = 1e-6

    # Normalise bands to reflectances if > 1 (e.g. 0..10000 scale)
    b2 = df["B2"].values
    b3 = df["B3"].values
    b4 = df["B4"].values
    b5 = df["B5"].values
    b8 = df["B8"].values
    b11 = df["B11"].values
    scale = 10000.0 if np.nanmedian(b8) > 10.0 else 1.0

    b2, b3, b4, b5, b8, b11 = (b2 / scale, b3 / scale, b4 / scale, b5 / scale, b8 / scale, b11 / scale)

    # Bare Soil Index
    df["BSI"] = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)
    # Soil-Adjusted Vegetation Index (L=0.5)
    df["SAVI"] = ((b8 - b4) / (b8 + b4 + 0.5 + eps)) * 1.5
    # Enhanced Vegetation Index
    evi_denom = b8 + 6.0 * b4 - 7.5 * b2 + 1.0
    df["EVI"] = np.clip(2.5 * (b8 - b4) / (evi_denom + eps), -2.0, 2.0)
    # Red-Edge NDVI
    df["NDRE"] = (b8 - b5) / (b8 + b5 + eps)

    return df


def enrich_dataset_from_ee(
    input_csv: Path | str,
    project: str,
    output_csv: Path | str | None = None,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """
    Enriches an existing point CSV by sampling seasonal & TWI features from GEE
    at the exact (lon, lat) of each row.
    """
    import ee

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"Loading base dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Total rows to enrich: {len(df):,}")

    # Check if already enriched
    target_cols = ["ndvi_wet", "ndvi_dry", "ndvi_seasonal_swing", "twi"]
    if all(col in df.columns for col in target_cols):
        print("✅ Dataset already contains all 4 seasonal & TWI features!")
        df = compute_spectral_indices_dataframe(df)
        if output_csv:
            df.to_csv(output_csv, index=False)
        return df

    # Connect to Earth Engine
    try:
        ee.Initialize(project=project)
        print("✅ Earth Engine initialized successfully")
    except Exception:
        try:
            ee.Authenticate(auth_mode="notebook")
            ee.Initialize(project=project)
            print("✅ Earth Engine initialized via notebook auth")
        except Exception as exc:
            raise RuntimeError(f"Earth Engine authentication failed: {exc}")

    ee_img = _build_ee_seasonal_image()

    # Sample points county-by-county to respect EE memory limits
    enriched_parts = []
    unique_regions = df["region"].unique() if "region" in df.columns else ["all"]

    for r_idx, region in enumerate(unique_regions):
        sub_df = df[df["region"] == region].copy() if "region" in df.columns else df.copy()
        print(f"\nProcessing region [{r_idx+1}/{len(unique_regions)}]: {region} ({len(sub_df):,} points)...")

        t0 = time.time()
        # Split into manageable batches of points
        num_batches = int(np.ceil(len(sub_df) / batch_size))
        region_results = []

        for b in range(num_batches):
            b_df = sub_df.iloc[b * batch_size : (b + 1) * batch_size]
            features = []
            for _, row in b_df.iterrows():
                geom = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
                features.append(ee.Feature(geom, {"row_id": int(row.name)}))

            fc = ee.FeatureCollection(features)
            sampled = ee_img.sampleRegions(
                collection=fc,
                scale=10,
                tileScale=16,
                geometries=False,
            )

            res = sampled.getInfo()["features"]
            res_dict = {f["properties"]["row_id"]: f["properties"] for f in res}

            # Map sampled values back to dataframe rows
            b_extracted = []
            for row_id in b_df.index:
                p = res_dict.get(row_id, {})
                b_extracted.append({
                    "ndvi_wet": p.get("ndvi_wet", np.nan),
                    "ndvi_dry": p.get("ndvi_dry", np.nan),
                    "ndvi_seasonal_swing": p.get("ndvi_seasonal_swing", np.nan),
                    "twi": p.get("twi", np.nan),
                })
            region_results.append(pd.DataFrame(b_extracted, index=b_df.index))

        res_df = pd.concat(region_results)
        for col in target_cols:
            sub_df[col] = res_df[col].values

        enriched_parts.append(sub_df)
        print(f"  -> {region} completed in {time.time()-t0:.1f}s")

    enriched_df = pd.concat(enriched_parts).sort_index()

    # Fill any sparse missing pixels using annual NDVI or county median
    if "NDVI" in enriched_df.columns:
        enriched_df["ndvi_wet"] = enriched_df["ndvi_wet"].fillna(enriched_df["NDVI"])
        enriched_df["ndvi_dry"] = enriched_df["ndvi_dry"].fillna(enriched_df["NDVI"])
        enriched_df["ndvi_seasonal_swing"] = enriched_df["ndvi_seasonal_swing"].fillna(0.0)
    enriched_df["twi"] = enriched_df["twi"].fillna(enriched_df["twi"].median())

    # Add in-situ spectral indices (BSI, SAVI, EVI, NDRE)
    enriched_df = compute_spectral_indices_dataframe(enriched_df)

    out_file = Path(output_csv) if output_csv else input_path
    enriched_df.to_csv(out_file, index=False)
    print(f"\n✅ Enriched dataset saved successfully -> {out_file} ({len(enriched_df):,} rows, {len(enriched_df.columns)} columns)")

    return enriched_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich dataset with GEE seasonal phenology and TWI features")
    parser.add_argument("--project", default="propertysatellite", help="GCP Project for Earth Engine")
    parser.add_argument("--input", default=str(MASTER_CSV), help="Path to input master CSV")
    parser.add_argument("--output", default=None, help="Path to output CSV (defaults to overwriting input)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for EE sampleRegions")
    args = parser.parse_args()

    enrich_dataset_from_ee(
        input_csv=args.input,
        project=args.project,
        output_csv=args.output,
        batch_size=args.batch_size,
    )

