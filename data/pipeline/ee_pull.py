"""
data/pipeline/ee_pull.py
────────────────────────
Pulls Sentinel-2 spectral features + WorldCover 2021 labels from Google Earth
Engine for all regions defined in configs/regions.yaml.

Outputs a CSV to paths.raw_csv (cached; re-run is a no-op if cache exists).

Run from repo root:
    python -m data.pipeline.ee_pull --project YOUR_GCP_PROJECT
"""

import argparse
import os
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import yaml

# ── Load config ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_CFG  = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
_RGN  = yaml.safe_load((_ROOT / "configs" / "regions.yaml").read_text())

YEAR        = _CFG["training"]["year"]
PER_CLASS   = _CFG["training"]["per_class_per_region"]
CLOUD_THRESH= _CFG["training"]["cloud_thresh"]
SEED        = _CFG["training"]["seed"]
CLASSES     = _RGN["classes"]
K           = len(CLASSES)
REMAP_FROM  = _RGN["worldcover_remap"]["from"]
REMAP_TO    = _RGN["worldcover_remap"]["to"]
REGIONS     = {k: v["bbox"] for k, v in _RGN["regions"].items()}
OUT_CSV     = _ROOT / _CFG["paths"]["raw_csv"]

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
IDX   = ["NDVI", "NDWI", "MNDWI", "NDBI"]
EXTRA = ["NDVI_p10", "NDVI_p90", "NDVI_stdDev", "elev", "slope"]
FEATURES = BANDS + IDX + EXTRA


# ── Earth Engine helpers ──────────────────────────────────────────────────────

def _retry(fn, tries: int = 3, wait: float = 10):
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:
            print(f"  retry {i+1}/{tries}: {str(exc)[:120]}")
            time.sleep(wait)
    return fn()


def _label_image() -> ee.Image:
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    return wc.remap(REMAP_FROM, REMAP_TO).rename("label").toInt()


def _feature_image(geom: ee.Geometry) -> ee.Image:
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
        .filterBounds(geom)
        .linkCollection(csp, ["cs_cdf"])
        .map(lambda im: im.updateMask(im.select("cs_cdf").gte(CLOUD_THRESH)))
    )

    def _add_idx(im):
        im = im.select(BANDS).divide(10000)
        ndvi  = im.normalizedDifference(["B8",  "B4" ]).rename("NDVI")
        ndwi  = im.normalizedDifference(["B3",  "B8" ]).rename("NDWI")
        mndwi = im.normalizedDifference(["B3",  "B11"]).rename("MNDWI")
        ndbi  = im.normalizedDifference(["B11", "B8" ]).rename("NDBI")
        return im.addBands([ndvi, ndwi, mndwi, ndbi])

    col        = s2.map(_add_idx)
    med        = col.select(BANDS + IDX).median()
    ndvi_stats = col.select("NDVI").reduce(
        ee.Reducer.percentile([10, 90]).combine(
            ee.Reducer.stdDev(), sharedInputs=True
        )
    )
    elev  = ee.Image("USGS/SRTMGL1_003").rename("elev")
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).rename("slope")
    return med.addBands(ndvi_stats).addBands(elev).addBands(slope).select(FEATURES)


def _pull_region(name: str, bbox: list) -> pd.DataFrame:
    geom = ee.Geometry.Rectangle(bbox)
    img  = _feature_image(geom).addBands(_label_image())

    fc = img.stratifiedSample(
        numPoints=0, classBand="label", region=geom, scale=10, seed=SEED,
        classValues=list(range(K)), classPoints=[PER_CLASS] * K,
        dropNulls=True, tileScale=4, geometries=True,
    )
    feats = _retry(lambda: fc.getInfo())["features"]
    rows  = [
        {**f["properties"],
         "lon": f["geometry"]["coordinates"][0],
         "lat": f["geometry"]["coordinates"][1]}
        for f in feats
    ]
    df = pd.DataFrame(rows)
    df["region"] = name

    # Area-weighted sample weight (Olofsson-style)
    hist = _retry(lambda: _label_image().reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(), geometry=geom, scale=100,
        maxPixels=int(1e10), bestEffort=True, tileScale=4,
    ).get("label").getInfo())
    share  = {int(float(k)): v for k, v in hist.items()}
    n_samp = df["label"].value_counts().to_dict()
    present= {c: share.get(c, 0) for c in n_samp}
    tot    = sum(present.values())
    df["weight"] = df["label"].map(
        lambda c: (present[c] / tot) / n_samp[c] if tot > 0 else 1.0
    )
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def pull(project: str, force: bool = False) -> pd.DataFrame:
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    print("Earth Engine ready")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if OUT_CSV.exists() and not force:
        df = pd.read_csv(OUT_CSV)
        print(f"Cache hit: {OUT_CSV}  ({df.shape[0]} rows)")
    else:
        parts = []
        for name, bbox in REGIONS.items():
            t0 = time.time()
            print(f"Sampling {name} …")
            parts.append(_pull_region(name, bbox))
            print(f"  {len(parts[-1])} rows  ({time.time()-t0:.0f}s)")
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(OUT_CSV, index=False)
        print(f"Saved {OUT_CSV}  ({df.shape[0]} rows)")

    df = df.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP project registered for EE")
    parser.add_argument("--force",   action="store_true", help="Ignore cache and re-pull")
    args = parser.parse_args()
    df = pull(args.project, args.force)
    print(df.groupby(["region", "label"]).size().unstack(fill_value=0))
