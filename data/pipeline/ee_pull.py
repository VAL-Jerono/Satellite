"""
data/pipeline/ee_pull.py
────────────────────────
Pulls Sentinel-2 spectral features + WorldCover 2021 labels from Google Earth
Engine for any subset of the 10 regions defined in configs/regions.yaml.

Outputs per-county chunk CSVs under data/raw/chunks_YEAR_nN/.
A master CSV is assembled by run_pipeline.py after all counties are done.
Progress is tracked in data/progress/ledger.json (never starts from scratch).

Run from repo root:
    # Pull all 10 counties (2 at a time via orchestrator):
    python -m data.pipeline.run_pipeline --project YOUR_GCP_PROJECT

    # Or pull specific counties directly:
    python -m data.pipeline.ee_pull --project YOUR_GCP_PROJECT --counties highland muranga
"""

import argparse
import os
import time
from pathlib import Path

# NOTE: `import ee` is intentionally deferred to inside functions so that
# importing this module never crashes when earthengine-api is not installed
# in the current Python environment (e.g. system Python on Colab).
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
REGIONS      = {k: v["bbox"]         for k, v in _RGN["regions"].items()}
CLIMATE_ZONE = {k: v.get("climate_zone", "unknown") for k, v in _RGN["regions"].items()}
OUT_CSV      = _ROOT / _CFG["paths"]["raw_csv"]

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


def _label_image() -> "ee.Image":
    import ee
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    return wc.remap(REMAP_FROM, REMAP_TO).rename("label").toInt()


def _feature_image(geom: "ee.Geometry") -> "ee.Image":
    import ee
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
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
        ee.Reducer.minMax().combine(ee.Reducer.stdDev(), sharedInputs=True)
    ).rename(["NDVI_p10", "NDVI_p90", "NDVI_stdDev"])
    elev  = ee.Image("USGS/SRTMGL1_003").rename("elev")
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).rename("slope")
    return med.addBands(ndvi_stats).addBands(elev).addBands(slope).select(FEATURES)


CHUNK_DIR = OUT_CSV.parent / f"chunks_{YEAR}_n{PER_CLASS}"

def _get_sub_bboxes(bbox: list, n_split: int = 4) -> list:
    min_x, min_y, max_x, max_y = bbox
    xs = np.linspace(min_x, max_x, n_split + 1)
    ys = np.linspace(min_y, max_y, n_split + 1)
    boxes = []
    for i in range(n_split):
        for j in range(n_split):
            boxes.append([xs[i], ys[j], xs[i+1], ys[j+1]])
    return boxes


def _pull_sub_tile(name: str, sub_idx: int, total_subs: int, sub_bbox: list, points_per_class: int) -> pd.DataFrame:
    import ee
    chunk_path = CHUNK_DIR / f"chunk_{name}_sub{sub_idx}.csv"
    if chunk_path.exists():
        d_cached = pd.read_csv(chunk_path)
        print(f"  [Cache hit] {name} sub-tile {sub_idx+1}/{total_subs} ({len(d_cached)} rows)")
        return d_cached

    t0 = time.time()
    print(f"  Sampling {name} sub-tile {sub_idx+1}/{total_subs} ...")
    geom = ee.Geometry.Rectangle(sub_bbox)
    img  = _feature_image(geom).addBands(_label_image())

    fc = img.stratifiedSample(
        numPoints=0, classBand="label", region=geom, scale=10, seed=SEED + sub_idx,
        classValues=list(range(K)), classPoints=[points_per_class] * K,
        dropNulls=True, tileScale=16, geometries=True,
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
    df["sub_idx"] = sub_idx
    df.to_csv(chunk_path, index=False)
    print(f"    -> Saved chunk: {chunk_path} ({len(df)} points in {time.time()-t0:.1f}s)")
    return df


def _pull_region(name: str, bbox: list, n_split: int = 4) -> pd.DataFrame:
    import ee
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    sub_bboxes = _get_sub_bboxes(bbox, n_split=n_split)
    total_subs = len(sub_bboxes)
    pts_per_sub = max(5, PER_CLASS // total_subs)

    parts = []
    for sub_i, sub_b in enumerate(sub_bboxes):
        parts.append(_pull_sub_tile(name, sub_i, total_subs, sub_b, pts_per_sub))
    df = pd.concat(parts, ignore_index=True)

    # Cached WorldCover class frequency histogram
    hist_path = CHUNK_DIR / f"hist_{name}.json"
    if hist_path.exists():
        import json
        with hist_path.open("r") as f:
            share = {int(k): v for k, v in json.load(f).items()}
    else:
        geom = ee.Geometry.Rectangle(bbox)
        hist = _retry(lambda: _label_image().reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(), geometry=geom, scale=100,
            maxPixels=int(1e10), bestEffort=True, tileScale=16,
        ).get("label").getInfo())
        share  = {int(float(k)): v for k, v in hist.items()}
        import json
        with hist_path.open("w") as f:
            json.dump(share, f)

    n_samp = df["label"].value_counts().to_dict()
    present= {c: share.get(c, 0) for c in n_samp}
    tot    = sum(present.values())
    df["weight"] = df["label"].map(
        lambda c: (present[c] / tot) / n_samp[c] if tot > 0 and n_samp.get(c, 0) > 0 else 1.0
    )
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def pull(
    project: str,
    force: bool = False,
    counties: "list[str] | None" = None,
    drive_dir: "Path | None" = None,
) -> pd.DataFrame:
    """
    Pull EE samples for the given counties (default: all 10).

    Parameters
    ----------
    project    : GCP project registered for Earth Engine
    force      : Ignore existing chunk CSVs and re-pull
    counties   : List of county names to pull (None = all)
    drive_dir  : Optional Google Drive backup directory
    """
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        # Credentials missing — attempt interactive auth (works in Colab)
        try:
            ee.Authenticate()
            ee.Initialize(project=project)
        except Exception as auth_exc:
            raise RuntimeError(
                f"Failed to initialize Earth Engine with project '{project}'.\n"
                "Please run `import ee; ee.Authenticate()` in a notebook cell first,\n"
                f"then retry. Original error: {auth_exc}"
            ) from auth_exc
    print("Earth Engine ready")

    target = {k: v for k, v in REGIONS.items() if counties is None or k in counties}
    if not target:
        raise ValueError(f"No valid counties in: {counties}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    for name, bbox in target.items():
        # Check if all sub-tile chunks already exist (skip if so and not forced)
        sub_bboxes = _get_sub_bboxes(bbox, n_split=4)
        chunks_done = all(
            (CHUNK_DIR / f"chunk_{name}_sub{i}.csv").exists()
            for i in range(len(sub_bboxes))
        )
        if chunks_done and not force:
            print(f"[Cache] All chunks exist for '{name}' — loading from disk")
            dfs = [pd.read_csv(CHUNK_DIR / f"chunk_{name}_sub{i}.csv")
                   for i in range(len(sub_bboxes))]
            parts.append(pd.concat(dfs, ignore_index=True))
        else:
            t0 = time.time()
            print(f"--- Sampling region: {name} ---")
            parts.append(_pull_region(name, bbox))
            print(f"Done {name}: {len(parts[-1])} total points in {time.time()-t0:.0f}s\n")

        # Drive backup of this county's chunks
        if drive_dir is not None:
            import shutil
            drive_chunks = Path(drive_dir) / f"chunks_{YEAR}_n{PER_CLASS}"
            drive_chunks.mkdir(parents=True, exist_ok=True)
            for cp in CHUNK_DIR.glob(f"chunk_{name}_*.csv"):
                target_cp = drive_chunks / cp.name
                if not target_cp.exists():
                    shutil.copy(cp, target_cp)
            hist_src = CHUNK_DIR / f"hist_{name}.json"
            if hist_src.exists():
                shutil.copy(hist_src, drive_chunks / hist_src.name)

    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pull EE samples for one or more Kenya counties."
    )
    parser.add_argument("--project",  required=True, help="GCP project registered for EE")
    parser.add_argument("--force",    action="store_true", help="Ignore cache and re-pull")
    parser.add_argument("--counties", nargs="+", default=None,
                        help="County names to pull (default: all 10)")
    args = parser.parse_args()
    df = pull(args.project, force=args.force, counties=args.counties)
    print(df.groupby(["region", "label"]).size().unstack(fill_value=0))
