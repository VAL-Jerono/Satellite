"""
data/pipeline/patch_export.py
──────────────────────────────
Takes the sampled point CSV and exports 64×64 pixel Sentinel-2 patches
(10 bands, 10 m resolution) centred on each point to local disk as .npy files.

Each patch is named: {region}_{label}_{idx}.npy
A manifest CSV (patches_dir/manifest.csv) maps filename → label, region, lon, lat, block.

This is the heavy step that makes the CNN training self-contained after this
point – no Earth Engine access needed during training.

Run from repo root:
    python -m data.pipeline.patch_export --project YOUR_GCP_PROJECT [--resume]
"""

import argparse
import ee
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = Path(__file__).resolve().parents[2]
_CFG  = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
_RGN  = yaml.safe_load((_ROOT / "configs" / "regions.yaml").read_text())

YEAR        = _CFG["training"]["year"]
CLOUD_THRESH= _CFG["training"]["cloud_thresh"]
PATCH_PX    = _CFG["training"]["patch_size"]          # 64
RESOLUTION  = 10                                       # metres
HALF_M      = (PATCH_PX // 2) * RESOLUTION            # 320 m half-width
SEED        = _CFG["training"]["seed"]
BANDS       = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
IN_CSV      = _ROOT / _CFG["paths"]["raw_csv"]
PATCHES_DIR = _ROOT / _CFG["paths"]["patches_dir"]


def _s2_composite(geom: ee.Geometry) -> ee.Image:
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2  = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
           .filterBounds(geom)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
           .linkCollection(csp, ["cs_cdf"])
           .map(lambda im: im.updateMask(im.select("cs_cdf").gte(CLOUD_THRESH))))
    return (s2.select(BANDS).median().divide(10000).clamp(0, 1)
              .toFloat())


def _export_patch(row, project: str, out_dir: Path) -> str | None:
    """Downloads one patch as an np.float32 array of shape (C, H, W)."""
    lon, lat = row["lon"], row["lat"]
    fname    = out_dir / f"{row['region']}_{int(row['label'])}_{row.name}.npy"
    if fname.exists():
        return str(fname)

    try:
        ee.Initialize(project=project)   # safe to call multiple times
        geom  = ee.Geometry.Point([lon, lat]).buffer(HALF_M).bounds()
        img   = _s2_composite(geom)
        arr   = np.array(img.sampleRectangle(
            region=geom, defaultValue=0
        ).get("array").getInfo(), dtype=np.float32)
        # arr shape: (H, W, C) → (C, H, W)
        arr = arr.transpose(2, 0, 1)
        # Resize to exact patch_size if EE returns slightly different dims
        if arr.shape[1] != PATCH_PX or arr.shape[2] != PATCH_PX:
            import cv2
            arr = np.stack([
                cv2.resize(arr[c], (PATCH_PX, PATCH_PX), interpolation=cv2.INTER_LINEAR)
                for c in range(arr.shape[0])
            ])
        np.save(fname, arr)
        return str(fname)
    except Exception as exc:
        print(f"  WARN patch {row.name} ({row['region']}): {exc}")
        return None


def export_patches(project: str, resume: bool = True, workers: int = 8):
    df = pd.read_csv(IN_CSV)
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PATCHES_DIR / "manifest.csv"

    if manifest_path.exists() and resume:
        done = set(pd.read_csv(manifest_path)["filename"])
    else:
        done = set()

    rows_todo = df[~df.apply(
        lambda r: (PATCHES_DIR / f"{r['region']}_{int(r['label'])}_{r.name}.npy").exists(),
        axis=1
    )]
    print(f"Patches to download: {len(rows_todo)} / {len(df)}")

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_export_patch, row, project, PATCHES_DIR): idx
            for idx, row in rows_todo.iterrows()
        }
        for i, fut in enumerate(as_completed(futures)):
            fname = fut.result()
            if fname:
                idx = futures[fut]
                r   = df.loc[idx]
                records.append(dict(
                    filename=fname, region=r["region"], label=int(r["label"]),
                    lon=r["lon"], lat=r["lat"], weight=r["weight"],
                ))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(futures)} done")

    new_df = pd.DataFrame(records)
    if manifest_path.exists():
        old_df = pd.read_csv(manifest_path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(manifest_path, index=False)
    print(f"Manifest saved: {manifest_path} ({len(new_df)} patches)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers",  type=int, default=8)
    args = parser.parse_args()
    export_patches(args.project, resume=not args.no_resume, workers=args.workers)
