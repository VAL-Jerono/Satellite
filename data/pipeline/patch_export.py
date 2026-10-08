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
# NOTE: `import ee` is deferred to inside functions — module imports cleanly
# without earthengine-api being available in the environment.
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


def _s2_composite(geom: "ee.Geometry") -> "ee.Image":
    import ee
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2  = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
           .filterBounds(geom)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
           .linkCollection(csp, ["cs_cdf"])
           .map(lambda im: im.updateMask(im.select("cs_cdf").gte(CLOUD_THRESH))))
    return (s2.select(BANDS).median().divide(10000).clamp(0, 1)
              .toFloat())


def _export_patch(row, project: str, out_dir: Path) -> "str | None":
    """Downloads one patch as an np.float32 array of shape (C, H, W)."""
    lon, lat = row["lon"], row["lat"]
    fname    = out_dir / f"{row['region']}_{int(row['label'])}_{row.name}.npy"
    if fname.exists():
        try:
            arr = np.load(fname)
            if arr.shape == (10, 64, 64):
                return str(fname)
            else:
                fname.unlink(missing_ok=True)
        except Exception:
            fname.unlink(missing_ok=True)

    try:
        import ee
        ee.Initialize(project=project)   # safe to call multiple times
        geom   = ee.Geometry.Point([lon, lat]).buffer(HALF_M).bounds()
        img    = _s2_composite(geom)
        sample = img.sampleRectangle(region=geom, defaultValue=0).getInfo()
        props  = sample.get("properties", sample)

        band_arrays = []
        import cv2
        for b in BANDS:
            raw = props.get(b, sample.get(b))
            if raw is None:
                raise KeyError(f"Band {b} missing from GEE sampleRectangle response")
            b_arr = np.array(raw, dtype=np.float32)
            if b_arr.ndim != 2:
                b_arr = b_arr.squeeze()
            if b_arr.shape[0] != PATCH_PX or b_arr.shape[1] != PATCH_PX:
                b_arr = cv2.resize(b_arr, (PATCH_PX, PATCH_PX), interpolation=cv2.INTER_LINEAR)
            band_arrays.append(b_arr)

        arr = np.stack(band_arrays, axis=0)  # shape: (10, 64, 64)
        np.save(fname, arr)
        return str(fname)
    except Exception as exc:
        print(f"  WARN patch {row.name} ({row['region']}): {exc}")
        return None


def export_patches(project: str, resume: bool = True, workers: int = 8):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        # Credentials not yet set — use Colab-safe notebook auth
        try:
            auth_mode = "notebook"  # correct mode for Colab interactive cells
            ee.Authenticate(auth_mode=auth_mode)
            ee.Initialize(project=project)
        except Exception as auth_exc:
            raise RuntimeError(
                "Earth Engine auth failed. Run this in its own notebook cell first:\n"
                "  import ee; ee.Authenticate(auth_mode='notebook'); "
                f"ee.Initialize(project='{project}')\n"
                f"Original error: {auth_exc}"
            ) from auth_exc

    csv_path = IN_CSV
    if not csv_path.exists():
        drive_csv = Path("/content/drive/MyDrive/land_atlas_baseline/samples_2021_n400.csv")
        if drive_csv.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(drive_csv, csv_path)
            print(f"Copied samples CSV from Drive: {drive_csv} -> {csv_path}")
        else:
            raise FileNotFoundError(
                f"Sample CSV not found at {csv_path} or {drive_csv}.\n"
                "Please make sure Google Drive is mounted or run ee_pull first."
            )

    df = pd.read_csv(csv_path)
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PATCHES_DIR / "manifest.csv"

    # Sync pre-existing patches & manifest from Drive if available
    drive_patches = Path("/content/drive/MyDrive/land_atlas_baseline/patches")
    if drive_patches.exists():
        drive_manifest = drive_patches / "manifest.csv"
        if drive_manifest.exists() and (not manifest_path.exists() or manifest_path.stat().st_size == 0):
            import shutil
            shutil.copy(drive_manifest, manifest_path)
            print(f"Copied manifest from Drive: {drive_manifest}")
        # Note: Do NOT loop over 27,000+ files to copy from Drive -> local disk
        # as Google Drive FUSE will freeze Colab. We read directly from Drive or process on-demand.


    # Find which patches still need to be downloaded
    rows_todo = df[~df.apply(
        lambda r: (PATCHES_DIR / f"{r['region']}_{int(r['label'])}_{r.name}.npy").exists(),
        axis=1
    )]
    print(f"Patches to download: {len(rows_todo)} / {len(df)}")

    if len(rows_todo) > 0:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_export_patch, row, project, PATCHES_DIR): idx
                for idx, row in rows_todo.iterrows()
            }
            for i, fut in enumerate(as_completed(futures)):
                fut.result()
                if (i + 1) % 100 == 0 or (i + 1) == len(futures):
                    print(f"  {i+1}/{len(futures)} downloaded")

    # Rebuild manifest index for all patch .npy files present on disk
    manifest_records = []
    for idx, r in df.iterrows():
        fname = PATCHES_DIR / f"{r['region']}_{int(r['label'])}_{idx}.npy"
        if fname.exists():
            manifest_records.append(dict(
                filename=str(fname), region=r["region"], label=int(r["label"]),
                lon=r["lon"], lat=r["lat"],
                weight=float(r.get("weight", 1.0)),   # safe: older chunk CSVs lack weight
            ))

    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Manifest saved: {manifest_path} ({len(manifest_df)} patches indexed)")

    # Backup patches & manifest to Drive so future Colab runs skip downloading
    if Path("/content/drive/MyDrive").exists():
        drive_out = Path("/content/drive/MyDrive/land_atlas_baseline/patches")
        drive_out.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(manifest_path, drive_out / "manifest.csv")
        for npy in PATCHES_DIR.glob("*.npy"):
            target = drive_out / npy.name
            if not target.exists():
                shutil.copy(npy, target)
        print(f"Backed up patches and manifest to Drive: {drive_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers",  type=int, default=8)
    args = parser.parse_args()
    export_patches(args.project, resume=not args.no_resume, workers=args.workers)
