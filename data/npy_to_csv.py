"""
data/npy_to_csv.py
──────────────────
Converts downloaded .npy patch files (shape 10×64×64 Sentinel-2) into a flat
tabular CSV suitable for Random Forest / XGBoost training.

For every patch the following columns are written:
  • Identity    : filename, region, label, lon, lat, weight
  • Band stats  : B2..B12 × {mean, std, min, max}  (40 cols)
  • Index stats : NDVI, NDWI, MNDWI, NDBI, EVI, SAVI, BSI, NDRE
                  × {mean, std, p10, p90}            (32 cols)
  • Texture     : spatial_texture_red, spatial_texture_nir (2 cols)

Total feature columns: 74

Usage (from repo root):
    python -m data.npy_to_csv                    # uses auto-discovered patch dir
    python -m data.npy_to_csv --patches-dir /content/Satellite/data/patches
    python -m data.npy_to_csv --out data/features.csv --workers 8

The script is resume-safe: if the output CSV already exists and --append is set,
only patches not yet in the CSV are processed and appended.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
_CFG  = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())

# Default search locations (Colab + local)
PATCH_SEARCH_DIRS = [
    _ROOT / _CFG["paths"]["patches_dir"],
    Path("/content/Satellite/data/patches"),
    Path("/content/drive/MyDrive/land_atlas_baseline/patches"),
]

DEFAULT_OUT = _ROOT / "data" / "features.csv"


# ── Feature extraction (imported from shared module) ─────────────────────────
from data.feature_engineering import extract_patch_features   # noqa: E402


def _process_row(args: tuple) -> Optional[dict]:
    """Worker function: load one .npy file and return a flat feature dict."""
    fname, region, label, lon, lat, weight = args
    try:
        arr = np.load(fname, allow_pickle=False).astype(np.float32)
        if arr.shape != (10, 64, 64):
            print(f"  SKIP {fname}: unexpected shape {arr.shape}")
            return None

        feats = extract_patch_features(arr)
        feats.update(
            filename=str(fname),
            region=region,
            label=int(label),
            lon=float(lon),
            lat=float(lat),
            weight=float(weight) if weight is not None else 1.0,
        )
        return feats
    except Exception as exc:
        print(f"  WARN {fname}: {exc}")
        return None


def find_patch_dir(override: Optional[str] = None) -> Optional[Path]:
    """Return the first patch directory that contains .npy files."""
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.extend(PATCH_SEARCH_DIRS)

    for d in candidates:
        if d.exists() and list(d.glob("*.npy")):
            return d
    return None


def build_manifest_from_disk(patch_dir: Path) -> pd.DataFrame:
    """
    Build a minimal manifest from raw .npy filenames when manifest.csv is
    absent or empty.  Filename convention: {region}_{label}_{idx}.npy
    """
    records = []
    for p in patch_dir.glob("*.npy"):
        parts = p.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            label = int(parts[-2])
            region = "_".join(parts[:-2])
            records.append(dict(
                filename=str(p),
                region=region,
                label=label,
                lon=float("nan"),
                lat=float("nan"),
                weight=1.0,
            ))
        except ValueError:
            continue
    return pd.DataFrame(records)


def load_manifest(patch_dir: Path) -> pd.DataFrame:
    """
    Always builds the primary patch list by scanning .npy filenames on disk
    (guarantees ALL downloaded patches are included, even if manifest.csv or
    chunk CSVs only cover a subset of counties).

    Then enriches with lon/lat/weight from manifest.csv where available.
    """
    # Step 1: ground truth = everything actually on disk
    df_disk = build_manifest_from_disk(patch_dir)
    print(f"  Disk scan: {len(df_disk)} .npy files found in {patch_dir}")

    if len(df_disk) == 0:
        return df_disk

    # Step 2: try to enrich with lon/lat from manifest.csv
    manifest_path = patch_dir / "manifest.csv"
    if manifest_path.exists():
        try:
            mdf = pd.read_csv(manifest_path)
            if len(mdf) > 0 and "filename" in mdf.columns:
                # Build a lookup: filename -> {lon, lat, weight}
                mdf["filename"] = mdf["filename"].astype(str)
                enrich = mdf.set_index("filename")[[c for c in ["lon", "lat", "weight"] if c in mdf.columns]]
                df_disk["filename"] = df_disk["filename"].astype(str)
                for col in enrich.columns:
                    df_disk[col] = df_disk["filename"].map(enrich[col])
                matched = df_disk["lon"].notna().sum()
                print(f"  Enriched {matched}/{len(df_disk)} patches with lon/lat from manifest.csv")
        except Exception as exc:
            print(f"  WARN: Could not read manifest.csv for enrichment: {exc}")

    return df_disk


def _flush(records: list, out_path: Path, write_header: bool) -> None:
    """Append a batch of records to the output CSV."""
    df = pd.DataFrame(records)

    # Reorder columns: identity first, then features
    id_cols = ["filename", "region", "label", "lon", "lat", "weight"]
    feat_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + feat_cols]

    df.to_csv(
        out_path,
        mode="a",
        header=write_header,
        index=False,
    )


def convert(
    patches_dir: Optional[str] = None,
    out_path: "str | Path" = DEFAULT_OUT,
    workers: int = 4,
    append: bool = True,
    chunk_size: int = 500,
) -> Path:
    """
    Main entry point.  Converts .npy patches to a flat tabular CSV.
    """
    out_path = Path(out_path)

    # ── Skip early if output CSV already exists and append=True ─────────────
    if append and out_path.exists() and out_path.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(out_path, usecols=["filename"])
            if len(existing_df) > 0:
                print(f"[npy_to_csv] Output CSV already exists with {len(existing_df)} rows at {out_path}.")
        except Exception:
            pass

    # ── Find patch directory ──────────────────────────────────────────────────
    patch_dir = find_patch_dir(patches_dir)
    if patch_dir is None:
        raise FileNotFoundError(
            "No .npy patch directory found.\n"
            f"Searched: {[str(d) for d in PATCH_SEARCH_DIRS]}\n"
            "Pass --patches-dir explicitly if patches are elsewhere."
        )
    print(f"[npy_to_csv] Patch directory: {patch_dir}")

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest = load_manifest(patch_dir)
    if len(manifest) == 0:
        raise ValueError(f"No patches found in {patch_dir}")

    # Ensure filename column is present as absolute path string
    if "filename" not in manifest.columns:
        manifest["filename"] = manifest.apply(
            lambda r: str(patch_dir / f"{r['region']}_{int(r['label'])}_{r.name}.npy"),
            axis=1,
        )

    # ── Skip already-processed patches (resume & skip logic) ───────────────────
    already_done: set = set()
    if append and out_path.exists() and out_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(out_path, usecols=["filename"])
            for fn in existing["filename"].dropna():
                s = str(fn)
                already_done.add(s)
                already_done.add(Path(s).name)  # match by basename as well
            print(f"  Existing output found: {len(existing)} rows already in {out_path}")
        except Exception as exc:
            print(f"  Warning loading existing CSV: {exc}")

    def _is_done(f_path: str) -> bool:
        s = str(f_path)
        return (s in already_done) or (Path(s).name in already_done)

    todo = manifest[~manifest["filename"].astype(str).apply(_is_done)]
    print(f"[npy_to_csv] Patches to process: {len(todo)} / {len(manifest)}")

    if len(todo) == 0:
        print(f"[npy_to_csv] Nothing to process — {out_path} is already up to date with {len(manifest)} patches.")
        return out_path

    # ── Build arg list ────────────────────────────────────────────────────────
    arg_list = [
        (
            row["filename"],
            row.get("region", "unknown"),
            row.get("label", -1),
            row.get("lon", float("nan")),
            row.get("lat", float("nan")),
            row.get("weight", 1.0),
        )
        for _, row in todo.iterrows()
    ]

    # ── Process patches ───────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (append and out_path.exists())

    buffer: list = []
    done = 0
    skipped = 0

    print(f"[npy_to_csv] Processing with {workers} worker(s) ...")

    if workers > 1:
        # Parallel: good for local runs with many cores
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_row, a): a for a in arg_list}
            for fut in as_completed(futures):
                result = fut.result()
                done += 1
                if result is None:
                    skipped += 1
                else:
                    buffer.append(result)

                if len(buffer) >= chunk_size:
                    _flush(buffer, out_path, write_header)
                    write_header = False
                    buffer = []

                if done % 1000 == 0 or done == len(arg_list):
                    print(f"  {done}/{len(arg_list)} processed, {skipped} skipped")
    else:
        # Sequential: safer for Colab / debugging
        for i, a in enumerate(arg_list):
            result = _process_row(a)
            done += 1
            if result is None:
                skipped += 1
            else:
                buffer.append(result)

            if len(buffer) >= chunk_size:
                _flush(buffer, out_path, write_header)
                write_header = False
                buffer = []

            if (i + 1) % 1000 == 0 or (i + 1) == len(arg_list):
                print(f"  {i+1}/{len(arg_list)} processed, {skipped} skipped")

    # Flush remainder
    if buffer:
        _flush(buffer, out_path, write_header)

    n_written = len(todo) - skipped
    print(f"\n[npy_to_csv] Done: {n_written} rows written → {out_path}")
    print(f"  ({skipped} patches skipped due to errors or wrong shape)")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert .npy Sentinel-2 patches to a flat tabular CSV."
    )
    parser.add_argument(
        "--patches-dir",
        default=None,
        help="Directory containing .npy patch files (auto-detected if omitted)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output CSV path (default: data/features.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes (default: 4; use 1 for Colab debugging)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV, skipping already-processed patches",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Flush to disk every N rows for crash resilience (default: 500)",
    )
    args = parser.parse_args()

    convert(
        patches_dir=args.patches_dir,
        out_path=args.out,
        workers=args.workers,
        append=args.append,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
