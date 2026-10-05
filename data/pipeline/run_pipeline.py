"""
data/pipeline/run_pipeline.py
──────────────────────────────
Batch-safe orchestrator for the 10-county Kenya Land Cover pipeline.

Reads ledger.json to know what has already been done, then processes
the remaining counties in configurable batches, saving progress after
every county so that Colab session timeouts lose at most one county.

Stages (in order)
─────────────────
  1. ee_pull       – pull Sentinel-2 + WorldCover samples from Earth Engine
  2. patch_export  – download 64x64 .npy patches per sampled point
  3. band_stats    – recompute per-band mean/std across ALL patches (if stale)
  4. merge_csv     – concatenate all per-county CSVs into one master CSV
  5. splits        – build buffered spatial-block CV + LORO splits

Run from repo root:
    python -m data.pipeline.run_pipeline --project YOUR_GCP_PROJECT

Key flags:
    --batch-size 2    Process 2 counties at a time (default 2)
    --counties highland muranga   Process only specific counties
    --stage ee_pull   Run only one stage (skip others)
    --force           Ignore ledger and reprocess everything
    --status          Print ledger status and exit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CFG  = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
_RGN  = yaml.safe_load((_ROOT / "configs" / "regions.yaml").read_text())

ALL_REGIONS   = list(_RGN["regions"].keys())
REGION_BBOXES = {k: v["bbox"] for k, v in _RGN["regions"].items()}
OUT_CSV       = _ROOT / _CFG["paths"]["raw_csv"]


def _drive_dir() -> "Path | None":
    p = Path("/content/drive/MyDrive/land_atlas_baseline")
    return p if p.parent.exists() else None


# ── Stage 1: EE Pull ──────────────────────────────────────────────────────────

def run_ee_pull(county: str, project: str, force: bool = False) -> bool:
    """Pull EE samples for a single county. Returns True on success."""
    from data.pipeline.ee_pull import pull, CHUNK_DIR
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [EE] Pulling {county} ...")
    t0 = time.time()
    try:
        drive = _drive_dir()
        pull(
            project=project,
            force=force,
            counties=[county],
            drive_dir=drive,
        )
        print(f"  [EE] {county} done in {time.time()-t0:.0f}s")
        return True
    except Exception as exc:
        print(f"  [EE] ERROR on {county}: {exc}")
        return False


# ── Stage 2: Patch Export ─────────────────────────────────────────────────────

def run_patch_export(county: str, project: str, workers: int = 8) -> bool:
    """Export .npy patches for a single county. Returns True on success."""
    import pandas as pd
    import ee

    # Find chunk CSVs for this county
    YEAR     = _CFG["training"]["year"]
    PER_CLS  = _CFG["training"]["per_class_per_region"]
    chunk_dir = _ROOT / "data" / "raw" / f"chunks_{YEAR}_n{PER_CLS}"
    county_chunks = list(chunk_dir.glob(f"chunk_{county}_sub*.csv"))

    if not county_chunks:
        print(f"  [PATCH] No chunk CSVs found for {county} - run ee_pull first")
        return False

    dfs = []
    for cp in sorted(county_chunks):
        try:
            dfs.append(pd.read_csv(cp))
        except Exception:
            pass
    if not dfs:
        return False

    df = pd.concat(dfs, ignore_index=True)
    print(f"  [PATCH] Exporting {len(df)} patches for {county} ...")

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        print(f"  [PATCH] EE Init error: {exc}")
        return False

    from data.pipeline.patch_export import _export_patch, PATCHES_DIR
    from concurrent.futures import ThreadPoolExecutor, as_completed

    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    # Only download missing patches
    rows_todo = df[~df.apply(
        lambda r: (PATCHES_DIR / f"{r['region']}_{int(r['label'])}_{r.name}.npy").exists(),
        axis=1
    )]
    print(f"  [PATCH] Need to download: {len(rows_todo)} / {len(df)}")

    if len(rows_todo) > 0:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_export_patch, row, project, PATCHES_DIR): idx
                for idx, row in rows_todo.iterrows()
            }
            done = 0
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done % 50 == 0 or done == len(futures):
                    print(f"  [PATCH] {done}/{len(futures)} downloaded")

    # Backup new patches to Drive
    drive = _drive_dir()
    if drive:
        import shutil
        drive_patches = drive / "patches"
        drive_patches.mkdir(parents=True, exist_ok=True)
        for npy in (PATCHES_DIR).glob(f"{county}_*.npy"):
            target = drive_patches / npy.name
            if not target.exists():
                shutil.copy(npy, target)
        print(f"  [PATCH] Backed up to Drive")

    return True


# ── Stage 3: Band Stats ───────────────────────────────────────────────────────

def run_band_stats() -> bool:
    from data.compute_band_stats import compute_dataset_band_stats, save_band_stats
    print("  [STATS] Recomputing band statistics across all patches ...")
    mean_vec, std_vec, count = compute_dataset_band_stats()
    if count == 0:
        print("  [STATS] WARN: No patches found yet — skipping stats update")
        return False
    stats_path = _ROOT / "data" / "band_stats.json"
    save_band_stats(mean_vec, std_vec, count, out_path=stats_path)
    print(f"  [STATS] Saved stats ({count} patches) -> {stats_path}")
    return True


# ── Stage 4: Merge CSV ────────────────────────────────────────────────────────

def run_merge_csv() -> bool:
    """Concatenate all per-county chunk CSVs into the master raw CSV."""
    import pandas as pd

    YEAR    = _CFG["training"]["year"]
    PER_CLS = _CFG["training"]["per_class_per_region"]
    chunk_dir = _ROOT / "data" / "raw" / f"chunks_{YEAR}_n{PER_CLS}"

    parts = []
    for county in ALL_REGIONS:
        county_chunks = sorted(chunk_dir.glob(f"chunk_{county}_sub*.csv"))
        if not county_chunks:
            print(f"  [MERGE] WARN: No chunks for {county} — skipping")
            continue
        dfs = []
        for cp in county_chunks:
            try:
                dfs.append(pd.read_csv(cp))
            except Exception:
                pass
        if dfs:
            parts.append(pd.concat(dfs, ignore_index=True))
            print(f"  [MERGE] {county}: {sum(len(d) for d in dfs)} rows")

    if not parts:
        print("  [MERGE] ERROR: No data to merge")
        return False

    df_all = pd.concat(parts, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_CSV, index=False)
    print(f"  [MERGE] Master CSV saved: {OUT_CSV} ({len(df_all)} rows)")

    # Backup to Drive
    drive = _drive_dir()
    if drive:
        import shutil
        drive.mkdir(parents=True, exist_ok=True)
        shutil.copy(OUT_CSV, drive / OUT_CSV.name)
        print(f"  [MERGE] Backed up master CSV to Drive")

    return True


# ── Stage 5: Spatial Splits ───────────────────────────────────────────────────

def run_splits(project: str) -> bool:
    from data.pipeline.spatial_splits import build_splits
    print("  [SPLITS] Building spatial block CV + LORO splits ...")
    try:
        summary = build_splits(project=project)
        print(f"  [SPLITS] Done: {summary['n_total']} points, {summary['n_blocks']} blocks")
        return True
    except Exception as exc:
        print(f"  [SPLITS] ERROR: {exc}")
        return False


# ── Manifest Rebuild ──────────────────────────────────────────────────────────

def rebuild_manifest() -> None:
    """Rebuild patches/manifest.csv from all .npy files currently on disk."""
    import pandas as pd
    import numpy as np
    from data.pipeline.patch_export import PATCHES_DIR

    YEAR    = _CFG["training"]["year"]
    PER_CLS = _CFG["training"]["per_class_per_region"]
    chunk_dir = _ROOT / "data" / "raw" / f"chunks_{YEAR}_n{PER_CLS}"

    # Load all chunk CSVs to get metadata (lon, lat, weight)
    meta_frames = []
    for cp in chunk_dir.glob("chunk_*.csv"):
        try:
            meta_frames.append(pd.read_csv(cp))
        except Exception:
            pass
    if not meta_frames:
        print("  [MANIFEST] No chunk CSVs found")
        return

    meta = pd.concat(meta_frames, ignore_index=True)

    records = []
    for idx, row in meta.iterrows():
        fname = PATCHES_DIR / f"{row['region']}_{int(row['label'])}_{idx}.npy"
        if fname.exists():
            records.append(dict(
                filename=str(fname),
                region=row["region"],
                label=int(row["label"]),
                lon=row["lon"],
                lat=row["lat"],
                weight=row.get("weight", 1.0),
            ))

    manifest_df = pd.DataFrame(records)
    manifest_path = PATCHES_DIR / "manifest.csv"
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  [MANIFEST] Rebuilt: {len(manifest_df)} patches indexed -> {manifest_path}")

    # Drive backup
    drive = _drive_dir()
    if drive:
        import shutil
        drive_patches = drive / "patches"
        drive_patches.mkdir(parents=True, exist_ok=True)
        shutil.copy(manifest_path, drive_patches / "manifest.csv")


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def run_pipeline(
    project: str,
    batch_size: int = 2,
    counties: "list[str] | None" = None,
    stage_filter: "str | None" = None,
    workers: int = 8,
    force: bool = False,
):
    from data.pipeline.ledger import Ledger

    drive = _drive_dir()
    led = Ledger(drive_dir=drive)

    print(led.summary())
    print()

    target_counties = counties if counties else ALL_REGIONS

    # ── Stage 1 & 2: EE Pull + Patch Export (county-by-county in batches) ────
    for stage in ("ee_pull", "patch_export"):
        if stage_filter and stage_filter != stage:
            continue

        pending = [c for c in target_counties if led.county_stage(c, stage) != "done"]
        if not pending:
            print(f"[Pipeline] All counties done for stage '{stage}' — skipping")
            continue

        print(f"\n[Pipeline] === Stage: {stage} | {len(pending)} counties remaining ===")
        # Process in batches
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            print(f"\n[Pipeline] Batch: {batch}")
            for county in batch:
                if led.county_stage(county, stage) == "done" and not force:
                    print(f"  [Pipeline] {county}/{stage} already done — skip")
                    continue
                led.mark_in_progress(county, stage)
                if stage == "ee_pull":
                    ok = run_ee_pull(county, project, force=force)
                else:
                    ok = run_patch_export(county, project, workers=workers)
                if ok:
                    led.mark_done(county, stage)
                else:
                    print(f"  [Pipeline] WARN: {county}/{stage} failed — will retry next run")

    if stage_filter and stage_filter not in ("ee_pull", "patch_export"):
        pass
    else:
        # After all patches, rebuild manifest
        if not stage_filter or stage_filter == "patch_export":
            print("\n[Pipeline] === Rebuilding manifest ===")
            rebuild_manifest()

    # ── Stage 3: Band Stats ───────────────────────────────────────────────────
    if not stage_filter or stage_filter == "band_stats":
        if led.global_flag("band_stats") == "stale" or force:
            print("\n[Pipeline] === Stage: band_stats ===")
            ok = run_band_stats()
            if ok:
                led.set_global("band_stats", "fresh")
        else:
            print("\n[Pipeline] Band stats are fresh — skipping recompute")

    # ── Stage 4: Merge CSV ────────────────────────────────────────────────────
    if not stage_filter or stage_filter == "merge_csv":
        print("\n[Pipeline] === Stage: merge CSV ===")
        run_merge_csv()
        led.set_global("splits_merged", "pending")  # CSV changed → splits need rebuild

    # ── Stage 5: Spatial Splits ───────────────────────────────────────────────
    if not stage_filter or stage_filter == "splits":
        print("\n[Pipeline] === Stage: spatial splits ===")
        ok = run_splits(project)
        if ok:
            led.set_global("splits_merged", "done")

    print("\n[Pipeline] === Final Status ===")
    print(led.summary())


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-safe 10-county pipeline orchestrator")
    parser.add_argument("--project",    required=True,  help="GCP project registered for Earth Engine")
    parser.add_argument("--batch-size", type=int, default=2, help="Counties to process per batch (default 2)")
    parser.add_argument("--counties",   nargs="+",      help="Process only these counties (space-separated)")
    parser.add_argument("--stage",      default=None,   help="Run only one stage: ee_pull|patch_export|band_stats|merge_csv|splits")
    parser.add_argument("--workers",    type=int, default=8, help="Thread workers for patch export")
    parser.add_argument("--force",      action="store_true", help="Ignore ledger, reprocess everything")
    parser.add_argument("--status",     action="store_true", help="Print ledger status and exit")
    args = parser.parse_args()

    if args.status:
        from data.pipeline.ledger import Ledger
        print(Ledger().summary())
    else:
        run_pipeline(
            project=args.project,
            batch_size=args.batch_size,
            counties=args.counties,
            stage_filter=args.stage,
            workers=args.workers,
            force=args.force,
        )
