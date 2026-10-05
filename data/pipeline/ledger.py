"""
data/pipeline/ledger.py
───────────────────────
Atomic progress ledger for the 10-county batch pipeline.

The ledger lives at  data/progress/ledger.json  (relative to repo root)
and Drive backup at  <OUT_DIR>/progress/ledger.json  (when running in Colab).

Stage values
────────────
  "pending"     – not started
  "in_progress" – started but not finished (may happen if session dies)
  "done"        – completed and verified

Global flags
────────────
  band_stats    – "fresh" | "stale" (stale → recompute before training)
  splits_merged – "pending" | "done"

Usage
─────
    from data.pipeline.ledger import Ledger
    led = Ledger()
    if led.county_stage("highland", "ee_pull") != "done":
        ...pull...
        led.mark_done("highland", "ee_pull")
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[2]
_PROGRESS_DIR = _ROOT / "data" / "progress"
_LEDGER_PATH  = _PROGRESS_DIR / "ledger.json"

STAGES = ("ee_pull", "patch_export", "splits")
StageStatus = Literal["pending", "in_progress", "done"]

ALL_REGIONS = [
    # Highland zone
    "highland", "muranga", "nyandarua",
    # Coastal zone
    "coast", "kwale", "lamu",
    # ASAL zone
    "arid", "turkana", "wajir",
    # Lake zone
    "lake_basin", "siaya",
    # Rift Valley zone
    "nakuru", "kajiado",
]


def _default_ledger() -> dict:
    return {
        "counties": {
            name: {stage: "pending" for stage in STAGES}
            for name in ALL_REGIONS
        },
        "global": {
            "band_stats":    "stale",   # recompute before training
            "splits_merged": "pending",
        },
        "meta": {
            "created_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "last_updated":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_counties": len(ALL_REGIONS),
        },
    }


class Ledger:
    """Thread-safe (single-process) progress ledger with atomic writes."""

    def __init__(self, ledger_path: "Path | None" = None, drive_dir: "Path | None" = None):
        self._path = Path(ledger_path) if ledger_path else _LEDGER_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Optional Drive backup path (Colab)
        self._drive_path = None
        if drive_dir is not None:
            self._drive_path = Path(drive_dir) / "progress" / "ledger.json"

        self._data = self._load()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        # Prefer Drive copy if local is missing (Colab restart scenario)
        if not self._path.exists():
            if self._drive_path and self._drive_path.exists():
                import shutil
                self._path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(self._drive_path, self._path)
                print(f"[Ledger] Restored from Drive: {self._drive_path}")
            else:
                data = _default_ledger()
                self._write(data)
                return data

        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            # Patch ledger if new counties were added since creation
            for name in ALL_REGIONS:
                if name not in data["counties"]:
                    data["counties"][name] = {s: "pending" for s in STAGES}
            return data
        except (json.JSONDecodeError, KeyError):
            print("[Ledger] WARN: corrupt ledger, resetting.")
            data = _default_ledger()
            self._write(data)
            return data

    def _write(self, data: "dict | None" = None) -> None:
        payload = data if data is not None else self._data
        payload.setdefault("meta", {})["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self._path)  # atomic on POSIX & Windows

        # Backup to Drive if available
        if self._drive_path:
            try:
                self._drive_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(self._path, self._drive_path)
            except Exception:
                pass  # best-effort

    # ── Reads ─────────────────────────────────────────────────────────────────

    def county_stage(self, county: str, stage: str) -> str:
        return self._data["counties"].get(county, {}).get(stage, "pending")

    def global_flag(self, key: str) -> str:
        return self._data["global"].get(key, "pending")

    def summary(self) -> str:
        lines = ["=== Pipeline Ledger ==="]
        for county, stages in self._data["counties"].items():
            row = "  ".join(f"{s}:{v[0].upper()}" for s, v in stages.items())
            lines.append(f"  {county:<12} {row}")
        g = self._data["global"]
        lines.append(f"  [global] band_stats={g['band_stats']}  splits={g['splits_merged']}")
        return "\n".join(lines)

    def pending_counties(self, stage: str) -> "list[str]":
        """Returns counties that still need work for a given stage."""
        return [
            c for c in ALL_REGIONS
            if self._data["counties"].get(c, {}).get(stage, "pending") != "done"
        ]

    def done_counties(self, stage: str) -> "list[str]":
        return [
            c for c in ALL_REGIONS
            if self._data["counties"].get(c, {}).get(stage, "pending") == "done"
        ]

    # ── Writes ────────────────────────────────────────────────────────────────

    def mark_in_progress(self, county: str, stage: str) -> None:
        self._data["counties"].setdefault(county, {})[stage] = "in_progress"
        self._write()

    def mark_done(self, county: str, stage: str) -> None:
        self._data["counties"].setdefault(county, {})[stage] = "done"
        # New patches -> band stats go stale
        if stage == "patch_export":
            self._data["global"]["band_stats"] = "stale"
        self._write()
        print(f"[Ledger] v {county} / {stage} marked DONE")

    def set_global(self, key: str, value: str) -> None:
        self._data["global"][key] = value
        self._write()

    def reset_county(self, county: str) -> None:
        """Force a county back to all-pending (useful for re-pulling a region)."""
        self._data["counties"][county] = {s: "pending" for s in STAGES}
        self._write()
        print(f"[Ledger] Reset {county} to all-pending")
