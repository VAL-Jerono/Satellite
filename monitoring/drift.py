"""
monitoring/drift.py
────────────────────
Evidently-based drift monitoring for the land-cover API.

Compares incoming request feature distributions against the reference
(training fold 0 features) to detect:
  - Data drift (feature shift, e.g. dry vs wet season)
  - Prediction drift (class distribution shift)

Run as a scheduled job or on-demand:
    python -m monitoring.drift --input-parquet /path/to/recent_predictions.parquet

Reports are written to monitoring/reports/ as HTML + JSON.
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT         = Path(__file__).resolve().parents[1]
_CFG          = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
REFERENCE_PATH= _ROOT / _CFG["monitoring"]["reference_data"]
REPORT_DIR    = _ROOT / _CFG["monitoring"]["drift_report_dir"]


def run_drift_report(current_parquet: str | Path, tag: str = "live") -> dict:
    """
    Compare current data against reference using Evidently.
    Returns a dict summary with drift scores.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, ClassificationPreset
        from evidently.metrics import DatasetDriftMetric
    except ImportError:
        raise RuntimeError(
            "Evidently not installed. Run: pip install evidently"
        )

    reference = pd.read_parquet(REFERENCE_PATH)
    current   = pd.read_parquet(current_parquet)

    # Keep only numeric feature columns present in both
    feat_cols = [c for c in reference.columns
                 if c not in ("filename","region","block","label","weight","bx","by")
                 and c in current.columns]

    ref_feat = reference[feat_cols + ["label"]].copy()
    cur_feat = current[feat_cols + ["label"]].copy()

    report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
    report.run(reference_data=ref_feat, current_data=cur_feat)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts        = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html_path = REPORT_DIR / f"drift_{tag}_{ts}.html"
    json_path = REPORT_DIR / f"drift_{tag}_{ts}.json"
    report.save_html(str(html_path))

    result = report.as_dict()
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Extract summary
    drift_metric = result.get("metrics", [{}])[0]
    n_drifted    = drift_metric.get("result", {}).get("number_of_drifted_columns", "?")
    n_total      = drift_metric.get("result", {}).get("number_of_columns", "?")
    share_drifted= drift_metric.get("result", {}).get("share_of_drifted_columns", None)

    summary = dict(
        tag=tag, timestamp=ts,
        n_features=n_total, n_drifted=n_drifted,
        share_drifted=round(float(share_drifted), 3) if share_drifted else None,
        html_report=str(html_path),
        json_report=str(json_path),
    )
    print(json.dumps(summary, indent=2))
    return summary


def simulate_seasonal_drift(output_parquet: str | Path = None) -> pd.DataFrame:
    """
    Generates a synthetic 'wet season' current dataset by shifting NDVI
    upward and MNDWI upward, simulating the dry→wet sensor drift.
    Used for Week 13 drift module demonstration.
    """
    ref = pd.read_parquet(REFERENCE_PATH)
    df  = ref.copy()

    np.random.seed(42)
    if "NDVI" in df.columns:
        df["NDVI"]  += np.random.normal(0.12, 0.04, len(df)).clip(-0.05, 0.25)
    if "MNDWI" in df.columns:
        df["MNDWI"] += np.random.normal(0.08, 0.03, len(df)).clip(-0.05, 0.20)
    if "NDWI" in df.columns:
        df["NDWI"]  += np.random.normal(0.06, 0.02, len(df)).clip(-0.05, 0.15)

    if output_parquet:
        Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_parquet, index=False)
        print(f"Simulated drift data → {output_parquet}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", help="Current-window features parquet")
    parser.add_argument("--simulate",      action="store_true",
                        help="Simulate dry→wet drift and run report")
    parser.add_argument("--tag",           default="live")
    args = parser.parse_args()

    if args.simulate:
        sim_path = REPORT_DIR / "simulated_wet_season.parquet"
        simulate_seasonal_drift(sim_path)
        run_drift_report(sim_path, tag="simulated_wet")
    elif args.input_parquet:
        run_drift_report(args.input_parquet, tag=args.tag)
    else:
        print("Pass --input-parquet or --simulate")
