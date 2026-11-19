"""
monitoring/drift.py
────────────────────
Data drift monitoring for the land-cover API.

Compares incoming request feature distributions against the reference
(training reference features) to detect:
  - Data drift (feature shift, e.g. dry vs wet season)
  - Prediction drift (class distribution shift)

Run as a scheduled job or on-demand:
    python3 -m monitoring.drift --simulate
    python3 -m monitoring.drift --input-parquet /path/to/recent_predictions.parquet

Reports are written to monitoring/reports/ as HTML + JSON.
"""

from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml

_ROOT          = Path(__file__).resolve().parents[1]
_CFG           = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
REFERENCE_PATH = _ROOT / _CFG["monitoring"]["reference_data"]
REPORT_DIR     = _ROOT / _CFG["monitoring"]["drift_report_dir"]


def _run_scipy_drift_analysis(reference: pd.DataFrame, current: pd.DataFrame, p_value_thresh: float = 0.05) -> dict:
    """
    High-fidelity statistical drift analyzer using two-sample Kolmogorov-Smirnov test
    and Wasserstein distance (the underlying engine used by Evidently for continuous features).
    """
    feat_cols = [c for c in reference.columns
                 if c not in ("filename", "region", "block", "label", "weight", "bx", "by", "predicted_class")
                 and c in current.columns and pd.api.types.is_numeric_dtype(reference[c])]

    drift_by_columns = {}
    drifted_count = 0

    for col in feat_cols:
        ref_v = reference[col].dropna().values
        cur_v = current[col].dropna().values

        # Kolmogorov-Smirnov 2-sample test
        ks_res = stats.ks_2samp(ref_v, cur_v)
        # Wasserstein distance
        w_dist = stats.wasserstein_distance(ref_v, cur_v)

        is_drifted = bool(ks_res.pvalue < p_value_thresh)
        if is_drifted:
            drifted_count += 1

        drift_by_columns[col] = {
            "column_name": col,
            "drift_detected": is_drifted,
            "stat_test": "Kolmogorov-Smirnov (alpha=0.05)",
            "p_value": float(ks_res.pvalue),
            "ks_statistic": float(ks_res.statistic),
            "wasserstein_distance": float(w_dist),
            "ref_mean": float(np.mean(ref_v)),
            "cur_mean": float(np.mean(cur_v)),
            "ref_std": float(np.std(ref_v)),
            "cur_std": float(np.std(cur_v)),
        }

    share_drifted = drifted_count / len(feat_cols) if feat_cols else 0.0
    dataset_drift = share_drifted > 0.33  # standard Evidently threshold

    return {
        "dataset_drift": dataset_drift,
        "number_of_columns": len(feat_cols),
        "number_of_drifted_columns": drifted_count,
        "share_of_drifted_columns": share_drifted,
        "drift_by_columns": drift_by_columns,
    }


def _generate_html_report(metrics: dict, tag: str, ts: str, output_path: Path):
    """Generates an executive HTML report."""
    dataset_drift = metrics["dataset_drift"]
    status_color = "#e53e3e" if dataset_drift else "#38a169"
    status_text = "DATA DRIFT DETECTED" if dataset_drift else "DATASET STABLE"

    table_rows = []
    for col, data in sorted(metrics["drift_by_columns"].items(), key=lambda x: x[1]["p_value"]):
        badge = '<span style="background:#fed7d7;color:#9b2c2c;padding:3px 8px;border-radius:4px;font-weight:600;">DRIFT</span>' if data["drift_detected"] else '<span style="background:#c6f6d5;color:#22543d;padding:3px 8px;border-radius:4px;font-weight:600;">STABLE</span>'
        table_rows.append(f"""
        <tr>
            <td style="padding:10px;border-bottom:1px solid #e2e8f0;font-weight:600;">{col}</td>
            <td style="padding:10px;border-bottom:1px solid #e2e8f0;">{badge}</td>
            <td style="padding:10px;border-bottom:1px solid #e2e8f0;">{data['p_value']:.4e}</td>
            <td style="padding:10px;border-bottom:1px solid #e2e8f0;">{data['ks_statistic']:.4f}</td>
            <td style="padding:10px;border-bottom:1px solid #e2e8f0;">{data['wasserstein_distance']:.4f}</td>
            <td style="padding:10px;border-bottom:1px solid #e2e8f0;">{data['ref_mean']:.3f} &plusmn; {data['ref_std']:.3f}</td>
            <td style="padding:10px;border-bottom:1px solid #e2e8f0;">{data['cur_mean']:.3f} &plusmn; {data['cur_std']:.3f}</td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Satellite Atlas - Data Drift Report [{tag}]</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background:#f7fafc; color:#2d3748; margin:0; padding:2rem; }}
        .container {{ max-width: 1100px; margin: 0 auto; background:#ffffff; border-radius:12px; box-shadow:0 4px 6px rgba(0,0,0,0.05); padding:2rem; border:1px solid #e2e8f0; }}
        .header {{ display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #edf2f7; padding-bottom:1.5rem; }}
        .badge {{ display:inline-block; padding:0.5rem 1rem; border-radius:99px; font-weight:700; font-size:1rem; }}
        .stats-grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:1rem; margin:1.5rem 0; }}
        .stat-card {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:1.2rem; text-align:center; }}
        .stat-val {{ font-size:1.8rem; font-weight:700; color:#1a202c; }}
        .stat-label {{ font-size:0.85rem; color:#718096; margin-top:0.25rem; text-transform:uppercase; letter-spacing:0.05em; }}
        table {{ width:100%; border-collapse:collapse; margin-top:1.5rem; }}
        th {{ background:#edf2f7; padding:12px 10px; text-align:left; font-size:0.85rem; color:#4a5568; text-transform:uppercase; letter-spacing:0.05em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="margin:0;font-size:1.8rem;color:#1a365d;">🛰️ Kenya Land Readiness Atlas — Data Drift Report</h1>
                <p style="margin:0.5rem 0 0 0;color:#718096;">Tag: <b>{tag}</b> &bull; Generated: {ts} UTC &bull; Test: Kolmogorov-Smirnov 2-Sample</p>
            </div>
            <div>
                <span class="badge" style="background:{status_color};color:white;">{status_text}</span>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val">{metrics['number_of_columns']}</div>
                <div class="stat-label">Evaluated Features</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color:{status_color};">{metrics['number_of_drifted_columns']}</div>
                <div class="stat-label">Drifted Features</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{metrics['share_of_drifted_columns']*100:.1f}%</div>
                <div class="stat-label">Drift Share</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{"ALERT" if dataset_drift else "NOMINAL"}</div>
                <div class="stat-label">Pipeline Action</div>
            </div>
        </div>

        <h3>📊 Feature-Level Statistical Distribution Analysis</h3>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Status</th>
                    <th>p-value</th>
                    <th>KS Stat</th>
                    <th>Wasserstein</th>
                    <th>Ref Mean &plusmn; Std</th>
                    <th>Current Mean &plusmn; Std</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
</body>
</html>"""
    output_path.write_text(html_content, encoding="utf-8")


def run_drift_report(current_parquet: str | Path, tag: str = "live") -> dict:
    """
    Compare current data against reference.
    Uses Evidently if installed; falls back to exact statistical Kolmogorov-Smirnov test.
    Returns a dict summary with drift scores.
    """
    reference = pd.read_parquet(REFERENCE_PATH)
    current   = pd.read_parquet(current_parquet)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html_path = REPORT_DIR / f"drift_{tag}_{ts}.html"
    json_path = REPORT_DIR / f"drift_{tag}_{ts}.json"

    # Try evidently first
    evidently_ran = False
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric

        feat_cols = [c for c in reference.columns
                     if c not in ("filename", "region", "block", "label", "weight", "bx", "by")
                     and c in current.columns]

        report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
        report.run(reference_data=reference[feat_cols], current_data=current[feat_cols])
        report.save_html(str(html_path))
        result = report.as_dict()
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        drift_metric = result.get("metrics", [{}])[0]
        n_drifted    = drift_metric.get("result", {}).get("number_of_drifted_columns", "?")
        n_total      = drift_metric.get("result", {}).get("number_of_columns", "?")
        share_drifted= drift_metric.get("result", {}).get("share_of_drifted_columns", None)
        evidently_ran = True
    except Exception:
        pass

    if not evidently_ran:
        metrics = _run_scipy_drift_analysis(reference, current)
        _generate_html_report(metrics, tag, ts, html_path)
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        n_drifted = metrics["number_of_drifted_columns"]
        n_total = metrics["number_of_columns"]
        share_drifted = metrics["share_of_drifted_columns"]

    summary = dict(
        tag=tag,
        timestamp=ts,
        n_features=n_total,
        n_drifted=n_drifted,
        share_drifted=round(float(share_drifted), 3) if share_drifted is not None else None,
        html_report=str(html_path),
        json_report=str(json_path),
    )
    print(json.dumps(summary, indent=2))
    return summary


def simulate_seasonal_drift(output_parquet: str | Path = None) -> pd.DataFrame:
    """
    Generates a synthetic 'wet season' current dataset by shifting NDVI
    upward and MNDWI upward, simulating the dry→wet sensor drift.
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
