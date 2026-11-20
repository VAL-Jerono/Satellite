# Kenya Land Readiness Atlas

**Evidence-led spatial machine-learning screening prototype**  
MSc Data Science & Analytics capstone · Strathmore University (iLabAfrica) · DSA 8401 Applied Machine Learning  
Candidate: Valerie Jerono · Submission: November 2026

## Overview

The Kenya Land Readiness Atlas explores whether openly available Earth-observation data and spatial machine learning can support **first-pass land-cover screening** across contrasting Kenyan environments. It combines Sentinel-2 Level-2A reflectance, SRTM terrain information and CHIRPS rainfall context, using ESA WorldCover 2021 as a reference label source. A six-class prediction is translated into provisional SETTLE, FARM, or PROTECT triage guidance.

This is a research prototype—not a cadastral or parcel product, legal determination, zoning instrument, land valuation, environmental approval, engineering assessment, or production-certified service. The reference labels are model-derived, not independent field observations. Model performance reported in the notebook is not independently reproduced from the packaged raw data and frozen splits.

## What is in this package

| Path | Contents |
|---|---|
| `CAPSTONE_NOTEBOOK.ipynb` | Project notebook and reported analysis outputs. Some operational examples are simulated or hard-coded demonstrations. |
| `data_pipeline/` | Earth Engine extraction, feature engineering, patch export, spatial split logic, region configuration, and county-summary utilities. |
| `model_training/` | Training code, configuration, five zone LightGBM artifacts, CNN checkpoint, stacking model, and ONNX artifact. |
| `api_service/` | FastAPI service source. Inspect fallback behavior before using; see limitations below. |
| `docker_configuration/` | Dockerfiles, Compose configurations, and Nginx configuration. Packaged configuration is not proof of successful deployment. |
| `reports_and_slides/` | Final report, business understanding, slide companion, and report figures/generator. |
| `AI_USE.md` | AI assistance and research integrity disclosure. |
| `CHECKLIST.md` | Submission inventory and evidence-status checklist. |
| `kenya_zones_map.png` | Study-zone map used by the report. |

The package does **not** include the observation-level raw training table, patch archive, split manifests, saved evaluation predictions, a `tests/` directory, or deployment/CI/load-test logs. Referenced paths in notebook cells may point to the original author's environment.

## Evidence status at a glance

| Claim or component | Status | How to read it |
|---|---|---|
| Pipeline, training and API source | **Packaged source** | Code is available for inspection; execution may require external access, missing inputs, and environment setup. |
| Model artifacts | **Packaged artifacts** | Files are present; their complete lineage to each reported metric is not independently established. |
| LOZO mean macro-F1 0.4324; zone scores 0.2732–0.5443 | **Notebook-reported** | Not independently reproduced; do not interpret as field accuracy. |
| Zone LightGBM 0.6271 and stacking 0.6504 | **Notebook-reported, different protocol** | Do not compare directly with LOZO values. |
| ECE 0.2525 → 0.0921, T=1.42 | **Notebook-reported** | Reliability-bin data and calibration split are not packaged. |
| Health, smoke and load examples | **Simulated / hard-coded** | Not evidence of live service behavior, uptime, throughput, latency, or CI success. |
| County action shares | **Notebook-reported sample summaries** | Shares of sampled observations, not county land-area proportions. |
| Independent field accuracy | **Not established** | No independent field-labelled sample is included. |

See `reports_and_slides/FINAL_REPORT.md` for the evidence register and metric provenance.

## Reproducing components

### 1. Inspect the notebook

Open `CAPSTONE_NOTEBOOK.ipynb` in Jupyter or another compatible notebook environment. Do not assume “Run All” is self-contained: source-data pulls require Google Earth Engine access and configured credentials, and some outputs in the saved notebook are illustrative rather than measured.

### 2. Data pipeline

The pipeline code is under `data_pipeline/`; its local README describes component-specific usage. A typical entry point is:

```bash
bash data_pipeline/run_pipeline.sh
```

Review the script and configuration before execution. Earth Engine credentials/project setup, dependencies, available data, and sufficient quota are prerequisites. The package does not include the raw table needed to reproduce the submitted metrics directly.

### 3. Model training

Model code and configuration are under `model_training/`. See `model_training/README.md` for dependencies and commands. Training from scratch requires the correct observation data, patch data, splits, compute environment, and configuration; these are not all included in this archive. Treat existing serialized models as submitted artifacts, not as independently verified reconstructions.

### 4. API inspection and local run

The API source is in `api_service/main.py`. It resolves paths relative to the parent of `api_service/`, expecting `configs/config.yaml`, `configs/regions.yaml`, `models/lgbm_zone_*.pkl`, `data/atlas/zone_confidence.json` and `data/county_summaries/`. In this archive those files live under `model_training/configs/` and `model_training/artifacts/`, so a staged directory layout (or a small path adjustment) is required before launch. No dependency lockfile is packaged; install the libraries the module imports (`fastapi`, `uvicorn`, `pandas`, `numpy`, `pyyaml`, `joblib`, `lightgbm`, and optionally `onnxruntime`, `earthengine-api`).

```bash
python -m uvicorn api_service.main:app --host 127.0.0.1 --port 8000
```

This is a development starting point, not a verified quickstart guarantee. Import-time configuration loading means the service fails at startup until the expected paths exist. Do not expose it publicly or use it for site decisions without remediation and validation.

### 5. Containers

Container definitions are in `docker_configuration/`. Review Compose files and required variables before use. The presence of a production-named Compose file does not establish that deployment has run or meets a service-level objective.

## Critical implementation and use limitations

- **Input fallback:** `/predict/point` has two non-informative paths. If a zone model is loaded but no patch or Earth Engine access is available, it scores a zero-valued feature vector with only slope set. If no zone model is loaded, it returns a uniform probability distribution labelled `model_type: "fallback"`. Neither output is site-specific. A safe implementation must return an explicit missing-input error instead.
- **Placeholder artifact fallback:** if the ONNX artifact is missing, API code can create a minimal placeholder graph. A production service should fail closed and verify artifact checksums rather than silently substitute a placeholder.
- **Legal rule layer:** action thresholds are prototype screening parameters, not statutory rules. PLUPA 2019 section 56 concerns county development control; riparian measurement is addressed in the 2021 General Development Permission and Control Regulations (Legal Notice 253), where widths vary. Do not treat a universal 30 m buffer or 15° prohibition as established by that section. The commonly cited EIA screening threshold for agricultural schemes is 100 ha, not 2 ha; applicability should be reviewed by qualified counsel.
- **No independent reference labels:** agreement with ESA WorldCover 2021 measures consistency with that product, not field-level accuracy.
- **No packaged automated tests:** there is no `tests/` folder in this archive. No test count or passing CI claim is made.
- **No verified production evidence:** notebook health output is mock data after an HTTP 403; load and smoke outputs are hard-coded; drift uses synthetic distributions. None demonstrates a running deployment.

## Project documents

- [Final technical report](reports_and_slides/FINAL_REPORT.md)
- [Business understanding retrospective](reports_and_slides/Business_Understanding.md)
- [Slide-by-slide companion and speaking notes](reports_and_slides/SLIDE_DECK.md)
- [AI use and research integrity disclosure](AI_USE.md)
- [Submission checklist](CHECKLIST.md)

## Suggested next steps before any pilot

1. Collect a stratified, independent field-reference sample and document interpretation protocols.
2. Freeze and package data splits, predictions, scoring scripts, and reliability-bin outputs.
3. Reconcile notebook methods, packaged code, serialized artifacts, and reported metric lineage.
4. Correct API fallback behavior and enforce artifact/version checks.
5. Obtain qualified legal review of every rule and user-facing statutory statement.
6. Execute and package automated tests, deployment logs, health checks, and measured load-test results.
7. Conduct a governed pilot with human review and a clear mechanism to contest or override outputs.

## Citation and contact

See the report references for data products, methods, and legal sources. Repository recorded in the project materials: <https://github.com/VAL-Jerono/Satellite>.
