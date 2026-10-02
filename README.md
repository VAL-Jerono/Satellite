# Kenya Land Readiness Atlas 🛰️

> **Capstone project** — a production-grade end-to-end satellite image analysis system for land-cover classification and readiness assessment over Kenya.

[![CI](https://github.com/your-org/land-atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/land-atlas/actions/workflows/ci.yml)

---

## What it does

Given a location in Kenya, the system:
1. **Classifies land cover** (built-up, cropland, trees, grass/shrub, water, bare) from a 64×64 Sentinel-2 patch using an EfficientNet-B0 fine-tuned CNN.
2. **Shows built-up trend** over time (NDBI time series from Earth Engine).
3. **Flags hazard** (flood risk from terrain + water class probability).
4. **Recommends action** — **settle / farm / protect** — using a transparent rule layer over model outputs (not a NN decision).

Labels come from **WorldCover 2021**. Accuracy is reported as agreement with WorldCover; an independent hand-labelled test set is required before citing numbers.

---

## Architecture

```
Earth Engine (Sentinel-2, WorldCover labels, SRTM)
  └─► ee_pull.py          # stratified point sampling, cached CSV
  └─► patch_export.py     # 64×64 .npy patch download
       └─► spatial_splits.py   # buffered block-CV + LORO folds
            └─► train.py       # EfficientNet-B0 fine-tune, MLflow tracked
                 └─► model.py  # ONNX export (CPU-first)
                      └─► api/main.py      # FastAPI: /predict/point, /tiles
                           └─► frontend/app.py  # Streamlit + Folium map
                                └─► monitoring/drift.py  # Evidently
```

All inference runs on **CPU** (ONNX Runtime). No GPU required for serving.

---

## Repo structure

```
Satellite/
├── configs/
│   ├── config.yaml          # training, model, serving, paths
│   └── regions.yaml         # 4 study regions, class colours, WorldCover remap
├── data/
│   ├── pipeline/
│   │   ├── ee_pull.py       # Step 1: EE point sampling → CSV
│   │   ├── patch_export.py  # Step 2: download 64×64 .npy patches
│   │   └── spatial_splits.py# Step 3: buffered block-CV folds
│   ├── raw/                 # sampled CSV (gitignored if large)
│   ├── patches/             # .npy files + manifest.csv
│   ├── splits/              # fold_k_train/val.csv + LORO CSVs
│   └── county_summaries/    # pre-aggregated county JSON/CSV
├── training/
│   ├── dataset.py           # PatchDataset (normalise, augment)
│   ├── model.py             # build_model() + export_onnx()
│   └── train.py             # full training loop + MLflow + ONNX export
├── models/                  # ONNX model + checkpoints (gitignored)
├── api/
│   ├── main.py              # FastAPI service
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit map app
│   └── requirements.txt
├── monitoring/
│   ├── drift.py             # Evidently drift + seasonal simulation
│   └── reports/
├── tests/
│   ├── test_model.py        # CNN forward + ONNX export
│   ├── test_splits.py       # spatial leakage checks
│   └── test_api.py          # API with mocked ONNX session
├── notebooks/
│   └── land_atlas_baseline.ipynb   # Colab baseline (LightGBM + spatial CV)
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt         # full project deps
└── .github/workflows/ci.yml
```

---

## Quickstart

### 1. Data pipeline (run on Colab or locally with EE access)

```bash
# Set your GCP project registered for Earth Engine
export GEE_PROJECT=your-project-id

# Step 1: Pull stratified points from EE → CSV (cached)
python -m data.pipeline.ee_pull --project $GEE_PROJECT

# Step 2: Download 64×64 patches as .npy (slow, do on Colab GPU for speed)
python -m data.pipeline.patch_export --project $GEE_PROJECT

# Step 3: Build spatial-block folds
python -m data.pipeline.spatial_splits
```

### 2. Train

```bash
# Train all 5 folds (MLflow tracked, ONNX exported from best fold)
python -m training.train

# Or single fold for quick smoke test:
python -m training.train --fold 0

# Leave-one-region-out:
python -m training.train --loro highland
```

Open MLflow at http://localhost:5000 (via `docker compose up mlflow`).

### 3. Run the full stack

```bash
# Copy your trained ONNX model to models/land_cover.onnx, then:
docker compose up

# API:      http://localhost:8000/docs
# Frontend: http://localhost:8501
# MLflow:   http://localhost:5000
```

### 4. Local dev (no Docker)

```bash
pip install -r requirements.txt

# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
streamlit run frontend/app.py
```

### 5. Drift monitoring

```bash
# Simulate dry→wet seasonal drift and run Evidently report:
python -m monitoring.drift --simulate
```

---

## Accuracy caveats

| What is reported | What it means |
|---|---|
| Block CV macro-F1 | Agreement with WorldCover within training regions; spatial leakage removed |
| LORO macro-F1 | Held-out region; the honest generalisation number |
| Weighted accuracy | Olofsson-style: class share × recall, so rare built-up doesn't distort headline |
| Neither | Ground truth; add 300–500 hand-labelled points before quoting any figure |

Spatial block size: **0.05°** (≈ 5.5 km). Buffer: 1 block (≈ 5.5 km gap between train and test).

---

## Dissertation link

`data/county_summaries/all_counties.csv` exports: `county_code`, `built_up_share`, `cropland_share`, `built_up_growth`, `flood_exposure`. Joins to HFVS / land allocation work as the buildability input layer.

---

## Timeline (course-aligned)

| Dates | Week | Milestone |
|---|---|---|
| 2–4 Oct | Before Wk 11 | ✅ Repo skeleton, configs, baseline notebook |
| 5–11 Oct | Wk 11 FastAPI+Docker | Data pipeline v1, spatial splits, API skeleton |
| 12–18 Oct | Wk 12 MLflow+CI | CNN fine-tune, experiment tracking, GitHub Actions |
| 19–25 Oct | Wk 13 Drift | Evidently monitoring, frontend, rule overlay. **Feature freeze 24 Oct** |
| 26 Oct | Wk 14 War room | Cloud deploy, load test |
| 2 Nov | Wk 15 | Demo, architecture review |

**Cut rule:** if behind by 19 Oct, drop LSTM forecast and cross-country test. Keep classifier + API + map.
