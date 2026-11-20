# Scripted and Reproducible Data Processing Pipeline

## Overview

This directory contains the complete, scripted, and fully reproducible data processing pipeline for the **Kenya Land Readiness Atlas**. The pipeline automates the entire journey from raw planetary-scale satellite imagery to validated, leak-free spatial partitions and statutory decision matrices.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & EXTRACTION (Google Earth Engine API via Colab Pro)           │
│    Sentinel-2 Level-2A (10 bands) + SRTM DEM + CHIRPS Rain + WorldCover     │
│    Script: ee_pull.py, ledger.py, patch_export.py                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. RADIOMETRIC CLEANING & FEATURE ENGINEERING (27 Features)                 │
│    Clipping [0, 1] + NDVI, EVI, SAVI, NDRE, NDWI, MNDWI, NDBI, BSI, TWI     │
│    Script: feature_engineering.py, enrich_features_ee.py                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. SPATIAL BLOCK CROSS-VALIDATION & LOZO PARTITIONING                       │
│    0.05° x 0.05° Grid (~5.5 km x 5.5 km) + 1-Block Buffer to Prevent Leakage │
│    Script: spatial_splits.py, zone_stratified_lgbm.py                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. STATUTORY AGGREGATION & COUNTY PROFILING                                 │
│    PLUPA 2019 §56 (30 m riparian buffer, 15° slope limit) + EMCA Cap 387    │
│    Script: build_county_summaries.py, regions.yaml                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. MASTER REPRODUCIBILITY ORCHESTRATION                                     │
│    Single-command execution across all stages                               │
│    Script: run_pipeline.sh, run_pipeline.py                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Script Inventory and Module Roles

| Script / Artifact | Operational Role in Data Pipeline | Primary Outputs |
|---|---|---|
| `run_pipeline.sh` | Master shell entrypoint that checks virtual environments, executes feature validation, aggregates county readiness, and verifies builds. | Full pipeline execution verification |
| `run_pipeline.py` | Master Python pipeline orchestrator executing end-to-end data processing, feature verification, and spatial fold generation. | Structured pipeline logs |
| `ee_pull.py` | Authenticated Google Earth Engine Python pipeline (`earthengine-api`) extracting cloud-masked Sentinel-2, SRTM, and CHIRPS data. | `data/raw/samples_2021_n400.csv` |
| `enrich_features_ee.py` | Computes server-side GEE annual reductions (median, cloud scoring) and extracts spatial points across 13 counties. | Point-level multi-band samples |
| `feature_engineering.py` | Mathematical library computing 27 non-linear remote sensing indices (NDVI, EVI, SAVI, NDRE, NDWI, MNDWI, NDBI, BSI, TWI). | Transformed feature arrays |
| `spatial_splits.py` | Divides Kenya into 0.05° spatial blocks, enforces 1-block buffer, and generates Leave-One-Zone-Out (LOZO) and GroupKFold train/val splits. | `data/splits/lozo_*.csv`, `split_summary.json` |
| `patch_export.py` | Extracts 10-channel Sentinel-2 spatial patches (`(10, 64, 64)` arrays) for the EfficientNet-B0 CNN. | `data/patches/` |
| `ledger.py` | Checkpoint and transaction ledger ensuring idempotency and failure recovery during large-scale GEE extraction runs. | Ingestion progress ledger |
| `zone_stratified_lgbm.py` | Pipeline module tying spatial folds to zone-stratified LightGBM feature matrices. | Training data partitions |
| `build_county_summaries.py` | Generates per-county statutory allocation JSON profiles and `all_counties.csv` for the FastAPI backend and React UI. | `data/county_summaries/*.json`, `all_counties.csv` |
| `regions.yaml` | Authoritative declarative mapping of 13 study counties across Kenya's 5 bioclimatic zones. | Configuration dictionary |

---

## How to Execute the Reproducible Data Pipeline

### Step 1: Run the Complete Master Pipeline
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Step 2: Regenerate Spatial Block Splits Independently
```bash
python3 spatial_splits.py
```

### Step 3: Recompute County Statutory Allocations
```bash
python3 build_county_summaries.py
```

### Step 4: Verify Feature Engineering on Sample Data
```bash
python3 -c "
import pandas as pd
from feature_engineering import compute_spectral_indices
df_sample = pd.DataFrame({
    'B2': [0.06], 'B3': [0.09], 'B4': [0.10], 'B5': [0.15],
    'B6': [0.22], 'B7': [0.25], 'B8': [0.27], 'B8A': [0.28],
    'B11': [0.24], 'B12': [0.20]
})
indices = compute_spectral_indices(df_sample)
print('Calculated remote sensing indices:')
for k, v in indices.items():
    print(f'  {k:8s}: {v[0]:.4f}')
"
```
