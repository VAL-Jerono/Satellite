# CRISP-DM Plan — Kenya Land Readiness Atlas
## Satellite Modeling to LORO F1 ≥ 0.60
### *Sub-pipeline of: Household-Guided, Land-Aware Affordable Housing Allocation in Kenya*

> **Scope of this document:** The satellite analysis only. Goal: produce a
> calibrated, geographically-generalising land-cover classifier whose outputs
> (`settle_area_km2`, `flood_risk_fraction`) can feed directly into the
> MILP allocation engine (Tier 3 of the master pipeline).
>
> **Primary success criterion:** LORO macro-F1 ≥ 0.60 across all climate zones.
> Current baseline: 0.42 mean LORO, 0.26 arid worst-case.

---

## Phase 1 — Business Understanding

### 1.1 What We Are Solving

The MILP allocator needs two satellite-derived numbers per county:

| Output | Description | Feeds Into |
|---|---|---|
| `settle_area_km2` | Probability-weighted area of Settle-class pixels | MILP land ceiling constraint |
| `flood_risk_fraction` | Share of Settle pixels with high TWI | Equity audit / penalty term |

Without a model that generalises across Kenya's five climate zones, those
numbers cannot be trusted — especially for ASAL counties (Turkana, Wajir,
Marsabit), which have the deepest housing vulnerability but were previously
classified at LORO F1 = 0.26.

### 1.2 Success Criteria

| Metric | Target | Current Baseline |
|---|---|---|
| LORO macro-F1 (mean, all zones) | **≥ 0.60** | 0.42 |
| LORO macro-F1 (arid zone, worst) | **≥ 0.50** | 0.26 |
| ECE (calibration) | **≤ 0.08** | 0.18 (LightGBM) |
| Block CV macro-F1 | **≥ 0.75** | 0.58 |

### 1.3 How This Plugs Into the Master Pipeline

```
[Satellite: this document]
        │
        │  settle_area_km2_c (per county, probability-weighted)
        │  flood_risk_fraction_c
        │  zone_confidence_weight_c
        ▼
[MILP Tier 3]  u_c ≤ land_units_c = settle_area_km2_c × density × zone_conf
```

Failing to hit LORO ≥ 0.60 means zone_confidence_weight will heavily
discount ASAL estimates, and the MILP will under-allocate to vulnerable
arid counties. Hitting ≥ 0.60 means the satellite constraint is trusted at
full weight.

---

## Phase 2 — Data Understanding

### 2.1 Data Sources

| Source | Description | Format |
|---|---|---|
| **Sentinel-2 SR** (COPERNICUS/S2_SR_HARMONIZED) | 10-band surface reflectance, annual + seasonal | GEE → EE pull |
| **Cloud Score+** (GOOGLE/CLOUD_SCORE_PLUS) | Cloud masking, cs_cdf ≥ 0.6 | Linked to S2 |
| **ESA WorldCover v200** | 2021 land-cover label map → remapped to 6 classes | GEE |
| **USGS SRTM** | Elevation + slope | GEE |
| **WWF HydroSHEDS 15ACC** | Flow accumulation for TWI | GEE |

### 2.2 Feature Set (23 features total — Week 12 upgrade)

| Group | Features | Count |
|---|---|---|
| S2 bands | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 | 10 |
| Spectral indices | NDVI, NDWI, MNDWI, NDBI | 4 |
| NDVI temporal range | NDVI_p10, NDVI_p90, NDVI_stdDev | 3 |
| **NEW** Seasonal phenology | ndvi_wet, ndvi_dry, ndvi_seasonal_swing | 3 |
| Topographic | elev, slope | 2 |
| **NEW** Flood risk | twi (Topographic Wetness Index) | 1 |

### 2.3 Label Scheme (6 classes)

| Class | Index | WorldCover Source Codes |
|---|---|---|
| built_up | 0 | 10 (built-up) |
| cropland | 1 | 20 (cropland) |
| trees | 2 | 30 (tree cover), 95 (mangrove) |
| grass_shrub | 3 | 40, 30 (shrubs) |
| water | 4 | 80 (water), 90 (wetland) |
| bare | 5 | 60 (bare), 70 (moss) |

### 2.4 Study Regions (13 counties, 5 climate zones)

| Climate Zone | Counties | Challenge |
|---|---|---|
| highland_zone | highland, muranga, nyandarua | Good baseline (LORO F1 ≈ 0.38) |
| coastal_zone | coast, kwale, lamu | Mangrove spectral uniqueness (LORO F1 ≈ 0.38) |
| asal_zone | arid, turkana, wajir | Bright bare soil = built_up confusion (LORO F1 ≈ 0.26) |
| lake_zone | lake_basin, siaya | Wetland/water spectral mixing (LORO F1 ≈ 0.37) |
| rift_zone | nakuru, kajiado | Soda lakes, savanna variability |

### 2.5 Known Data Quality Issues

| Issue | Evidence | Impact |
|---|---|---|
| ASAL class deficit (built_up, water) | wajir: 25 water samples; arid: 22 | Model starved of rare classes in key zone |
| Global normalisation blurs zone signals | NDVI=0.20 means grass in Turkana, degraded farm in Nyandarua | Primary driver of LORO gap |
| No seasonal split in NDVI stats | Annual p10/p90 collapse wet/dry signal | Cropland ≡ forest in temporal space |
| No building density feature | built_up class = slum + tower blocks | Misclassification of sparse ASAL settlements |

---

## Phase 3 — Data Preparation

### 3.1 Pipeline Stages

```
Stage A  GEE Pull          ee_pull.py
         ├─ 13 counties × 4 sub-tiles × PER_CLASS stratified samples
         ├─ 23 features (10 bands + 4 idx + 3 temporal + 2 topo + TWI + seasonal)
         ├─ climate_zone column added per region
         └─ Output: data/raw/samples_2021_n400.csv

Stage B  Patch Export       data/pipeline/patch_export.py
         ├─ Each sample row → 64×64 .npy patch extracted from GEE at 10m
         └─ Output: data/patches/{region}_{label}_{idx}.npy

Stage C  Band Statistics    data/compute_band_stats.py
         ├─ Global:  data/band_stats.json (10 means, 10 stds)
         └─ Per-zone: data/band_stats_by_zone.json (5 zones × 10 bands)

Stage D  Feature CSV        data/npy_to_csv.py
         ├─ For GBDT tabular training: extract 74 patch-level features
         └─ Output: data/features.csv

Stage E  Spatial Splits     data/pipeline/spatial_splits.py
         ├─ Block CV: data/splits/fold_{k}_train/val.csv  (k=0..4)
         ├─ LORO:     data/splits/loro_{region}_train/val.csv  (13 regions)
         └─ LOZO:     data/splits/lozo_{zone}_train/val.csv    (5 zones)
```

### 3.2 Normalisation Strategy

**Old (broken):** Single global (mean, std) — NDVI of 0.20 gets normalised identically regardless of zone.

**New (required):** Zone-stratified normalisation — compute (mean_z, std_z) for each of 5 climate zones.

### 3.3 Class Imbalance Handling

- Stratified EE sampling: PER_CLASS=400 per class per region
- Prevalence weights: each sample weighted by class_share / n_samples
- Focal loss: γ=1.5 + class-frequency alpha weights
- ASAL rare-class augmentation: oversample built_up and water 3× in LOZO arid training

---

## Phase 4 — Modelling

### 4.1 Model Progression

```
Level 0  Tabular Baseline (already done)
         Decision Tree + LightGBM on 23 spectral features
         Results: Block CV macro-F1 ≈ 0.58 | LORO ≈ 0.42

Level 1  Zone-Stratified LightGBM (NEW)
         One model per climate zone; LOZO evaluation
         Target: LORO ≥ 0.55 | LOZO ≥ 0.50

Level 2  EfficientNet-B0 + Spectral Channel Attention (Champion CNN)
         10-band 64×64 patches; FocalLoss + Mixup + CosineAnnealing
         Target: Block CV ≥ 0.75 | LORO ≥ 0.60

Level 3  Temperature Scaling Calibration
         Scalar T fit on held-out fold; ECE ≤ 0.08
```

### 4.2 LOZO Strategies for ASAL

1. Zone-normalisation at inference using arid-zone band statistics
2. NDBI + BSI + SAVI as primary bare/built discriminants
3. TWI as flood gate for wet-area disambiguation in dry landscapes

---

## Phase 5 — Evaluation

### 5.1 Evaluation Protocol

| Test | Description | Primary Metric |
|---|---|---|
| Block CV (5-fold) | Blocks ≈ 5.5 km, 1-block buffer | Macro-F1, ECE |
| LORO (13 counties) | Train on 12, test on 1 | Macro-F1 per county |
| LOZO (5 zones) | Train on 4 zones, test on 1 | Macro-F1 per zone |
| Calibration | Temperature scaling on held-out fold | ECE (10 bins) |

### 5.2 Zone Targets

| Zone | LOZO F1 Target |
|---|---|
| highland | ≥ 0.68 |
| coastal | ≥ 0.60 |
| **asal** | **≥ 0.50** |
| lake | ≥ 0.62 |
| rift | ≥ 0.65 |

### 5.3 Critical Confusion Pairs

1. bare ↔ built_up (ASAL — inflates settle_area_km2 if wrong)
2. cropland ↔ trees (coastal — Farm vs. Protect label)
3. water ↔ bare (ASAL — dry riverbeds vs. open water)

---

## Phase 6 — Deployment

### 6.1 Outputs

```
models/land_cover.onnx                   ONNX model for FastAPI serving
models/temp_scaling_T.json               Temperature T per zone
data/atlas/county_land_stats.csv         settle_area_km2 + flood_risk_fraction per county
data/atlas/zone_confidence.json          LOZO F1 per zone (confidence weights for MILP)
```

### 6.2 MILP Integration Parameters

```yaml
land:
  settle_area_km2:      {county: value}   # from atlas_export.py
  flood_risk_fraction:  {county: value}
  zone_confidence:      {asal_zone: 0.83}  # LOZO F1 values
  density_assumption:   200                # units per km² (policy param)
  min_settle_floor_km2: 5.0               # activation gate
```

---

## What To Expect (Summary)

| Phase | Deliverable | Expected Result |
|---|---|---|
| EDA | Class distribution by zone; spectral signature plots | ASAL class deficit confirmed; seasonal swing visible |
| LightGBM baseline | Block CV + LORO table | Block CV ≈ 0.58–0.65 / LORO ≈ 0.42–0.50 |
| Zone-stratified LightGBM | One model per zone; LOZO table | LOZO arid improves to ≈ 0.48 |
| EfficientNet-B0 CNN | 5-fold CV + LORO + LOZO | Block CV ≈ 0.72–0.78 / LORO ≥ 0.60 |
| Temperature scaling | ECE plot before/after | ECE drops to ≤ 0.08 |
| Atlas export | county_land_stats.csv | 47-row table ready for MILP |
