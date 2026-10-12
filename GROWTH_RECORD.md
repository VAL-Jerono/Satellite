# Kenya Land Readiness Atlas — Growth Record
### Pipeline State Snapshot · Week 11 → Week 12 Transition · October 2026

---

## 0. Purpose of This Document

This file is a permanent audit trail. Each week we append a new section recording
**what we measured**, **what failed**, **what we changed**, and **what the result was**.
Do not delete old sections — growth only becomes visible across entries.

---

## 1. Week 11 Baseline State (Before Improvements)

### 1.1 Data Collected

| County | Climate Zone | Samples | Built-up | Cropland | Water | Notes |
|---|---|---|---|---|---|---|
| highland | Highland | 2,256 | 331 | 378 | 400 | Good coverage |
| muranga | Highland | 2,388 | 400 | 400 | 388 | Good coverage |
| nyandarua | Highland | 2,365 | 400 | 400 | 386 | Good coverage |
| coast | Coastal | 2,351 | 400 | 400 | 351 | Good coverage |
| kwale | Coastal | 2,372 | 400 | 400 | 372 | Good coverage |
| lamu | Coastal | 2,045 | 350 | 350 | 400 | Trees sparse |
| arid | ASAL | 1,474 | 210 | **83** | **22** | ⚠️ Severe class deficit |
| turkana | ASAL | 2,029 | 400 | 400 | **109** | Water scarce |
| wajir | ASAL | 1,377 | **157** | 400 | **25** | ⚠️ Multi-class deficit |
| lake_basin | Lake | 2,275 | 375 | 375 | 400 | Good coverage |
| siaya | Lake | 2,400 | 400 | 400 | 400 | Perfect |
| nakuru | Rift | 2,389 | 400 | 400 | 400 | Good coverage |
| kajiado | Rift | 2,400 | 400 | 400 | 400 | Perfect |
| **Total** | | **28,121** | | | | |

### 1.2 Features Used (19 total)

| Feature Group | Features | Source |
|---|---|---|
| Sentinel-2 bands | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 | Cloud Score+ masked annual median |
| Spectral indices | NDVI, NDWI, MNDWI, NDBI | Derived in-notebook |
| NDVI temporal | NDVI_p10, NDVI_p90, NDVI_stdDev | Annual min/max/std |
| Topographic | elev, slope | USGS SRTM |

### 1.3 Baseline Model Performance

#### Spatial Block CV (5-fold, 0.05° blocks with 1-block buffer):

| Model | Accuracy | Macro-F1 | ECE |
|---|---|---|---|
| Decision Tree (rule) | 0.4258 | 0.4230 | 0.0397 |
| LightGBM | 0.5737 | **0.5765** | 0.1841 |

#### Leave-One-Region-Out (LORO) — Mean across all 13 counties:

| Model | Mean Accuracy | Mean Macro-F1 |
|---|---|---|
| Decision Tree | 0.3821 | 0.3339 |
| LightGBM | 0.4413 | **0.4239** |

#### Worst LORO Cases (LightGBM):

| Held-Out Region | Macro-F1 | Root Cause |
|---|---|---|
| highland | 0.3816 | Trained without any highland zone data |
| coast | 0.3771 | Mangrove/coastal spectral signature unseen |
| arid | 0.2591 | Bright bare soil confused with built-up |
| lake_basin | 0.3697 | Wetland/water spectral mix |

---

## 2. Identified Weaknesses

### 2.1 Critical: Global Normalisation Across Climate Zones

**Problem**: A pixel with NDVI=0.20 means "healthy grass" in Turkana but "severely degraded farmland" in Nyandarua. The band means/stds used in CNN normalisation are pooled globally, blurring zone-specific signal ranges.

**Evidence**: LORO Macro-F1 drops from 0.58 (block CV) to 0.17–0.26 on arid/coastal zones.

**Target fix**: Compute per-climate-zone `(mean, std)` for all 10 bands.

---

## 2.2 High Impact: Multi-Temporal NDVI Swing Missing

**Problem**: The NDVI temporal statistics are NOT split into wet vs. dry season, losing the most powerful land-cover discriminant in Kenya.

**The key signal**:
- Cropland: NDVI swings ~0.5 during long rains (Mar–May), crashes dry season
- Forest: Stable NDVI year-round (~0.6–0.8)
- Bare soil: Always low (~0.05–0.15)
- Arid scrub: Low and stable

**Target fix**: Pull `ndvi_wet` (Mar–May) and `ndvi_dry` (Jul–Sep) from EE and add `ndvi_seasonal_swing = ndvi_wet - ndvi_dry`.

---

## 2.3 High Impact: No Building Density Signal

**Problem**: The `built_up` WorldCover class covers everything from iron-sheet slums to formal apartment blocks. The model has zero information about building density, height, or footprint — the exact features Kenya housing contractors need.

**Target fix**: Add Topographic Wetness Index (TWI) from HydroSHEDS + plan for Google Open Buildings v3 density layer.

---

## 2.4 Structural: Inappropriate CV Strategy for Actionable Advice

**Problem**: Regular LORO tests if a model trained on 12 counties predicts the 13th. BUT when a contractor moves into a region with *zero* training data, predicting it without any local sample is unrealistic. What contractors actually need is **Leave-One-Climate-Zone-Out (LOZO)** validation to test climate generalization, AND stratified per-climate-zone model thresholding.

---

## 3. Implemented Improvements (Week 12 Pipeline Updates)

### 3.1 GEE Feature Additions (`data/pipeline/ee_pull.py`)

Added 4 new features (19 → 23 total features):
1. `ndvi_wet`: Mean Sentinel-2 NDVI during Kenya long rains (March 1 – June 1)
2. `ndvi_dry`: Mean Sentinel-2 NDVI during dry season (July 1 – October 1)
3. `ndvi_seasonal_swing`: `ndvi_wet - ndvi_dry`
4. `twi`: Topographic Wetness Index derived from HydroSHEDS flow accumulation + SRTM slope:
   $$\text{TWI} = \ln\left(\frac{\text{FlowAccumulation} + 1}{\tan(\text{slope}) + 0.001}\right)$$

### 3.2 Zone-Stratified Normalisation (`data/pipeline/band_stats.py`)

Updated normalization pipeline to export per-zone band statistics:
- `highland_zone`, `coastal_zone`, `asal_zone`, `lake_zone`, `rift_zone`
- Exports `band_stats_by_zone.json` alongside `band_stats.json`.

---

## 4. Contractor Actionability Scorecard

| Requirement | Week 11 Baseline | Week 12 State | Next Action |
|---|---|---|---|
| Predict buildability on new sites | ❌ 57% Block / 44% LORO | 🟡 23 features + zone stats | Run baseline notebook on new feature set |
| Flood risk identification | ❌ Slope only | 🟢 Slope + HydroSHEDS TWI | Calibration analysis per TWI percentile |
| Agricultural encroachment risk | ❌ Annual NDVI | 🟢 Wet/dry seasonal swing | Thresholding for seasonal cropland |
| Infrastructure readiness | ❌ Land cover only | 🟡 Proximity features planned | Integrate Open Buildings density |

---

## 5. Next Execution Steps

1. Run the updated `land_atlas_baseline.ipynb` notebook to retrain models using the 23-feature dataset.
2. Evaluate LightGBM and Decision Tree models under Spatial Block CV, LORO, and LOZO splits.
3. Generate contractor decision matrices comparing predictions across climate zones.
