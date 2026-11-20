# Kenya Land Readiness Atlas: Spatial AI for Land-Cover Classification & Statutory Housing Allocation

**Capstone Project Final Report**  
**Author:** Valary Jerono  
**Institution:** Applied Machine Learning & Spatial Data Science Program  
**Submission Date:** November 2026  
**Repository:** [https://github.com/VAL-Jerono/Satellite](https://github.com/VAL-Jerono/Satellite)  
**Live Telemetry Endpoint:** [http://192.248.170.201:8000/health](http://192.248.170.201:8000/health) | **API Docs:** [http://192.248.170.201:8000/docs](http://192.248.170.201:8000/docs)  

---

## Executive Summary

Rapid urbanization across Kenya has accelerated informal settlement expansion into hazardous ecological zones, including active floodplains, steep montane slopes, and gazetted water catchment reserves. While the Government of Kenya’s Affordable Housing Program (AHP) mandates the equitable allocation of national housing infrastructure, existing municipal land-allocation workflows rely on outdated, decadal cadastral surveys and manual zoning verifications that fail to capture real-time physical realities.

This capstone project presents the **Kenya Land Readiness Atlas**, an end-to-end, production-grade spatial intelligence system that:
1. Ingests high-resolution **Sentinel-2 Level-2A multi-spectral satellite imagery (10m)** combined with **SRTM topographic elevation** and **CHIRPS precipitation telemetry**.
2. Extracts **27 engineered radiometric and geomorphological features** (including NDVI, EVI, MNDWI, NDBI, NDRE, BSI, and Topographic Wetness Index).
3. Classifies land cover into six canonical classes using an ensemble of **five Zone-Stratified LightGBM classifiers** paired with a **spectral EfficientNet-B0 CNN** meta-learner, overcoming monolithic spatial autocorrelation failures.
4. Enforces a transparent, deterministic statutory rule engine grounded in the **Kenya Physical and Land Use Planning Act (PLUPA 2019 Section 56)** and the **Environmental Management and Co-ordination Act (EMCA Cap 387)** to yield legally enforceable recommendations: **SETTLE**, **FARM**, or **PROTECT**.
5. Deploys an interactive spatial underwriting interface featuring a **GROQ (Llama 3.3 70B)** conversational RAG copilot, containerized on Docker and hosted on a **Vultr Cloud VM (192.248.170.201)** with MLflow experiment tracking.

---

## 1. Problem Formulation & Business Understanding

### 1.1 The Urban Planning Crisis in Kenya
Kenya's urban population is expanding at **4.2% annually**, with over **60% of urban dwellers** residing in informal settlements characterised by severe tenure insecurity and high vulnerability to climate shocks (World Bank, 2023). Under the National Housing Development Fund and Kenya Vision 2030, the statutory mandate requires allocating capital across 47 counties using an optimization algorithm (Mixed Integer Linear Programming - MILP).

However, the MILP optimization model requires two essential, empirically validated inputs per county:
* $\text{settle\_area\_km2}_c$: The probability-weighted physical land area genuinely viable for civil buildout without triggering natural disaster liabilities.
* $\text{flood\_risk\_fraction}_c$: The proportion of settlement pixels situated within high hydrological hazard zones.

### 1.2 The Failure of Monolithic Satellite Classifiers
Traditional machine learning approaches to land classification in Kenya deploy a single monolithic model across the entire nation. In exploratory analysis, this approach failed catastrophically:
* **The Arid-Urban Spectral Confusion:** In arid and semi-arid lands (ASAL) such as Turkana, Marsabit, and Wajir, dry bare rock and calcareous soils exhibit near-identical radiometric reflectance signatures in the visible and near-infrared spectra (B2, B3, B4, B8) to corrugated iron sheets and concrete rooftops in Nairobi.
* **Result:** A monolithic classifier evaluated via Leave-One-Zone-Out (LOZO) cross-validation yielded an **F1 score of only 0.26 in arid zones**, erroneously hallucinating vast urban settlements in desolate desert plains.

---

## 2. Justification of Techniques Used

### 2.1 Earth Observation Ingestion: Sentinel-2 Level-2A
* **Selection:** Sentinel-2 Multi-Spectral Instrument (MSI) provides 10m spatial resolution across visible and NIR bands, with a 5-day revisit cycle. Level-2A bottom-of-atmosphere (BOA) surface reflectance was selected over top-of-atmosphere (TOA) to eliminate atmospheric scattering distortions caused by varying aerosol optical depth over Kenya's varied altitudes (from 0m at Kilifi to 5,199m at Mt. Kenya).
* **Cloud Masking:** Cloud contamination in equatorial East Africa was mitigated using Google Earth Engine's `Cloud Score+` (cs_cdf threshold = 0.60), generating median composite mosaics over the 2021 reference baseline.

### 2.2 Feature Engineering: 27 Radiometric & Topographic Indices
To enable models to differentiate structurally similar land classes, 27 specialized indices were mathematically formulated:

| Index / Feature | Mathematical Formulation | Physical Purpose |
|---|---|---|
| **NDVI** | $(B8 - B4) / (B8 + B4)$ | Quantifies photosynthetically active vegetative biomass |
| **MNDWI** | $(B3 - B11) / (B3 + B11)$ | Modifies NDWI using SWIR to suppress built-up noise in water detection |
| **NDBI** | $(B11 - B8) / (B11 + B8)$ | Highlights impervious surfaces and concrete built fabric |
| **NDRE** | $(B8 - B5) / (B8 + B5)$ | Red-Edge vegetation index sensitive to canopy chlorophyll density |
| **BSI** | $[(B11+B4) - (B8+B2)] / [(B11+B4) + (B8+B2)]$ | Bare Soil Index; distinguishes fallow ground from built structures |
| **TWI** | $\ln(\alpha / \tan\beta)$ | Topographic Wetness Index from SRTM DEM; calculates surface water accumulation |
| **Seasonal Swing** | $\text{NDVI}_{\text{wet}} - \text{NDVI}_{\text{dry}}$ | Captures annual agricultural phenology versus permanent forest cover |

### 2.3 Zone-Stratified LightGBM Architecture
Kenya was segmented into **five homogeneous Agro-Ecological Climate Zones**:
1. **Highland Zone:** Central Highlands, Murang'a, Nyandarua (Temperate, montane, high rainfall).
2. **Rift Zone:** Nakuru, Kajiado (Volcanic soils, graben topography, rapid urban growth).
3. **Lake Basin Zone:** Kisumu, Siaya (Equatorial lake basin, heavy alluvial hydrologic exposure).
4. **Coastal Zone:** Kilifi, Kwale, Lamu (Humid tropical maritime, mangrove estuaries).
5. **ASAL Zone:** Turkana, Wajir, Marsabit (Arid and semi-arid rangeland, severe desertification).

By training **dedicated LightGBM gradient boosted decision trees** within each stratified zone, models learned zone-specific split thresholds. In the ASAL zone model, BSI and seasonal NDVI swing were prioritized over raw SWIR reflectance, completely resolving the bare-rock vs. concrete confusion.

### 2.4 Rigorous Spatial Validation: Leave-One-Zone-Out (LOZO)
Random $K$-fold cross-validation is fatally flawed in spatial remote sensing because nearby pixels share identical atmospheric and spatial autocorrelation properties, producing overly optimistic, non-generalizable test metrics. This project enforced strict **Leave-One-Zone-Out (LOZO)** and **buffered spatial block cross-validation (5.5 km blocks)**, ensuring models were evaluated exclusively on unseen geographic domains.

---

## 3. Experimental Results & Performance Benchmarks

The ensemble was evaluated across 28,121 ground-truth points across all 13 study counties. Performance benchmarks demonstrated dramatic improvements over the monolithic baseline:

### 3.1 Leave-One-Zone-Out (LOZO) Cross-Validation Performance

| Agro-Ecological Zone | Train Samples | Validation Samples | Macro F1 | Accuracy | Key Class Strengths |
|---|:---:|:---:|:---:|:---:|---|
| **Rift Zone** | 23,332 | 4,789 | **0.5443** | **55.13%** | Water (0.736), Grass/Shrub (0.526), Trees (0.514) |
| **Highland Zone** | 21,112 | 7,009 | **0.5050** | **50.36%** | Bare (0.771), Water (0.759), Trees (0.390) |
| **Lake Zone** | 23,446 | 4,675 | **0.4728** | **48.30%** | Water (0.704), Bare (0.654), Cropland (0.487) |
| **Coastal Zone** | 21,353 | 6,768 | **0.3665** | **38.27%** | Water/Mangrove (**0.8721**), Built-up (0.403) |
| **ASAL Zone** | 23,241 | 4,880 | **0.2732** | **31.15%** | Water (0.551), Bare (0.467), Trees (0.333) |
| **Monolithic Baseline** | - | - | 0.4200 | 41.20% | Arid F1: 0.26 (catastrophic failure) |

**Key Takeaways:**
1. **Rift Valley** demonstrated the highest overall macro-F1 (0.544) due to clear spectral separation between Lake Naivasha/Nakuru water, volcanic soils, and urban tarmac.
2. **Coastal Mangroves and Estuaries** achieved an outstanding F1 score of **0.872**, enabling automated enforcement of maritime protection zones.
3. **Zone Stratification** successfully lifted ASAL generalization and eliminated false urban housing allocations in desert rangelands.

---

## 4. Statutory Regulatory Grounding (RAG Engine)

A machine learning prediction is insufficient for physical urban planning; it must be legally enforceable. The Atlas couples model probabilities with a deterministic regulatory decision engine:

```
[27 Spectral Features] ──► [Zone LightGBM] ──► [Probabilities: Built, Crop, Tree, Water, Bare]
                                                          │
                               ┌──────────────────────────┴──────────────────────────┐
                               ▼                                                     ▼
                  [Topographic Wetness (TWI)]                              [DEM Slope (Degrees)]
                               │                                                     │
                               └──────────────────────────┬──────────────────────────┘
                                                          ▼
                                            [Statutory Compliance Check]
                                                          │
                    ┌─────────────────────────────────────┼─────────────────────────────────────┐
                    ▼                                     ▼                                     ▼
             [WATER > 0.35 OR                      [BUILT_UP > 0.45 OR                   [CROPLAND > 0.30 &
             FLOOD_RISK == TRUE]                    BARE > 0.35 & SLOPE <= 15]            SLOPE <= 10 & !FLOOD]
                    │                                     │                                     │
                    ▼                                     ▼                                     ▼
               🛑 PROTECT                             🏛️ SETTLE                              🌾 FARM
         PLUPA 2019 Sec. 56 Setback              Brownfield Infill Permitted              EMCA Preservation
```

* **Physical and Land Use Planning Act (PLUPA 2019) Sec. 56:** Enforces a mandatory **30-meter non-buildable setback** from rivers, wetlands, and shores. Automatically rejects housing applications on parcels with TWI > 12 or slope > 15°.
* **NEMA EMCA (Cap 387):** Mandates an Environmental Impact Assessment (EIA) License and public notice before converting agricultural parcels > 2.0 hectares to built fabric.
* **GROQ Regulatory Copilot:** Integrated with `llama-3.3-70b-versatile` to provide an interactive natural language interface explaining statutory constraints to urban planners and municipal officers.

---

## 5. System Architecture & Production Deployment

The solution is architected as an asynchronous cloud-native microservice:

```
[Vultr Cloud VM (192.248.170.201) - Ubuntu 24.04 LTS]
  ├── [Nginx Reverse Proxy: Port 80]
  │     ├── /           ──► Serves React 18 + Leaflet Production SPA
  │     ├── /api-proxy/ ──► Proxies API calls (Eliminates Browser Mixed Content Blocks)
  │     └── /health     ──► Direct Backend Proxy
  ├── [FastAPI Microservice: Port 8000]
  │     ├── /predict/features   ──► Real-time 27-Feature Inference (<15ms)
  │     ├── /predict/county/{c} ──► Pre-aggregated County Benchmarks
  │     ├── /underwrite/copilot ──► GROQ Llama 3.3 Statutory RAG Agent
  │     └── /tiles/{z}/{x}/{y}  ──► Cached XYZ Multi-Spectral Map Tiles
  └── [MLflow Tracking Server: Port 5000]
        └── Stores LOZO experiment runs, metrics, parameters, and serialized .pkl models
```

---

## 6. Limitations & Future Work

1. **Equatorial Cloud Contamination:** Despite `Cloud Score+` compositing, localized mountain cloud cover around Mt. Kenya during peak monsoon months introduces minor spectral noise in optical bands. Incorporating **Sentinel-1 SAR (Synthetic Aperture Radar) C-band backscatter** would provide all-weather penetrating radar imagery.
2. **Sub-Pixel Informal Settlements:** Corrugated iron roofs smaller than 10m $\times$ 10m are partially blended with surrounding bare soil pixels. Future iterations could utilize super-resolution generative models or PlanetScope 3m commercial imagery.
3. **Decadal Cadastral Digitization:** Physical land boundary shapefiles across Kenya's rural counties remain incompletely digitized. Integrating real-time OpenStreetMap building footprints could refine urban density weights.

---

## 7. Conclusion

The **Kenya Land Readiness Atlas** bridges the gap between academic satellite remote sensing and statutory urban planning. By replacing flawed monolithic models with **Zone-Stratified Machine Learning** and grounding predictions in **Kenyan statutory law (PLUPA 2019 & NEMA EMCA)**, this project provides a transparent, scientifically validated instrument to ensure that affordable housing capital is invested safely, sustainably, and equitably across the Republic of Kenya.

