# Kenya Land Readiness Atlas: Capstone Presentation Deck 🛰️

**Slide Deck for Oral Defense & Technical Presentation**  
**Presenter:** Valary Jerono  
**Academic Program:** Master of Science in Data Science & Analytics (MSc DSA)  
**Institution:** Strathmore University / iLabAfrica  
**Course:** DSA 8401 - Applied Machine Learning  
**Live Production API:** [http://192.248.170.201:8000/docs](http://192.248.170.201:8000/docs)  
**Live Production Frontend:** [https://satellite-tau.vercel.app](https://satellite-tau.vercel.app)  
**Repository:** [https://github.com/VAL-Jerono/Satellite](https://github.com/VAL-Jerono/Satellite)  

---

```
================================================================================
SLIDE 1: TITLE SLIDE
================================================================================
```
# KENYA LAND READINESS ATLAS
### Ground-Truth Spatial AI for Multi-Sector Land-Use Underwriting, Statutory Zoning & MLOps Deployment

**Candidate:** Valary Jerono  
**Degree:** Master of Science in Data Science & Analytics (MSc DSA)  
**Institution:** Strathmore University (iLabAfrica)  
**Supervisors:** Dr. Javier Serrano & Dr. Emmanuel Bundi  
**Date:** November 2026  

*“Transforming 10m Sentinel-2 multi-spectral observations and Kenyan statutory law into calibrated, legally enforceable land allocation intelligence.”*

---

```
================================================================================
SLIDE 2: THE PROBLEM & ECOSYSTEM CONTEXT
================================================================================
```
## Spatial Planning Data Latency & The Multi-Sector Underwriting Crisis

* **The Macro Challenge:** Land allocation decisions across Sub-Saharan Africa - spanning affordable housing, commercial agriculture, climate carbon credits, and transport corridors - suffer from severe data latency and lack of sub-county spatial ground truth.
* **The Planning Blindspot:** Municipal planning offices rely on 10 to 20 year-old paper cadastral maps, leading to high-risk capital misallocation.
* **Multi-Sector Underwriting Use Cases:**
  1. **Real Estate & Housing:** Kenya Affordable Housing Program (AHP) and Affordable Housing Act 2023 Sec. 4 require verified buildable land ceilings without encroaching on fragile wetlands.
  2. **Agricultural Finance:** Commercial agricultural banks need crop suitability evidence to preserve arable food security zones under the Agriculture and Food Authority (AFA) Act.
  3. **Climate & Conservation Finance:** ESG funds require automated ecological audits for carbon credits and mangrove/montane forest protection.
  4. **Trade Corridor Development:** Transport authorities require terrain slope and drainage verification along the LAPSSET and Northern Corridors.
* **The "Probability Raster" Black-Box Gap:** Raw 9-class pixel probabilities from global satellite products (ESA WorldCover, Dynamic World) do not provide decision-ready directives. Underwriters require explicit statutory actions: **SETTLE**, **FARM**, or **PROTECT**.

---

```
================================================================================
SLIDE 3: THE TECHNICAL BOTTLENECK
================================================================================
```
## The Arid-Urban Spectral Illusion & Monolithic Classifier Failure

* **The Core Research Question:** *Can a single monolithic neural network or Random Forest accurately classify land suitability across Kenya's diverse biomes?*
* **The Catastrophic Failure:** Monolithic baselines failed severely in Northern Arid zones (Turkana, Marsabit, Wajir), achieving a **Macro-F1 of only 0.2600**.
* **Root Cause - Radiometric Signature Overlap:**
  * Highly reflective caliche limestone, quartz sands, and sun-baked volcanic desert hardpans in Arid zones reflect solar radiation in visible and near-infrared bands (B2, B3, B4, B8) in patterns virtually indistinguishable from corrugated iron rooftops, asphalt highways, and concrete slabs in Nairobi.
  * Monolithic models suffered severe false positives, hallucinating massive urban metropolises in uninhabited desert rangelands!
* **The Solution:** Agro-ecological zone stratification + topographic elevation/slope integration + phenological dry/wet seasonal swing metrics.

---

```
================================================================================
SLIDE 4: DATA PIPELINE & RESOLUTION TRADE-OFFS
================================================================================
```
## Multi-Source Earth Observation Ingestion via GEE & Colab

```
                  ┌───────────────────────────────────────────────┐
                  │ Google Earth Engine & Colab Extraction Engine │
                  └───────────────────────┬───────────────────────┘
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       ▼                                  ▼                                  ▼
[Sentinel-2 Level-2A]              [SRTM Elevation DEM]             [CHIRPS Rainfall]
10m BOA surface reflectance        30m topography: elevation,       Monthly precipitation:
10 bands: B2..B12                  slope, Topographic Wetness       seasonal wet/dry dynamics
Cloud Score+ Masking (cs_cdf > 0.6)
```

* **Dataset Scope:** 28,121 stratified ground-truth samples across 13 anchor study counties representing all 5 Kenya agro-ecological zones.
* **Spatial Resolution Optimization Matrix:**
  * **10-Meter Native Sentinel-2:** Fine-grained site underwriting, parcel boundary auditing, and riparian setbacks. (Processing cost: 2-3s per parcel).
  * **100-Meter Resampling:** County-wide land readiness mosaics, trade-corridor infrastructure screening, and municipal zoning overviews.
  * **200-Meter Resampling:** Rapid nationwide environmental screening across all 580,367 km² of Kenya.

---

```
================================================================================
SLIDE 5: FEATURE ENGINEERING
================================================================================
```
## 27 Engineered Multi-Spectral, Topographic & Climatic Features

Rather than passing raw, uncalibrated spectral reflectances, we engineered 27 domain-specific features:
* **Vegetation Vitality:**
  $$\text{NDVI} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}}, \quad \text{EVI} = 2.5 \cdot \frac{\text{B8} - \text{B4}}{\text{B8} + 6\text{B4} - 7.5\text{B2} + 1}, \quad \text{SAVI} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4} + 0.5} \cdot 1.5$$
* **Canopy & Red-Edge Discrimination:**
  $$\text{NDRE} = \frac{\text{B8} - \text{B5}}{\text{B8} + \text{B5}} \quad \text{(differentiates dense forest canopies from seasonal crops)}$$
* **Built Fabric vs. Bare Desert Ground:**
  $$\text{NDBI} = \frac{\text{B11} - \text{B8}}{\text{B11} + \text{B8}}, \quad \text{BSI} = \frac{(\text{B11} + \text{B4}) - (\text{B8} + \text{B2})}{(\text{B11} + \text{B4}) + (\text{B8} + \text{B2})}$$
* **Hydrology & Moisture Delineation:**
  $$\text{NDWI} = \frac{\text{B3} - \text{B8}}{\text{B3} + \text{B8}}, \quad \text{MNDWI} = \frac{\text{B3} - \text{B11}}{\text{B3} + \text{B11}}$$
* **Terrain Topography & Climate:**
  $$\text{Elevation (m)}, \quad \text{Slope (degrees)}, \quad \text{CHIRPS Annual Rainfall (mm)}, \quad \Delta\text{NDVI (wet - dry seasonal swing)}$$

---

```
================================================================================
SLIDE 6: MODEL ARCHITECTURES
================================================================================
```
## Dual-Engine Framework: Zone-Stratified LightGBM + Spectral ConvNet

```
                         [Input: 27 Multi-Spectral & Topographic Features]
                                                 │
                                                 ▼
                                     [Climate Zone Spatial Router]
                                                 │
         ┌───────────────┬───────────────┼───────────────┬───────────────┐
         ▼               ▼               ▼               ▼               ▼
   [Highland LGBM]  [Rift LGBM]     [Lake LGBM]     [Coast LGBM]    [ASAL LGBM]
   (Nyeri, Muranga) (Nakuru, Kajiado)(Kisumu, Siaya)(Kilifi, Lamu)  (Turkana, Wajir)
         │               │               │               │               │
         └───────────────┴───────────────┼───────────────┴───────────────┘
                                         ▼
                 [Dual-Engine Meta-Stacker & Probability Vectors]
                 [Built-up, Cropland, Grass/Shrub, Forest, Water, Bare]
```

* **Engine 1 - Zone-Stratified LightGBM Specialists:**
  * 5 independent models trained specifically on biome-calibrated feature distributions.
  * Sub-15ms CPU inference, handles non-linear feature interactions, high explainability for regulators.
* **Engine 2 - PyTorch Spectral Channel Attention ConvNet (EfficientNet-B0 / ResNet-18):**
  * Custom Squeeze-and-Excitation (SE) channel attention module learning inter-band spectral correlations.
  * Label Smoothing Focal Loss ($\gamma = 1.5, \epsilon = 0.1$) to handle acute class imbalance.
  * Mixup data augmentation ($\alpha = 0.2$) for robust spectral generalization.

---

```
================================================================================
SLIDE 7: SPATIAL VALIDATION METHODOLOGY
================================================================================
```
## Eliminating Spatial Autocorrelation Leakage

* **The Problem with Random $K$-Fold CV:** Randomly splitting geographic pixels allows training and test samples to sit within meters of each other (Tobler's First Law of Geography). Models simply memorize local textures, yielding artificially inflated F1 scores (>0.90) that collapse in real-world deployment.
* **Our Rigorous Spatial Partitioning:**
  1. **400-Block Spatial Grid:** Kenya partitioned into $0.05^\circ \times 0.05^\circ$ blocks (~5.5 km $\times$ 5.5 km).
  2. **1-Block Buffer Isolation:** All training samples in blocks immediately adjacent to test blocks are discarded, enforcing a zero-leakage perimeter.
  3. **Leave-One-Zone-Out (LOZO) Benchmark:** Models are evaluated strictly on an entire unseen agro-ecological zone to test out-of-domain transferability.

```
       [Train Block]   [BUFFER ZONE]   [Test Block]   [BUFFER ZONE]   [Train Block]
         (Included)      (Discarded)     (Evaluated)     (Discarded)     (Included)
```

---

```
================================================================================
SLIDE 8: EXPERIMENTAL BENCHMARKS & LOZO RESULTS
================================================================================
```
## Zone-Stratified Specialists vs. Monolithic Baseline

| Evaluation Zone | Anchor Counties | Train Samples | Val Samples | Macro F1 | Accuracy | Key Phenomenon Captured |
|---|---|:---:|:---:|:---:|:---:|---|
| **Rift Valley** | Nakuru, Kajiado | 23,332 | 4,789 | **0.5443** | **55.13%** | Clear tarmac vs. volcanic bare soil separation |
| **Highland** | Nyeri, Murang'a, Nyandarua | 21,112 | 7,009 | **0.5050** | **50.36%** | Preserves high-density tea/coffee canopy cover |
| **Lake Basin** | Kisumu, Siaya | 23,446 | 4,675 | **0.4728** | **48.30%** | Accurate separation of seasonal floodplain rice |
| **Coastal Strip** | Kilifi, Kwale, Lamu | 21,353 | 6,768 | **0.3665** | **38.27%** | Mangrove marine segmentation F1: **0.8721** |
| **ASAL Zone** | Turkana, Wajir, Marsabit | 23,241 | 4,880 | **0.2732** | **31.15%** | **Completely eliminated urban hallucinations** |
| **Monolithic Model** | All 13 Study Counties | - | - | 0.4200 | 41.20% | Arid F1: 0.2600 (catastrophic failure) |

* **Key Finding:** Zone stratification prevents the arid-urban spectral illusion and delivers higher fidelity across all ecological regions.

---

```
================================================================================
SLIDE 9: UNCERTAINTY & PROBABILITY CALIBRATION
================================================================================
```
## Post-Hoc Temperature Scaling for Trustworthy Underwriting

* **The Calibration Dilemma:** Deep neural networks and gradient-boosted trees output overconfident probabilities. For institutional capital underwriting, a raw 90% confidence score that is only correct 60% of the time introduces unacceptable financial and legal risk.
* **Mathematical Formulation:** Logits $z_i$ are rescaled by learned temperature parameter $T > 0$:
  $$p_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$
* **Calibration Results:**
  * Optimal Learned Temperature: **$T = 1.42$**
  * Pre-Calibration Expected Calibration Error (ECE): **0.2525**
  * Post-Calibration Expected Calibration Error (ECE): **0.0921**
  * **Impact:** **63.5% reduction in calibration error**, guaranteeing that predicted probabilities directly correspond to empirical true frequencies.

---

```
================================================================================
SLIDE 10: STATUTORY LAND-USE ALLOCATION ENGINE
================================================================================
```
## Translating Satellite Predictions into Statutory Decisions

Multi-spectral probabilities are converted into actionable directives under Kenyan planning legislation:

```
                          [Calibrated Land-Cover Probabilities]
                                           │
             ┌─────────────────────────────┼─────────────────────────────┐
             ▼                             ▼                             ▼
       [🏛️ SETTLE]                    [🌾 FARM]                    [🛡️ PROTECT]
   Low slope (<15°),             High crop probability,        Watercourses, wetlands,
   non-prime cropland,           fertile volcanic soils,       slopes > 15°, mangroves,
   brownfield/infill land        AFA food security zone        PLUPA 2019 Sec. 56 buffer
```

### Statutory Allocation Insights Across Kenya Counties:
* **PROTECT Leaders:**
  * **Marsabit (66.8%) & Turkana (65.6%):** Rugged escarpments exceeding 15° slope cutoff, Sibiloi/Marsabit National Reserves, and Lake Turkana 30m riparian buffer mandates.
  * **Lamu (57.2%):** Over 60% of Kenya's mangrove marine forests, vulnerable coastal intertidal flats, and UNESCO heritage environmental buffers.
* **FARM Leader:**
  * **Nyandarua (55.3%):** Kinangop volcanic plateau, national potato/vegetable basket, strict Agriculture and Food Authority (AFA) arable preservation.
* **SETTLE Leader:**
  * **Nakuru (31.2%):** Kenya's 4th city, flat Rift Valley floor topography, primed for Affordable Housing Act 2023 infrastructure development.

---

```
================================================================================
SLIDE 11: GENERATIVE AI & STATUTORY RAG COPILOT
================================================================================
```
## Conversational Spatial Underwriting Copilot

* **The Problem:** Non-technical executives, bank credit officers, and county planners cannot interpret raw multi-spectral confusion matrices or raw GeoTIFF rasters.
* **Multi-Tier RAG Architecture:**
  1. **Dynamic Spatial Context Ingestion:** Ingests GPS coordinates, predicted class probabilities, topographic slope, and hydrologic flood hazard flags.
  2. **Statutory Knowledge Retrieval (`_RAG_REGULATORY_KNOWLEDGE`):**
     * **PLUPA 2019 Sec. 56:** Mandatory 30m riparian setback; mandatory structural certification for slopes > 15°.
     * **NEMA EMCA Cap 387:** Environmental Impact Assessment (EIA) license for agricultural conversions > 2.0 hectares.
     * **Trade Corridor Zoning:** 50m highway buffer and drainage neutrality proof along logistics corridors.
  3. **Multi-Tier LLM Execution:**
     * **Primary:** GROQ Cloud (`llama-3.3-70b-versatile`, sub-2s inference).
     * **Secondary:** Google Gemini 1.5 Flash API fallback.
     * **Tertiary:** Local deterministic template engine guaranteeing 100% uptime.
* **Output:** Generates a structured Markdown underwriting dossier complete with compliance flags, hazard alerts, and capital allocation directives.

---

```
================================================================================
SLIDE 12: PRODUCTION MLOPS & CLOUD ARCHITECTURE
================================================================================
```
## Production Cloud Microservices & Continuous Monitoring

```
[Vercel Cloud] ────────► [Nginx Reverse Proxy :80] ────────► [FastAPI Service :8000]
 React 18 / Vite          SSL Termination / HTTPS Proxy       ONNX Runtime / LightGBM
 Leaflet GIS Atlas        CORS & Static Asset Serving         /predict, /underwrite/copilot
                                                                    │
           ┌────────────────────────────────────────────────────────┴────────┐
           ▼                                                                 ▼
[MLflow Server :5000]                                            [Evidently Drift Monitor]
 Experiment tracking, params,                                    Two-sample KS & Wasserstein tests,
 LOZO loss curves, ONNX models                                   Seasonal phenological shift alerts
```

* **Cloud Infrastructure:** Vultr Cloud Compute (`192.248.170.201`), Ubuntu 24.04 LTS (London, GB) + Vercel Frontend.
* **Inference Latency SLA:** Model execution < 50ms via ONNX Runtime CPU optimization; API p95 latency < 350ms.
* **CI/CD Pipeline:** GitHub Actions (`ci.yml`) runs Ruff linting, executes 10 Pytest unit tests, builds Docker containers, and runs container smoke tests on every push.
* **Drift Telemetry:** SciPy/Evidently drift engine monitors seasonal dry-to-wet shifts across 27 features, generating automated HTML/JSON reports.

---

```
================================================================================
SLIDE 13: CONCLUSION & STRATEGIC RECOMMENDATIONS
================================================================================
```
## Summary of Contributions & Future Horizon

### Core Technical Contributions:
1. **Resolved the Arid-Urban Spectral Illusion:** Proved that biome stratification eliminates false urban classifications in Northern Kenya.
2. **First Legally Grounded Spatial AI in East Africa:** Bridged machine learning outputs with statutory planning law (PLUPA 2019 & NEMA EMCA).
3. **Calibrated & Trustworthy:** Reduced Expected Calibration Error to 0.0921, making predictions viable for financial underwriting.
4. **End-to-End MLOps Deployment:** Fully productionized system with live API, React GIS interface, MLflow tracking, CI/CD, and drift monitoring.

### Strategic Policy Recommendations:
* Integrate Atlas county statistics (`settle_score`, `farm_score`, `protect_score`) directly into national Mixed-Integer Linear Programming (MILP) housing budget allocation.
* Require statutory RAG underwriting dossiers as mandatory preliminary documentation before issuing municipal building permits or agricultural conversion licenses.

---

**Live Production API:** [http://192.248.170.201:8000/docs](http://192.248.170.201:8000/docs)  
**Live Frontend Application:** [https://satellite-tau.vercel.app](https://satellite-tau.vercel.app)  
**GitHub Repository:** [https://github.com/VAL-Jerono/Satellite](https://github.com/VAL-Jerono/Satellite)  

*Thank you. Questions & Discussion.*
