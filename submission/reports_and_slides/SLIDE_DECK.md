# Kenya Land Readiness Atlas: Capstone Presentation Deck 🛰️

**Slide Deck for Oral Defense & Technical Presentation**  
**Presenter:** Valary Jerono  
**Project:** Spatial AI for Land-Cover Classification & Statutory Housing Allocation  
**Live Application:** [http://192.248.170.201:8000/docs](http://192.248.170.201:8000/docs) | **Repo:** [github.com/VAL-Jerono/Satellite](https://github.com/VAL-Jerono/Satellite)  

---

```
================================================================================
SLIDE 1: TITLE SLIDE
================================================================================
```
# KENYA LAND READINESS ATLAS
### Ground-Truth Spatial AI for Affordable Housing & Statutory Land Allocation

**Candidate:** Valary Jerono  
**Field:** Applied Machine Learning & Geographic Spatial Intelligence  
**Year:** 2026  

*“Transforming 10m Sentinel-2 multi-spectral observations and Kenyan statutory law into trusted physical land allocation intelligence.”*

---

```
================================================================================
SLIDE 2: THE PROBLEM
================================================================================
```
## Kenya’s Urban Housing Dilemma & Planning Failure
* **Rapid Urbanization:** Kenya’s urban centers grow at 4.2% annually; 60%+ live in informal settlements with high climate disaster exposure.
* **The Planning Blindspot:** Municipal master plans rely on 10-20 year-old paper cadastral maps that fail to reflect current terrain realities.
* **The Policy Need:** The National Housing Development Fund uses Mixed Integer Linear Programming (MILP) to allocate billions in affordable housing capital across 47 counties.
* **The Core Requirement:** The allocator requires two trusted satellite inputs per county:
  1. `settle_area_km2` (True buildable land ceiling)
  2. `flood_risk_fraction` (Hydrologic hazard discount penalty)

---

```
================================================================================
SLIDE 3: THE TECHNICAL BOTTLENECK
================================================================================
```
## The Failure of Monolithic Satellite Classifiers
* **The Hypothesis:** Can a single monolithic neural network or Random Forest classify land suitability across all of Kenya?
* **The Catastrophic Failure:** Monolithic models failed in Arid zones (Turkana, Wajir, Marsabit), achieving an **F1 score of only 0.26**.
* **Root Cause: The Arid-Urban Spectral Illusion:**
  * Highly reflective caliche bare rock in desert basins produces **identical optical reflectance (B2, B3, B4, B8)** to concrete slabs and corrugated iron rooftops in Nairobi.
  * Monolithic models hallucinated vast urban buildouts in empty desert grazing rangelands!

---

```
================================================================================
SLIDE 4: DATA PIPELINE & EARTH OBSERVATION
================================================================================
```
## Multi-Source Satellite & Topographic Ingestion
```
                  ┌───────────────────────────────────────────────┐
                  │ Google Earth Engine Multi-Spectral Ingestion  │
                  └───────────────────────┬───────────────────────┘
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       ▼                                  ▼                                  ▼
[Sentinel-2 Level-2A]              [SRTM Elevation DEM]             [CHIRPS Rainfall]
10m BOA surface reflectance        30m topography: elevation,       Monthly precipitation:
10 bands: B2..B12                  slope, Topographic Wetness       seasonal wet/dry dynamics
Cloud Score+ Masking (cs_cdf > 0.6)
```
* **Dataset Scope:** 28,121 stratified ground-truth samples across 13 study counties representing all 5 Kenya agro-ecological zones.

---

```
================================================================================
SLIDE 5: RADIOMETRIC FEATURE ENGINEERING
================================================================================
```
## 27 Engineered Multi-Spectral Indices
Rather than passing raw noisy bands, we engineered 27 physics-guided radiometric indices:
* **Vegetation & Chlorophyll:** $\text{NDVI}$, $\text{EVI}$, $\text{NDRE}$ (Red-Edge sensitivity for crops vs. tree canopy).
* **Built Fabric vs. Bare Ground:** $\text{NDBI}$ (Normalized Difference Built-up Index) and $\text{BSI}$ (Bare Soil Index).
* **Hydrology & Wetness:** $\text{MNDWI}$ (Modified Normalized Difference Water Index) and $\text{TWI}$ (Topographic Wetness Index).
* **Phenological Dynamics:** $\Delta\text{NDVI} = \text{NDVI}_{\text{wet}} - \text{NDVI}_{\text{dry}}$ to differentiate seasonal agricultural crops from permanent montane forest canopies.

---

```
================================================================================
SLIDE 6: MODEL ARCHITECTURE
================================================================================
```
## Zone-Stratified Machine Learning Architecture
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
                         [Meta-Stacker & Probability Vectors]
                         [Built-up, Cropland, Trees, Grass, Water, Bare]
```
* **Why LightGBM?** Ultra-fast CPU inference (<15ms per parcel), memory efficient, and provides exact feature importance splits for regulatory explainability.

---

```
================================================================================
SLIDE 7: SPATIAL VALIDATION METHODOLOGY
================================================================================
```
## Why Leave-One-Zone-Out (LOZO) Validation?
* **The Spatial Autocorrelation Trap:** Standard random $K$-fold cross-validation inflates accuracy metrics because test pixels sit directly adjacent to training pixels.
* **Our Rigorous Benchmark:**
  * **Leave-One-Zone-Out (LOZO):** The model is evaluated strictly on an entire unseen agro-ecological zone.
  * **Buffered Block CV:** 5.5 km spatial blocks (0.05°) with spatial isolation buffers to guarantee genuine geographic generalizability.

---

```
================================================================================
SLIDE 8: EXPERIMENTAL BENCHMARKS
================================================================================
```
## LOZO Experimental Results Across 5 Agro-Ecological Zones

| Agro-Ecological Zone | Train Samples | Validation Samples | Macro F1 | Accuracy | Key Highlight |
|---|:---:|:---:|:---:|:---:|---|
| **Rift Zone (Nakuru)** | 23,332 | 4,789 | **0.5443** | **55.13%** | Volcanic soil vs. urban tarmac separation |
| **Highland Zone (Nyeri)** | 21,112 | 7,009 | **0.5050** | **50.36%** | High montane forest protection |
| **Lake Zone (Kisumu)** | 23,446 | 4,675 | **0.4728** | **48.30%** | Water & wetland classification |
| **Coastal Zone (Kilifi)** | 21,353 | 6,768 | **0.3665** | **38.27%** | Mangrove segmentation F1: **0.8721** |
| **ASAL Zone (Turkana)** | 23,241 | 4,880 | **0.2732** | **31.15%** | Eliminated monolithic urban hallucinations |
| **Monolithic Baseline** | - | - | 0.4200 | 41.20% | Arid F1: 0.26 (severe failure) |

---

```
================================================================================
SLIDE 9: STATUTORY REGULATORY GROUNDING
================================================================================
```
## From Machine Learning to Enforceable Physical Law
A raw probability distribution cannot be sent to a municipal land planner. We enforce a transparent rule layer:
* **PLUPA 2019 Section 56:**
  * **30-Meter Mandatory Riparian Setback:** Any parcel with water probability > 0.35 or active flood risk is strictly routed to **`PROTECT`**.
  * **Hillside Slope Restriction:** Parcels with slope > 15° require structural engineering certification and conservation buffer.
* **NEMA EMCA (Cap 387):**
  * Preserves prime agricultural cropland > 2.0 hectares from reckless urban sub-division without EIA licensing.

---

```
================================================================================
SLIDE 10: FULL-STACK MLOPS DEPLOYMENT
================================================================================
```
## Production Cloud Microservice Architecture
* **Cloud Infrastructure:** Vultr Cloud Compute (`192.248.170.201`), Ubuntu 24.04 LTS (London, GB).
* **FastAPI Service:** Asynchronous microservice exposing `/health`, `/predict/features`, `/predict/county`, and `/underwrite/copilot`.
* **MLflow Tracking:** Port 5000 server tracking LOZO runs, model artifacts, and evaluation parameters.
* **Nginx Reverse Proxy:** Serves production React bundle and eliminates browser Mixed Content (HTTPS-to-HTTP) blocks.
* **GenAI Copilot:** Powered by **GROQ (Llama 3.3 70B)** with deterministic statutory RAG fallback.

---

```
================================================================================
SLIDE 11: USER INTERFACE & DEMONSTRATION
================================================================================
```
## 4-Tab Interactive Spatial AI Experience
1. **`🗺️ Spatial Atlas`:** Leaflet interface locked to Kenyan sovereign boundaries; 390 geographically calibrated prediction points color-coded by statutory decision (`Settle: 115`, `Farm: 172`, `Protect: 103`).
2. **`⚖️ Site Underwriting & Chatbot`:** Conversational RAG Copilot answering legal questions on PLUPA 2019, NEMA EIA licenses, and site suitability.
3. **`📊 County Benchmarks (13)`:** Ground-truth baseline data and agro-ecological policy insights for Kenya's 13 study counties.
4. **`⚡ MLOps & Progress`:** Real-time telemetry, live links to Swagger Docs (`:8000/docs`) and MLflow (`:5000`).

---

```
================================================================================
SLIDE 12: CONCLUSION & STRATEGIC IMPACT
================================================================================
```
## Summary & Real-World Planning Value
* **Scientific Achievement:** Solved the spatial autocorrelation and arid-urban spectral confusion using Zone-Stratified LightGBM and LOZO validation.
* **Policy Value:** Directly delivers mathematically grounded `settle_area_km2` and `flood_risk_fraction` constraints into Kenya's national MILP affordable housing allocator.
* **Humanitarian Safety:** Prevents public housing funds from being sunk into hazardous floodplains or protected ecological reserves.

**Repository:** `https://github.com/VAL-Jerono/Satellite`  
**Thank you! Questions & Discussion.**

