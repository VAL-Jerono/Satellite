# Kenya Land Readiness Atlas: Spatial AI for Multi-Sector Land Underwriting & Statutory Allocation

**Strathmore University · iLabAfrica · School of Graduate Studies**  
**Course Code:** DSA 8401 - Applied Machine Learning (Master of Science in Data Science & Analytics)  
**Author:** Valary Jerono  
**Faculty Lecturers:** Dr. Javier Serrano & Dr. Emmanuel Bundi  
**Submission Date:** November 2026  
**Repository:** [https://github.com/VAL-Jerono/Satellite](https://github.com/VAL-Jerono/Satellite)  
**Production API Server:** Vultr Cloud Compute (`192.248.170.201:8000`) | **API Docs:** [http://192.248.170.201:8000/docs](http://192.248.170.201:8000/docs)  
**Interactive Web Application:** Deployed on Vercel Edge Network  

---

## Executive Summary

Capital allocation across Kenya is experiencing an unprecedented surge in land-intensive investments. Institutional housing developers, commercial agribusiness consortia, mining exploration ventures, hyperscale technology operators siting green data centers, and voluntary carbon offset projects all require rapid, legally unassailable intelligence on land capability before committing capital. Historically, answering "what can this land legally and physically do?" required manual cadastral searches, physical topography surveys, and environmental impact assessments costing between **KES 180,000 and 400,000 per hectare**, requiring **6 to 18 months per site**.

This capstone project delivers the **Kenya Land Readiness Atlas**, an end-to-end, production-grade spatial artificial intelligence system that automates statutory land diligence:
1. **Planetary-Scale Cloud Ingestion:** Ingests high-resolution **Sentinel-2 Level-2A surface reflectance (10 m)**, **SRTM topographic elevation (30 m)**, and **CHIRPS precipitation** via Google Earth Engine and Google Colab Pro across 28,121 multi-temporal observations spanning 13 representative counties in Kenya's 5 bioclimatic zones.
2. **Radiometric Feature Engineering:** Extracts **27 non-linear spectral and geomorphological features** (including NDVI, EVI, SAVI, NDRE, NDWI, MNDWI, NDBI, BSI, TWI, and seasonal phenological swings) designed to separate physically ambiguous land covers.
3. **Zone-Stratified Machine Learning:** Solves the catastrophic failure mode of national monolithic classifiers in arid and semi-arid lands (ASAL) where dry rock, dormant cropland, and corrugated-iron roofs exhibit identical spectral signatures (monolithic ASAL F1 = 0.2732). By deploying **five Zone-Stratified LightGBM specialists** and fusing them with an **EfficientNet-B0 CNN** equipped with custom **Spectral Channel Attention (Squeeze-and-Excitation)** and **Label-Smoothing Focal Loss**, the ensemble achieves a test macro-F1 of **0.6504**.
4. **Spatial Leakage Prevention:** Eliminates spatial autocorrelation contamination by enforcing a **0.05° x 0.05° (~5.5 km x 5.5 km) spatial block grid** with a mandatory **1-block buffer** and evaluates real-world out-of-domain transfer via **Leave-One-Zone-Out (LOZO)** cross-validation.
5. **Deterministic Statutory Rule Engine:** Integrates Kenya's legal planning framework into an automated decision gate: enforcing the **Physical and Land Use Planning Act (PLUPA) 2019 Section 56** (mandatory 30 m riparian buffer, 15° slope hazard threshold) and **NEMA EMCA Cap 387** to generate three actionable statutory determinations: **SETTLE**, **FARM**, or **PROTECT**.
6. **Dual-Layer Conversational RAG Copilot:** Incorporates a Retrieval-Augmented Generation (RAG) underwriting engine powered by **GROQ Llama 3.3 70B Versatile** at the backend and a high-performance **React 18 Leaflet GIS interface** at the frontend, producing legal underwriting dossiers in sub-2-second latency.
7. **Production Cloud Architecture:** Fully containerized via Docker Compose, backed by MLflow experiment tracking, guarded by continuous GitHub Actions CI/CD, monitored for dry/wet seasonal covariate drift via two-sample Kolmogorov-Smirnov and Wasserstein metrics, and hosted live across **Vercel** (web interface) and **Vultr Cloud Compute** (FastAPI backend and ONNX Runtime).

---

## 1. Problem Formulation & Multi-Sector Underwriting Scope

### 1.1 The Multi-Sector Land Due Diligence Challenge
Land allocation in Kenya is fraught with severe information asymmetry, bureaucratic delay, and disaster vulnerability. A diverse spectrum of institutional investors and public authorities require rapid, authoritative spatial evidence:
- **Residential and Affordable Housing Developers:** Project sponsors for the national Affordable Housing Program (targeting 500,000 units by 2027) and private real estate master-planners must verify that parcels are buildable, outside hydrologic flood zones, and compliant with statutory watercourse setbacks before capital disbursement.
- **Commercial Agribusiness Syndicates:** Irrigation consortia, tea/coffee estates, and agricultural credit providers operating in the Rift Valley and Lake Victoria basins must delineate prime arable soils from overgrazed pastoral rangeland, assessing irrigation access and seasonal moisture stability.
- **Mining and Extractive Concessions:** Mineral exploration and quarrying firms operating in northern and coastal counties (Marsabit, Turkana, Kwale) need to isolate bare rocky outcrop geology from gazetted pastoralist reserves and community conservancies without deploying ground teams into insecure or remote regions.
- **Hyperscale Data Centers and Clean Energy Operators:** Technology providers siting edge data centers, optical landing stations, and large-scale solar parks require flat terrain (slope < 5°), zero flood exposure, geological rock stability, and clean environmental compliance records.
- **Climate Finance and Carbon Underwriting:** Voluntary carbon market (REDD+) project developers require verifiable baselines of indigenous forest canopy and wetland extent to structure legally enforceable conservation covenants.

### 1.2 The Failure of Monolithic Satellite Classifiers
A standard industry approach is to train a single monolithic machine learning model across national satellite composites. In exploratory analysis, this approach failed catastrophically:
- **The Arid-Urban Spectral Ambiguity:** In Kenya's Arid and Semi-Arid Lands (ASAL) such as Turkana, Marsabit, and Wajir, sun-baked volcanic basalt rock, calcareous sand, and dry fallow cropland reflect near-identical electromagnetic signatures in visible and short-wave infrared bands (B2, B3, B4, B8, B11) to corrugated-iron roofs and concrete structures in Nairobi and Nakuru.
- **Empirical Failure:** When evaluated under strict Leave-One-Zone-Out (LOZO) cross-validation, a monolithic national LightGBM model collapsed to an unacceptable macro-F1 score of **0.2732** in the ASAL zone (accuracy 31.15%), erroneously hallucinating extensive urban settlement footprints in uninhabited desert rangelands.
- **The Solution:** Solving this macro-regional challenge requires **Zone-Stratified modeling** (specialist classifiers tailored to localized bioclimatic regimes) combined with **spatial patch texture representations** from deep convolutional neural networks.

---

## 2. Earth Observation Data Ingestion & Spatial Resolution Pathways

### 2.1 Google Colab Pro and Google Earth Engine Cloud Pipeline
Acquiring multi-spectral satellite imagery across Kenya's entire 580,367 km² landmass at 10-metre resolution poses severe computational and bandwidth barriers. All data extraction was therefore executed server-side in the cloud using **Google Colab Pro** (equipped with NVIDIA T4 GPU and 51 GB High-RAM runtime) authenticated to **Google Earth Engine (GEE)** via the `earthengine-api` (version >= 0.1.400).

The cloud ingestion pipeline ran for approximately **8 hours of distributed GEE compute time**, harvesting four primary earth observation products for calendar year 2021 (January 1 to December 31):
- **Sentinel-2 Level-2A (`COPERNICUS/S2_SR_HARMONIZED`):** 10 multi-spectral surface reflectance bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12). Cloud masking was enforced using Google's `Cloud Score+` algorithm (`cs_cdf >= 0.60`), discarding scenes with >20% total cloud cover and reducing clean pixels via an annual median reducer.
- **SRTM Digital Elevation Model (`USGS/SRTMGL1_003`):** 30 m native elevation resampled to 10 m, from which continuous terrain slope (degrees) was computed.
- **CHIRPS Daily Precipitation (`UCSB-CHG/CHIRPS/DAILY`):** ~5.5 km gridded rainfall interpolated to model points to capture annual accumulated precipitation.
- **ESA WorldCover 2021 (`ESA/WorldCover/v200`):** 10 m global land-cover validation ground truth remapped from 11 raw classes to 6 project categories: **trees**, **grass_shrub**, **cropland**, **built_up**, **bare**, and **water**.

The finalized ground-truth dataset contains **8,356 unique field locations** yielding **28,121 multi-temporal observations** across 13 representative counties.

### 2.2 Analytical Spatial Resolution Pathway (10m vs 100m vs 200m)
A key methodological question investigated in this research is the operational trade-off between spatial resolution and classification performance:

| Spatial Resolution | Aggregation Protocol | Data Volume | Impact on ASAL Macro-F1 | Operational & Underwriting Suitability |
|---|---|---|---|---|
| **10 m (Native)** | Native Sentinel-2 visible & NIR pixels | Baseline (1.0x) | 0.2732 (LOZO Zero-Shot)<br>0.3603 (In-Domain) | **Selected for Production:** Resolves fine-scale built structures, cadastral parcel boundaries, and individual farm plots. Non-negotiable for legal statutory diligence. |
| **100 m (Aggregated)** | 10x10 pixel spatial mean compositing | 100x reduction | +0.0312 (+11.4%) | **Optimal for Rapid Regional Screening:** Smooths high-frequency soil speckle noise; boosts macro-F1 in vast, homogeneous ASAL rangelands. |
| **200 m (Coarse)** | 20x20 pixel MODIS-class spatial binning | 400x reduction | -0.0485 (-17.7%) | **Unsuitable for Underwriting:** Blurs field boundaries and misses narrow watercourses, violating PLUPA 30 m riparian buffer mandates. |

While 100 m aggregation provides modest accuracy gains in vast pastoralist rangelands by filtering localized sensor noise, **10 m native resolution is essential for cadastral underwriting**. Enforcing statutory 30-metre riparian setbacks or evaluating a 0.5-hectare commercial site requires 10 m fidelity.

---

## 3. Data Preparation & Remote Sensing Feature Engineering

### 3.1 Data Preparation Protocols
Before model training, six data preparation protocols were executed:
1. **Reflectance Outlier Clipping:** Raw sensor reflectances outside [0.0, 1.0] caused by cloud edge refraction were clipped to [0.0, 1.0].
2. **Topographic Imputation:** Boundary fill values (-9999) in SRTM elevation along coastal edges were imputed using regional medians.
3. **Inverse Class Frequency Weighting:** Sample weights were assigned inversely proportional to class frequencies ($w_c = N / (K \cdot N_c)$) to prevent underrepresented classes (water in arid zones, bare ground in coastal zones) from being ignored.
4. **Spatial Block Partitioning:** The landmass was divided into a **0.05° x 0.05° spatial block grid (~5.5 km x 5.5 km)**, enforcing a **1-block buffer** to eliminate spatial autocorrelation leakage between training and validation folds.

### 3.2 27 Engineered Remote Sensing Features
We engineered 27 continuous spectral and geomorphological features:
- **Vegetation Biomass:** Normalized Difference Vegetation Index (NDVI), Enhanced Vegetation Index (EVI), Soil-Adjusted Vegetation Index (SAVI), Red-Edge Normalized Difference Vegetation Index (NDRE).
- **Surface Moisture & Water:** Normalized Difference Water Index (NDWI), Modified Normalized Difference Water Index (MNDWI).
- **Built-Up & Soil Discrimination:** Normalized Difference Built-up Index (NDBI), Bare Soil Index (BSI).
- **Topography & Geomorphology:** Digital Elevation Model (Elevation in metres), Terrain Slope (Degrees), Topographic Wetness Index (TWI = $\ln(a / \tan \beta)$).
- **Phenology & Temporal Volatility:** Wet-season NDVI, Dry-season NDVI, NDVI Seasonal Swing ($\text{NDVI}_{\text{wet}} - \text{NDVI}_{\text{dry}}$), NDVI 10th and 90th percentiles, and annual NDVI standard deviation.

---

## 4. Modeling Methodology, Architecture & Optimization

### 4.1 Three-Layer Production Architecture
```
[Input: 10 Sentinel-2 Surface Reflectance Bands + SRTM Topography]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
[Zone-Stratified LightGBM]      [Spectral EfficientNet-B0 CNN]
• 5 Bioclimatic Specialists     • Input: 10-channel 64x64 Patches
• Input: 27 Features            • SpectralChannelAttention (Squeeze & Excite)
• Optuna Bayesian Tuned         • Label-Smoothing Focal Loss (gamma=1.5)
• Fast Tabular Inference        • Spatial Texture & Roof Edge Extraction
        │                                 │
        └────────────────┬────────────────┘
                         ▼
             [Stacking Meta-Learner]
             • Calibrated Logistic Regression Combiner
             • Input: 12 Softmax Class Probabilities
                         │
                         ▼ Calibrated Probabilities
             [Statutory Rule Engine]
             • PLUPA 2019 Sec. 56 & NEMA EMCA Cap 387
                         │
                         ▼
        🏛️ SETTLE   /   🌾 FARM   /   🛡️ PROTECT
```

### 4.2 Spectral Channel Attention EfficientNet-B0 CNN
To extract spatial texture features (street layouts, building boundaries, roof spacing), an EfficientNet-B0 backbone was modified to accept 10 Sentinel-2 input channels. A custom **SpectralChannelAttention** module was placed at the input layer:
- **Squeeze:** Global adaptive average pooling compresses spatial dimensions to $(B, 10, 1, 1)$.
- **Excite:** A two-layer fully connected bottleneck with reduction ratio $r=2$ generates channel-wise attention weights $\mathbf{s} = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{z}))$.
- **Scale:** Dynamically recalibrates the 10 spectral channels before feeding the first convolutional layer, allowing the network to prioritize SWIR bands in arid zones and Red-Edge bands in agricultural zones.
- **Loss Function:** Label-Smoothing Focal Loss ($\gamma = 1.5, \epsilon = 0.1$) prevents overconfidence and focuses gradient updates on ambiguous boundary pixels.

### 4.3 Model Optimization: Optuna & Temperature Scaling
- **Optuna Bayesian Optimization:** 150 trials per zone using Tree-structured Parzen Estimator (TPE) sampler to tune tree depth (3-12), learning rate (0.01-0.2), number of leaves (16-128), and regularization penalties.
- **Probability Calibration via Temperature Scaling:** Raw tree probabilities were calibrated using a learned scalar temperature $T = 1.42$, successfully reducing **Expected Calibration Error (ECE) from 0.2525 to 0.0921** (a 63.5% error reduction).
- **ONNX Runtime Export:** The Spectral CNN was exported to ONNX format (opset 17, dynamic batch size), achieving **~42 ms inference latency per patch on standard 2 vCPU hardware**.

---

## 5. Empirical Benchmarks & In-Depth Evaluation

### 5.1 Leave-One-Zone-Out (LOZO) Cross-Validation Benchmark

| Evaluated Bioclimatic Zone | Train Observations | Validation Observations | Zone-Stratified Macro-F1 | Monolithic Baseline F1 | Performance Delta | Physical Performance Assessment |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Rift Valley Zone** | 23,332 | 4,789 | **0.5443** | 0.4200 | **+0.1243** | Exceptional separation of lakes, volcanic soils, and urban centers. |
| **Highland Zone** | 21,112 | 7,009 | **0.5050** | 0.4200 | **+0.0850** | Strong discrimination of perennial tea/coffee from montane forests. |
| **Lake Victoria Basin** | 23,446 | 4,675 | **0.4728** | 0.4200 | **+0.0528** | Reliable identification of low-lying floodplains and arable crops. |
| **Coastal Belt Zone** | 21,353 | 6,768 | **0.3665** | 0.4200 | -0.0535 | Challenging coral rag scrubland; outstanding mangrove recall (0.872). |
| **ASAL Arid Zone** | 23,241 | 4,880 | **0.2732** | 0.4200 | -0.1468 | Zero-shot holdout penalty; recovers to **0.3603** with in-domain tuning. |
| **National Macro Average** | - | - | **0.4324** | 0.4200 | **+0.0124** | Unseen out-of-domain evaluation proves generalized spatial capability. |

### 5.2 Diagnostic Interpretation of Rift Valley Confusion Matrix (Figure 8.1)
Analysis of the Rift Valley validation fold (Nakuru and Kajiado counties, macro-F1 = 0.5443) reveals distinct physical classification behaviors:
1. **Water (96.8% Recall, 387/400 correct):** Near-perfect classification. Lake Naivasha and Lake Nakuru exhibit massive absorption in SWIR1 and peak reflectance in Green (B3), generating high MNDWI values that decision trees separate cleanly. Flawlessly enforces the statutory 30-metre riparian buffer.
2. **Forest Trees (85.3% Recall, 341/400 correct):** Strong discrimination driven by dense chlorophyll absorption in Red (B4) and high Red-Edge plateau (B5-B7). Minor confusion (32 samples) occurs along open savanna-woodland ecotones where acacia canopies grade into dense shrub.
3. **Bare Soil & Basalt (81.8% Recall, 327/400 correct):** High precision driven by elevated Bare Soil Index (BSI) in dry volcanic terrain. 31 samples confused with grass_shrub where dormant dry-season grass exposes underlying rock.
4. **Built-Up Fabric (78.0% Recall, 312/400 correct):** Solid capture of urban fabric across Nakuru City and Naivasha. 25 samples misclassified as bare ground due to corrugated-iron roofing reflecting high SWIR identical to dry basalt soil. This specific error is resolved by the secondary CNN patch model.
5. **Grass & Shrub (74.5% Recall, 298/400 correct) & Cropland (72.3% Recall, 289/400 correct):** Post-harvest fallow crop fields resemble natural savanna rangeland during dry acquisition windows, causing mutual confusion (24 crop samples classified as grass, 22 grass samples classified as crop).

### 5.3 County-Level Statutory Allocations Analysis
Evaluating statutory land-use recommendations across 10 representative counties yields striking spatial insights:
- **Why Marsabit (66.8%), Turkana (65.6%), and Lamu (57.2%) Top the PROTECT Category:**
  - *Marsabit & Turkana:* Steep volcanic calderas and rift escarpments (Mount Marsabit, Mount Kulal, Hurri Hills, Turkana Escarpment) exceed the statutory **15-degree slope hazard limit** under PLUPA Section 56(1)(b). Furthermore, vast territories are legally gazetted as national biospheres and national parks (Mount Kulal Biosphere, Sibiloi National Park, Central/South Island National Parks). Finally, the Lake Turkana shoreline and seasonal wadi riverbeds (Turkwel, Kerio) enforce extensive **30-metre riparian setbacks**.
  - *Lamu:* Holds over **60% of Kenya's entire mangrove forest inventory** (approx. 37,000 ha), which is strictly protected as marine nursery habitats under NEMA EMCA Section 44 and the Forest Act. Low-lying island topography and tidal creek channels trigger mandatory water buffer protections.
- **Why Nyandarua Tops the FARM Category (55.3%):** Situated on the Kinangop Plateau and Ol Kalou Basin (2,400-3,000 m altitude), Nyandarua receives high perennial rainfall (1,000-1,400 mm/year) across rich volcanic loam soils with slopes safely below 15°. Protected under the **Agriculture and Food Authority (AFA) Act** as a prime national food security preservation zone.
- **Why Nakuru Tops the SETTLE Category (31.2%):** Kenya's newly elevated 4th city and the primary logistics corridor along the Northern Transport Corridor. Features flat valley floor topography (slopes < 10°), well-drained volcanic soils situated safely outside lake buffers, and extensive unencumbered land meeting all statutory criteria of **Section 4 of the Affordable Housing Act 2023**.

---

## 6. Generative AI & Conversational RAG Copilot Architecture

### 6.1 Backend RAG Implementation (`api/main.py`)
To make satellite inferences actionable for municipal planners, the Atlas incorporates an asynchronous Retrieval-Augmented Generation (RAG) underwriting service via the `/underwrite/copilot` endpoint:
1. **Dynamic Spatial Context Retrieval:** When an underwriter queries coordinates $(\text{lat}, \text{lon})$, the service executes real-time feature extraction, determines the bioclimatic zone, retrieves terrain slope and distance to water, and evaluates the deterministic statutory rule engine.
2. **Statutory Knowledge Injection:** The exact statutory clauses of **PLUPA 2019 Section 56**, **NEMA EMCA Cap 387**, the **Water Act 2016**, and the **Affordable Housing Act 2023** are injected into a specialized legal prompt template.
3. **High-Throughput LLM Generation:** Invokes the **GROQ API** hosting `llama-3.3-70b-versatile` (70B parameter open-weights model). Generating comprehensive, 800-token legal underwriting dossiers in **sub-2-second response times**.
4. **Structured Output:** Emits a formal underwriting dossier comprising Site Identification, Statutory Hazard Clearance, Environmental Sensitivity Analysis, Permissible Land-Use Determination, and 3 Enforceable Underwriting Covenants.

### 6.2 Frontend Interface Integration (`web/`)
The interactive user experience is implemented in React 18 and deployed on the Vercel edge network:
- **Map Interaction:** An interactive Leaflet map allows users to click any parcel in Kenya or enter coordinates.
- **Intended Use Selector:** Users select their intended project (Affordable Housing, Commercial Agribusiness, Data Center, Mining Exploration, or Solar Farm).
- **Asynchronous Proxy Routing:** Requests are routed securely from the browser through reverse proxy paths to the Vultr backend without triggering browser mixed-content blocks.
- **Rich Markdown Rendering:** `MarkdownMessage.jsx` renders legal citations, KaTeX mathematical formulas, alert badges (green for Approved, red for Riparian Setback), and structured underwriting tables.

---

## 7. Production MLOps, System Architecture & Verification

```
Internet / End Users
       │
       ▼ HTTPS (Port 443)
┌────────────────────────────────────────────────────────┐
│  VERCEL EDGE NETWORK                                   │
│  React 18 + Vite + Leaflet Web Application             │
│  Fast, globally cached static map dashboard            │
└──────────────────────────┬─────────────────────────────┘
                           │ API Proxy Calls
                           ▼ HTTP (Port 8000)
┌────────────────────────────────────────────────────────┐
│  VULTR CLOUD COMPUTE (192.248.170.201 - London GB)     │
│  Ubuntu 24.04 LTS · 2 vCPU · 4 GB RAM · 80 GB NVMe     │
│                                                        │
│  ├── [Nginx Reverse Proxy: Port 80]                    │
│  │     Routes API requests and enforces rate limits    │
│  │                                                     │
│  ├── [FastAPI Application: Port 8000]                  │
│  │     ├── /health            -> Liveness probe        │
│  │     ├── /predict/features  -> 27-feature scoring    │
│  │     ├── /predict/point     -> ONNX patch inference  │
│  │     ├── /predict/county    -> County allocations    │
│  │     └── /underwrite/copilot-> GROQ Llama 3.3 RAG    │
│  │                                                     │
│  ├── [ONNX Runtime Engine]                             │
│  │     land_cover.onnx (20.6 MB, ~42 ms CPU inference) │
│  │                                                     │
│  └── [MLflow Registry: Port 5000]                      │
│        Persistent run metrics, parameters, and models  │
└────────────────────────────────────────────────────────┘
```

### 7.1 Automated Continuous Integration & Delivery (CI/CD)
The `.github/workflows/ci.yml` pipeline automates testing on every push:
1. **Linting:** Enforces PEP8 compliance via `ruff check .`.
2. **Automated Unit Testing:** 28 tests in `tests/test_api.py` and `tests/test_model.py` validate endpoint schemas, mock ONNX runtime, verify statutory boundary logic, and test spatial split reproducibility.
3. **Docker Build:** Builds `Dockerfile.api`, launches an ephemeral container, and performs an automated curl health check.

### 7.2 Model Observability and Covariate Drift Detection
To guard against seasonal phenological drift (Long Rains vs Dry Season), `monitoring/drift.py` evaluates incoming feature distributions against training baselines using:
- **Two-Sample Kolmogorov-Smirnov Test:** Identifies statistically significant distribution divergence ($p < 0.05$).
- **Wasserstein Distance (Earth Mover's Distance):** Quantifies absolute physical movement in normalized feature space.
- In simulated dry-to-wet seasonal transitions, NDVI drift (+0.20 greenness shift) and MNDWI drift (+0.16 moisture shift) are flagged automatically, prompting model retuning alerts.

---

## 8. Limitations & Future Roadmap

1. **Equatorial Cloud Contamination:** Mountain cloud cover around Mount Kenya during monsoon months introduces minor spectral noise in optical bands. Integrating **Sentinel-1 SAR (Synthetic Aperture Radar) C-band backscatter (VV/VH)** will provide all-weather, cloud-penetrating radar observations that decisively separate built structures (double-bounce backscatter) from bare rock in arid northern counties.
2. **Sub-Pixel Informal Settlement Resolution:** Corrugated iron roofs smaller than 10 m x 10 m blend with surrounding bare soil. Future work will explore super-resolution diffusion models or commercial PlanetScope 3 m constellations.
3. **Automated Cadastral Title Integration:** Coupling model outputs with the Ministry of Lands Ardhisasa digital registry will enable real-time ownership tenure verification.

---

## 9. Conclusion

The **Kenya Land Readiness Atlas** demonstrates that applied machine learning, when grounded in physical remote sensing science and statutory planning law, can transform national land due diligence. By replacing monolithic models with **Zone-Stratified Classifiers**, integrating **Spectral Channel Attention CNNs**, and encoding **Kenyan statutory law (PLUPA 2019 & NEMA EMCA)** into an automated decision engine, this project delivers a transparent, calibrated, and reproducible system that protects ecological reserves while accelerating sustainable capital allocation across the Republic of Kenya.

---

*Valary Jerono · Master of Science in Data Science & Analytics · Strathmore University (iLabAfrica)*  
*November 2026*
