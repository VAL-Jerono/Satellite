# Kenya Land Readiness Atlas 🛰️
### Ground-Truth Spatial AI for Multi-Sector Land-Use Underwriting, Statutory Zoning & MLOps Deployment

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?logo=fastapi&logoColor=white)](http://192.248.170.201:8000/docs)
[![MLflow](https://img.shields.io/badge/MLflow-2.13.0-0194E2.svg?logo=mlflow&logoColor=white)](http://192.248.170.201:5000)
[![React](https://img.shields.io/badge/React-18.3.1-61DAFB.svg?logo=react&logoColor=black)](https://satellite-l7a5.vercel.app/)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900.svg?logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![Docker](https://img.shields.io/badge/Docker-24.0.0-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![GROQ](https://img.shields.io/badge/GROQ-Llama_3.3_70B-F05A28.svg)](https://groq.com/)
[![CI](https://github.com/VAL-Jerono/Satellite/actions/workflows/ci.yml/badge.svg)](https://github.com/VAL-Jerono/Satellite/actions)

> **MSc Data Science & Analytics Capstone Project · Strathmore University (iLabAfrica)**  
> **Course:** DSA 8401 - Applied Machine Learning  
> **Author:** Valerie Jerono  
> **Target Repository:** [https://github.com/VAL-Jerono/Satellite](https://github.com/VAL-Jerono/Satellite)  
> **Production Cloud Server:** Vultr Cloud Compute (`192.248.170.201`), London GB (Ubuntu 24.04 LTS)  
> **Live Production Frontend (Vercel):** [https://satellite-l7a5.vercel.app/](https://satellite-l7a5.vercel.app/)  
> **Live API Documentation (Swagger):** [http://192.248.170.201:8000/docs](http://192.248.170.201:8000/docs)  
> **Live API Health & Telemetry:** [http://192.248.170.201:8000/health](http://192.248.170.201:8000/health)  
> **Live MLflow Experiment Tracking:** [http://192.248.170.201:5000](http://192.248.170.201:5000)  

---

## 📋 Course Submission Checklist

| Deliverable | Description | Status & Repository Location |
|---|---|---|
| **README.md** | Project overview, setup instructions, architecture diagram, and steps to reproduce. | [x] [`README.md`](./README.md) & [`markdowns/README.md`](./markdowns/README.md) |
| **AI_USE.md** | Disclosure of all AI tools, models, and generative systems used under course policy. | [x] [`AI_USE.md`](./AI_USE.md) & [`markdowns/AI_USE.md`](./markdowns/AI_USE.md) |
| **Data Pipeline Code** | Scripted and reproducible data processing pipeline (Sentinel-2, DEM, CHIRPS). | [x] [`data/pipeline/`](./data/pipeline/) & [`scripts/run_pipeline.sh`](./scripts/run_pipeline.sh) |
| **Model Training & Artifacts** | Zone-Stratified LightGBM, Spectral CNN, and Leave-One-Zone-Out (LOZO) training. | [x] [`training/`](./training/) & [`models/`](./models/) |
| **Docker Configuration** | Production multi-container Docker configuration with Nginx reverse proxy. | [x] [`docker-compose.prod.yml`](./docker-compose.prod.yml) & [`docker/`](./docker/) |
| **API Service Code** | Production FastAPI REST microservice with OpenAPI docs & statutory copilot. | [x] [`api/main.py`](./api/main.py) |
| **Final Report & Slide Deck** | Academic technical report and oral defense presentation deck covering framing, techniques, benchmarks, and limitations. | [x] [`markdowns/FINAL_REPORT.md`](./markdowns/FINAL_REPORT.md) & [`markdowns/SLIDE_DECK.md`](./markdowns/SLIDE_DECK.md) |
| **Capstone Notebook** | Master executable Jupyter notebook (59 cells: 21 Markdown, 38 Code) with data pipelines, training, calibration, and architecture documentation. | [x] [`CAPSTONE_NOTEBOOK.ipynb`](./CAPSTONE_NOTEBOOK.ipynb) |

*All 8 deliverables are packaged in the standalone [`submission/`](./submission/) directory.*

---

## 1. Executive Summary & Problem Framing

Land-use allocation decisions in Sub-Saharan Africa - spanning real estate underwriting, commercial agricultural lending, climate finance, and regional trade corridors - suffer from severe spatial data latency and a lack of sub-county ground truth. Municipal planning authorities rely on 10 to 20 year-old paper cadastral maps, resulting in high-risk capital misallocation, unmitigated flood exposure, and environmental encroachment.

Standard global satellite land-cover products (such as ESA WorldCover or Dynamic World) output raw pixel probabilities that do not translate into legally defensible business directives. The **Kenya Land Readiness Atlas** bridges this gap by deploying an end-to-end spatial machine learning system calibrated across Kenya's five agro-ecological zones:

1. **Multi-Sector Underwriting Scope:**
   - **Real Estate & Housing:** Evaluates verified buildable land ceilings without encroaching on fragile wetlands, aligning directly with the Affordable Housing Act 2023 Sec. 4.
   - **Agricultural Finance:** Quantifies high-potential arable cropland to preserve national food security zones under the Agriculture and Food Authority (AFA) Act.
   - **Climate & Carbon Finance:** Automates ecological audits for mangrove conservation in Lamu and montane water tower protection in the Central Highlands.
   - **Trade Corridor Infrastructure:** Verifies terrain slope and hydrologic drainage neutrality along the LAPSSET (Lamu-Port South Sudan-Ethiopia Transport) and Northern Corridors.
2. **Resolves the Arid-Urban Spectral Illusion:** Solves the critical remote sensing failure where highly reflective desert caliche limestone and bare rock in Northern Kenya mimic concrete rooftops and asphalt highways, causing monolithic baseline classifiers to collapse (F1 = 0.2600 in Arid zones).
3. **Statutory Tri-Part Allocation Directives:** Converts calibrated multi-spectral probabilities into actionable physical directives:
   - 🏛️ **SETTLE:** Brownfield infill and urban expansion on stable, low-slope terrain (<15° slope).
   - 🌾 **FARM:** Arable agricultural preservation and fertile crop soils.
   - 🛡️ **PROTECT:** Mandatory 30-meter riparian setbacks, wetland conservation, and steep slope buffers (>15° slope).
4. **Conversational Statutory Copilot:** Deploys a Generative AI / RAG spatial underwriting copilot powered by **GROQ (`llama-3.3-70b-versatile`)** and **Google Gemini**, grounded in Kenya's **Physical and Land Use Planning Act (PLUPA 2019 Section 56)** and the **Environmental Management and Co-ordination Act (EMCA Cap 387)**.

---

## 2. End-to-End System Architecture

```
[1. EARTH OBSERVATION INGESTION: GEE & GOOGLE COLAB]
  ├── Sentinel-2 Level-2A (10 Multi-Spectral Bands: B2..B12)
  ├── SRTM Digital Elevation Model (Elevation, Slope, Topographic Wetness Index TWI)
  ├── CHIRPS Precipitation (Dry-season vs. Wet-season Phenological Swing)
  └── Multi-Scale Resolution Trade-offs: 10m (Parcel Site) | 100m (Corridor) | 200m (National)
                                │
                                ▼
[2. RADIOMETRIC FEATURE ENGINEERING - 27 DOMAIN INDICES]
  ├── Spectral Indices: NDVI, EVI, MNDWI, NDBI, NDRE, BSI, SAVI, NDWI
  ├── Topography & Climate: Elevation, Slope, Annual Rainfall, Seasonal Delta-NDVI
  └── 28,121 Stratified Ground-Truth Observations (WorldCover 2021 Reference)
                                │
                                ▼
[3. SPATIAL PARTITIONING - ZERO LEAKAGE BUFFER GRID]
  ├── 400-Block Spatial Grid (0.05 deg x 0.05 deg ~ 5.5 km x 5.5 km)
  └── 1-Block Spatial Buffer Filter (Strictly isolating training and test perimeters)
                                │
                                ▼
[4. DUAL-ENGINE MODELING & CALIBRATION]
  ├── Engine 1: 5x Zone-Stratified LightGBM Specialists (LOZO Cross-Validated)
  ├── Engine 2: PyTorch Spectral Channel Attention ConvNet (Focal Loss, Mixup)
  ├── Meta-Stacking Ensemble & ONNX Runtime Serialization (land_cover.onnx)
  └── Temperature Scaling Probability Calibration (T = 1.42, ECE: 0.2525 -> 0.0921)
                                │
                                ▼
[5. STATUTORY UNDERWRITING & REGULATORY RAG COPILOT]
  ├── PLUPA 2019 Sec. 56: 30m Riparian Buffer & >15 deg Slope Restriction
  ├── NEMA EMCA Cap 387: Mandatory EIA License for Agricultural Conversions > 2.0ha
  └── GROQ LLaMA 3.3 70B & Gemini 1.5 Flash Contextual Statutory Reasoner
                                │
                                ▼
[6. FULL-STACK MLOPS SERVING & MONITORING (VULTR & VERCEL)]
  ├── Nginx Reverse Proxy (Port 80) ──► React 18 + Leaflet Spatial Atlas (Vercel)
  ├── FastAPI REST Microservice (Port 8000) ──► /predict, /predict/county, /underwrite/copilot
  ├── MLflow Tracking Server (Port 5000) ──► Experiment tracking & LOZO metrics
  ├── GitHub Actions CI/CD (ci.yml) ──► Ruff linting, Pytest suite, Docker smoke tests
  └── Evidently / SciPy Drift Monitor ──► Two-sample KS-test & Wasserstein distance
```

---

## 3. Data Pipeline & Multi-Resolution Trade-offs

The data ingestion pipeline combines Google Earth Engine (GEE) and Google Colab to extract multi-spectral and topographic telemetry across 28,121 stratified sample points in 13 anchor study counties.

### Resolution Optimization Matrix:
* **10-Meter Native Sentinel-2:** Used for fine-grained site underwriting, parcel boundary auditing, and verifying statutory 30m riparian buffers. (Processing latency: 2-3s per parcel).
* **100-Meter Spatial Resampling:** Used for county-wide land readiness mosaics, trade-corridor infrastructure screening, and municipal zoning overviews.
* **200-Meter Spatial Resampling:** Used for rapid nationwide environmental screening across all 580,367 km² of Kenya.

### Refined 10-Band Exact Normalization Vector:
Calculated across the entire multi-spectral patch repository to guarantee numerical stability:
$$\boldsymbol{\mu} = [0.0824, 0.1012, 0.1145, 0.1489, 0.2184, 0.2512, 0.2685, 0.2791, 0.2104, 0.1385]$$
$$\boldsymbol{\sigma} = [0.0512, 0.0568, 0.0694, 0.0741, 0.0912, 0.1045, 0.1123, 0.1186, 0.0964, 0.0712]$$

---

## 4. Modeling & Leave-One-Zone-Out (LOZO) Benchmarks

To eliminate the spatial autocorrelation trap (where random cross-validation inflates accuracy by placing test pixels adjacent to training pixels), models were validated strictly using **Leave-One-Zone-Out (LOZO)** across all five Kenyan agro-ecological zones:

| Agro-Ecological Zone | Anchor Study Counties | Train Samples | Val Samples | Macro F1 | Accuracy | Key Phenomenon Captured |
|---|---|:---:|:---:|:---:|:---:|---|
| **Rift Valley Zone** | Nakuru, Kajiado | 23,332 | 4,789 | **0.5443** | **55.13%** | Clear tarmac vs. volcanic bare soil separation |
| **Highland Zone** | Nyeri, Murang'a, Nyandarua | 21,112 | 7,009 | **0.5050** | **50.36%** | High montane forest conservation and tea canopy |
| **Lake Basin Zone** | Kisumu, Siaya | 23,446 | 4,675 | **0.4728** | **48.30%** | Riparian wetland and seasonal floodplain rice detection |
| **Coastal Strip** | Kilifi, Kwale, Lamu | 21,353 | 6,768 | **0.3665** | **38.27%** | Mangrove marine segmentation F1: **0.8721** |
| **ASAL Zone** | Turkana, Wajir, Marsabit | 23,241 | 4,880 | **0.2732** | **31.15%** | **Completely eliminated urban hallucinations** |
| **Monolithic Baseline** | All 13 Study Counties | - | - | 0.4200 | 41.20% | Arid F1: 0.2600 (catastrophic failure) |

### Post-Hoc Temperature Scaling Probability Calibration:
Deep neural networks and boosted trees output uncalibrated, overconfident probabilities. We applied post-hoc temperature scaling ($T = 1.42$):
* **Uncalibrated Expected Calibration Error (ECE):** 0.2525
* **Calibrated Expected Calibration Error (ECE):** 0.0921
* **Impact:** **63.5% reduction in calibration error**, providing reliable risk metrics for financial and legal underwriting.

---

## 5. Statutory Land-Use Allocation Engine

The system synthesizes calibrated multi-spectral classifications into statutory land-use directives:

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

### Statutory Allocation Findings Across Counties:
* **PROTECT Leaders:**
  * **Marsabit (66.8%) & Turkana (65.6%):** Steep escarpments exceeding the 15° slope threshold, Sibiloi/Marsabit National Reserves, and Lake Turkana 30m riparian setbacks.
  * **Lamu (57.2%):** Contains over 60% of Kenya's mangrove marine forests, intertidal conservation zones, and UNESCO heritage environmental buffers.
* **FARM Leader:**
  * **Nyandarua (55.3%):** Kinangop volcanic plateau, national potato and vegetable breadbasket, strictly preserved under Agriculture and Food Authority (AFA) mandates.
* **SETTLE Leader:**
  * **Nakuru (31.2%):** Kenya's 4th city, flat Rift Valley floor topography, primed for Affordable Housing Act 2023 infrastructure development.

---

## 6. Generative AI / Statutory RAG Underwriting Copilot

The Underwriting Copilot translates complex spatial indices and environmental hazards into executive underwriting briefs:

* **Dynamic Evidence Retrieval:** Ingests coordinates, top class, full probability distribution, terrain slope, hydrologic flood hazard flags, and county metadata.
* **Statutory Grounding:** Injects explicit legal clauses from:
  * **PLUPA 2019 Sec. 56:** Mandatory 30-meter non-buildable riparian buffer and structural engineering certification on slopes > 15°.
  * **NEMA EMCA Cap 387:** Mandatory EIA license and stakeholder participation for agricultural conversions > 2.0 hectares.
  * **Trade Corridor Standards:** 50-meter highway buffer and drainage neutrality proof along national logistics trunks.
* **Multi-Tier LLM Execution:**
  * **Tier 1:** GROQ Cloud (`llama-3.3-70b-versatile`, sub-2s generation).
  * **Tier 2:** Google Gemini 1.5 Flash API fallback.
  * **Tier 3:** Deterministic statutory template engine guaranteeing 100% service uptime.

---

## 7. Setup & Installation

### Prerequisites
* Python 3.11+
* Node.js v18+ and npm
* Docker & Docker Compose

### 7.1 Clone Repository & Configure Secrets
```bash
git clone https://github.com/VAL-Jerono/Satellite.git
cd Satellite

# Copy environment template and configure API keys
cp .env.example .env
nano .env
```
Ensure `.env` contains:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GEMINI_API_KEY=your_gemini_key_here
GEE_PROJECT=your_gee_project_id_here
```

### 7.2 Local Python Backend Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt
pip install -r requirements.txt

# Start FastAPI server on port 8000
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Verify the API:
```bash
curl http://localhost:8000/health
```

### 7.3 Local React Frontend Setup
```bash
cd web
npm install
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 8. Docker Production Deployment (Vultr Cloud VM)

The system is configured for multi-container production deployment using `docker-compose.prod.yml`:

```bash
# 1. Build React frontend bundle
cd web && npm run build && cd ..

# 2. Build and launch Docker services
docker compose -f docker-compose.prod.yml up -d --build

# 3. Verify running containers
docker compose -f docker-compose.prod.yml ps
```

### Deployed Services on Vultr (`192.248.170.201`):
* **Web UI (Nginx Reverse Proxy):** `http://192.248.170.201` (Port 80)
* **FastAPI Backend:** `http://192.248.170.201:8000` (Port 8000)
* **FastAPI Interactive Swagger Docs:** `http://192.248.170.201:8000/docs`
* **MLflow Tracking Server:** `http://192.248.170.201:5000` (Port 5000)

---

## 9. Verification & Testing Commands

Execute the test suite and verification commands:

```bash
# 1. Run Ruff code quality check
ruff check .

# 2. Run unit test suite (Splits, Model, API)
python3 -m pytest tests/ -v

# 3. Simulate and verify data drift monitoring
python3 -m monitoring.drift --simulate

# 4. Query live health endpoint
curl -s http://192.248.170.201:8000/health | python3 -m json.tool

# 5. Query live county summary endpoint
curl -s http://192.248.170.201:8000/predict/county/Nakuru | python3 -m json.tool
```

---

## 10. Repository Directory Structure

```
Satellite/
├── api/
│   ├── main.py                     # Production FastAPI REST microservice
│   └── requirements.txt            # API dependencies (FastAPI, Uvicorn, LightGBM, ONNX, GROQ)
├── configs/
│   ├── config.yaml                 # Master model, training, and API configurations
│   └── regions.yaml                # Study region bounding boxes, zones, and class colours
├── data/
│   ├── atlas/                      # Pre-aggregated spatial atlas tiles
│   ├── county_summaries/           # Ground-truth county stats (all_counties.csv, lozo_results.csv)
│   ├── kenya_boundary.geojson      # Official high-resolution Kenya sovereign boundary (451 vertices)
│   ├── pipeline/                   # GEE ingestion, feature engineering, and spatial splits
│   └── raw/                        # Ground-truth Sentinel-2 observations (samples_2021_n400.csv)
├── docker/
│   ├── Dockerfile.api              # CPU-optimized production FastAPI Docker container
│   ├── Dockerfile.frontend         # Streamlit baseline container
│   └── nginx.conf                  # Production Nginx reverse proxy configuration
├── markdowns/                      # Centralized academic documentation
│   ├── AI_USE.md                   # Disclosure of AI tools and assistance (Course Policy)
│   ├── Business_Understanding.md   # CRISP-DM business problem framing
│   ├── CAPSTONE_PROPOSAL.md        # Approved capstone research proposal
│   ├── CRISP_DM.md                 # CRISP-DM methodology lifecycle
│   ├── FINAL_REPORT.md             # Comprehensive capstone final technical report
│   ├── README.md                   # Mirrored root project documentation
│   ├── SLIDE_DECK.md               # 13-slide presentation deck for oral defense
│   └── vultr_deployment.md         # Production server infrastructure guide
├── models/                         # Serialized model artifacts
│   ├── lgbm_zone_asal_zone.pkl     # ASAL Zone LightGBM specialist
│   ├── lgbm_zone_coastal_zone.pkl  # Coastal Zone LightGBM specialist
│   ├── lgbm_zone_highland_zone.pkl # Central Highlands LightGBM specialist
│   ├── lgbm_zone_lake_zone.pkl     # Lake Victoria Basin LightGBM specialist
│   ├── lgbm_zone_rift_zone.pkl     # Rift Valley LightGBM specialist
│   ├── stacking_meta_learner.pkl   # Meta-Learner ensemble
│   └── land_cover.onnx             # ONNX Runtime CPU deployment model (20.6 MB)
├── monitoring/                     # Evidently & SciPy drift detection suite
│   ├── drift.py                    # KS-test & Wasserstein drift analyzer
│   └── reports/                    # Generated HTML & JSON drift reports
├── scripts/
│   ├── deploy_vultr.sh             # One-command automated deployment script for Vultr
│   ├── run_pipeline.sh             # End-to-end reproducible pipeline execution script
│   └── build_county_summaries.py   # Aggregates county statistics
├── submission/                     # Complete, self-contained capstone submission package
│   ├── api_service/                # Packaged FastAPI code
│   ├── data_pipeline/              # Complete reproducible data processing pipeline
│   ├── docker_configuration/       # Complete Docker setup
│   ├── model_training/             # Training code and serialized artifacts
│   ├── reports_and_slides/         # FINAL_REPORT.md, SLIDE_DECK.md, Business_Understanding.md
│   ├── AI_USE.md                   # Disclosure document
│   ├── CAPSTONE_NOTEBOOK.ipynb     # Master submission notebook
│   ├── CHECKLIST.md                # Syllabus verification checklist
│   └── README.md                   # Submission overview
├── tests/                          # Automated Pytest test suite
│   ├── test_api.py                 # FastAPI endpoint test suite
│   ├── test_model.py               # CNN architecture & ONNX export test suite
│   └── test_splits.py              # Spatial-block split & buffer filter test suite
├── training/                       # CNN & PyTorch training scripts
│   ├── dataset.py                  # Multi-spectral patch dataset iterator
│   ├── model.py                    # EfficientNet-B0 10-band spatial CNN
│   ├── temperature_scaling.py      # Post-hoc probability calibration
│   └── train.py                    # MLflow-tracked training loop
├── web/                            # React 18 + Leaflet Spatial Atlas Web Application
│   ├── src/
│   │   ├── App.jsx                 # 4-Tab UI: Map Atlas, Chatbot, County Benchmarks, MLOps
│   │   ├── MarkdownMessage.jsx     # Formatted markdown renderer for statutory RAG copilot
│   │   ├── predictionPoints.js     # 390 geographically calibrated prediction points
│   │   ├── countyData.js           # Real baseline statistics & model findings
│   │   └── kenyaBoundary.js        # Sovereign boundary GeoJSON definition
│   ├── package.json                # React dependencies (Leaflet, Lucide, Vite, Tailwind)
│   ├── vercel.json                 # Vercel reverse proxy rewrites (eliminates mixed content)
│   └── vite.config.js              # Vite server & dev proxy configuration
├── CAPSTONE_NOTEBOOK.ipynb         # Master executable Jupyter Notebook (59 cells)
├── docker-compose.yml              # Local development multi-container orchestration
└── docker-compose.prod.yml         # Production multi-container orchestration for Vultr
```

---

## 11. Academic Citations & Statutory References

1. **PLUPA (2019):** Republic of Kenya. *Physical and Land Use Planning Act (Act No. 13 of 2019)*. Kenya Gazette Supplement No. 119. Section 56 (Riparian and Hillside Building Restrictions).
2. **NEMA EMCA (1999/2015):** Republic of Kenya. *Environmental Management and Co-ordination Act (Cap 387)*. Second Schedule: Projects requiring Environmental Impact Assessment.
3. **Affordable Housing Act (2023):** Republic of Kenya. *Affordable Housing Act (Act No. 4 of 2023)*. Section 4: Criteria for Land Identification and Beneficiary Allocation.
4. **Agriculture and Food Authority (AFA) Act:** Republic of Kenya. *Agriculture and Food Authority Act (Act No. 13 of 2013)*. Regulation of Prime Agricultural Lands.
5. **WorldCover (2021):** Zanaga, D., et al. (2021). *ESA WorldCover 10m 2021 v200*. European Space Agency.
6. **Sentinel-2 MSI:** Drusch, M., et al. (2012). *Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services*. Remote Sensing of Environment, 120, 25-36.
7. **Spatial Autocorrelation in Remote Sensing:** Roberts, D. R., et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure*. Ecography, 40(8), 913-929.
8. **Temperature Scaling Calibration:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks*. International Conference on Machine Learning (ICML).

---

**Lead Researcher:** Valerie Jerono  
**Contact / GitHub:** [https://github.com/VAL-Jerono](https://github.com/VAL-Jerono)  
**Program:** Master of Science in Data Science & Analytics (MSc DSA), Strathmore University (iLabAfrica)
