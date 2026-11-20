# Kenya Land Readiness Atlas 🛰️
### Ground-Truth Spatial AI for Affordable Housing & Statutory Land-Use Allocation

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?logo=fastapi&logoColor=white)](http://192.248.170.201:8000/docs)
[![MLflow](https://img.shields.io/badge/MLflow-2.13.0-0194E2.svg?logo=mlflow&logoColor=white)](http://192.248.170.201:5000)
[![React](https://img.shields.io/badge/React-18.3.1-61DAFB.svg?logo=react&logoColor=black)](http://192.248.170.201)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900.svg?logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![Docker](https://img.shields.io/badge/Docker-24.0.0-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![GROQ](https://img.shields.io/badge/GROQ-Llama_3.3_70B-F05A28.svg)](https://groq.com/)

> **Capstone Project - Final Submission**  
> **Author:** Valary Jerono  
> **Repository:** [https://github.com/VAL-Jerono/Satellite](https://github.com/VAL-Jerono/Satellite)  
> **Production Cloud Server:** Vultr Cloud Compute (`192.248.170.201`), London GB (Ubuntu 24.04 LTS)  
> **Live API Documentation (Swagger):** [http://192.248.170.201:8000/docs](http://192.248.170.201:8000/docs)  
> **Live API Health & Telemetry:** [http://192.248.170.201:8000/health](http://192.248.170.201:8000/health)  
> **Live MLflow Experiment Tracking:** [http://192.248.170.201:5000](http://192.248.170.201:5000)  

---

## 📋 Capstone Submission Checklist

| Deliverable | Description | Status & Repository Location |
|---|---|---|
| **README.md** | Project overview, setup instructions, architecture diagram, and steps to reproduce. | ✅ [`README.md`](./README.md) |
| **AI_USE.md** | Disclosure of all AI tools, models, and generative systems used under course policy. | ✅ [`AI_USE.md`](./AI_USE.md) |
| **Data Pipeline Code** | Scripted and reproducible data processing pipeline (Sentinel-2, DEM, CHIRPS). | ✅ [`data/pipeline/`](./data/pipeline/) & [`scripts/run_pipeline.sh`](./scripts/run_pipeline.sh) |
| **Model Training & Artifacts** | Zone-Stratified LightGBM, Spectral CNN, and Leave-One-Zone-Out (LOZO) training. | ✅ [`training/`](./training/) & [`models/`](./models/) |
| **Docker Configuration** | Production multi-container Docker configuration with Nginx reverse proxy. | ✅ [`docker-compose.prod.yml`](./docker-compose.prod.yml) & [`docker/`](./docker/) |
| **API Service Code** | Production FastAPI REST microservice with OpenAPI docs & statutory copilot. | ✅ [`api/main.py`](./api/main.py) |
| **Final Report & Deck** | Academic technical report and oral defense presentation deck covering framing, techniques, benchmarks, and limitations. | ✅ [`FINAL_REPORT.md`](./FINAL_REPORT.md) & [`SLIDE_DECK.md`](./SLIDE_DECK.md) |

---

## 1. Project Overview

The **Kenya Land Readiness Atlas** is a production-grade spatial artificial intelligence platform designed to eliminate hazardous, blind land allocation under Kenya's Affordable Housing Program (AHP). By processing **10m Sentinel-2 Level-2A multi-spectral satellite imagery**, **SRTM 30m digital elevation models**, and **CHIRPS precipitation telemetry**, the Atlas:

1. **Resolves the Arid-Urban Spectral Confusion:** Solves the classic remote sensing failure where bare desert rock in northern Kenya mimics concrete roofs by deploying **five Zone-Stratified LightGBM classifiers** calibrated across Kenya's agro-ecological zones.
2. **Enforces Kenyan Statutory Planning Law:** Bridges pure computer vision and physical urban planning by grounding multi-spectral probabilities in the **Physical and Land Use Planning Act (PLUPA 2019 Section 56)** and the **Environmental Management and Co-ordination Act (EMCA Cap 387)**.
3. **Outputs Deterministic Statutory Recommendations:**
   * 🏛️ **SETTLE:** Brownfield infill and urban expansion on stable, low-slope terrain.
   * 🌾 **FARM:** Arable agricultural cropland preservation and agro-pastoral rangeland.
   * 🛡️ **PROTECT:** Mandatory 30-meter riparian setbacks, wetland conservation, and steep montane forest buffers (>15° slope).
4. **Interactive Regulatory RAG Chatbot:** Integrates **GROQ (`llama-3.3-70b-versatile`)** and **Google Gemini** with a curated statutory knowledge base, enabling urban planners to query zoning legality in natural language.

---

## 2. System Architecture Diagram

```
[1. EARTH OBSERVATION INGESTION]
  ├── Sentinel-2 Level-2A (10 Multi-Spectral Bands: B2..B12)
  ├── SRTM Digital Elevation Model (Elevation, Slope, Topographic Wetness Index)
  └── CHIRPS Precipitation (Dry-season vs. Wet-season Phenological Swing)
                               │
                               ▼
[2. FEATURE ENGINEERING - 27 RADIOMETRIC & TOPOGRAPHIC INDICES]
  ├── NDVI, EVI, MNDWI, NDBI, NDRE, BSI, TWI, Slope, Elevation
  └── 28,121 Stratified Ground-Truth Observations (WorldCover 2021 Reference)
                               │
                               ▼
[3. MODEL ENSEMBLE - ZONE-STRATIFIED MACHINE LEARNING]
  ├── Climate Zone Spatial Routing: Highland, Rift, Lake, Coast, ASAL
  ├── 5x Specialized LightGBM Classifiers (Trained via Leave-One-Zone-Out LOZO)
  └── Spectral EfficientNet-B0 CNN + Stacking Meta-Learner (.pkl & .onnx)
                               │
                               ▼
[4. REGULATORY DECISION LAYER - STATUTORY RAG ENGINE]
  ├── PLUPA 2019 Sec. 56: 30m Riparian Buffer & >15° Slope Construction Restriction
  ├── NEMA EMCA: Mandatory EIA License on Agricultural Conversion > 2.0ha
  └── GROQ Llama 3.3 70B & Gemini 1.5 Flash Contextual Statutory Reasoner
                               │
                               ▼
[5. PRODUCTION SERVING & MLOPS - VULTR CLOUD (192.248.170.201)]
  ├── Nginx Reverse Proxy (Port 80) ──► React 18 + Leaflet Spatial Atlas
  ├── FastAPI Microservice (Port 8000) ──► /predict/features, /predict/county, /underwrite/copilot
  └── MLflow Tracking Server (Port 5000) ──► Model Registry & LOZO Metrics
```

---

## 3. Performance Benchmarks: Leave-One-Zone-Out (LOZO)

To avoid spatial autocorrelation pitfalls inherent in remote sensing, models were validated strictly using **Leave-One-Zone-Out (LOZO)** across all five agro-ecological zones:

| Agro-Ecological Zone | Study Counties | Train Samples | Val Samples | Macro F1 | Accuracy | Key Highlight |
|---|---|:---:|:---:|:---:|:---:|---|
| **Rift Zone** | Nakuru, Kajiado | 23,332 | 4,789 | **0.5443** | **55.13%** | Clear tarmac vs. volcanic bare soil separation |
| **Highland Zone** | Nyeri, Murang'a, Nyandarua | 21,112 | 7,009 | **0.5050** | **50.36%** | High montane forest conservation |
| **Lake Zone** | Kisumu, Siaya | 23,446 | 4,675 | **0.4728** | **48.30%** | Riparian wetland detection |
| **Coastal Zone** | Kilifi, Kwale, Lamu | 21,353 | 6,768 | **0.3665** | **38.27%** | Mangrove marine segmentation F1: **0.8721** |
| **ASAL Zone** | Turkana, Wajir, Marsabit | 23,241 | 4,880 | **0.2732** | **31.15%** | Eliminated monolithic urban hallucinations |
| **Monolithic Baseline** | All 13 Counties | - | - | 0.4200 | 41.20% | Arid F1: 0.26 (catastrophic failure) |

---

## 4. Setup & Installation

### Prerequisites
* Python 3.11+
* Node.js v18+ and npm
* Docker & Docker Compose (optional for containerized deployment)

### 4.1 Clone Repository & Configure Secrets
```bash
git clone https://github.com/VAL-Jerono/Satellite.git
cd Satellite

# Copy environment template and add your API keys
cp .env.example .env   # or edit .env directly
nano .env
```
Ensure `.env` contains:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GEMINI_API_KEY=your_gemini_key_here
GEE_PROJECT=your_gee_project_id_here
```

### 4.2 Local Python Backend Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt
pip install -r requirements.txt

# Start FastAPI server on port 8000
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Test the API:
```bash
curl http://localhost:8000/health
```

### 4.3 Local React Frontend Setup
```bash
cd web
npm install
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 5. Docker Production Deployment (Vultr Cloud VM)

The system is configured for multi-container production deployment using `docker-compose.prod.yml`:

```bash
# 1. Build React frontend bundle
cd web && npm run build && cd ..

# 2. Build and launch Docker services
docker compose -f docker-compose.prod.yml up -d --build

# 3. Verify running containers
docker compose -f docker-compose.prod.yml ps
```

### Services Deployed on Vultr (`192.248.170.201`):
* **Web UI (Nginx Reverse Proxy):** `http://192.248.170.201` (Port 80)
* **FastAPI Backend:** `http://192.248.170.201:8000` (Port 8000)
* **FastAPI Interactive Swagger Docs:** `http://192.248.170.201:8000/docs`
* **MLflow Tracking Server:** `http://192.248.170.201:5000` (Port 5000)

---

## 6. Steps to Reproduce the Full ML Pipeline

To execute the data processing and training pipeline from scratch:

```bash
# Execute master pipeline runner script
bash scripts/run_pipeline.sh
```

Or step-by-step:

```bash
# Step 1: Feature Engineering & Topographic Radiometric Indices
python3 -m data.feature_engineering

# Step 2: Spatial Block & Leave-One-Zone-Out (LOZO) Splits
python3 -m data.pipeline.spatial_splits

# Step 3: Train 5 Zone-Stratified LightGBM Models
python3 -m data.pipeline.zone_stratified_lgbm

# Step 4: Train Spectral EfficientNet-B0 CNN (PyTorch)
python3 -m training.train --epochs 40 --batch-size 64

# Step 5: Build Pre-Aggregated County Benchmarks
python3 scripts/build_county_summaries.py

# Step 6: Verify Model Artifacts
python3 scripts/verify_models.py
```

---

## 7. Repository Directory Structure

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
├── docs/                           # Architecture specs, CRISP-DM reports, and deployment guides
├── models/                         # Serialized model artifacts
│   ├── lgbm_zone_asal_zone.pkl     # ASAL Zone LightGBM
│   ├── lgbm_zone_coastal_zone.pkl  # Coastal Zone LightGBM
│   ├── lgbm_zone_highland_zone.pkl # Central Highlands LightGBM
│   ├── lgbm_zone_lake_zone.pkl     # Lake Victoria Basin LightGBM
│   ├── lgbm_zone_rift_zone.pkl     # Rift Valley LightGBM
│   ├── stacking_meta_learner.pkl   # Meta-Learner ensemble
│   └── land_cover.onnx             # ONNX Runtime CPU deployment model
├── scripts/
│   ├── deploy_vultr.sh             # One-command automated deployment script for Vultr
│   ├── run_pipeline.sh             # End-to-end reproducible pipeline execution script
│   └── build_county_summaries.py   # Aggregates county statistics
├── training/                       # CNN & PyTorch training scripts
│   ├── dataset.py                  # Multi-spectral patch dataset iterator
│   ├── model.py                    # EfficientNet-B0 10-band spatial CNN
│   └── train.py                    # MLflow-tracked training loop
├── web/                            # React 18 + Leaflet Spatial Atlas Web Application
│   ├── src/
│   │   ├── App.jsx                 # 4-Tab UI: Map Atlas, Chatbot, County Benchmarks, MLOps
│   │   ├── MarkdownMessage.jsx     # Formatted markdown renderer for statutory RAG copilot
│   │   ├── predictionPoints.js     # 390 geographically calibrated prediction points
│   │   ├── countyData.js           # Real baseline statistics & model findings
│   │   ├── kenyaBoundary.js        # Sovereign boundary GeoJSON definition
│   │   └── constants.js            # Class colours, action metadata, and study regions
│   ├── package.json                # React dependencies (Leaflet, Lucide icons, Vite, Tailwind)
│   ├── vercel.json                 # Vercel reverse proxy rewrites (eliminates mixed content)
│   └── vite.config.js              # Vite server & dev proxy configuration
├── AI_USE.md                       # Disclosure of AI tools and assistance (Course Policy)
├── FINAL_REPORT.md                 # Comprehensive capstone final technical report
├── SLIDE_DECK.md                   # 12-slide presentation deck for oral defense
├── docker-compose.yml              # Local development multi-container orchestration
└── docker-compose.prod.yml         # Production multi-container orchestration for Vultr
```

---

## 8. Academic Citations & Legal References

1. **PLUPA (2019):** Republic of Kenya. *Physical and Land Use Planning Act (Act No. 13 of 2019)*. Kenya Gazette Supplement No. 119. Section 56 (Riparian and Hillside Building Restrictions).
2. **NEMA EMCA (1999/2015):** Republic of Kenya. *Environmental Management and Co-ordination Act (Cap 387)*. Second Schedule: Projects requiring Environmental Impact Assessment.
3. **WorldCover (2021):** Zanaga, D., et al. (2021). *ESA WorldCover 10m 2021 v200*. European Space Agency.
4. **Sentinel-2 MSI:** Drusch, M., et al. (2012). *Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services*. Remote Sensing of Environment, 120, 25-36.
5. **Spatial Autocorrelation in Remote Sensing:** Roberts, D. R., et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure*. Ecography, 40(8), 913-929.

---

**Lead Researcher:** Valary Jerono  
**Contact / GitHub:** [https://github.com/VAL-Jerono](https://github.com/VAL-Jerono)
