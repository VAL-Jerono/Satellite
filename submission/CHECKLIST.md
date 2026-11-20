# Strathmore University DSA 8401 Capstone Submission Checklist

**Student Author:** Valerie Jerono  
**Institution:** Strathmore University, iLabAfrica  
**Course:** DSA 8401 Applied Machine Learning (Capstone)  
**Project:** Kenya Land Readiness Atlas: Spatial AI for Multi-Sector Land Underwriting  
**Submission Package:** Complete and Verified  

---

## Submission Checklist Verification

- [x] **README.md**
  - **Location:** `submission/README.md`
  - **Contents:** Project overview, end-to-end setup instructions, full multi-tier architecture diagram, reproduction guide, and live deployment links.
  - **Live URL:** Vultr Cloud Compute (`http://192.248.170.201`) | Vercel Web Dashboard

- [x] **AI_USE.md**
  - **Location:** `submission/AI_USE.md`
  - **Contents:** Explicit disclosure of all generative and agentic AI tools utilized during the capstone, detailing their roles in data extraction, model scaffolding, debugging, and document styling in full compliance with course ethical guidelines.

- [x] **Data Pipeline Code**
  - **Location:** `submission/data_pipeline/`
  - **Contents:** Comprehensive, scripted, and reproducible end-to-end data processing suite:
    - `run_pipeline.sh` & `run_pipeline.py`: Master automated pipeline orchestrators.
    - `ee_pull.py` & `enrich_features_ee.py`: Cloud-based Google Earth Engine extraction and enrichment scripts.
    - `feature_engineering.py`: Full remote sensing module computing all 27 spectral and topographic indices.
    - `spatial_splits.py`: Spatial block cross-validation (0.05° grid), 1-block spatial buffer, and LOZO split generator.
    - `patch_export.py`: 10-band Sentinel-2 spatial patch extractor (`(10, 64, 64)` arrays).
    - `ledger.py`: Multi-county ingestion state ledger and checkpoint manager.
    - `build_county_summaries.py`: Automated county-level statutory land allocation aggregator.
    - `regions.yaml`: Declarative bioclimatic zone mappings.
    - `README.md`: In-depth reproduction guide, architecture diagram, and execution commands.

- [x] **Model Training Code and Artifacts**
  - **Location:** `submission/model_training/`
  - **Contents:** 
    - `code/`: `train.py` (LightGBM training loop with spatial block cross-validation and MLflow logging), `model.py` (SpectralChannelAttention EfficientNet-B0 CNN), `dataset.py` (PyTorch 10-band spatial patch dataset loader), and `loss.py` (Focal Loss).
    - `artifacts/`: 5 serialized Zone-Stratified LightGBM models (`lgbm_zone_*.pkl`), exported ONNX graph (`land_cover.onnx`), PyTorch checkpoint (`best_spectral_cnn.pt`), and calibrated ensemble meta-learner (`stacking_meta_learner.pkl`).
    - `configs/`: Pinned hyperparameters in `lgbm_tuned_params.yaml` and model configuration in `config.yaml`.

- [x] **Docker Configuration**
  - **Location:** `submission/docker_configuration/`
  - **Contents:** `Dockerfile.api` (FastAPI + ONNX Runtime Python 3.11 image), `Dockerfile.frontend` (Streamlit map container), `nginx.conf` (reverse proxy configuration), `docker-compose.yml` (local multi-container stack), `docker-compose.prod.yml` (production cloud stack), and `.dockerignore`.

- [x] **API Service Code**
  - **Location:** `submission/api_service/`
  - **Contents:** Complete FastAPI production application (`main.py`, `dependencies.py`, `schemas.py`) exposing 5 endpoints: `/health`, `/predict/features`, `/predict/point`, `/predict/county/{name}`, and `/underwrite/copilot` (backed by GROQ Llama 3.3 70B statutory RAG).
  - **Statutory Rules:** Automated encoding of Physical and Land Use Planning Act 2019 Section 56 (30 m riparian buffer, 15 deg slope restriction) and NEMA EMCA Cap 387.

- [x] **Final Report or Slide Deck**
  - **Location:** `submission/reports_and_slides/`
  - **Contents:** 
    - `FINAL_REPORT.md`: Comprehensive capstone technical report covering multi-sector problem framing, remote sensing science justification, Leave-One-Zone-Out (LOZO) empirical benchmarks, probability calibration, and policy inferences.
    - `SLIDE_DECK.md`: Formal oral defense slide deck with slide-by-slide speaker notes.
    - `Business_Understanding.md`: In-depth CRISP-DM Phase 1 business understanding document.

- [x] **Capstone Notebook**
  - **Location:** `submission/CAPSTONE_NOTEBOOK.ipynb`
  - **Contents:** Submission-ready Jupyter notebook containing all 15 required syllabus sections, rich narration in every markdown cell, Figure 1.1 with verified Kenya sovereign boundary GeoJSON, detailed interpretation of Confusion Matrix Figure 8.1, zero em-dashes, and a comprehensive repository architecture guide as its final cell.

---

*Verified and Compiled: November 2026*  
*Candidate: Valerie Jerono (Strathmore University)*  
