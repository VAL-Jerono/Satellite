# AI Assistance & Tools Disclosure (AI_USE.md)

**Master of Science in Data Science & Analytics (MSc DSA)**  
**Strathmore University / iLabAfrica**  
**Course:** DSA 8401 - Applied Machine Learning  
**Project Title:** Kenya Land Readiness Atlas: Ground-Truth Spatial AI for Multi-Sector Land-Use Underwriting, Statutory Zoning & MLOps Deployment  
**Candidate / Researcher:** Valerie Jerono  
**Supervisors / Lecturers:** Dr. Javier Serrano & Dr. Emmanuel Bundi  
**Academic Year:** 2026  

---

## 1. Executive Statement of AI Use & Academic Integrity

In accordance with Strathmore University academic integrity regulations and the DSA 8401 Course Policy, this document provides a comprehensive, transparent disclosure of the Artificial Intelligence (AI) tools, Large Language Models (LLMs), and automated assistance utilized throughout the research, engineering, training, validation, documentation, and deployment of the **Kenya Land Readiness Atlas**.

The core intellectual contributions - including the framing of the multi-sector spatial underwriting bottleneck, the discovery of the arid-urban spectral illusion, the formulation of the 400-block spatial grid with 1-block buffer isolation, the design of the Leave-One-Zone-Out (LOZO) benchmark across Kenya's five agro-ecological zones, the statutory integration of PLUPA 2019 and NEMA EMCA legislation, and all analytical conclusions - were conceived, directed, and evaluated by the human author.

AI systems were utilized in two distinct capacities:
1. **Development-Time Assistance:** As cognitive accelerators for boilerplate code generation, refactoring, unit test drafting, Docker container configuration, and formatting.
2. **Runtime Production Component:** As an integrated Generative AI / Retrieval-Augmented Generation (RAG) inference engine powering the conversational Underwriting Copilot on the live platform.

---

## 2. Comprehensive Inventory of AI Systems & Services Used

| System / Model | Provider / Architecture | Mode of Use | Project Role | Verification & Quality Control |
|---|---|:---:|---|---|
| **Antigravity (Gemini & Claude)** | Google DeepMind / Anthropic | Development | Interactive code scaffolding, CI/CD workflow drafting, bash pipeline scripts, Docker debugging | 100% manual review, local compilation, unit test validation (`pytest`), static analysis (`ruff`) |
| **GROQ Cloud (`llama-3.3-70b-versatile`)** | Groq / Meta AI | Runtime Service | Primary LLM engine for the `/underwrite/copilot` endpoint; synthesizes statutory dossiers | Grounded via curated Kenyan planning law context; verified with deterministic fallback |
| **Google Gemini 1.5 Flash** | Google Cloud / Vertex AI | Runtime Service | Secondary fallback LLM for spatial underwriting synthesis | Parameterized prompt templates; fallback to deterministic statutory engine |
| **GitHub Copilot** | GitHub / OpenAI | Development | IDE-level inline autocomplete for standard Python, React, and CSS boilerplate | Manual syntax review, type checking, build validation |
| **Google Earth Engine (GEE) APIs** | Google Earth Engine | Data Ingestion | Automated Cloud Score+ QA masking ($cs\_cdf \ge 0.6$) and multi-spectral median compositing | Visual verification in GEE Code Editor against high-resolution optical basemaps |

---

## 3. Granular Breakdown by Project Lifecycle (CRISP-DM Phases)

### Phase 1: Business & Problem Understanding
* **Human Responsibility:** Formulated the primary research questions; identified the multi-sector underwriting crisis spanning Affordable Housing (Affordable Housing Act 2023 Sec. 4), commercial agricultural preservation (AFA Act), climate carbon credits, and transit corridors (LAPSSET); and conceptualized the tri-part land allocation directives (`SETTLE`, `FARM`, `PROTECT`).
* **AI Assistance:** Assisted in structuring the comparative framing between municipal paper cadastral delays and real-time satellite telemetry, and reviewing formatting for the CRISP-DM methodology document.

### Phase 2: Earth Observation Data Engineering & Extraction
* **Human Responsibility:** Designed the multi-source data ingestion pipeline integrating Sentinel-2 Level-2A (10 bands), SRTM 30m Digital Elevation Model, and CHIRPS rainfall telemetry; established the 10m vs. 100m vs. 200m spatial resolution trade-offs; verified ground-truth class remapping from ESA WorldCover 2021 into six canonical classes.
* **AI Assistance:** Drafted vectorized NumPy and Pandas routines for calculating the 27 radiometric and topographic spectral indices (NDVI, EVI, MNDWI, NDBI, NDRE, BSI, SAVI, NDWI, TWI, and phenological delta-NDVI).
* **Human Verification:** All mathematical formulations were validated against published remote sensing literature (Rouse et al., 1974 for NDVI; McFeeters, 1996 for NDWI; Rikimaru et al., 2002 for BSI; Barnes et al., 2000 for NDRE). Exact per-band normalization vectors ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) were computed directly from the 28,121 stratified samples.

### Phase 3: Spatial Partitioning & Zero-Leakage Architecture
* **Human Responsibility:** Formulated the solution to spatial autocorrelation leakage (Tobler's First Law of Geography); designed the 400-block spatial grid ($0.05^\circ \times 0.05^\circ \approx 5.5\text{ km} \times 5.5\text{ km}$); instituted the 1-block spatial buffer filter that discards adjacent boundary samples; and established the Leave-One-Zone-Out (LOZO) benchmark across the five climate zones.
* **AI Assistance:** Provided helper code for geographic coordinate binning (`bx = lon // 0.05`, `by = lat // 0.05`) and set-based spatial adjacency filtering in `data/pipeline/spatial_splits.py`.
* **Human Verification:** Confirmed mathematically and empirically via automated tests (`tests/test_splits.py`) that zero training blocks overlap with or border validation blocks.

### Phase 4: Model Development, Optimization & Calibration
* **Human Responsibility:** Hypothesized and proved that the monolithic classifier failure in Northern Arid zones (Macro-F1 = 0.2600) was driven by the arid-urban spectral illusion; engineered the five Zone-Stratified LightGBM specialists; designed the PyTorch Spectral Channel Attention ConvNet with Label Smoothing Focal Loss ($\gamma = 1.5, \epsilon = 0.1$) and Mixup ($\alpha = 0.2$); executed post-hoc Temperature Scaling ($T = 1.42$).
* **AI Assistance:** Assisted in generating PyTorch training loop boilerplate, MLflow metric logging hooks, and ONNX Runtime export wrappers (`models/land_cover.onnx`).
* **Human Verification:** Evaluated all models on actual out-of-domain LOZO splits; verified the 63.5% reduction in Expected Calibration Error (ECE from 0.2525 down to 0.0921); confirmed all training metrics directly from generated checkpoints without synthetic inflation.

### Phase 5: Statutory RAG Underwriting Copilot Architecture
* **Human Responsibility:** Curated and verified the statutory planning knowledge base grounded in:
  1. *Kenya Physical and Land Use Planning Act (PLUPA 2019) Sec. 56:* Mandatory 30-meter riparian setback from watercourses and mandatory structural certification for slopes > 15°.
  2. *NEMA Environmental Management and Co-ordination Act (EMCA Cap 387):* Environmental Impact Assessment (EIA) license for agricultural conversions > 2.0 hectares.
  3. *East African Trade Corridor Zoning Standards:* 50-meter highway buffer and drainage neutrality proof along national logistics trunks.
* **AI Assistance:** Synthesized the multi-tier API routing logic in `api/main.py` connecting GROQ Cloud, Gemini 1.5 Flash, and the deterministic fallback generator.
* **Human Verification:** Manually reviewed prompt templates and outputs to eliminate statutory hallucinations; designed the client-side fallback in `web/src/App.jsx` ensuring that the application functions even when external LLM APIs are offline.

### Phase 6: Full-Stack MLOps, CI/CD & Production Deployment
* **Human Responsibility:** Architected the production microservices topology across Vultr Cloud Compute (`192.248.170.201`) and Vercel; configured Nginx reverse proxying to resolve Mixed Content blocks; designed the continuous data drift monitoring strategy using two-sample Kolmogorov-Smirnov and Wasserstein distance tests.
* **AI Assistance:** Drafted Dockerfiles (`Dockerfile.api`, `Dockerfile.frontend`), `docker-compose.prod.yml`, GitHub Actions workflow (`ci.yml`), and Evidently / SciPy drift script (`monitoring/drift.py`).
* **Human Verification:** Deployed and tested all services on live cloud infrastructure; validated that GitHub Actions CI builds and passes all 10 unit tests and smoke tests; verified that drift monitoring accurately flags seasonal shifts via simulated wet-season parquet tests.

---

## 4. Representative Prompts & System Instructions Disclosed

The runtime Generative AI Copilot relies on strictly constrained, grounded prompt templates. The exact system prompt embedded in `api/main.py` is disclosed below:

```text
You are an expert GIS Spatial Underwriter for African Trade Corridors and Real Estate.
Synthesize an executive Underwriting Dossier for a site in Kenya with the following evidence:
- Location: Lat {lat}, Lon {lon} ({county_name})
- Model Prediction: {prediction} (Probabilities: {probabilities_json})
- Recommendation: {recommendation}
- Flood Hazard Flag: {flood_risk}, Slope: {slope_deg}°
- Relevant Spatial Planning Regulations (RAG Context):
  * Kenya Physical and Land Use Planning Act (PLUPA 2019) Sec. 56: Mandatory 30m riparian setback;
    development on slopes >15° requires specialized structural engineering certification.
  * NEMA Environmental Management and Co-ordination Act (EMCA Cap 387): Conversion of classified
    agricultural land >2 hectares mandates Environmental Impact Assessment (EIA) License.
  * East African Trade Corridor Zoning Standard: Commercial logistics hubs along national transit
    trunks must maintain 50m highway buffer & demonstrate hydrological drainage neutral impact.

Format your response as a professional Markdown report with sections:
1. Executive Land Suitability Brief
2. Regulatory Compliance & Environmental Hazard Assessment
3. Strategic Underwriting & Capital Allocation Recommendation
Keep it concise, rigorous, and professional.
```

---

## 5. Data Privacy, Governance & Intellectual Property

1. **No Personally Identifiable Information (PII):** No human subject data, private citizen names, or confidential personal information were ingested, processed, or transmitted to any external AI service.
2. **Open Public Earth Observation:** All multi-spectral satellite imagery and elevation data are derived from publicly accessible, non-commercial research quotas provided by Copernicus ESA (Sentinel-2), NASA/USGS (SRTM), and UCSB (CHIRPS).
3. **No Closed-Source Intellectual Property Infringement:** All algorithmic code, feature engineering routines, and model training pipelines were developed specifically for this project or adapted from standard open-source libraries (Scikit-Learn, LightGBM, PyTorch, FastAPI, Leaflet) under their respective MIT/Apache licenses.

---

## 6. Reproducibility & Deterministic Fallback Guarantee

To guarantee complete scientific reproducibility without dependency on commercial AI API keys:
1. **Fully Scripted Data & Training Pipelines:** The entire data processing, feature extraction, spatial splitting, and model training workflow can be executed deterministically from source code:
   ```bash
   bash scripts/run_pipeline.sh
   ```
2. **Local Offline Inference:** All machine learning inference (`LightGBM`, `ONNX Runtime`) runs 100% locally on CPU without making external network calls.
3. **Deterministic Statutory Fallback:** If `GROQ_API_KEY` or `GEMINI_API_KEY` are unset or unreachable, both the FastAPI backend (`_build_fallback_dossier`) and the React frontend (`generateLocalRAGAnswer`) automatically engage a rule-based statutory engine that generates fully compliant underwriting briefs based on vetted legal statutes.

---

## 7. Declaration of Authorship

I hereby confirm that I am the sole human author of this capstone project. I take full intellectual and ethical responsibility for the design, methodology, code, results, analysis, and conclusions presented in this work. All uses of Artificial Intelligence tools have been fully, accurately, and transparently disclosed in accordance with the standards of Strathmore University.

**Candidate Name:** Valerie Jerono  
**Degree Program:** Master of Science in Data Science & Analytics (MSc DSA)  
**Institution:** Strathmore University / iLabAfrica  
**Date:** 20 November 2026  
**Signature:** *Valerie* Jerono*  
