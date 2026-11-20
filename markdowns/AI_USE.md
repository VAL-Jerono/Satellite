# AI Assistance & Tools Disclosure (AI_USE.md)

**Project:** Kenya Land Readiness Atlas: Spatial AI for Affordable Housing & Land Use Allocation  
**Course:** Capstone Project - Applied Machine Learning & Spatial Intelligence  
**Author:** Valary Jerono  
**Academic Year:** 2026  

---

## 1. Executive Statement of AI Use

In accordance with the academic integrity and AI use guidelines of this course, this document provides a transparent, detailed disclosure of the Artificial Intelligence (AI) tools, models, and generative systems utilized during the ideation, engineering, training, documentation, and deployment of the **Kenya Land Readiness Atlas**.

The core scientific hypotheses, spatial research questions, problem framing, data curation strategies, regulatory compliance rules, and final conclusions were conceived and directed by the human author. AI systems served strictly as cognitive accelerators, code scaffolding assistants, refactoring aids, and runtime inference components for the statutory Retrieval-Augmented Generation (RAG) copilot.

---

## 2. Inventory of AI Systems & Services Used

| Tool / Model | Provider / Architecture | Primary Role in Project | Verification Mechanism |
|---|---|---|---|
| **Antigravity / Gemini & Claude** | Google DeepMind / Anthropic | Code scaffolding, test generation, bash automation, Docker troubleshooting | Full manual review, unit tests, local compilation, runtime logging |
| **GROQ API (`llama-3.3-70b-versatile`)** | Groq / Meta Llama 3.3 | Runtime spatial copilot for statutory planning queries (PLUPA 2019, NEMA EMCA) | Grounded with deterministic regulatory context strings; fallback to rule engine |
| **Google Gemini 1.5 Flash** | Google Cloud Vertex AI / AI Studio | Secondary fallback LLM for spatial underwriting synthesis | Grounded prompt templates; fallback to deterministic statutory text |
| **GitHub Copilot** | GitHub / OpenAI Codex | IDE-level inline autocomplete for standard Python/React boilerplate | Code syntax validation, linting (`ruff`), and build tests |
| **Google Earth Engine (GEE)** | Google Earth Engine API | Automated multi-spectral cloud masking (`Cloud Score+`), radiometric median compositing | Visual validation via GEE Code Editor against high-res imagery |

---

## 3. Granular Breakdown by Project Phase

### Phase 1: Problem Understanding & Architecture Design
* **Assistance Received:** Brainstorming comparative baselines between standard random Cross-Validation (K-Fold) versus spatially blocked Leave-One-Zone-Out (LOZO) validation.
* **Human Ownership:** Framing the target metrics around housing density ceilings under the National Housing Development Fund and integrating with Mixed Integer Linear Programming (MILP).

### Phase 2: Data Pipeline & Radiometric Feature Engineering
* **Assistance Received:** Drafting vectorized NumPy and Pandas calculations for 27 radiometric and topographic spectral indices (NDVI, EVI, MNDWI, NDBI, NDRE, BSI, TWI, dry/wet season differentials).
* **Human Verification:** All mathematical formulas were cross-checked against standard remote sensing literature (e.g., McFeeters 1996 for NDWI, Rouse 1974 for NDVI, Rikimaru 2002 for BSI).

### Phase 3: Model Architecture & Training
* **Assistance Received:** Generating boilerplate for LightGBM hyperparameter optimization and PyTorch dataset iterators for multi-spectral patch extraction.
* **Human Verification:** The core zone-stratification insight-recognizing that bare arid rock in Turkana spectrally mimics urban concrete rooftops in Nairobi-was identified through human exploratory data analysis (EDA). The models were trained and serialized on actual Sentinel-2 observations without synthetic data tampering.

### Phase 4: Full-Stack Web Interface & Leaflet Cartography
* **Assistance Received:** Refactoring React Leaflet layer switches, implementing Vite reverse proxies to bypass browser Mixed Content restrictions on Vultr IP endpoints, and creating the Markdown parser component (`MarkdownMessage.jsx`).
* **Human Verification:** Verified map boundary accuracy using the official 451-point sovereign boundary GeoJSON from national survey data, ensuring zero border bleeding into Uganda or Somalia.

### Phase 5: Statutory Regulatory Grounding (RAG Engine)
* **Assistance Received:** Synthesizing complex legal clauses into concise prompt templates for the LLM.
* **Human Verification:** All legal citations were manually audited against official statutory publications:
  - *Physical and Land Use Planning Act (PLUPA 2019), Section 56 (30m riparian setback & steep slope restrictions).*
  - *Environmental Management and Co-ordination Act (EMCA Cap 387), Second Schedule (Mandatory EIA for agricultural conversions >2.0ha).*
  - *East African Trade Corridor Transport Trunk Zoning Standards.*

---

## 4. Academic Integrity & Reproducibility Statement

1. **No Plagiarism of Analytical Work:** All model evaluations (Macro-F1, accuracy, Confusion Matrices, LOZO splits) are the direct output of executed Python code stored in `models/` and `data/county_summaries/`.
2. **Reproducibility:** A reviewer can reproduce the entire pipeline from scratch using `scripts/run_pipeline.sh` without requiring an active AI subscription.
3. **Deterministic Fallbacks:** The API and frontend are engineered with deterministic fallback algorithms; even in the event of third-party LLM service outages, the system operates deterministically on vetted statutory rules.

**Signed:**  
*Valary Jerono*  
Lead Researcher & Capstone Candidate

