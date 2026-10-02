# Business Understanding — Kenya Land Readiness Atlas
### CRISP-DM Phase 1 · October 2026

---

## 0. Document purpose

This document executes the four mandatory tasks of the CRISP-DM Business
Understanding phase for the **Kenya Land Readiness Atlas** capstone project.
It grounds every modelling decision in a real-world need, inventories what prior
work already provides, names the gaps this project fills, and translates both
into concrete, measurable targets.

---

## 1. Business Objectives

### 1.1 Primary stakeholder problem

Land-use decisions in Kenya — where to build housing, where to protect
farmland, where to conserve watershed — are made with limited spatial
evidence at the county level. County spatial plans exist on paper but are
rarely backed by timely, sub-county-scale land-cover data that a planner can
query interactively. Two structural gaps drive this:

- **Data latency**: global products such as WorldCover and Dynamic World are
  released annually or with multi-month lag; county planners cannot act on
  them in-season.
- **Interpretability**: a 9-class probability raster is not a decision. A
  planner needs a translated recommendation — settle, farm, or protect — with
  a stated reason and an honesty flag on model uncertainty.

### 1.2 Objectives by stakeholder axis

| Axis | Stakeholder | What they need | What the Atlas provides |
|---|---|---|---|
| **Land planning** | County spatial planners | Sub-county land-cover map, built-up growth rate, hazard overlay | Interactive map, per-click predictions, county export table |
| **Climate risk** | NGOs, DRR agencies | Flood-exposed settlement area, trend into hazard zones | Flood flag + built-up growth into high-water-probability zones |
| **Academic / capstone** | Course examiners, dissertation committee | Honest spatial CV, calibrated uncertainty, reproducible pipeline | Block CV + LORO + calibration plots, all tracked in MLflow |
| **Dissertation linkage** | Personal research | County-level buildability inputs for HFVS allocation model | `county_code, built_up_share, cropland_share, built_up_growth, flood_exposure` export table |

### 1.3 Business success criteria (non-technical)

1. A planner can click a location on the web map and read a land-cover mix
   and a settle/farm/protect recommendation within 5 seconds.
2. The recommendation traces to a visible rule, not a black-box output.
3. The system runs without a GPU and without a paid API on demo day.
4. The county export table joins to at least one external dataset (HFVS or
   Kenya National Bureau of Statistics county codes) without manual editing.

---

## 2. Assessing the Current Situation

### 2.1 Resources available

| Resource | Status | Notes |
|---|---|---|
| Sentinel-2 SR imagery (2017–2024) | ✅ Available via Google Earth Engine | Cloud Score+ mask available |
| WorldCover 2021 v200 | ✅ Available in EE | 10 m, 9 classes; used as training labels |
| SRTM elevation + slope | ✅ Available in EE | Used for hazard rule |
| Dynamic World | ✅ Available in EE | Comparison baseline, not labels |
| GCP project + EE registration | ⚠️ Pending approval this week | Blocks data pull |
| Compute: Colab/Kaggle free tier | ✅ Available | ~4 h GPU/day Colab; 30 h/week Kaggle |
| AlphaEarth embeddings | ✅ Hosted in EE | Precomputed 64-dim annual at 10 m; not open-source |
| BigEarthNet / EuroSAT pretrained weights | ✅ Public | Torchvision-compatible; usable as ablation arm |
| OlmoEarth | ✅ Public paper; weights TBC | Uses southern Kenya land cover as partner task |
| Sen1Floods11 benchmark | ✅ Public | 446 hand-labelled chips; Ghana + Nigeria test sets |
| Repo scaffold | ✅ Done | Data pipeline, CNN, API, Streamlit, Docker, CI all coded |
| Hand-labelled test points | ❌ Not yet | 300–500 needed before quoting accuracy to anyone |

### 2.2 Constraints

| Constraint | Detail |
|---|---|
| **Compute ceiling** | Colab free GPU. Mixed-precision + 64×64 patches + gradient accumulation ×4 is the design response |
| **Time** | 31 days to demo (2 Nov). Feature freeze 24 Oct. Cut rule: if behind by 19 Oct, drop forecast and cross-country test |
| **Labels are the bottleneck** | WorldCover is a model output, not ground truth. Agreement ≠ accuracy. Hand-labelled points are non-negotiable before citing numbers |
| **CPU-only serving** | Class load test runs on CPU. ONNX export mandatory; GPU-dependent model is a liability |
| **EE API rate limits** | Bulk patch download needs retry + threading; ~400 points per call avoids quota errors |

### 2.3 Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| EE quota exceeded during patch export | Medium | High | `patch_export.py` has ThreadPool + resume; start with 1 region |
| WorldCover label noise in arid zone | High | Medium | Report per-region accuracy; flag arid separately in results |
| Built-up over-prediction on metallic roofs / bare soil | High | Medium | Include NDBI + brightness features; hand-label test catches this |
| Colab session reset mid-training | Medium | High | Checkpoint every epoch to Drive; ONNX export from best checkpoint |
| AlphaEarth embedding coverage gaps over Kenya | Unknown | Medium | Verify tile coverage before building embedding pipeline |
| Spatial CV pessimism raised by reviewer | Low | Medium | Report both block CV and LORO; cite Wadoux et al. counter-argument in limitations |

### 2.4 What prior work already provides

> **Key principle**: you inherit every row below. Your novelty is the integration, the Kenya transfer evaluation, and calibration honesty.

#### Labels and global baselines — inherited

| Product | What it gives you | Your use |
|---|---|---|
| **WorldCover 2021** (Zanaga et al.) | 10 m global land cover, 9 classes, Africa | Training labels; remapped to 6 classes |
| **Dynamic World** (Brown et al. 2022) | Near-real-time 9-class probabilities from deep learning on S2 | Comparison baseline on held-out test set |
| **Esri 2020 / Digital Earth Africa cropland** | Independent Africa products | Secondary comparison for cropland class |

#### Pretraining and embeddings — available to borrow

| Dataset / Model | What it gives you | Your planned use |
|---|---|---|
| **EuroSAT** (Helber et al. 2019) | First large-scale S2 patch benchmark; CNN weights | Ablation arm against ImageNet init |
| **BigEarthNet** (Sumbul et al. 2019) | 590 k S2 patches, 43 classes; 60-model benchmark confirms domain pretraining consistently beats ImageNet | Pretrained backbone weights for ablation |
| **SEN12MS** (Schmitt et al.) | S1 + S2 + MODIS triplets, 180 k scenes | Potential S1 extension in stretch goal |
| **AlphaEarth** (Google 2025) | 64-dim annual embeddings at 10 m, EE-hosted; best gains on Africa crop masks | Ablation: frozen embeddings + LightGBM vs fine-tuned CNN |
| **OlmoEarth** | Open-source foundation model; partner tasks in southern Kenya + Nandi — highest geographic relevance | LORO comparison if weights available |

#### Kenya and East Africa existing results

| Study | Method | Location | Accuracy | Key gap |
|---|---|---|---|---|
| Murang'a avocado LULC | Random forest, GEE | Murang'a County | 8 classes | No spatial CV |
| Nandi crop type | RF, 15-day S2 composites | Nandi County | OA 0.80–0.91 | Single county, random split |
| Uasin Gishu agri/non-agri | S1 CoV, no in-situ | Uasin Gishu County | 87.5% OA | No labels, no transfer test |
| OlmoEarth | Foundation model | Southern Kenya + Nandi | Not yet reported | Directly comparable to your LORO |

**Field gap**: all Kenyan studies are single-county, classical ML, with random or no CV. Multi-county deep learning with honest spatial transfer evaluation does not exist in the published literature for Kenya.

#### Built-up mapping — prior work and its warnings

| Study | Method | Region | Relevance |
|---|---|---|---|
| Sub-Saharan urban 1995–2015 | Multi-sensor | 45 cities, avg F1 0.93 | Shows what a well-resourced study achieves |
| HUR dataset | DeepLabV3 | Africa 2016–2022, 10 m | Available comparison raster |
| Open Buildings 2.5D Temporal | Google | Africa | Annual building footprints; plausibility check |

**Known failure mode** (applies directly to your built-up class): classifiers over-predict built-up where bright bare soil or metallic roofing dominates, and under-predict sparse or vegetation-obscured settlements. The hand-labelled test set must include both failure types across all four regions.

#### Flood hazard — prior work

| Study | Finding relevant to your work |
|---|---|
| **Sen1Floods11** (Bonafilia et al.) | Standard benchmark; Ghana + Nigeria test chips included |
| MNDWI vs. deep model | MNDWI approached deep-learning performance on two hold-out sets — your CNN must beat this baseline to justify itself |
| Nairobi + Nyeri flood growth | 2–100× built-up growth into flood-exposed zones — directly supports your hazard flag narrative |

#### Validation methodology — what reviewers will cite

| Reference | Claim | Your response |
|---|---|---|
| Spatial CV leakage studies | Random CV overestimates by up to 28% | Buffered block CV (0.05°, 1-block gap) + LORO is the direct response; `test_splits.py` asserts zero leakage |
| Wadoux et al. (counter-argument) | Spatial CV can be overly pessimistic; probability-sampled design-based inference preferred | Acknowledged in limitations; hand-labelled stratified test set is the design-based element |
| Olofsson et al. (2014) | Area-weighted accuracy for map accuracy assessment | Implemented in `ee_pull.py` sample weights; reported in all confusion matrices |

### 2.5 What prior work does NOT cover — this project's scope

| Gap | What this project does |
|---|---|
| Multi-region CNN transfer within Kenya | Fine-tuned EfficientNet-B0 with LORO across highland, coast, arid, lake basin |
| Calibrated uncertainty on spatial hold-out | ECE + reliability plot on LORO test sets |
| Deployed system (any Kenya study) | FastAPI + Docker + Streamlit + Evidently + MLflow in one compose stack |
| Transparent decision layer | Settle/farm/protect rule with rationale string; not a NN output |
| Dissertation bridge | County-level export table linking land-cover to HFVS buildability inputs |

---

## 3. Data Mining Goals

Specific, measurable translations of the business objectives above.

### 3.1 Land-cover classifier

| Goal | Metric | Minimum target | Evaluation protocol |
|---|---|---|---|
| Classify 6 classes from 64×64 S2 patches | Macro-F1 | ≥ 0.70 | Buffered 5-fold block CV, block = 0.05° |
| Transfer across contrasting regions | LORO macro-F1 | ≥ 0.55 on all 4 held-out regions | Leave-one-region-out |
| Beat LightGBM baseline | Macro-F1 gap | CNN > LGBM on same folds | From `land_atlas_baseline.ipynb` |
| Beat MNDWI threshold for water | Water class F1 | CNN > MNDWI @ best threshold | Evaluated on hand-labelled test set |
| Calibrated confidence | ECE | ≤ 0.10 | 10-bin reliability plot on LORO test sets |
| Independent accuracy (when labels ready) | Area-weighted OA | Reported; no target until points exist | 300–500 hand-labelled stratified points (Olofsson et al.) |

### 3.2 Built-up trend

| Goal | Metric | Target |
|---|---|---|
| Annual built-up share per county, 2017–2023 | NDBI / built-up class time series | Monotonicity check; plausibility vs. Open Buildings |
| Counties with growth into flood zone | % new built-up ∩ P(water) > 0.4 | Reported descriptively; supports hazard narrative |

### 3.3 Hazard flag

| Goal | Metric | Target |
|---|---|---|
| Flag flood exposure per location | Agreement with MNDWI threshold | Precision ≥ 0.75 at 20% recall (Sen1Floods11 if time allows) |
| Transparent rule | Rationale string per API response | 100% coverage — every prediction includes a reason |

### 3.4 Serving performance

| Goal | Metric | Target |
|---|---|---|
| CPU inference latency | p95 `/predict/point` | ≤ 5 s |
| Throughput under load | RPS at 10 concurrent users | ≥ 2 RPS without 5xx |
| Cold start | `docker compose up` → first healthy response | ≤ 60 s |

### 3.5 Drift detection (Week 13 deliverable)

| Goal | Metric | Target |
|---|---|---|
| Detect dry→wet season feature shift | Evidently dataset drift flag | Drift detected on simulated wet-season data; not on reference |
| Report automation | HTML + JSON generated | `python -m monitoring.drift --simulate` runs clean |

### 3.6 Ablation study (Week 12, time-permitting)

| Arm | Init strategy | Hypothesis |
|---|---|---|
| A | Random (no pretrain) | Lowest LORO |
| B | ImageNet weights, averaged to 10 bands | Mid |
| C | BigEarthNet / EuroSAT pretrained | Mid–high |
| D | AlphaEarth frozen embeddings + LightGBM | Comparable to B; no GPU needed |
| E (submitted) | ImageNet → S2 full fine-tune | Highest LORO |

---

## 4. Project Plan

### 4.1 Timeline

| Dates | Course week | Build | Quality gate |
|---|---|---|---|
| 2–4 Oct | Before Wk 11 | GCP/EE registration, repo scaffold ✅ | Repo visible to supervisors |
| 5–11 Oct | **Wk 11**: FastAPI + Docker | Data pipeline v1, spatial splits, API in Docker | `/health` returns 200; zero-leakage test passes |
| 12–18 Oct | **Wk 12**: MLflow + CI | CNN fine-tune, all folds, CI badge green | EfficientNet-B0 LORO > LightGBM LORO |
| 19–25 Oct | **Wk 13**: Drift | Evidently, frontend, rule overlay. **FEATURE FREEZE 24 Oct** | Map click → prediction in < 5 s; drift report generated |
| 26 Oct | **Wk 14**: War room | Cloud deploy, load test, peer debug | 2 RPS @ 10 concurrent; Docker cold-start ≤ 60 s |
| 2 Nov | **Wk 15** | Demo, architecture review, impact story | All quality gates green |

**Cut rule**: if behind by 19 October, drop ConvLSTM/GRU forecast and cross-country test. Keep classifier + API + map. A working deployed system outranks an ambitious unfinished one.

### 4.2 Task breakdown

| Task | Module | Week | Blocker |
|---|---|---|---|
| GCP project + EE registration | Infrastructure | Now | Must clear by 4 Oct |
| `ee_pull.py` across 4 regions | Data | Wk 11 | EE access |
| `patch_export.py` overnight | Data | Wk 11 | ee_pull |
| `spatial_splits.py` | Data | Wk 11 | patch_export |
| API Docker smoke test | API | Wk 11 | None |
| EfficientNet-B0 fine-tune, fold 0 | Training | Wk 12 | Patches |
| All-fold CV + all LORO runs | Training | Wk 12 | Fold 0 clean |
| MLflow experiment comparison (ablation) | Training | Wk 12 | All folds done |
| GitHub Actions CI passing | DevOps | Wk 12 | Repo public |
| Evidently dry→wet simulation | Monitoring | Wk 13 | Reference data |
| Streamlit map live vs. API | Frontend | Wk 13 | API healthy |
| Rule overlay — all 3 actions tested | API | Wk 13 | Frontend |
| Hand-label 300 stratified points | Validation | Wk 13–14 | Basemap access |
| Cloud VM deploy | Serving | Wk 14 | Docker images |
| Load test (10 concurrent, 60 s) | Serving | Wk 14 | Cloud deploy |
| County export table (≥ 4 rows) | Output | Wk 13 | LORO results |

### 4.3 Tools

| Layer | Tool | Notes |
|---|---|---|
| Data pull | `earthengine-api` ≥ 0.1.400 | Colab for EE auth |
| Training | PyTorch + torchvision ≥ 2.2 | Colab GPU or Kaggle |
| Experiment tracking | MLflow ≥ 2.13 | Dockerised, `./mlruns` |
| Model export | ONNX Runtime ≥ 1.18 | CPU provider only |
| API | FastAPI + uvicorn ≥ 0.111 | 1 worker, OMP_NUM_THREADS=2 |
| Frontend | Streamlit + Folium ≥ 1.35 | Port 8501 |
| Monitoring | Evidently ≥ 0.4.30 | HTML + JSON reports |
| CI | GitHub Actions | ruff + pytest + Docker build |
| Containers | Docker Compose ≥ 3.9 | api + frontend + mlflow |

### 4.4 Budget

| Item | Estimated cost |
|---|---|
| Google Earth Engine | Free (non-commercial research) |
| Colab / Kaggle GPU | Free (upgrade Colab Pro if quota binds: ~$12) |
| GCP VM for load test (e2-standard-2, ~30 h) | ~$3 |
| All other tools | Free / open source |
| **Total** | **< $15** |

### 4.5 Quality gates summary

**End of Wk 11**
- [ ] `ee_pull.py` returns ≥ 1,200 points across ≥ 2 regions without error
- [ ] `test_splits.py::test_no_leakage` passes (zero block-level overlap)
- [ ] `GET /health` returns `{"status":"ok"}` from Docker

**End of Wk 12**
- [ ] Block CV macro-F1 logged in MLflow for both LightGBM and CNN
- [ ] EfficientNet-B0 LORO macro-F1 > LightGBM LORO macro-F1
- [ ] CI badge green on `main`

**End of Wk 13 (feature freeze)**
- [ ] All 5-fold block CV + 4 LORO runs completed and logged
- [ ] ECE ≤ 0.10 on at least one LORO fold
- [ ] Map click → recommendation in < 5 s
- [ ] Evidently drift report generated for simulated wet season
- [ ] County export table populated (≥ 4 counties)

**Demo day (2 Nov)**
- [ ] `docker compose up` → healthy in ≤ 60 s on CPU-only VM
- [ ] Live click → prediction + rationale in < 5 s
- [ ] Load test: ≥ 2 RPS @ 10 concurrent, no 5xx
- [ ] Confusion matrix + LORO results visible in MLflow or deck
- [ ] Limitations section present: WorldCover ≠ ground truth; no hand-labelled set yet

---

## 5. Novelty statement (for proposal and report)

| | Prior work provides | This project adds |
|---|---|---|
| **Labels** | WorldCover, Dynamic World, EuroSAT, BigEarthNet | WorldCover remap + hand-labelled validation points |
| **Model** | CNN benchmarks, AlphaEarth, OlmoEarth | EfficientNet-B0 fine-tune with 10-band first-conv; ablation across init strategies |
| **Kenya geography** | County-level RF studies (Murang'a, Nandi, Uasin Gishu) | Multi-county deep learning with LORO spatial transfer |
| **Hazard** | Sen1Floods11 benchmark | MNDWI + water-class probability rule; beat-MNDWI baseline test |
| **Validation** | Spatial CV and leakage literature | Buffered block CV + LORO + calibration + hand-labelled stratified test |
| **Deployment** | None — all studies are offline | FastAPI + Docker + Streamlit + Evidently + MLflow, CPU-serving |
| **Decision layer** | None | Transparent rule overlay (settle / farm / protect) with rationale string |

> **The novelty claim**: cross-region transfer evaluation of a fine-tuned multispectral CNN for Kenya land cover, delivered as a calibrated, CPU-serving, interpretable system with a transparent recommendation layer. No single element is new; the combination is.

---

*CRISP-DM Phase 1 · Kenya Land Readiness Atlas · 2 October 2026*
*Update this document after each weekly quality gate.*
