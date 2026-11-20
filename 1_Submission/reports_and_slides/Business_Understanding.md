# Business Understanding — Kenya Land Readiness Atlas

**CRISP-DM Phase 1 · Final retrospective · November 2026**  
**Candidate:** Valerie Jerono · MSc Data Science & Analytics, Strathmore University (iLabAfrica)  
**Course:** DSA 8401 Applied Machine Learning

## 1. Purpose and decision context

This document records the business problem, intended users, requirements, and final status of the Kenya Land Readiness Atlas. It is a retrospective account of the submitted prototype—not a forward-looking project plan and not a claim of operational approval.

The project addresses a practical information gap: planners, lenders, and conservation teams may need an early, spatially explicit view of land-cover context before commissioning a survey or specialist assessment. The Atlas tests whether open Earth-observation data can support that **first-pass screening** across diverse Kenyan environments.

The system is not a cadastral product, parcel survey, title service, engineering assessment, environmental approval, zoning instrument, or substitute for statutory review. It cannot establish that land is legally or physically suitable for a proposed use.

## 2. Stakeholders and decisions

| Stakeholder | Potential decision supported | Responsible use of prototype |
|---|---|---|
| County physical-planning staff | Which locations warrant site inspection or a fuller planning assessment? | Review broad land-cover and terrain signals; consult county plans and qualified professionals. |
| Housing or infrastructure sponsors | Which candidate sites should advance to survey and due diligence? | Use as an initial screening prompt only; not as a buildability or compliance determination. |
| Agricultural lenders and cooperatives | Which parcels warrant agronomic review? | Use crop-likelihood as a triage signal; verify soil, tenure, access, water, and current land use independently. |
| Conservation and climate-finance teams | Where should field verification be prioritised? | Use relative land-cover signals to guide investigation; do not infer legal protection status or carbon eligibility. |
| Examiners and technical reviewers | Is the analytical workflow documented and its evidence appropriately qualified? | Inspect code, notebook, artifacts, protocol distinctions, and evidence register. |

## 3. Business objectives and measurable criteria

The initial intent was translated into testable criteria. Status reflects the evidence included in this submission, not an assumption that code presence equals successful operation.

| Objective | Success criterion | Final status | Evidence / limitation |
|---|---|---|---|
| Make a location-level screen available | A coordinate produces a site-specific prediction in an end-to-end test within five seconds | **Not verified** | The API exists, but `/predict/point` can fall back to a zero feature vector when patch/GEE inputs are unavailable; no measured end-to-end latency is packaged. |
| Make recommendation logic inspectable | Every recommendation includes a transparent action and rationale | **Partially evidenced** | A deterministic rule layer is implemented. Its thresholds and statutory framing need correction and review; no packaged test suite proves full response coverage. |
| Support CPU-oriented inference | Core model path can run without a GPU | **Implemented, not operationally certified** | CPU-oriented artifacts and service code are present; no reproducible deployment transcript or resource benchmark is supplied. |
| Support independent validation | Report metrics reproducible from frozen splits and predictions | **Not met** | Notebook-reported results are retained with provenance, but raw observations, split files, and prediction outputs are absent. |
| Support responsible planning interpretation | Legal context is accurate and appropriately caveated | **Not met for current prototype rules** | Earlier fixed setback/slope and EIA claims were not defensible as cited; this submission corrects the narrative and recommends legal review before any operational use. |
| Demonstrate county-level summaries | County summaries can be independently joined and interpreted | **Not verified** | Notebook contains sample-based allocation summaries; the underlying county output files are not packaged. Shares are not estimates of county area. |

## 4. Scope and delivered concept

The submitted workflow combines Sentinel-2 Level-2A reflectance, SRTM terrain derivatives, CHIRPS rainfall context, and ESA WorldCover 2021 reference labels. The notebook reports 28,121 observations, 13 county/region strata, five bioclimatic zones, six remapped classes, and 27 engineered features. Five zone-specific LightGBM models, a spectral CNN, a stacking learner, temperature scaling, and a transparent three-action rules layer are described. A FastAPI service and Docker configuration provide a serving scaffold.

The choice of six broad classes and three provisional actions is deliberate: it makes a technical output easier to discuss with non-specialists. But the action names are not statutory categories. The underlying labels are inherited from a model-derived global map, and a correct-looking recommendation does not validate either the prediction or its legal interpretation.

### Out of scope

- Determining title, tenure, boundaries, easements, or parcel ownership.
- Issuing development permission or certifying compliance with Kenyan law.
- Estimating national or county land-cover area from the stratified sample.
- Replacing field surveys, environmental impact assessment, geotechnical work, flood modelling, or agronomic analysis.
- Claiming independent classification accuracy without a separately collected reference sample.
- Claiming a production service-level objective, deployed uptime, or measured load capacity.

## 5. Assessment of resources and current situation

| Resource | Submission status | Implication |
|---|---|---|
| Data extraction and feature pipeline | Source under `data_pipeline/` | Rerun depends on Earth Engine access, configuration, and upstream inputs. |
| Observation-level training table | Not included | Reported metrics cannot be recomputed end to end from this archive alone. |
| Split manifests and saved predictions | Not included | Protocol and metric reproduction require additional artifacts. |
| Model code and serialized artifacts | Present under `model_training/` | Presence alone does not establish artifact-to-report metric lineage. |
| API and deployment configuration | Present under `api_service/` and `docker_configuration/` | Scaffold can be inspected; production readiness has not been demonstrated. |
| Automated tests | No `tests/` directory in the package | No test count or passing-suite claim is made. |
| Independent field labels | Not included | No field accuracy or area-adjusted accuracy claim is justified. |
| Monitoring/load/health evidence | Notebook includes simulated or hard-coded demonstrations | These are not measurements of a running deployment. |

## 6. Risks: anticipated and realized

| Risk | What happened / status | Required mitigation before real use |
|---|---|---|
| Reference-label error, especially in arid terrain | Realized as a core validity limitation: WorldCover is a model-derived reference, not field truth. | Design stratified, independent field interpretation; include ambiguous surfaces and rare classes. |
| Geographic transfer failure | Realized: notebook-reported LOZO macro-F1 ranges from 0.2732 (ASAL) to 0.5443 (Rift), mean 0.4324. | Collect targeted labels in underperforming zones; evaluate with a frozen protocol and uncertainty intervals. |
| Spatial leakage or dependence | Buffered 0.05° blocks are described; the raw split artifacts are unavailable for audit. | Package split IDs and code-generated leakage diagnostics; test buffer sensitivity and larger spatial scales. |
| Non-site-specific API fallback | Realized in source: the point route can score a zero-filled vector if no actual point features are available. | Fail closed with an explicit missing-input status; never present fallback output as a coordinate assessment. |
| Placeholder model substitution | Realized in source: a missing ONNX artifact can trigger a generated placeholder. | Fail startup in production when the expected artifact is absent; pin checksums and model versions. |
| Legal misstatement | Realized in earlier materials; fixed narrative now distinguishes PLUPA section 56 from the 2021 development-control regulations. | Obtain qualified legal review; version sources and map rules to exact provisions and jurisdiction. |
| Misleading operational claims | Notebook health response, load test and smoke checks are mock/hard-coded demonstrations; no operational logs accompany them. | Run auditable tests in the target environment and package timestamped raw results. |
| Sample-selection bias | Observations are stratified, not a probability sample of land area. | Use an explicit probability sampling design and design weights for map-area estimates. |

## 7. Results relevant to business acceptance

The notebook reports a leave-one-zone-out mean macro-F1 of **0.4324** (Rift 0.5443; Highland 0.5050; Lake 0.4728; Coastal 0.3665; ASAL 0.2732). It separately reports **0.6271** for zone LightGBM and **0.6504** for stacking in a benchmark inventory. These are different evaluation protocols and are not like-for-like. Temperature scaling is reported to reduce ECE from **0.2525** to **0.0921** at *T* = 1.42; reliability-bin data and the calibration split are not included.

These results support a research prototype and motivate further validation. They do not meet a defensible acceptance threshold for field deployment, legal underwriting, or a county planning decision. All reported model metrics remain notebook-reported, not independently reproduced from packaged raw data.

## 8. Legal and governance boundary

The Physical and Land Use Planning Act 2019, section 56, concerns county development-control powers; it is not the source of a universal 30 m riparian buffer or a 15° slope prohibition. Riparian reserve measurement appears in the Physical and Land Use Planning (General Development Permission and Control) Regulations, 2021, Legal Notice No. 253, Part V/regulation 14; widths vary by watercourse. The frequently repeated EIA screening threshold for agricultural schemes is 100 hectares, not 2 hectares. These rules are jurisdiction- and context-sensitive and require counsel to confirm the current law and its application.

Accordingly, this package treats slope, water proximity, and action thresholds as **prototype screening parameters**, not legal rules. The system must not produce a compliance verdict. Any future legal copilot should use a versioned authoritative corpus, citation-level retrieval, explicit uncertainty, and human review.

## 9. Final business recommendation

Advance only as a controlled research and demonstration prototype. The next investment should be an evidence programme rather than a more complex model: (1) independent stratified field reference data; (2) frozen and packaged train/validation/test splits, predictions, and scoring scripts; (3) zone-specific error and calibration analysis; (4) qualified legal review of every rule and user-facing statement; and (5) fail-closed API behavior followed by instrumented deployment and load testing.

The business value at this stage is the inspectable workflow and the visibility it gives to geographic failure—not a readiness score that can be used to approve land. A future operational pilot should begin only after all five gates are documented and signed off by accountable human reviewers.

## 10. Companion materials

- [Final report](./FINAL_REPORT.md) — methods, results, legal analysis, and evidence register.
- [Slide companion and speaker notes](./SLIDE_DECK.md) — 10-slide defense narrative with sources and timing.
- [Figures and generation script](./figures/) — report visual assets.
- Submission-root [README](../README.md) and [AI use disclosure](../AI_USE.md).
