# Kenya Land Readiness Atlas — Defense Deck Companion

**10-slide presentation · November 2026**  
**Candidate:** Valerie Jerono · MSc Data Science & Analytics, Strathmore University (iLabAfrica)  
**Course:** DSA 8401 Applied Machine Learning

**Delivered deck:** [Kenya Land Readiness Atlas — Evidence-led Capstone Defense](manus-resource://manus-slides/deck/kenya-land-readiness-atlas-evide-mwp41y1a-d2d71327)

This companion mirrors the final 10-slide deck. Values tagged **[N]** are reported in the submitted notebook and have not been independently reproduced. **[C]** denotes findings from inspection of packaged source/code or artifacts. **[I]** denotes illustrative material, not an empirical map or result. The full evidence register and detailed citations are in [`FINAL_REPORT.md`](./FINAL_REPORT.md).

**Suggested total delivery time:** 8 minutes, plus questions. Speaker notes are a delivery guide, not additional evidence.

## Slide 1 — Kenya Land Readiness Atlas (0:30)

**On-slide message:** An evidence-led spatial machine-learning screening prototype. Candidate and programme metadata.

**Evidence:** Project identity and date [C]. Cover landscape image [I], generated for presentation design; not a satellite observation.

**Speaker notes:** The Atlas asks whether open Earth-observation signals can help identify where a site merits closer investigation. The phrase “screening prototype” is deliberate: this work does not certify land suitability, title, or legal compliance. I’ll focus on the method, the limits of the evidence, and the steps needed before a responsible pilot.

## Slide 2 — A first-pass screen, not a permit (0:50)

**On-slide message:** Intended users include planners, housing and infrastructure sponsors, agricultural lenders, and conservation teams. The system can inform triage, never replace surveys or approvals.

**Evidence:** Scope and API boundaries [C]. Planning-office image [I].

**Speaker notes:** The decision problem is not simply to produce a land-cover map. A user needs context and a reason to investigate. The prototype offers broad probabilities and a small, inspectable action layer. It cannot establish whether a parcel is buildable or whether a proposal meets statutory requirements. That boundary matters because the reference labels and operational evidence are incomplete.

## Slide 3 — Four inputs, one explicit label caveat (0:50)

**On-slide message:** Sentinel-2, SRTM, CHIRPS, and ESA WorldCover 2021 inform 28,121 sampled observations, 13 strata, five zones, six classes, and 27 features.

**Evidence:** Counts and feature design [N]; source product selection and package contents [C]. Input strip image [I]. WorldCover is a model-derived reference, not field truth.

**Speaker notes:** The notebook reports 28,121 observations and a near-balanced six-class sample across five zones. Near balance is a sampling choice; it does not describe Kenya’s landscape composition. Most importantly, labels come from ESA WorldCover. Agreement with that product is a useful consistency measure, but it is not independent field accuracy. The raw observation table and frozen split files are not in the archive.

## Slide 4 — A transparent workflow, with a spatially harder test (0:50)

**On-slide message:** Engineer features, split by 0.05-degree blocks with a one-block buffer, train zone specialists and a CNN, then apply transparent provisional rules.

**Evidence:** Workflow and block size [N/C].

**Speaker notes:** Random pixel splitting can make nearby, highly similar locations appear in both training and test data. The project uses geographic blocks and a buffer, then reports leave-one-zone-out results as a harder transfer test. This reduces one form of leakage; it does not prove all spatial dependence is eliminated. The split manifests are absent, so the implementation and metrics still need an independently reproducible audit.

## Slide 5 — The hard transfer case is the ASAL zone (1:10)

**On-slide message:** LOZO macro-F1: Rift 0.5443, Highland 0.5050, Lake 0.4728, Coastal 0.3665, ASAL 0.2732; mean 0.4324. Separate benchmark inventory reports 0.6271 zone LightGBM and 0.6504 stacking.

**Evidence:** All scores [N]. Classification visual [I]. Protocols differ; do not compare the benchmark inventory directly to LOZO.

**Speaker notes:** The central result is not a single headline score. Transfer varies materially by zone, with ASAL weakest. The notebook also reports higher scores for zone LightGBM and stacking in a separate benchmark table, but the evaluation protocols are different. The values should not be treated as one leaderboard. Every score here is notebook-reported; this package lacks raw observations and predictions for independent recomputation.

## Slide 6 — Transfer failure is a finding (0:55)

**On-slide message:** ASAL LOZO macro-F1 is 0.2732; an in-domain ASAL specialist is separately reported at 0.3603. These answer different questions.

**Evidence:** Scores [N]. Landscape image [I].

**Speaker notes:** Holding out an entire zone asks whether a model trained elsewhere transfers to an unseen environment. It is intentionally demanding. The in-domain specialist value reported for ASAL answers a different question. Both are modest, and neither supports confident use for a real site. The next priority is independent reference data in the failure zones, not a claim that stratification has solved the problem.

## Slide 7 — Calibration is promising; it still needs replication (0:45)

**On-slide message:** Reported ECE decreases from 0.2525 to 0.0921 after temperature scaling at T=1.42, a 63.5% reduction.

**Evidence:** Calibration values [N]. Reliability bins and calibration split are absent.

**Speaker notes:** Better calibration can make probability outputs more interpretable, but expected calibration error depends on binning and evaluation design. The notebook does not include the reliability bins or the exact calibration split. Treat this as a promising reported result, not proof that served probabilities are calibrated for every zone or site. Replication and zone-specific reliability analysis are prerequisites for decision use.

## Slide 8 — Readable triage, not statutory zoning (0:45)

**On-slide message:** SETTLE, FARM, and PROTECT summarize provisional prototype actions. No output is a legal designation. Images are illustrative.

**Evidence:** Rule categories [C/N]. Images [I]. Legal caveat: PLUPA 2019 section 56 is not a universal 30 m/15° rule; riparian regulation appears in 2021 LN 253, Part V/regulation 14, with widths that vary.

**Speaker notes:** A transparent rule layer makes it easier to inspect why a recommendation appeared. But an inspectable rule is not necessarily a legally correct rule. The prior materials overstated the legal basis for fixed buffers and slope restrictions. This deck treats thresholds only as screening parameters. A qualified legal review must precede any use in a planning or lending workflow.

## Slide 9 — A service scaffold, not verified production (0:55)

**On-slide message:** API, artifacts, and Docker configuration are packaged. Deployment, CI, uptime, latency, and load evidence are not verified. Two unsafe fallbacks require remediation.

**Evidence:** Source inspection [C]. Architecture image [I].

**Speaker notes:** The package includes a FastAPI service and model artifacts, but that is not the same as verified production operation. Source inspection found that point prediction may use a zero feature vector when actual point inputs are missing, and the service can create a placeholder ONNX graph when the expected artifact is absent. The correct production behavior is to fail closed, report missing inputs, and verify model checksums. Notebook mock health and hard-coded test tables are not operational measurements.

## Slide 10 — Strengthen the evidence before scaling (0:30)

**On-slide message:** Collect stratified field labels; freeze and reproduce evaluation; review rules with counsel; fail closed and measure operations.

**Evidence:** Recommended governance steps [C]. Fieldwork image [I].

**Speaker notes:** The next phase should be evidence-led: collect independent reference labels, freeze splits and scoring artifacts, validate calibration by zone, review the rules with counsel, and run instrumented service tests. That work is more valuable than adding complexity to the model. The contribution today is an inspectable prototype and an explicit account of where its evidence ends.

## Timing and Q&A

| Slide | Target time |
|---|---:|
| 1 | 0:30 |
| 2 | 0:50 |
| 3 | 0:50 |
| 4 | 0:50 |
| 5 | 1:10 |
| 6 | 0:55 |
| 7 | 0:45 |
| 8 | 0:45 |
| 9 | 0:55 |
| 10 | 0:30 |
| **Total** | **8:00** |

For questions, keep three distinctions clear: **reference-label agreement is not field accuracy; LOZO is not in-domain performance; and packaged code is not verified production operation.**
