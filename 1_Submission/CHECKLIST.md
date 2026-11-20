# Submission Checklist — Kenya Land Readiness Atlas

**Submission package review · November 2026**  
**Candidate:** Valerie Jerono · MSc Data Science & Analytics, Strathmore University (iLabAfrica)  
**Course:** DSA 8401 Applied Machine Learning

This checklist records what is actually present in the submitted archive and distinguishes packaged artifacts from claims that still require verification. A file's presence does not establish that it has run successfully or is production-ready.

| Deliverable | Package location | Status and qualification |
|---|---|---|
| Project overview and setup | `README.md` | Aligned with the actual package tree; includes reproducibility constraints and safety caveats. |
| AI use disclosure | `AI_USE.md` | Discloses development assistance, documentation/slide drafting assistance, and simulated notebook outputs. |
| Business understanding | `reports_and_slides/Business_Understanding.md` | Final CRISP-DM retrospective with stakeholders, outcomes, realized risks, and scope boundaries. |
| Final report | `reports_and_slides/FINAL_REPORT.md` | Evidence-led report; metrics labeled notebook-reported; legal claims corrected and qualified. |
| Presentation | Slides resource; companion at `reports_and_slides/SLIDE_DECK.md` | 10-slide evidence-led defense deck; companion includes slide content, speaker notes, and timing. |
| Notebook | `CAPSTONE_NOTEBOOK.ipynb` | 59-cell project notebook. Some operational outputs are mock, hard-coded, or synthetic demonstrations, not measurements. |
| Data pipeline | `data_pipeline/` | Source code and configuration packaged; raw observation data and patch archive not included. |
| Model code and artifacts | `model_training/` | Code, configurations, five zone LightGBM artifacts, CNN checkpoint, meta-learner, and ONNX file present. Full metric lineage is not independently reproduced. |
| API service | `api_service/main.py` | Source packaged. Includes unsafe fallback behavior that must be corrected before any operational use. |
| Container configuration | `docker_configuration/` | Dockerfiles, Compose files, and Nginx configuration packaged. No successful deployment transcript included. |
| Automated tests | No `tests/` directory | **Not included.** No test count, passing-suite, or CI claim is made. |
| Independent field validation | Not included | **Not performed or documented in this archive.** WorldCover is a model-derived reference, not field ground truth. |
| Operational evidence | Not included | Mock health output, literal load/smoke tables, and a synthetic drift exercise are demonstrations, not production telemetry. |
| Report figures | `reports_and_slides/figures/` | Figures and `make_figures.py` included; charts visualize notebook-transcribed values, not newly recomputed model results. |
| Study-area map | `kenya_zones_map.png` | Included; annotation includes an 8,356 unique-location figure that cannot be recomputed without the raw table. |

## Evidence labels used across the submission

- **A — Reproducible source or artifact:** source or model file exists in the archive; execution may still depend on missing data, credentials, or configuration.
- **B — Notebook-reported:** output appears in the submitted notebook; not independently rerun during package refinement.
- **C — Simulated or illustrative:** synthetic distributions, hard-coded outputs, or presentation imagery; not empirical operational or geographic evidence.
- **D — Unverified external claim:** a claim requiring external logs, field observations, or other primary evidence not supplied in the archive.

## Corrections applied to the final package

- [x] Replaced "ground truth" with **WorldCover reference labels**.
- [x] Separated LOZO transfer results (mean macro-F1 0.4324) from benchmark-protocol results (zone LightGBM 0.6271; stacking 0.6504).
- [x] Corrected ONNX size to approximately **387 KB (0.4 MB)** as packaged.
- [x] Removed conflicting test counts and stated that tests are not packaged.
- [x] Identified mock health, hard-coded load/smoke outputs, and synthetic drift as demonstrations.
- [x] Disclosed API zero-feature scoring and placeholder-model fallbacks.
- [x] Reframed statutory thresholds as provisional screening parameters; corrected PLUPA, riparian, and EIA discussion.
- [x] Labeled county allocation percentages as shares of sampled observations, not county land area.
- [x] Added report figures and a reproducible figure-generation script.
- [x] Cross-linked the report, business understanding, slide companion, README, and AI-use disclosure.

## Final author responsibilities

- [ ] Personally verify every citation and quoted legal statement against the current official instrument.
- [ ] Review and approve all AI-assisted document and slide edits before submission.
- [ ] Confirm course-specific formatting, declaration, and signature requirements.
- [ ] If presenting, confirm the Slides resource opens and the speaker notes are visible.
