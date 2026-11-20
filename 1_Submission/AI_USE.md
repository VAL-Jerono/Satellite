# AI Use and Research Integrity Disclosure

**Project:** Kenya Land Readiness Atlas — spatial machine-learning screening prototype  
**Candidate:** Valerie Jerono · MSc Data Science & Analytics, Strathmore University (iLabAfrica)  
**Course:** DSA 8401 Applied Machine Learning  
**Disclosure date:** 20 November 2026

## 1. Statement of responsibility

This disclosure describes the AI-assisted work represented in the submitted package and the evidence available to support those statements. The human author remains responsible for the research questions, methodological choices, code and data decisions, interpretation, legal review, and final submitted claims. AI assistance does not establish correctness or authorship, and all consequential outputs require human verification.

The project uses ESA WorldCover 2021 as a **reference label source**. It is a model-derived land-cover product, not ground-truth field data. The submitted package contains no independent field sample. Therefore, model agreement with WorldCover must not be represented as field accuracy.

## 2. Tool and system inventory

| Tool / system | Use represented in the project | Evidence boundary |
|---|---|---|
| Generative AI coding assistants used during development (including the tools named in earlier project notes: Antigravity, GitHub Copilot, and chat assistants) | Assistance with code scaffolding, debugging, refactoring, and drafting technical text, as described in project records. | The package does not preserve a complete prompt/session log or a line-by-line attribution of generated code. No claim of exhaustive provenance is made. The author must review all retained code. |
| Groq-hosted Llama endpoint and Google Gemini endpoint | The API source includes an optional external-LLM copilot pathway with fallback logic. | Source code documents intended behavior; no usage logs or verified production calls are included. The endpoint is not evidence of a deployed RAG system or legally grounded advice. |
| Google Earth Engine APIs | Data extraction and derived-feature workflow in `data_pipeline/`. | Inputs and outputs are not wholly bundled; reproducibility depends on Earth Engine access, project configuration, and source availability. |
| Jupyter, Python scientific/ML libraries and deterministic scripts | Data analysis, model development, metrics and plots described in the notebook and source. | Notebook metrics have not been independently rerun from packaged raw observations and split artifacts. |
| Manus Slides and image generation tools | Assistance creating the polished defense deck and its illustrative imagery in this submission session. | Generated photographic-style images are illustrative. They are not satellite observations, model outputs, or evidence of project field sites. |
| Manus AI assistance in documentation | Structural editing, evidence auditing, consistency review, and drafting/rewrite assistance for the final report, README, business understanding, AI_USE disclosure, slide companion, and checklist. | The submitted content was grounded in the provided package and notebook and qualified where evidence was absent. The human author must review and approve the final wording before academic submission. |

No claim is made that every tool/model version or every prompt used in earlier development can be reconstructed from the package. This document should be amended if the author has a more complete record of development-time systems.

## 3. Human and AI roles by project phase

### Business understanding and scope

The human author owns the problem framing, intended stakeholder decisions, CRISP-DM choices, and final scope. AI assistance helped organize and edit explanatory prose. The final materials explicitly restrict use to preliminary screening and remove unsupported claims of legal enforceability or production readiness.

### Data engineering

The repository contains Earth Engine extraction, feature engineering, and spatial-split code. Automated assistance may have contributed boilerplate or debugging, but the exact contribution cannot be attributed line by line from the submitted evidence. Human responsibility includes confirming collection identifiers, scale/resampling choices, masking logic, label remapping, feature definitions, and data provenance. These checks are not equivalent to independent replication.

### Modeling and evaluation

The notebook and packaged source describe zone LightGBM models, a spectral CNN, stacking, temperature scaling, and spatial validation. Human responsibility includes selecting evaluation protocols, interpreting their differences, reporting the weak ASAL transfer result, and distinguishing model-derived reference agreement from field accuracy. The reported metrics are not independently recomputed for this disclosure. No claim is made that a simulated output is a model measurement.

### API, copilot, and operations

The API source contains external LLM integration/fallback logic and also contains important unsafe development fallbacks: point predictions can use zero-filled features when valid site data are unavailable, and missing ONNX files may cause a placeholder graph to be generated. Those behaviors are disclosed as limitations, not endorsed for production. The legal prompt/context in earlier code and documents contained inaccurate statutory claims; the revised documentation corrects the narrative and requires qualified legal review. No live deployment, production uptime, load capacity, CI pass, or test count is asserted here.

### Documentation and presentation

AI assistance was used during this refinement to audit the archive, identify contradictions, improve organization and language, produce the companion documentation, and draft the 10-slide presentation and speaker-note narrative. Figures were generated from notebook-transcribed values and labeled as reported or illustrative. This assistance does not replace the author's obligation to inspect the sources, verify citations, and approve the final submission.

## 4. Representative assistance patterns

The following are representative descriptions of assistance, not verbatim historical prompt transcripts:

- “Review this submission archive and notebook; identify inconsistencies between claims, source code, artifacts, and measured evidence. Do not treat notebook literals as measurements.”
- “Rewrite the business understanding as a final CRISP-DM retrospective; keep intended use, achieved status, risks, and non-claims separate.”
- “Build an evidence-led presentation from the stated metrics; label notebook-reported results, separate different validation protocols, and mark generated imagery as illustrative.”
- “Review legal statements against authoritative Kenyan sources and avoid asserting fixed rules where the cited provision does not establish them.”

## 5. Simulated and illustrative notebook outputs

The following notebook outputs are **not empirical measurements of a running service**:

- A health-check example was displayed as a hard-coded `mock_health` dictionary after the HTTP request returned 403.
- Load-test values are represented by a DataFrame of literals, not captured load-test telemetry.
- Smoke-test results are hard-coded pass strings, not outputs from an executed test suite.
- The drift exercise uses synthetic distributions and demonstrates an analysis pattern; it is not observed production drift.

These may illustrate intended methods, but must not be described as verified uptime, throughput, latency, deployment status, CI success, or field behavior. The package includes no `tests/` directory, so no automated test count or passing test-suite claim is made.

## 6. Verification and risk controls

The refinement applied these checks to claims used in final materials:

1. Cross-checked file/path and implementation claims against the extracted submission tree and relevant code.
2. Kept notebook-reported metric values explicitly attributed to the notebook and not independently reproduced.
3. Separated LOZO transfer scores from other benchmark scores measured under different protocols.
4. Corrected “ground truth” wording to “WorldCover reference labels” and avoided field-accuracy claims.
5. Inspected API fallback behavior and disclosed that some point responses may not be site-specific.
6. Distinguished prototype rule parameters from legal requirements and corrected statutory citations in the companion materials.
7. Identified generated deck images as illustrative rather than geographic or model evidence.

These are documentary checks, not a substitute for rerunning the training pipeline, field validation, legal advice, or operational testing. Prior to submission, the candidate should personally verify each citation, generated artifact, and course-specific disclosure requirement.

## 7. Data, privacy, and governance

The packaged analysis concerns public Earth-observation, terrain, and rainfall data. The package does not document ingestion of personal data. The optional external LLM pathway may transmit request context to third-party providers if enabled; any deployment must review provider terms, minimize payloads, protect credentials, and obtain institutional approval as applicable. This disclosure does not independently certify privacy or licensing compliance.

## 8. Declaration

I acknowledge that AI systems assisted with parts of the code-development workflow and, for this refinement, with auditing and drafting documentation and presentation materials. I take responsibility for reviewing the final submission, identifying AI assistance in accordance with course policy, and ensuring that claims and citations are accurate to the best of my knowledge.

**Candidate:** Valerie Jerono  
**Institution:** Strathmore University / iLabAfrica  
**Date:** 20 November 2026  
**Signature:** To be completed by the candidate if required by course policy.
