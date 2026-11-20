# DSA 8401: Applied Machine Learning

**Course Outline & Syllabus**
Master in Data Science and Analytics, Strathmore University (iLabAfrica Centre)
**Semester start:** Monday, 27 July 2026

---

## Unit Information

| | |
|---|---|
| **Unit Title & Code** | DSA 8401: Applied Machine Learning |
| **Lecturers** | Javier Serrano & Emmanuel Bundi |
| **Prerequisite** | Data Mining |
| **Contact Hours** | 3 hours weekly (1.5 h lecture + 1.5 h practical lab), Mondays |
| **Office Hours** | [To be confirmed] |
| **Email** | jgarcia@strathmore.edu, ebundi@strathmore.edu |

---

## 1. Course Aim

This course provides students with a comprehensive understanding of the core concepts, algorithms, and methodologies of machine learning, enabling them to effectively analyse data, solve real-world problems, and build predictive models. Furthermore, this modernised course bridges the gap between theoretical model training and production-grade deployment, equipping students with the MLOps and Generative AI skills required to build, deploy, and monitor end-to-end machine learning systems in industry environments.

Practical work is anchored in the African technological and financial ecosystem (fintech solutions addressing under-banking, trade-corridor underwriting, and medical data modelling) so that graduates can apply these systems directly in regional industry contexts.

## 2. Course Learning Outcomes

By the end of this course, students will be able to:

1. **Understand** the intuition and formal foundations of modern machine learning and deep learning methods, including how, why, and when they work.
2. **Apply** machine learning techniques to solve practical, real-world problems using rigorous data engineering and feature extraction.
3. **Build, containerize, and deploy** ML models as scalable APIs using modern MLOps practices (Docker, FastAPI, CI/CD).
4. **Implement and fine-tune** modern Generative AI architectures, including Transformers, Large Language Models (LLMs), and Retrieval-Augmented Generation (RAG).
5. **Monitor** deployed models for data and concept drift, ensuring long-term reliability and fairness in production.

## 3. Course Content Overview

1. Fundamentals of ML & end-to-end pipelines
2. Advanced data pre-processing & feature engineering
3. Supervised learning (classical & ensemble)
4. Unsupervised learning & dimensionality reduction
5. Deep learning (FNN, CNN, RNN/Transformers)
6. Modern Generative AI (LLMs, RAG)
7. MLOps: model serving, containerization, and monitoring

**Course arc:** Foundations & Pipelines → Feature Eng. & Evaluation → Ensembles & Tuning → Unsupervised & Dim. Red. → Deep Learning (CNN/RNN) → GenAI: LLMs & RAG → MLOps: Serve & Monitor → Capstone: Deploy

## 4. Weekly Schedule & Activities

Each session runs three hours: a 1.5-hour lecture (slide deck) followed by a 1.5-hour practical session (Jupyter/Python notebook).

| Wk | Date | Topic & Intended Outcomes | Activities / Labs |
|---|---|---|---|
| 1 | 27 Jul | **Course Presentation & End-to-End ML Fundamentals**: understand the ML lifecycle; set up professional dev environments. | Setup Git, VS Code, virtual envs. Build a baseline ML pipeline from scratch. |
| 2 | 03 Aug | **Advanced Data Prep & Feature Engineering**: handle messy real-world data; prevent data leakage. | Advanced feature engineering, encoding strategies, handling missing data. |
| 3 | 10 Aug | **Classical ML & Rigorous Evaluation**: master cross-validation and advanced metrics. | Imbalanced data (SMOTE), ROC-AUC, Precision-Recall curves. |
| 4 | 17 Aug | **Ensemble Methods & Hyperparameter Tuning**: optimise tree-based models. | Random Forests, XGBoost/LightGBM. Automated tuning with Optuna. |
| 5 | 24 Aug | **Unsupervised Learning & Dimensionality Reduction**: discover hidden patterns; reduce feature-space complexity. | Customer segmentation (K-Means, DBSCAN); PCA/t-SNE visualisation. |
| 6 | 31 Aug | **Deep Learning Fundamentals**: transition from Scikit-Learn to DL frameworks. | Build and train MLPs using PyTorch/TensorFlow. |
| 7 | 07 Sep | **Computer Vision with CNNs**: apply transfer learning to custom image classification. | Fine-tuning pre-trained ResNet/EfficientNet architectures. |
| 8 | 14 Sep | **Mid-Semester Break**: Capstone project proposals & dataset approvals due. | No class. |
| 9 | 21 Sep | **Sequence Models & Time Series**: forecast temporal data with advanced sequence models. | LSTMs/GRUs for time series; intro to Temporal Fusion Transformers. |
| 10 | 28 Sep | **Modern GenAI: Transformers & LLMs**: understand and apply modern Generative AI architectures. | HuggingFace ecosystem, fine-tuning LLMs, building RAG pipelines. |
| 11 | 05 Oct | **MLOps I: Model Packaging & Serving**: wrap models in production-ready APIs; containerize. | Build a FastAPI application and containerize it with Docker. |
| 12 | 12 Oct | **MLOps II: Experiment Tracking & CI/CD**: automate model testing; track experiments. | Integrate MLflow; build a GitHub Actions CI/CD pipeline. |
| 13 | 19 Oct | **Model Monitoring, Drift & System Design**: detect degradation; design for scalability and ethics. | Data/concept drift detection (Evidently AI); ML system design principles. |
| 14 | 26 Oct | **Capstone Project: Deploy & Test**: deploy end-to-end systems; conduct peer reviews. | "War room" session: cloud deployment, load testing, peer debugging. |
| 15 | 02 Nov | **Capstone Project Presentations**: communicate technical architecture and business impact. | Live demos, architecture review, final presentations. |

## 5. Capstone Project: Localised Themes

The capstone must deliver an end-to-end ML system: **data pipeline → model training → API deployment (Docker/FastAPI) → frontend**. Projects default to complex, messy, real-world data. Suggested tracks, anchored in the African tech and fintech ecosystem:

- **Fintech / under-banking:** alternative-data credit scoring for thin-file applicants using mobile-money (M-Pesa-style) transaction logs.
- **Trade-corridor underwriting:** risk modelling for cross-border SME lending along East African trade corridors.
- **Health informatics:** treatment-policy modelling with Markov decision processes and reinforcement learning on clinical pathway data.
- **Fraud & AML:** streaming anomaly detection on payment rails with concept-drift monitoring.

Proposals and dataset approvals are due at the mid-semester break (Week 8).

## 6. Academic Assessment

> **Note:** the traditional written final examination has been replaced with a practical midterm and a heavily weighted capstone project, to align with industry hiring standards.

| Assessment Type | Weight | Description |
|---|---|---|
| Lab Assignments & Quizzes | 20% | Bi-weekly practical labs (e.g. Dockerizing a model, building a RAG pipeline). |
| Midterm Exam | 20% | A 2/3-hour exam focusing on data processing, model training, and evaluation. |
| Capstone Project, Participation & Peer Review | 60% | End-to-end ML system: data pipeline, model training, API deployment (Docker/FastAPI), and a frontend. Includes active class participation and constructive peer reviews during capstone testing. |
| **Total** | **100%** | |

## 7. Course Delivery Methodology

- Interactive lectures (theory & concepts): 1.5 hours weekly, slide-deck driven.
- Hands-on practical labs (code & implementation): 1.5 hours weekly, Jupyter/Python notebooks.
- Code reviews and peer programming.
- Comprehensive end-to-end capstone project.

## 8. Course Materials

**Primary Textbook**

- A. Géron. *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, Third Edition, 2022.

**Supplementary Materials (Provided via Virtual Campus)**

- C. Bishop. *Deep Learning: Foundations and Concepts*, 2024.
- C. Huyen. *Designing Machine Learning Systems* (for MLOps and system design).
- HuggingFace official documentation and courses.
- Selected research papers on Transformers, LLMs, and RAG architectures.

## 9. Course Policies & Guidelines

**Classes**

1. Punctuality is fundamental.
2. Active participation in class discussions and lab sessions is encouraged.

**Assignments and Course Work**

1. Plagiarism is a serious offence. If detected in any form in course work and assignments, the following will apply:
   - **a.** In partial or non-serious cases (such as not citing whole word-for-word quotes), half the total possible marks of the assignment are duly struck off.
   - **b.** In serious cases (such as whole duplication of a paper), a zero policy will apply, i.e. all offending assignments will be awarded a mark of zero.

   *Note: the level of seriousness referred to above is at the discretion of the lecturer. Appeals are possible through the relevant channels.*
2. Referencing: APA system.
3. Notwithstanding the above, collaboration in course work is certainly encouraged as this promotes team spirit and group synergy, provided originality is preserved.

## 10. Communication Channels

1. **Virtual Campus:** all course materials, announcements, and assignment submissions are handled here.
2. **Institutional email:** for official communication and queries.

---

**First session:** Monday, 27 July 2026. Please bring a laptop with Git installed and a working Python environment.