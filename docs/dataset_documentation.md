# FaceTwin Dataset Documentation & Acquisition Strategy

## 1. Public Clinical & Video Datasets

### 1.1 PPMI (Parkinson's Progression Markers Initiative)
- **Source:** [PPMI Information Portal](https://www.ppmi-info.org/)
- **Modalities:** Longitudinal clinical scores (MDS-UPDRS Parts I, II, III, IV), MoCA, Hoehn & Yahr staging, patient demographics, and genetic markers.
- **Access Protocol:** Qualified academic researchers can register and execute a standardized Data Use Agreement (DUA).
- **Project Role:** Provides the clinical ground truth and structured patient tabular feature vectors for multimodal fusion and longitudinal trajectory modeling.

### 1.2 YouTubePD (Benchmark for Parkinson's Disease Analysis)
- **Source:** UIUC / NeurIPS 2023 Datasets & Benchmarks Track
- **Modalities:** In-the-wild video recordings with clinician annotations for speech, facial regions, and motor symptom severity (scale 0–5).
- **Access Protocol:** Public GitHub benchmark repository (`prepare_data.py`).
- **Project Role:** Naturalistic video evaluation for facial masking (hypomimia) detection in realistic lighting conditions.

### 1.3 TULIP (3D Precision Assessment of Parkinson's Disease)
- **Source:** Duke University / CVPR 2024 (Zenodo)
- **Modalities:** Multi-camera synchronized RGB video of Parkinson's patients performing standardized MDS-UPDRS Part III clinical motor exams.
- **Access Protocol:** Restricted research access via application form and institutional review.
- **Project Role:** High-resolution validation of kinematic movement parameters during clinical motor tasks.

### 1.4 Facial Palsy & Stroke Asymmetry Datasets
- **YouTube Facial Palsy (YFP):** Clinically annotated video clips of unilateral and bilateral facial weakness.
- **Roboflow & Kaggle Palsy Benchmarks:** Annotated images and video frames labeled for eye weakness, facial droop, and asymmetric smile dynamics.
- **Project Role:** Evaluates the generalization of the bilateral asymmetry index and range-of-motion pipeline for stroke monitoring.

### 1.5 Facial Action Coding & Expression Benchmarks (Baselines)
- **Cohn-Kanade (CK+):** 593 FACS AU-annotated sequences from 123 subjects.
- **DISFA (Denver Intensity of Spontaneous Facial Action):** Frame-by-frame 12-AU intensity ratings across 130,000 video frames.
- **Project Role:** Verification and calibration of geometric Action Unit (AU) proxy metrics.

---

## 2. Synthetic Data Strategy & Integrity Standards

To enable reproducible unit testing, continuous integration, and pipeline validation prior to external DUA authorization:
1. **Explicit Labeling:** All synthetic datasets reside exclusively in `data/synthetic/` and carry explicit metadata tags:
   ```json
   {
     "is_synthetic": true,
     "generation_method": "parametric_gaussian_mixture",
     "intended_use": "ci_pipeline_verification"
   }
   ```
2. **Zero Fabrication Policy:** Synthetic datasets are used exclusively for pipeline verification, API stress-testing, and software engineering unit tests. No academic accuracy claims, metrics, or thesis conclusions will be derived from synthetic data.
