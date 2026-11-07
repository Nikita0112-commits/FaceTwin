# FaceTwin: Multimodal Human Digital Twin for Parkinson's Disease & Stroke Monitoring

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## M.Tech Major Research Project

**Project Title:** FaceTwin: An AI-Enabled Multimodal Human Digital Twin Integrating Facial Video Analytics with Clinical Data for Parkinson's Disease and Stroke Monitoring

---

## 1. Research Objectives & Clinical Disclaimer

FaceTwin investigates an engineering framework integrating:
- **Computer Vision:** Real-time 478-point 3D facial landmark mesh extraction using Google MediaPipe.
- **Biomechanical Facial Feature Engineering:** Asymmetry indices, Action Unit (AU) proxies, Eye Aspect Ratio (EAR) blink kinetics, and mouth range-of-motion (ROM).
- **Temporal Deep Learning:** Bidirectional LSTM with self-attention over sequential facial movement dynamics.
- **Clinical Data Fusion:** Late fusion of facial kinematic embeddings with structured MDS-UPDRS clinical scores and patient demographics.
- **Digital Twin State Estimation:** Linear Kalman Filter data assimilation providing optimal continuous state tracking with uncertainty bounds.

> **CRITICAL CLINICAL NOTICE:** FaceTwin is an academic research software platform designed for objective longitudinal movement monitoring and research. It does **not** provide independent automated medical diagnoses for Parkinson's disease or acute stroke.

---

## 2. Core Architecture

```
                    ┌─────────────────────────┐
                    │    Daily Facial Video   │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Video Preprocessing   │
                    │   & Normalization       │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   MediaPipe 478 Mesh    │
                    │   & Iris Extraction     │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  Biomechanical Features │
                    │ (Asymmetry, Blink, ROM) │
                    └─────┬─────────────┬─────┘
                          │             │
        ┌─────────────────┴─┐         ┌─┴────────────────┐
        │  Static Features  │         │ Sequential Window│
        └─────────┬─────────┘         └────────┬─────────┘
                  │                            │
                  ▼                            ▼
        ┌───────────────────┐        ┌───────────────────┐
        │  Classical ML     │        │  Bi-LSTM Temporal │
        │  (RF, XGBoost)    │        │  Model + Attention│
        └─────────┬─────────┘        └─────────┬─────────┘
                  │                            │
                  └─────────────┬──────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐   ┌───────────────────────┐
                    │  Late Multimodal Fusion │◄──┤ Structured Clinical   │
                    │                         │   │ Data (MDS-UPDRS, etc.)│
                    └───────────┬─────────────┘   └───────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Kalman Filter State   │
                    │   Estimator Engine      │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Digital Twin State    │
                    │   & Progression Traj.   │
                    └─────────────────────────┘
```

---

## 3. Directory Layout

```
facetwin/
├── configs/            # Configuration management (YAML)
├── data/               # Raw, processed, feature, and synthetic datasets
├── docs/               # Architecture, methodology, and dataset provenance
├── experiments/        # Checkpoints and evaluation results
├── notebooks/          # Exploratory and experimental notebooks
├── src/facetwin/       # Modular core library
│   ├── core/           # Configuration, exceptions, structured logging
│   ├── data/           # Patient models and clinical data ingestion
│   ├── preprocessing/  # Video stabilization, frame extraction, MediaPipe 478
│   ├── features/       # Asymmetry, blink rate, movement dynamics
│   ├── models/         # Classical ML, LSTM, and multimodal fusion
│   ├── digital_twin/   # Kalman filter data assimilation and progression
│   └── api/            # FastAPI service layer
├── tests/              # Pytest unit and integration test suite
└── scripts/            # Dataset acquisition and pipeline runners
```

---

## 4. Setup & Installation

### Requirements
- Python 3.10 or 3.11 (Python 3.11 recommended)
- Virtual environment tool (`venv` or `conda`)

```bash
# 1. Create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install FaceTwin in editable development mode
pip install -e .
```

### Running Tests
```bash
pytest tests/ -v
```

---

## 5. Development Phases

- **Phase 0:** Environment setup, directory hierarchy, configuration & core framework *(Current)*
- **Phase 1:** Clinical data ingestion & dataset exploration
- **Phase 2:** Video preprocessing & 478-landmark extraction
- **Phase 3:** Biomechanical feature engineering
- **Phase 4:** Baseline machine learning models
- **Phase 5:** LSTM temporal sequence modeling
- **Phase 6:** Multimodal fusion & ablation studies
- **Phase 7:** Kalman filter digital twin state estimation
- **Phase 8:** FastAPI service & visualization dashboard
- **Phase 9:** Scientific evaluation, benchmarking & documentation
- **Phase 10:** Final thesis integration
