# FaceTwin Research Methodology

## 1. Formal Research Questions

- **RQ1 (Facial Kinematic Discriminability):** Can MediaPipe 478-point 3D facial landmarks and derived biomechanical features (bilateral asymmetry index, EAR blink rate, mouth range-of-motion) reliably differentiate hypomimia and facial droop patterns from healthy baselines?
- **RQ2 (Multimodal Synergy):** Does the late fusion of facial kinematic features with structured clinical metrics (MDS-UPDRS Part III scores, disease duration, levodopa equivalent daily dose) statistically outperform either unimodal stream?
- **RQ3 (Temporal Dynamics):** Does sequence modeling via Bidirectional LSTM with temporal attention capture movement velocity and hypomimia onset dynamics more accurately than static window aggregations?
- **RQ4 (State Estimation & Tracking):** Does Kalman filter state estimation provide lower variance and clinically meaningful trajectory forecasting compared to discrete unconstrained longitudinal predictions?
- **RQ5 (Domain Transferability):** Can the feature extraction and digital twin architecture transfer between Parkinson's hypomimia and acute/post-stroke facial asymmetry monitoring without architectural restructuring?

---

## 2. Experimental Design & Validation Protocols

### 2.1 Subject-Independent Validation
To avoid data leakage across frames and video sessions of the same patient:
- **Partitioning Strategy:** Grouped Stratified $K$-Fold cross-validation ($K=5$) and Leave-One-Subject-Out (LOSO) cross-validation where applicable.
- **Data Leakage Mitigation:** All scalers, feature selectors, and imputation pipelines are fit strictly on training splits before transforming validation/test splits.

### 2.2 Performance Metrics
1. **Classification Tasks (Healthy vs. Impaired / Severity Grading):**
   - Area Under the Receiver Operating Characteristic curve (AUC-ROC)
   - Macro-averaged F1-Score
   - Sensitivity (Recall) and Specificity
   - Cohen's Kappa ($\kappa$)
2. **Regression & Continuous State Tracking (MDS-UPDRS Motor Prediction & Progression):**
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - Pearson ($r$) and Spearman ($\rho$) correlation coefficients
3. **Kalman State Estimator Evaluation:**
   - Innovation sequence whiteness testing
   - State estimation error covariance convergence
   - 95% Confidence Interval coverage bounds

### 2.3 Ablation Studies
| Experiment ID | Video Kinematics | Bi-LSTM Stream | Clinical Scores | Kalman State Estimator |
|---|---|---|---|---|
| EXP-01 | Baseline (Static ML) | No | No | No |
| EXP-02 | Static ML + LSTM | Yes | No | No |
| EXP-03 | Clinical Only | No | Yes | No |
| EXP-04 | Full Dual-Stream | Yes | Yes | No |
| EXP-05 (Proposed) | Full Dual-Stream | Yes | Yes | Yes (Digital Twin) |
