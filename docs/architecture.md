# FaceTwin System Architecture

## 1. Overview & High-Level Design

FaceTwin is an experimental multimodal human digital twin platform. The design decouples:
1. **Perception Layer:** Video acquisition, face detection, 478-point mesh extraction (MediaPipe Tasks API).
2. **Kinematic Feature Extraction:** Computation of geometric asymmetry, eye aspect ratio (EAR) blink metrics, and mouth range-of-motion (ROM).
3. **Temporal Dynamics:** Sequence windowing feeding a Bidirectional LSTM network with attention.
4. **Multimodal Data Assimilation:** Combining kinematic predictions with structured clinical markers (e.g. MDS-UPDRS motor scores).
5. **Digital Twin Engine:** Kalman filtering for optimal recursive state tracking and progression velocity estimation.

---

## 2. Module Boundaries & Data Contracts

### 2.1 Video Preprocessing & Landmark Extraction (`facetwin.preprocessing`)
- **Input:** Video stream or file path (`.mp4`, `.avi`, `.mov`).
- **Processing:** Resizing to 640x480, frame rate normalization to 30 FPS, CLAHE contrast enhancement, MediaPipe 478 landmark detection.
- **Output:** Frame-indexed landmark coordinates $\mathbf{L} \in \mathbb{R}^{T \times 478 \times 3}$ (where $x, y, z$ coordinates are normalized to $[0, 1]$ or relative bounding box coordinates).

### 2.2 Feature Engineering (`facetwin.features`)
- **Facial Asymmetry Index (FAI):** Normalized distance ratios between left and right facial bilaterally symmetric pairs relative to the central facial midline (nose bridge landmark 168 to chin landmark 152).
- **Blink Dynamics:** Eye Aspect Ratio (EAR):
  $$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$
- **Mouth Range of Motion (ROM):** Vertical aperture over inter-commissure distance.
- **Output:** Feature matrix $\mathbf{X}_{\text{kinematic}} \in \mathbb{R}^{T \times D}$ where $D$ is the feature dimension.

### 2.3 Dual-Stream Modeling & Fusion (`facetwin.models`)
- **Stream 1 (Static Engineered ML):** Summary statistics (mean, std, min, max, skewness) of kinematics fed into Random Forest / Support Vector Classifier / XGBoost.
- **Stream 2 (Temporal Sequence):** Frame-level sequence $\mathbf{X}_{\text{kinematic}}[t:t+W]$ fed into a 2-layer Bi-LSTM with self-attention.
- **Late Multimodal Fusion:** Probability-level or representation-level concatenation with clinical vector $\mathbf{c}_{\text{clinical}}$:
  $$\hat{y}_{\text{fused}} = \sigma(\mathbf{W}_v \mathbf{h}_{\text{video}} + \mathbf{W}_c \mathbf{c}_{\text{clinical}} + b)$$

### 2.4 Digital Twin Engine (`facetwin.digital_twin`)
- **State Vector:**
  $$\mathbf{x}_k = [s_{\text{hypomimia}}, s_{\text{asymmetry}}, s_{\text{blink}}, s_{\text{motor}}, \dot{s}_{\text{hypomimia}}, \dot{s}_{\text{asymmetry}}]^T$$
- **Linear Discrete State Space:**
  $$\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{w}_{k-1}, \quad \mathbf{w} \sim \mathcal{N}(0, \mathbf{Q})$$
  $$\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{v} \sim \mathcal{N}(0, \mathbf{R})$$
- **Output:** Smoothed digital twin trajectory with covariance $\mathbf{P}_k$, providing confidence intervals over progression.
