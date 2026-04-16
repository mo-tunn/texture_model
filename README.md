# Texture Analysis Pipeline - Hybrid Documentation (v3 & v5)

This documentation explains the training processes of two independent models (**v3** and **v5**) that evaluate skin texture roughness on a scale of 0-100, and the **Hybrid Decision Mechanism** created by combining these two models in the test environment (`local_tester.py`).

---

## 1. General Architecture and Purpose

The goal is to objectively score skin texture (acne, scars, large pores, fine lines). For this purpose, instead of a single omnipotent model, two models with different strengths were trained, and a hybrid system was established:

1.  **v3 Model (Good Skin Expert):** A safe model trained only with high-quality general face data (FFHQ), making predictions within a relatively narrow score range. (Reference: `Texture_Analysis_Colab_v3.ipynb`)
2.  **v5 Model (Problematic Skin Expert):** A model trained by adding thousands of specific acne and scar data in addition to FFHQ data, instantly penalizing flaws. (Reference: `Texture_Analysis_Colab_v5.ipynb`)

During the test phase (inference), the scores of these two models are normalized and combined using specific threshold and weighting rules.

---

## 2. v3 Model: Balanced Training (Good Skin Oriented)

The v3 model focuses on the FFHQ dataset, which consists of high-resolution, general face photos.

### Features:
* **Dataset:** ~10,000 FFHQ-256 images.
* **Focus:** Understanding the general structure of the skin, identifying smooth skin as "smooth".
* **Labeling Strategy:** Automated scoring via formula using 8 different mathematical features (Fourier Entropy, GLCM, Ra, Rq, etc.) without human intervention.
* **Balancing:** Creating a balanced distribution by `oversampling` rare score ranges.
* **Model Features:** EfficientNetB0 based, Huber Loss. Target MAE (Mean Absolute Error) `5 - 8`.
* **Characteristic:** Avoids extreme scoring (very good or very bad), the scoring range is concentrated between **20 - 70**. It is passive.

---

## 3. v5 Model: Semi-Supervised and Human Calibration (Problematic Skin Oriented)

The v5 model is designed to correctly penalize specific problematic skins (severe acne, scars) where v3 falls short. It uses a Semi-Supervised Learning architecture.

### Features:
* **Dataset:** * `FFHQ` (Normal Faces) ~5,000 images
    * `Acne` (Acne-prone Faces) ~1,000 images
    * `Scar` (Scarred Faces) ~500 images
* **Labeling Strategy (Human Calibration):** 1. 300 random images (100 from each category) were selected from the system.
    2. Manually scored by a human between 0-100 (0: Very bad, 100: Perfect).
    3. An **XGBoost Regressor** (Calibration Model) was trained by mapping the manual labels to 12 mathematical features.
* **Label Propagation:** The XGBoost model applied what it learned from the 300 manually scored images to the remaining 6,000+ images, labeling the entire dataset.
* **Main Model Training:** The 6,000+ images labeled with XGBoost decisions were retrained using the EfficientNetB0 architecture.
* **Characteristic:** It immediately pulls the score down to the 5-10 range for acne, being quite **aggressive** and a flaw hunter. The scoring range is predominantly between **5 - 40**.

---

## 4. Mathematical Feature Extraction

Signal processing tools that measure physical roughness were used while labeling the training sets (directly in V3, to support calibration in V5):

1.  **Fourier Entropy & Spatial Entropy**: Pixel complexity.
2.  **DWT Energy (Wavelet)**: Energy level in fine details.
3.  **Ra, Rq (ISO 4287)**: Physical surface roughness standards.
4.  **GLCM (Contrast, Homogeneity, etc.)**: Harshness in color transitions between neighboring pixels.
5.  **MLC, Laplacian Variance, Gradient Magnitude**: Edge/Corner detection.

---

## 5. Inference: v3 and v5 Hybrid Extraction (`local_tester.py`)

The system running on the end-user side runs the v3 and v5 models *together* on a live camera or image. It reconciles the two different decisions obtained with a mathematical "Weighted Decision Mechanism" to give a final skin score.

### Phase 1: Normalization

The raw scores produced due to model characteristics are not in the same space. Therefore, both are standardized to a 0-100 scale according to determined distribution characteristics.

* `v3_norm = (v3_raw - 20) / (70 - 20) * 100` (The range is expanded to `20-70` for a smooth increase).
* `v5_norm = (v5_raw - 5) / (40 - 5) * 100`

### Phase 2: Weighted Decision Rules

A `Threshold = 50.0` is accepted during the decision phase (Hybrid Score Calculation). Above 50 means good, below means problematic skin. The algorithm looks at the `norm` equivalent of the two scores.

1.  **Exception Protection (Preventing v5 Inflation):** If `v5_raw < 28` (Faulty, Bad) **AND** `v3_raw >= 55` (Great, Clean), the system falls into a contradiction. In this contradiction, to prevent v5's norm score from mathematically peaking and artificially skewing the score, it is ordered not to exceed the *`v5_norm = min(v5_norm, 45.0)`* limit.
2.  **Perfection (Rule 1 - 90% v3):** If `v3_norm >= 80`, the skin is indisputably smooth. The system ignores v5's aggressiveness and gives 90% weight to v3's decision.
3.  **Common Good (Rule 2 - 80% v3):** If both models pass the threshold (`v3_norm >= 50` and `v5_norm >= 50`), the skin is good. 80% trust is placed in v3, the good skin expert.
4.  **Common Bad (Rule 3 - 80% v5):** If neither model passes the threshold (`v3_norm < 50` and `v5_norm < 50`), the skin is problematic. 80% trust is placed in v5, the problematic skin expert.
5.  **Pessimistic Contradiction (Rule 4 - 80% v5):** If v3 liked the skin but v5 didn't (`v5_norm < 50`). Because v5 is an aggressive model, there is an 80% chance it found a microscopic scar or acne somewhere. To avoid missing the disease, 80% trust is placed in v5, the pessimistic expert.
6.  **Optimistic Contradiction (Rule 5 - 50% Indecision):** If v5 says "very good", but v3, which is normally passive, says "no, bad" (`v3_norm < 50`), indecision prevails. In this case, the reliability of the two models is taken as half-and-half `%50` and averaged. 

---

## 6. Conclusion and Distribution Output

| Score Range | UI Interpretation | Color Code |
| :--- | :--- | :--- |
| **70 - 100** | Smooth skin | Green |
| **50 - 69** | Normal texture | Light Green |
| **35 - 49** | Moderate texture | Orange |
| **0 - 34** | Prominent texture | Red |

Thanks to the flow above, the v3 model, which only works on normal faces, and the v5 model, which only detects damaged skin, are balanced through a "Committee (Ensemble)" style of work.
