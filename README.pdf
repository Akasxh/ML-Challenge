# ML Challenge 2025 — Smart Product Pricing (Team **Apex**)

**Submission Date:** 13/10/2025

<div align="center">

| Name               |                      GitHub                      |                            LinkedIn                            | Department @ IIT Patna |
| :----------------- | :----------------------------------------------: | :------------------------------------------------------------: | :--------------------: |
| **S Akash**        |       [@Akasxh](https://github.com/Akasxh)       |  [LinkedIn](https://www.linkedin.com/in/s-akash-project-gia/)  |           EEE          |
| **Anirudh D Bhat** |      [@RudhaTR](https://github.com/RudhaTR)      |     [LinkedIn](https://www.linkedin.com/in/anirudh-d-bhat/)    |           CSE          |
| **Akash S**        | [@akash1764591](https://github.com/akash1764591) |   [LinkedIn](https://www.linkedin.com/in/akash-s-473781263/)   |           EEE          |
| **Ammar Ahmad**    |  [@ammarrahmad](https://github.com/ammarrahmad)  | [LinkedIn](https://www.linkedin.com/in/ammar-ahmad-5a8343251/) |         AI & DS        |

</div>

---

## 1) Executive Summary

We tackle **product price prediction from semi‑structured catalog text** as a **text regression** problem. After building a **hybrid data preparation pipeline** (Regex for fast parsing + a targeted small LLM for imputation), we fine‑tune **DeBERTa‑v3‑large** with a lightweight regression head.

* **Best validation SMAPE:** **42.22%** on a 7,500‑sample hold‑out split
* **Other metrics:** **MAE:** $9.44, **R²:** 0.29
* **Training size:** 67,499 samples; **epochs:** 10

The key insight is that **pragmatic feature engineering**—extracting value/unit/name via Regex and using an LLM *only where needed*—yields cleaner inputs at a fraction of the compute of fully LLM‑driven structuring.

---

## 2) Problem & Data

### 2.1 Task

Predict a continuous **price** from a textual **catalog_content** field (semi‑structured lines with optional bullets). Images were available but largely redundant with text for this challenge.

### 2.2 Dataset Fields

* `serial_id`, `price` (target), `image_link`, `catalog_content` (primary feature)

### 2.3 EDA Highlights

* **Right‑skewed target** → apply `log1p(price)` during training
* **Semi‑structured text** with (often) this pattern:

  1. Item name / short description
  2. Numeric value/quantity (e.g., `12`, `16.9`)
  3. Unit (e.g., `oz`, `count`)
  4. Longer description / bullets (optional)
* **Image information** mostly duplicated in text → **text‑only** approach favored for compute efficiency (we would have augmented with VLM when text was insufficient, but compute/time constraints prevented it)

---

## 3) Methodology

### 3.1 Data Preparation (Hybrid Regex + Targeted LLM)

1. **Regex extraction** of **item name**, **value**, **unit**, plus concatenation of remaining text as **description**.
2. **Targeted LLM imputation** for rows where Regex fails (fills `None` / `NA` gracefully).
3. **Sensible defaults** for residual nulls (e.g., `unit = "count"`, `description = "no description"`).
4. **Structured prompt string** fed to the model:

```
Category: [category] [SEP] description: [description] [SEP] Amount: [value] [unit]
```

5. Apply **log1p** transform to `price` for training; use **expm1** for inverse transform at inference.

### 3.2 Model Architecture

* **Backbone:** `microsoft/deberta-v3-large`
* **Head:** MLP regression head on top of final encoder representation (CLS/pooled)
* **Loss:** Huber during training (stable for heavy‑tailed targets)

<p align="center">
  <img src="https://github.com/user-attachments/assets/a7127382-e4b3-49b5-be4a-60573feca2eb" alt="pipeline" width="680" height="362" />
</p>

### 3.3 Training Setup (Key Hyperparameters)

* Max sequence length **160**
* Dropout **0.2**
* Encoder LR **2e‑5**; Head LR **1e‑3**
* Batch size **16**
* Epochs **10** (best run)

---

## 4) Results

We use **SMAPE** as the primary metric:

SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)

### 4.1 Final Hold‑out (7,500 samples)

* **SMAPE:** **42.22%**
* **MAE:** **$9.44**
* **R²:** **0.29**
* **Training epochs:** **10**, **Layers:** 512→256 MLP head, **Train Loss (Huber) @ epoch 10:** 0.0276

### 4.2 Selected Experiment Snapshots (Ablations)

**A) Head depth/epochs vs. performance (DeBERTa‑v3‑large):**

| Epochs | MLP Head (dims) |  MAE ($) |    R²    | SMAPE (%) |
| :----: | :-------------: | :------: | :------: | :-------: |
|    5   |    512 → 256    |   9.53   |   0.28   |   42.69   |
|    5   |    768 → 384    |   9.63   |   0.29   |   42.86   |
|    7   |    512 → 256    |   9.50   |   0.28   |   42.68   |
| **10** |  **512 → 256**  | **9.44** | **0.29** | **42.22** |

**B) Model family comparisons (same pipeline):**

| Approach                               | Validation SMAPE (%) |
| :------------------------------------- | :------------------: |
| **DeBERTa‑v3‑large + head (final)**    |       **42.22**      |
| DeBERTa‑v3‑base + head                 |         43.59        |
| DistilBERT + head (baseline)           |         45.00        |
| Sentence Embeddings + Neural Net       |         49.00        |
| Sentence Embeddings + XGBoost          |         52.00        |
| BERT Encodings + XGBoost               |        ~50.00        |
| LLM SFT (phi‑3‑mini‑instruct, 1 epoch) |     ~48.00 (slow)    |
| IFT (Qwen‑7B / phi‑3‑mini‑instruct)    |  Abandoned (compute) |

> **Takeaway:** Larger encoders with careful regularization and a small regression head consistently outperform static‑embedding approaches and quick SFT/IFT attempts within realistic compute budgets.

---

## 5) Discussion & Lessons

* **Data > Model Size (alone):** The Regex→LLM hybrid pipeline delivered cleaner, more consistent inputs **without** the cost of fully LLM‑structuring the entire dataset.
* **Target transformation matters:** `log1p` notably stabilized optimization for a heavy‑tailed price distribution.
* **Vision deprioritized intentionally:** For this dataset, **text captured most of the useful variance**; image features were not critical under compute constraints.

---

## 6) Reproducibility (High‑Level)

1. **Prepare data:** run parsing to extract `name/value/unit/description`; apply targeted LLM fills; form structured text field; compute `log1p(price)`.
2. **Train:** fine‑tune `microsoft/deberta-v3-large` with the hyperparameters above; use Huber loss.
3. **Evaluate:** compute SMAPE/MAE/R² on the 7,500‑sample hold‑out.
4. **Infer:** predict on new `catalog_content`; inverse‑transform with `expm1`.

> **Code & Scripts:**
> **Drive (full code & experiments):** `https://drive.google.com/file/d/1WwFyzrKUbCFXXAKv7RyQjrae3UFAftzB/view?usp=sharing`

---

## 7) Appendix

### A) Additional Experiment Notes (excerpts)

* **Epoch 7/7:** Train Huber Loss ≈ **0.0470** → Test SMAPE **42.68%** (MAE $9.50, R² 0.28)
* **Epoch 10/10:** Train Huber Loss **0.0276** → **Best** Test SMAPE **42.22%** (MAE $9.44, R² 0.29)

### B) Potential Next Steps

* Lightweight **category detection** and **“premium” heuristics** using small models/features to further de‑skew residuals.
* Targeted **vision augmentation** only when Regex+LLM confidence is low.
* **Quantization / LoRA** for faster ablation on larger backbones.
* Calibrated **price intervals** (e.g., conformal) for actionable uncertainty estimates.

---

**Contact:** See team table (GitHub / LinkedIn) for profiles.
