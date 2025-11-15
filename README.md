# Sepsis Early Detection ML System

Sepsis is a life-threatening condition where **every hour of delay increases mortality risk**.  
This project builds a **machine learning pipeline for early sepsis risk detection** using structured clinical data, with a strong focus on:

- **Class imbalance handling** (balanced datasets via SMOTE/SMOTENC)
- **Clinically realistic features** (vitals, labs, comorbidities, outcomes)
- **Clear model performance metrics** (ROC-AUC, PR-AUC, confusion matrix)
- **Governance-friendly structure** (reproducible training + metrics artifacts)

---

## 🔍 Problem Statement

Hospitals struggle to:

- Detect high-risk sepsis patients early enough
- Prioritize limited ICU resources
- Reduce **sepsis-related mortality** and **30-day readmission rates**

This project frames it as a **supervised ML problem**:

> Given a patient's structured clinical features, predict the probability that the patient is septic (`is_sepsis = 1`) early enough for clinicians to act.

---

## 📦 Data Overview

This project uses a **synthetic, PII-free clinical dataset** that mimics real hospital data while avoiding HIPAA issues.

**Source file:**

- `data/synthetic_clinical_dataset.csv`  
  – 10,000 patients (scalable to millions)  
  – Fully de-identified, synthetic

**Example columns:**

- **Demographics:** `age`, `sex`, `bmi`
- **Vitals / Labs:** `systolic_bp`, `diastolic_bp`, `glucose`, `cholesterol`, `creatinine`
- **Comorbidities:** `diabetes`, `hypertension`
- **Diagnosis:** `diagnosis` ∈ {`Normal`, `Pneumonia`, `Heart Failure`, `Sepsis`}
- **Outcomes:**  
  - `mortality` (0/1)  
  - `readmission_30d` (0/1)

To address **severe class imbalance**, we generate three balanced datasets:

- `data/balanced_sepsis.csv`  
  - Target: `is_sepsis` (1 if diagnosis == `Sepsis`, else 0)  
  - Features: numeric + one-hot–encoded clinical features  
  - Balanced via **SMOTE**: 50% sepsis / 50% non-sepsis

- `data/balanced_mortality.csv`  
  - Target: `mortality`  
  - Raw clinical format (categorical + numeric)  
  - Balanced via **SMOTENC**

- `data/balanced_readmission.csv`  
  - Target: `readmission_30d`  
  - Raw clinical format  
  - Balanced via **SMOTENC**

> ⚠️ These datasets are **synthetic** and intended for portfolio, research, and pipeline demonstration.  
> No real patient data is used; no PHI/PII is present.

---

## 🧠 Model & Training Pipeline (Sepsis)

Current focus: **Sepsis Early Detection** using `balanced_sepsis.csv`.

**Target:**

- `is_sepsis` ∈ {0, 1}

**Model:**

- `GradientBoostingClassifier` (sklearn)
- Trained on a **perfectly balanced** dataset via SMOTE

**Pipeline:**

1. Load `data/balanced_sepsis.csv`
2. Split into train/validation (stratified, 80/20)
3. Apply `StandardScaler` to numeric feature matrix
4. Train `GradientBoostingClassifier` with:
   - `n_estimators = 200`
   - `learning_rate = 0.05`
   - `max_depth = 3`
   - `random_state = 42`
5. Evaluate on validation set:
   - ROC-AUC
   - PR-AUC (precision–recall, important for rare events)
   - Confusion matrix
   - Full classification report
6. Persist artifacts:
   - Model → `models/sepsis_gb_model.joblib`
   - Metrics → `reports/sepsis_metrics.json`

---

## 🚀 How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/<your-username>/sepsis-early-detection-ml.git
cd sepsis-early-detection-ml
```

### 2. Create Virtual Environment & Install Dependencies
```python3 -m venv .venv
source .venv/bin/activate  # Mac / Linux
# .venv\Scripts\activate   # Windows (PowerShell)
pip install --upgrade pip
pip install -r requirements.txt
```
### 3. Train the Sepsis Model
```python train_sepsis.py```

You should see logs like:
```📂 Loading data from data/balanced_sepsis.csv ...
✅ Loaded 18716 rows, XX features.
   Class balance: {0: 9358, 1: 9358}
🧪 Train size: 14972, Val size: 3744
🚀 Training GradientBoostingClassifier...
✅ Training complete.
📈 ROC-AUC: 0.9xxx
📈 PR-AUC : 0.9xxx
Confusion matrix (rows=true, cols=pred):
[[TN FP]
 [FN TP]]
💾 Model saved to: models/sepsis_gb_model.joblib
📝 Metrics saved to: reports/sepsis_metrics.json
```
### 📊 Metrics & Outputs

Key artifacts after training:
* models/sepsis_gb_model.joblib\
→ Serialized scikit-learn pipeline (scaler + classifier)
* reports/sepsis_metrics.json\
→ Contains:
  * roc_auc
  * pr_auc
  * confusion_matrix
  * classification_report

You can load and inspect them:
```
import json
from pathlib import Path

with open(Path("reports/sepsis_metrics.json")) as f:
    metrics = json.load(f)

print("ROC-AUC:", metrics["roc_auc"])
print("PR-AUC:", metrics["pr_auc"])
```
### 🧱 Project Structure
```sepsis_early_detection/
├── data/
│   ├── synthetic_clinical_dataset.csv        # source synthetic clinical data
│   ├── balanced_sepsis.csv                   # SMOTE-balanced sepsis dataset
│   ├── balanced_mortality.csv                # SMOTENC-balanced mortality data
│   └── balanced_readmission.csv              # SMOTENC-balanced readmission data
├── models/
│   └── sepsis_gb_model.joblib                # trained model artifact
├── reports/
│   └── sepsis_metrics.json                   # ROC-AUC, PR-AUC, confusion matrix, report
├── train_sepsis.py                           # training script
├── requirements.txt
└── README.md
```
### Governance, Risk & Compliance Notes
Although this project uses synthetic, non-PHI data, it is designed in a way that could be safely extended to real settings:
* No PHI/PII:  
No names, MRNs, dates of birth, addresses, or identifiers.
* Data Minimization & Explainability:  
Uses structured features (vitals, labs, comorbidities) that can be clinically interpreted and audited.
* Imbalance Handling:  
Uses SMOTE/SMOTENC on training data only, clearly documented.
* Avoids naïve resampling on raw hospital data without governance.
### Extensible Compliance Hooks:
In a real deployment, you’d add:
* Access control & audit trails
* PHI scrubbing before logging
* Model monitoring (drift, performance degradation)
* Bias checks across age/sex/comorbidity slices
### Roadmap
Planned improvements:
* ✅ Add mortality & readmission prediction models using balanced_mortality.csv and balanced_readmission.csv
* ✅ Add calibration plots (reliability curves, Brier score)
* 🔜 Add a FastAPI microservice exposing /predict_sepsis_risk
* 🔜 Add SHAP-based explainability per prediction (feature attributions)
* 🔜 Add basic drift monitoring and alerting hooks (e.g., batch notebook)
### 💼 Why This Project Matters (Portfolio Angle)
This project demonstrates that you can:
* Take a messy clinical prediction problem (sepsis)
* Design a balanced, drift-aware training dataset
* Train and evaluate a solid ML model with meaningful metrics
* Package it into a reproducible pipeline with saved artifacts
* Reason about governance and compliance, not just code
