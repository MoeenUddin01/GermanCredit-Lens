# CLAUDE.md — Architecture, Standards & Automation Playbook
## Project: GermanCredit-Lens (Asymmetric Credit Risk Engine)

This document establishes the production engineering standards, architectural boundaries, and coding invariants for the `GermanCredit-Lens` system. Every line of code written by the AI Coding Agent must comply completely with this specification.

---

## 1. System Engineering Principles

### 1.1 Objective & Constraints
The objective is to engineer a highly modular machine learning pipeline centered around `XGBoost` and custom `scikit-learn` transformers utilizing the raw `german.data` text dataset. The model evaluates creditworthiness under a non-negotiable asymmetric business cost constraint:
* **True Label Mapping:** `1` = Good Credit Risk, `2` = Bad Credit Risk.
* **Asymmetric Cost Matrix:** Misclassifying a high-risk applicant ("Bad") as a safe applicant ("Good") carries an administrative penalty weight of **5**, whereas misclassifying a safe applicant as high-risk carries a penalty weight of **1**.
* **Optimization Driver:** Custom evaluation metrics must direct hyperparameter optimization toward minimizing total financial risk exposure, rather than purely maximizing overall accuracy.

### 1.2 State Persistence & Artifact Invariant
To ensure deterministic serving during future frontend application integration and inference phases, **no component may calculate its parameters on the fly during inference**. 
* Any module computing parameters on the training slice (e.g., categorical encoders, numerical scalers, missing-value indicators, spatial transformers) must implement the `scikit-learn` Estimator API (`fit`/`transform`/`fit_transform`).
* All fitted objects must be serialized and persisted into the root-level `/artifacts/` directory immediately following execution.

---

## 2. Directory Topology & System Boundaries

The project boundary matches the following verified structural mapping. Do not create unapproved structural components or drift from this pattern:

```text
.
├── dataset
│   ├── processed/              # Version-controlled, cached engine data tensors
│   └── raw
│       ├── german.data         # Source space-separated file containing text categories
│       └── german.data-numeric # Benchmark Strathclyde numeric vector variant
├── artifacts/                  # Production asset store (Serialized JSON schemas, .joblib scalers, encoders, model binaries)
├── notebooks
│   └── preprocessing.ipynb    # Local experimentation, discovery, and rapid prototyping
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── loader.py           # Structural IO, column naming, and split stratification
│   │   └── scaling.py          # Custom scikit-learn transformers for pipeline normalization
│   ├── model
│   │   ├── __init__.py
│   │   ├── evaluation.py       # Asymmetric cost-matrix scoring formulas and threshold tuning
│   │   ├── train.py            # Operational training pipeline driver
│   │   └── xgboost_model.py    # Custom wrappers around the XGBoost training API
│   └── pipelines
│       ├── __init__.py
│       └── model_training.py   # High-level orchestration script joining data to models
├── claude.md                   # System configuration and rules engine
├── main.py                     # Entry point for production execution
├── pyproject.toml              # Modern Python dependency definition (via uv/pip)
├── README.md                   # End-user operational documentation
└── uv.lock                     # Deterministic, lockfile dependency graph
```

## 3. Structural Quality Requirements

### 3.1 Typing, Signatures, and Modern Python Style
* **Strict Typing:** All functional and method interfaces must specify explicit type hints via the typing or built-in collection modules. Sub-type components from third parties (e.g., pd.DataFrame, np.ndarray, xgb.Booster) must be fully denoted.
* **Docstring Standard:** Every class, method, and function must expose an exhaustive Google-style docstring explaining args, returns, and specific exception risks.
* **Error Handling:** Avoid blind try/except: pass paradigms. Write specific catch blocks for expected filesystem and math-bound errors (e.g., FileNotFoundError, ValueError), and emit proper system traces using standard Python logging. Do not use plain print statements.

### 3.2 Machine Learning Invariants
* **Data Leakage Mitigation:** Splitting operations must run before any feature manipulation occurs. Encoders and scalers must execute fit solely against the training array partition.
* **Deterministic Execution:** Seed every pseudo-random mechanism. Pass an explicit random_state parameter down into splits (train_test_split), training initializations, cross-validation runs, and estimator objects.
* **Serialization Invariant:** Utilize joblib or native json/safetensors configurations to store operational pipeline states inside /artifacts/. Ensure files are saved with precise structural version labels.

## 4. Feature and Target Domain Specifications

When parsing the text-based german.data matrix, columns must be named explicitly to mimic the fields found in the documentation. Use the following precise indices and maps:

| Column Index | Field Structural Name | Feature Type |
|--------------|---------------------|--------------|
| 0 | status_checking | Categorical |
| 1 | duration_months | Numerical |
| 2 | credit_history | Categorical |
| 3 | purpose | Categorical |
| 4 | credit_amount | Numerical |
| 5 | savings_bonds | Categorical |
| 6 | employment_since | Categorical |
| 7 | installment_rate | Numerical |
| 8 | personal_status_sex | Categorical |
| 9 | other_debtors | Categorical |
| 10 | present_residence | Numerical |
| 11 | property | Categorical |
| 12 | age_years | Numerical |
| 13 | other_installment | Categorical |
| 14 | housing | Categorical |
| 15 | existing_credits | Numerical |
| 16 | job | Categorical |
| 17 | dependents | Numerical |
| 18 | telephone | Categorical |
| 19 | foreign_worker | Categorical |
| 20 (Target) | credit_risk | Target Categorical Label (1=Good, 2=Bad) |
