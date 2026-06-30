# GermanCredit-Lens

An asymmetric credit risk evaluation engine built with XGBoost and custom scikit-learn transformers. This project implements a production-grade machine learning pipeline for creditworthiness assessment under asymmetric business cost constraints.

## Overview

GermanCredit-Lens evaluates credit risk using the German Credit dataset with a focus on minimizing financial risk exposure rather than pure accuracy. The system implements an asymmetric cost matrix where misclassifying high-risk applicants as safe carries a 5x penalty compared to the reverse error.

### Key Features

- **Asymmetric Cost Optimization**: Custom evaluation metrics that minimize financial risk exposure
- **Production-Grade Pipeline**: Modular scikit-learn transformers with fit/transform API for deterministic inference
- **Data Leakage Prevention**: Stratified splits and training-only parameter fitting
- **Artifact Persistence**: All fitted objects serialized to `/artifacts/` for reproducible serving
- **Type Safety**: Full type hints and Google-style docstrings throughout

## Installation

### Prerequisites

- Python 3.12 or higher
- uv (recommended) or pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd GermanCredit-Lens

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Project Structure

```
.
├── artifacts/                  # Production asset store (serialized models, scalers, encoders)
├── dataset
│   ├── processed/              # Cached engine data tensors
│   └── raw
│       ├── german.data         # Source space-separated text dataset
│       └── german.data-numeric # Benchmark numeric variant
├── notebooks
│   └── preprocessing.ipynb    # Experimentation and prototyping
├── src
│   ├── data
│   │   ├── loader.py           # Data loading, column naming, stratified splits
│   │   ├── preprocessing.py    # Categorical data cleaning
│   │   └── scaling.py          # Feature scaling and one-hot encoding
│   ├── model
│   │   ├── evaluation.py       # Asymmetric cost-matrix scoring
│   │   ├── train.py            # Training pipeline driver
│   │   └── xgboost_model.py    # XGBoost wrapper implementation
│   └── pipelines
│       └── model_training.py   # High-level orchestration
├── CLAUDE.md                   # Architecture standards and coding invariants
├── main.py                     # Entry point for production execution
├── pyproject.toml              # Dependency management
└── README.md                   # This file
```

## Usage

### Basic Training Pipeline

```python
from src.data.loader import GermanDataLoader
from src.data.preprocessing import GermanCreditPreprocessor
from src.data.scaling import GermanCreditFeatureScaler
from src.model.train import ModelTrainer

# Load and split data
loader = GermanDataLoader("dataset/raw/german.data", random_state=42)
df = loader.load_raw_data()
X_train, X_test, y_train, y_test = loader.split_data(df, test_size=0.2)

# Preprocess categorical features
preprocessor = GermanCreditPreprocessor()
preprocessor.fit(X_train)
X_train_clean = preprocessor.transform(X_train)
X_test_clean = preprocessor.transform(X_test)

# Scale and encode features
scaler = GermanCreditFeatureScaler()
scaler.fit(X_train_clean)
X_train_scaled = scaler.transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_clean)

# Train model with asymmetric cost optimization
trainer = ModelTrainer(random_state=42)
model = trainer.train(X_train_scaled, y_train)

# Evaluate with asymmetric cost matrix
from src.model.evaluation import AsymmetricCostEvaluator
evaluator = AsymmetricCostEvaluator()
metrics = evaluator.evaluate(model, X_test_scaled, y_test)
```

### Running the Full Pipeline

```bash
python main.py
```

## Asymmetric Cost Matrix

The project optimizes for business cost rather than pure accuracy:

| Actual \ Predicted | Good (0) | Bad (1) |
|-------------------|----------|---------|
| **Good (0)**       | 0        | 1       |
| **Bad (1)**        | 5        | 0       |

- **False Negative (Bad → Good)**: Cost = 5 (high-risk applicant approved)
- **False Positive (Good → Bad)**: Cost = 1 (safe applicant rejected)

This cost structure reflects the higher financial risk of approving a bad credit applicant compared to rejecting a good one.

## Data Schema

The dataset uses 21 columns as specified in CLAUDE.md:

| Index | Column Name | Type |
|-------|-------------|------|
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
| 20 | credit_risk | Target (1=Good, 2=Bad) |

## Development Guidelines

All development must follow the standards specified in `CLAUDE.md`:

- **Strict Typing**: All functions must have explicit type hints
- **Google Docstrings**: Every class/method requires exhaustive documentation
- **Error Handling**: Specific catch blocks with proper logging (no bare except)
- **No Print Statements**: Use logging instead
- **Deterministic Execution**: Seed all random operations with `random_state`
- **Artifact Persistence**: Serialize all fitted objects to `/artifacts/`

## License

[Specify your license here]

## Contributing

[Specify contribution guidelines here]
