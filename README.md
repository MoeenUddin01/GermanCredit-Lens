# GermanCredit-Lens

An asymmetric credit risk evaluation engine built with XGBoost and custom scikit-learn transformers. This project implements a production-grade machine learning pipeline for creditworthiness assessment under asymmetric business cost constraints.

## Overview

GermanCredit-Lens evaluates credit risk using the German Credit dataset with a focus on minimizing financial risk exposure rather than pure accuracy. The system implements an asymmetric cost matrix where misclassifying high-risk applicants as safe carries a 5x penalty compared to the reverse error.

### Key Features

- **Asymmetric Cost Optimization**: Custom evaluation metrics that minimize financial risk exposure
- **Hyperparameter Tuning**: GridSearchCV with Stratified 5-Fold cross-validation using asymmetric cost as the optimization objective
- **Class Imbalance Handling**: Dynamic `scale_pos_weight` calculation for balanced training
- **Production-Grade Pipeline**: Modular scikit-learn transformers with fit/transform API for deterministic inference
- **Data Leakage Prevention**: Stratified splits and training-only parameter fitting
- **Artifact Persistence**: All fitted objects serialized to `/artifacts/` for reproducible serving (JSON and text formats)
- **Separate Training & Test Reports**: Standalone evaluation reports for both training and test sets
- **Type Safety**: Full type hints and Google-style docstrings throughout
- **Prediction API**: FastAPI backend serving real-time inference via `POST /predict`
- **Interactive UI**: Next.js frontend with glassmorphic neobank design for credit assessment

## Installation

### Prerequisites

- Python 3.10 or higher
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
├── artifacts/                  # Production asset store
│   ├── evaluation_result/      # Test set evaluation reports
│   ├── training_evaluation_result/  # Training set evaluation reports
│   └── model_training_result/  # Hyperparameter tuning metadata
├── dataset
│   ├── processed/              # Cached engine data tensors
│   └── raw
│       ├── german.data         # Source space-separated text dataset
│       └── german.data-numeric # Benchmark numeric variant
├── api/
│   ├── main.py                 # FastAPI prediction server (POST /predict)
│   └── requirements.txt        # Backend Python dependencies
├── notebooks
│   └── preprocessing.ipynb    # Experimentation and prototyping
├── src
│   ├── data
│   │   ├── loader.py           # Data loading, column naming, stratified splits
│   │   ├── preprocessing.py    # Categorical data cleaning
│   │   └── scaling.py          # Feature scaling and one-hot encoding
│   ├── model
│   │   ├── evaluation.py       # Asymmetric cost-matrix scoring and threshold tuning
│   │   ├── train.py            # Training metadata tracker
│   │   └── xgboost_model.py    # XGBoost wrapper implementation
│   └── pipelines
│       └── model_training.py   # End-to-end orchestration pipeline
├── frontend/
│   ├── src/
│   │   ├── app/                # Next.js App Router pages
│   │   ├── components/         # UI components and forms
│   │   ├── services/           # API client layer
│   │   └── hooks/              # Custom React hooks
│   ├── public/                 # Static assets
│   ├── package.json
│   ├── tsconfig.json
│   └── tailwind.config.js      # Neon-lime design tokens
├── CLAUDE.md                   # Architecture standards and coding invariants
├── main.py                     # CLI entry point for production execution
├── pyproject.toml              # Dependency management
└── README.md                   # This file
```

## Usage

### Running the Full Pipeline

```bash
# Default execution
python main.py

# With custom parameters
python main.py --data_path "dataset/raw/german.data" --test_size 0.2 --random_state 42
```

### Pipeline Steps

The orchestration pipeline executes in this order:

1. **Load Data** — Load the raw dataset and perform stratified train-test split
2. **Preprocess Categoricals** — Strip whitespace and cast to proper types
3. **Scale & Encode** — Scale numerical features, one-hot encode categoricals
4. **Hyperparameter Tuning** — GridSearchCV with Stratified 5-Fold CV using asymmetric cost scorer
5. **Save Training Report** — Persist training set evaluation metrics as standalone report
6. **Threshold Optimization** — Find optimal decision threshold minimizing asymmetric cost on training data
7. **Test Evaluation** — Evaluate final model on held-out test set using optimized threshold
8. **Save Test Report** — Persist test set evaluation metrics as standalone report

### Prediction API

Start the FastAPI server to serve real-time predictions using the trained model:

```bash
# Install API dependencies
uv pip install -r api/requirements.txt

# Start the server
cd GermanCredit-Lens && uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API exposes two endpoints:

- `GET /health` — Check if the model is loaded and server is operational
- `POST /predict` — Submit a credit application and receive a risk assessment

Example prediction request:

```bash
curl -s http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "status_checking": "A11",
    "duration_months": 24,
    "credit_history": "A32",
    "purpose": "A42",
    "credit_amount": 2500,
    "savings_bonds": "A65",
    "employment_since": "A73",
    "installment_rate": 4,
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "present_residence": 3,
    "property": "A122",
    "age_years": 30,
    "other_installment": "A143",
    "housing": "A152",
    "existing_credits": 1,
    "job": "A173",
    "dependents": 1,
    "telephone": "A191",
    "foreign_worker": "A201"
  }'
```

Response:

```json
{
  "risk_score": 0.5121,
  "risk_probability": 0.5121,
  "prediction": "HIGH RISK — Rejected",
  "prediction_label": 1,
  "threshold": 0.49,
  "explanation": "This prediction uses a defensive threshold of 0.49..."
}
```

### Interactive Web UI

A premium neobank-style frontend is available for visual credit assessment:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` to access the landing page. Navigate to `/dashboard` to submit credit applications interactively. The UI features:

- **Glassmorphic design** with neon lime-green accent (#CCFF00) on dark surfaces
- **3D interactive mockup** with mouse-tracked perspective tilt
- **Full 20-field credit form** with floating labels and select dropdowns
- **Animated results** showing risk probability, decision threshold, and verdict
- **Real-time API integration** with the FastAPI backend (default `http://localhost:8000`)

### Artifacts Produced

After a successful run, the following files are generated:

| Artifact | Location | Format |
|----------|----------|--------|
| XGBoost model | `artifacts/xgboost_model.joblib` | joblib |
| Categorical preprocessor | `artifacts/categorical_preprocessor.joblib` | joblib |
| Feature scaler & encoder | `artifacts/*.joblib` | joblib |
| Training evaluation report | `artifacts/training_evaluation_result/training_evaluation_report_*.json` | JSON / TXT |
| Test evaluation report | `artifacts/evaluation_result/test_evaluation_report_*.json` | JSON / TXT |
| Hyperparameter tuning metadata | `artifacts/model_training_result/hyperparameter_tuning_*.json` | JSON / TXT |

### Programmatic Usage

```python
from src.data.loader import GermanDataLoader
from src.data.preprocessing import GermanCreditPreprocessor
from src.data.scaling import GermanCreditFeatureScaler
from src.model.xgboost_model import GermanCreditXGBClassifier
from src.model.evaluation import evaluate_predictions, optimize_threshold
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
from src.model.evaluation import calculate_asymmetric_cost

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

# Calculate scale_pos_weight
scale_pos_weight = int((y_train == 0).sum()) / int((y_train == 1).sum())

# Hyperparameter tuning with asymmetric cost scorer
base_model = GermanCreditXGBClassifier(
    scale_pos_weight=scale_pos_weight, random_state=42
)
cost_scorer = make_scorer(calculate_asymmetric_cost, greater_is_better=False)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid={
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    },
    scoring=cost_scorer,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)
model = grid_search.best_estimator_

# Optimize threshold and evaluate
y_train_probs = model.predict_proba(X_train_scaled)[:, 1]
optimal_threshold = optimize_threshold(y_train, y_train_probs)

y_test_probs = model.predict_proba(X_test_scaled)[:, 1]
metrics = evaluate_predictions(y_test, y_test_probs, threshold=optimal_threshold)
```

## Model Performance

The current model achieves the following metrics using the optimal decision threshold (0.49):

| Metric | Train | Test | Gap | Interpretation |
|--------|-------|------|-----|----------------|
| Accuracy | 0.7575 | 0.6800 | +7.8% | Moderate gap, acceptable |
| Precision | 0.5596 | 0.4792 | +8.0% | Moderate gap |
| Recall | 0.9000 | 0.7667 | +13.3% | Noticeable drop on unseen data |
| F1-Score | 0.6901 | 0.5897 | +10.0% | Moderate gap |
| ROC-AUC | 0.8774 | 0.7592 | +11.8% | Moderate gap |
| **Asymmetric Cost** | **0.3625** | **0.6000** | **+65%** | Cost generalizes moderately |

The model exhibits **mild overfitting** — gaps of 7–13% across standard metrics, which is typical for a small dataset (1000 samples). The GridSearchCV regularization (`max_depth=4`, `learning_rate=0.01`, `n_estimators=50`) effectively prevents severe overfitting.

The model is **not underfit** — both train and test metrics show it learned meaningful patterns from the data.

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
