"""

"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulse-api")

app = FastAPI(title="pulse. — Credit Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "categorical_preprocessor.joblib"
SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.joblib"

COLUMN_NAMES = [
    "status_checking", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_bonds", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "present_residence", "property",
    "age_years", "other_installment", "housing", "existing_credits", "job",
    "dependents", "telephone", "foreign_worker",
]

CATEGORICAL_COLUMNS = [
    "status_checking", "credit_history", "purpose", "savings_bonds",
    "employment_since", "personal_status_sex", "other_debtors", "property",
    "other_installment", "housing", "job", "telephone", "foreign_worker",
]

_model = None
_preprocessor = None
_scaler = None


def load_artifacts() -> None:
    global _model, _preprocessor, _scaler
    try:
        _model = joblib.load(MODEL_PATH)
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
        _scaler = joblib.load(SCALER_PATH)
        logger.info("All artifacts loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise


class CreditApplication(BaseModel):
    status_checking: str = Field(..., description="Status of existing checking account")
    duration_months: float = Field(..., ge=1, description="Loan duration in months")
    credit_history: str = Field(..., description="Credit history")
    purpose: str = Field(..., description="Loan purpose")
    credit_amount: float = Field(..., ge=100, description="Credit amount in DM")
    savings_bonds: str = Field(..., description="Savings account/bonds")
    employment_since: str = Field(..., description="Employment duration")
    installment_rate: float = Field(..., ge=1, le=4, description="Installment rate as percentage of disposable income")
    personal_status_sex: str = Field(..., description="Personal status and sex")
    other_debtors: str = Field(..., description="Other debtors / guarantors")
    present_residence: float = Field(..., ge=1, description="Years in current residence")
    property: str = Field(..., description="Property")
    age_years: float = Field(..., ge=18, description="Age in years")
    other_installment: str = Field(..., description="Other installment plans")
    housing: str = Field(..., description="Housing")
    existing_credits: float = Field(..., ge=1, description="Number of existing credits at this bank")
    job: str = Field(..., description="Employment category")
    dependents: float = Field(..., ge=1, description="Number of dependents")
    telephone: str = Field(..., description="Telephone")
    foreign_worker: str = Field(..., description="Foreign worker")


class PredictionResponse(BaseModel):
    risk_score: float
    risk_probability: float
    prediction: str
    prediction_label: int
    threshold: float
    explanation: str


@app.on_event("startup")
def startup() -> None:
    load_artifacts()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model_loaded": str(_model is not None)}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication) -> PredictionResponse:
    if _model is None or _preprocessor is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the pipeline first.")

    try:
        input_dict = application.model_dump()
        df = pd.DataFrame([input_dict], columns=COLUMN_NAMES)

        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        df_clean = _preprocessor.transform(df)
        df_scaled = _scaler.transform(df_clean)

        proba = _model.predict_proba(df_scaled)
        risk_probability = float(proba[0][1])

        threshold = 0.49
        prediction_label = 1 if risk_probability >= threshold else 0
        prediction = "HIGH RISK — Rejected" if prediction_label == 1 else "LOW RISK — Approved"

        explanation = (
            "This prediction uses a defensive threshold of 0.49 optimized under "
            "a 5:1 asymmetric cost matrix. Misclassifying a high-risk applicant "
            "costs 5x more than rejecting a safe one."
        )

        return PredictionResponse(
            risk_score=round(risk_probability, 4),
            risk_probability=round(risk_probability, 4),
            prediction=prediction,
            prediction_label=prediction_label,
            threshold=threshold,
            explanation=explanation,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
