"""
Master pipeline orchestrator for German Credit Risk model training.

This script functions as the primary engine execution script that imports
all modular components from the src/ directory, chains them sequentially
to prevent data leakage, executes training and evaluation runs, and saves
all final assets to the /artifacts/ ecosystem.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data.loader import GermanDataLoader
from src.data.preprocessing import GermanCreditPreprocessor
from src.data.scaling import GermanCreditFeatureScaler
from src.model.evaluation import (
    calculate_asymmetric_cost,
    evaluate_predictions,
    optimize_threshold,
    save_evaluation_results,
)
from src.model.train import GermanCreditTrainer
from src.model.xgboost_model import GermanCreditXGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_training_pipeline(data_path: str = "dataset/raw/german.data") -> None:
    """
    Execute the complete model training and evaluation pipeline.

    This function orchestrates the entire workflow from data loading through
    model training, threshold optimization, and evaluation, ensuring proper
    artifact persistence and data leakage prevention.

    Pipeline execution order:
        1. Load data and perform stratified train-test split
        2. Fit and transform preprocessor, save artifact
        3. Fit and transform feature scaler, save artifact
        4. Hyperparameter tuning with GridSearchCV using asymmetric cost scorer
        5. Optimize decision threshold on training data
        6. Evaluate on test set with optimized threshold
        7. Save evaluation report and summary metadata

    Args:
        data_path: Path to the raw german.data file.
            Defaults to "dataset/raw/german.data".

    Raises:
        FileNotFoundError: If the data file does not exist.
        RuntimeError: If any pipeline step fails.
    """
    logger.info("=" * 80)
    logger.info("STARTING GERMAN CREDIT RISK TRAINING PIPELINE")
    logger.info("=" * 80)

    # Initialize random state for reproducibility
    random_state = 42

    try:
        # =====================================================================
        # STEP 1: Load data and perform stratified train-test split
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading data and performing stratified train-test split")
        logger.info("=" * 80)

        loader = GermanDataLoader(data_path=data_path, random_state=random_state)
        df = loader.load_raw_data()
        X_train, X_test, y_train, y_test = loader.split_data(df, test_size=0.2)

        logger.info(
            f"Data split complete: "
            f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples"
        )

        # =====================================================================
        # STEP 2: Fit and transform preprocessor, save artifact
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Preprocessing categorical features")
        logger.info("=" * 80)

        preprocessor = GermanCreditPreprocessor()
        preprocessor.fit(X_train)
        X_train_clean = preprocessor.transform(X_train)
        X_test_clean = preprocessor.transform(X_test)

        logger.info(
            f"Preprocessing complete: "
            f"Train shape: {X_train_clean.shape}, Test shape: {X_test_clean.shape}"
        )

        # Save preprocessor artifact
        preprocessor.save_transformer("artifacts/categorical_preprocessor.joblib")
        logger.info("Preprocessor artifact saved to artifacts/")

        # =====================================================================
        # STEP 3: Fit and transform feature scaler, save artifact
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Scaling features and one-hot encoding")
        logger.info("=" * 80)

        scaler = GermanCreditFeatureScaler()
        scaler.fit(X_train_clean)
        X_train_scaled = scaler.transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test_clean)

        logger.info(
            f"Feature scaling complete: "
            f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}"
        )

        # Save scaler artifact
        scaler.save_artifacts("artifacts")
        logger.info("Feature scaler artifact saved to artifacts/")

        # =====================================================================
        # STEP 4: Hyperparameter tuning with GridSearchCV
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Hyperparameter tuning with GridSearchCV")
        logger.info("=" * 80)

        # Calculate scale_pos_weight to handle class imbalance
        # scale_pos_weight = count(majority_class_0) / count(minority_class_1)
        count_class_0 = int((y_train == 0).sum())
        count_class_1 = int((y_train == 1).sum())
        scale_pos_weight = count_class_0 / count_class_1

        logger.info(
            f"Class distribution in training set: "
            f"Class 0 (Good): {count_class_0}, Class 1 (Bad): {count_class_1}"
        )
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

        # Create custom asymmetric cost scorer (we want to MINIMIZE cost)
        cost_scorer = make_scorer(
            calculate_asymmetric_cost,
            greater_is_better=False
        )
        logger.info("Custom asymmetric cost scorer created")

        # Define hyperparameter grid for regularization
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        }
        logger.info(f"Hyperparameter grid: {param_grid}")

        # Initialize base model
        base_model = GermanCreditXGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state
        )

        # Set up Stratified 5-Fold Cross-Validation
        stratified_kfold = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state
        )
        logger.info("Stratified 5-Fold Cross-Validation configured")

        # Initialize GridSearchCV with custom scorer
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=cost_scorer,
            cv=stratified_kfold,
            n_jobs=-1,
            verbose=1
        )

        # Fit GridSearchCV on training data
        logger.info("Starting hyperparameter search...")
        grid_search.fit(X_train_scaled, y_train)

        # Extract best estimator and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logger.info("=" * 80)
        logger.info("HYPERPARAMETER SEARCH COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best asymmetric cost score: {best_score:.4f}")

        # Update the model with the best estimator
        model = best_model

        # Calculate training set evaluation metrics
        logger.info("Calculating training set evaluation metrics on tuned model")
        y_train_probs = model.predict_proba(X_train_scaled)[:, 1]
        optimal_threshold = optimize_threshold(y_train, y_train_probs)
        training_metrics = evaluate_predictions(
            y_true=y_train,
            y_probs=y_train_probs,
            threshold=optimal_threshold
        )

        logger.info("Training set evaluation metrics:")
        logger.info(f"  Accuracy: {training_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {training_metrics['precision']:.4f}")
        logger.info(f"  Recall: {training_metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {training_metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {training_metrics['roc_auc']:.4f}")
        logger.info(f"  Asymmetric Cost: {training_metrics['asymmetric_cost']:.4f}")
        logger.info(f"  Optimal Threshold: {optimal_threshold:.4f}")

        # Save training evaluation metrics as standalone report
        training_metrics_with_opt = dict(training_metrics)
        training_metrics_with_opt["optimal_threshold"] = float(optimal_threshold)
        save_evaluation_results(
            metrics=training_metrics_with_opt,
            output_dir="artifacts/training_evaluation_result",
            prefix="training_evaluation_report"
        )
        logger.info("Training evaluation report saved to artifacts/training_evaluation_result/")

        # Save model artifact
        model.save_model("artifacts/xgboost_model.joblib")
        logger.info("Model artifact saved to artifacts/")

        # Save hyperparameter tuning metadata with training metrics
        tuning_metadata = {
            "best_parameters": best_params,
            "best_score": float(best_score),
            "scale_pos_weight": float(scale_pos_weight),
            "param_grid": param_grid,
            "cv_strategy": "Stratified 5-Fold",
            "random_state": random_state,
            "training_evaluation_metrics": {
                "accuracy": float(training_metrics["accuracy"]),
                "precision": float(training_metrics["precision"]),
                "recall": float(training_metrics["recall"]),
                "f1_score": float(training_metrics["f1_score"]),
                "roc_auc": float(training_metrics["roc_auc"]),
                "asymmetric_cost": float(training_metrics["asymmetric_cost"]),
                "threshold": float(training_metrics["threshold"]),
                "optimal_threshold": float(optimal_threshold),
            },
        }

        output_path = Path("artifacts/model_training_result")
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuning_file = output_path / f"hyperparameter_tuning_{timestamp}.json"

        with open(tuning_file, "w", encoding="utf-8") as f:
            json.dump(tuning_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Hyperparameter tuning metadata saved to {tuning_file}")

        # Also save as text file
        tuning_txt_file = output_path / f"hyperparameter_tuning_{timestamp}.txt"
        with open(tuning_txt_file, "w", encoding="utf-8") as f:
            json.dump(tuning_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Hyperparameter tuning metadata saved to {tuning_txt_file}")

        # =====================================================================
        # STEP 5: Optimize decision threshold on training data
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Optimizing decision threshold on training data")
        logger.info("=" * 80)

        # Get probability predictions on training set
        y_train_probs = model.predict_proba(X_train_scaled)[:, 1]

        # Optimize threshold to minimize asymmetric cost
        optimal_threshold = optimize_threshold(y_train, y_train_probs)

        logger.info(f"Optimal threshold found: {optimal_threshold:.4f}")

        # =====================================================================
        # STEP 6: Evaluate on test set with optimized threshold
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Evaluating model on test set with optimized threshold")
        logger.info("=" * 80)

        # Get probability predictions on test set
        y_test_probs = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate with optimized threshold
        test_metrics = evaluate_predictions(
            y_true=y_test,
            y_probs=y_test_probs,
            threshold=optimal_threshold
        )

        logger.info("Test set evaluation complete")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        logger.info(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
        logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"Test Asymmetric Cost: {test_metrics['asymmetric_cost']:.4f}")

        # =====================================================================
        # STEP 7: Save evaluation report and summary metadata
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Saving evaluation report and summary metadata")
        logger.info("=" * 80)

        # Add optimal threshold to metrics
        test_metrics["optimal_threshold"] = optimal_threshold

        # Save evaluation results
        save_evaluation_results(
            metrics=test_metrics,
            output_dir="artifacts/evaluation_result"
        )
        logger.info("Evaluation report saved to artifacts/evaluation_result/")

        # =====================================================================
        # Pipeline completion
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nSummary:")
        logger.info(f"  - Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"  - Test asymmetric cost: {test_metrics['asymmetric_cost']:.4f}")
        logger.info(f"  - Test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  - Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info("\nAll artifacts saved to artifacts/ directory")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise RuntimeError(f"Pipeline failed due to missing file: {e}") from e

    except ValueError as e:
        logger.error(f"Value error in pipeline: {e}")
        raise RuntimeError(f"Pipeline failed due to invalid data: {e}") from e

    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}")
        raise RuntimeError(f"Pipeline failed unexpectedly: {e}") from e


if __name__ == "__main__":
    run_training_pipeline()
