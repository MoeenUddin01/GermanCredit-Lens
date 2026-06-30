"""
Master pipeline orchestrator for German Credit Risk model training.

This script functions as the primary engine execution script that imports
all modular components from the src/ directory, chains them sequentially
to prevent data leakage, executes training and evaluation runs, and saves
all final assets to the /artifacts/ ecosystem.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.data.loader import GermanDataLoader
from src.data.preprocessing import GermanCreditPreprocessor
from src.data.scaling import GermanCreditFeatureScaler
from src.model.evaluation import (
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
        4. Train XGBoost model via trainer, save model
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
        # STEP 4: Train XGBoost model via trainer, save model
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Training XGBoost model")
        logger.info("=" * 80)

        # Initialize model with hyperparameters
        model = GermanCreditXGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1.0,
            random_state=random_state
        )

        # Initialize trainer
        trainer = GermanCreditTrainer(random_state=random_state)

        # Prepare evaluation set for training monitoring
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]

        # Train model
        model = trainer.train_model(
            model=model,
            X_train=X_train_scaled,
            y_train=y_train,
            eval_set=eval_set
        )

        # Save model artifact
        model.save_model("artifacts/xgboost_model.joblib")
        logger.info("Model artifact saved to artifacts/")

        # Save training metadata
        trainer.save_training_metadata("artifacts/model_training_result")
        logger.info("Training metadata saved to artifacts/model_training_result/")

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
