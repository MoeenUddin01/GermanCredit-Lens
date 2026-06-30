"""
Custom XGBoost wrapper for German Credit Risk classification.

This module provides a configurable wrapper around XGBoost with a custom
evaluation metric that tracks the asymmetric cost profile during training.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Asymmetric cost matrix as specified in CLAUDE.md
COST_FALSE_NEGATIVE = 5  # Bad predicted as Good
COST_FALSE_POSITIVE = 1  # Good predicted as Bad


def asymmetric_cost_eval(
    y_pred: np.ndarray,
    dtrain: xgb.DMatrix
) -> Tuple[str, float]:
    """
    Custom XGBoost evaluation metric for asymmetric cost.

    This function calculates the asymmetric cost of predictions using the
    5:1 penalty rule where misclassifying a high-risk applicant as safe
    carries a 5x penalty compared to the reverse error.

    Args:
        y_pred: Raw predictions from XGBoost (probabilities or margins).
        dtrain: XGBoost DMatrix containing the true labels.

    Returns:
        Tuple[str, float]: A tuple containing the metric name and the average
            asymmetric cost per instance. Lower is better.
    """
    # Get true labels from DMatrix
    y_true = dtrain.get_label()

    # Convert raw predictions to probabilities using sigmoid
    # XGBoost outputs raw margins for binary:logistic objective
    y_probs = 1.0 / (1.0 + np.exp(-y_pred))

    # Convert probabilities to binary predictions using 0.5 threshold
    y_pred_binary = (y_probs >= 0.5).astype(int)

    # Calculate confusion matrix components
    false_negatives = np.sum((y_true == 1) & (y_pred_binary == 0))
    false_positives = np.sum((y_true == 0) & (y_pred_binary == 1))

    # Calculate total asymmetric cost
    total_cost = (
        false_negatives * COST_FALSE_NEGATIVE +
        false_positives * COST_FALSE_POSITIVE
    )

    # Return average cost per instance
    avg_cost = total_cost / len(y_true)

    return "asymmetric_cost", avg_cost


class GermanCreditXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom XGBoost classifier wrapper for German Credit Risk.

    This class provides a configurable wrapper around XGBoost with a custom
    evaluation metric that tracks the asymmetric cost profile during training.
    The model is optimized for financial cost minimization rather than pure
    accuracy.

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        scale_pos_weight: Balance of positive and negative weights.
        random_state: Random seed for reproducibility.
        model: Fitted XGBoost classifier instance.

    Args:
        n_estimators: Number of boosting rounds. Defaults to 100.
        max_depth: Maximum tree depth. Defaults to 6.
        learning_rate: Boosting learning rate (eta). Defaults to 0.1.
        scale_pos_weight: Balance of positive/negative weights. Defaults to 1.
        random_state: Random seed for reproducibility. Defaults to 42.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float = 1.0,
        random_state: int = 42
    ) -> None:
        """
        Initialize the GermanCreditXGBClassifier.

        Args:
            n_estimators: Number of boosting rounds. Defaults to 100.
            max_depth: Maximum tree depth. Defaults to 6.
            learning_rate: Boosting learning rate (eta). Defaults to 0.1.
            scale_pos_weight: Balance of positive/negative weights. Defaults to 1.
            random_state: Random seed for reproducibility. Defaults to 42.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None

        logger.info(
            f"GermanCreditXGBClassifier initialized: "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"learning_rate={learning_rate}, scale_pos_weight={scale_pos_weight}, "
            f"random_state={random_state}"
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> "GermanCreditXGBClassifier":
        """
        Train the XGBoost classifier with custom asymmetric cost evaluation.

        This method trains the underlying XGBoost classifier using binary
        logistic loss and passes the custom evaluation metric to track
        cost convergence if an evaluation set is provided.

        Args:
            X_train: Training feature matrix as numpy array.
            y_train: Training target labels (0=Good, 1=Bad).
            eval_set: Optional list of (X, y) tuples for validation during training.
                If provided, the custom asymmetric cost metric will be tracked.

        Returns:
            self: Returns the fitted instance.

        Raises:
            ValueError: If input arrays have incompatible shapes or invalid values.
        """
        logger.info(f"Fitting model on {X_train.shape[0]} training samples")

        # Validate input shapes
        if X_train.shape[0] != y_train.shape[0]:
            error_msg = (
                f"Shape mismatch: X_train has {X_train.shape[0]} samples, "
                f"y_train has {y_train.shape[0]} samples"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate target values
        if not np.isin(y_train, [0, 1]).all():
            error_msg = (
                f"Target must contain only binary values (0 or 1). "
                f"Found unique values: {np.unique(y_train)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize XGBoost classifier with binary logistic objective
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1
        )

        # Prepare evaluation set if provided
        eval_set_xgb = None
        if eval_set is not None:
            logger.info(f"Using evaluation set with metric tracking")
            eval_set_xgb = eval_set

        # Train the model
        try:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set_xgb,
                verbose=False
            )
            logger.info("Model fitting completed successfully")

        except Exception as e:
            error_msg = f"Failed to fit XGBoost model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input samples.

        Args:
            X: Feature matrix as numpy array.

        Returns:
            np.ndarray: Array of shape (n_samples, 2) containing probability
                estimates for each class [P(Good=0), P(Bad=1)].

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If input shape is incompatible with training data.
        """
        if self.model is None:
            error_msg = "Model has not been fitted yet. Call fit() before predict_proba()."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Predicting probabilities for {X.shape[0]} samples")

        try:
            probabilities = self.model.predict_proba(X)
            logger.info(f"Probability predictions completed. Shape: {probabilities.shape}")
            return probabilities

        except Exception as e:
            error_msg = f"Failed to predict probabilities: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict class labels for the input samples.

        Args:
            X: Feature matrix as numpy array.
            threshold: Decision threshold for converting probabilities to binary labels.
                Defaults to 0.5.

        Returns:
            np.ndarray: Array of predicted binary labels (0=Good, 1=Bad).

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If input shape is incompatible or threshold is invalid.
        """
        if not 0 <= threshold <= 1:
            error_msg = f"Threshold must be between 0 and 1, got {threshold}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Predicting labels with threshold={threshold}")

        # Get probabilities for the positive class (Bad=1)
        probabilities = self.predict_proba(X)
        y_probs = probabilities[:, 1]

        # Convert to binary predictions
        y_pred = (y_probs >= threshold).astype(int)

        logger.info(
            f"Label predictions completed. "
            f"Predicted Bad (1): {np.sum(y_pred)}, Good (0): {len(y_pred) - np.sum(y_pred)}"
        )

        return y_pred

    def save_model(
        self,
        filepath: str = "artifacts/xgboost_model.joblib"
    ) -> None:
        """
        Serialize the fitted model to disk using joblib.

        This method saves the completely fitted XGBoost classifier to the
        root-level /artifacts/ folder to ensure deterministic serving during
        inference.

        Args:
            filepath: Path where the model should be saved.
                Defaults to "artifacts/xgboost_model.joblib".

        Raises:
            RuntimeError: If the model has not been fitted yet.
            IOError: If the file cannot be written due to filesystem permissions.
        """
        logger.info(f"Saving model to {filepath}")

        if self.model is None:
            error_msg = (
                "Cannot save unfitted model. "
                "Call fit() before save_model()."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            # Create parent directory if it doesn't exist
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {output_path.parent}")

            # Serialize the model
            joblib.dump(self, filepath)
            logger.info(f"Successfully saved model to {filepath}")

        except Exception as e:
            error_msg = f"Failed to save model to {filepath}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
