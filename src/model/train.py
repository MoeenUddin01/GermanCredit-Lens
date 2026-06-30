"""
Training core engine module for German Credit Risk model.

This module handles fitting the XGBoost wrapper model, tracking performance
metrics during training, and saving operational training metadata to the
artifacts directory.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.model.xgboost_model import GermanCreditXGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GermanCreditTrainer:
    """
    Training manager for German Credit Risk XGBoost model.

    This class handles the complete training workflow including model fitting,
    metric tracking during training, and serialization of training metadata
    for reproducibility and audit trails.

    Attributes:
        training_metadata: Dictionary storing training parameters and metrics.

    Args:
        random_state: Random seed for reproducibility. Defaults to 42.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Initialize the GermanCreditTrainer.

        Args:
            random_state: Random seed for reproducibility. Defaults to 42.
        """
        self.random_state = random_state
        self.training_metadata: Dict[str, Any] = {
            "random_state": random_state,
            "training_start_time": None,
            "training_end_time": None,
            "training_duration_seconds": None,
            "model_parameters": {},
            "training_metrics": {},
            "validation_metrics": {},
        }
        logger.info(
            f"GermanCreditTrainer initialized with random_state={random_state}"
        )

    def train_model(
        self,
        model: GermanCreditXGBClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> GermanCreditXGBClassifier:
        """
        Train the XGBoost model with metric tracking.

        This method executes the model's fit routine and logs training
        progress including metric trends across boosting rounds if validation
        evaluation hooks are provided.

        Args:
            model: GermanCreditXGBClassifier instance to train.
            X_train: Training feature matrix as numpy array.
            y_train: Training target labels (0=Good, 1=Bad).
            eval_set: Optional list of (X, y) tuples for validation during training.
                If provided, validation metrics will be tracked and logged.

        Returns:
            GermanCreditXGBClassifier: The fitted model instance.

        Raises:
            ValueError: If input arrays have incompatible shapes or invalid values.
            RuntimeError: If model fitting fails.
        """
        logger.info("Starting model training")
        logger.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")

        # Record training start time
        self.training_metadata["training_start_time"] = datetime.now().isoformat()
        start_time = time.time()

        # Record model parameters
        self.training_metadata["model_parameters"] = {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "scale_pos_weight": model.scale_pos_weight,
            "random_state": model.random_state,
        }
        logger.info(f"Model parameters: {self.training_metadata['model_parameters']}")

        # Record training set information
        self.training_metadata["training_metrics"] = {
            "n_samples": X_train.shape[0],
            "n_features": X_train.shape[1],
            "class_distribution": {
                "good (0)": int((y_train == 0).sum()),
                "bad (1)": int((y_train == 1).sum()),
            },
        }
        logger.info(
            f"Training class distribution: "
            f"Good (0): {self.training_metadata['training_metrics']['class_distribution']['good (0)']}, "
            f"Bad (1): {self.training_metadata['training_metrics']['class_distribution']['bad (1)']}"
        )

        # Record validation set information if provided
        if eval_set is not None:
            logger.info(f"Validation set provided with {len(eval_set)} evaluation set(s)")
            self.training_metadata["validation_metrics"] = {
                "n_eval_sets": len(eval_set),
                "eval_set_sizes": [
                    {
                        "n_samples": eval_X.shape[0],
                        "n_features": eval_X.shape[1],
                        "class_distribution": {
                            "good (0)": int((eval_y == 0).sum()),
                            "bad (1)": int((eval_y == 1).sum()),
                        },
                    }
                    for eval_X, eval_y in eval_set
                ],
            }

        # Fit the model
        try:
            logger.info("Initiating model fitting...")
            model.fit(X_train, y_train, eval_set=eval_set)
            logger.info("Model fitting completed successfully")

        except Exception as e:
            error_msg = f"Model fitting failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Record training end time and duration
        end_time = time.time()
        self.training_metadata["training_end_time"] = datetime.now().isoformat()
        self.training_metadata["training_duration_seconds"] = end_time - start_time

        logger.info(
            f"Training completed in "
            f"{self.training_metadata['training_duration_seconds']:.2f} seconds"
        )

        # Log training results if available from XGBoost
        if hasattr(model.model, "evals_result"):
            evals_result = model.model.evals_result()
            logger.info("Training evaluation results available")
            for eval_name, metrics in evals_result.items():
                logger.info(f"Evaluation set: {eval_name}")
                for metric_name, metric_values in metrics.items():
                    final_value = metric_values[-1]
                    logger.info(
                        f"  {metric_name}: {final_value:.4f} "
                        f"(over {len(metric_values)} rounds)"
                    )
                    # Store final metrics in metadata
                    if "validation_metrics" not in self.training_metadata:
                        self.training_metadata["validation_metrics"] = {}
                    if eval_name not in self.training_metadata["validation_metrics"]:
                        self.training_metadata["validation_metrics"][eval_name] = {}
                    self.training_metadata["validation_metrics"][eval_name][
                        metric_name
                    ] = {
                        "final_value": float(final_value),
                        "n_rounds": len(metric_values),
                    }

        return model

    def save_training_metadata(
        self,
        output_dir: str = "artifacts/model_training_result"
    ) -> None:
        """
        Save training metadata to a JSON file in the artifacts directory.

        This method creates the target folder if missing and serializes
        parameters, training runtime details, and training/validation metric
        histories into a file named model_training_report.json.

        Args:
            output_dir: Directory path where the training report should be saved.
                Defaults to "artifacts/model_training_result".

        Raises:
            IOError: If the file cannot be written due to filesystem permissions.
            ValueError: If training metadata is incomplete or empty.
        """
        logger.info(f"Saving training metadata to {output_dir}")

        # Validate training metadata
        if not self.training_metadata.get("training_start_time"):
            error_msg = (
                "Training metadata is incomplete. "
                "No training has been performed yet."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {output_path}")

            # Generate output file path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_path / f"model_training_report_{timestamp}.json"

            # Add save timestamp to metadata
            metadata_to_save = self.training_metadata.copy()
            metadata_to_save["save_timestamp"] = timestamp
            metadata_to_save["save_date"] = datetime.now().isoformat()

            # Save metadata to JSON file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved training metadata to {output_file}")

        except Exception as e:
            error_msg = f"Failed to save training metadata to {output_dir}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
