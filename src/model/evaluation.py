"""
Business performance evaluation module for German Credit Risk.

This module implements custom metric calculations based on the asymmetric
cost matrix specified in CLAUDE.md, calculates financial loss profiles,
and provides threshold optimization for minimizing business cost.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Asymmetric cost matrix as specified in CLAUDE.md
# Actual=Bad (1), Predicted=Good (0) -> Penalty = 5
# Actual=Good (0), Predicted=Bad (1) -> Penalty = 1
COST_FALSE_NEGATIVE = 5  # Bad predicted as Good
COST_FALSE_POSITIVE = 1  # Good predicted as Bad


def calculate_asymmetric_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: bool = True
) -> float:
    """
    Calculate total penalty cost based on the asymmetric cost matrix.

    This function computes the financial cost of misclassifications using
    the 5:1 penalty rule where misclassifying a high-risk applicant as safe
    carries a 5x penalty compared to the reverse error.

    Args:
        y_true: Ground truth binary labels (0=Good, 1=Bad).
        y_pred: Predicted binary labels (0=Good, 1=Bad).
        average: If True, returns average cost per instance.
            If False, returns total cost. Defaults to True.

    Returns:
        float: Total asymmetric cost or average cost per instance.

    Raises:
        ValueError: If input arrays have different shapes or contain invalid values.
    """
    logger.info("Calculating asymmetric cost")

    # Validate input shapes
    if y_true.shape != y_pred.shape:
        error_msg = (
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate input values
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        error_msg = (
            "Input arrays must contain only binary values (0 or 1). "
            f"Found unique values in y_true: {np.unique(y_true)}, "
            f"y_pred: {np.unique(y_pred)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Calculate confusion matrix components
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))

    logger.info(
        f"Confusion matrix: TN={true_negatives}, FP={false_positives}, "
        f"FN={false_negatives}, TP={true_positives}"
    )

    # Calculate total cost using asymmetric penalties
    total_cost = (
        false_negatives * COST_FALSE_NEGATIVE +
        false_positives * COST_FALSE_POSITIVE
    )

    logger.info(
        f"Total asymmetric cost: {total_cost} "
        f"(FN: {false_negatives} × {COST_FALSE_NEGATIVE} = {false_negatives * COST_FALSE_NEGATIVE}, "
        f"FP: {false_positives} × {COST_FALSE_POSITIVE} = {false_positives * COST_FALSE_POSITIVE})"
    )

    # Return average cost per instance if requested
    if average:
        avg_cost = total_cost / len(y_true)
        logger.info(f"Average cost per instance: {avg_cost:.4f}")
        return avg_cost

    return total_cost


def evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate predictions using standard metrics and asymmetric cost.

    This function converts probability scores to binary labels using the
    specified threshold, calculates standard classification metrics
    (Accuracy, Precision, Recall, F1-Score, ROC-AUC), and computes the
    custom asymmetric total cost.

    Args:
        y_true: Ground truth binary labels (0=Good, 1=Bad).
        y_probs: Predicted probability scores for the positive class (Bad=1).
        threshold: Decision threshold for converting probabilities to binary labels.
            Defaults to 0.5.

    Returns:
        Dict[str, float]: Dictionary containing all calculated metrics:
            - accuracy: Overall classification accuracy
            - precision: Precision score (positive predictive value)
            - recall: Recall score (sensitivity, true positive rate)
            - f1_score: F1 score (harmonic mean of precision and recall)
            - roc_auc: Area under the ROC curve
            - asymmetric_cost: Average asymmetric cost per instance
            - threshold: The threshold used for evaluation

    Raises:
        ValueError: If input arrays have different shapes or threshold is invalid.
    """
    logger.info(f"Evaluating predictions with threshold={threshold}")

    # Validate threshold
    if not 0 <= threshold <= 1:
        error_msg = f"Threshold must be between 0 and 1, got {threshold}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate input shapes
    if y_true.shape != y_probs.shape:
        error_msg = (
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_probs has shape {y_probs.shape}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Convert probabilities to binary predictions
    y_pred = (y_probs >= threshold).astype(int)

    logger.info(
        f"Converted probabilities to binary predictions. "
        f"Predicted Bad (1): {np.sum(y_pred)}, Good (0): {len(y_pred) - np.sum(y_pred)}"
    )

    # Calculate standard metrics
    metrics: Dict[str, float] = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

    # Calculate ROC-AUC (handle edge case where all labels are the same)
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
    except ValueError as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
        metrics["roc_auc"] = 0.0

    # Calculate asymmetric cost
    metrics["asymmetric_cost"] = calculate_asymmetric_cost(y_true, y_pred, average=True)
    metrics["threshold"] = threshold

    # Log all metrics
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        if key != "threshold":
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    return metrics


def optimize_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    step: float = 0.01
) -> float:
    """
    Find the optimal decision threshold that minimizes asymmetric cost.

    This function iterates across decision thresholds and selects the one
    that minimizes the total asymmetric cost, rather than using the default
    0.5 threshold which may not be optimal for business cost optimization.

    Args:
        y_true: Ground truth binary labels (0=Good, 1=Bad).
        y_probs: Predicted probability scores for the positive class (Bad=1).
        min_threshold: Minimum threshold to evaluate. Defaults to 0.01.
        max_threshold: Maximum threshold to evaluate. Defaults to 0.99.
        step: Increment step for threshold search. Defaults to 0.01.

    Returns:
        float: The threshold that minimizes the asymmetric cost.

    Raises:
        ValueError: If input arrays have different shapes or threshold range is invalid.
    """
    logger.info(
        f"Optimizing threshold from {min_threshold} to {max_threshold} "
        f"with step {step}"
    )

    # Validate input shapes
    if y_true.shape != y_probs.shape:
        error_msg = (
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_probs has shape {y_probs.shape}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate threshold range
    if not 0 <= min_threshold < max_threshold <= 1:
        error_msg = (
            f"Invalid threshold range: min={min_threshold}, max={max_threshold}. "
            "Must satisfy 0 <= min < max <= 1"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if step <= 0:
        error_msg = f"Step must be positive, got {step}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Generate threshold values to test
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    thresholds = np.clip(thresholds, 0, 1)  # Ensure thresholds stay within [0, 1]

    logger.info(f"Testing {len(thresholds)} threshold values")

    # Evaluate cost at each threshold
    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        cost = calculate_asymmetric_cost(y_true, y_pred, average=True)

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    logger.info(
        f"Optimal threshold found: {best_threshold:.2f} "
        f"with asymmetric cost: {best_cost:.4f}"
    )

    # Evaluate metrics at optimal threshold for logging
    optimal_metrics = evaluate_predictions(y_true, y_probs, best_threshold)
    logger.info(
        f"Metrics at optimal threshold: "
        f"Accuracy={optimal_metrics['accuracy']:.4f}, "
        f"Precision={optimal_metrics['precision']:.4f}, "
        f"Recall={optimal_metrics['recall']:.4f}, "
        f"F1={optimal_metrics['f1_score']:.4f}"
    )

    return best_threshold


def save_evaluation_results(
    metrics: Dict[str, float],
    output_dir: str = "artifacts/evaluation_result"
) -> None:
    """
    Save evaluation metrics to a JSON file in the artifacts directory.

    This function ensures the output directory exists and saves the metrics
    dictionary to a timestamped JSON file named test_evaluation_report.json
    for persistent tracking of model performance.

    Args:
        metrics: Dictionary containing evaluation metrics to save.
        output_dir: Directory path where the evaluation report should be saved.
            Defaults to "artifacts/evaluation_result".

    Raises:
        IOError: If the file cannot be written due to filesystem permissions.
        ValueError: If metrics dictionary is empty.
    """
    logger.info(f"Saving evaluation results to {output_dir}")

    # Validate metrics dictionary
    if not metrics:
        error_msg = "Metrics dictionary is empty. Cannot save empty results."
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {output_path}")

        # Generate output file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"test_evaluation_report_{timestamp}.json"

        # Add timestamp to metrics for tracking
        metrics_with_timestamp = metrics.copy()
        metrics_with_timestamp["timestamp"] = timestamp
        metrics_with_timestamp["evaluation_date"] = datetime.now().isoformat()

        # Save metrics to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics_with_timestamp, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved evaluation results to {output_file}")

    except Exception as e:
        error_msg = f"Failed to save evaluation results to {output_dir}: {e}"
        logger.error(error_msg)
        raise IOError(error_msg) from e
