"""
Categorical preprocessing module for German Credit Risk dataset.

This module provides a scikit-learn compatible transformer for early-stage
data cleaning and categorical feature preparation before scaling or vectorization.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Categorical columns as specified in CLAUDE.md Section 4
CATEGORICAL_COLUMNS = [
    "status_checking",
    "credit_history",
    "purpose",
    "savings_bonds",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]


class GermanCreditPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for categorical data preprocessing.

    This transformer handles early-stage data cleaning and categorical feature
    preparation before scaling or vectorization. It ensures proper data type
    casting and removes trailing whitespace from categorical entries.

    The transformer implements the scikit-learn Estimator API (fit/transform)
    to ensure deterministic serving during inference phases.

    Attributes:
        categorical_columns (list): List of categorical column names to process.

    Args:
        categorical_columns: List of column names to treat as categorical.
            Defaults to the columns specified in CLAUDE.md Section 4.
    """

    def __init__(
        self,
        categorical_columns: Optional[list[str]] = None
    ) -> None:
        """
        Initialize the GermanCreditPreprocessor.

        Args:
            categorical_columns: List of column names to treat as categorical.
                If None, uses the default categorical columns from CLAUDE.md.
        """
        self.categorical_columns = (
            categorical_columns if categorical_columns is not None
            else CATEGORICAL_COLUMNS
        )
        logger.info(
            f"GermanCreditPreprocessor initialized with "
            f"{len(self.categorical_columns)} categorical columns"
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> "GermanCreditPreprocessor":
        """
        Fit the preprocessor to the input DataFrame.

        This method evaluates the input DataFrame structure and stores
        necessary state. For static string cleanups, no stateful calculation
        is strictly needed, but the API pattern is preserved for scikit-learn
        compatibility.

        Args:
            X: Input DataFrame containing features.
            y: Target variable (ignored, present for scikit-learn API compatibility).

        Returns:
            self: Returns the instance itself.

        Raises:
            ValueError: If required categorical columns are missing from the DataFrame.
        """
        logger.info("Fitting GermanCreditPreprocessor")

        # Validate that categorical columns exist in the DataFrame
        missing_cols = [
            col for col in self.categorical_columns if col not in X.columns
        ]
        if missing_cols:
            error_msg = (
                f"Missing categorical columns in DataFrame: {missing_cols}. "
                f"Available columns: {list(X.columns)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"Preprocessor fit successfully. "
            f"Validated {len(self.categorical_columns)} categorical columns."
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by cleaning categorical features.

        This method creates a deep copy of the DataFrame, ensures categorical
        columns are cast to proper data types, and strips trailing whitespace
        from string entries.

        Args:
            X: Input DataFrame containing features to transform.

        Returns:
            pd.DataFrame: Cleaned DataFrame with proper categorical types.

        Raises:
            ValueError: If required categorical columns are missing from the DataFrame.
        """
        logger.info("Transforming data with GermanCreditPreprocessor")

        # Validate that categorical columns exist
        missing_cols = [
            col for col in self.categorical_columns if col not in X.columns
        ]
        if missing_cols:
            error_msg = (
                f"Missing categorical columns in DataFrame: {missing_cols}. "
                f"Available columns: {list(X.columns)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Create deep copy to prevent side effects
        logger.info("Creating deep copy of input DataFrame")
        X_transformed = X.copy(deep=True)

        # Process each categorical column
        for col in self.categorical_columns:
            logger.debug(f"Processing column: {col}")

            # Strip trailing whitespace from string entries
            if X_transformed[col].dtype == "object":
                logger.debug(f"Stripping whitespace from column: {col}")
                X_transformed[col] = X_transformed[col].astype(str).str.strip()

            # Cast to category or string type
            try:
                X_transformed[col] = X_transformed[col].astype("category")
                logger.debug(f"Cast column {col} to category type")
            except (TypeError, ValueError) as e:
                # Fallback to string if category conversion fails
                logger.warning(
                    f"Failed to cast column {col} to category: {e}. "
                    "Falling back to string type."
                )
                X_transformed[col] = X_transformed[col].astype("string")

        logger.info(
            f"Transformation complete. Output shape: {X_transformed.shape}"
        )

        return X_transformed

    def save_transformer(
        self,
        filepath: str = "artifacts/categorical_preprocessor.joblib"
    ) -> None:
        """
        Serialize the preprocessor instance to disk using joblib.

        This method saves the fitted transformer state to ensure deterministic
        serving during inference. The parent directory is created if it does
        not exist.

        Args:
            filepath: Path where the transformer should be saved.
                Defaults to "artifacts/categorical_preprocessor.joblib".

        Raises:
            IOError: If the file cannot be written due to filesystem permissions.
        """
        logger.info(f"Saving preprocessor to {filepath}")

        try:
            # Create parent directory if it doesn't exist
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {output_path.parent}")

            # Serialize the transformer
            joblib.dump(self, filepath)
            logger.info(f"Successfully saved preprocessor to {filepath}")

        except Exception as e:
            error_msg = f"Failed to save preprocessor to {filepath}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
