"""
Feature scaling and vectorization module for German Credit Risk dataset.

This module provides a scikit-learn compatible transformer that handles
numerical scaling and categorical one-hot encoding, producing numerical
arrays ready for XGBoost ingestion.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Numerical columns as specified in CLAUDE.md Section 4
NUMERICAL_COLUMNS = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "present_residence",
    "age_years",
    "existing_credits",
    "dependents",
]

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


class GermanCreditFeatureScaler(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for feature scaling and vectorization.

    This transformer processes the output of the preprocessor, fits encoders
    and scalers exclusively on the training matrix to prevent data leakage,
    and outputs numerical arrays ready for XGBoost ingestion.

    The transformer uses:
    - StandardScaler for numerical feature normalization
    - OneHotEncoder for categorical feature vectorization

    Attributes:
        numerical_columns: List of numerical column names to scale.
        categorical_columns: List of categorical column names to encode.
        column_transformer: Fitted ColumnTransformer instance.
        feature_names_out: List of expanded feature names after one-hot encoding.

    Args:
        numerical_columns: List of column names to treat as numerical.
            Defaults to columns specified in CLAUDE.md Section 4.
        categorical_columns: List of column names to treat as categorical.
            Defaults to columns specified in CLAUDE.md Section 4.
    """

    def __init__(
        self,
        numerical_columns: Optional[list[str]] = None,
        categorical_columns: Optional[list[str]] = None
    ) -> None:
        """
        Initialize the GermanCreditFeatureScaler.

        Args:
            numerical_columns: List of column names to treat as numerical.
                If None, uses the default numerical columns from CLAUDE.md.
            categorical_columns: List of column names to treat as categorical.
                If None, uses the default categorical columns from CLAUDE.md.
        """
        self.numerical_columns = (
            numerical_columns if numerical_columns is not None
            else NUMERICAL_COLUMNS
        )
        self.categorical_columns = (
            categorical_columns if categorical_columns is not None
            else CATEGORICAL_COLUMNS
        )

        # Initialize transformers
        self.column_transformer = ColumnTransformer(
            transformers=[
                (
                    "num_scaler",
                    StandardScaler(),
                    self.numerical_columns
                ),
                (
                    "cat encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    self.categorical_columns
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        self.feature_names_out: list[str] = []

        logger.info(
            f"GermanCreditFeatureScaler initialized with "
            f"{len(self.numerical_columns)} numerical columns and "
            f"{len(self.categorical_columns)} categorical columns"
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> "GermanCreditFeatureScaler":
        """
        Fit the feature scaler to the training DataFrame.

        This method fits the internal ColumnTransformer exclusively against
        the training DataFrame to prevent data leakage. It also extracts and
        tracks the final expanded feature names after one-hot encoding.

        Args:
            X: Training DataFrame containing features.
            y: Target variable (ignored, present for scikit-learn API compatibility).

        Returns:
            self: Returns the instance itself.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        logger.info("Fitting GermanCreditFeatureScaler on training data")

        # Validate that all required columns exist
        all_required_cols = self.numerical_columns + self.categorical_columns
        missing_cols = [
            col for col in all_required_cols if col not in X.columns
        ]
        if missing_cols:
            error_msg = (
                f"Missing columns in DataFrame: {missing_cols}. "
                f"Available columns: {list(X.columns)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Fit the ColumnTransformer on training data only
        logger.info(
            f"Fitting ColumnTransformer on {X.shape[0]} training samples"
        )
        self.column_transformer.fit(X)

        # Extract and store feature names after transformation
        try:
            self.feature_names_out = list(
                self.column_transformer.get_feature_names_out()
            )
            logger.info(
                f"Extracted {len(self.feature_names_out)} feature names "
                "after transformation"
            )
        except AttributeError:
            # Fallback for older sklearn versions
            logger.warning(
                "Could not extract feature names automatically. "
                "Using placeholder names."
            )
            n_features = self.column_transformer.transform(
                X.head(1)
            ).shape[1]
            self.feature_names_out = [
                f"feature_{i}" for i in range(n_features)
            ]

        logger.info("Feature scaler fitting completed successfully")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the input DataFrame into a numerical array.

        This method applies numerical scaling to numerical fields and one-hot
        encoding to categorical features, producing a pure numeric numpy array
        ready for training or prediction.

        Args:
            X: Input DataFrame containing features to transform.

        Returns:
            np.ndarray: Transformed numerical array ready for XGBoost.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        logger.info(f"Transforming {X.shape[0]} samples")

        # Validate that all required columns exist
        all_required_cols = self.numerical_columns + self.categorical_columns
        missing_cols = [
            col for col in all_required_cols if col not in X.columns
        ]
        if missing_cols:
            error_msg = (
                f"Missing columns in DataFrame: {missing_cols}. "
                f"Available columns: {list(X.columns)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Transform the data
        X_transformed = self.column_transformer.transform(X)

        logger.info(
            f"Transformation complete. Output shape: {X_transformed.shape}"
        )

        return X_transformed

    def save_artifacts(
        self,
        output_dir: str = "artifacts"
    ) -> None:
        """
        Serialize the fitted feature scaler to disk.

        This method saves the completely fitted transformer instance to the
        root-level /artifacts/ folder as feature_scaler.joblib to ensure
        deterministic serving during inference.

        Args:
            output_dir: Directory path where artifacts should be saved.
                Defaults to "artifacts".

        Raises:
            IOError: If the file cannot be written due to filesystem permissions.
            RuntimeError: If the transformer has not been fitted yet.
        """
        logger.info(f"Saving feature scaler artifacts to {output_dir}")

        # Check if transformer has been fitted
        if not self.feature_names_out:
            error_msg = (
                "Cannot save unfitted transformer. "
                "Call fit() before save_artifacts()."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {output_path}")

            # Serialize the transformer
            artifact_path = output_path / "feature_scaler.joblib"
            joblib.dump(self, artifact_path)
            logger.info(f"Successfully saved feature scaler to {artifact_path}")

        except Exception as e:
            error_msg = f"Failed to save artifacts to {output_dir}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
