"""
Data loader module for German Credit Risk dataset.

This module provides functionality to load the raw German Credit dataset,
apply proper column naming, transform target labels, and perform stratified
train-test splits to prevent data leakage.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Column names as specified in CLAUDE.md Section 4
COLUMN_NAMES = [
    "status_checking",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_bonds",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age_years",
    "other_installment",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "credit_risk",
]


class GermanDataLoader:
    """
    Production-grade data loader for the German Credit Risk dataset.

    This class handles loading the raw space-separated dataset, applying
    standardized column names, transforming target labels to binary format,
    and performing stratified train-test splits to prevent data leakage.

    Attributes:
        data_path (str): Path to the raw german.data file.
        random_state (int): Random seed for reproducible splits.

    Args:
        data_path: File system path to the raw dataset file.
        random_state: Random seed for reproducible data splitting. Defaults to 42.
    """

    def __init__(self, data_path: str, random_state: int = 42) -> None:
        """
        Initialize the GermanDataLoader.

        Args:
            data_path: Path to the raw german.data file.
            random_state: Random seed for reproducible splits. Defaults to 42.
        """
        self.data_path = Path(data_path)
        self.random_state = random_state
        logger.info(
            f"GermanDataLoader initialized with data_path='{self.data_path}', "
            f"random_state={self.random_state}"
        )

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load the raw German Credit dataset from the space-separated file.

        This method reads the dataset, applies the standardized column names
        as specified in CLAUDE.md Section 4, and performs basic validation.

        Returns:
            pd.DataFrame: Loaded dataset with proper column names.

        Raises:
            FileNotFoundError: If the data file does not exist at the specified path.
            ValueError: If the loaded data does not have the expected number of columns.
        """
        if not self.data_path.exists():
            error_msg = f"Data file not found at path: {self.data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading data from {self.data_path}")

        try:
            df = pd.read_csv(
                self.data_path,
                sep=r"\s+",
                header=None,
                names=COLUMN_NAMES,
                engine="python"
            )
        except Exception as e:
            error_msg = f"Failed to read data file: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        # Validate column count
        if len(df.columns) != len(COLUMN_NAMES):
            error_msg = (
                f"Expected {len(COLUMN_NAMES)} columns but found {len(df.columns)}. "
                "Data file may be corrupted or format has changed."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"Successfully loaded dataset with shape {df.shape}. "
            f"Columns: {list(df.columns)}"
        )

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into stratified train and test sets.

        This method separates features from the target, transforms the target
        labels from the original encoding (1=Good, 2=Bad) to standard binary
        format (0=Good, 1=Bad), and performs a stratified split to maintain
        class distribution across train and test sets.

        Args:
            df: Input DataFrame containing features and target.
            test_size: Proportion of data to allocate to the test set. Defaults to 0.2.

        Returns:
            Tuple containing:
                - X_train: Training feature DataFrame
                - X_test: Testing feature DataFrame
                - y_train: Training target Series (binary encoded)
                - y_test: Testing target Series (binary encoded)

        Raises:
            ValueError: If the target column 'credit_risk' is not found in the DataFrame.
        """
        if "credit_risk" not in df.columns:
            error_msg = "Target column 'credit_risk' not found in DataFrame."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Splitting features and target")

        # Separate features and target
        X = df.drop(columns=["credit_risk"])
        y = df["credit_risk"]

        logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

        # Transform target: 1 (Good) -> 0, 2 (Bad) -> 1
        logger.info("Transforming target labels: 1 (Good) -> 0, 2 (Bad) -> 1")
        y_transformed = y.map({1: 0, 2: 1})

        # Validate transformation
        if y_transformed.isna().any():
            unexpected_values = y[~y.isin([1, 2])].unique()
            error_msg = (
                f"Found unexpected values in target: {unexpected_values}. "
                "Expected only 1 (Good) and 2 (Bad)."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"Target distribution after transformation: "
            f"Good (0): {(y_transformed == 0).sum()}, Bad (1): {(y_transformed == 1).sum()}"
        )

        # Perform stratified train-test split
        logger.info(
            f"Performing stratified train-test split with test_size={test_size}, "
            f"random_state={self.random_state}"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_transformed,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_transformed
        )

        logger.info(
            f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape}. "
            f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}"
        )

        logger.info(
            f"Train target distribution: Good (0): {(y_train == 0).sum()}, "
            f"Bad (1): {(y_train == 1).sum()}"
        )
        logger.info(
            f"Test target distribution: Good (0): {(y_test == 0).sum()}, "
            f"Bad (1): {(y_test == 1).sum()}"
        )

        return X_train, X_test, y_train, y_test
