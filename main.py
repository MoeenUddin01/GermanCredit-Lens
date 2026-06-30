"""
Primary entry point for GermanCredit-Lens training pipeline.

This script parses execution flags from the terminal, triggers the master
training pipeline, and ensures clean execution logging.
"""

import argparse
import logging
import sys

from src.pipelines.model_training import run_training_pipeline

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main entry point for the GermanCredit-Lens training pipeline.

    Parses command-line arguments and executes the training pipeline
    with the specified configuration.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="GermanCredit-Lens: Asymmetric Credit Risk Engine Training Pipeline"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/raw/german.data",
        help="Path to the raw german.data file. Defaults to 'dataset/raw/german.data'."
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to allocate to the test set. Defaults to 0.2."
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility. Defaults to 42."
    )

    args = parser.parse_args()

    logger.info("Starting GermanCredit-Lens training pipeline")
    logger.info(f"Configuration:")
    logger.info(f"  - data_path: {args.data_path}")
    logger.info(f"  - test_size: {args.test_size}")
    logger.info(f"  - random_state: {args.random_state}")

    try:
        # Execute the training pipeline
        run_training_pipeline(data_path=args.data_path)

        logger.info("Training pipeline completed successfully")
        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        logger.error("Please ensure the data file exists at the specified path.")
        sys.exit(1)

    except ValueError as e:
        logger.error(f"Value error: {e}")
        logger.error("Please check your input data and configuration.")
        sys.exit(1)

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        logger.error("Pipeline execution failed. Check the logs above for details.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        logger.error("An unexpected error occurred during pipeline execution.")
        sys.exit(1)


if __name__ == "__main__":
    main()

