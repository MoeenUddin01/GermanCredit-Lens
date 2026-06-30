"""
Pipelines module for German Credit Risk orchestration.

This module provides high-level orchestration scripts that join
data processing components with model training components.
"""

from .model_training import run_training_pipeline

__all__ = [
    "run_training_pipeline",
]
