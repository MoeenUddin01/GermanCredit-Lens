"""
Pipelines module for German Credit Risk orchestration.

This module provides high-level orchestration scripts that join
data processing components with model training components.
"""

from .model_training import ModelTrainingPipeline

__all__ = [
    "ModelTrainingPipeline",
]
