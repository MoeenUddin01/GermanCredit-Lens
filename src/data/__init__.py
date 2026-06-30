"""
Data processing module for German Credit Risk dataset.

This module provides data loading, preprocessing, and feature scaling
functionality for the German Credit Risk pipeline.
"""

from .loader import GermanDataLoader
from .preprocessing import GermanCreditPreprocessor
from .scaling import GermanCreditFeatureScaler

__all__ = [
    "GermanDataLoader",
    "GermanCreditPreprocessor",
    "GermanCreditFeatureScaler",
]
