"""
Customer Churn Prediction - src package

Modules:
- data_loader: dataset loading, cleaning and validation
- feature_engineering: target separation and derived features
- pipeline: the single fit-once / serve-once modeling pipeline
- config: configuration settings
"""

from .data_loader import load_telco_data, clean_data, validate_data, get_dataset_info
from .feature_engineering import separate_features_and_target, engineer_features
from .pipeline import (
    build_pipeline,
    add_engineered_features,
    FeatureEngineer,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from . import config

__version__ = "2.0.0"
__author__ = "Davide Infantino"
__description__ = "Customer Churn Prediction - end-to-end ML pipeline + Streamlit app"

__all__ = [
    # Data loading
    'load_telco_data',
    'clean_data',
    'validate_data',
    'get_dataset_info',
    # Feature engineering
    'separate_features_and_target',
    'engineer_features',
    # Pipeline
    'build_pipeline',
    'add_engineered_features',
    'FeatureEngineer',
    'NUMERIC_FEATURES',
    'CATEGORICAL_FEATURES',
    # Modules
    'config',
]
