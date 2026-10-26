"""
Configuration settings for the Customer Churn Prediction project.

This module contains all configurable parameters used throughout the project.
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# config.py lives in src/, so the project root is its parent's parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
SCREENSHOTS_DIR = PROJECT_ROOT / 'screenshots'
REPORTS_DIR = PROJECT_ROOT / 'reports'

# NOTE: directories are intentionally NOT created at import time. Creating
# filesystem side effects on `import config` previously spawned empty src/data,
# src/models, ... folders. Create dirs explicitly where you actually write.

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Dataset file path
DATASET_PATH = DATA_DIR / 'WA_Fn-UseC_-_Telco_Customer_Churn.csv'

# Train-test split ratio
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model artifact paths (single serialized pipeline + metadata)
PIPELINE_PATH = MODELS_DIR / 'churn_pipeline.pkl'
METADATA_PATH = MODELS_DIR / 'metadata.json'

# Logistic Regression hyperparameters
LR_PARAMS = {
    'random_state': RANDOM_STATE,
    'max_iter': 1000,
    'solver': 'lbfgs',
}

# Random Forest hyperparameters
RF_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
}

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# Numerical features for scaling
NUMERIC_FEATURES_TO_SCALE = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
]

# Binary categorical features (Yes/No)
BINARY_CATEGORICAL_FEATURES = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
]

# Multi-class categorical features
MULTICLASS_CATEGORICAL_FEATURES = [
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod',
]

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Cross-validation folds
CV_FOLDS = 5

# Metrics for model selection
PRIMARY_METRIC = 'f1'  # Can be: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'

# Threshold for model selection
METRIC_THRESHOLD = 0.55

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Figure size for plots
FIGURE_SIZE = (12, 6)

# Plot style
PLOT_STYLE = 'whitegrid'

# Color palette
COLORS = {
    'no_churn': '#2ecc71',      # Green
    'churn': '#e74c3c',          # Red
    'primary': '#3498db',        # Blue
    'secondary': '#9b59b6',      # Purple
    'accent': '#e67e22',         # Orange
}

# DPI for saved figures
FIGURE_DPI = 300

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

# Page configuration
PAGE_TITLE = "Customer Churn Prediction"
PAGE_ICON = "🔮"
LAYOUT = "wide"

# Cache time (in seconds)
CACHE_TTL = 3600  # 1 hour

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log level
LOG_LEVEL = 'INFO'

# Log file
LOG_FILE = PROJECT_ROOT / 'project.log'

# ============================================================================
# FEATURE IMPORTANCE CONFIGURATION
# ============================================================================

# Number of top features to display
TOP_N_FEATURES = 15

# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================

# Churn probability threshold for high-risk classification
CHURN_THRESHOLD = 0.5

# Risk levels
RISK_LEVELS = {
    'low': (0, 0.3),
    'medium': (0.3, 0.7),
    'high': (0.7, 1.0),
}

# ============================================================================
# DATA QUALITY CONFIGURATION
# ============================================================================

# Maximum allowed missing values percentage
MAX_MISSING_PERCENT = 10

# Minimum samples required
MIN_SAMPLES = 100

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_config_dict():
    """
    Get all configuration as a dictionary.
    
    Returns:
        dict: Configuration dictionary
    """
    return {
        'paths': {
            'project_root': str(PROJECT_ROOT),
            'data_dir': str(DATA_DIR),
            'models_dir': str(MODELS_DIR),
            'dataset_path': str(DATASET_PATH),
        },
        'model': {
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'cv_folds': CV_FOLDS,
        },
        'visualization': {
            'figure_size': FIGURE_SIZE,
            'dpi': FIGURE_DPI,
            'style': PLOT_STYLE,
        }
    }


if __name__ == "__main__":
    # Print configuration
    import json
    print("Configuration Summary:")
    print(json.dumps(get_config_dict(), indent=2))
