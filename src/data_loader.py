"""
Data loading and initial preprocessing module for Customer Churn Prediction project.

This module handles:
- Downloading or loading the Telco Customer Churn dataset
- Basic data validation and cleaning
- Data type conversions
- Handling missing values
"""

import pandas as pd
import os

DEFAULT_DATASET_PATH = 'data/WA_Fn-UseC_-_Telco_Customer_Churn.csv'
KAGGLE_URL = 'https://www.kaggle.com/blastchar/telco-customer-churn'


def load_telco_data(data_path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset from disk.

    The dataset is a required input: this function never fabricates data. If the
    file is missing or looks synthetic, it fails loudly so the problem surfaces
    instead of silently poisoning training with random noise.

    Args:
        data_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset

    Raises:
        FileNotFoundError: If the dataset is not present on disk
        ValueError: If the dataset fails the authenticity guardrail
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Download the real Telco Customer "
            f"Churn dataset from {KAGGLE_URL} and place it there. "
            "This project does NOT generate synthetic data."
        )

    print(f"✓ Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    _assert_is_real_telco(df)
    return df


def _assert_is_real_telco(df: pd.DataFrame) -> None:
    """Guardrail blocking placeholder/synthetic datasets from entering the pipeline.

    Catches the failure mode where a random sample (target independent of every
    feature) is mistaken for the real dataset, which makes any model degenerate
    to the majority-class baseline.
    """
    if 'customerID' in df.columns and df['customerID'].astype(str).str.startswith('ID-').mean() > 0.5:
        raise ValueError(
            "Dataset looks synthetic: customerIDs use the 'ID-xxxxx' placeholder "
            "format, not the real Telco format (e.g. '7590-VHVEG')."
        )
    if {'Contract', 'Churn'} <= set(df.columns):
        rates = df.groupby('Contract')['Churn'].apply(lambda s: (s == 'Yes').mean())
        if len(rates) > 1 and (rates.max() - rates.min()) < 0.10:
            raise ValueError(
                "No churn signal across Contract types (spread < 0.10): the data "
                "is almost certainly synthetic/shuffled, not the real Telco set."
            )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning and validation.
    
    Operations:
    - Handle missing values
    - Fix data type inconsistencies
    - Remove duplicates
    - Validate data ranges
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df = df.copy()

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"✓ Removed {initial_rows - len(df)} duplicate rows")

    # Strip surrounding whitespace from categorical values (avoids 'Yes' vs 'Yes ').
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # TotalCharges ships as text with blank cells -> coerce to numeric, leaving
    # NaN. Missing-value imputation is delegated to the modeling pipeline so that
    # training and serving share exactly one imputation strategy (no skew, and
    # no pandas-3.0 chained-inplace pitfalls).
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        n_missing = int(df['TotalCharges'].isna().sum())
        if n_missing:
            print(f"✓ Found {n_missing} blank TotalCharges (imputed in the pipeline)")

    # Remove customer ID as it's not useful for prediction
    if 'customerID' in df.columns:
        df = df.drop(columns='customerID')

    print(f"✓ Data cleaning completed. Final shape: {df.shape}")

    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the dataset for expected structure and content.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_cols = ['gender', 'SeniorCitizen', 'tenure', 'Churn']
    
    # Check for required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠ Warning: Missing columns {missing_cols}")
        return False
    
    # Check for Churn column and valid values
    if df['Churn'].isin(['Yes', 'No']).sum() != len(df):
        print("⚠ Warning: Churn column contains unexpected values")
        return False
    
    print(f"✓ Data validation passed. Dataset shape: {df.shape}")
    return True


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        
    Returns:
        dict: Dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict(),
        'churn_rate': (df['Churn'] == 'Yes').sum() / len(df) if 'Churn' in df.columns else None,
    }
    
    return info


if __name__ == "__main__":
    # Example usage
    df = load_telco_data()
    df = clean_data(df)
    validate_data(df)
    
    info = get_dataset_info(df)
    print("\nDataset Information:")
    print(f"Shape: {info['shape']}")
    print(f"Missing values: {info['missing_values']}")
    print(f"Churn rate: {info['churn_rate']:.2%}")
