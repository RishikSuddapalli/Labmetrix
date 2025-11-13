# scripts/prepare_data.py
import pandas as pd
import numpy as np
from streamlit_config import config

def clean_numeric_column(x):
    """Clean numeric columns by handling NaN and converting safely"""
    try:
        if pd.isna(x):  # Check for NaN
            return None
        return int(float(x))  # Convert safely through float
    except (ValueError, TypeError):
        return None  # Return None for anything invalid

def prepare_dataset():
    """Prepare and clean the dataset"""
    try:
        # Load data
        df = pd.read_csv(config.DATA_PATH)
        print(f"Original data shape: {df.shape}")
        
        # Apply cleaning to specified columns
        for col in config.CLEANING_REQUIRED_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric_column).astype('float64')
                print(f"Cleaned column: {col}")
        
        # Handle missing values
        from src.data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        df = preprocessor.handle_missing_values(df)
        
        # Calculate derived features
        from src.utils import EMIPredictUtils
        utils = EMIPredictUtils()
        df = utils.calculate_financial_health_metrics(df)
        
        # Save cleaned data
        cleaned_path = "data/EMI_dataset_cleaned.csv"
        df.to_csv(cleaned_path, index=False)
        print(f"Cleaned data saved to: {cleaned_path}")
        print(f"Final data shape: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

if __name__ == "__main__":
    prepare_dataset()