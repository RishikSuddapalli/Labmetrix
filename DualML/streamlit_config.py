# streamlit_config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # ===== STREAMLIT CLOUD DEPLOYMENT CONFIGURATION =====
    STREAMLIT_SERVER_PORT = 8501
    STREAMLIT_SERVER_ADDRESS = "localhost"
    STREAMLIT_BROWSER_GATHER_USAGE_STATS = False
    STREAMLIT_THEME_BASE = "light"
    
    # ===== MLFLOW CONFIGURATION =====
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "EMIPredict_AI")
    MLFLOW_ARTIFACT_LOCATION = os.getenv("MLFLOW_ARTIFACT_LOCATION", "./mlruns")
    
    # ===== DATA CONFIGURATION =====
    DATA_PATH = os.getenv("DATA_PATH", "data/emi_prediction_dataset.csv")
    # Set SAMPLE_SIZE to None to load all records, or specify a number to limit
    SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "404800")) or None  # 0 or empty means load all
    DATA_CHUNK_SIZE = int(os.getenv("DATA_CHUNK_SIZE", "1000000"))  # For large file processing
    
    # ===== MODEL CONFIGURATION =====
    CLASSIFICATION_MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "Gradient Boosting"]
    REGRESSION_MODELS = ["Linear Regression", "Random Forest", "XGBoost", "Gradient Boosting"]
    
    # Model training parameters
    TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
    CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
    
    # ===== FEATURE CONFIGURATION =====
    # Core numerical features (updated to match dataset)
    NUMERICAL_FEATURES = [
        'age', 'monthly_salary', 'years_of_employment', 'family_size',
        'dependents', 'school_fees', 'college_fees', 'travel_expenses',
        'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount',
        'credit_score', 'bank_balance', 'emergency_fund', 'requested_amount',
        'requested_tenure', 'max_monthly_emi'
    ]
    
    # Core categorical features (updated to match dataset)
    CATEGORICAL_FEATURES = [
        'gender', 'marital_status', 'education', 'employment_type',
        'company_type', 'house_type', 'existing_loans', 'emi_scenario',
        'emi_eligibility'
    ]
    
    # Derived features (will be calculated)
    DERIVED_FEATURES = [
        'expense_to_income_ratio', 'affordability_ratio',
        'risk_score', 'financial_stability_score'
    ]
    
    # ===== DATA CLEANING CONFIGURATION =====
    # Columns that need special cleaning (based on data types and potential issues)
    CLEANING_REQUIRED_COLUMNS = [
        'age', 'monthly_salary', 'bank_balance', 'credit_score',
        'years_of_employment', 'current_emi_amount', 'emergency_fund',
        'requested_amount', 'max_monthly_emi'
    ]
    
    # Zero-inflated columns (from analysis)
    ZERO_INFLATED_COLUMNS = {
        'monthly_rent': 0.0,  # Will be updated during analysis
        'school_fees': 0.0,
        'college_fees': 0.0,
        'current_emi_amount': 0.0,
        'emergency_fund': 0.0
    }
    
    # Potentially correlated feature pairs (to be verified during EDA)
    CORRELATED_FEATURE_PAIRS = [
        ('bank_balance', 'emergency_fund'),
        ('current_emi_amount', 'existing_loans'),
        ('dependents', 'family_size'),
        ('monthly_rent', 'house_type'),
        ('groceries_utilities', 'other_monthly_expenses'),
        ('monthly_salary', 'requested_amount'),
        ('age', 'years_of_employment')
    ]
    
    # ===== PERFORMANCE CONFIGURATION =====
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "True").lower() == "true"
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
    ENABLE_PREDICTION_CACHING = os.getenv("ENABLE_PREDICTION_CACHING", "True").lower() == "true"
    
    # ===== SECURITY CONFIGURATION =====
    ENABLE_AUTH = os.getenv("ENABLE_AUTH", "False").lower() == "true"
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    
    # ===== BUSINESS RULES CONFIGURATION =====
    # EMI Eligibility thresholds (can be adjusted based on business rules)
    ELIGIBILITY_THRESHOLDS = {
        'MIN_CREDIT_SCORE': 650,
        'MAX_DEBT_TO_INCOME': 0.5,
        'MIN_DISPOSABLE_INCOME_RATIO': 0.25,
        'MIN_EMPLOYMENT_YEARS': 1,
        'MAX_LOAN_TO_INCOME': 0.7,
        'MIN_LOAN_AMOUNT': 10000,
        'MAX_LOAN_AMOUNT': 5000000,
        'MIN_LOAN_TENURE': 6,  # months
        'MAX_LOAN_TENURE': 84  # months (7 years)
    }
    
    # Risk scoring weights
    RISK_WEIGHTS = {
        'credit_score': 0.3,
        'debt_to_income': 0.3,
        'expense_to_income': 0.2,
        'employment_stability': 0.2
    }
    
    # ===== VISUALIZATION CONFIGURATION =====
    CHART_THEME = "plotly_white"
    COLOR_SCHEME = {
        'eligible': '#28a745',
        'high_risk': '#ffc107',
        'not_eligible': '#dc3545',
        'low_risk': '#17a2b8',
        'medium_risk': '#fd7e14'
    }

class DevelopmentConfig(Config):
    """Development specific configuration"""
    DEBUG = True
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    ENABLE_CACHING = True

class ProductionConfig(Config):
    """Production specific configuration"""
    DEBUG = False
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    ENABLE_CACHING = True

class StreamlitCloudConfig(Config):
    """Streamlit Cloud specific configuration"""
    DEBUG = False
    MLFLOW_TRACKING_URI = "./mlruns"  # Relative path for Streamlit Cloud
    ENABLE_CACHING = True
    MAX_UPLOAD_SIZE_MB = 100  # Lower limit for cloud

# Determine which configuration to use
def get_config():
    env = os.getenv("STREAMLIT_ENV", "development")
    
    if env == "production":
        return ProductionConfig()
    elif env == "streamlit_cloud":
        return StreamlitCloudConfig()
    else:
        return DevelopmentConfig()

# Create config instance
config = get_config()