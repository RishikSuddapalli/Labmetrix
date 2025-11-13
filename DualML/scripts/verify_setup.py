# scripts/verify_setup.py
import os
import sys
from streamlit_config import config

def verify_setup():
    print("🔍 Verifying EMIPredict AI Setup...")
    
    # Check directories
    required_dirs = ['data', 'models', 'mlruns', 'logs', 'src']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Missing directory: {dir_name}")
    
    # Check data file
    if os.path.exists(config.DATA_PATH):
        print(f"✅ Data file exists: {config.DATA_PATH}")
    else:
        print(f"⚠️ Data file not found: {config.DATA_PATH}")
    
    # Check MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        print(f"✅ MLflow configured with URI: {config.MLFLOW_TRACKING_URI}")
    except Exception as e:
        print(f"❌ MLflow setup error: {e}")
    
    # Check dependencies
    try:
        import pandas as pd
        import streamlit as st
        import plotly
        print("✅ Core dependencies loaded")
    except ImportError as e:
        print(f"❌ Dependency error: {e}")
    
    print("🎯 Setup verification complete!")

if __name__ == "__main__":
    verify_setup()