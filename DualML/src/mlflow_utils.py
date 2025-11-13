# src/mlflow_utils.py
import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime
import os
from streamlit_config import config

class MLflowManager:
    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
    
    def setup_experiment(self, experiment_name):
        """Setup MLflow experiment"""
        mlflow.set_experiment(experiment_name)
        return experiment_name
    
    def log_classification_experiment(self, model, model_name, X_test, y_test, predictions, params=None):
        """Log classification model experiment"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Set the experiment
        mlflow.set_experiment("EMI_DUAL_ML")
        
        with mlflow.start_run(run_name=f"{model_name}_Classification"):
            # Log parameters
            if params:
                mlflow.log_params(params)
            mlflow.log_param("model_type", "classification")
            mlflow.log_param("model_name", model_name)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
            
            # Log model with consistent naming
            model_name_clean = model_name.replace(" ", "_")
            mlflow.sklearn.log_model(model, f"{model_name_clean}_Classification")
            
            # Log artifacts
            results_df = pd.DataFrame({
                'actual': y_test,
                'predicted': predictions
            })
            results_df.to_csv("predictions.csv", index=False)
            mlflow.log_artifact("predictions.csv")
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
    
    def log_regression_experiment(self, model, model_name, X_test, y_test, predictions, params=None):
        """Log regression model experiment"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Set the experiment
        mlflow.set_experiment("EMI_DUAL_ML")
        
        with mlflow.start_run(run_name=f"{model_name}_Regression"):
            # Log parameters
            if params:
                mlflow.log_params(params)
            mlflow.log_param("model_type", "regression")
            mlflow.log_param("model_name", model_name)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / np.where(y_test == 0, 1, y_test))) * 100
            
            # Log metrics
            mlflow.log_metrics({
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "mape": mape
            })
            
            # Log model with consistent naming
            model_name_clean = model_name.replace(" ", "_")
            mlflow.sklearn.log_model(model, f"{model_name_clean}_Regression")
            
            return {
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "mape": mape
            }