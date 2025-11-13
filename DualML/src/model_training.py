# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, 
    mean_absolute_error, r2_score
)
import mlflow
import mlflow.sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Any, Tuple, Union
import logging
from datetime import datetime
import numpy as np

from src.utils import EMIPredictUtils
from src.mlflow_utils import MLflowManager

class ModelTrainer:
    """Machine Learning model trainer for EMIPredict AI"""
    
    def __init__(self, experiment_name='EMI_DUAL_ML'):
        self.utils = EMIPredictUtils()
        self.logger = self.utils.logger
        self.classification_models = {}
        self.regression_models = {}
        self.experiment_name = experiment_name
        self.mlflow_manager = MLflowManager()
        self.mlflow_manager.setup_experiment(self.experiment_name)
        
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for model training
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, preprocessor, feature_names)
        """
        try:
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Identify feature types
            numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            
            self.logger.info(f"Numerical features: {len(numerical_features)}")
            self.logger.info(f"Categorical features: {len(categorical_features)}")
            
            # Create preprocessor
            numerical_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y if target_column == 'emi_eligibility' else None
            )
            
            # Get feature names after preprocessing
            preprocessor.fit(X_train)
            feature_names = self._get_feature_names(preprocessor, numerical_features, categorical_features)
            
            self.logger.info(f"Data prepared: {len(X_train)} training, {len(X_test)} test samples")
            
            return X_train, X_test, y_train, y_test, preprocessor, feature_names
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def _get_feature_names(self, preprocessor, numerical_features: List[str], 
                          categorical_features: List[str]) -> List[str]:
        """Get feature names after preprocessing"""
        feature_names = numerical_features.copy()
        
        # Add categorical feature names
        categorical_transformer = preprocessor.named_transformers_['cat']
        if hasattr(categorical_transformer, 'get_feature_names_out'):
            cat_features = categorical_transformer.get_feature_names_out(categorical_features)
            feature_names.extend(cat_features)
        else:
            # Fallback for older sklearn versions
            for feature in categorical_features:
                unique_vals = categorical_transformer.categories_[categorical_features.index(feature)]
                for val in unique_vals:
                    feature_names.append(f"{feature}_{val}")
        
        return feature_names
    
    def train_classification_model(self, X_train, y_train, X_test, y_test, model_name='Random Forest', params=None):
        """
        Train a classification model with MLflow tracking
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model to train
            params: Hyperparameters for the model
            
        Returns:
            Trained model and evaluation metrics
        """
        try:
            # Set up MLflow run
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize model
            if model_name == 'Random Forest':
                model = RandomForestClassifier(**params) if params else RandomForestClassifier()
            elif model_name == 'Logistic Regression':
                model = LogisticRegression(**params) if params else LogisticRegression()
            elif model_name == 'XGBoost':
                model = XGBClassifier(**params) if params else XGBClassifier()
            elif model_name == 'Gradient Boosting':
                model = GradientBoostingClassifier(**params) if params else GradientBoostingClassifier()
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Log to MLflow
            metrics = self.mlflow_manager.log_classification_experiment(
                model=model,
                model_name=model_name,
                X_test=X_test,
                y_test=y_test,
                params=params
            )
            
            # Store model
            self.classification_models[model_name] = model
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                plt.figure(figsize=(10, 6))
                plt.title(f"Feature Importances - {model_name}")
                plt.bar(range(X_train.shape[1]), importances[indices])
                plt.xticks(range(X_train.shape[1]), 
                          [X_train.columns[i] for i in indices], 
                          rotation=90)
                plt.tight_layout()
                
                # Log feature importance plot
                importance_path = f"feature_importance_{model_name}.png"
                plt.savefig(importance_path)
                mlflow.log_artifact(importance_path)
                plt.close()
            
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"Error training {model_name}: {str(e)}")
            mlflow.end_run(status="FAILED")
            raise
    
    def train_regression_models(self, X_train, X_test, y_train, y_test, 
                              preprocessor, model_types: List[str] = None,
                              use_mlflow: bool = True) -> Dict[str, Any]:
        """
        Train multiple regression models for maximum EMI prediction
        
        Args:
            X_train, X_test, y_train, y_test: Training and test data
            preprocessor: Fitted preprocessor
            model_types: List of model types to train
            use_mlflow: Whether to log to MLflow
            
        Returns:
            Dictionary with training results and models
        """
        if model_types is None:
            model_types = ['linear_regression', 'random_forest', 'xgboost']
        
        results = {
            'models': {},
            'metrics': {},
            'best_model': None,
            'best_score': 0  # Using R² score for regression
        }
        
        # Define model configurations
        model_configs = {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {'regressor__n_estimators': [100, 200], 'regressor__max_depth': [10, 20]}
            },
            'xgboost': {
                'model': XGBRegressor(random_state=42),
                'params': {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.1, 0.01]}
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.1, 0.01]}
            },
            'svm': {
                'model': SVR(),
                'params': {'regressor__C': [0.1, 1, 10], 'regressor__kernel': ['linear', 'rbf']}
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {'regressor__max_depth': [10, 20, None], 'regressor__min_samples_split': [2, 5]}
            }
        }
        
        if use_mlflow:
            mlflow.set_experiment("Max_EMI_Regression")
        
        for model_type in model_types:
            if model_type not in model_configs:
                self.logger.warning(f"Unknown model type: {model_type}")
                continue
            
            try:
                self.logger.info(f"Training {model_type}...")
                
                if use_mlflow:
                    with mlflow.start_run(run_name=model_type):
                        model, metrics = self._train_single_regression_model(
                            model_configs[model_type], preprocessor,
                            X_train, X_test, y_train, y_test,
                            use_mlflow=True
                        )
                else:
                    model, metrics = self._train_single_regression_model(
                        model_configs[model_type], preprocessor,
                        X_train, X_test, y_train, y_test,
                        use_mlflow=False
                    )
                
                # Store results
                results['models'][model_type] = model
                results['metrics'][model_type] = metrics
                
                # Update best model (using R² score)
                if metrics['r2_score'] > results['best_score']:
                    results['best_score'] = metrics['r2_score']
                    results['best_model'] = model_type
                
                self.logger.info(f"{model_type} trained with R²: {metrics['r2_score']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_type}: {str(e)}")
                continue
        
        return results
    
    def _train_single_regression_model(self, model_config, preprocessor,
                                     X_train, X_test, y_train, y_test, 
                                     use_mlflow: bool = True):
        """Train a single regression model"""
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model_config['model'])
        ])
        
        # Hyperparameter tuning if parameters are specified
        if model_config['params']:
            grid_search = GridSearchCV(
                pipeline, 
                model_config['params'], 
                cv=5, 
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
        else:
            # For models without hyperparameters (like Linear Regression)
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
            cv_score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2').mean()
        
        # Predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100,
            'best_params': best_params,
            'cv_score': cv_score
        }
        
        # Log to MLflow
        if use_mlflow:
            if best_params:
                mlflow.log_params(best_params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != 'best_params'})
            mlflow.sklearn.log_model(best_model, "model")
            
            # Log feature importance if available
            if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
                feature_importance = dict(zip(
                    [f"feature_{i}" for i in range(len(best_model.named_steps['regressor'].feature_importances_))],
                    best_model.named_steps['regressor'].feature_importances_
                ))
                mlflow.log_dict(feature_importance, "feature_importance.json")
        
        return best_model, metrics
    
    def evaluate_model_performance(self, model, X_test, y_test, model_type: str = 'classification'):
        """
        Comprehensive model performance evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_type: Type of model ('classification' or 'regression')
            
        Returns:
            Dictionary with evaluation results
        """
        evaluation = {}
        
        try:
            # Predictions
            if model_type == 'classification':
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Classification metrics
                evaluation['accuracy'] = accuracy_score(y_test, y_pred)
                evaluation['precision'] = precision_score(y_test, y_pred, average='weighted')
                evaluation['recall'] = recall_score(y_test, y_pred, average='weighted')
                evaluation['f1_score'] = f1_score(y_test, y_pred, average='weighted')
                evaluation['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                
                # Confusion matrix
                evaluation['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
                
                # Classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                evaluation['classification_report'] = report
                
            else:  # regression
                y_pred = model.predict(X_test)
                
                # Regression metrics
                evaluation['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                evaluation['mae'] = mean_absolute_error(y_test, y_pred)
                evaluation['r2_score'] = r2_score(y_test, y_pred)
                evaluation['mape'] = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100
                
                # Residual analysis
                residuals = y_test - y_pred
                evaluation['residual_mean'] = np.mean(residuals)
                evaluation['residual_std'] = np.std(residuals)
            
            self.logger.info(f"Model evaluation completed for {model_type}")
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            evaluation['error'] = str(e)
        
        return evaluation
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        try:
            # Get the actual estimator from the pipeline
            if hasattr(model, 'named_steps'):
                estimator = model.named_steps[list(model.named_steps.keys())[-1]]
            else:
                estimator = model
            
            if hasattr(estimator, 'feature_importances_'):
                importance_scores = estimator.feature_importances_
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_scores
                }).sort_values('importance', ascending=False)
                
                return importance_df
            else:
                self.logger.warning("Model does not have feature_importances_ attribute")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error extracting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def create_ensemble_model(self, models: Dict[str, Any], model_type: str = 'classification'):
        """
        Create ensemble model from multiple trained models
        
        Args:
            models: Dictionary of trained models
            model_type: Type of ensemble ('classification' or 'regression')
            
        Returns:
            Ensemble model
        """
        try:
            from sklearn.ensemble import VotingClassifier, VotingRegressor
            
            if model_type == 'classification':
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in models.items()],
                    voting='soft'
                )
            else:
                ensemble = VotingRegressor(
                    estimators=[(name, model) for name, model in models.items()]
                )
            
            self.logger.info(f"Created {model_type} ensemble with {len(models)} models")
            return ensemble
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble model: {str(e)}")
            raise
    
    def save_training_results(self, results: Dict[str, Any], model_type: str, 
                            feature_names: List[str], output_dir: str = "models"):
        """
        Save training results and models
        
        Args:
            results: Training results dictionary
            model_type: Type of models ('classification' or 'regression')
            feature_names: List of feature names
            output_dir: Output directory
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"{output_dir}/{model_type}_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save individual models
            for model_name, model in results['models'].items():
                model_path = f"{save_dir}/{model_name}.pkl"
                joblib.dump(model, model_path)
            
            # Save metrics
            metrics_path = f"{save_dir}/metrics.json"
            with open(metrics_path, 'w') as f:
                import json
                # Convert numpy types to Python types for JSON serialization
                metrics_serializable = {}
                for model_name, metrics in results['metrics'].items():
                    metrics_serializable[model_name] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in metrics.items()
                    }
                json.dump(metrics_serializable, f, indent=2)
            
            # Save feature names
            features_path = f"{save_dir}/features.json"
            with open(features_path, 'w') as f:
                json.dump(feature_names, f, indent=2)
            
            # Save summary
            summary = {
                'best_model': results.get('best_model'),
                'best_score': float(results.get('best_score', 0)),
                'timestamp': timestamp,
                'model_type': model_type,
                'total_models': len(results['models'])
            }
            summary_path = f"{save_dir}/summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Training results saved to {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving training results: {str(e)}")
            raise

# Create model trainer instance
model_trainer = ModelTrainer()