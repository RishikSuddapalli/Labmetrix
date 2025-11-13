# pages/2_🤖_Model_Training.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mlflow
import mlflow.sklearn
from streamlit_config import config
from src.ui.navigation import render_sidebar_nav
from src.model_training import ModelTrainer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Model Training & Management",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'current_experiment' not in st.session_state:
    st.session_state.current_experiment = None
if 'mlflow_runs' not in st.session_state:
    st.session_state.mlflow_runs = []

# Initialize model trainer
try:
    model_trainer = ModelTrainer(experiment_name="EMI_Prediction_Models")
except Exception as e:
    st.error(f"Failed to initialize model trainer: {str(e)}")
    st.stop()

def validate_training_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate data for model training
    
    Args:
        data: Input DataFrame to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if data is None or not isinstance(data, pd.DataFrame):
        return False, "❌ No valid data found. Please load data first."
    
    if data.empty:
        return False, "❌ The dataset is empty. Please check your data source."
    
    # Required columns
    required_columns = [
        'age', 'monthly_salary', 'credit_score', 'requested_amount',
        'requested_tenure', 'emi_eligibility', 'max_monthly_emi'
    ]
    
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        return False, f"❌ Missing required columns: {', '.join(missing)}"
    
    # Check for missing values
    missing_values = data[required_columns].isnull().sum()
    if missing_values.sum() > 0:
        missing_cols = missing_values[missing_values > 0].index.tolist()
        return False, f"❌ Missing values found in columns: {', '.join(missing_cols)}"
    
    return True, "✅ Data is ready for model training"

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Return available models with their default parameters"""
    return {
        "classification": {
            "Logistic Regression": {"C": 1.0, "max_iter": 1000},
            "Random Forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
            "Gradient Boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}
        },
        "regression": {
            "Linear Regression": {},
            "Random Forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
            "Gradient Boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}
        }
    }

def plot_metrics(metrics: Dict[str, float], title: str = "Model Metrics") -> None:
    """Plot model metrics using Plotly"""
    if not metrics:
        return
        
    fig = px.bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        title=title,
        labels={'x': 'Metric', 'y': 'Score'},
        color=list(metrics.keys())
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    st.pyplot(fig)

@st.cache_data
def get_mlflow_runs(experiment_name: str = None) -> List[Dict[str, Any]]:
    """Get MLflow runs for the specified experiment"""
    try:
        client = mlflow.tracking.MlflowClient()
        if experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
            if not experiment:
                return []
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attributes.start_time DESC"]
            )
        else:
            runs = client.search_runs(experiment_ids=[])
            
        return [{
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'metrics': run.data.metrics,
            'params': run.data.params,
            'tags': run.data.tags
        } for run in runs]
    except Exception as e:
        st.error(f"Error fetching MLflow runs: {str(e)}")
        return []

def show_model_training(data: pd.DataFrame) -> None:
    """Show model training interface"""
    st.header("🎯 Train New Model")
    
    if data is None or data.empty:
        st.warning("Please load data first in the Data Analysis page.")
        return
    
    # Data validation
    is_valid, message = validate_training_data(data)
    if not is_valid:
        st.error(message)
        return
    
    # Model type selection
    model_type = st.radio("Select Model Type", ["Classification", "Regression"])
    
    # Target selection
    target_col = st.selectbox(
        "Select Target Variable",
        ["emi_eligibility"] if model_type == "Classification" else ["max_monthly_emi"]
    )
    
    # Feature selection
    feature_cols = st.multiselect(
        "Select Features",
        [col for col in data.columns if col != target_col],
        default=[col for col in data.columns if col != target_col]
    )
    
    # Model selection
    available_models = get_available_models()["classification" if model_type == "Classification" else "regression"]
    selected_model = st.selectbox("Select Model", list(available_models.keys()))
    
    # Hyperparameter tuning
    st.subheader("Hyperparameters")
    params = {}
    default_params = available_models[selected_model]
    
    for param, default_val in default_params.items():
        if isinstance(default_val, bool):
            params[param] = st.checkbox(param, value=default_val, key=f"param_{param}")
        elif isinstance(default_val, int):
            params[param] = st.number_input(
                param, 
                value=default_val, 
                min_value=1, 
                step=1,
                key=f"param_{param}"
            )
        elif isinstance(default_val, float):
            params[param] = st.number_input(
                param, 
                value=default_val, 
                min_value=0.0, 
                step=0.01,
                key=f"param_{param}"
            )
    
    # Train/test split
    st.subheader("Train/Test Split")
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random State", 42)
    
    # MLflow options
    st.subheader("Experiment Tracking")
    use_mlflow = st.checkbox("Enable MLflow Tracking", True)
    experiment_name = st.text_input("Experiment Name", f"EMI_{model_type}_{datetime.now().strftime('%Y%m%d')}")
    
    # Train button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Prepare data
                X = data[feature_cols]
                y = data[target_col]
                
                # Convert categorical features
                categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                        ],
                        remainder='passthrough'
                    )
                    X = preprocessor.fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y if model_type == "Classification" else None
                )
                
                # Train model
                if model_type == "Classification":
                    model, metrics = model_trainer.train_classification_model(
                        X_train, y_train, X_test, y_test,
                        model_name=selected_model,
                        params=params
                    )
                else:
                    model, metrics = model_trainer.train_regression_model(
                        X_train, y_train, X_test, y_test,
                        model_name=selected_model,
                        params=params
                    )
                
                # Store model in session state
                model_key = f"{model_type}_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.trained_models[model_key] = {
                    'model': model,
                    'type': model_type.lower(),
                    'metrics': metrics,
                    'features': feature_cols,
                    'target': target_col,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.success("✅ Model trained successfully!")
                
                # Show metrics
                st.subheader("Model Performance")
                plot_metrics(metrics)
                
                # For classification, show confusion matrix
                if model_type == "Classification":
                    y_pred = model.predict(X_test)
                    plot_confusion_matrix(y_test, y_pred)
                
                # For regression, show actual vs predicted
                else:
                    y_pred = model.predict(X_test)
                    fig = px.scatter(
                        x=y_test, y=y_pred,
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title='Actual vs Predicted Values'
                    )
                    fig.add_shape(
                        type="line",
                        x0=y_test.min(), y0=y_test.min(),
                        x1=y_test.max(), y1=y_test.max(),
                        line=dict(color="red", dash="dash")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
                st.exception(e)

def show_model_registry():
    """Show MLflow model registry"""
    st.header("📚 Model Registry")
    
    try:
        # Get all experiments
        experiments = mlflow.search_experiments()
        experiment_names = [exp.name for exp in experiments]
        
        if not experiment_names:
            st.info("No experiments found in MLflow")
            return
        
        # Select experiment
        selected_exp = st.selectbox("Select Experiment", experiment_names)
        
        # Get runs for selected experiment
        runs = get_mlflow_runs(selected_exp)
        
        if not runs:
            st.info(f"No runs found for experiment: {selected_exp}")
            return
        
        # Show runs in a table
        runs_df = pd.DataFrame([{
            'Run ID': run['run_id'],
            'Start Time': datetime.fromtimestamp(run['start_time']/1000).strftime('%Y-%m-%d %H:%M:%S'),
            'Status': run['status'],
            'Model': run['params'].get('model_name', 'N/A'),
            'Metrics': ', '.join([f"{k}: {v:.4f}" for k, v in run['metrics'].items()])
        } for run in runs])
        
        st.dataframe(runs_df, use_container_width=True)
        
        # Show run details
        if st.checkbox("Show Run Details"):
            selected_run_id = st.selectbox("Select Run", runs_df['Run ID'].tolist())
            selected_run = next(run for run in runs if run['run_id'] == selected_run_id)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Parameters")
                st.json(selected_run['params'])
                
            with col2:
                st.subheader("Metrics")
                st.json(selected_run['metrics'])
                
            # Show artifacts if available
            if 'artifacts' in selected_run:
                st.subheader("Artifacts")
                for artifact in selected_run['artifacts']:
                    st.write(f"- {artifact}")
    
    except Exception as e:
        st.error(f"Error accessing MLflow: {str(e)}")

def show_deployment():
    """Show model deployment interface"""
    st.header("🚀 Deploy Model")
    
    if not st.session_state.trained_models:
        st.warning("No models available for deployment. Please train a model first.")
        return
    
    # Select model to deploy
    model_options = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Select Model to Deploy", model_options)
    
    model_info = st.session_state.trained_models[selected_model]
    
    # Show model info
    st.subheader("Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Type", model_info['type'].capitalize())
        st.metric("Target Variable", model_info['target'])
    
    with col2:
        st.metric("Training Date", model_info['timestamp'].split('T')[0])
        st.metric("Number of Features", len(model_info['features']))
    
    # Deployment options
    st.subheader("Deployment Options")
    deployment_target = st.radio(
        "Deploy to",
        ["Local File System", "MLflow Model Registry", "REST API"]
    )
    
    if deployment_target == "Local File System":
        model_name = st.text_input("Model Name", f"{model_info['type']}_model")
        if st.button("💾 Save Model"):
            try:
                os.makedirs("models", exist_ok=True)
                model_path = f"models/{model_name}.pkl"
                joblib.dump(model_info['model'], model_path)
                st.success(f"✅ Model saved to {model_path}")
            except Exception as e:
                st.error(f"❌ Error saving model: {str(e)}")
    
    elif deployment_target == "MLflow Model Registry":
        model_name = st.text_input("Model Name", f"emi_{model_info['type']}_model")
        if st.button("📤 Register Model"):
            try:
                with mlflow.start_run():
                    mlflow.sklearn.log_model(
                        model_info['model'],
                        "model",
                        registered_model_name=model_name
                    )
                    mlflow.log_metrics(model_info['metrics'])
                    mlflow.set_tag("model_type", model_info['type'])
                st.success("✅ Model registered successfully!")
            except Exception as e:
                st.error(f"❌ Error registering model: {str(e)}")
    
    elif deployment_target == "REST API":
        st.warning("REST API deployment is not yet implemented.")

def main():
    # Sidebar navigation
    st.title("🤖 Machine Learning Model Training")
    st.markdown("Train and evaluate classification and regression models for financial risk assessment")
    
    # Consistent sidebar navigation - only once at the top
    render_sidebar_nav("🤖 Model Training")
    
    # Check if data is loaded in session state
    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        st.warning("⚠️ No data loaded. Please load your data from the main dashboard first.")
        if st.button("Go to Dashboard"):
            st.switch_page("app.py")
        return
    
    # Get data from session state
    data = st.session_state.current_data
    
    # Show model training tabs
    page = st.sidebar.radio(
        "Model Operations",
        ["Train Model", "Model Registry", "Deploy Model"]
    )
    
    # Validate the data for training before showing any model pages
    is_valid, message = validate_training_data(data)
    if not is_valid:
        st.error(f"❌ {message}")
        st.warning("Please ensure your dataset contains all required columns and try again.")
        
        if st.button("View Data Analysis"):
            st.switch_page("pages/1_📊_Data_Analysis.py")
        return
    
    # Show the selected page only if data is valid
    if page == "Train Model":
        show_model_training(data)
    elif page == "Model Registry":
        show_model_registry()
    elif page == "Deploy Model":
        show_deployment()
    
    # Show data source info
    data_source = "Sample Data" if st.session_state.get('using_sample_data', False) \
                 else st.session_state.get('data_path', 'Unknown source')
    
    with st.sidebar:
        st.info(f"**Data Source:** {data_source}")
        st.info(f"**Records:** {len(data):,}")
        st.info(f"**Features:** {len(data.columns)}")
        
        # Data quality indicators
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            st.warning(f"⚠️ {missing_values} missing values detected")
        
        # Check for class imbalance in target
        if 'emi_eligibility' in data.columns:
            class_balance = data['emi_eligibility'].value_counts(normalize=True) * 100
            st.info(f"**Class Balance (Eligibility):**")
            for cls, pct in class_balance.items():
                st.info(f"- {cls}: {pct:.1f}%")
    
    # Initialize MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Data Preparation", 
        "🎯 Classification", 
        "📈 Regression", 
        "📊 MLflow Tracking"
    ])
    
    with tab1:
        show_data_preparation(data)
    
    with tab2:
        show_classification_training(data)
    
    with tab3:
        show_regression_training(data)
    
    with tab4:
        show_mlflow_tracking()

def show_data_preparation(data):
    st.header("🔧 Data Preparation for Model Training")
    
    st.subheader("Feature Selection")
    
    # Available features strictly from config and present in data
    from streamlit_config import config
    configured_num = [c for c in config.NUMERICAL_FEATURES if c in data.columns]
    configured_cat = [c for c in config.CATEGORICAL_FEATURES if c in data.columns]
    
    # Remove targets from selection lists
    numerical_features = [c for c in configured_num if c != 'max_monthly_emi']
    categorical_features = [c for c in configured_cat if c != 'emi_eligibility']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Features**")
        selected_numerical = st.multiselect(
            "Select numerical features:",
            numerical_features,
            default=numerical_features[:5] if len(numerical_features) > 5 else numerical_features,
            key="num_features"
        )
    
    with col2:
        st.write("**Categorical Features**")
        selected_categorical = st.multiselect(
            "Select categorical features:",
            categorical_features,
            default=categorical_features[:3] if len(categorical_features) > 3 else categorical_features,
            key="cat_features"
        )
    
    # Feature engineering options
    st.subheader("Feature Engineering Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scale_features = st.checkbox("Scale Numerical Features", value=True)
        encode_categorical = st.checkbox("Encode Categorical Features", value=True)
    
    with col2:
        handle_imbalance = st.checkbox("Handle Class Imbalance", value=True)
        create_interactions = st.checkbox("Create Interaction Features", value=False)
    
    with col3:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        random_state = st.number_input("Random State", value=42)
    
    # Data preview
    if st.button("Prepare Data for Training", type="primary"):
        with st.spinner("Preparing data..."):
            try:
                # Prepare features and targets
                X = data[selected_numerical + selected_categorical].copy()
                y_classification = data['emi_eligibility']
                # Standardize regression target
                y_regression = data['max_monthly_emi']
                
                # Store in session state
                st.session_state.X = X
                st.session_state.y_classification = y_classification
                st.session_state.y_regression = y_regression
                st.session_state.feature_names = selected_numerical + selected_categorical
                
                st.success("Data prepared successfully!")
                
                # Show data summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Features", len(selected_numerical + selected_categorical))
                with col2:
                    st.metric("Training Samples", f"{(100-test_size)}%")
                with col3:
                    st.metric("Test Samples", f"{test_size}%")
                
                # Show feature importance preview (using correlation with targets)
                st.subheader("Feature-Target Relationships")
                
                # For classification
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_classification)
                
                temp_data = X.select_dtypes(include=[np.number]).copy()
                temp_data['target'] = y_encoded
                
                classification_corr = temp_data.corr()['target'].abs().sort_values(ascending=False)
                classification_corr = classification_corr[classification_corr.index != 'target']
                
                fig1 = px.bar(
                    x=classification_corr.head(10).values,
                    y=classification_corr.head(10).index,
                    orientation='h',
                    title='Top Features for Classification (Correlation with EMI Eligibility)',
                    labels={'x': 'Absolute Correlation', 'y': 'Feature'}
                )
                st.plotly_chart(fig1, width='stretch')
                
                # For regression
                regression_corr = X.select_dtypes(include=[np.number]).corrwith(y_regression).abs().sort_values(ascending=False)
                
                fig2 = px.bar(
                    x=regression_corr.head(10).values,
                    y=regression_corr.head(10).index,
                    orientation='h',
                    title='Top Features for Regression (Correlation with Max EMI)',
                    labels={'x': 'Absolute Correlation', 'y': 'Feature'}
                )
                st.plotly_chart(fig2, width='stretch')
                
            except Exception as e:
                st.error(f"Error preparing data: {str(e)}")

def show_classification_training(data):
    st.header("🎯 EMI Eligibility Classification")
    
    if 'X' not in st.session_state:
        st.warning("Please prepare data first in the Data Preparation tab.")
        return
    
    X = st.session_state.X
    y = st.session_state.y_classification
    
    st.subheader("Model Selection and Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Select Classification Models**")
        
        models_config = {
            "Logistic Regression": st.checkbox("Logistic Regression", value=True),
            "Random Forest": st.checkbox("Random Forest", value=True),
            "XGBoost": st.checkbox("XGBoost", value=True),
            "Support Vector Machine": st.checkbox("SVM", value=False),
            "Decision Tree": st.checkbox("Decision Tree", value=False)
        }
        
        selected_models = [model for model, selected in models_config.items() if selected]
        
        if not selected_models:
            st.warning("Please select at least one model to train.")
            return
    
    with col2:
        st.write("**Training Configuration**")
        
        test_size = st.slider("Test Size (%)", 10, 40, 20, key="cls_test_size")
        cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)
        use_mlflow = st.checkbox("Log to MLflow", value=True)
    
    # Model hyperparameters
    st.subheader("Model Hyperparameters")
    
    hyperparams = {}
    
    if "Logistic Regression" in selected_models:
        with st.expander("Logistic Regression Parameters"):
            hyperparams['logistic_regression'] = {
                'C': st.number_input("C (Inverse Regularization)", value=1.0, key='lr_c'),
                'max_iter': st.number_input("Max Iterations", value=1000, key='lr_max_iter')
            }
    
    if "Random Forest" in selected_models:
        with st.expander("Random Forest Parameters"):
            hyperparams['random_forest'] = {
                'n_estimators': st.number_input("Number of Trees", value=100, key='rf_n_est'),
                'max_depth': st.number_input("Max Depth", value=10, key='rf_max_depth'),
                'min_samples_split': st.number_input("Min Samples Split", value=2, key='rf_min_split')
            }
    
    if "XGBoost" in selected_models:
        with st.expander("XGBoost Parameters"):
            hyperparams['xgboost'] = {
                'n_estimators': st.number_input("Number of Trees", value=100, key='xgb_n_est'),
                'max_depth': st.number_input("Max Depth", value=6, key='xgb_max_depth'),
                'learning_rate': st.number_input("Learning Rate", value=0.1, key='xgb_lr')
            }
    
    # Train models
    if st.button("Train Classification Models", type="primary"):
        with st.spinner("Training classification models..."):
            try:
                results = train_classification_models(
                    X, y, selected_models, hyperparams, test_size/100, 
                    cv_folds, use_mlflow
                )
                
                # Display results
                st.subheader("📊 Classification Results")
                
                # Results table
                results_df = pd.DataFrame(results)
                if results_df.empty:
                    st.error("No classification models produced results. Please adjust settings and retry.")
                    return
                st.dataframe(results_df, use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Accuracy comparison
                    fig = px.bar(
                        results_df,
                        x='Model',
                        y='Accuracy',
                        title='Model Accuracy Comparison',
                        color='Accuracy',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # F1-Score comparison
                    fig = px.bar(
                        results_df,
                        x='Model',
                        y='F1-Score',
                        title='Model F1-Score Comparison',
                        color='F1-Score',
                        color_continuous_scale='plasma'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Best model
                best_model_idx = results_df['Accuracy'].astype(float).idxmax()
                best_model = results_df.iloc[best_model_idx]
                
                st.success(f"🎉 Best Model: **{best_model['Model']}** with Accuracy: **{best_model['Accuracy']}**")
                
                # Store best model in session state
                st.session_state.best_classification_model = best_model['Model']
                st.session_state.classification_results = results_df
                
            except Exception as e:
                st.error(f"Error training classification models: {str(e)}")

def show_regression_training(data):
    st.header("📈 Maximum EMI Amount Regression")
    
    if 'X' not in st.session_state:
        st.warning("Please prepare data first in the Data Preparation tab.")
        return
    
    X = st.session_state.X
    y = st.session_state.y_regression
    
    st.subheader("Model Selection and Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Select Regression Models**")
        
        models_config = {
            "Linear Regression": st.checkbox("Linear Regression", value=True, key='lin_reg'),
            "Random Forest": st.checkbox("Random Forest", value=True, key='rf_reg'),
            "XGBoost": st.checkbox("XGBoost", value=True, key='xgb_reg'),
            "Support Vector Regressor": st.checkbox("SVR", value=False, key='svr_reg'),
            "Decision Tree": st.checkbox("Decision Tree", value=False, key='dt_reg')
        }
        
        selected_models = [model for model, selected in models_config.items() if selected]
        
        if not selected_models:
            st.warning("Please select at least one model to train.")
            return
    
    with col2:
        st.write("**Training Configuration**")
        
        test_size = st.slider("Test Size (%)", 10, 40, 20, key="reg_test_size")
        cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5, key="reg_cv")
        use_mlflow = st.checkbox("Log to MLflow", value=True, key="reg_mlflow")
    
    # Model hyperparameters
    st.subheader("Model Hyperparameters")
    
    hyperparams = {}
    
    if "Random Forest" in selected_models:
        with st.expander("Random Forest Regressor Parameters"):
            hyperparams['random_forest'] = {
                'n_estimators': st.number_input("Number of Trees", value=100, key='rf_reg_n_est'),
                'max_depth': st.number_input("Max Depth", value=10, key='rf_reg_max_depth'),
                'min_samples_split': st.number_input("Min Samples Split", value=2, key='rf_reg_min_split')
            }
    
    if "XGBoost" in selected_models:
        with st.expander("XGBoost Regressor Parameters"):
            hyperparams['xgboost'] = {
                'n_estimators': st.number_input("Number of Trees", value=100, key='xgb_reg_n_est'),
                'max_depth': st.number_input("Max Depth", value=6, key='xgb_reg_max_depth'),
                'learning_rate': st.number_input("Learning Rate", value=0.1, key='xgb_reg_lr')
            }
    
    # Train models
    if st.button("Train Regression Models", type="primary"):
        with st.spinner("Training regression models..."):
            try:
                results = train_regression_models(
                    X, y, selected_models, hyperparams, test_size/100, 
                    cv_folds, use_mlflow
                )
                
                # Display results
                st.subheader("📊 Regression Results")
                
                # Results table
                results_df = pd.DataFrame(results)
                if results_df.empty:
                    st.error("No regression models produced results. Please adjust settings and retry.")
                    return
                st.dataframe(results_df, width='stretch')
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # RMSE comparison (lower is better)
                    fig = px.bar(
                        results_df,
                        x='Model',
                        y='RMSE',
                        title='Model RMSE Comparison (Lower is Better)',
                        color='RMSE',
                        color_continuous_scale='viridis_r'
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    # R² comparison (higher is better)
                    fig = px.bar(
                        results_df,
                        x='Model',
                        y='R² Score',
                        title='Model R² Score Comparison (Higher is Better)',
                        color='R² Score',
                        color_continuous_scale='plasma'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Best model
                best_model_idx = results_df['R² Score'].astype(float).idxmax()
                best_model = results_df.iloc[best_model_idx]
                
                st.success(f"🎉 Best Model: **{best_model['Model']}** with R² Score: **{best_model['R² Score']}**")
                
                # Store best model in session state
                st.session_state.best_regression_model = best_model['Model']
                st.session_state.regression_results = results_df
                
            except Exception as e:
                st.error(f"Error training regression models: {str(e)}")

def show_mlflow_tracking():
    st.header("📊 MLflow Experiment Tracking")
    
    st.info("""
    MLflow tracks all model experiments, including parameters, metrics, and artifacts. 
    You can compare model performance and manage model versions through the MLflow UI.
    """)
    
    # MLflow status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("MLflow Status")
        
        if os.path.exists("mlruns"):
            st.success("✅ MLflow tracking is active")
            
            # Count experiments
            try:
                experiments = mlflow.search_experiments()
                st.metric("Number of Experiments", len(experiments))
                
                # Get recent runs
                recent_runs = mlflow.search_runs(experiment_ids=[e.experiment_id for e in experiments], 
                                               max_results=5)
                if not recent_runs.empty:
                    st.subheader("Recent Runs")
                    st.dataframe(recent_runs[['experiment_id', 'run_id', 'start_time', 'tags.mlflow.runName']], 
                               use_container_width=True)
            except:
                st.warning("Could not fetch MLflow experiments")
        
        else:
            st.warning("MLflow directory not found. Run model training to create experiments.")
    
    with col2:
        st.subheader("MLflow Actions")
        
        if st.button("Open MLflow UI"):
            st.code("mlflow ui --port 5000")
            st.info("Run the above command in your terminal to open the MLflow dashboard")
        
        if st.button("View Experiment Comparison"):
            show_experiment_comparison()
    
    # Experiment comparison
    st.subheader("Experiment Comparison")
    
    try:
        # Get all experiments and their runs
        experiments = mlflow.search_experiments()
        all_runs = []
        
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not runs.empty:
                runs['experiment_name'] = exp.name
                all_runs.append(runs)
        
        if all_runs:
            combined_runs = pd.concat(all_runs, ignore_index=True)
            
            # Filter relevant columns
            metric_cols = [col for col in combined_runs.columns if col.startswith('metrics.')]
            param_cols = [col for col in combined_runs.columns if col.startswith('params.')]
            
            display_cols = ['experiment_name', 'run_id', 'start_time'] + metric_cols[:5]  # Show first 5 metrics
            
            st.dataframe(combined_runs[display_cols], width='stretch')
        else:
            st.info("No MLflow runs found. Train models to see experiments here.")
    
    except Exception as e:
        st.warning(f"Could not load MLflow experiments: {str(e)}")

def train_classification_models(X, y, models, hyperparams, test_size, cv_folds, use_mlflow):
    """Train classification models and return results"""
    
    # Preprocessing
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    results = []
    
    # MLflow experiment setup
    if use_mlflow:
        try:
            mlflow.set_experiment("EMI_DUAL_ML")
            # Ensure tracking URI is set
            if not mlflow.tracking.get_tracking_uri():
                mlflow.set_tracking_uri("mlruns")  # Local file-based tracking
        except Exception as e:
            st.warning(f"MLflow setup warning: {str(e)}. Continuing without MLflow tracking.")
            use_mlflow = False
    
    for model_name in models:
        try:
            # Create model and pipeline
            model = create_classification_model(model_name, hyperparams)
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # MLflow run context
            if use_mlflow:
                try:
                    with mlflow.start_run(run_name=model_name, nested=True) as run:
                        # Train model
                        pipeline.fit(X_train, y_train)
                        
                        # Get the run ID immediately after starting the run
                        run_id = run.info.run_id
                        
                        # Predictions
                        y_pred = pipeline.predict(X_test)
                        y_pred_proba = pipeline.predict_proba(X_test)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                        
                        # Additional metrics for classification
                        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
                        mae = mean_absolute_error(y_test, y_pred)
                        mape = mean_absolute_percentage_error(y_test, y_pred)
                        
                        # Log to MLflow
                        mlflow.log_params(model.get_params())
                        mlflow.log_metrics({
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1,
                            "roc_auc": auc,
                            "mae": mae,
                            "mape": mape,
                            "r2_score": r2_score(y_test, y_pred)
                        })
                        
                        # Log model with consistent naming
                        model_name_clean = model_name.replace(" ", "_")
                        mlflow.sklearn.log_model(
                            pipeline, 
                            f"{model_name_clean}_Classification",
                            registered_model_name=f"{model_name_clean}_Classification"
                        )
                        
                        # Store run info in results
                        results.append({
                            "Model": model_name,
                            "Accuracy": f"{accuracy:.4f}",
                            "Precision": f"{precision:.4f}",
                            "Recall": f"{recall:.4f}",
                            "F1-Score": f"{f1:.4f}",
                            "ROC-AUC": f"{auc:.4f}",
                            "MAE": f"{mae:.4f}",
                            "MAPE": f"{mape:.2%}",
                            "R²": f"{r2_score(y_test, y_pred):.4f}",
                            "MLflow Run ID": run_id
                        })
                        
                        # Store model in session state
                        st.session_state[f'{model_name.lower().replace(" ", "_")}_cls'] = pipeline
                        
                except Exception as e:
                    st.error(f"MLflow error in {model_name}: {str(e)}")
                    # Fall back to non-MLflow training
                    use_mlflow = False
                    
            # Non-MLflow training or fallback
            if not use_mlflow:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                results.append({
                    "Model": model_name,
                    "Accuracy": f"{accuracy:.4f}",
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                    "F1-Score": f"{f1:.4f}",
                    "ROC-AUC": f"{auc:.4f}",
                    "MAE": f"{mae:.4f}",
                    "MAPE": f"{mape:.2%}",
                    "R²": f"{r2_score(y_test, y_pred):.4f}",
                    "MLflow Run ID": "Not tracked"
                })
                
                st.session_state[f'{model_name.lower().replace(" ", "_")}_cls'] = pipeline
            else:
                # Train without MLflow
                model = create_classification_model(model_name, hyperparams)
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                results.append({
                    "Model": model_name,
                    "Accuracy": f"{accuracy:.4f}",
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                    "F1-Score": f"{f1:.4f}",
                    "ROC-AUC": f"{auc:.4f}",
                    "MAE": f"{mae:.4f}",
                    "MAPE": f"{mape:.2%}",
                    "R²": f"{r2_score(y_test, y_pred):.4f}"
                })
                
                st.session_state[f'{model_name.lower().replace(" ", "_")}_cls'] = pipeline
                
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            continue
    
    return results

def train_regression_models(X, y, models, hyperparams, test_size, cv_folds, use_mlflow):
    """Train regression models and return results"""
    
    # Preprocessing
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    results = []
    
    # MLflow experiment
    if use_mlflow:
        mlflow.set_experiment("EMI_DUAL_ML")
    
    for model_name in models:
        try:
            if use_mlflow:
                with mlflow.start_run(run_name=model_name):
                    # Train model
                    model = create_regression_model(model_name, hyperparams)
                    
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', model)
                    ])
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = pipeline.predict(X_test)
                    
                    # Calculate metrics
                    from sklearn.metrics import (
                        mean_squared_error,
                        mean_absolute_error,
                        r2_score,
                        mean_absolute_percentage_error,
                        explained_variance_score,
                        max_error,
                    )
                    
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # as percentage
                    
                    # Additional metrics
                    y_pred_train = pipeline.predict(X_train)
                    train_r2 = r2_score(y_train, y_pred_train)
                    
                    # Log to MLflow
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics({
                        "rmse": rmse,
                        "mae": mae,
                        "r2_score": r2,
                        "mape": mape,
                        "train_r2": train_r2,
                        "explained_variance": explained_variance_score(y_test, y_pred),
                        "max_error": max_error(y_test, y_pred)
                    })
                    
                    # Log model with consistent naming
                    model_name_clean = model_name.replace(" ", "_")
                    mlflow.sklearn.log_model(pipeline, f"{model_name_clean}_Regression")
                    
                    results.append({
                        "Model": model_name,
                        "RMSE": f"₹{rmse:,.0f}",
                        "MAE": f"₹{mae:,.0f}",
                        "R² Score": f"{r2:.4f}",
                        "MAPE": f"{mape:.2f}%",
                        "Train R²": f"{train_r2:.4f}",
                        "Explained Variance": f"{explained_variance_score(y_test, y_pred):.4f}",
                        "Max Error": f"₹{max_error(y_test, y_pred):,.0f}"
                    })
                    
                    # Store model in session state
                    st.session_state[f'{model_name.lower().replace(" ", "_")}_reg'] = pipeline
            else:
                # Train without MLflow
                model = create_regression_model(model_name, hyperparams)
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score, max_error
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                
                # Additional metrics
                y_pred_train = pipeline.predict(X_train)
                train_r2 = r2_score(y_train, y_pred_train)
                
                results.append({
                    "Model": model_name,
                    "RMSE": f"₹{rmse:,.0f}",
                    "MAE": f"₹{mae:,.0f}",
                    "R² Score": f"{r2:.4f}",
                    "MAPE": f"{mape:.2f}%",
                    "Train R²": f"{train_r2:.4f}",
                    "Explained Variance": f"{explained_variance_score(y_test, y_pred):.4f}",
                    "Max Error": f"₹{max_error(y_test, y_pred):,.0f}"
                })
                
                st.session_state[f'{model_name.lower().replace(" ", "_")}_reg'] = pipeline
                
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            continue
    
    return results

def create_classification_model(model_name, hyperparams):
    """Create classification model instance"""
    if model_name == "Logistic Regression":
        params = hyperparams.get('logistic_regression', {})
        return LogisticRegression(**params, random_state=42)
    
    elif model_name == "Random Forest":
        params = hyperparams.get('random_forest', {})
        return RandomForestClassifier(**params, random_state=42)
    
    elif model_name == "XGBoost":
        params = hyperparams.get('xgboost', {})
        return XGBClassifier(**params, random_state=42)
    
    elif model_name == "Support Vector Machine":
        from sklearn.svm import SVC
        return SVC(probability=True, random_state=42)
    
    elif model_name == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=42)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def create_regression_model(model_name, hyperparams):
    """Create regression model instance"""
    if model_name == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    
    elif model_name == "Random Forest":
        params = hyperparams.get('random_forest', {})
        return RandomForestRegressor(**params, random_state=42)
    
    elif model_name == "XGBoost":
        params = hyperparams.get('xgboost', {})
        return XGBRegressor(**params, random_state=42)
    
    elif model_name == "Support Vector Regressor":
        from sklearn.svm import SVR
        return SVR()
    
    elif model_name == "Decision Tree":
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(random_state=42)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def show_experiment_comparison():
    """Show detailed experiment comparison"""
    st.subheader("Detailed Experiment Comparison")
    
    try:
        experiments = mlflow.search_experiments()
        all_runs = []
        
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not runs.empty:
                runs['experiment_name'] = exp.name
                all_runs.append(runs)
        
        if all_runs:
            combined_runs = pd.concat(all_runs, ignore_index=True)
            
            # Create comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy/R² by experiment
                if 'metrics.accuracy' in combined_runs.columns:
                    fig = px.box(combined_runs, x='experiment_name', y='metrics.accuracy', 
                               title='Accuracy Distribution by Experiment')
                    st.plotly_chart(fig, use_container_width=True)
                elif 'metrics.r2_score' in combined_runs.columns:
                    fig = px.box(combined_runs, x='experiment_name', y='metrics.r2_score', 
                               title='R² Score Distribution by Experiment')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Run duration analysis
                if 'start_time' in combined_runs.columns and 'end_time' in combined_runs.columns:
                    combined_runs['duration'] = (pd.to_datetime(combined_runs['end_time']) - 
                                               pd.to_datetime(combined_runs['start_time'])).dt.total_seconds()
                    fig = px.box(combined_runs, x='experiment_name', y='duration', 
                               title='Run Duration by Experiment (seconds)')
                    st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not create experiment comparison: {str(e)}")

if __name__ == "__main__":
    main()