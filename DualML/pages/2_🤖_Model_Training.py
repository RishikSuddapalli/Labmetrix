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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score, max_error
)
import joblib
import os
from datetime import datetime

def validate_training_data(data):
    """Validate data for model training and return (is_valid, message) tuple"""
    if data is None or not isinstance(data, pd.DataFrame):
        return False, "No valid data found. Please load data first."
    
    if data.empty:
        return False, "The dataset is empty. Please check your data source."
    
    # Check for required columns allowing aliases (original vs current dataset)
    required_aliases = {
        'age': ['age'],
        'monthly_salary': ['monthly_salary'],
        'bank_balance': ['bank_balance'],
        'loan_amount': ['loan_amount', 'requested_amount'],
        'cibil_score': ['cibil_score', 'credit_score'],
        'emi_eligibility': ['emi_eligibility'],
        'max_emi': ['max_affordable_emi', 'max_monthly_emi']
    }
    missing_groups = [
        '/'.join(options) for options in required_aliases.values()
        if not any(opt in data.columns for opt in options)
    ]
    if missing_groups:
        return False, f"Missing required columns for training: {', '.join(missing_groups)}"
    
    return True, "Data is ready for model training"

def main():
    st.set_page_config(
        page_title="Model Training - EMIPredict AI",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Machine Learning Model Training")
    st.markdown("Train and evaluate classification and regression models for financial risk assessment")
    # Consistent sidebar navigation
    render_sidebar_nav("🤖 Model Training")
    
    # Check if data is loaded in session state
    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        st.warning("⚠️ No data loaded. Please load your data from the main dashboard first.")
        if st.button("Go to Dashboard"):
            st.switch_page("app.py")
        return
    
    # Get data from session state
    data = st.session_state.current_data
    
    # Validate the data for training
    is_valid, message = validate_training_data(data)
    if not is_valid:
        st.error(f"❌ {message}")
        st.warning("Please ensure your dataset contains all required columns and try again.")
        
        if st.button("View Data Analysis"):
            st.switch_page("pages/1_📊_Data_Analysis.py")
        return
    
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
    
    # Available features
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variables from feature lists
    if 'emi_eligibility' in categorical_features:
        categorical_features.remove('emi_eligibility')
    # Remove whichever regression target exists
    for target_col in ['max_affordable_emi', 'max_monthly_emi']:
        if target_col in numerical_features:
            numerical_features.remove(target_col)
    
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
                # Pick available regression target
                y_regression = data['max_affordable_emi'] if 'max_affordable_emi' in data.columns else data['max_monthly_emi']
                
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
    
    # MLflow experiment
    if use_mlflow:
        mlflow.set_experiment("EMI_DUAL_ML")
    
    for model_name in models:
        try:
            if use_mlflow:
                with mlflow.start_run(run_name=model_name):
                    # Train model
                    model = create_classification_model(model_name, hyperparams)
                    
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', model)
                    ])
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
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
                    mlflow.sklearn.log_model(pipeline, f"{model_name_clean}_Classification")
                    
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
                    
                    # Store model in session state
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
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
                    
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