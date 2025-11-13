# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
import mlflow.sklearn
import joblib
import sqlite3
from datetime import datetime
import os
from streamlit_config import config
from src.ui.navigation import render_sidebar_nav
# Page configuration
st.set_page_config(
    page_title="EMIPredict AI - Financial Risk Assessment",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .success-text { color: #28a745; }
    .warning-text { color: #ffc107; }
    .danger-text { color: #dc3545; }
    /* Hide default Streamlit multipage sidebar navigation */
    [data-testid="stSidebarNav"] { display: none !important; }
    /* Sidebar radio navigation enhancements */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label {
        display: block;
        padding: 8px 12px;
        border-radius: 8px;
        transition: background 0.2s ease, transform 0.15s ease;
        cursor: pointer;
        margin-bottom: 4px;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label:hover {
        background: rgba(31, 119, 180, 0.08);
        transform: translateX(3px);
    }
    /* Highlight the selected tab */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label:has(input:checked) {
        background: rgba(31, 119, 180, 0.15);
        border-left: 4px solid #1f77b4;
        color: #1f77b4;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with default values
def initialize_session_state():
    """Initialize all required session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'data_preprocessor' not in st.session_state:
        from src.data_preprocessing import DataPreprocessor
        st.session_state.data_preprocessor = DataPreprocessor()
    if 'data_path' not in st.session_state:
        st.session_state.data_path = config.DATA_PATH

# Initialize session state
initialize_session_state()

class EMIPredictAI:
    def __init__(self):
        import logging
        self.logger = logging.getLogger(__name__)
        self.data = None
        self.models = {}
        self.mlflow_uri = "mlruns"
        
    def load_sample_data(self):
        """Load the emi_prediction_dataset.csv file"""
        import os
        from pathlib import Path
        
        # Define possible locations for the dataset
        possible_paths = [
            os.path.join('data', 'emi_prediction_dataset.csv'),  # Local development
            os.path.join(os.getcwd(), 'data', 'emi_prediction_dataset.csv'),  # Absolute path
            'emi_prediction_dataset.csv',  # Current directory
            os.path.join(os.path.dirname(__file__), 'data', 'emi_prediction_dataset.csv')  # Module-relative path
        ]
        
        df = None
        found_path = None
        
        # Try each possible path
        for dataset_path in possible_paths:
            try:
                self.logger.info(f"Attempting to load dataset from: {dataset_path}")
                if os.path.exists(dataset_path):
                    df = pd.read_csv(dataset_path)
                    found_path = dataset_path
                    self.logger.info(f"Successfully loaded dataset from {dataset_path}")
                    break
            except Exception as e:
                self.logger.warning(f"Failed to load from {dataset_path}: {str(e)}")
                continue
        
        if df is None:
            error_msg = (
                "Could not find or load 'emi_prediction_dataset.csv' in any of these locations:\n"
                f"{chr(10).join(possible_paths)}\n\n"
                "Please ensure the file exists in one of these locations."
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Sanitize columns and coerce numeric types
        try:
            # Ensure all column names are strings and non-empty
            new_cols = []
            for i, c in enumerate(df.columns):
                name = str(c).strip() if c is not None and str(c).strip() != '' else f"col_{i}"
                new_cols.append(name)
            df.columns = new_cols

            # Coerce expected numeric columns
            numeric_cols = [
                'monthly_salary','years_of_employment','monthly_rent','family_size','dependents',
                'school_fees','college_fees','travel_expenses','groceries_utilities','other_monthly_expenses',
                'current_emi_amount','credit_score','bank_balance','emergency_fund',
                'requested_amount','requested_tenure','max_monthly_emi'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Fill numeric NaNs where appropriate
            fill_zeros = [c for c in numeric_cols if c in df.columns and c != 'credit_score']
            for c in fill_zeros:
                df[c] = df[c].fillna(0)
            if 'credit_score' in df.columns:
                df['credit_score'] = df['credit_score'].fillna(0)

            # Compute ratios/targets if missing
            if 'max_monthly_emi' not in df.columns or df['max_monthly_emi'].isna().all():
                if all(c in df.columns for c in [
                    'monthly_salary','monthly_rent','school_fees','college_fees',
                    'travel_expenses','groceries_utilities','other_monthly_expenses','current_emi_amount'
                ]):
                    total_exp = (
                        df['monthly_rent'] + df['school_fees'] + df['college_fees'] +
                        df['travel_expenses'] + df['groceries_utilities'] + df['other_monthly_expenses'] +
                        df['current_emi_amount']
                    )
                    disposable = (df['monthly_salary'] - total_exp).clip(lower=0)
                    df['max_monthly_emi'] = (disposable * 0.4).clip(lower=500)

            if 'emi_eligibility' not in df.columns and all(c in df.columns for c in ['monthly_salary','credit_score']):
                total_exp = 0
                for c in ['monthly_rent','school_fees','college_fees','travel_expenses','groceries_utilities','other_monthly_expenses','current_emi_amount']:
                    if c in df.columns:
                        total_exp = total_exp + df[c]
                disposable = (df['monthly_salary'] - total_exp).fillna(0)
                cond1 = (disposable > 0.4 * df['monthly_salary']) & (df['credit_score'] > 700)
                cond2 = (disposable > 0.2 * df['monthly_salary']) & (df['credit_score'] > 600)
                df['emi_eligibility'] = np.select([cond1, cond2], ['Eligible','High_Risk'], default='Not_Eligible')
        except Exception as e:
            self.logger.warning(f"Error sanitizing sample data: {str(e)}")

        # Store the data in session state
        st.session_state.current_data = df
        st.session_state.data_loaded = True
        st.session_state.data_path = found_path

        return df

    def calculate_financial_ratios(self, df):
        """Calculate key financial ratios"""
        df['debt_to_income'] = (df['current_emi_amount'] / df['monthly_salary']).fillna(0)
        df['expense_to_income'] = (
            (df['monthly_rent'] + df['school_fees'] + df['college_fees'] + 
             df['travel_expenses'] + df['groceries_utilities'] + 
             df['other_monthly_expenses'] + df['current_emi_amount']) / 
            df['monthly_salary']
        ).fillna(0)
        df['affordability_ratio'] = (
            (df['monthly_salary'] - 
             (df['monthly_rent'] + df['school_fees'] + df['college_fees'] + 
              df['travel_expenses'] + df['groceries_utilities'] + 
              df['other_monthly_expenses'])) / 
            df['monthly_salary']
        ).fillna(0)
        
        # Risk scoring
        df['employment_stability'] = df['years_of_employment'] / 30  # Normalized
        df['financial_stability'] = (df['bank_balance'] + df['emergency_fund']) / df['monthly_salary']
        df['risk_score'] = (
            0.3 * (1 - df['employment_stability']) +
            0.3 * df['debt_to_income'] +
            0.2 * df['expense_to_income'] +
            0.2 * (1 - (df['credit_score'] - 300) / 550)
        )
        
        return df
    
    def load_and_clean_data(self, file_path=None):
        """
        Load and clean the dataset with proper data cleaning and validation
        
        Args:
            file_path (str, optional): Path to the data file. If None, uses emi_prediction_dataset.csv.
            
        Returns:
            pd.DataFrame: Cleaned and validated dataset
        """
        try:
            from streamlit_config import config
            import os
            from pathlib import Path

            # Use default path if not provided
            if file_path is None:
                file_path = os.path.join('data', 'emi_prediction_dataset.csv')

            # Build a robust list of candidate paths
            filename = os.path.basename(file_path)
            base_dir = Path(__file__).resolve().parent
            candidates = [
                Path(file_path),
                base_dir / file_path,
                base_dir / 'data' / filename,
                Path.cwd() / 'data' / filename,
                Path.cwd() / filename,
            ]

            # Choose first existing path
            chosen = None
            for p in candidates:
                if p.exists():
                    chosen = p
                    break

            if chosen is None:
                # As a last resort, try a case-insensitive match within data directories that exist
                search_dirs = [
                    base_dir / 'data',
                    Path('data'),
                    base_dir,
                    Path.cwd(),
                ]
                for d in search_dirs:
                    if d.exists() and d.is_dir():
                        for f in d.iterdir():
                            if f.is_file() and f.name.lower() == filename.lower():
                                chosen = f
                                break
                    if chosen is not None:
                        break

            if chosen is None:
                raise FileNotFoundError(
                    f"Could not find the dataset file '{filename}'. "
                    "Ensure it is committed to the repository under the 'data' folder."
                )

            # Update session state
            st.session_state.data_path = str(chosen)
            self.logger.info(f"Loading dataset from {chosen}")

            # Read CSV with safe defaults
            df = pd.read_csv(chosen, low_memory=False)

            # Sanitize columns and dtypes
            try:
                # Ensure all column names are strings and non-empty
                new_cols = []
                for i, c in enumerate(df.columns):
                    name = str(c).strip() if c is not None and str(c).strip() != '' else f"col_{i}"
                    new_cols.append(name)
                df.columns = new_cols

                # Coerce expected numeric columns
                numeric_cols = [
                    'monthly_salary','years_of_employment','monthly_rent','family_size','dependents',
                    'school_fees','college_fees','travel_expenses','groceries_utilities','other_monthly_expenses',
                    'current_emi_amount','credit_score','bank_balance','emergency_fund',
                    'requested_amount','requested_tenure','max_monthly_emi'
                ]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Fill obviously numeric NaNs where appropriate to avoid downstream math errors
                fill_zeros = [c for c in numeric_cols if c in df.columns and c != 'credit_score']
                for c in fill_zeros:
                    df[c] = df[c].fillna(0)

                if 'credit_score' in df.columns:
                    df['credit_score'] = df['credit_score'].fillna(0)

            except Exception as e:
                self.logger.warning(f"Error during data sanitization: {str(e)}")

            # Compute ratios/targets if missing
            try:
                if 'max_monthly_emi' not in df.columns or df['max_monthly_emi'].isna().all():
                    if all(c in df.columns for c in [
                        'monthly_salary','monthly_rent','school_fees','college_fees',
                        'travel_expenses','groceries_utilities','other_monthly_expenses','current_emi_amount'
                    ]):
                        total_exp = (
                            df['monthly_rent'] + df['school_fees'] + df['college_fees'] +
                            df['travel_expenses'] + df['groceries_utilities'] + df['other_monthly_expenses'] +
                            df['current_emi_amount']
                        )
                        disposable = (df['monthly_salary'] - total_exp).clip(lower=0)
                        df['max_monthly_emi'] = (disposable * 0.4).clip(lower=500)

                if 'emi_eligibility' not in df.columns and all(c in df.columns for c in ['monthly_salary','credit_score']):
                    # Simple heuristic based on disposable income and credit score
                    total_exp = 0
                    for c in ['monthly_rent','school_fees','college_fees','travel_expenses','groceries_utilities','other_monthly_expenses','current_emi_amount']:
                        if c in df.columns:
                            total_exp = total_exp + df[c]
                    disposable = (df['monthly_salary'] - total_exp).fillna(0)
                    cond1 = (disposable > 0.4 * df['monthly_salary']) & (df['credit_score'] > 700)
                    cond2 = (disposable > 0.2 * df['monthly_salary']) & (df['credit_score'] > 600)
                    df['emi_eligibility'] = np.select([cond1, cond2], ['Eligible','High_Risk'], default='Not_Eligible')
            except Exception as e:
                self.logger.warning(f"Error computing derived targets: {str(e)}")

            self.logger.info(f"Successfully loaded {len(df):,} records from {chosen}")

            # Store the data in session state
            st.session_state.current_data = df
            st.session_state.data_loaded = True
            st.session_state.data_path = str(chosen)

            return df

        except Exception as e:
            self.logger.error(f"Unexpected error in load_and_clean_data: {str(e)}")
            raise
# Create required directories on startup
def initialize_application():
    """Initialize application directories and configuration"""
    import os
    from pathlib import Path
    
    # Create required directories if they don't exist
    required_dirs = ['data', 'models', 'mlruns', 'logs', 'tmp']
    for dir_name in required_dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
            # Verify directory is writable
            test_file = os.path.join(dir_name, '.test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            print(f"Warning: Could not create or write to directory '{dir_name}': {str(e)}")
    
    # Set MLflow tracking URI (robust to relative paths)
    try:
        import mlflow
        base_dir = Path(__file__).resolve().parent
        env_uri = os.getenv("MLFLOW_TRACKING_URI", config.MLFLOW_TRACKING_URI)

        def is_uri(s: str) -> bool:
            return "://" in s or s.startswith("sqlite://")

        tracking_uri = env_uri
        if not is_uri(tracking_uri):
            # Treat as filesystem path; make absolute inside app directory
            path = Path(tracking_uri)
            if not path.is_absolute():
                path = base_dir / path
            # Ensure directory exists
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create MLflow directory '{path}': {e}")
            tracking_uri = str(path)

        mlflow.set_tracking_uri(tracking_uri)
        print(f"Application initialized with MLflow tracking URI: {tracking_uri}")
    except Exception as e:
        print(f"Warning: Could not initialize MLflow: {str(e)}")
    
    # Verify data directory is accessible
    data_dir = 'data'
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        print(f"Warning: Data directory '{data_dir}' does not exist or is not a directory")
    
    # Check if dataset file exists
    dataset_path = os.path.join(data_dir, 'emi_prediction_dataset.csv')
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset file not found at {dataset_path}")
        # Try to find it in other locations
        possible_locations = [
            os.path.join(os.getcwd(), 'data', 'emi_prediction_dataset.csv'),
            'emi_prediction_dataset.csv',
            os.path.join(os.path.dirname(__file__), 'data', 'emi_prediction_dataset.csv')
        ]
        for loc in possible_locations:
            if os.path.exists(loc):
                print(f"Note: Dataset found at: {loc}")
                break
        else:
            print("Error: Could not find 'emi_prediction_dataset.csv' in any expected location")

# Call initialization at the start of main()
def main():
    initialize_application()
    st.markdown('<h1 class="main-header">💰 EMIPredict AI - Intelligent Financial Risk Assessment</h1>', 
                unsafe_allow_html=True)
    
    # Initialize the application
    app = EMIPredictAI()

    # Shared sidebar navigation (Dashboard as active for this page)
    render_sidebar_nav("🏠 Dashboard")
    page = "🏠 Dashboard"
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading and cleaning financial data..."):
            app = EMIPredictAI()
            try:
                app.data = app.load_and_clean_data("data/emi_prediction_dataset.csv")
                st.session_state.current_data = app.data
                st.session_state.data_loaded = True
                st.success(f"Successfully loaded {len(app.data)} records")
                
                # Show data quality information
                with st.expander("Data Quality Summary"):
                    st.write(f"**Dataset Shape:** {app.data.shape}")
                    st.write(f"**Missing Values:** {app.data.isnull().sum().sum()}")
                    st.write(f"**Data Types:**")
                    st.write(app.data.dtypes.value_counts())
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.info("Loading sample data instead...")
                app.data = app.load_sample_data()
                app.data = app.calculate_financial_ratios(app.data)
                st.session_state.current_data = app.data
                st.session_state.data_loaded = True
    else:
        # On reruns, hydrate the in-memory app instance from session state
        app.data = st.session_state.get('current_data', None)

    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard(app)
    elif page == "📊 Data Analysis":
        show_data_analysis(app)
    elif page == "🤖 Model Training":
        # Redirect to advanced Model Training page (with Data Preparation tab)
        st.switch_page("pages/2_🤖_Model_Training.py")
        return
    elif page == "📈 Predictions":
        show_predictions(app)
    elif page == "⚙️ Admin Panel":
        show_admin_panel(app)

def show_dashboard(app):
    st.header("📊 Financial Risk Assessment Dashboard")
    if app.data is None:
        st.error("No data loaded. Please load data from the dashboard or check for data loading errors.")
        return
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(app.data)
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        eligible_rate = (app.data['emi_eligibility'] == 'Eligible').mean() * 100
        st.metric("Eligibility Rate", f"{eligible_rate:.1f}%")
    
    with col3:
        avg_emi = app.data['max_monthly_emi'].mean()
        st.metric("Avg Max EMI", f"₹{avg_emi:,.0f}")
    
    with col4:
        avg_credit_score = app.data['credit_score'].mean()
        st.metric("Avg Credit Score", f"{avg_credit_score:.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Eligibility distribution
        fig = px.pie(
            app.data, 
            names='emi_eligibility',
            title='EMI Eligibility Distribution',
            color='emi_eligibility',
            color_discrete_map={
                'Eligible': '#28a745',
                'High_Risk': '#ffc107', 
                'Not_Eligible': '#dc3545'
            }
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # EMI scenarios
        fig = px.histogram(
            app.data,
            x='emi_scenario',
            color='emi_eligibility',
            title='EMI Scenarios by Eligibility',
            barmode='group'
        )
        st.plotly_chart(fig, width='stretch')
    
    # Financial metrics
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            app.data, 
            y='max_monthly_emi',
            title='Maximum Monthly EMI Distribution'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.scatter(
            app.data,
            x='monthly_salary',
            y='max_monthly_emi',
            color='emi_eligibility',
            title='Salary vs Maximum EMI'
        )
        st.plotly_chart(fig, width='stretch')

def show_data_analysis(app):
    st.header("📊 Data Analysis")
    if app.data is None:
        st.error("No data loaded. Please load data from the dashboard or check for data loading errors.")
        return
    st.dataframe(app.data.head(100), width='stretch')
    # Data overview
    st.subheader("Data Overview")
    #st.dataframe(app.data.head(100), width='stretch')
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(app.data.describe(), width='stretch')
    
    # Correlation analysis
    st.subheader("Correlation Heatmap")
    numeric_cols = app.data.select_dtypes(include=[np.number]).columns
    corr_matrix = app.data[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        aspect="auto"
    )
    st.plotly_chart(fig, width='stretch')
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_feature1 = st.selectbox(
            "Select Feature 1",
            options=numeric_cols,
            key="feature1"
        )
        fig = px.histogram(app.data, x=selected_feature1, title=f"Distribution of {selected_feature1}")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        selected_feature2 = st.selectbox(
            "Select Feature 2", 
            options=numeric_cols,
            key="feature2"
        )
        fig = px.box(app.data, y=selected_feature2, title=f"Box Plot of {selected_feature2}")
        st.plotly_chart(fig, width='stretch')

def show_model_training(app):
    st.header("🤖 Model Training")
    if app.data is None:
        st.error("No data loaded. Please load data from the dashboard or check for data loading errors.")
        return
    # Quick access to full-featured Model Training (includes Data Preparation tab)
    with st.expander("Need the Data Preparation workflow?"):
        st.write("Open the advanced Model Training page that includes the full Data Preparation tab.")
        if st.button("Open Advanced Model Training (with Data Preparation)"):
            st.switch_page("pages/2_🤖_Model_Training.py")
    st.metric("Records", f"{len(app.data):,}")
    tab1, tab2, tab3 = st.tabs(["Classification", "Regression", "MLflow Tracking"])
    
    with tab1:
        st.subheader("EMI Eligibility Classification")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("Train classification models to predict EMI eligibility")
            
            if st.button("Train Classification Models", type="primary"):
                with st.spinner("Training classification models..."):
                    train_classification_models(app)
        
        with col2:
            st.metric("Target", "emi_eligibility")
            st.metric("Classes", "3")
            #st.metric("Records", f"{len(app.data):,}")
    
    with tab2:
        st.subheader("Maximum EMI Amount Regression")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("Train regression models to predict maximum affordable EMI")
            
            if st.button("Train Regression Models", type="primary"):
                with st.spinner("Training regression models..."):
                    train_regression_models(app)
        
        with col2:
            st.metric("Target", "max_monthly_emi")
            st.metric("Type", "Continuous")
            st.metric("Range", "₹500 - ₹50,000")
    
    with tab3:
        st.subheader("MLflow Experiment Tracking")
        st.info("Model experiments are tracked using MLflow")
        
        # Display recent experiments
        if os.path.exists("mlruns"):
            st.success("MLflow tracking active")
            if st.button("View MLflow Dashboard"):
                st.code("mlflow ui --port 5000")
                st.info("Run the above command in terminal to open MLflow dashboard")
        else:
            st.warning("MLflow directory not found. Run model training first.")

def show_predictions(app):
    st.header("📈 Real-time Predictions")
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.subheader("Individual Financial Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 25, 60, 35)
            monthly_salary = st.number_input("Monthly Salary (₹)", 15000, 200000, 50000)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            years_of_employment = st.slider("Years of Employment", 0, 30, 5)
        
        with col2:
            existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
            current_emi_amount = st.number_input("Current EMI Amount (₹)", 0, 30000, 0)
            monthly_rent = st.number_input("Monthly Rent (₹)", 0, 50000, 15000)
            total_monthly_expenses = st.number_input("Total Monthly Expenses (₹)", 5000, 100000, 25000)
            requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 1500000, 300000)
            requested_tenure = st.slider("Requested Tenure (months)", 3, 84, 24)
        
        if st.button("Assess Eligibility", type="primary"):
            # Mock prediction (replace with actual model prediction)
            disposable_income = monthly_salary - total_monthly_expenses - current_emi_amount
            eligibility_prob = min(disposable_income / monthly_salary * 2, 1)
            
            if eligibility_prob > 0.6 and credit_score > 700:
                eligibility = "Eligible"
                color = "success"
            elif eligibility_prob > 0.3 and credit_score > 600:
                eligibility = "High Risk"
                color = "warning"
            else:
                eligibility = "Not Eligible"
                color = "danger"
            
            max_emi = disposable_income * 0.4
            
            # Display results
            st.subheader("Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("EMI Eligibility", eligibility)
            
            with col2:
                st.metric("Maximum Affordable EMI", f"₹{max_emi:,.0f}")
            
            with col3:
                st.metric("Risk Level", color.replace("success", "Low").replace("warning", "Medium").replace("danger", "High"))
            
            # Detailed analysis
            st.subheader("Financial Analysis")
            
            analysis_data = {
                "Metric": ["Monthly Salary", "Total Expenses", "Disposable Income", 
                          "Debt-to-Income Ratio", "Affordability Ratio", "Credit Score"],
                "Value": [f"₹{monthly_salary:,}", f"₹{total_monthly_expenses:,}", 
                         f"₹{disposable_income:,}", 
                         f"{(current_emi_amount/monthly_salary)*100:.1f}%",
                         f"{(disposable_income/monthly_salary)*100:.1f}%",
                         f"{credit_score}"]
            }
            
            st.table(pd.DataFrame(analysis_data))

def show_admin_panel(app):
    st.header("⚙️ Admin Panel")
    if app.data is None:
        st.error("No data loaded. Please load data from the dashboard or check for data loading errors.")
        return
    tab1, tab2, tab3 = st.tabs(["CRUD Operations", "Data Quality", "System Monitoring"])
    
    with tab1:
        st.subheader("Create, Read, Update, Delete Operations")
        
        operation = st.selectbox("Select Operation", 
                               ["View Data", "Add Record", "Update Record", "Delete Record"])
        
        if operation == "View Data":
            st.dataframe(app.data, width='stretch')
        
        elif operation == "Add Record":
            st.info("Add new financial record")
            # Implementation for adding records
        
        elif operation == "Update Record":
            st.info("Update existing records")
            # Implementation for updating records
        
        elif operation == "Delete Record":
            st.info("Delete records")
            # Implementation for deleting records
    
    with tab2:
        st.subheader("Data Quality Assessment")
        
        # Data quality metrics
        completeness = (1 - app.data.isnull().sum() / len(app.data)) * 100
        uniqueness = (app.data.nunique() / len(app.data)) * 100
        
        quality_df = pd.DataFrame({
            'Feature': app.data.columns,
            'Completeness (%)': completeness,
            'Uniqueness (%)': uniqueness
        })
        
        st.dataframe(quality_df, width='stretch')
        
        # Data quality visualization
        fig = px.bar(quality_df, x='Feature', y='Completeness (%)', 
                     title='Data Completeness by Feature')
        st.plotly_chart(fig, width='stretch')
    
    with tab3:
        st.subheader("System Monitoring")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Models", "6")
            st.metric("Data Records", f"{len(app.data):,}")
            st.metric("MLflow Experiments", "12")
        
        with col2:
            st.metric("Avg Response Time", "0.8s")
            st.metric("Success Rate", "99.2%")
            st.metric("System Uptime", "99.9%")
        
        with col3:
            st.metric("Memory Usage", "45%")
            st.metric("CPU Usage", "32%")
            st.metric("Storage", "1.2GB/5GB")

def train_classification_models(app):
    """Train classification models for EMI eligibility prediction"""
    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("EMI_Eligibility_Classification")
        
        # Prepare data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Feature selection and preprocessing
        features = ['age', 'monthly_salary', 'credit_score', 'years_of_employment', 
                   'debt_to_income', 'expense_to_income', 'affordability_ratio', 
                   'risk_score', 'existing_loans', 'current_emi_amount']
        
        X = app.data[features]
        y = app.data['emi_eligibility']
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(random_state=42)
        }
        
        results = []
        
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                # Train model
                if model_name == "Logistic Regression":
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                
                # Log parameters and metrics
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": auc
                })
                
                # Log model with explicit task suffix
                mlflow.sklearn.log_model(model, model_name.replace(" ", "_") + "_Classification")
                
                results.append({
                    "Model": model_name,
                    "Accuracy": f"{accuracy:.4f}",
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                    "F1-Score": f"{f1:.4f}",
                    "ROC-AUC": f"{auc:.4f}"
                })
        
        # Display results
        results_df = pd.DataFrame(results)
        st.subheader("Classification Model Results")
        st.dataframe(results_df, width='stretch')
        
        # Save best model
        best_model_idx = results_df['Accuracy'].astype(float).idxmax()
        best_model_name = results_df.iloc[best_model_idx]['Model']
        st.success(f"Best Model: {best_model_name}")
        
    except Exception as e:
        st.error(f"Error training classification models: {str(e)}")

def train_regression_models(app):
    """Train regression models for maximum EMI prediction"""
    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Max_EMI_Regression")
        
        # Prepare data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Feature selection
        features = ['age', 'monthly_salary', 'credit_score', 'years_of_employment',
                   'debt_to_income', 'expense_to_income', 'affordability_ratio',
                   'risk_score', 'existing_loans', 'current_emi_amount',
                   'monthly_rent', 'bank_balance', 'emergency_fund']
        
        X = app.data[features]
        y = app.data['max_monthly_emi']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(random_state=42)
        }
        
        results = []
        
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                # Train model
                if model_name == "Linear Regression":
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # Log parameters and metrics
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "mape": mape
                })
                
                # Log model with explicit task suffix
                mlflow.sklearn.log_model(model, model_name.replace(" ", "_") + "_Regression")
                
                results.append({
                    "Model": model_name,
                    "RMSE": f"₹{rmse:,.0f}",
                    "MAE": f"₹{mae:,.0f}",
                    "R² Score": f"{r2:.4f}",
                    "MAPE": f"{mape:.2f}%"
                })
        
        # Display results
        results_df = pd.DataFrame(results)
        st.subheader("Regression Model Results")
        st.dataframe(results_df, width='stretch')
        
        # Save best model
        best_model_idx = results_df['R² Score'].astype(float).idxmax()
        best_model_name = results_df.iloc[best_model_idx]['Model']
        st.success(f"Best Model: {best_model_name}")
        
    except Exception as e:
        st.error(f"Error training regression models: {str(e)}")

if __name__ == "__main__":
    main()
