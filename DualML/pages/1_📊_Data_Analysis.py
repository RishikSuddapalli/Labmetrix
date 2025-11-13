# pages/1_📊_Data_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from src.ui.navigation import render_sidebar_nav
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

def validate_data(data):
    """Validate the loaded data and return (is_valid, message) tuple"""
    if data is None or not isinstance(data, pd.DataFrame):
        return False, "No valid data found. Please load data first."
    
    if data.empty:
        return False, "The dataset is empty. Please check your data source."
    
    # Check for required columns 
    required_columns = [
        'age', 'monthly_salary', 'bank_balance', 'requested_amount', 
        'credit_score', 'emi_eligibility', 'max_monthly_emi'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}. Please check your dataset."
    
    return True, "Data validation successful"

def main():
    st.set_page_config(
        page_title="Data Analysis - EMIPredict AI",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Comprehensive analysis of financial data and EMI patterns")
    # Consistent sidebar navigation
    render_sidebar_nav("📊 Data Analysis")
    
    # Check if data is loaded in session state
    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        st.warning("⚠️ No data loaded. Please load your data from the main dashboard first.")
        if st.button("Go to Dashboard"):
            st.switch_page("app.py")
        return
    
    # Get data from session state
    data = st.session_state.current_data
    
    # Validate the data
    is_valid, message = validate_data(data)
    if not is_valid:
        st.error(f"❌ {message}")
        if st.button("Reload Data"):
            st.rerun()
        return
    
    # Show data source info
    data_source = "Sample Data" if st.session_state.get('using_sample_data', False) \
                 else st.session_state.get('data_path', 'Unknown source')
    st.sidebar.info(f"**Data Source:** {data_source}")
    st.sidebar.info(f"**Records:** {len(data):,}")
    st.sidebar.info(f"**Features:** {len(data.columns)}")
    
    # Initialize classes
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()

    # --- Remove duplicate columns if any ---
    duplicated_cols = data.columns[data.columns.duplicated()].unique()
    if len(duplicated_cols) > 0:
        data = data.loc[:, ~data.columns.duplicated()]
        st.warning(f"Duplicate columns found and removed: {', '.join(duplicated_cols)}.")

    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Dataset Overview", 
    "💰 Financial Analysis", 
    "🎯 Target Analysis",
    "📊 Feature Analysis",
    "🔍 Correlation Analysis",
    "⚡ Data Quality Report"  # New tab
])
    
    with tab1:
        show_dataset_overview(data)
    
    with tab2:
        show_financial_analysis(data)
    
    with tab3:
        show_target_analysis(data)
    
    with tab4:
        show_feature_analysis(data)
    
    with tab5:
        show_correlation_analysis(data)
    
    with tab6:
        show_data_quality_report(data)

def show_dataset_overview(data):
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        st.metric("Number of Features", f"{len(data.columns)}")
    
    with col3:
        missing_values = data.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_values}")
    
    with col4:
        duplicate_rows = data.duplicated().sum()
        st.metric("Duplicate Records", f"{duplicate_rows}")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(data.head(100), use_container_width=True)
    
    # Data information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Types")
        dtype_info = pd.DataFrame(data.dtypes, columns=['Data Type'])
        st.dataframe(dtype_info, use_container_width=True)
    
    with col2:
        st.subheader("Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_df = pd.DataFrame({
        'Column': data.columns,
        'Missing_Count': data.isnull().sum(),
        'Missing_Percentage': (data.isnull().sum() / len(data)) * 100
    }).sort_values('Missing_Percentage', ascending=False)
    
    fig = px.bar(
        missing_df[missing_df['Missing_Count'] > 0],
        x='Column',
        y='Missing_Percentage',
        title='Missing Values by Column (%)',
        color='Missing_Percentage',
        color_continuous_scale='reds'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_financial_analysis(data):
    st.header("💰 Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Income distribution
        fig = px.histogram(
            data, 
            x='monthly_salary',
            nbins=50,
            title='Monthly Salary Distribution',
            labels={'monthly_salary': 'Monthly Salary (₹)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Credit score distribution
        fig = px.histogram(
            data,
            x='credit_score',
            nbins=50,
            title='Credit Score Distribution',
            color_discrete_sequence=['green']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Expense breakdown
        expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
                       'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
        expense_data = data[expense_cols].mean().reset_index()
        expense_data.columns = ['Expense Type', 'Average Amount']
        
        fig = px.pie(
            expense_data,
            values='Average Amount',
            names='Expense Type',
            title='Average Monthly Expenses Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Bank balance vs emergency fund
        # Remove duplicate columns before plotting
        duplicated_cols = data.columns[data.columns.duplicated()].unique()
        if len(duplicated_cols) > 0:
            data = data.loc[:, ~data.columns.duplicated()]
            st.warning(f"Duplicate columns found and removed before plotting: {', '.join(duplicated_cols)}.")
        fig = px.scatter(
            data,
            x='bank_balance',
            y='emergency_fund',
            color='emi_eligibility',
            title='Bank Balance vs Emergency Fund',
            labels={
                'bank_balance': 'Bank Balance (₹)',
                'emergency_fund': 'Emergency Fund (₹)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Financial ratios analysis
    st.subheader("Financial Ratios Analysis")
    
    ratio_cols = [col for col in data.columns if 'ratio' in col.lower() or 'score' in col.lower()]
    
    if ratio_cols:
        col1, col2, col3 = st.columns(3)
        
        for i, ratio_col in enumerate(ratio_cols[:6]):  # Show first 6 ratios
            col = [col1, col2, col3][i % 3]
            with col:
                fig = px.histogram(
                    data, 
                    x=ratio_col,
                    title=f'Distribution of {ratio_col}',
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)

def show_target_analysis(data):
    st.header("🎯 Target Variable Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # EMI Eligibility distribution
        eligibility_counts = data['emi_eligibility'].value_counts()
        fig = px.pie(
            values=eligibility_counts.values,
            names=eligibility_counts.index,
            title='EMI Eligibility Distribution',
            color=eligibility_counts.index,
            color_discrete_map={
                'Eligible': '#28a745',
                'High_Risk': '#ffc107',
                'Not_Eligible': '#dc3545'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Maximum EMI distribution
        fig = px.histogram(
            data,
            x='max_monthly_emi',
            nbins=50,
            title='Maximum Monthly EMI Distribution',
            labels={'max_monthly_emi': 'Maximum Monthly EMI (₹)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # EMI scenarios by eligibility
        fig = px.sunburst(
            data,
            path=['emi_scenario', 'emi_eligibility'],
            title='EMI Scenarios by Eligibility'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Maximum EMI by employment type
        fig = px.box(
            data,
            x='employment_type',
            y='max_monthly_emi',
            color='emi_eligibility',
            title='Maximum EMI by Employment Type'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Target analysis by demographics
    st.subheader("Target Analysis by Demographics")
    
    demo_cols = ['age', 'gender', 'marital_status', 'education', 'employment_type']
    
    for demo_col in demo_cols:
        if demo_col in data.columns:
            fig = px.histogram(
                data,
                x=demo_col,
                color='emi_eligibility',
                barmode='group',
                title=f'EMI Eligibility by {demo_col.replace("_", " ").title()}',
                color_discrete_map={
                    'Eligible': '#28a745',
                    'High_Risk': '#ffc107',
                    'Not_Eligible': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis(data):
    st.header("📊 Feature Analysis")
    
    # Select feature to analyze
    feature_options = [col for col in data.columns if col not in ['emi_eligibility', 'max_monthly_emi']]
    selected_feature = st.selectbox("Select Feature to Analyze", feature_options)
    
    if selected_feature:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            if data[selected_feature].dtype in ['int64', 'float64']:
                fig = px.histogram(
                    data,
                    x=selected_feature,
                    nbins=50,
                    title=f'Distribution of {selected_feature}',
                    marginal='box'
                )
            else:
                value_counts = data[selected_feature].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f'Distribution of {selected_feature}',
                    labels={'x': selected_feature, 'y': 'Count'}
                )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Relationship with target
            if data[selected_feature].dtype in ['int64', 'float64']:
                fig = px.box(
                    data,
                    x='emi_eligibility',
                    y=selected_feature,
                    title=f'{selected_feature} by EMI Eligibility',
                    color='emi_eligibility',
                    color_discrete_map={
                        'Eligible': '#28a745',
                        'High_Risk': '#ffc107',
                        'Not_Eligible': '#dc3545'
                    }
                )
            else:
                cross_tab = pd.crosstab(data[selected_feature], data['emi_eligibility'])
                fig = px.bar(
                    cross_tab,
                    barmode='group',
                    title=f'{selected_feature} vs EMI Eligibility'
                )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.subheader(f"Statistical Summary - {selected_feature}")
        
        if data[selected_feature].dtype in ['int64', 'float64']:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{data[selected_feature].mean():.2f}")
            with col2:
                st.metric("Median", f"{data[selected_feature].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{data[selected_feature].std():.2f}")
            with col4:
                st.metric("Missing Values", f"{data[selected_feature].isnull().sum()}")
        else:
            value_counts = data[selected_feature].value_counts()
            st.dataframe(value_counts, use_container_width=True)

def show_correlation_analysis(data):
    st.header("🔍 Correlation Analysis")

    # Remove duplicate columns if any
    duplicated_cols = data.columns[data.columns.duplicated()].unique()
    if len(duplicated_cols) > 0:
        data = data.loc[:, ~data.columns.duplicated()]
        st.warning(f"Duplicate columns found and removed in correlation analysis: {', '.join(duplicated_cols)}.")
    
    # Select numerical columns for correlation
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    
    corr_matrix = data[numerical_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations with target
    st.subheader("Top Correlations with Target Variables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'emi_eligibility' in data.columns:
            # For classification target, we need to encode it first
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            encoded_eligibility = le.fit_transform(data['emi_eligibility'])
            temp_data = data[numerical_cols].copy()
            temp_data['emi_eligibility_encoded'] = encoded_eligibility
            
            eligibility_corr = temp_data.corr()['emi_eligibility_encoded'].abs().sort_values(ascending=False)
            eligibility_corr = eligibility_corr[eligibility_corr.index != 'emi_eligibility_encoded']
            
            fig = px.bar(
                x=eligibility_corr.head(10).values,
                y=eligibility_corr.head(10).index,
                orientation='h',
                title='Top Features Correlated with EMI Eligibility',
                labels={'x': 'Absolute Correlation', 'y': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'max_monthly_emi' in data.columns:
            emi_corr = data[numerical_cols].corr()['max_monthly_emi'].abs().sort_values(ascending=False)
            emi_corr = emi_corr[emi_corr.index != 'max_monthly_emi']
            
            fig = px.bar(
                x=emi_corr.head(10).values,
                y=emi_corr.head(10).index,
                orientation='h',
                title='Top Features Correlated with Max Monthly EMI',
                labels={'x': 'Absolute Correlation', 'y': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature relationships scatter plots
    st.subheader("Feature Relationship Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("X-axis Feature", numerical_cols, key='x_feature')
    with col2:
        y_feature = st.selectbox("Y-axis Feature", numerical_cols, key='y_feature')
    
    if x_feature and y_feature:
        if x_feature == y_feature:
            st.warning("Please select different features for X and Y.")
            return
        # Remove duplicate columns again right before plotting
        duplicated_cols = data.columns[data.columns.duplicated()].unique()
        if len(duplicated_cols) > 0:
            data = data.loc[:, ~data.columns.duplicated()]
            st.warning(f"Duplicate columns found and removed before plotting: {', '.join(duplicated_cols)}.")
        fig = px.scatter(
            data,
            x=x_feature,
            y=y_feature,
            color='emi_eligibility',
            title=f'{x_feature} vs {y_feature}',
            color_discrete_map={
                'Eligible': '#28a745',
                'High_Risk': '#ffc107',
                'Not_Eligible': '#dc3545'
            },
            trendline='lowess'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        correlation = data[x_feature].corr(data[y_feature])
        st.metric(f"Correlation between {x_feature} and {y_feature}", f"{correlation:.3f}")

        
def show_data_quality_report(data):
    st.header("🔍 Data Quality Report")
    
    from src.data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    
    # Generate quality report
    quality_report = preprocessor.get_data_quality_report(data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", quality_report['basic_info']['total_records'])
    with col2:
        st.metric("Total Columns", quality_report['basic_info']['total_columns'])
    with col3:
        st.metric("Memory Usage", f"{quality_report['basic_info']['memory_usage_mb']:.1f} MB")
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_data = []
    for col, info in quality_report['missing_values'].items():
        if info['missing_count'] > 0:
            missing_data.append({
                'Column': col,
                'Missing Count': info['missing_count'],
                'Missing %': info['missing_percentage']
            })
    
    if missing_data:
        missing_df = pd.DataFrame(missing_data).sort_values('Missing %', ascending=False)
        st.dataframe(missing_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            missing_df.head(10),
            x='Column',
            y='Missing %',
            title='Top 10 Columns with Missing Values',
            color='Missing %'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found in the dataset!")
    
    # Correlation issues
    if quality_report['correlation_issues']:
        st.subheader("High Correlation Issues")
        corr_data = []
        for col1, col2, corr_value in quality_report['correlation_issues']:
            corr_data.append({
                'Feature 1': col1,
                'Feature 2': col2,
                'Correlation': corr_value
            })
        
        corr_df = pd.DataFrame(corr_data).sort_values('Correlation', ascending=False)
        st.dataframe(corr_df, use_container_width=True)
    
    # Imbalance issues
    if quality_report['imbalance_issues']:
        st.subheader("Class Imbalance Issues")
        for issue in quality_report['imbalance_issues']:
            st.warning(
                f"Column '{issue['column']}' is imbalanced: "
                f"'{issue['dominant_value']}' represents {issue['percentage']:.1f}% of values"
            )

if __name__ == "__main__":
    main()