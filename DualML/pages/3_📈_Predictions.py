# pages/3_📈_Predictions.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from src.ui.navigation import render_sidebar_nav

from datetime import datetime
import json

def main():
    st.set_page_config(
        page_title="Predictions - EMIPredict AI",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Real-time Financial Predictions")
    st.markdown("Get instant EMI eligibility assessment and maximum EMI amount predictions")
    # Consistent sidebar navigation
    render_sidebar_nav("📈 Predictions")
    
    # Check if models are trained
    models_available = any(key.endswith('_cls') or key.endswith('_reg') for key in st.session_state.keys())
    
    if not models_available:
        st.warning("⚠️ No trained models found. Please train models first in the Model Training section.")
        st.info("You can still use the prediction interface with default rules-based assessment.")
    
    # Create tabs for different prediction types
    tab1, tab2, tab3 = st.tabs([
        "👤 Individual Assessment", 
        "📊 Batch Prediction", 
        "📋 Prediction History"
    ])
    
    with tab1:
        show_individual_prediction()
    
    with tab2:
        show_batch_prediction()
    
    with tab3:
        show_prediction_history()

def show_individual_prediction():
    st.header("👤 Individual Financial Assessment")
    
    st.info("""
    Provide the applicant's financial details to get instant EMI eligibility assessment 
    and maximum affordable EMI amount prediction.
    """)
    
    # Create form for user input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal & Employment Details")
            
            age = st.slider("Age", 25, 60, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            years_of_employment = st.slider("Years of Employment", 0, 30, 5)
            company_type = st.selectbox("Company Type", ["Startup", "MNC", "MSME", "Large Enterprise"])
        
        with col2:
            st.subheader("Financial Details")
            
            monthly_salary = st.number_input("Monthly Salary (₹)", 15000, 200000, 50000, step=5000)
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
            monthly_rent = st.number_input("Monthly Rent (₹)", 0, 50000, 15000, step=1000)
            family_size = st.slider("Family Size", 1, 6, 3)
            dependents = st.slider("Number of Dependents", 0, 4, 1)
            
            st.subheader("Monthly Expenses")
            school_fees = st.number_input("School Fees (₹)", 0, 20000, 5000, step=1000)
            college_fees = st.number_input("College Fees (₹)", 0, 50000, 10000, step=1000)
            travel_expenses = st.number_input("Travel Expenses (₹)", 1000, 30000, 8000, step=1000)
            groceries_utilities = st.number_input("Groceries & Utilities (₹)", 3000, 30000, 12000, step=1000)
            other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 1000, 20000, 5000, step=1000)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Financial Status")
            
            existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
            current_emi_amount = st.number_input("Current EMI Amount (₹)", 0, 30000, 0, step=1000)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            bank_balance = st.number_input("Bank Balance (₹)", 0, 500000, 100000, step=10000)
            emergency_fund = st.number_input("Emergency Fund (₹)", 0, 200000, 50000, step=10000)
        
        with col4:
            st.subheader("Loan Application Details")
            
            emi_scenario = st.selectbox("EMI Scenario", [
                "E-commerce Shopping", "Home Appliances", "Vehicle", 
                "Personal Loan", "Education"
            ])
            requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 1500000, 300000, step=10000)
            requested_tenure = st.slider("Requested Tenure (months)", 3, 84, 24)
        
        # Submit button
        submitted = st.form_submit_button("Assess Eligibility", type="primary")
    
    if submitted:
        with st.spinner("Analyzing financial profile..."):
            # Prepare input data
            input_data = prepare_input_data(
                age, gender, marital_status, education, monthly_salary, employment_type,
                years_of_employment, company_type, house_type, monthly_rent, family_size,
                dependents, school_fees, college_fees, travel_expenses, groceries_utilities,
                other_monthly_expenses, existing_loans, current_emi_amount, credit_score,
                bank_balance, emergency_fund, emi_scenario, requested_amount, requested_tenure
            )
            
            # Get predictions
            eligibility, max_emi, risk_level, probability, details = get_predictions(input_data)
            
            # Display results
            display_prediction_results(
                eligibility, max_emi, risk_level, probability, details, 
                requested_amount, requested_tenure
            )
            
            # Store prediction in history
            store_prediction_history(input_data, eligibility, max_emi, risk_level, probability)

def show_batch_prediction():
    st.header("📊 Batch Prediction")
    
    st.info("""
    Upload a CSV file with multiple applicant records to get batch predictions. 
    The file should contain the same features as the individual assessment form.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    with col2:
        st.write("**Template**")
        st.download_button(
            "Download Template",
            data=create_template_csv(),
            file_name="batch_prediction_template.csv",
            mime="text/csv"
        )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_data)} records")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(batch_data.head(10), use_container_width=True)
            
            # Process batch predictions
            if st.button("Process Batch Predictions", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    results = process_batch_predictions(batch_data)
                    
                    # Display results
                    st.subheader("Batch Prediction Results")
                    st.dataframe(results, use_container_width=True)
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        "Download Results as CSV",
                        data=csv,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("Batch Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        eligible_count = (results['EMI_Eligibility'] == 'Eligible').sum()
                        st.metric("Eligible", eligible_count)
                    
                    with col2:
                        high_risk_count = (results['EMI_Eligibility'] == 'High_Risk').sum()
                        st.metric("High Risk", high_risk_count)
                    
                    with col3:
                        not_eligible_count = (results['EMI_Eligibility'] == 'Not_Eligible').sum()
                        st.metric("Not Eligible", not_eligible_count)
                    
                    with col4:
                        avg_max_emi = results['Max_Monthly_EMI'].mean()
                        st.metric("Avg Max EMI", f"₹{avg_max_emi:,.0f}")
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            results,
                            names='EMI_Eligibility',
                            title='Eligibility Distribution',
                            color='EMI_Eligibility',
                            color_discrete_map={
                                'Eligible': '#28a745',
                                'High_Risk': '#ffc107',
                                'Not_Eligible': '#dc3545'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.histogram(
                            results,
                            x='Max_Monthly_EMI',
                            nbins=20,
                            title='Maximum EMI Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_prediction_history():
    st.header("📋 Prediction History")
    
    # Initialize prediction history in session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if not st.session_state.prediction_history:
        st.info("No prediction history available. Make some predictions to see them here.")
        return
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    # Display history
    st.subheader("Recent Predictions")
    st.dataframe(history_df, use_container_width=True)
    
    # Statistics
    st.subheader("Prediction Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_predictions = len(history_df)
        st.metric("Total Predictions", total_predictions)
    
    with col2:
        approval_rate = (history_df['Eligibility'] == 'Eligible').mean() * 100
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    
    with col3:
        avg_max_emi = history_df['Max_EMI'].mean()
        st.metric("Average Max EMI", f"₹{avg_max_emi:,.0f}")
    
    # Clear history button
    if st.button("Clear History", type="secondary"):
        st.session_state.prediction_history = []
        st.rerun()

def prepare_input_data(age, gender, marital_status, education, monthly_salary, employment_type,
                      years_of_employment, company_type, house_type, monthly_rent, family_size,
                      dependents, school_fees, college_fees, travel_expenses, groceries_utilities,
                      other_monthly_expenses, existing_loans, current_emi_amount, credit_score,
                      bank_balance, emergency_fund, emi_scenario, requested_amount, requested_tenure):
    """Prepare input data for prediction with only the specified features"""
    
    # Convert all numeric inputs to float
    try:
        # Convert numeric fields
        age = float(age) if age is not None else 0
        monthly_salary = float(monthly_salary) if monthly_salary is not None else 0
        years_of_employment = float(years_of_employment) if years_of_employment is not None else 0
        monthly_rent = float(monthly_rent) if monthly_rent is not None else 0
        family_size = float(family_size) if family_size is not None else 0
        dependents = float(dependents) if dependents is not None else 0
        school_fees = float(school_fees) if school_fees is not None else 0
        college_fees = float(college_fees) if college_fees is not None else 0
        travel_expenses = float(travel_expenses) if travel_expenses is not None else 0
        groceries_utilities = float(groceries_utilities) if groceries_utilities is not None else 0
        other_monthly_expenses = float(other_monthly_expenses) if other_monthly_expenses is not None else 0
        current_emi_amount = float(current_emi_amount) if current_emi_amount is not None else 0
        credit_score = float(credit_score) if credit_score is not None else 0
        bank_balance = float(bank_balance) if bank_balance is not None else 0
        emergency_fund = float(emergency_fund) if emergency_fund is not None else 0
        requested_amount = float(requested_amount) if requested_amount is not None else 0
        requested_tenure = float(requested_tenure) if requested_tenure is not None else 0
    except (ValueError, TypeError) as e:
        st.error(f"Error converting input values: {str(e)}")
        # Set default values in case of conversion error
        age = 0
        monthly_salary = 0
        years_of_employment = 0
        monthly_rent = 0
        family_size = 0
        dependents = 0
        school_fees = 0
        college_fees = 0
        travel_expenses = 0
        groceries_utilities = 0
        other_monthly_expenses = 0
        current_emi_amount = 0
        credit_score = 0
        bank_balance = 0
        emergency_fund = 0
        requested_amount = 0
        requested_tenure = 0
    
    # Calculate financial ratios
    total_monthly_expenses = (
        monthly_rent + school_fees + college_fees + travel_expenses + 
        groceries_utilities + other_monthly_expenses + current_emi_amount
    )
    
    disposable_income = monthly_salary - total_monthly_expenses
    debt_to_income = current_emi_amount / monthly_salary if monthly_salary > 0 else 0
    
    # Calculate binary features
    school_fees_is_zero = 1 if school_fees == 0 else 0
    monthly_rent_is_zero = 1 if monthly_rent == 0 else 0
    current_emi_amount_is_zero = 1 if current_emi_amount == 0 else 0
    college_fees_is_zero = 1 if college_fees == 0 else 0
    
    # Include all specified features plus required binary features
    input_data = {
        'age': float(age),
        'gender': str(gender) if gender is not None else 'Unknown',
        'marital_status': str(marital_status) if marital_status is not None else 'Unknown',
        'education': str(education) if education is not None else 'Unknown',
        'monthly_salary': float(monthly_salary),
        'employment_type': str(employment_type) if employment_type is not None else 'Unknown',
        'years_of_employment': float(years_of_employment),
        'company_type': str(company_type) if company_type is not None else 'Unknown',
        'house_type': str(house_type) if house_type is not None else 'Unknown',
        'monthly_rent': float(monthly_rent),
        'family_size': float(family_size),
        'dependents': float(dependents),
        'school_fees': float(school_fees),
        'college_fees': float(college_fees),
        'travel_expenses': float(travel_expenses),
        'groceries_utilities': float(groceries_utilities),
        'other_monthly_expenses': float(other_monthly_expenses),
        'existing_loans': 1 if str(existing_loans).lower() in ['yes', 'true', '1'] else 0,
        'current_emi_amount': float(current_emi_amount),
        'credit_score': float(credit_score),
        'bank_balance': float(bank_balance),
        'emergency_fund': float(emergency_fund),
        'emi_scenario': str(emi_scenario) if emi_scenario is not None else 'Unknown',
        'requested_amount': float(requested_amount),
        'requested_tenure': float(requested_tenure),
        'debt_to_income': float(debt_to_income),
        # Add binary features
        'school_fees_is_zero': int(school_fees_is_zero),
        'monthly_rent_is_zero': int(monthly_rent_is_zero),
        'current_emi_amount_is_zero': int(current_emi_amount_is_zero),
        'college_fees_is_zero': int(college_fees_is_zero)
    }
    
    return input_data

def get_predictions(input_data):
    """Get predictions using trained models or fallback rules"""
    
    # Try to use trained models first
    try:
        # Prepare features in the exact order expected by the model
        model_features = [
            'age', 'gender', 'marital_status', 'education', 'monthly_salary',
            'employment_type', 'years_of_employment', 'company_type', 'house_type',
            'monthly_rent', 'family_size', 'dependents', 'school_fees', 'college_fees',
            'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
            'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance',
            'emergency_fund', 'emi_scenario', 'requested_amount', 'requested_tenure',
            'debt_to_income', 'school_fees_is_zero', 'monthly_rent_is_zero',
            'current_emi_amount_is_zero', 'college_fees_is_zero'
        ]
        
        # Create a clean copy of input data with proper type conversion
        clean_data = {}
        for k in model_features:
            val = input_data.get(k, 0)  # Default to 0 if key doesn't exist
            # Convert numeric fields to float, leave others as is
            if k in ['age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
                    'family_size', 'dependents', 'school_fees', 'college_fees',
                    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
                    'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
                    'requested_amount', 'requested_tenure', 'debt_to_income',
                    'school_fees_is_zero', 'monthly_rent_is_zero',
                    'current_emi_amount_is_zero', 'college_fees_is_zero']:
                try:
                    clean_data[k] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    clean_data[k] = 0.0
            # Handle existing_loans specifically
            elif k == 'existing_loans':
                if isinstance(val, str):
                    clean_data[k] = 1 if val.lower() in ['yes', 'true', '1'] else 0
                else:
                    clean_data[k] = 1 if val else 0
            # Handle other string fields
            else:
                clean_data[k] = str(val) if val is not None else 'Unknown'
        
        # Create DataFrame with consistent column order and proper dtypes
        input_df = pd.DataFrame([clean_data[col] for col in model_features], 
                              index=model_features).T
        
        # Ensure all numeric columns are float
        for col in input_df.select_dtypes(include=['int', 'float']).columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)
        
        # Use classification model if available
        classification_models = [key for key in st.session_state.keys() if key.endswith('_cls')]
        if classification_models:
            try:
                model_key = classification_models[0]
                classifier = st.session_state[model_key]
                
                # Get feature names the model was trained on
                try:
                    if hasattr(classifier, 'feature_names_in_'):
                        model_features = classifier.feature_names_in_
                        # Ensure all required features are present
                        missing_features = set(model_features) - set(input_df.columns)
                        if missing_features:
                            for f in missing_features:
                                input_df[f] = 0.0  # Add missing features with default value 0
                        input_df = input_df[model_features]  # Reorder columns to match training
                except Exception as e:
                    st.warning(f"Feature handling warning: {str(e)}")
                    # Continue with current features if there's an error
                
                # Predict
                eligibility_pred = classifier.predict(input_df)[0]
                eligibility_proba = classifier.predict_proba(input_df)[0]
                
                # Map prediction to label
                if hasattr(classifier, 'classes_'):
                    # If model has classes_ attribute, use it for mapping
                    eligibility_map = {i: cls for i, cls in enumerate(classifier.classes_)}
                    eligibility = str(eligibility_map.get(eligibility_pred, 'High_Risk'))
                else:
                    # Default mapping
                    eligibility_map = {0: 'Eligible', 1: 'High_Risk', 2: 'Not_Eligible'}
                    eligibility = str(eligibility_map.get(eligibility_pred, 'High_Risk'))
                
                probability = float(max(eligibility_proba))
            except Exception as e:
                st.warning(f"Classification model warning: {str(e)}. Using fallback rules.")
                eligibility, probability = rules_based_assessment(input_data)
        else:
            # Fallback to rules-based assessment
            eligibility, probability = rules_based_assessment(input_data)
        
        # Use regression model if available
        regression_models = [key for key in st.session_state.keys() if key.endswith('_reg')]
        if regression_models:
            try:
                model_key = regression_models[0]
                regressor = st.session_state[model_key]
                
                # Get feature names the model was trained on
                try:
                    if hasattr(regressor, 'feature_names_in_'):
                        model_features = regressor.feature_names_in_
                        # Ensure all required features are present
                        missing_features = set(model_features) - set(input_df.columns)
                        if missing_features:
                            for f in missing_features:
                                input_df[f] = 0.0  # Add missing features with default value 0
                        input_df = input_df[model_features]  # Reorder columns to match training
                except Exception as e:
                    st.warning(f"Feature handling warning: {str(e)}")
                    # Continue with current features if there's an error
                
                max_emi = float(regressor.predict(input_df)[0])
                # Ensure max_emi is not negative
                max_emi = max(0, max_emi)
            except Exception as e:
                st.warning(f"Regression model warning: {str(e)}. Using fallback rules.")
                max_emi = rules_based_max_emi(input_data)
        else:
            # Fallback to rules-based calculation
            max_emi = rules_based_max_emi(input_data)
        
        # Determine risk level
        risk_level = determine_risk_level(eligibility, probability, input_data)
        
        # Generate detailed analysis
        details = generate_detailed_analysis(input_data, eligibility, max_emi, risk_level)
        
        return str(eligibility), float(max_emi), str(risk_level), float(probability), details
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        
        # Fallback to rules-based assessment
        try:
            eligibility, probability = rules_based_assessment(input_data)
            max_emi = rules_based_max_emi(input_data)
            risk_level = determine_risk_level(eligibility, probability, input_data)
            details = generate_detailed_analysis(input_data, eligibility, max_emi, risk_level)
            
            return str(eligibility), float(max_emi), str(risk_level), float(probability), details
        except Exception as fallback_error:
            st.error(f"Fallback prediction failed: {str(fallback_error)}")
            # Return default values if everything fails
            return 'High_Risk', 0.0, 'High', 0.5, {}

def rules_based_assessment(input_data):
    """Rules-based fallback for eligibility assessment"""
    
    disposable_income = (
        input_data['monthly_salary'] - 
        input_data['monthly_rent'] - 
        input_data['school_fees'] - 
        input_data['college_fees'] - 
        input_data['travel_expenses'] - 
        input_data['groceries_utilities'] - 
        input_data['other_monthly_expenses'] - 
        input_data['current_emi_amount']
    )
    
    disposable_ratio = disposable_income / input_data['monthly_salary']
    
    if (disposable_ratio > 0.4 and input_data['credit_score'] > 700 and 
        input_data['debt_to_income'] < 0.3):
        return 'Eligible', 0.85
    elif (disposable_ratio > 0.2 and input_data['credit_score'] > 600 and 
          input_data['debt_to_income'] < 0.5):
        return 'High_Risk', 0.65
    else:
        return 'Not_Eligible', 0.25

def rules_based_max_emi(input_data):
    """Rules-based fallback for maximum EMI calculation"""
    
    disposable_income = (
        input_data['monthly_salary'] - 
        input_data['monthly_rent'] - 
        input_data['school_fees'] - 
        input_data['college_fees'] - 
        input_data['travel_expenses'] - 
        input_data['groceries_utilities'] - 
        input_data['other_monthly_expenses'] - 
        input_data['current_emi_amount']
    )
    
    # Conservative estimate: 40% of disposable income
    max_emi = disposable_income * 0.4
    
    # Apply constraints
    max_emi = max(max_emi, 500)  # Minimum
    max_emi = min(max_emi, 50000)  # Maximum
    
    return max_emi

def determine_risk_level(eligibility, probability, input_data):
    """Determine risk level based on multiple factors"""
    
    if eligibility == 'Eligible' and probability > 0.8:
        return 'Low'
    elif eligibility == 'Eligible':
        return 'Medium-Low'
    elif eligibility == 'High_Risk' and probability > 0.6:
        return 'Medium'
    elif eligibility == 'High_Risk':
        return 'Medium-High'
    else:
        return 'High'

def generate_detailed_analysis(input_data, eligibility, max_emi, risk_level):
    """Generate detailed financial analysis with available features"""
    
    # Calculate total monthly expenses
    total_expenses = (
        input_data['monthly_rent'] + 
        input_data['school_fees'] + 
        input_data['college_fees'] + 
        input_data['travel_expenses'] + 
        input_data['groceries_utilities'] + 
        input_data['other_monthly_expenses'] + 
        input_data['current_emi_amount']
    )
    
    # Calculate derived metrics
    monthly_salary = input_data['monthly_salary']
    disposable_income = monthly_salary - total_expenses
    debt_to_income = input_data.get('debt_to_income', 0)
    expense_to_income = total_expenses / monthly_salary if monthly_salary > 0 else 0
    
    # Calculate binary features
    school_fees_is_zero = 1 if input_data['school_fees'] == 0 else 0
    monthly_rent_is_zero = 1 if input_data['monthly_rent'] == 0 else 0
    current_emi_amount_is_zero = 1 if input_data['current_emi_amount'] == 0 else 0
    college_fees_is_zero = 1 if input_data['college_fees'] == 0 else 0
    
    # Add binary features to input_data
    input_data.update({
        'school_fees_is_zero': school_fees_is_zero,
        'monthly_rent_is_zero': monthly_rent_is_zero,
        'current_emi_amount_is_zero': current_emi_amount_is_zero,
        'college_fees_is_zero': college_fees_is_zero
    })
    
    # Prepare analysis dictionary
    analysis = {
        'monthly_salary': monthly_salary,
        'total_expenses': total_expenses,
        'disposable_income': disposable_income,
        'debt_to_income_ratio': debt_to_income,
        'expense_to_income_ratio': expense_to_income,
        'affordability_ratio': (monthly_salary - total_expenses) / monthly_salary if monthly_salary > 0 else 0,
        'credit_score': input_data['credit_score'],
        'financial_stability': (input_data['bank_balance'] + input_data['emergency_fund']) / monthly_salary if monthly_salary > 0 else 0,
        'employment_stability': input_data['years_of_employment'] / 30  # Normalized
    }
    
    return analysis

def display_prediction_results(eligibility, max_emi, risk_level, probability, details, requested_amount, requested_tenure):
    """Display prediction results in an organized manner"""
    
    st.success("## Assessment Complete!")
    
    # Results header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Eligibility with color coding
        if eligibility == 'Eligible':
            st.metric("EMI Eligibility", "✅ Eligible", delta="Recommended", delta_color="normal")
        elif eligibility == 'High_Risk':
            st.metric("EMI Eligibility", "⚠️ High Risk", delta="Conditional", delta_color="off")
        else:
            st.metric("EMI Eligibility", "❌ Not Eligible", delta="Not Recommended", delta_color="inverse")
    
    with col2:
        st.metric("Maximum Affordable EMI", f"₹{max_emi:,.0f}")
    
    with col3:
        st.metric("Risk Level", risk_level)
    
    with col4:
        st.metric("Confidence Score", f"{probability:.1%}")
    
    # Detailed analysis
    st.subheader("📋 Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Financial Metrics**")
        
        metrics_data = {
            "Metric": [
                "Monthly Salary", "Total Monthly Expenses", "Disposable Income",
                "Debt-to-Income Ratio", "Expense-to-Income Ratio", 
                "Affordability Ratio", "Credit Score"
            ],
            "Value": [
                f"₹{details['monthly_salary']:,}",
                f"₹{details['total_expenses']:,.0f}",
                f"₹{details['disposable_income']:,.0f}",
                f"{details['debt_to_income_ratio']:.1%}",
                f"{details['expense_to_income_ratio']:.1%}",
                f"{details['affordability_ratio']:.1%}",
                f"{details['credit_score']}"
            ],
            "Status": [
                "✅ Good" if details['monthly_salary'] > 50000 else "⚠️ Average",
                "✅ Good" if details['total_expenses'] / details['monthly_salary'] < 0.6 else "⚠️ High",
                "✅ Good" if details['disposable_income'] > 0.3 * details['monthly_salary'] else "⚠️ Low",
                "✅ Good" if details['debt_to_income_ratio'] < 0.3 else "⚠️ High",
                "✅ Good" if details['expense_to_income_ratio'] < 0.6 else "⚠️ High",
                "✅ Good" if details['affordability_ratio'] > 0.4 else "⚠️ Low",
                "✅ Excellent" if details['credit_score'] > 750 else 
                "✅ Good" if details['credit_score'] > 650 else "⚠️ Average"
            ]
        }
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    with col2:
        st.write("**Loan Application Analysis**")
        
        # Calculate proposed EMI
        monthly_interest_rate = 0.01  # 1% monthly for demonstration
        proposed_emi = calculate_emi(requested_amount, monthly_interest_rate, requested_tenure)
        
        loan_analysis = {
            "Parameter": [
                "Requested Loan Amount",
                "Requested Tenure",
                "Proposed Monthly EMI",
                "Maximum Affordable EMI",
                "EMI Affordability Gap",
                "Recommended Action"
            ],
            "Value": [
                f"₹{requested_amount:,}",
                f"{requested_tenure} months",
                f"₹{proposed_emi:,.0f}",
                f"₹{max_emi:,.0f}",
                f"₹{proposed_emi - max_emi:,.0f}" if proposed_emi > max_emi else "₹0",
                "Reduce Amount" if proposed_emi > max_emi else "Proceed"
            ]
        }
        
        st.dataframe(pd.DataFrame(loan_analysis), use_container_width=True)
        
        # Recommendations
        st.write("**Recommendations**")
        
        if eligibility == 'Eligible':
            if proposed_emi <= max_emi:
                st.success("✅ Loan application meets all criteria. Recommended for approval.")
            else:
                st.warning(f"⚠️ Reduce loan amount to ₹{calculate_affordable_loan(max_emi, monthly_interest_rate, requested_tenure):,.0f} to stay within affordable EMI.")
        
        elif eligibility == 'High_Risk':
            st.warning("""
            ⚠️ Conditional approval recommended with following conditions:
            - Higher interest rate may apply
            - Additional collateral may be required
            - Consider co-applicant for better terms
            """)
        
        else:
            st.error("""
            ❌ Application does not meet eligibility criteria. Consider:
            - Improving credit score
            - Reducing existing debt
            - Increasing income stability
            - Reapplying after 6 months
            """)

def calculate_emi(principal, monthly_rate, tenure):
    """Calculate EMI amount"""
    if monthly_rate == 0:
        return principal / tenure
    return principal * monthly_rate * (1 + monthly_rate) ** tenure / ((1 + monthly_rate) ** tenure - 1)

def calculate_affordable_loan(affordable_emi, monthly_rate, tenure):
    """Calculate affordable loan amount based on EMI"""
    if monthly_rate == 0:
        return affordable_emi * tenure
    return affordable_emi * ((1 + monthly_rate) ** tenure - 1) / (monthly_rate * (1 + monthly_rate) ** tenure)

def process_batch_predictions(batch_data):
    """Process batch predictions for multiple records"""
    
    results = []
    
    for index, row in batch_data.iterrows():
        try:
            # Convert row to input data format
            input_data = row.to_dict()
            
            # Get predictions
            eligibility, max_emi, risk_level, probability, details = get_predictions(input_data)
            
            results.append({
                'ID': index + 1,
                'EMI_Eligibility': eligibility,
                'Max_Monthly_EMI': max_emi,
                'Risk_Level': risk_level,
                'Confidence_Score': probability,
                'Monthly_Salary': input_data.get('monthly_salary', 0),
                'Credit_Score': input_data.get('credit_score', 0),
                'Debt_to_Income_Ratio': input_data.get('debt_to_income', 0)
            })
        
        except Exception as e:
            st.error(f"Error processing record {index + 1}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def create_template_csv():
    """Create template CSV for batch predictions"""
    
    template_data = {
        'age': [35],
        'gender': ['Male'],
        'marital_status': ['Married'],
        'education': ['Graduate'],
        'monthly_salary': [50000],
        'employment_type': ['Private'],
        'years_of_employment': [5],
        'company_type': ['MNC'],
        'house_type': ['Rented'],
        'monthly_rent': [15000],
        'family_size': [3],
        'dependents': [1],
        'school_fees': [5000],
        'college_fees': [10000],
        'travel_expenses': [8000],
        'groceries_utilities': [12000],
        'other_monthly_expenses': [5000],
        'existing_loans': [1],
        'current_emi_amount': [0],
        'credit_score': [650],
        'bank_balance': [100000],
        'emergency_fund': [50000],
        'emi_scenario': ['Personal Loan'],
        'requested_amount': [300000],
        'requested_tenure': [24]
    }
    
    template_df = pd.DataFrame(template_data)
    return template_df.to_csv(index=False)

def store_prediction_history(input_data, eligibility, max_emi, risk_level, probability):
    """Store prediction in history"""
    
    prediction_record = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Age': input_data['age'],
        'Monthly_Salary': input_data['monthly_salary'],
        'Credit_Score': input_data['credit_score'],
        'Requested_Amount': input_data['requested_amount'],
        'Eligibility': eligibility,
        'Max_EMI': max_emi,
        'Risk_Level': risk_level,
        'Confidence': f"{probability:.1%}"
    }
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    st.session_state.prediction_history.append(prediction_record)
    
    # Keep only last 100 records
    if len(st.session_state.prediction_history) > 100:
        st.session_state.prediction_history = st.session_state.prediction_history[-100:]

if __name__ == "__main__":
    main()