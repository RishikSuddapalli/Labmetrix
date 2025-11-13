# src/feature_engineering.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.financial_ratios = {}
    
    def create_financial_ratios(self, df):
        """Create comprehensive financial ratios"""
        # Debt-to-Income Ratio
        df['debt_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary']
        
        # Expense-to-Income Ratio
        total_expenses = df['monthly_rent'] + df['school_fees'] + df['college_fees'] + \
                        df['travel_expenses'] + df['groceries_utilities'] + \
                        df['other_monthly_expenses']
        df['expense_to_income_ratio'] = total_expenses / df['monthly_salary']
        
        # Savings Ratio
        df['savings_ratio'] = (df['monthly_salary'] - total_expenses - df['current_emi_amount']) / df['monthly_salary']
        
        # Loan-to-Income Ratio
        df['loan_to_income_ratio'] = df['requested_amount'] / (df['monthly_salary'] * 12)
        
        # Financial Stability Score
        df['financial_stability_score'] = (
            (df['bank_balance'] / df['monthly_salary']) * 0.3 +
            (df['emergency_fund'] / df['monthly_salary']) * 0.3 +
            (df['years_of_employment'] / 30) * 0.2 +
            ((df['credit_score'] - 300) / 550) * 0.2
        )
        
        # Risk Capacity Index
        df['risk_capacity_index'] = (
            (df['monthly_salary'] / df['monthly_salary'].max()) * 0.25 +
            (1 - df['debt_to_income_ratio']).clip(0, 1) * 0.25 +
            (df['financial_stability_score']) * 0.25 +
            ((df['age'] - 25) / 35) * 0.25  # Age between 25-60
        )
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between key variables"""
        # Salary-Credit interaction
        df['salary_credit_interaction'] = df['monthly_salary'] * (df['credit_score'] / 850)
        
        # Employment-Stability interaction
        df['employment_stability_interaction'] = df['years_of_employment'] * df['financial_stability_score']
        
        # Debt-Burden indicator
        df['high_debt_burden'] = ((df['debt_to_income_ratio'] > 0.4) & 
                                 (df['expense_to_income_ratio'] > 0.6)).astype(int)
        
        return df
    
    def create_emi_scenario_features(self, df):
        """Create features specific to EMI scenarios"""
        # Scenario-specific risk weights
        scenario_risk_weights = {
            'E-commerce Shopping': 1.0,
            'Home Appliances': 1.1,
            'Education': 1.2,
            'Personal Loan': 1.3,
            'Vehicle': 1.4
        }
        
        df['scenario_risk_weight'] = df['emi_scenario'].map(scenario_risk_weights)
        
        # Tenure adjustment factor
        df['tenure_adjustment_factor'] = np.where(
            df['requested_tenure'] <= 12, 1.0,
            np.where(df['requested_tenure'] <= 36, 1.1, 1.2)
        )
        
        return df