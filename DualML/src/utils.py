# src/utils.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import pickle
import os
from typing import Dict, List, Any, Union
import warnings
warnings.filterwarnings('ignore')

class EMIPredictUtils:
    """Utility class for EMIPredict AI platform"""
    def __init__(self):
        self.logger = self.setup_logging()
    
    @staticmethod
    def setup_logging():
        """Setup logging configuration for the application"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/emipredict.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def clean_numeric_column(self, x):
        """Clean numeric columns by handling NaN and converting safely"""
        try:
            if pd.isna(x):  # Check for NaN
                return None
            return int(float(x))  # Convert safely through float
        except (ValueError, TypeError):
            return None  # Return None for anything invalid
    
    def validate_financial_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive validation of financial data
        
        Args:
            df: Input DataFrame with financial data
            
        Returns:
            Dictionary with validation results and errors
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Check for required columns
            required_columns = [
                'age', 'monthly_salary', 'credit_score', 'emi_eligibility', 
                'max_monthly_emi'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_results['errors'].append(f"Missing required columns: {missing_columns}")
                validation_results['is_valid'] = False
            
            # Check data types
            numeric_columns = ['age', 'monthly_salary', 'credit_score', 'max_monthly_emi']
            for col in numeric_columns:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        validation_results['errors'].append(f"Column {col} should be numeric")
                        validation_results['is_valid'] = False
            
            # Check for negative values in financial columns
            financial_cols = ['monthly_salary', 'bank_balance', 'emergency_fund', 'max_monthly_emi']
            for col in financial_cols:
                if col in df.columns:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        validation_results['warnings'].append(
                            f"Negative values found in {col}: {negative_count} records"
                        )
            
            # Validate credit score range
            if 'credit_score' in df.columns:
                invalid_credit = ((df['credit_score'] < 300) | (df['credit_score'] > 850)).sum()
                if invalid_credit > 0:
                    validation_results['warnings'].append(
                        f"Credit scores outside valid range (300-850): {invalid_credit} records"
                    )
            
            # Validate age range
            if 'age' in df.columns:
                invalid_age = ((df['age'] < 18) | (df['age'] > 100)).sum()
                if invalid_age > 0:
                    validation_results['warnings'].append(
                        f"Ages outside reasonable range (18-100): {invalid_age} records"
                    )
            
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                validation_results['warnings'].append(
                    f"Missing values found: {missing_values} total"
                )
            
            # Calculate basic statistics
            validation_results['stats'] = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'missing_values': missing_values,
                'duplicate_records': df.duplicated().sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
            
            self.logger.info(f"Data validation completed: {len(validation_results['errors'])} errors, "
                           f"{len(validation_results['warnings'])} warnings")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Data validation failed: {str(e)}")
        
        return validation_results
    
    
    def calculate_financial_health_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive financial health metrics with handling for zero-inflated columns
        
        Args:
            df: Input DataFrame with financial data
            
        Returns:
            DataFrame with additional financial health metrics
        """
        try:
            df = df.copy()
            
            # Handle zero-inflated columns by creating indicators
            zero_inflated_columns = ['monthly_rent', 'school_fees', 'college_fees', 'current_emi_amount']
            for col in zero_inflated_columns:
                if col in df.columns:
                    indicator_name = f'{col}_is_zero'
                    df[indicator_name] = (df[col] == 0).astype(int)
            
            # Calculate total monthly expenses (handle zeros appropriately)
            expense_columns = [
                'monthly_rent', 'school_fees', 'college_fees', 'travel_expenses',
                'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount'
            ]
            
            # Only use columns that exist in the dataframe
            available_expenses = [col for col in expense_columns if col in df.columns]
            df['total_monthly_expenses'] = df[available_expenses].sum(axis=1)
            
            # Financial ratios with zero handling
            df['debt_to_income_ratio'] = np.where(
                df['monthly_salary'] > 0,
                df['current_emi_amount'] / df['monthly_salary'],
                0
            )
            
            df['expense_to_income_ratio'] = np.where(
                df['monthly_salary'] > 0,
                df['total_monthly_expenses'] / df['monthly_salary'],
                1  # If no salary, expenses exceed income
            )
            
            df['disposable_income'] = df['monthly_salary'] - df['total_monthly_expenses']
            
            df['savings_ratio'] = np.where(
                df['monthly_salary'] > 0,
                df['disposable_income'] / df['monthly_salary'],
                0
            )
            
            # Affordability metrics
            df['emi_affordability'] = (df['disposable_income'] * 0.4).clip(0)  # 40% of disposable income, non-negative
            
            # Risk scores with robust calculations
            df['credit_risk_score'] = np.where(
                df['credit_score'].notna(),
                (850 - df['credit_score']) / 550,  # Normalized risk (0-1)
                0.5  # Default risk if credit score missing
            )
            
            df['debt_risk_score'] = df['debt_to_income_ratio'].clip(0, 1)
            df['expense_risk_score'] = df['expense_to_income_ratio'].clip(0, 1)
            
            # Employment stability
            df['employment_stability'] = (df['years_of_employment'] / 30).clip(0, 1)  # Normalized
            
            # Financial stability (handle missing bank_balance and emergency_fund)
            financial_cushion = np.where(
                (df['bank_balance'].notna()) & (df['emergency_fund'].notna()),
                (df['bank_balance'] + df['emergency_fund']) / df['monthly_salary'],
                3  # Default cushion of 3 months if data missing
            )
            df['financial_cushion'] = financial_cushion
            
            df['financial_stability_score'] = (
                df['employment_stability'] * 0.3 +
                (df['financial_cushion'] / 12).clip(0, 1) * 0.3 +  # 12 months cushion = 1
                (1 - df['credit_risk_score']) * 0.2 +
                (1 - df['debt_risk_score']) * 0.2
            ).clip(0, 1)
            
            # Overall risk score
            df['overall_risk_score'] = (
                df['credit_risk_score'] * 0.3 +
                df['debt_risk_score'] * 0.3 +
                df['expense_risk_score'] * 0.2 +
                (1 - df['employment_stability']) * 0.2
            ).clip(0, 1)
            
            # Risk categories
            conditions = [
                df['overall_risk_score'] <= 0.3,
                df['overall_risk_score'] <= 0.6,
                df['overall_risk_score'] > 0.6
            ]
            choices = ['Low Risk', 'Medium Risk', 'High Risk']
            df['risk_category'] = np.select(conditions, choices, default='Medium Risk')
            
            self.logger.info("Financial health metrics calculated successfully with zero-inflation handling")
            
        except Exception as e:
            self.logger.error(f"Error calculating financial health metrics: {str(e)}")
            raise
        
        return df

    def generate_financial_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive financial report
        
        Args:
            df: DataFrame with financial data and calculated metrics
            
        Returns:
            Dictionary with financial analysis report
        """
        report = {
            'summary': {},
            'risk_analysis': {},
            'eligibility_analysis': {},
            'recommendations': []
        }
        
        try:
            # Summary statistics
            report['summary'] = {
                'total_applicants': len(df),
                'avg_monthly_salary': df['monthly_salary'].mean(),
                'avg_credit_score': df['credit_score'].mean(),
                'avg_debt_to_income': df['debt_to_income_ratio'].mean(),
                'avg_financial_stability': df['financial_stability_score'].mean()
            }
            
            # Risk analysis
            risk_distribution = df['risk_category'].value_counts().to_dict()
            report['risk_analysis'] = {
                'risk_distribution': risk_distribution,
                'high_risk_applicants': risk_distribution.get('High Risk', 0),
                'medium_risk_applicants': risk_distribution.get('Medium Risk', 0),
                'low_risk_applicants': risk_distribution.get('Low Risk', 0)
            }
            
            # Eligibility analysis
            if 'emi_eligibility' in df.columns:
                eligibility_dist = df['emi_eligibility'].value_counts().to_dict()
                report['eligibility_analysis'] = {
                    'eligibility_distribution': eligibility_dist,
                    'approval_rate': (eligibility_dist.get('Eligible', 0) / len(df)) * 100
                }
            
            # Generate recommendations
            recommendations = self._generate_business_recommendations(df)
            report['recommendations'] = recommendations
            
            self.logger.info("Financial report generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating financial report: {str(e)}")
            report['error'] = str(e)
        
        return report
    
    def _generate_business_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate business recommendations based on data analysis"""
        recommendations = []
        
        try:
            # Analyze risk distribution
            risk_counts = df['risk_category'].value_counts()
            high_risk_pct = (risk_counts.get('High Risk', 0) / len(df)) * 100
            
            if high_risk_pct > 30:
                recommendations.append(
                    f"High proportion of high-risk applicants ({high_risk_pct:.1f}%). "
                    "Consider tightening credit criteria or increasing interest rates for high-risk categories."
                )
            
            # Analyze debt levels
            high_debt_pct = (df['debt_to_income_ratio'] > 0.4).mean() * 100
            if high_debt_pct > 25:
                recommendations.append(
                    f"Significant portion of applicants with high debt burden ({high_debt_pct:.1f}%). "
                    "Recommend implementing stricter debt-to-income limits."
                )
            
            # Analyze credit scores
            poor_credit_pct = (df['credit_score'] < 600).mean() * 100
            if poor_credit_pct > 20:
                recommendations.append(
                    f"Substantial number of applicants with poor credit ({poor_credit_pct:.1f}%). "
                    "Consider alternative credit assessment methods or requiring collateral."
                )
            
            # Analyze employment stability
            low_employment_stability = (df['employment_stability'] < 0.3).mean() * 100
            if low_employment_stability > 15:
                recommendations.append(
                    f"Many applicants show low employment stability ({low_employment_stability:.1f}%). "
                    "Recommend requiring longer employment history or additional income verification."
                )
            
            if not recommendations:
                recommendations.append(
                    "Portfolio appears healthy. Current risk assessment criteria are appropriate."
                )
                
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def save_model_artifacts(self, model, model_name: str, feature_names: List[str], 
                           metrics: Dict[str, Any], model_type: str = 'classification') -> str:
        """
        Save model artifacts to disk
        
        Args:
            model: Trained model object
            model_name: Name of the model
            feature_names: List of feature names
            metrics: Dictionary of model metrics
            model_type: Type of model ('classification' or 'regression')
            
        Returns:
            Path to saved model directory
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = f"models/{model_type}/{model_name}_{timestamp}"
            
            # Create directory
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            with open(f"{model_dir}/model.pkl", 'wb') as f:
                pickle.dump(model, f)
            
            # Save feature names
            with open(f"{model_dir}/features.json", 'w') as f:
                json.dump(feature_names, f, indent=2)
            
            # Save metrics
            with open(f"{model_dir}/metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_type': model_type,
                'timestamp': timestamp,
                'feature_count': len(feature_names),
                'training_date': datetime.now().isoformat()
            }
            with open(f"{model_dir}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model artifacts saved to {model_dir}")
            return model_dir
            
        except Exception as e:
            self.logger.error(f"Error saving model artifacts: {str(e)}")
            raise
    
    def load_model_artifacts(self, model_dir: str):
        """
        Load model artifacts from disk
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            Tuple of (model, feature_names, metrics, metadata)
        """
        try:
            # Load model
            with open(f"{model_dir}/model.pkl", 'rb') as f:
                model = pickle.load(f)
            
            # Load feature names
            with open(f"{model_dir}/features.json", 'r') as f:
                feature_names = json.load(f)
            
            # Load metrics
            with open(f"{model_dir}/metrics.json", 'r') as f:
                metrics = json.load(f)
            
            # Load metadata
            with open(f"{model_dir}/metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.logger.info(f"Model artifacts loaded from {model_dir}")
            return model, feature_names, metrics, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading model artifacts: {str(e)}")
            raise
    
    def create_sample_financial_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create realistic sample financial data for testing
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with sample financial data
        """
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(25, 60, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
            'marital_status': np.random.choice(['Single', 'Married'], n_samples, p=[0.4, 0.6]),
            'education': np.random.choice(['High School', 'Graduate', 'Post Graduate', 'Professional'], 
                                        n_samples, p=[0.2, 0.5, 0.2, 0.1]),
            'monthly_salary': np.random.normal(50000, 20000, n_samples).clip(15000, 200000),
            'employment_type': np.random.choice(['Private', 'Government', 'Self-employed'], 
                                              n_samples, p=[0.6, 0.3, 0.1]),
            'years_of_employment': np.random.exponential(5, n_samples).clip(0, 30),
            'house_type': np.random.choice(['Rented', 'Own', 'Family'], n_samples, p=[0.5, 0.4, 0.1]),
            'monthly_rent': np.random.normal(15000, 5000, n_samples).clip(0, 50000),
            'family_size': np.random.randint(1, 6, n_samples),
            'dependents': np.random.randint(0, 4, n_samples),
            'school_fees': np.random.normal(5000, 2000, n_samples).clip(0, 20000),
            'college_fees': np.random.normal(10000, 5000, n_samples).clip(0, 50000),
            'travel_expenses': np.random.normal(8000, 3000, n_samples).clip(1000, 30000),
            'groceries_utilities': np.random.normal(12000, 4000, n_samples).clip(3000, 30000),
            'other_monthly_expenses': np.random.normal(5000, 2000, n_samples).clip(1000, 20000),
            'existing_loans': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'current_emi_amount': np.random.normal(8000, 4000, n_samples).clip(0, 30000),
            'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
            'bank_balance': np.random.normal(100000, 50000, n_samples).clip(0, 500000),
            'emergency_fund': np.random.normal(50000, 20000, n_samples).clip(0, 200000),
            'emi_scenario': np.random.choice([
                'E-commerce Shopping', 'Home Appliances', 'Vehicle', 
                'Personal Loan', 'Education'
            ], n_samples),
            'requested_amount': np.random.normal(300000, 200000, n_samples).clip(10000, 1500000),
            'requested_tenure': np.random.randint(3, 84, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate financial metrics
        df = self.calculate_financial_health_metrics(df)
        
        # Generate realistic target variables based on financial health
        conditions = [
            (df['overall_risk_score'] <= 0.3) & (df['credit_score'] > 700),
            (df['overall_risk_score'] <= 0.6) & (df['credit_score'] > 600),
            (df['overall_risk_score'] > 0.6) | (df['credit_score'] <= 600)
        ]
        choices = ['Eligible', 'High_Risk', 'Not_Eligible']
        df['emi_eligibility'] = np.select(conditions, choices, default='High_Risk')
        
        # Generate max monthly EMI (regression target)
        df['max_monthly_emi'] = df['emi_affordability'].clip(500, 50000)
        
        self.logger.info(f"Generated sample data with {n_samples} records")
        return df
    
    def format_currency(self, amount: float) -> str:
        """Format amount as Indian currency"""
        if amount >= 10000000:  # 1 crore
            return f"₹{amount/10000000:.2f}Cr"
        elif amount >= 100000:  # 1 lakh
            return f"₹{amount/100000:.2f}L"
        else:
            return f"₹{amount:,.0f}"
    
    def format_percentage(self, value: float) -> str:
        """Format value as percentage"""
        return f"{value:.1%}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for monitoring"""
        import psutil
        import platform
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
# Create utility instance
utils = EMIPredictUtils()