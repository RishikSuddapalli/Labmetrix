import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

class DataPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def clean_numeric_column(self, x):
        """Clean numeric columns by handling NaN and converting safely"""
        try:
            if pd.isna(x):  # Check for NaN
                return None
            return int(float(x))  # Convert safely through float
        except (ValueError, TypeError):
            return None  # Return None for anything invalid
    
    def load_and_validate_data(self, file_path):
        """
        Load and validate the dataset with full data loading and duplicate detection
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Cleaned and validated DataFrame
        """
        try:
            self.logger.info(f"Starting data loading from {file_path}")
            
            # Define numeric columns that need special handling for malformed values
            self.numeric_columns = [
                'age', 'monthly_salary', 'bank_balance', 'current_emi_amount',
                'requested_tenure', 'credit_score', 'years_of_employment', 'family_size',
                'dependents', 'school_fees', 'college_fees', 'travel_expenses',
                'groceries_utilities', 'other_monthly_expenses', 'emergency_fund',
                'requested_amount', 'max_monthly_emi'
            ]
            
            # Load all columns as strings first to handle malformed values
            dtype_dict = {col: str for col in self.numeric_columns}
            
            # First pass to get the total number of rows and check for duplicate columns
            self.logger.info("Performing initial file scan...")
            total_rows = 0
            with open(file_path, 'r') as f:
                total_rows = sum(1 for _ in f) - 1  # Subtract 1 for header
            self.logger.info(f"Total rows in file: {total_rows:,}")
            
            # Read column names to check for duplicates
            df_sample = pd.read_csv(file_path, nrows=1)
            cols = pd.Series(df_sample.columns)
            duplicate_columns = df_sample.columns[df_sample.columns.duplicated()].tolist()
            
            if duplicate_columns:
                self.logger.warning(f"Found duplicate columns: {duplicate_columns}")
                # Keep only first occurrence of each column
                df_sample = df_sample.loc[:, ~df_sample.columns.duplicated()]
            
            self.logger.info(f"Loading {total_rows:,} rows with {len(df_sample.columns)} columns")
            
            # Read the full dataset in a single pass with optimized parameters
            self.logger.info("Loading full dataset...")
            df = pd.read_csv(
                file_path,
                dtype=dtype_dict,
                low_memory=False,
                engine='c',  # Use C engine for better performance
                float_precision='high',
                encoding='utf-8',
                on_bad_lines='warn'  # Warn about malformed lines but continue
            )
            
            # Log initial stats
            self.logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
            self.logger.info(f"Initial columns: {df.columns.tolist()}")
            
            # Check for and remove duplicate columns (case-insensitive)
            df.columns = [str(col).strip() for col in df.columns]  # Clean column names
            df = df.loc[:, ~df.columns.duplicated()]  # Remove exact duplicates
            
            # Check for case-insensitive duplicates
            lower_cols = [str(col).lower() for col in df.columns]
            if len(lower_cols) != len(set(lower_cols)):
                self.logger.warning("Found case-insensitive duplicate columns")
                # Keep first occurrence of each column (case-insensitive)
                seen = {}
                keep_cols = []
                for col in df.columns:
                    lower = str(col).lower()
                    if lower not in seen:
                        seen[lower] = True
                        keep_cols.append(col)
                    else:
                        self.logger.warning(f"Removing duplicate column: {col}")
                df = df[keep_cols]
            
            # Ensure no duplicate rows
            initial_count = len(df)
            df = df.drop_duplicates()
            if len(df) < initial_count:
                self.logger.warning(f"Removed {initial_count - len(df):,} duplicate rows")
            
            # Clean the dataset
            self.logger.info("Starting data cleaning...")
            df = self.clean_dataset(df)
            
            # Final type conversion to ensure Arrow compatibility
            for col in df.select_dtypes(include=['number']).columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            self.logger.info(f"Successfully loaded and validated {len(df)} records")
            self.logger.info(f"Columns in dataset: {df.columns.tolist()}")
            self.logger.info(f"Data types after loading:\n{df.dtypes}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_numeric_value(self, x):
        """
        Clean a single numeric value.
        Handles various input types, including strings with multiple decimal points,
        and ensures Arrow-compatible output.
        """
        # Handle None, NaN, or empty strings
        if pd.isna(x) or x is None or x == '' or x == 'nan':
            return 0.0
            
        # Convert to string if not already
        if not isinstance(x, str):
            x = str(x)
            
        try:
            # Remove any non-numeric characters except decimal point and negative sign
            clean_str = ''.join(c for c in x if c.isdigit() or c in '.-')
            if not clean_str:  # If nothing left after cleaning
                return 0.0
                
            # Handle multiple decimal points by keeping only the first one
            if clean_str.count('.') > 1:
                parts = clean_str.split('.')
                clean_str = f"{parts[0]}.{''.join(parts[1:])}"
                
            # Remove any remaining non-numeric characters except first decimal point and minus sign
            has_decimal = False
            result = []
            for c in clean_str:
                if c == '-':
                    if not result:  # Only allow minus at the start
                        result.append(c)
                elif c == '.':
                    if not has_decimal:  # Only allow one decimal point
                        result.append(c)
                        has_decimal = True
                elif c.isdigit():
                    result.append(c)
            
            clean_str = ''.join(result)
            if not clean_str or clean_str == '-':  # Handle cases where only a minus sign remains
                return 0.0
                
            float_val = float(clean_str)
            
            # Convert to int if it's a whole number, otherwise keep as float
            if float_val.is_integer():
                return int(float_val)
            return float_val
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error converting value '{x}': {str(e)}")
            return 0.0
    
    def clean_dataset(self, df):
        """
        Apply comprehensive data cleaning to the dataset.
        Handles duplicate columns, missing values, and ensures proper data types.
        """
        if df is None or df.empty:
            self.logger.warning("Empty or None DataFrame received for cleaning")
            return pd.DataFrame()
            
        df_clean = df.copy()
        
        # Log initial state
        self.logger.info(f"Starting dataset cleaning. Initial shape: {df_clean.shape}")
        self.logger.info(f"Initial columns: {df_clean.columns.tolist()}")
        
        # Clean all numeric columns
        for col in self.numeric_columns:
            if col in df_clean.columns:
                try:
                    # Log initial non-null count for this column
                    initial_non_null = df_clean[col].count()
                    
                    # Apply cleaning function
                    df_clean[col] = df_clean[col].apply(self.clean_numeric_value)
                    
                    # Convert to numeric, coercing any remaining errors to NaN
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    
                    # Fill any remaining NaN values with 0.0
                    null_count = df_clean[col].isna().sum()
                    if null_count > 0:
                        self.logger.warning(f"Column {col} had {null_count} values that couldn't be converted to numbers")
                        df_clean[col] = df_clean[col].fillna(0.0)
                    
                    # Log cleaning results
                    final_non_null = df_clean[col].count()
                    self.logger.info(
                        f"Cleaned column: {col} | "
                        f"Type: {df_clean[col].dtype} | "
                        f"Non-null: {final_non_null}/{len(df_clean)} | "
                        f"Range: {df_clean[col].min():.2f} to {df_clean[col].max():.2f}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error cleaning column {col}: {str(e)}")
                    # If there's an error, try a basic conversion
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
        
        # Ensure all numeric columns are properly typed
        for col in df_clean.select_dtypes(include=['number']).columns:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                if df_clean[col].isna().any():
                    df_clean[col] = df_clean[col].fillna(0.0)
            except Exception as e:
                self.logger.error(f"Error processing numeric column {col}: {str(e)}")
                df_clean[col] = 0.0
        
        # Handle missing values
        df_clean = self.handle_missing_values(df_clean)
        
        # Handle zero-inflated columns
        df_clean = self.handle_zero_inflated_columns(df_clean)
        
        # Final type conversion to ensure Arrow compatibility
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
        
        # Ensure no duplicate columns (case-insensitive)
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        
        # Log final state
        self.logger.info(f"Dataset cleaning complete. Final shape: {df_clean.shape}")
        self.logger.info(f"Final columns: {df_clean.columns.tolist()}")
        self.logger.info(f"Final data types:\n{df_clean.dtypes}")
        
        return df_clean
    
    def handle_zero_inflated_columns(self, df):
        """Handle columns with high percentage of zeros"""
        df_clean = df.copy()
        
        # Columns with high zero percentage (based on your analysis)
        zero_inflated_columns = {
            'monthly_rent': 58.6,  # 58.6% zeros
            'school_fees': 45.7,   # 45.7% zeros
            'college_fees': 72.9,  # 72.9% zeros
            'current_emi_amount': 60.1  # 60.1% zeros
        }
        
        # Create indicator variables for zero values
        for col, zero_percentage in zero_inflated_columns.items():
            if col in df_clean.columns:
                indicator_name = f'{col}_is_zero'
                df_clean[indicator_name] = (df_clean[col] == 0).astype(int)
                self.logger.info(f"Created indicator for {col}: {zero_percentage}% zeros")
        
        return df_clean
    
    def _perform_data_quality_checks(self, df):
        """Perform comprehensive data quality assessment"""
        checks = {
            'Missing Values': df.isnull().sum().sum(),
            'Duplicate Records': df.duplicated().sum(),
            'Zero Salary Records': (df['monthly_salary'] == 0).sum() if 'monthly_salary' in df.columns else 0,
            'Invalid Credit Scores': ((df['credit_score'] < 300) | (df['credit_score'] > 850)).sum() if 'credit_score' in df.columns else 0,
            'Negative Expenses': (df.select_dtypes(include=[np.number]) < 0).sum().sum()
        }
        
        # Log specific column missing values
        high_missing_cols = ['bank_balance', 'credit_score', 'education', 'monthly_rent', 'emergency_fund']
        for col in high_missing_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                if missing_count > 0:
                    self.logger.warning(f"{col}: {missing_count} missing values ({missing_pct:.2f}%)")
        
        for check, count in checks.items():
            if count > 0:
                self.logger.warning(f"{check}: {count} issues found")
    
    def handle_missing_values(self, df):
        """Handle missing values with sophisticated imputation"""
        df_clean = df.copy()
        
        # Numerical columns - use median imputation with consideration of zeros
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                # For zero-inflated columns, consider using mode if more than 50% are zeros
                zero_percentage = (df_clean[col] == 0).sum() / len(df_clean) * 100
                if zero_percentage > 50:
                    # Use 0 for zero-inflated columns
                    df_clean[col].fillna(0, inplace=True)
                    self.logger.info(f"Filled missing {col} with 0 (zero-inflated: {zero_percentage:.1f}%)")
                else:
                    # Use median for normal numerical columns
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    self.logger.info(f"Filled missing {col} with median: {median_val}")
        
        # Categorical columns - mode imputation with 'Unknown' category
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                self.logger.info(f"Filled missing {col} with mode: {mode_val}")
        
        return df_clean
    
    def handle_high_correlations(self, df, threshold=0.8):
        """Handle highly correlated features based on your analysis"""
        df_reduced = df.copy()
        
        # Highly correlated pairs based on your analysis
        correlated_pairs = [
            ('bank_balance', 'emergency_fund'),
            ('current_emi_amount', 'existing_loans'),
            ('dependents', 'family_size'),
            ('monthly_rent', 'house_type'),
            ('groceries_utilities', 'other_monthly_expenses')
        ]
        
        # Features to consider keeping (usually the more fundamental ones)
        features_to_keep = [
            'bank_balance',  # Keep over emergency_fund
            'current_emi_amount',  # Keep over existing_loans (more quantitative)
            'family_size',  # Keep over dependents
            'house_type',  # Keep over monthly_rent (more fundamental)
            'monthly_salary'  # Keep as fundamental feature
        ]
        
        features_to_remove = []
        
        for feature1, feature2 in correlated_pairs:
            if feature1 in df_reduced.columns and feature2 in df_reduced.columns:
                # Calculate correlation
                correlation = df_reduced[feature1].corr(df_reduced[feature2])
                if abs(correlation) > threshold:
                    # Decide which one to remove
                    if feature1 in features_to_keep and feature2 not in features_to_keep:
                        features_to_remove.append(feature2)
                        self.logger.info(f"Removing {feature2} (correlated with {feature1}, r={correlation:.3f})")
                    elif feature2 in features_to_keep and feature1 not in features_to_keep:
                        features_to_remove.append(feature1)
                        self.logger.info(f"Removing {feature1} (correlated with {feature2}, r={correlation:.3f})")
                    else:
                        # If both or neither are in keep list, remove the second one
                        features_to_remove.append(feature2)
                        self.logger.info(f"Removing {feature2} (correlated with {feature1}, r={correlation:.3f})")
        
        # Remove the highly correlated features
        features_to_remove = list(set(features_to_remove))
        df_reduced = df_reduced.drop(columns=features_to_remove)
        
        self.logger.info(f"Removed {len(features_to_remove)} highly correlated features")
        
        return df_reduced
    
    def encode_categorical_variables(self, df, categorical_columns):
        """Encode categorical variables with handling for imbalanced classes"""
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            # Handle gender imbalance by creating binary encoding
            if col == 'gender':
                # Map to binary with handling for imbalance
                df_encoded[col] = df_encoded[col].map({'Male': 1, 'Female': 0})
                self.logger.info(f"Binary encoded {col} (Male:1, Female:0)")
            else:
                # Use label encoding for other categorical variables
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                self.logger.info(f"Label encoded {col}")
        
        return df_encoded
    
    def get_data_quality_report(self, df):
        """Generate comprehensive data quality report"""
        report = {
            'basic_info': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            },
            'missing_values': {},
            'data_types': {},
            'correlation_issues': [],
            'imbalance_issues': []
        }
        
        # Missing values analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            report['missing_values'][col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            }
        
        # Data types
        report['data_types'] = dict(df.dtypes.astype(str))
        
        # Correlation analysis (simplified)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = []
            for col in upper_tri.columns:
                for idx in upper_tri.index:
                    if not pd.isna(upper_tri[col][idx]) and upper_tri[col][idx] > 0.8:
                        high_corr_pairs.append((col, idx, upper_tri[col][idx]))
            report['correlation_issues'] = high_corr_pairs
        
        # Imbalance analysis for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True)
            if len(value_counts) > 0:
                max_pct = value_counts.iloc[0] * 100
                if max_pct > 70:  # Threshold for imbalance
                    report['imbalance_issues'].append({
                        'column': col,
                        'dominant_value': value_counts.index[0],
                        'percentage': max_pct
                    })
        
        return report