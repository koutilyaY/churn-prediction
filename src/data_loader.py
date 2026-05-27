import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_raw_data(filepath):
    """Load raw CSV data"""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Preprocess the telco churn dataset:
    - Handle missing values
    - Encode categorical variables
    - Scale numeric features
    """
    df = df.copy()
    
    # Drop customerID (not useful)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Fix TotalCharges (sometimes has spaces)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing with 0 (new customers haven't paid yet)
        df = df.fillna({'TotalCharges': 0})
    
    # Fix Churn (Yes/No → 1/0)
    if 'Churn' in df.columns:
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Identify categorical and numeric columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target from numeric cols
    if 'Churn' in numeric_cols:
        numeric_cols.remove('Churn')
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    return df_encoded

def split_features_target(df, target_col='Churn'):
    """Separate features and target"""
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    else:
        X = df
        y = None
    return X, y

def create_processed_data(input_path, output_path):
    """Load raw data, preprocess, and save"""
    df = load_raw_data(input_path)
    df_processed = preprocess_data(df)
    df_processed.to_csv(output_path, index=False)
    print(f"✓ Processed data saved to {output_path}")
    print(f"  Shape: {df_processed.shape}")
    print(f"  Columns: {df_processed.columns.tolist()}")
    return df_processed
