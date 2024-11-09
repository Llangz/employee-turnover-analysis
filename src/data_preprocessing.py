import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path='data/HR_comma_sep.csv'):
    """Load and perform initial data preprocessing"""
    df = pd.read_csv(file_path)
    
    # Display basic information
    print("Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:")
    print(df.duplicated().sum())
    
    return df

def prepare_data_for_analysis(df):
    """Prepare data for analysis by creating dummy variables"""
    return pd.get_dummies(df, columns=['Department', 'salary'])

def prepare_data_for_modeling(df):
    """Prepare data for modeling including scaling"""
    df_model = pd.get_dummies(df, columns=['Department', 'salary'])
    X = df_model.drop('left', axis=1)
    y = df_model['left']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns, scaler