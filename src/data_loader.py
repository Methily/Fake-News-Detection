import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_fraud_data(file_path: str = "data/fake.csv") -> pd.DataFrame:
    """Load the fraud detection dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data into training and testing sets."""
    # Assuming 'target' is your fraud label column
    X = df.drop('target', axis=1)  # Features
    y = df['target']  # Target variable
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def validate_data(df: pd.DataFrame) -> bool:
    """Basic data validation."""
    if df.empty:
        print("Error: Dataset is empty")
        return False
    
    if df.isnull().sum().sum() > 0:
        print("Warning: Dataset contains missing values")
    
    return True