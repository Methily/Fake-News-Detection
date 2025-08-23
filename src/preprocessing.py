#!/usr/bin/env python3
"""
Text preprocessing utilities for fake news classification
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def load_fraud_dataset(filepath=None):
    """
    Load the fake news dataset
    """
    if filepath is None:
        # Try different possible locations
        possible_paths = [
            
            "data/fake.csv",
            
            "../data/fake.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        
        if filepath is None:
            raise FileNotFoundError("Dataset not found! Please ensure fake.csv is in data/ folder")
    
    print(f"📂 Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape}")
    
    return df

def validate_dataset(df):
    """
    Validate that this is the correct fake news dataset
    """
    required_label = 'label'
    possible_text_columns = ['cleaned_text', 'text']
    
    # Check label
    if required_label not in df.columns:
        raise ValueError("Missing required 'label' column")
    
    # Check text column
    text_col = None
    for col in possible_text_columns:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        raise ValueError("Missing text column. Expected one of: 'cleaned_text' or 'text'")
    
    unique_labels = sorted(df[required_label].dropna().unique())
    print(f"🔎 Found text column: '{text_col}'")
    print(f"🔎 Unique labels: {unique_labels}")
    if not set(unique_labels).issubset({0, 1}):
        print(f"⚠️ Warning: Expected binary labels 0/1, found {unique_labels}")
    
    print("✅ Dataset validation passed")
    return True

def clean_data(df):
    """
    Clean the dataset
    """
    print("🧹 Cleaning data...")
    
    initial_rows = len(df)
    
    # Drop obvious duplicates
    df_clean = df.drop_duplicates()
    
    # Drop rows where label is missing
    if 'label' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['label'])
    
    # Ensure text column has no NaNs
    if 'cleaned_text' in df_clean.columns:
        df_clean['cleaned_text'] = df_clean['cleaned_text'].fillna('')
    if 'text' in df_clean.columns:
        df_clean['text'] = df_clean['text'].fillna('')
    
    final_rows = len(df_clean)
    removed_rows = initial_rows - final_rows
    
    if removed_rows > 0:
        print(f"   Removed {removed_rows:,} rows ({removed_rows/initial_rows*100:.2f}%)")
    else:
        print("   No rows removed")
    
    print(f"✅ Cleaned dataset: {df_clean.shape}")
    return df_clean

def analyze_class_balance(df):
    """
    Analyze class distribution
    """
    if 'label' not in df.columns:
        print("⚠️ No 'label' column found for balance analysis")
        return None
    
    print("📊 Analyzing class balance...")
    
    class_counts = df['label'].value_counts().sort_index()
    total = len(df)
    
    balance_info = {}
    for class_label, count in class_counts.items():
        percentage = count / total * 100
        balance_info[int(class_label)] = {
            'count': int(count),
            'percentage': float(percentage)
        }
        print(f"   Label {class_label}: {count:,} samples ({percentage:.3f}%)")
    
    # Determine if imbalanced
    if len(balance_info) >= 2:
        minority_percentage = min(info['percentage'] for info in balance_info.values())
        if minority_percentage < 20:
            print("⚠️ Imbalanced dataset detected")
    
    return balance_info

def prepare_features_target(df):
    """
    Select text feature and target variable for fake news
    """
    if 'label' not in df.columns:
        raise ValueError("No 'label' column found for target variable")
    
    print("🎯 Selecting text and target...")
    
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else None)
    if text_col is None:
        raise ValueError("No text column found. Expected 'cleaned_text' or 'text'")
    
    X_text = df[text_col].astype(str)
    y = df['label'].astype(int)
    
    print(f"   Using text column: {text_col}")
    print(f"   Samples: {len(X_text):,}")
    
    return X_text, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    """
    print(f"✂️ Splitting data (train: {int((1-test_size)*100)}%, test: {int(test_size*100)}%)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    
    # Check class distribution in splits
    if hasattr(y_train, 'value_counts'):
        train_pos_rate = y_train.mean() * 100
        test_pos_rate = y_test.mean() * 100
        print(f"   Train positive rate: {train_pos_rate:.3f}%")
        print(f"   Test positive rate: {test_pos_rate:.3f}%")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, save_scaler=True, scaler_path="models/scaler.pkl"):
    """
    Deprecated for text data. Kept for backward compatibility.
    """
    raise NotImplementedError("Scaling is not applicable for text data. Use TF-IDF vectorization instead.")

def vectorize_text(X_train_text, X_test_text, save_vectorizer=True, vectorizer_path="models/tfidf_vectorizer.pkl"):
    """
    Vectorize text using TF-IDF
    """
    print("🧮 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=50000,
        ngram_range=(1, 2),
        strip_accents='unicode',
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)
    
    if save_vectorizer:
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        import joblib
        joblib.dump(vectorizer, vectorizer_path)
        print(f"💾 Vectorizer saved to: {vectorizer_path}")
    
    print("✅ Text vectorization complete")
    return X_train_vec, X_test_vec, vectorizer

def full_preprocessing_pipeline(filepath=None):
    """
    Complete preprocessing pipeline for fake news text classification
    """
    print("🚀 Starting full preprocessing pipeline...\n")
    
    # Step 1: Load data
    df = load_fraud_dataset(filepath)
    
    # Step 2: Validate data
    validate_dataset(df)
    
    # Step 3: Clean data
    df_clean = clean_data(df)
    
    # Step 4: Analyze balance
    balance_info = analyze_class_balance(df_clean)
    
    # Step 5: Prepare text and target
    X_text, y = prepare_features_target(df_clean)
    
    # Step 6: Split data
    X_train_text, X_test_text, y_train, y_test = split_data(X_text, y)
    
    # Step 7: Vectorize text
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train_text, X_test_text)
    
    print("\n🎉 Preprocessing pipeline complete!")
    
    return {
        'X_train': X_train_vec,
        'X_test': X_test_vec,
        'y_train': y_train,
        'y_test': y_test,
        'vectorizer': vectorizer,
        'balance_info': balance_info,
        'original_df': df_clean
    }

# Example usage
if __name__ == "__main__":
    try:
        result = full_preprocessing_pipeline()
        print(f"\n📊 Final Results:")
        print(f"   Training samples: {result['X_train'].shape[0]:,}")
        print(f"   Test samples: {result['X_test'].shape[0]:,}")
        print(f"   Features: {result['X_train'].shape[1]}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        print("Make sure you have the fake.csv file in the data/ folder")