#!/usr/bin/env python3
"""
Fake News Detection - Data Exploration
Save this as a .ipynb file or run as Python script
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_fraud_data():
    """Explore the fake news dataset"""
    print("🔍 FAKE NEWS DATA EXPLORATION")
    print("="*50)
    
    # Load data
    try:
        if pd.io.common.file_exists("../data/fake.csv"):
            df = pd.read_csv("../data/fake.csv")
        elif pd.io.common.file_exists("data/fake.csv"):
            df = pd.read_csv("data/fake.csv")
        else:
            print("❌ Dataset not found!")
            return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Basic info
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"📏 Total rows: {len(df):,}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"\n🔍 Missing values per column:")
    for col, miss in missing.items():
        if miss > 0:
            print(f"   {col}: {miss}")
    if missing.sum() == 0:
        print("   ✅ No missing values!")
    
    # Data types
    print(f"\n📋 Data types:")
    print(df.dtypes.value_counts())
    
    # First few rows
    print(f"\n👀 First 5 rows:")
    print(df.head())
    
    # Check columns for fake news
    if 'label' in df.columns:
        print(f"\n🎯 LABEL ANALYSIS:")
        class_counts = df['label'].value_counts()
        pos_rate = class_counts.get(1, 0) / len(df) * 100
        print(f"   Real (0): {class_counts.get(0, 0):,}")
        print(f"   Fake (1): {class_counts.get(1, 0):,}")
        print(f"   Fake rate: {pos_rate:.3f}%")
    else:
        print("❌ Missing 'label' column (expected 0/1)")
    
    # Show a couple of texts
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else None)
    if text_col:
        print(f"\n📝 Sample texts from '{text_col}':")
        print(df[text_col].head(3).tolist())
    else:
        print("❌ Missing text columns ('cleaned_text' or 'text')")
    
    return df

def create_simple_plots(df):
    """Create basic visualizations"""
    if 'label' not in df.columns:
        print("Cannot create plots - no 'label' column found")
        return
    
    print("📊 Creating basic visualizations...")
    
    # Class distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    df['label'].value_counts().plot(kind='bar')
    plt.title('Class Distribution\n(0=Real, 1=Fake)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    
    # Amount distribution
    if 'Amount' in df.columns:
        plt.subplot(1, 2, 2)
        
        # Plot both classes
        normal_amounts = df[df['Class'] == 0]['Amount']
        fraud_amounts = df[df['Class'] == 1]['Amount']
        
        plt.hist(normal_amounts, bins=50, alpha=0.7, label='Normal', density=True)
        plt.hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', density=True)
        plt.xlabel('Amount')
        plt.ylabel('Density')
        plt.title('Transaction Amount Distribution')
        plt.legend()
        plt.yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Basic plots created!")

# Run the exploration
if __name__ == "__main__":
    df = explore_fraud_data()
    
    if df is not None:
        try:
            create_simple_plots(df)
        except Exception as e:
            print(f"⚠️ Could not create plots: {e}")
            print("This is normal if running without display")
    
    print("\n🎉 Data exploration complete!")