#!/usr/bin/env python3
"""
Fake news classification training script
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def load_data():
    """Load the fake news dataset"""
    print("📂 Loading fake news dataset...")
    
    try:
        if os.path.exists("data/fake.csv"):
            df = pd.read_csv("data/fake.csv")
        elif os.path.exists("../data/fake.csv"):
            df = pd.read_csv("../data/fake.csv")
        else:
            print("❌ Dataset not found! Please place fake.csv in data/")
            return None
            
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Validate essential columns
        if 'label' not in df.columns:
            print("❌ Missing 'label' column for target")
            return None
        if 'cleaned_text' not in df.columns and 'text' not in df.columns:
            print("❌ Missing text column. Need 'cleaned_text' or 'text'")
            return None
            
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def prepare_data(df):
    """Prepare text data and target"""
    print("🔧 Preparing data...")
    
    df = df.dropna(subset=['label']).copy()
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
    df[text_col] = df[text_col].fillna('').astype(str)
    
    X_text = df[text_col]
    y = df['label'].astype(int)
    
    # Class distribution
    pos_count = int(y.sum())
    neg_count = int(len(y) - pos_count)
    pos_rate = pos_count / len(y) * 100
    print(f"📈 Class Distribution:")
    print(f"   Real (0): {neg_count:,} ({100-pos_rate:.2f}%)")
    print(f"   Fake (1): {pos_count:,} ({pos_rate:.2f}%)")
    
    return X_text, y

def train_models(X_train, y_train):
    """Train text classification models"""
    print("🤖 Training models...")
    
    models = {}
    
    # Logistic Regression (good baseline for TF-IDF)
    print("  📊 Training Logistic Regression...")
    lr_model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=2000,
        solver='liblinear'
    )
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    # Linear SVM via SGD (hinge loss)
    print("  ⚡ Training Linear SVM (SGDClassifier)...")
    svm_model = SGDClassifier(
        loss='hinge',
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        tol=1e-3
    )
    svm_model.fit(X_train, y_train)
    models['Linear SVM (SGD)'] = svm_model
    
    # Random Forest (more complex, usually better performance)
    print("  🌳 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    print("✅ Models trained successfully!")
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models"""
    print("📊 Evaluating models...")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔍 Evaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_scores = None
        if hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_test)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_scores) if y_scores is not None else float('nan')
        
        # Get precision, recall, f1 from classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1_score = report['1']['f1-score']
        
        results[name] = {
            'model': model,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        if y_scores is not None:
            print(f"   ROC-AUC: {roc_auc:.4f}")
        else:
            print("   ROC-AUC: N/A (no probability/scores)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1_score:.4f}")
    
    return results

def save_best_model(results, vectorizer):
    """Save the best performing model and the TF-IDF vectorizer"""
    print("💾 Saving models...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Find best model (highest F1 score)
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_model = results[best_model_name]['model']
    
    print(f"🏆 Best model: {best_model_name}")
    print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
    
    # Save all models
    for name, result in results.items():
        filename = name.lower().replace(' ', '_') + '_model.pkl'
        joblib.dump(result['model'], f'models/{filename}')
        print(f"   ✅ Saved: models/{filename}")
    
    # Save best model separately
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    print("✅ All models saved!")
    return best_model_name, results[best_model_name]

def main():
    """Main training pipeline"""
    print("🚀 Starting fake news training pipeline...\n")
    
    # Step 1: Load data
    df = load_data()
    if df is None:
        return
    
    # Step 2: Prepare data
    X_text, y = prepare_data(df)
    
    # Step 3: Split data
    print("✂️ Splitting data...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train_text):,}")
    print(f"   Testing samples: {len(X_test_text):,}")
    
    # Step 4: Vectorize text (TF-IDF)
    try:
        from src.preprocessing import vectorize_text
    except Exception:
        from preprocessing import vectorize_text
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train_text, X_test_text)
    
    # Step 5: Train models
    models = train_models(X_train_vec, y_train)
    
    # Step 6: Evaluate models
    results = evaluate_models(models, X_test_vec, y_test)
    
    # Step 7: Save best model
    best_name, best_result = save_best_model(results, vectorizer)
    
    print(f"\n🎉 Training complete!")
    print(f"🏆 Best model: {best_name}")
    print(f"📁 Models saved in: models/")
    print(f"🎯 Best F1-Score: {best_result['f1_score']:.4f}")
    print(f"🎯 Best Recall: {best_result['recall']:.4f}")

if __name__ == "__main__":
    main()