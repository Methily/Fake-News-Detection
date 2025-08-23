"""
Model evaluation utilities for fake news text classification
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def load_saved_model(model_path="models/best_model.pkl"):
    """Load a saved model"""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def calculate_fraud_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive metrics for binary text classification
    """
    # Basic confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics manually for clarity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    metrics = {
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'accuracy': accuracy
    }
    
    # ROC-AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            metrics['roc_auc'] = roc_auc
        except Exception as e:
            print(f"⚠️ Could not calculate ROC-AUC: {e}")
            metrics['roc_auc'] = None
    
    return metrics

def print_fraud_evaluation(metrics, model_name="Model"):
    """
    Print a comprehensive evaluation report for fake news detection
    """
    print(f"\n🎯 {model_name.upper()} EVALUATION REPORT")
    print("="*50)
    
    cm = metrics['confusion_matrix']
    print(f"📊 CONFUSION MATRIX:")
    print(f"                 Predicted")
    print(f"              Real    Fake")
    print(f"   Real       {cm['tn']:6d}  {cm['fp']:5d}")
    print(f"   Fake       {cm['fn']:6d}  {cm['tp']:5d}")
    
    print(f"\n🎯 KEY FRAUD DETECTION METRICS:")
    print(f"   Precision:    {metrics['precision']:.4f} (of predicted frauds, % actually fraud)")
    print(f"   Recall:       {metrics['recall']:.4f} (of actual frauds, % we caught) ⭐")
    print(f"   Specificity:  {metrics['specificity']:.4f} (of normal transactions, % correctly identified)")
    print(f"   F1-Score:     {metrics['f1_score']:.4f} (balance of precision & recall)")
    print(f"   Accuracy:     {metrics['accuracy']:.4f} (overall correctness)")
    
    if metrics.get('roc_auc'):
        print(f"   ROC-AUC:      {metrics['roc_auc']:.4f} (overall model performance)")
    
    # Fraud-specific insights
    fraud_caught = cm['tp']
    fraud_missed = cm['fn']
    false_alarms = cm['fp']
    
    print(f"\n🚨 DETECTION PERFORMANCE:")
    print(f"   Fakes caught:      {fraud_caught:,}")
    print(f"   Fakes missed:      {fraud_missed:,}")
    print(f"   False alarms:      {false_alarms:,}")
    
    if fraud_missed > 0:
        cost_ratio = false_alarms / fraud_missed if fraud_missed > 0 else float('inf')
        print(f"   False alarm ratio: {cost_ratio:.2f} false alarms per missed fraud")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if metrics['recall'] < 0.7:
        print("   ⚠️ Low recall - many fake articles are being missed!")
        print("   → Consider lowering classification threshold")
        print("   → Try ensemble methods or different algorithms")
    
    if metrics['precision'] < 0.5:
        print("   ⚠️ Low precision - too many false alarms!")
        print("   → Consider raising classification threshold")
        print("   → Review feature engineering")
    
    if metrics['f1_score'] > 0.8:
        print("   ✅ Excellent F1-score - good balance of precision and recall")
    elif metrics['f1_score'] > 0.6:
        print("   ✅ Good F1-score - reasonable performance")
    else:
        print("   ⚠️ Low F1-score - model needs improvement")

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix with proper labels for fraud detection
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    
    # Add text annotations for better understanding
    tn, fp, fn, tp = cm.ravel()
    plt.text(0.5, -0.1, f'TN={tn:,}', transform=plt.gca().transAxes, ha='center')
    plt.text(1.5, -0.1, f'FP={fp:,}', transform=plt.gca().transAxes, ha='center')
    plt.text(0.5, 1.1, f'FN={fn:,}', transform=plt.gca().transAxes, ha='center')
    plt.text(1.5, 1.1, f'TP={tp:,}', transform=plt.gca().transAxes, ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve", save_path=None):
    """
    Plot ROC curve for fraud detection
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 ROC curve saved to: {save_path}")
    
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve", save_path=None):
    """
    Plot Precision-Recall curve - especially important for imbalanced datasets
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Baseline (random classifier performance)
    baseline = sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='red', linestyle='--', alpha=0.7,
                label=f'Baseline (Random) = {baseline:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 PR curve saved to: {save_path}")
    
    plt.show()

def comprehensive_evaluation(model, X_test, y_test, model_name="Model", save_plots=False):
    """
    Perform comprehensive evaluation of a fake news model
    """
    print(f"🔍 Performing comprehensive evaluation of {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    # Probability of positive class if available
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_pred_proba = model.decision_function(X_test)
    else:
        y_pred_proba = None
    
    # Calculate metrics
    metrics = calculate_fraud_metrics(y_test, y_pred, y_pred_proba)
    
    # Print evaluation report
    print_fraud_evaluation(metrics, model_name)
    
    # Create plots
    plot_dir = "evaluation_plots" if save_plots else None
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Confusion Matrix
    cm_path = f"{plot_dir}/{model_name.lower().replace(' ', '_')}_confusion_matrix.png" if save_plots else None
    plot_confusion_matrix(y_test, y_pred, f"{model_name} - Confusion Matrix", cm_path)
    
    # ROC Curve
    roc_path = f"{plot_dir}/{model_name.lower().replace(' ', '_')}_roc_curve.png" if save_plots else None
    if y_pred_proba is not None:
        plot_roc_curve(y_test, y_pred_proba, f"{model_name} - ROC Curve", roc_path)
    
    # Precision-Recall Curve
    pr_path = f"{plot_dir}/{model_name.lower().replace(' ', '_')}_pr_curve.png" if save_plots else None
    if y_pred_proba is not None:
        plot_precision_recall_curve(y_test, y_pred_proba, f"{model_name} - Precision-Recall Curve", pr_path)
    
    return metrics, y_pred, y_pred_proba

def compare_models(models_dict, X_test, y_test, save_plots=False):
    """
    Compare multiple models side by side
    """
    print("🏆 COMPARING MULTIPLE MODELS")
    print("="*60)
    
    results = {}
    
    # Evaluate each model
    for name, model in models_dict.items():
        metrics, y_pred, y_pred_proba = comprehensive_evaluation(
            model, X_test, y_test, name, save_plots
        )
        results[name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    # Create comparison table
    print("\n📊 MODEL COMPARISON SUMMARY:")
    print("-" * 80)
    print(f"{'Model':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-" * 80)
    
    best_f1 = 0
    best_model = ""
    
    for name, result in results.items():
        m = result['metrics']
        roc_auc = m.get('roc_auc', 0) or 0
        
        print(f"{name:<20} {m['precision']:<10.4f} {m['recall']:<10.4f} "
              f"{m['f1_score']:<10.4f} {roc_auc:<10.4f}")
        
        if m['f1_score'] > best_f1:
            best_f1 = m['f1_score']
            best_model = name
    
    print("-" * 80)
    print(f"🏆 BEST MODEL: {best_model} (F1-Score: {best_f1:.4f})")
    
    return results, best_model

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing evaluation utilities...")
    
    # Check if we have saved models
    model_path = "models/best_model.pkl"
    if os.path.exists(model_path):
        print("✅ Found saved model, loading for demo...")
        model = load_saved_model(model_path)
        
        if model:
            print("📊 Model loaded successfully!")
            print(f"   Model type: {type(model).__name__}")
            
            # Check if we have test data
            try:
                from preprocessing import full_preprocessing_pipeline
                print("🔄 Running preprocessing to get test data...")
                result = full_preprocessing_pipeline()
                
                X_test = result['X_test']
                y_test = result['y_test']
                
                print("🎯 Performing evaluation...")
                metrics, _, _ = comprehensive_evaluation(model, X_test, y_test, "Demo Model")
                
                print("\n✅ Evaluation demo complete!")
                
            except Exception as e:
                print(f"⚠️ Could not run full demo: {e}")
                print("Run training first: python src/train.py")
    else:
        print("⚠️ No saved model found")
        print("Train a model first: python src/train.py")
    
    print("\n🎉 Evaluation utilities ready!")