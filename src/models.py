import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    precision_score, recall_score, f1_score, 
    classification_report, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Churn rate - Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """Train baseline logistic regression model"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train random forest model"""
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, X_test=None, y_test=None):
    """Train XGBoost model with early stopping"""
    eval_set = [(X_test, y_test)] if X_test is not None else None
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    
    if eval_set:
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    else:
        model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Performance")
    print(f"{'='*50}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"{'='*50}\n")
    
    return {
        'model': model_name,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

def compare_models(models_dict, X_test, y_test):
    """Compare multiple models and return results"""
    results = []
    for name, model in models_dict.items():
        result = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(result)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'Model': r['model'],
            'AUC': r['auc'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1-Score': r['f1']
        }
        for r in results
    ])
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70 + "\n")
    
    return results, comparison_df

def find_optimal_threshold(y_test, y_proba, fp_cost=500, fn_cost=5000):
    """
    Find optimal prediction threshold based on cost-benefit analysis
    fp_cost: Cost of false positive (retention spend)
    fn_cost: Cost of false negative (lost customer)
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Calculate costs for each threshold
    costs = []
    for threshold in thresholds:
        preds = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        cost = (fp * fp_cost) + (fn * fn_cost)
        costs.append(cost)
    
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    min_cost = costs[optimal_idx]
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL THRESHOLD ANALYSIS")
    print(f"{'='*60}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Minimum Total Cost: ${min_cost:,.0f}")
    print(f"  - False Positive Cost (wasted spend): ${thresholds[optimal_idx] * fp_cost:,.0f}")
    print(f"  - False Negative Cost (lost customers): ${(1-thresholds[optimal_idx]) * fn_cost:,.0f}")
    print(f"{'='*60}\n")
    
    return optimal_threshold

def save_model(model, filepath):
    """Save trained model to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {filepath}")

def load_model(filepath):
    """Load trained model from pickle file"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
