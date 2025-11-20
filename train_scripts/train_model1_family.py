#!/usr/bin/env python3
"""
Train Model 1: OS Family Classifier

Dataset: Masaryk (TCP SYN flow-level features)
Task: Classify OS family (Windows, Linux, macOS, Android, iOS, BSD)
Algorithm: XGBoost (best for medium-large datasets with categorical targets)

Input:  data/processed/masaryk_processed.csv
Output: models/model1_os_family.pkl
        models/model1_feature_names.pkl
        models/model1_label_encoder.pkl
        results/model1_evaluation.json
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import argparse
from datetime import datetime

# ML libraries
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"ERROR: Missing required library: {e}")
    print("\nInstall with:")
    print("  pip install xgboost scikit-learn matplotlib seaborn")
    sys.exit(1)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def select_features(df, verbose=True):
    """
    Select relevant features for OS family classification

    Flow-level features from Masaryk dataset that are most discriminative
    for OS family fingerprinting.
    """

    # Core TCP fingerprinting features (CRITICAL for OS identification!)
    tcp_features = [
        # Basic TCP features
        'tcp_syn_size',
        'tcp_win_size',  # Added alias for tcp_window_size

        # TCP Options (CRITICAL - Different OSes have unique patterns)
        'tcp_win_scale_forward',  # NEW: HIGH importance
        'tcp_win_scale_backward',  # NEW: HIGH importance
        'tcp_mss_forward',  # NEW: HIGH importance
        'tcp_mss_backward',  # NEW: HIGH importance
        'tcp_sack_permitted_forward',  # NEW: MEDIUM importance
        'tcp_sack_permitted_backward',  # NEW: MEDIUM importance
        # 'tcp_timestamp_forward',  # REMOVED - 0.00% availability in Masaryk
        # 'tcp_timestamp_backward',  # REMOVED - 0.00% availability in Masaryk
        'tcp_nop_forward',  # NEW: LOW importance
        'tcp_nop_backward',  # NEW: LOW importance
    ]

    # IP-level features (ENHANCED with critical discriminators)
    ip_features = [
        'ttl',  # Time to Live
        'initial_ttl',  # NEW: Estimated initial TTL (64=Linux/Mac, 128=Windows)
        'df_flag_forward',  # NEW: Don't Fragment flag forward
        'df_flag_backward',  # NEW: Don't Fragment flag backward
        'ip_tos',  # NEW: MEDIUM importance - Type of Service
        'max_ttl_forward',  # NEW: Maximum TTL observed forward
        'max_ttl_backward',  # NEW: Maximum TTL observed backward
    ]

    # Flow-level behavioral features
    flow_features = [
        'flow_duration',
        'pkt_count',
        'total_bytes',
        'pkt_rate',
        'byte_rate',
        'avg_pkt_size',
        'pkt_count_forward',
        'pkt_count_backward',
    ]

    # Port-based features (some services are OS-specific)
    port_features = [
        'src_port',
        'dst_port',
    ]

    # Combine all features
    all_features = tcp_features + ip_features + flow_features + port_features

    # Select only features that exist in the dataframe
    available_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]

    if verbose:
        print(f"\nFeature Selection:")
        print(f"  Available: {len(available_features)}")
        print(f"  Missing: {len(missing_features)}")
        if missing_features:
            print(f"\n  Missing features: {missing_features[:10]}")
            if len(missing_features) > 10:
                print(f"    ... and {len(missing_features) - 10} more")

    return available_features


def handle_missing_values(X, strategy='median', verbose=True):
    """
    Handle missing values in feature matrix

    Strategy options:
    - 'median': Replace with median (best for continuous features)
    - 'mean': Replace with mean
    - 'zero': Replace with 0
    """
    if verbose:
        missing_before = X.isnull().sum().sum()
        print(f"\nMissing Values:")
        print(f"  Total missing: {missing_before:,}")

    # Only apply median/mean to numeric columns
    if strategy == 'median':
        X_filled = X.fillna(X.select_dtypes(include=['number']).median())
    elif strategy == 'mean':
        X_filled = X.fillna(X.select_dtypes(include=['number']).mean())
    elif strategy == 'zero':
        X_filled = X.fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if verbose:
        missing_after = X_filled.isnull().sum().sum()
        print(f"  After imputation: {missing_after:,}")

    return X_filled


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_xgboost_model(X_train, y_train, X_val, y_val,
                        class_weights=None, verbose=True):
    """
    Train XGBoost classifier with optimized hyperparameters

    XGBoost is chosen for Model 1 because:
    - Handles large datasets well
    - Built-in handling of missing values
    - Robust to imbalanced classes with scale_pos_weight
    - Generally better accuracy than Random Forest
    """

    if verbose:
        print("\nTraining XGBoost Model...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Validation samples: {len(X_val):,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")

    # Calculate class weights if imbalanced
    if class_weights is None:
        unique, counts = np.unique(y_train, return_counts=True)
        max_count = counts.max()
        class_weights = {cls: max_count / count for cls, count in zip(unique, counts)}

        if verbose:
            print(f"\n  Class distribution in training:")
            for cls, count in zip(unique, counts):
                print(f"    Class {cls}: {count:,} samples (weight: {class_weights[cls]:.2f})")

    # Calculate sample weights
    sample_weights = np.array([class_weights[y] for y in y_train])

    # XGBoost parameters optimized for OS family classification
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_train)),
        'max_depth': 8,  # Prevent overfitting
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,  # Minimum loss reduction for split
        'min_child_weight': 3,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1,
    }

    if verbose:
        print(f"\n  XGBoost parameters:")
        for key, value in params.items():
            if key not in ['use_label_encoder', 'eval_metric', 'objective']:
                print(f"    {key}: {value}")

    # Create and train model
    model = xgb.XGBClassifier(**params)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )

    if verbose:
        print(f"\n  Training complete!")
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Best score: {model.best_score:.4f}")

    return model, class_weights


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, label_encoder, output_dir, verbose=True):
    """
    Comprehensive model evaluation with metrics and visualizations
    """

    if verbose:
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    if verbose:
        print(f"\nOverall Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

    # Per-class metrics
    class_names = label_encoder.classes_
    class_report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True
    )

    if verbose:
        print(f"\nPer-Class Performance:")
        print(f"  {'OS Family':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print(f"  {'-'*60}")
        for cls in class_names:
            metrics = class_report[cls]
            print(f"  {cls:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1-score']:<10.3f} {int(metrics['support']):<10}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Model 1: OS Family Classification - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'model1_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"\n  Confusion matrix saved: {cm_path}")
    plt.close()

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': model.get_booster().feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    if verbose:
        print(f"\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Model 1: Top 20 Feature Importances')
    plt.tight_layout()

    fi_path = os.path.join(output_dir, 'model1_feature_importance.png')
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"  Feature importance plot saved: {fi_path}")
    plt.close()

    # Save evaluation results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'XGBoost',
        'task': 'OS Family Classification',
        'dataset': 'Masaryk',
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        },
        'per_class_metrics': {
            cls: {
                'precision': float(class_report[cls]['precision']),
                'recall': float(class_report[cls]['recall']),
                'f1_score': float(class_report[cls]['f1-score']),
                'support': int(class_report[cls]['support'])
            }
            for cls in class_names
        },
        'confusion_matrix': cm.tolist(),
        'class_names': class_names.tolist(),
        'feature_importance': feature_importance.to_dict('records')[:20]
    }

    results_path = os.path.join(output_dir, 'model1_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"  Evaluation results saved: {results_path}")

    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Model 1: OS Family Classifier (Masaryk dataset)'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/masaryk_processed.csv',
        help='Input CSV file with preprocessed Masaryk data'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for trained model'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Output directory for evaluation results'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )

    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Fraction of training data for validation (default: 0.1)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Print header
    if verbose:
        print("="*70)
        print("MODEL 1: OS FAMILY CLASSIFIER (MASARYK DATASET)")
        print("="*70)
        print(f"\nInput:  {args.input}")
        print(f"Output: {args.output_dir}/model1_os_family.pkl")
        print(f"Algorithm: XGBoost")

    # Load data
    if verbose:
        print(f"\n[1/6] Loading data...")

    if not os.path.exists(args.input):
        print(f"\nERROR: Input file not found: {args.input}")
        print(f"\nRun preprocessing first:")
        print(f"  python scripts/preprocess_masaryk.py")
        sys.exit(1)

    df = pd.read_csv(args.input)
    if verbose:
        print(f"  Loaded {len(df):,} records")
        print(f"  Columns: {len(df.columns)}")

    # Check for required columns
    if 'os_family' not in df.columns:
        print(f"\nERROR: 'os_family' column not found in dataset!")
        print(f"Available columns: {list(df.columns)[:10]}")
        sys.exit(1)

    # Remove rows with missing OS family
    df = df[df['os_family'].notna()]
    if verbose:
        print(f"  Records with OS family: {len(df):,}")

    # Class distribution
    if verbose:
        print(f"\n  OS Family distribution:")
        for os_family, count in df['os_family'].value_counts().items():
            pct = (count / len(df)) * 100
            print(f"    {os_family:<15}: {count:>8,} ({pct:>5.2f}%)")

    # Feature selection
    if verbose:
        print(f"\n[2/6] Feature engineering...")

    feature_columns = select_features(df, verbose=verbose)

    if len(feature_columns) == 0:
        print(f"\nERROR: No usable features found!")
        sys.exit(1)

    # Prepare X and y
    X = df[feature_columns].copy()
    y = df['os_family'].values

    # Handle missing values
    X = handle_missing_values(X, strategy='median', verbose=verbose)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if verbose:
        print(f"\n  Encoded {len(label_encoder.classes_)} classes:")
        for idx, cls in enumerate(label_encoder.classes_):
            print(f"    {idx}: {cls}")

    # Train/test split (stratified to preserve class distribution)
    if verbose:
        print(f"\n[3/6] Splitting data...")
        print(f"  Test size: {args.test_size * 100:.0f}%")
        print(f"  Validation size: {args.val_size * 100:.0f}%")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_encoded,
        test_size=args.test_size,
        stratify=y_encoded,
        random_state=args.random_state
    )

    # Further split training into train and validation
    val_size_adjusted = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size_adjusted,
        stratify=y_train_full,
        random_state=args.random_state
    )

    if verbose:
        print(f"  Training: {len(X_train):,} samples")
        print(f"  Validation: {len(X_val):,} samples")
        print(f"  Test: {len(X_test):,} samples")

    # Train model
    if verbose:
        print(f"\n[4/6] Training model...")

    model, class_weights = train_xgboost_model(
        X_train, y_train, X_val, y_val,
        verbose=verbose
    )

    # Evaluate model
    if verbose:
        print(f"\n[5/6] Evaluating model...")

    os.makedirs(args.results_dir, exist_ok=True)
    results = evaluate_model(
        model, X_test, y_test, label_encoder,
        args.results_dir, verbose=verbose
    )

    # Save model and metadata
    if verbose:
        print(f"\n[6/6] Saving model...")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(args.output_dir, 'model1_os_family.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    if verbose:
        print(f"  Model saved: {model_path}")

    # Save feature names
    feature_names_path = os.path.join(args.output_dir, 'model1_feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_columns, f)
    if verbose:
        print(f"  Feature names saved: {feature_names_path}")

    # Save label encoder
    encoder_path = os.path.join(args.output_dir, 'model1_label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    if verbose:
        print(f"  Label encoder saved: {encoder_path}")

    # Save class weights
    weights_path = os.path.join(args.output_dir, 'model1_class_weights.pkl')
    with open(weights_path, 'wb') as f:
        pickle.dump(class_weights, f)
    if verbose:
        print(f"  Class weights saved: {weights_path}")

    # Final summary
    if verbose:
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"\nModel 1 Performance:")
        print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"  F1-Score: {results['metrics']['f1_score']:.4f}")
        print(f"\nFiles saved:")
        print(f"  {model_path}")
        print(f"  {feature_names_path}")
        print(f"  {encoder_path}")
        print(f"  {weights_path}")
        print(f"\nNext steps:")
        print(f"  1. Train Model 2a: python scripts/train_model2a_legacy.py")
        print(f"  2. Train Model 2b: python scripts/train_model2b_modern.py")
        print(f"  3. Test ensemble: python scripts/predict_ensemble.py")


if __name__ == '__main__':
    main()
