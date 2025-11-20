#!/usr/bin/env python3
"""
Train Model 2a: Legacy OS Version Classifier

Dataset: nPrint (TCP SYN packet-level features)
Task: Classify legacy OS versions (Windows 7/8, Ubuntu 14.04/16.04, CentOS, Debian 8/9)
Algorithm: XGBoost (best for medium-large datasets)

Input:  data/processed/nprint_packets.csv
Output: models/model2a_legacy_os.pkl
        models/model2a_feature_names.pkl
        models/model2a_label_encoder.pkl
        results/model2a_evaluation.json
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
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report
    )
    from imblearn.over_sampling import SMOTE
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"ERROR: Missing required library: {e}")
    print("\nInstall with:")
    print("  pip install xgboost scikit-learn imbalanced-learn matplotlib seaborn")
    sys.exit(1)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def select_features(df, verbose=True):
    """
    Select packet-level features for OS version classification

    nPrint dataset provides rich packet-level TCP/IP features
    """

    # TCP/IP fingerprinting features (ENHANCED - most discriminative for OS fingerprinting!)
    tcp_features = [
        'tcp_window_size',
        'tcp_window_scale',  # Window scale option
        'tcp_mss',  # Maximum Segment Size
        'tcp_options_order',  # CRITICAL: String feature - needs encoding
        'tcp_flags',
        'tcp_timestamp_val',  # NEW: CRITICAL! Timestamp value (granularity differs by OS)
        'tcp_timestamp_ecr',  # NEW: CRITICAL! Timestamp echo reply
        'tcp_sack_permitted',  # NEW: MEDIUM importance
        'tcp_urgent_ptr',  # NEW: LOW importance
    ]

    # IP-level features (ENHANCED with critical discriminators!)
    ip_features = [
        'ttl',
        'initial_ttl',
        'df_flag',  # Don't Fragment
        'ip_len',  # IP packet length
        'ip_id',  # NEW: CRITICAL! Windows=incremental, Linux=randomized
        'ip_tos',  # NEW: MEDIUM - Type of Service / DSCP
    ]

    # Network metadata (may help)
    network_features = [
        'src_port',
        'dst_port',
        'protocol',
    ]

    # Combine
    all_features = tcp_features + ip_features + network_features

    # Select only available features
    available_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]

    if verbose:
        print(f"\nFeature Selection:")
        print(f"  Available: {len(available_features)}")
        print(f"  Missing: {len(missing_features)}")
        if missing_features:
            print(f"  Missing: {missing_features}")

    return available_features


def encode_tcp_options(df, column='tcp_options_order', verbose=True):
    """
    Encode TCP options order as categorical feature

    TCP options order is HIGHLY discriminative for OS fingerprinting!
    Example: "MSS:NOP:WS:NOP:NOP:SACK" uniquely identifies certain OS versions
    """

    if column not in df.columns:
        if verbose:
            print(f"  Warning: {column} not in dataset, skipping encoding")
        return df, []

    # Get unique option patterns
    unique_options = df[column].fillna('NONE').unique()

    if verbose:
        print(f"\n  TCP Options encoding:")
        print(f"    Unique patterns: {len(unique_options)}")

    # Create one-hot encoding for most common patterns (top 20)
    top_patterns = df[column].fillna('NONE').value_counts().head(20).index.tolist()

    new_cols = []
    for pattern in top_patterns:
        col_name = f"tcp_opt_{pattern.replace(':', '_')[:30]}"  # Limit length
        df[col_name] = (df[column] == pattern).astype(int)
        new_cols.append(col_name)

    if verbose:
        print(f"    Created {len(new_cols)} one-hot encoded features")

    return df, new_cols


def handle_missing_values(X, strategy='median', verbose=True):
    """Handle missing values in feature matrix"""

    if verbose:
        missing_before = X.isnull().sum().sum()
        print(f"\nMissing Values:")
        print(f"  Total missing: {missing_before:,}")

    if strategy == 'median':
        X_filled = X.fillna(X.median())
    elif strategy == 'mean':
        X_filled = X.fillna(X.mean())
    elif strategy == 'zero':
        X_filled = X.fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # For any remaining NaN (e.g., all-NaN columns), fill with 0
    X_filled = X_filled.fillna(0)

    if verbose:
        missing_after = X_filled.isnull().sum().sum()
        print(f"  After imputation: {missing_after:,}")

    return X_filled


# ============================================================================
# DATA BALANCING
# ============================================================================

def apply_smote(X_train, y_train, min_samples_threshold=100, verbose=True):
    """
    Apply SMOTE to balance classes with < min_samples_threshold samples

    SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic
    samples for minority classes to prevent model bias.
    """

    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))

    minority_classes = [cls for cls, count in class_dist.items()
                       if count < min_samples_threshold]

    if len(minority_classes) == 0:
        if verbose:
            print(f"\n  No minority classes (all have ≥{min_samples_threshold} samples)")
            print(f"  Skipping SMOTE")
        return X_train, y_train

    if verbose:
        print(f"\n  Applying SMOTE for minority classes:")
        for cls in minority_classes:
            print(f"    Class {cls}: {class_dist[cls]} samples")

    # Determine sampling strategy
    # SMOTE to bring minority classes up to min_samples_threshold
    sampling_strategy = {
        cls: max(min_samples_threshold, count)
        for cls, count in class_dist.items()
    }

    try:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=min(5, min(counts) - 1) if min(counts) > 1 else 1
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        if verbose:
            print(f"\n  After SMOTE:")
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            for cls, count in zip(unique_new, counts_new):
                print(f"    Class {cls}: {count} samples")
            print(f"\n  Total samples: {len(X_train):,} → {len(X_resampled):,}")

        return X_resampled, y_resampled

    except Exception as e:
        if verbose:
            print(f"\n  WARNING: SMOTE failed: {e}")
            print(f"  Continuing without SMOTE")
        return X_train, y_train


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_xgboost_model(X_train, y_train, X_val, y_val,
                        class_weights=None, verbose=True):
    """
    Train XGBoost classifier for legacy OS version classification
    """

    if verbose:
        print("\nTraining XGBoost Model...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Validation samples: {len(X_val):,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")

    # Calculate class weights if not provided
    if class_weights is None:
        unique, counts = np.unique(y_train, return_counts=True)
        max_count = counts.max()
        class_weights = {cls: max_count / count for cls, count in zip(unique, counts)}

        if verbose:
            print(f"\n  Class distribution:")
            for cls, count in zip(unique, counts):
                print(f"    Class {cls}: {count:,} samples (weight: {class_weights[cls]:.2f})")

    # Sample weights
    sample_weights = np.array([class_weights[y] for y in y_train])

    # XGBoost parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_train)),
        'max_depth': 10,  # Deeper for detailed OS version classification
        'learning_rate': 0.1,
        'n_estimators': 300,  # More trees for complex patterns
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'min_child_weight': 2,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1,
    }

    if verbose:
        print(f"\n  XGBoost hyperparameters:")
        for key in ['max_depth', 'learning_rate', 'n_estimators', 'subsample']:
            print(f"    {key}: {params[key]}")

    # Train model
    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=False
    )

    if verbose:
        print(f"\n  Training complete!")
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Best validation score: {model.best_score:.4f}")

    return model, class_weights


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, label_encoder, output_dir, verbose=True):
    """Comprehensive model evaluation"""

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
        output_dict=True,
        zero_division=0
    )

    if verbose:
        print(f"\nPer-Class Performance:")
        print(f"  {'OS Version':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print(f"  {'-'*70}")
        for cls in class_names:
            if cls in class_report:
                metrics = class_report[cls]
                print(f"  {cls:<25} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                      f"{metrics['f1-score']:<10.3f} {int(metrics['support']):<10}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Model 2a: Legacy OS Classification - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'model2a_confusion_matrix.png')
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
        print(f"\nTop 15 Most Important Features:")
        for idx, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']:<35} {row['importance']:.4f}")

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Model 2a: Top 20 Feature Importances')
    plt.tight_layout()

    fi_path = os.path.join(output_dir, 'model2a_feature_importance.png')
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"  Feature importance plot saved: {fi_path}")
    plt.close()

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'XGBoost',
        'task': 'Legacy OS Version Classification',
        'dataset': 'nPrint',
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
            for cls in class_names if cls in class_report
        },
        'confusion_matrix': cm.tolist(),
        'class_names': class_names.tolist(),
        'feature_importance': feature_importance.to_dict('records')[:20]
    }

    results_path = os.path.join(output_dir, 'model2a_evaluation.json')
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
        description='Train Model 2a: Legacy OS Classifier (nPrint dataset)'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/nprint_packets.csv',
        help='Input CSV file with preprocessed nPrint data'
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
        '--use-smote',
        action='store_true',
        help='Apply SMOTE to balance minority classes'
    )

    parser.add_argument(
        '--smote-threshold',
        type=int,
        default=100,
        help='Minimum samples per class before applying SMOTE (default: 100)'
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

    # Header
    if verbose:
        print("="*70)
        print("MODEL 2a: LEGACY OS CLASSIFIER (nPRINT DATASET)")
        print("="*70)
        print(f"\nInput:  {args.input}")
        print(f"Output: {args.output_dir}/model2a_legacy_os.pkl")
        print(f"Algorithm: XGBoost")
        if args.use_smote:
            print(f"Balancing: SMOTE (threshold: {args.smote_threshold} samples)")

    # Load data
    if verbose:
        print(f"\n[1/7] Loading data...")

    if not os.path.exists(args.input):
        print(f"\nERROR: Input file not found: {args.input}")
        print(f"\nRun preprocessing first:")
        print(f"  python scripts/preprocess_nprint.py")
        sys.exit(1)

    df = pd.read_csv(args.input)
    if verbose:
        print(f"  Loaded {len(df):,} records")

    # Check for required columns
    if 'os_label' not in df.columns:
        print(f"\nERROR: 'os_label' column not found!")
        sys.exit(1)

    # Remove missing labels
    df = df[df['os_label'].notna()]
    if verbose:
        print(f"  Records with OS labels: {len(df):,}")

    # Show OS distribution
    if verbose:
        print(f"\n  OS Version distribution (top 15):")
        for os_label, count in df['os_label'].value_counts().head(15).items():
            pct = (count / len(df)) * 100
            print(f"    {str(os_label):<30}: {count:>8,} ({pct:>5.2f}%)")

    # Feature engineering
    if verbose:
        print(f"\n[2/7] Feature engineering...")

    # Encode TCP options
    df, tcp_opt_cols = encode_tcp_options(df, verbose=verbose)

    # Select features
    feature_columns = select_features(df, verbose=verbose)
    feature_columns.extend(tcp_opt_cols)  # Add encoded TCP option features

    if len(feature_columns) == 0:
        print(f"\nERROR: No usable features found!")
        sys.exit(1)

    # Prepare X and y
    X = df[feature_columns].copy()
    y = df['os_label'].values

    # Handle missing values
    X = handle_missing_values(X, strategy='median', verbose=verbose)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if verbose:
        print(f"\n  Encoded {len(label_encoder.classes_)} OS versions")

    # Train/validation/test split
    if verbose:
        print(f"\n[3/7] Splitting data...")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_encoded,
        test_size=args.test_size,
        stratify=y_encoded,
        random_state=args.random_state
    )

    val_size_adjusted = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size_adjusted,
        stratify=y_train_full,
        random_state=args.random_state
    )

    if verbose:
        print(f"  Training: {len(X_train):,}")
        print(f"  Validation: {len(X_val):,}")
        print(f"  Test: {len(X_test):,}")

    # Apply SMOTE if requested
    if args.use_smote:
        if verbose:
            print(f"\n[4/7] Applying SMOTE...")
        X_train, y_train = apply_smote(
            X_train, y_train,
            min_samples_threshold=args.smote_threshold,
            verbose=verbose
        )
    else:
        if verbose:
            print(f"\n[4/7] Skipping SMOTE (--use-smote not specified)")

    # Train model
    if verbose:
        print(f"\n[5/7] Training model...")

    model, class_weights = train_xgboost_model(
        X_train, y_train, X_val, y_val,
        verbose=verbose
    )

    # Evaluate
    if verbose:
        print(f"\n[6/7] Evaluating model...")

    os.makedirs(args.results_dir, exist_ok=True)
    results = evaluate_model(
        model, X_test, y_test, label_encoder,
        args.results_dir, verbose=verbose
    )

    # Save model
    if verbose:
        print(f"\n[7/7] Saving model...")

    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, 'model2a_legacy_os.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    if verbose:
        print(f"  Model saved: {model_path}")

    feature_names_path = os.path.join(args.output_dir, 'model2a_feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_columns, f)
    if verbose:
        print(f"  Feature names saved: {feature_names_path}")

    encoder_path = os.path.join(args.output_dir, 'model2a_label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    if verbose:
        print(f"  Label encoder saved: {encoder_path}")

    weights_path = os.path.join(args.output_dir, 'model2a_class_weights.pkl')
    with open(weights_path, 'wb') as f:
        pickle.dump(class_weights, f)
    if verbose:
        print(f"  Class weights saved: {weights_path}")

    # Summary
    if verbose:
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"\nModel 2a Performance:")
        print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"  F1-Score: {results['metrics']['f1_score']:.4f}")
        print(f"\nNext steps:")
        print(f"  Train Model 2b: python scripts/train_model2b_modern.py")


if __name__ == '__main__':
    main()
