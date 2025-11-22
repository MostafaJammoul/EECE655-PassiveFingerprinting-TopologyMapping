#!/usr/bin/env python3
"""
Train Model 1: OS Family Classifier

Dataset: Masaryk (TCP SYN flow-level features)
Task: Classify OS family (Windows, Linux, macOS, Android, iOS, BSD)
Algorithm: XGBoost (best for medium-large datasets with categorical targets)

Input:  data/processed/masaryk.csv
Output: models/model1_os_family.pkl
        models/model1_feature_names.pkl
        models/model1_label_encoder.pkl
        models/model1_categorical_encoders.pkl
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
        'tcp_syn_ttl',  # TCP SYN TTL value
        'tcp_flags_a',  # TCP flags string (e.g., "---AP-SF") - requires encoding
        'syn_ack_flag',  # SYN-ACK flag indicator

        # TCP Options (CRITICAL - Different OSes have unique patterns)
        'tcp_option_window_scale_forward',  # NEW: HIGH importance
        'tcp_option_window_scale_backward',  # NEW: HIGH importance
        'tcp_option_maximum_segment_size_forward',  # NEW: HIGH importance
        'tcp_option_maximum_segment_size_backward',  # NEW: HIGH importance
        'tcp_option_selective_ack_permitted_forward',  # NEW: MEDIUM importance
        'tcp_option_selective_ack_permitted_backward',  # NEW: MEDIUM importance
        # 'tcp_timestamp_forward',  # REMOVED - 0.00% availability in Masaryk
        # 'tcp_timestamp_backward',  # REMOVED - 0.00% availability in Masaryk
        'tcp_option_no_operation_forward',  # NEW: LOW importance
        'tcp_option_no_operation_backward',  # NEW: LOW importance
    ]

    # IP-level features (ENHANCED with critical discriminators)
    ip_features = [
        'initial_ttl',  # Estimated initial TTL (64=Linux/Mac, 128=Windows)
        'ipv4_dont_fragment_forward',  # Don't Fragment flag forward
        'ipv4_dont_fragment_backward',  # Don't Fragment flag backward
        'ip_tos',  # Type of Service
        'maximum_ttl_forward',  # Maximum TTL observed forward
        'maximum_ttl_backward',  # Maximum TTL observed backward
        'l3_proto',  # L3 protocol
        'l4_proto',  # L4 protocol
    ]

    # Flow-level behavioral features
    flow_features = [
        'bytes_a',  # Total bytes in direction A
        'packets_a',  # Total packets in direction A
        'packet_total_count_forward',  # Packet count forward
        'packet_total_count_backward',  # Packet count backward
        'total_bytes',  # Total bytes (derived feature)
    ]

    # Port-based features (some services are OS-specific)
    port_features = [
        'src_port',
        'dst_port',
    ]

    # NPM timing features - REMOVED (not available in CESNET, not needed for deployment)
    # npm_features = []

    # TLS fingerprinting features (string features - require encoding)
    tls_features = [
        'tls_handshake_type',
        'tls_client_version',
        'tls_cipher_suites',
        'tls_extension_types',
        'tls_elliptic_curves',
        'tls_client_key_length',
        'tls_ja3_fingerprint',
    ]

    # Combine all features (NPM features excluded - not available in CESNET)
    all_features = tcp_features + ip_features + flow_features + port_features + tls_features

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


def encode_categorical_features(X, verbose=True):
    """
    Encode categorical/string features using LabelEncoder

    String features like tcp_flags_a, TLS features need to be encoded
    to numeric values before training.
    """
    from sklearn.preprocessing import LabelEncoder

    # Identify string/object columns
    string_columns = X.select_dtypes(include=['object']).columns.tolist()

    if not string_columns:
        if verbose:
            print(f"\nCategorical Encoding:")
            print(f"  No string features to encode")
        return X, {}

    if verbose:
        print(f"\nCategorical Encoding:")
        print(f"  Encoding {len(string_columns)} string features:")
        for col in string_columns:
            unique_count = X[col].nunique()
            print(f"    {col}: {unique_count} unique values")

    X_encoded = X.copy()
    encoders = {}

    for col in string_columns:
        le = LabelEncoder()
        # Handle NaN values by filling with a placeholder
        X_encoded[col] = X_encoded[col].fillna('__MISSING__')
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le

    return X_encoded, encoders


def handle_missing_values(X, strategy='median', verbose=True):
    """
    Handle missing values in feature matrix

    Strategy options:
    - 'median': Replace with median (best for continuous features)
    - 'mean': Replace with mean
    - 'zero': Replace with 0

    Note: This should be called AFTER encode_categorical_features()
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
        'early_stopping_rounds': 20,  # Moved to constructor in XGBoost 2.0+
        'random_state': 42,
        'n_jobs': -1,
    }

    if verbose:
        print(f"\n  XGBoost parameters:")
        for key, value in params.items():
            if key not in ['eval_metric', 'objective', 'early_stopping_rounds']:
                print(f"    {key}: {value}")

    # Create and train model
    model = xgb.XGBClassifier(**params)

    # Train with validation set
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
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
        default='data/processed/masaryk.csv',
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
        '--use-adasyn',
        action='store_true',
        help='Use ADASYN oversampling to handle class imbalance (requires imbalanced-learn)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Create dated folder name for this training run
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dated_folder = f"FamilyClassifier_{date_str}"

    # Update output directories to use dated folders
    args.output_dir = os.path.join(args.output_dir, dated_folder)
    args.results_dir = os.path.join(args.results_dir, dated_folder)

    # Print header
    if verbose:
        print("="*70)
        print("MODEL 1: OS FAMILY CLASSIFIER (MASARYK DATASET)")
        print("="*70)
        print(f"\nInput:  {args.input}")
        print(f"Output: {args.output_dir}/")
        print(f"Results: {args.results_dir}/")
        print(f"Algorithm: XGBoost")
        if args.use_adasyn:
            print(f"Oversampling: ADASYN")

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

    # Encode categorical/string features (e.g., tcp_flags_a, TLS features)
    X, categorical_encoders = encode_categorical_features(X, verbose=verbose)

    # Handle missing values (after encoding)
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

    # Apply ADASYN oversampling if requested
    if args.use_adasyn:
        if verbose:
            print(f"\n[3.5/6] Applying ADASYN oversampling...")

        try:
            from imblearn.over_sampling import ADASYN

            # Show class distribution before ADASYN
            if verbose:
                print(f"\n  Class distribution BEFORE ADASYN:")
                unique, counts = np.unique(y_train, return_counts=True)
                for cls, count in zip(unique, counts):
                    cls_name = label_encoder.inverse_transform([cls])[0]
                    print(f"    {cls_name:<15}: {count:>6,}")

            # Apply ADASYN
            adasyn = ADASYN(random_state=args.random_state)
            X_train, y_train = adasyn.fit_resample(X_train, y_train)

            # Show class distribution after ADASYN
            if verbose:
                print(f"\n  Class distribution AFTER ADASYN:")
                unique, counts = np.unique(y_train, return_counts=True)
                for cls, count in zip(unique, counts):
                    cls_name = label_encoder.inverse_transform([cls])[0]
                    print(f"    {cls_name:<15}: {count:>6,}")

                print(f"\n  ✓ ADASYN completed: {len(X_train):,} training samples")

        except ImportError:
            print(f"\n  ✗ ERROR: imbalanced-learn not installed!")
            print(f"  Install with: pip install imbalanced-learn")
            sys.exit(1)

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

    # Save categorical encoders (for string features like tcp_flags_a, TLS features)
    cat_encoder_path = os.path.join(args.output_dir, 'model1_categorical_encoders.pkl')
    with open(cat_encoder_path, 'wb') as f:
        pickle.dump(categorical_encoders, f)
    if verbose:
        print(f"  Categorical encoders saved: {cat_encoder_path}")

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
