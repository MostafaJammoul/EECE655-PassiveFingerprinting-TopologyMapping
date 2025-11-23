#!/usr/bin/env python3
"""
Train Linux Expert Model: Linux Distribution/Version Classifier

Dataset: linux.csv (Linux flows from Masaryk/CESNET)
Task: Classify specific Linux distribution/version
Algorithm: XGBoost with optional SMOTE + class weights

Input:  data/processed/linux.csv
Output: models/LinuxExpert_<timestamp>/linux_expert.pkl
        results/LinuxExpert_<timestamp>/linux_expert_evaluation.json
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
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    from sklearn.utils.class_weight import compute_class_weight
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"ERROR: Missing required library: {e}")
    print("\nInstall with:")
    print("  pip install xgboost scikit-learn matplotlib seaborn imbalanced-learn")
    sys.exit(1)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def select_features(df, verbose=True):
    """
    Select relevant features for Linux version classification

    Same 25-feature structure as Masaryk dataset (used by Android, Windows, etc.)
    """

    # TCP features
    tcp_features = [
        'tcp_syn_size',
        'tcp_win_size',
        'tcp_syn_ttl',
        'tcp_flags_a',
        'syn_ack_flag',
        'tcp_option_window_scale_forward',
        'tcp_option_selective_ack_permitted_forward',
        'tcp_option_maximum_segment_size_forward',
        'tcp_option_no_operation_forward',
    ]

    # IP features
    ip_features = [
        'l3_proto',
        'ip_tos',
        'maximum_ttl_forward',
        'ipv4_dont_fragment_forward',
    ]

    flow_features = [
        'src_port',
        'packet_total_count_forward',
        'packet_total_count_backward',
        'total_bytes',
    ]

    # TLS features - CRITICAL for discrimination
    tls_features = [
        'tls_ja3_fingerprint',
        'tls_cipher_suites',
        'tls_extension_types',
        'tls_elliptic_curves',
        'tls_client_version',
        'tls_handshake_type',
        'tls_client_key_length',
    ]

    derived_features = [
        'initial_ttl',
    ]

    all_features = tcp_features + ip_features + flow_features + tls_features + derived_features

    available_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]

    if verbose:
        print(f"\nFeature Selection (Linux Expert):")
        print(f"  Available: {len(available_features)}/25 features")
        if missing_features:
            print(f"  Missing: {missing_features}")

    return available_features


def encode_categorical_features(X, encoders=None, verbose=True):
    """Encode categorical/string features using LabelEncoder"""
    from sklearn.preprocessing import LabelEncoder

    string_columns = X.select_dtypes(include=['object']).columns.tolist()

    if not string_columns:
        if verbose:
            print(f"\nCategorical Encoding:")
            print(f"  No string features to encode")
        return X, {}

    if verbose:
        print(f"\nCategorical Encoding:")
        print(f"  Encoding {len(string_columns)} string features")

    X_encoded = X.copy()

    if encoders is None:
        encoders = {}
        for col in string_columns:
            le = LabelEncoder()
            X_encoded[col] = X_encoded[col].fillna('__MISSING__')
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le

            if verbose:
                print(f"    {col}: {len(le.classes_)} unique values")
    else:
        for col in string_columns:
            if col in encoders:
                le = encoders[col]
                X_encoded[col] = X_encoded[col].fillna('__MISSING__')
                X_encoded[col] = X_encoded[col].apply(
                    lambda x: x if x in le.classes_ else '__MISSING__'
                )
                X_encoded[col] = le.transform(X_encoded[col].astype(str))
            else:
                if verbose:
                    print(f"  WARNING: No encoder found for {col}, filling with 0")
                X_encoded[col] = 0

    return X_encoded, encoders


def apply_smote(X, y, verbose=True, balance_strategy='full', use_borderline=False):
    """
    Apply SMOTE oversampling for class imbalance

    Args:
        balance_strategy: 'full' (balance to majority), 'moderate' (balance to median),
                         'minimal' (balance smallest class only)
        use_borderline: If True, use BorderlineSMOTE (focuses on boundary samples)

    CRITICAL: Imputes NaN values before SMOTE
    """
    if verbose:
        print(f"\nClass Imbalance Handling (SMOTE):")
        print(f"  Original class distribution:")
        for label, count in pd.Series(y).value_counts().items():
            print(f"    {label}: {count:,}")

    try:
        from sklearn.impute import SimpleImputer

        # Check if there are NaN values
        has_nan = X.isnull().any().any() if isinstance(X, pd.DataFrame) else np.isnan(X).any()

        if has_nan:
            if verbose:
                nan_cols = X.columns[X.isnull().any()].tolist() if isinstance(X, pd.DataFrame) else []
                print(f"\n  Imputing NaN values in {len(nan_cols)} features before SMOTE")

            imputer = SimpleImputer(strategy='median')
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1]),
                index=X.index if isinstance(X, pd.DataFrame) else range(X.shape[0])
            )
        else:
            X_imputed = X

        # SMOTE with configurable balancing strategy
        unique, counts = np.unique(y, return_counts=True)
        n_classes = len(unique)

        # Determine target count based on strategy
        if balance_strategy == 'full':
            target_count = max(counts)
        elif balance_strategy == 'moderate':
            target_count = int(np.median(counts))
        else:  # minimal
            target_count = sorted(counts)[1]

        if verbose:
            smote_type = "BorderlineSMOTE" if use_borderline else "SMOTE"
            print(f"\n  Balancing strategy: {balance_strategy.upper()} with {smote_type}")
            print(f"  Target count: {target_count:,}")
            print(f"  Starting distribution:")
            for cls, count in zip(unique, counts):
                print(f"    Class {cls}: {count:,}")

        # Apply SMOTE iteratively
        if use_borderline:
            smote = BorderlineSMOTE(random_state=42, k_neighbors=3, sampling_strategy='minority')
        else:
            smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy='minority')

        X_resampled = X_imputed.copy()
        y_resampled = y.copy()

        if verbose:
            print(f"\n  Applying iterative SMOTE (stopping when target reached)...")

        # Iterate until all minority classes reach target_count
        max_iterations = n_classes - 1
        for iteration in range(max_iterations):
            unique_current, counts_current = np.unique(y_resampled, return_counts=True)
            min_count = min(counts_current)

            if min_count >= target_count:
                if verbose:
                    print(f"    Target reached after {iteration} iteration(s)")
                break

            X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

            if verbose:
                unique_iter, counts_iter = np.unique(y_resampled, return_counts=True)
                print(f"    Iteration {iteration + 1}: {dict(zip(unique_iter, counts_iter))}")

        if verbose:
            print(f"\n  After SMOTE:")
            for label, count in pd.Series(y_resampled).value_counts().items():
                print(f"    {label}: {count:,}")
            print(f"\n  Total samples: {len(y):,} → {len(y_resampled):,} (+{len(y_resampled)-len(y):,})")

        return X_resampled, y_resampled

    except Exception as e:
        print(f"\n  WARNING: SMOTE failed: {e}")
        print(f"  Falling back to original data with class weights only")
        return X, y


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_linux_expert(input_path='data/processed/linux.csv',
                       output_dir='models',
                       results_dir='results',
                       use_smote=True,
                       balance_strategy='moderate',
                       use_borderline=False,
                       cross_validate=False,
                       verbose=True):
    """Train Linux Expert Model for distribution/version classification"""

    # Create timestamped directories
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"LinuxExpert_{timestamp}"

    model_output_dir = os.path.join(output_dir, run_name)
    results_output_dir = os.path.join(results_dir, run_name)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    if verbose:
        print("="*80)
        print("LINUX EXPERT MODEL - TRAINING")
        print("="*80)
        print(f"\nTask: Linux distribution/version classification")
        print(f"Algorithm: XGBoost with {'SMOTE + ' if use_smote else ''}class weights")
        print(f"Input: {input_path}")
        print(f"\nOutput directories:")
        print(f"  Models: {model_output_dir}")
        print(f"  Results: {results_output_dir}")

    # Load data
    if verbose:
        print(f"\n[1/8] Loading data...")

    if not os.path.exists(input_path):
        print(f"\n✗ ERROR: Input file not found: {input_path}")
        print(f"\nPlease create linux.csv first using:")
        print(f"  - Extract Linux flows from Masaryk/CESNET datasets")
        print(f"  - Ensure format matches masaryk_android.csv (25 features + os_label)")
        return None, None

    df = pd.read_csv(input_path)

    if verbose:
        print(f"  Loaded {len(df):,} samples")

    if 'os_label' not in df.columns:
        print(f"\n✗ ERROR: Dataset must contain 'os_label' column")
        return None, None

    # Check class distribution
    if verbose:
        print(f"\n  OS Label distribution:")
        for os_label, count in df['os_label'].value_counts().items():
            percentage = (count / len(df)) * 100
            print(f"    {os_label:<30}: {count:>6,} ({percentage:>5.1f}%)")

    # Select features
    if verbose:
        print(f"\n[2/8] Selecting features...")

    feature_cols = select_features(df, verbose=verbose)

    X = df[feature_cols].copy()
    y = df['os_label'].copy()

    # Encode categorical features
    if verbose:
        print(f"\n[3/8] Encoding categorical features...")

    X_encoded, categorical_encoders = encode_categorical_features(X, verbose=verbose)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if verbose:
        print(f"\n  Target classes: {label_encoder.classes_.tolist()}")

    # Split train/test
    if verbose:
        print(f"\n[4/8] Splitting train/test (80/20, stratified)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    if verbose:
        print(f"  Train: {len(X_train):,} samples")
        print(f"  Test:  {len(X_test):,} samples")

    # Apply SMOTE (optional)
    if use_smote:
        if verbose:
            print(f"\n[5/8] Applying SMOTE to training set...")
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train,
                                                            verbose=verbose,
                                                            balance_strategy=balance_strategy,
                                                            use_borderline=use_borderline)
    else:
        if verbose:
            print(f"\n[5/8] Skipping SMOTE (disabled)")
        X_train_resampled = X_train
        y_train_resampled = y_train

    # Compute class weights
    unique_classes = np.unique(y_train_resampled)
    class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=y_train_resampled)
    class_weights = dict(zip(unique_classes, class_weights_array))

    if verbose:
        print(f"\n[6/8] Computing class weights...")
        print(f"  Class weights (balanced):")
        for cls_idx, weight in class_weights.items():
            cls_name = label_encoder.classes_[cls_idx]
            print(f"    {cls_name}: {weight:.2f}")

    # Train model
    if verbose:
        print(f"\n[7/8] Training XGBoost model...")

    # Use conservative hyperparameters (can be tuned later)
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.05,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective='multi:softmax',
        num_class=len(unique_classes)
    )

    # Apply class weights
    sample_weights = np.array([class_weights[label] for label in y_train_resampled])

    if verbose:
        print(f"  Training with {len(unique_classes)} classes...")

    model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights, verbose=False)

    # Cross-validation (optional)
    if cross_validate:
        if verbose:
            print(f"\n  Performing 5-fold cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
        if verbose:
            fold_scores_str = [f'{s*100:.2f}%' for s in cv_scores]
            print(f"  Fold scores: {fold_scores_str}")
            print(f"  CV Mean Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Evaluate
    if verbose:
        print(f"\n[8/8] Evaluating on test set...")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if verbose:
        print(f"\n{'='*80}")
        print(f"TEST SET ACCURACY: {accuracy*100:.2f}%")
        print(f"{'='*80}")

    # Classification report
    class_names = label_encoder.classes_.tolist()
    if verbose:
        print(f"\nDetailed Classification Report:")
        print(f"{'-'*80}")
        print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if verbose:
        print(f"\nConfusion Matrix:")
        print(f"{'-'*80}")
        header = "True \\ Pred"
        print(f"{header:<15}", end="")
        for cls in class_names:
            print(f"{cls:<15}", end="")
        print()
        print(f"{'-'*80}")

        for i, true_class in enumerate(class_names):
            print(f"{true_class:<15}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i][j]:<15}", end="")
            print()

    # Feature importance
    importance = model.feature_importances_
    feature_names = X_encoded.columns.tolist()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    if verbose:
        print(f"\n\nTop 15 Most Important Features:")
        print(f"{'-'*80}")
        for idx, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']:<40} {row['importance']:.4f}")

    # Save model
    model_path = os.path.join(model_output_dir, 'linux_expert.pkl')
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'categorical_encoders': categorical_encoders,
        'feature_columns': feature_cols,
        'class_names': class_names
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    if verbose:
        print(f"\n\nModel saved to: {model_path}")

    # Save evaluation results
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(class_names))
    )

    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }

    results = {
        'model_type': 'LinuxExpert',
        'classes': class_names,
        'test_accuracy': float(accuracy),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'used_smote': use_smote,
        'balance_strategy': balance_strategy if use_smote else None,
        'timestamp': timestamp,
        'feature_importance': importance_df.head(15).to_dict('records'),
    }

    eval_path = os.path.join(results_output_dir, 'linux_expert_evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"Evaluation results saved to: {eval_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nRun: {run_name}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Model: {model_path}")
    print(f"Results: {eval_path}")

    return model_data, accuracy


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Linux Expert Model for Linux distribution/version classification'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/linux.csv',
        help='Path to Linux dataset CSV file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save model files'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save evaluation results'
    )

    parser.add_argument(
        '--no-smote',
        action='store_true',
        help='Disable SMOTE oversampling (use class weights only)'
    )

    parser.add_argument(
        '--balance-strategy',
        type=str,
        choices=['full', 'moderate', 'minimal'],
        default='moderate',
        help='SMOTE balancing strategy: full (balance to majority), moderate (balance to median - default), minimal (balance smallest class only)'
    )

    parser.add_argument(
        '--borderline-smote',
        action='store_true',
        help='Use BorderlineSMOTE instead of regular SMOTE (focuses on decision boundary samples)'
    )

    parser.add_argument(
        '--cross-validate',
        action='store_true',
        help='Perform 5-fold cross-validation'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Train model
    model, accuracy = train_linux_expert(
        input_path=args.input,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        use_smote=not args.no_smote,
        balance_strategy=args.balance_strategy,
        use_borderline=args.borderline_smote,
        cross_validate=args.cross_validate,
        verbose=not args.quiet
    )

    if model is None:
        sys.exit(1)
