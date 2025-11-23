#!/usr/bin/env python3
"""
Train Android Expert Model: Android Version Classifier

Dataset: masaryk_android.csv (Android flows from Masaryk)
Task: Classify specific Android version (7, 8, 9, 10)
Algorithm: XGBoost with optional SMOTE + class weights

Input:  data/processed/masaryk_android.csv
Output: models/AndroidExpert_<timestamp>/android_expert.pkl
        results/AndroidExpert_<timestamp>/android_expert_evaluation.json
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
    Select relevant features for Android version classification

    CRITICAL: Removes constant features that provide no discrimination
    """

    # TCP features (EXCLUDING constant syn_ack_flag)
    tcp_features = [
        'tcp_syn_size',
        'tcp_win_size',
        'tcp_syn_ttl',
        'tcp_flags_a',
        # 'syn_ack_flag',  # REMOVED: Constant (all = 1)
        'tcp_option_window_scale_forward',
        # 'tcp_option_selective_ack_permitted_forward',  # REMOVED: Very low variance
        'tcp_option_maximum_segment_size_forward',
        # 'tcp_option_no_operation_forward',  # REMOVED: Very low variance
    ]

    # IP features (EXCLUDING constant l3_proto)
    ip_features = [
        # 'l3_proto',  # REMOVED: Constant (all = 4)
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

    # TLS features (EXCLUDING constant tls_client_version)
    # TLS is 99.9% available - CRITICAL for discrimination!
    tls_features = [
        'tls_ja3_fingerprint',
        'tls_cipher_suites',
        'tls_extension_types',
        'tls_elliptic_curves',
        # 'tls_client_version',  # REMOVED: Constant (all = 771.0)
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
        print(f"\nFeature Selection (optimized for Android):")
        print(f"  Available: {len(available_features)}/19 (removed 6 constant/low-variance)")
        print(f"  Removed constant: syn_ack_flag, l3_proto, tls_client_version")
        print(f"  Removed low variance: tcp_option_selective_ack_permitted_forward, tcp_option_no_operation_forward")
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

    SMOTE is more reliable than ADASYN for multiclass balancing because it always
    generates the requested number of synthetic samples (not adaptive).

    Args:
        balance_strategy: 'full' (balance to majority), 'moderate' (balance to median),
                         'minimal' (balance smallest class only)
        use_borderline: If True, use BorderlineSMOTE (focuses on boundary samples,
                       less noise than regular SMOTE)

    CRITICAL: Imputes NaN values before SMOTE (SMOTE doesn't handle NaN)
    """
    if verbose:
        print(f"\nClass Imbalance Handling (SMOTE):")
        print(f"  Original class distribution:")
        for label, count in pd.Series(y).value_counts().items():
            print(f"    {label}: {count:,}")

    try:
        # CRITICAL FIX: Impute NaN values before SMOTE
        # SMOTE doesn't accept NaN values
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
        class_counts = dict(zip(unique, counts))

        # Determine target count based on strategy
        if balance_strategy == 'full':
            # Balance all to majority class
            target_count = max(counts)
        elif balance_strategy == 'moderate':
            # Balance all to median count (less synthetic noise)
            target_count = int(np.median(counts))
        else:  # minimal
            # Balance only smallest class to second smallest
            target_count = sorted(counts)[1]

        if verbose:
            smote_type = "BorderlineSMOTE" if use_borderline else "SMOTE"
            print(f"\n  Balancing strategy: {balance_strategy.upper()} with {smote_type}")
            print(f"  Target count: {target_count:,}")
            print(f"  Starting distribution:")
            for cls, count in zip(unique, counts):
                print(f"    Class {cls}: {count:,}")

        # Apply SMOTE iteratively (use BorderlineSMOTE for less noise)
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
            # Check current distribution
            unique_current, counts_current = np.unique(y_resampled, return_counts=True)
            min_count = min(counts_current)

            # Stop if smallest class has reached or exceeded target
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

def train_android_expert(input_path='data/processed/masaryk_android.csv',
                         output_dir='models',
                         results_dir='results',
                         use_adasyn=True,
                         balance_strategy='moderate',
                         use_borderline=False,
                         cross_validate=False,
                         verbose=True):
    """Train Android Expert Model for version classification"""

    # Create timestamped directories
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"AndroidExpert_{timestamp}"

    model_output_dir = os.path.join(output_dir, run_name)
    results_output_dir = os.path.join(results_dir, run_name)

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    print("="*80)
    print("ANDROID EXPERT MODEL - TRAINING")
    print("="*80)
    print(f"\nTask: Android version classification (7, 8, 9, 10)")
    print(f"Algorithm: XGBoost with {'SMOTE + ' if use_adasyn else ''}class weights")
    print(f"Input: {input_path}")
    print(f"\nOutput directories:")
    print(f"  Models: {model_output_dir}")
    print(f"  Results: {results_output_dir}")

    # Load data
    if verbose:
        print(f"\n[1/8] Loading dataset...")

    if not os.path.exists(input_path):
        print(f"\nERROR: File not found: {input_path}")
        print("\nPlease run the Android extraction script first:")
        print("  python extract_android.py")
        return None, 0

    df = pd.read_csv(input_path)

    if verbose:
        print(f"  Loaded {len(df):,} flows")

    # Check target column
    if 'os_label' not in df.columns:
        print(f"\nERROR: 'os_label' column not found!")
        return None, 0

    # Verify all samples are Android
    os_dist = df['os_label'].value_counts()
    if verbose:
        print(f"\n  Android version distribution:")
        for os_name, count in os_dist.items():
            print(f"    {os_name}: {count:,} ({count/len(df)*100:.2f}%)")

    # Calculate imbalance ratio
    if len(os_dist) > 1:
        max_count = os_dist.max()
        min_count = os_dist.min()
        imbalance_ratio = max_count / min_count
        print(f"\n  Imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio < 3:
            print(f"  ✓ Moderate imbalance - ADASYN optional")
        else:
            print(f"  ⚠ Higher imbalance - ADASYN recommended")

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
        X_encoded, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    if verbose:
        print(f"  Train: {len(X_train):,} samples")
        print(f"  Test:  {len(X_test):,} samples")

    # Apply SMOTE (optional)
    if use_adasyn:
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
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_resampled),
        y=y_train_resampled
    )

    sample_weights = np.array([class_weights[label] for label in y_train_resampled])

    if verbose:
        print(f"\n  Class weights (balanced):")
        for i, cls in enumerate(label_encoder.classes_):
            if i < len(class_weights):
                print(f"    {cls}: {class_weights[i]:.3f}")

    # Train XGBoost model
    # TUNED hyperparameters from tune_android_hyperparameters.py
    # Best CV F1-weighted: 0.7506
    if verbose:
        print(f"\n[6/8] Training XGBoost model (tuned hyperparameters)...")

    model = xgb.XGBClassifier(
        n_estimators=708,                      # Tuned (was 500)
        max_depth=19,                          # Tuned (was 12) - much deeper!
        learning_rate=0.070,                   # Tuned (was 0.05)
        subsample=0.989,                       # Tuned (was 0.9)
        colsample_bytree=0.904,                # Tuned (was 0.9)
        min_child_weight=1,                    # Tuned (same as before)
        gamma=0.475,                           # Tuned (was 0.0)
        reg_alpha=0.434,                       # Tuned (was 0.1)
        reg_lambda=1.618,                      # Tuned (was 1.0)
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    if verbose:
        print(f"  Using optimized hyperparameters from tuning:")
        print(f"    - n_estimators: 708 (was 500)")
        print(f"    - max_depth: 19 (was 12)")
        print(f"    - learning_rate: 0.070 (was 0.05)")
        print(f"    - reg_alpha: 0.434, reg_lambda: 1.618")
        print(f"    - Expected CV F1: 0.7506 (75.06%)")

    model.fit(
        X_train_resampled,
        y_train_resampled,
        sample_weight=sample_weights,
        verbose=verbose
    )

    if verbose:
        print(f"  ✓ Model trained")

    # Evaluate on test set
    if verbose:
        print(f"\n[7/8] Evaluating on test set...")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if verbose:
        print(f"\n  Test Accuracy: {accuracy*100:.2f}%")

    # Classification report
    class_names = label_encoder.classes_.tolist()

    print(f"\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

    # Get per-class metrics for JSON output
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

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nConfusion Matrix:")
    header = "True \\ Pred"
    print(f"{header:<15}", end="")
    for cls in class_names:
        print(f"{cls:<15}", end="")
    print()
    print("-" * (15 + 15 * len(class_names)))

    for i, true_cls in enumerate(class_names):
        print(f"{true_cls:<15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:<15}", end="")
        print()

    # Feature importance
    if verbose:
        print(f"\n" + "="*80)
        print("FEATURE IMPORTANCE (Top 15)")
        print("="*80)

    feature_importance = model.feature_importances_
    feature_names = X_encoded.columns.tolist()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print(f"\n{'Rank':<6} {'Feature':<45} {'Importance':<12}")
    print("-" * 65)
    for i, row in importance_df.head(15).iterrows():
        print(f"{importance_df.index.get_loc(i)+1:<6} {row['feature']:<45} {row['importance']*100:>10.2f}%")

    # Cross-validation (optional)
    if cross_validate:
        if verbose:
            print(f"\n[8/8] Cross-validation (5-fold)...")

        cv_scores = cross_val_score(
            model, X_train_resampled, y_train_resampled,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )

        print(f"\n  CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
        fold_scores_str = [f'{s*100:.2f}%' for s in cv_scores]
        print(f"  Fold scores: {fold_scores_str}")

    # Save model and artifacts
    if verbose:
        print(f"\n[8/8] Saving model and artifacts...")

    # Save model
    model_path = os.path.join(model_output_dir, 'android_expert.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved model: {model_path}")

    # Save feature names
    feature_names_path = os.path.join(model_output_dir, 'android_expert_feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"  ✓ Saved feature names: {feature_names_path}")

    # Save label encoder
    label_encoder_path = os.path.join(model_output_dir, 'android_expert_label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  ✓ Saved label encoder: {label_encoder_path}")

    # Save categorical encoders
    cat_encoders_path = os.path.join(model_output_dir, 'android_expert_categorical_encoders.pkl')
    with open(cat_encoders_path, 'wb') as f:
        pickle.dump(categorical_encoders, f)
    print(f"  ✓ Saved categorical encoders: {cat_encoders_path}")

    # Save evaluation results
    eval_results = {
        'model_type': 'Android Expert (Version Classifier)',
        'algorithm': 'XGBoost',
        'dataset': input_path,
        'n_samples': len(df),
        'n_features': len(feature_names),
        'classes': class_names,
        'test_accuracy': float(accuracy),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'used_smote': use_adasyn,
        'balance_strategy': balance_strategy if use_adasyn else None,
        'timestamp': timestamp,
        'feature_importance': importance_df.head(15).to_dict('records'),
    }

    eval_path = os.path.join(results_output_dir, 'android_expert_evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"  ✓ Saved evaluation: {eval_path}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Android Expert Model - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    cm_path = os.path.join(results_output_dir, 'android_expert_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved confusion matrix: {cm_path}")
    plt.close()

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance_df.head(20).plot(x='feature', y='importance', kind='barh', legend=False)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Android Expert Model - Feature Importance (Top 20)')
    plt.gca().invert_yaxis()

    fi_path = os.path.join(results_output_dir, 'android_expert_feature_importance.png')
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved feature importance: {fi_path}")
    plt.close()

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nRun: {run_name}")
    print(f"Model directory: {model_output_dir}")
    print(f"Results directory: {results_output_dir}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    return model, accuracy


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Android Expert Model for version classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_scripts/train_android_expert.py

  # Disable ADASYN (use class weights only)
  python train_scripts/train_android_expert.py --no-adasyn

  # Enable cross-validation
  python train_scripts/train_android_expert.py --cross-validate
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/masaryk_android.csv',
        help='Input CSV file with Android flows'
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
        '--no-adasyn',
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
        help='Use BorderlineSMOTE instead of regular SMOTE (focuses on decision boundary samples, less noise)'
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
    model, accuracy = train_android_expert(
        input_path=args.input,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        use_adasyn=not args.no_adasyn,
        balance_strategy=args.balance_strategy,
        use_borderline=args.borderline_smote,
        cross_validate=args.cross_validate,
        verbose=not args.quiet
    )

    if model is None:
        sys.exit(1)

    print(f"\n✓ Training successful!")
    print(f"  Accuracy: {accuracy*100:.2f}%")


if __name__ == '__main__':
    main()
