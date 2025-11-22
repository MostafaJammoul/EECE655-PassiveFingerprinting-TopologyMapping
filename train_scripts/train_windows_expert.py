#!/usr/bin/env python3
"""
Train Windows Expert Model: Windows Version Classifier

Dataset: nprint_windows_flows.csv (flows initiated by Windows machines)
Task: Classify specific Windows version (7, 8, 10, Vista)
Algorithm: XGBoost with ADASYN oversampling + class weights

Input:  data/processed/nprint_windows_flows.csv
Output: models/windows_expert.pkl
        models/windows_expert_feature_names.pkl
        models/windows_expert_label_encoder.pkl
        models/windows_expert_categorical_encoders.pkl
        results/windows_expert_evaluation.json
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
    from imblearn.over_sampling import ADASYN
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
    Select relevant features for Windows version classification

    Uses same 25 features as Model 1 (Masaryk) but for Windows-specific
    discrimination between versions 7, 8, 10, Vista.
    """

    # Core TCP fingerprinting features (9 features)
    tcp_features = [
        'tcp_syn_size',
        'tcp_win_size',
        'tcp_syn_ttl',
        'tcp_flags_a',  # TCP flags string - requires encoding
        'syn_ack_flag',

        # TCP Options - Forward direction only
        'tcp_option_window_scale_forward',
        'tcp_option_selective_ack_permitted_forward',
        'tcp_option_maximum_segment_size_forward',
        'tcp_option_no_operation_forward',
    ]

    # IP-level features (4 features)
    ip_features = [
        'l3_proto',
        'ip_tos',
        'maximum_ttl_forward',
        'ipv4_dont_fragment_forward',
    ]

    # Flow metadata (4 features)
    flow_features = [
        'src_port',
        'packet_total_count_forward',
        'packet_total_count_backward',
        'total_bytes',
    ]

    # TLS fingerprinting features (7 features - require encoding)
    tls_features = [
        'tls_ja3_fingerprint',
        'tls_cipher_suites',
        'tls_extension_types',
        'tls_elliptic_curves',
        'tls_client_version',
        'tls_handshake_type',
        'tls_client_key_length',
    ]

    # Derived features (1 feature)
    derived_features = [
        'initial_ttl',
    ]

    # Combine all 25 features
    all_features = tcp_features + ip_features + flow_features + tls_features + derived_features

    # Select only features that exist in the dataframe
    available_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]

    if verbose:
        print(f"\nFeature Selection:")
        print(f"  Available: {len(available_features)}/25")
        print(f"  Missing: {len(missing_features)}")
        if missing_features:
            print(f"  Missing features: {missing_features}")

    return available_features


def encode_categorical_features(X, encoders=None, verbose=True):
    """
    Encode categorical/string features using LabelEncoder

    String features like tcp_flags_a, TLS features need to be encoded
    to numeric values before training.

    Args:
        X: DataFrame with features
        encoders: Dict of pre-fitted encoders (for inference). If None, fit new encoders.
        verbose: Print progress

    Returns:
        X_encoded: DataFrame with encoded features
        encoders: Dict of fitted LabelEncoders
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
        print(f"  Encoding {len(string_columns)} string features")

    X_encoded = X.copy()

    if encoders is None:
        # Training mode - fit new encoders
        encoders = {}
        for col in string_columns:
            le = LabelEncoder()
            # Handle NaN values by replacing with a special token
            X_encoded[col] = X_encoded[col].fillna('__MISSING__')
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le

            if verbose:
                print(f"    {col}: {len(le.classes_)} unique values")
    else:
        # Inference mode - use pre-fitted encoders
        for col in string_columns:
            if col in encoders:
                le = encoders[col]
                X_encoded[col] = X_encoded[col].fillna('__MISSING__')

                # Handle unseen categories
                X_encoded[col] = X_encoded[col].apply(
                    lambda x: x if x in le.classes_ else '__MISSING__'
                )
                X_encoded[col] = le.transform(X_encoded[col].astype(str))
            else:
                if verbose:
                    print(f"  WARNING: No encoder found for {col}, filling with 0")
                X_encoded[col] = 0

    return X_encoded, encoders


def apply_adasyn(X, y, verbose=True):
    """
    Apply ADASYN (Adaptive Synthetic Sampling) to balance classes

    ADASYN focuses on generating synthetic samples for minority classes
    near the decision boundary (harder-to-learn samples).

    Better than SMOTE for Windows 10 (severe minority: 966 vs 5,784 samples).
    """
    if verbose:
        print(f"\nClass Imbalance Handling (ADASYN):")
        print(f"  Original class distribution:")
        for label, count in pd.Series(y).value_counts().items():
            print(f"    {label}: {count:,}")

    try:
        adasyn = ADASYN(random_state=42, n_neighbors=5, sampling_strategy='minority')
        X_resampled, y_resampled = adasyn.fit_resample(X, y)

        if verbose:
            print(f"\n  After ADASYN:")
            for label, count in pd.Series(y_resampled).value_counts().items():
                print(f"    {label}: {count:,}")
            print(f"\n  Total samples: {len(y):,} → {len(y_resampled):,} (+{len(y_resampled)-len(y):,})")

        return X_resampled, y_resampled

    except Exception as e:
        print(f"\n  WARNING: ADASYN failed: {e}")
        print(f"  Falling back to original data with class weights only")
        return X, y


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_windows_expert(input_path='data/processed/nprint_windows_flows.csv',
                         output_dir='models',
                         results_dir='results',
                         use_adasyn=True,
                         cross_validate=False,
                         verbose=True):
    """
    Train Windows Expert Model for version classification

    Args:
        input_path: Path to nprint_windows_flows.csv
        output_dir: Directory to save model files
        results_dir: Directory to save evaluation results
        use_adasyn: Apply ADASYN oversampling for class imbalance
        cross_validate: Perform cross-validation
        verbose: Print progress

    Returns:
        model: Trained XGBoost model
        accuracy: Test accuracy
    """

    # Create timestamped directories
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"WindowsExpert_{timestamp}"

    model_output_dir = os.path.join(output_dir, run_name)
    results_output_dir = os.path.join(results_dir, run_name)

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    print("="*80)
    print("WINDOWS EXPERT MODEL - TRAINING")
    print("="*80)
    print(f"\nTask: Windows version classification (7, 8, 10, Vista)")
    print(f"Algorithm: XGBoost with ADASYN + class weights")
    print(f"Input: {input_path}")
    print(f"\nOutput directories:")
    print(f"  Models: {model_output_dir}")
    print(f"  Results: {results_output_dir}")

    # Load data
    if verbose:
        print(f"\n[1/8] Loading dataset...")

    if not os.path.exists(input_path):
        print(f"\nERROR: File not found: {input_path}")
        print("\nPlease run the Windows extraction script first:")
        print("  python preprocess_scripts/analyze_nprint_flows.py \\")
        print("    --pcap data/raw/nprint/os-100-packet.pcapng \\")
        print("    --output data/processed/nprint_windows_flows.csv \\")
        print("    --use-windows-ips")
        return None, 0

    df = pd.read_csv(input_path)

    if verbose:
        print(f"  Loaded {len(df):,} flows")

    # Check target column
    if 'os_family' not in df.columns:
        print(f"\nERROR: 'os_family' column not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return None, 0

    # Verify all samples are Windows
    os_dist = df['os_family'].value_counts()
    if verbose:
        print(f"\n  Windows version distribution:")
        for os_name, count in os_dist.items():
            print(f"    {os_name}: {count:,} ({count/len(df)*100:.2f}%)")

    # Filter to only Windows versions (should already be filtered, but double-check)
    windows_versions = ['Windows 7', 'Windows 8', 'Windows 10', 'Windows Vista', 'Windows XP']
    df = df[df['os_family'].isin(windows_versions)].copy()

    if len(df) == 0:
        print(f"\nERROR: No Windows samples found!")
        return None, 0

    # Select features
    if verbose:
        print(f"\n[2/8] Selecting features...")

    feature_cols = select_features(df, verbose=verbose)

    X = df[feature_cols].copy()
    y = df['os_family'].copy()

    # Encode categorical features
    if verbose:
        print(f"\n[3/8] Encoding categorical features...")

    X_encoded, categorical_encoders = encode_categorical_features(X, verbose=verbose)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if verbose:
        print(f"\n  Target classes: {label_encoder.classes_.tolist()}")

    # Split train/test (stratified to preserve class distribution)
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

    # Apply ADASYN to training set only
    if use_adasyn:
        if verbose:
            print(f"\n[5/8] Applying ADASYN to training set...")
        X_train_resampled, y_train_resampled = apply_adasyn(X_train, y_train, verbose=verbose)
    else:
        if verbose:
            print(f"\n[5/8] Skipping ADASYN (disabled)")
        X_train_resampled = X_train
        y_train_resampled = y_train

    # Compute class weights (for additional weighting in loss function)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_resampled),
        y=y_train_resampled
    )

    # Convert to sample weights
    sample_weights = np.array([class_weights[label] for label in y_train_resampled])

    if verbose:
        print(f"\n  Class weights (balanced):")
        for i, cls in enumerate(label_encoder.classes_):
            if i < len(class_weights):
                print(f"    {cls}: {class_weights[i]:.3f}")

    # Train XGBoost model
    if verbose:
        print(f"\n[6/8] Training XGBoost model...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

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

    # Detailed classification report
    class_names = label_encoder.classes_.tolist()

    print(f"\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nConfusion Matrix:")
    print(f"{'True \\ Pred':<15}", end="")
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
        print(f"  Fold scores: {[f'{s*100:.2f}%' for s in cv_scores]}")

    # Save model and artifacts
    if verbose:
        print(f"\n[8/8] Saving model and artifacts...")

    # Save model
    model_path = os.path.join(model_output_dir, 'windows_expert.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved model: {model_path}")

    # Save feature names
    feature_names_path = os.path.join(model_output_dir, 'windows_expert_feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"  ✓ Saved feature names: {feature_names_path}")

    # Save label encoder
    label_encoder_path = os.path.join(model_output_dir, 'windows_expert_label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  ✓ Saved label encoder: {label_encoder_path}")

    # Save categorical encoders
    cat_encoders_path = os.path.join(model_output_dir, 'windows_expert_categorical_encoders.pkl')
    with open(cat_encoders_path, 'wb') as f:
        pickle.dump(categorical_encoders, f)
    print(f"  ✓ Saved categorical encoders: {cat_encoders_path}")

    # Save evaluation results
    eval_results = {
        'model_type': 'Windows Expert (Version Classifier)',
        'algorithm': 'XGBoost',
        'dataset': input_path,
        'n_samples': len(df),
        'n_features': len(feature_names),
        'classes': class_names,
        'test_accuracy': float(accuracy),
        'used_adasyn': use_adasyn,
        'timestamp': timestamp,
        'feature_importance': importance_df.head(15).to_dict('records'),
    }

    eval_path = os.path.join(results_output_dir, 'windows_expert_evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"  ✓ Saved evaluation: {eval_path}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Windows Expert Model - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    cm_path = os.path.join(results_output_dir, 'windows_expert_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved confusion matrix: {cm_path}")
    plt.close()

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance_df.head(20).plot(x='feature', y='importance', kind='barh', legend=False)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Windows Expert Model - Feature Importance (Top 20)')
    plt.gca().invert_yaxis()

    fi_path = os.path.join(results_output_dir, 'windows_expert_feature_importance.png')
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
        description='Train Windows Expert Model for version classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_scripts/train_windows_expert.py

  # With custom input
  python train_scripts/train_windows_expert.py --input data/processed/nprint_windows_flows.csv

  # Disable ADASYN (use class weights only)
  python train_scripts/train_windows_expert.py --no-adasyn

  # Enable cross-validation
  python train_scripts/train_windows_expert.py --cross-validate
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/nprint_windows_flows.csv',
        help='Input CSV file with Windows flows'
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
        help='Disable ADASYN oversampling (use class weights only)'
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
    model, accuracy = train_windows_expert(
        input_path=args.input,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        use_adasyn=not args.no_adasyn,
        cross_validate=args.cross_validate,
        verbose=not args.quiet
    )

    if model is None:
        sys.exit(1)

    print(f"\n✓ Training successful!")
    print(f"  Accuracy: {accuracy*100:.2f}%")


if __name__ == '__main__':
    main()
