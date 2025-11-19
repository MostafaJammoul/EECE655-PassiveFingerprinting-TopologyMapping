#!/usr/bin/env python3
"""
Train Model 2b: Modern OS Version Classifier

Dataset: CESNET Idle (TCP SYN packet-level features)
Task: Classify modern OS versions (Win10/11, Ubuntu 22/24, Fedora, macOS 13+, Android)
Algorithm: Random Forest (more robust for small datasets)

CRITICAL: This dataset has only ~1.8k samples!
Solutions implemented:
1. Random Forest (less prone to overfitting than XGBoost)
2. SMOTE for minority classes
3. Class merging option (combine similar OS versions)
4. Extensive cross-validation

Input:  data/processed/cesnet_idle_packets.csv
Output: models/model2b_modern_os.pkl
        models/model2b_feature_names.pkl
        models/model2b_label_encoder.pkl
        results/model2b_evaluation.json
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
import re

# ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report
    )
    from imblearn.over_sampling import ADASYN, SMOTE
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"ERROR: Missing required library: {e}")
    print("\nInstall with:")
    print("  pip install scikit-learn imbalanced-learn matplotlib seaborn")
    sys.exit(1)


# ============================================================================
# CLASS GROUPING (to combat small dataset)
# ============================================================================

def merge_similar_os_versions(df, os_column='os_label', verbose=True):
    """
    Merge similar OS versions to increase samples per class

    Examples:
    - "Windows 10" + "Windows 11" → "Windows 10/11"
    - "Ubuntu 22.04" + "Ubuntu 24.04" → "Ubuntu 22+"
    - "macOS 13" + "macOS 14" + "macOS 15" → "macOS 13+"
    - "Fedora 36" + "Fedora 37" + "Fedora 38" → "Fedora 36+"
    """

    if verbose:
        print(f"\nMerging similar OS versions...")
        print(f"  Original distribution:")
        for os_label, count in df[os_column].value_counts().head(15).items():
            print(f"    {str(os_label):<30}: {count:>6,}")

    df_merged = df.copy()

    def merge_os_label(label):
        """Apply merging rules"""
        label_str = str(label).lower()

        # Windows: Merge 10 and 11
        if 'windows 10' in label_str or 'windows 11' in label_str or 'win10' in label_str or 'win11' in label_str:
            return 'Windows 10/11'
        elif 'windows' in label_str:
            return 'Windows Other'

        # Ubuntu: Merge 22.04 and 24.04
        if 'ubuntu 22' in label_str or 'ubuntu 24' in label_str:
            return 'Ubuntu 22+'
        elif 'ubuntu 20' in label_str or 'ubuntu 21' in label_str:
            return 'Ubuntu 20+'
        elif 'ubuntu' in label_str:
            return 'Ubuntu Other'

        # Debian: Group modern versions
        if 'debian 11' in label_str or 'debian 12' in label_str or 'debian 13' in label_str:
            return 'Debian 11+'
        elif 'debian' in label_str:
            return 'Debian Other'

        # Fedora: Merge recent versions
        if any(f'fedora {v}' in label_str for v in range(36, 45)):
            return 'Fedora 36+'
        elif 'fedora' in label_str:
            return 'Fedora Other'

        # macOS: Merge Ventura, Sonoma, Sequoia (13, 14, 15)
        if any(f'macos {v}' in label_str for v in [13, 14, 15]):
            return 'macOS 13+'
        elif any(f'macos {v}' in label_str for v in [11, 12]):
            return 'macOS 11/12'
        elif 'macos' in label_str or 'darwin' in label_str:
            return 'macOS Other'

        # Android: Keep as-is (usually already grouped)
        if 'android' in label_str:
            return 'Android'

        # BSD variants: Group together
        if 'bsd' in label_str or 'freebsd' in label_str or 'openbsd' in label_str:
            return 'BSD'

        # Keep original if no rule matches
        return label

    df_merged[os_column] = df_merged[os_column].apply(merge_os_label)

    if verbose:
        print(f"\n  After merging:")
        for os_label, count in df_merged[os_column].value_counts().items():
            print(f"    {str(os_label):<30}: {count:>6,}")
        print(f"\n  Reduced from {df[os_column].nunique()} to {df_merged[os_column].nunique()} classes")

    return df_merged


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def select_features(df, verbose=True):
    """
    Select packet-level features for modern OS classification

    CESNET dataset has rich packet-level features
    """

    # TCP fingerprinting features
    tcp_features = [
        'tcp_window_size',
        'tcp_window_scale',
        'tcp_mss',
        'tcp_options_order',  # String - needs encoding
        'tcp_flags',
    ]

    # IP features
    ip_features = [
        'ttl',
        'initial_ttl',
        'df_flag',
        'ip_len',
    ]

    # Network
    network_features = [
        'src_port',
        'dst_port',
        'protocol',
    ]

    all_features = tcp_features + ip_features + network_features
    available_features = [f for f in all_features if f in df.columns]

    if verbose:
        print(f"\nFeature Selection:")
        print(f"  Available: {len(available_features)}")

    return available_features


def encode_tcp_options(df, column='tcp_options_order', max_patterns=15, verbose=True):
    """
    Encode TCP options order

    Limit to top patterns to avoid sparse features with small dataset
    """

    if column not in df.columns:
        if verbose:
            print(f"  Warning: {column} not found")
        return df, []

    # Get top patterns
    top_patterns = df[column].fillna('NONE').value_counts().head(max_patterns).index.tolist()

    if verbose:
        print(f"\n  TCP Options encoding:")
        print(f"    Unique patterns: {df[column].nunique()}")
        print(f"    Using top {len(top_patterns)} patterns")

    new_cols = []
    for pattern in top_patterns:
        col_name = f"tcp_opt_{pattern.replace(':', '_')[:30]}"
        df[col_name] = (df[column] == pattern).astype(int)
        new_cols.append(col_name)

    return df, new_cols


def handle_missing_values(X, strategy='median', verbose=True):
    """Handle missing values"""

    if verbose:
        missing_before = X.isnull().sum().sum()
        print(f"\nMissing Values: {missing_before:,}")

    if strategy == 'median':
        X_filled = X.fillna(X.median())
    elif strategy == 'mean':
        X_filled = X.fillna(X.mean())
    else:
        X_filled = X.fillna(0)

    X_filled = X_filled.fillna(0)

    if verbose:
        missing_after = X_filled.isnull().sum().sum()
        print(f"  After imputation: {missing_after:,}")

    return X_filled


# ============================================================================
# ADVANCED SAMPLING
# ============================================================================

def apply_adasyn(X_train, y_train, min_samples=50, verbose=True):
    """
    Apply ADASYN (Adaptive Synthetic Sampling)

    ADASYN is better than SMOTE for very imbalanced datasets
    """

    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))

    if verbose:
        print(f"\n  Class distribution before ADASYN:")
        for cls, count in class_dist.items():
            print(f"    Class {cls}: {count} samples")

    # Check if we need ADASYN
    if min(counts) >= min_samples:
        if verbose:
            print(f"\n  All classes have ≥{min_samples} samples, skipping ADASYN")
        return X_train, y_train

    # Sampling strategy: bring all classes to at least min_samples
    sampling_strategy = {
        cls: max(min_samples, count)
        for cls, count in class_dist.items()
    }

    try:
        # Use ADASYN for adaptive sampling
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=42,
            n_neighbors=min(5, min(counts) - 1) if min(counts) > 1 else 1
        )
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

        if verbose:
            print(f"\n  After ADASYN:")
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            for cls, count in zip(unique_new, counts_new):
                print(f"    Class {cls}: {count} samples")
            print(f"\n  Total: {len(X_train):,} → {len(X_resampled):,}")

        return X_resampled, y_resampled

    except Exception as e:
        if verbose:
            print(f"\n  ADASYN failed: {e}")
            print(f"  Trying SMOTE...")

        try:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42,
                k_neighbors=min(3, min(counts) - 1) if min(counts) > 1 else 1
            )
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            if verbose:
                print(f"  SMOTE successful!")
                print(f"  Total: {len(X_train):,} → {len(X_resampled):,}")

            return X_resampled, y_resampled

        except Exception as e2:
            if verbose:
                print(f"  SMOTE also failed: {e2}")
                print(f"  Continuing without resampling")
            return X_train, y_train


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_random_forest_model(X_train, y_train, X_val, y_val,
                              class_weights=None, verbose=True):
    """
    Train Random Forest classifier

    Random Forest chosen for Model 2b because:
    - More robust to overfitting with small data
    - No need for extensive hyperparameter tuning
    - Naturally handles class imbalance with class_weight
    """

    if verbose:
        print("\nTraining Random Forest Model...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Validation samples: {len(X_val):,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")

    # Calculate class weights
    if class_weights is None:
        unique, counts = np.unique(y_train, return_counts=True)
        max_count = counts.max()
        class_weights = {cls: max_count / count for cls, count in zip(unique, counts)}

        if verbose:
            print(f"\n  Class distribution:")
            for cls, count in zip(unique, counts):
                print(f"    Class {cls}: {count:,} (weight: {class_weights[cls]:.2f})")

    # Random Forest parameters (conservative for small dataset)
    params = {
        'n_estimators': 300,  # More trees for stability
        'max_depth': 15,  # Moderate depth to prevent overfitting
        'min_samples_split': 5,  # Require at least 5 samples to split
        'min_samples_leaf': 2,  # Require at least 2 samples per leaf
        'max_features': 'sqrt',  # Use sqrt of features for each split
        'bootstrap': True,
        'class_weight': class_weights,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }

    if verbose:
        print(f"\n  Random Forest hyperparameters:")
        for key in ['n_estimators', 'max_depth', 'min_samples_split', 'max_features']:
            print(f"    {key}: {params[key]}")

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Validation score
    val_score = model.score(X_val, y_val)

    if verbose:
        print(f"\n  Training complete!")
        print(f"  Validation accuracy: {val_score:.4f}")

    return model, class_weights


# ============================================================================
# CROSS-VALIDATION (critical for small datasets)
# ============================================================================

def perform_cross_validation(X, y, n_splits=5, verbose=True):
    """
    Perform stratified k-fold cross-validation

    Critical for small datasets to estimate true performance
    """

    if verbose:
        print(f"\nPerforming {n_splits}-Fold Cross-Validation...")

    # Use same parameters as main model
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    class_weights = {cls: max_count / count for cls, count in zip(unique, counts)}

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)

    if verbose:
        print(f"  Fold scores: {[f'{s:.4f}' for s in scores]}")
        print(f"  Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    return scores


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, label_encoder, output_dir, verbose=True):
    """Comprehensive evaluation"""

    if verbose:
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
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

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Model 2b: Modern OS Classification - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'model2b_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"\n  Confusion matrix saved: {cm_path}")
    plt.close()

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    if verbose:
        print(f"\nTop 15 Most Important Features:")
        for idx, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']:<35} {row['importance']:.4f}")

    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Model 2b: Top 20 Feature Importances')
    plt.tight_layout()

    fi_path = os.path.join(output_dir, 'model2b_feature_importance.png')
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"  Feature importance saved: {fi_path}")
    plt.close()

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'RandomForest',
        'task': 'Modern OS Version Classification',
        'dataset': 'CESNET Idle',
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

    results_path = os.path.join(output_dir, 'model2b_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"  Results saved: {results_path}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Model 2b: Modern OS Classifier (CESNET dataset)'
    )

    parser.add_argument('--input', type=str, default='data/processed/cesnet_idle_packets.csv')
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--merge-classes', action='store_true',
                       help='Merge similar OS versions to combat small dataset')
    parser.add_argument('--use-adasyn', action='store_true',
                       help='Apply ADASYN/SMOTE for minority classes')
    parser.add_argument('--adasyn-threshold', type=int, default=50,
                       help='Min samples before applying ADASYN (default: 50)')
    parser.add_argument('--cross-validate', action='store_true',
                       help='Perform k-fold cross-validation')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.1)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print("="*70)
        print("MODEL 2b: MODERN OS CLASSIFIER (CESNET DATASET)")
        print("="*70)
        print(f"\nInput:  {args.input}")
        print(f"Output: {args.output_dir}/model2b_modern_os.pkl")
        print(f"Algorithm: Random Forest")
        if args.merge_classes:
            print(f"Class merging: ENABLED")
        if args.use_adasyn:
            print(f"Balancing: ADASYN (threshold: {args.adasyn_threshold})")

    # Load data
    if verbose:
        print(f"\n[1/8] Loading data...")

    if not os.path.exists(args.input):
        print(f"\nERROR: File not found: {args.input}")
        print(f"Run: python scripts/preprocess_cesnet_idle.py")
        sys.exit(1)

    df = pd.read_csv(args.input)
    if verbose:
        print(f"  Loaded {len(df):,} records")

    if 'os_label' not in df.columns:
        print(f"\nERROR: 'os_label' column not found!")
        sys.exit(1)

    df = df[df['os_label'].notna()]
    if verbose:
        print(f"  With OS labels: {len(df):,}")

    # Merge classes if requested
    if args.merge_classes:
        if verbose:
            print(f"\n[2/8] Merging similar OS versions...")
        df = merge_similar_os_versions(df, verbose=verbose)
    else:
        if verbose:
            print(f"\n[2/8] Skipping class merging")
            print(f"  TIP: Use --merge-classes to improve performance on small dataset")

    # Feature engineering
    if verbose:
        print(f"\n[3/8] Feature engineering...")

    df, tcp_opt_cols = encode_tcp_options(df, max_patterns=15, verbose=verbose)
    feature_columns = select_features(df, verbose=verbose)
    feature_columns.extend(tcp_opt_cols)

    X = df[feature_columns].copy()
    y = df['os_label'].values

    X = handle_missing_values(X, verbose=verbose)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if verbose:
        print(f"\n  Encoded {len(label_encoder.classes_)} classes")

    # Cross-validation (recommended for small datasets)
    if args.cross_validate:
        if verbose:
            print(f"\n[4/8] Cross-validation...")
        cv_scores = perform_cross_validation(X, y_encoded, n_splits=5, verbose=verbose)
    else:
        if verbose:
            print(f"\n[4/8] Skipping cross-validation")

    # Split data
    if verbose:
        print(f"\n[5/8] Splitting data...")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_encoded, test_size=args.test_size, stratify=y_encoded, random_state=args.random_state
    )

    val_size_adjusted = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size_adjusted,
        stratify=y_train_full, random_state=args.random_state
    )

    if verbose:
        print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Apply ADASYN
    if args.use_adasyn:
        if verbose:
            print(f"\n[6/8] Applying ADASYN...")
        X_train, y_train = apply_adasyn(X_train, y_train, args.adasyn_threshold, verbose)
    else:
        if verbose:
            print(f"\n[6/8] Skipping ADASYN")

    # Train
    if verbose:
        print(f"\n[7/8] Training model...")

    model, class_weights = train_random_forest_model(X_train, y_train, X_val, y_val, verbose=verbose)

    # Evaluate
    if verbose:
        print(f"\n[8/8] Evaluating...")

    os.makedirs(args.results_dir, exist_ok=True)
    results = evaluate_model(model, X_test, y_test, label_encoder, args.results_dir, verbose)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'model2b_modern_os.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(args.output_dir, 'model2b_feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_columns, f)
    with open(os.path.join(args.output_dir, 'model2b_label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(args.output_dir, 'model2b_class_weights.pkl'), 'wb') as f:
        pickle.dump(class_weights, f)

    if verbose:
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"\nModel 2b Performance:")
        print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"  F1-Score: {results['metrics']['f1_score']:.4f}")
        print(f"\n⚠️  IMPORTANT: With only ~1.8k samples, monitor for overfitting!")
        print(f"   Recommendations:")
        print(f"   - Use --merge-classes to combine similar OSs")
        print(f"   - Use --use-adasyn for minority class balancing")
        print(f"   - Use --cross-validate to estimate true performance")


if __name__ == '__main__':
    main()
