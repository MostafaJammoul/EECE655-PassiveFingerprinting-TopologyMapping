#!/usr/bin/env python3
"""
Train Model 2b: Modern OS Version Classifier (FIXED VERSION)

CRITICAL FIXES:
1. REMOVED port numbers (application-specific, not OS-specific)
2. IP ID converted to behavioral features (sequential vs random)
3. Added port category features (well-known, ephemeral, etc.)
4. Increased regularization to prevent single-feature dominance

Dataset: CESNET Idle (TCP SYN packet-level features)
Task: Classify modern OS versions (Win10/11, Ubuntu 22/24, Fedora, macOS 13+, Android)
Algorithm: Random Forest (more robust for small datasets)

Input:  data/processed/cesnet_idle_packets.csv
Output: models/model2b_modern_os_fixed.pkl
        results/model2b_evaluation_fixed.json
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
# IMPROVED FEATURE ENGINEERING (FIXED)
# ============================================================================

def extract_ip_id_behavioral_features(df, verbose=True):
    """
    Extract behavioral features from IP ID instead of raw values

    CRITICAL LIMITATION: CESNET dataset only has SINGLE SYN packets per connection!
    We CANNOT determine if IP IDs are sequential or random from a single packet.

    SOLUTION: For single-packet datasets, we REMOVE IP ID features entirely.
    IP ID requires multiple packets from the same host to analyze incrementing behavior.

    If you want to use IP ID:
    1. Reprocess dataset with --all-tcp flag
    2. Group packets by source IP
    3. Calculate IP ID increment statistics per host
    """

    if 'ip_id' not in df.columns:
        if verbose:
            print("  ℹ️  ip_id column not found, skipping IP ID features")
        return df

    if verbose:
        print("\n  ⚠️  IP ID REMOVED (single-packet limitation)")
        print("     - Cannot determine sequential/random from single packets")
        print("     - Requires multiple packets per host for behavioral analysis")
        print("     - Use --all-tcp flag during preprocessing to enable IP ID features")

    # DO NOT create any IP ID features for single-packet datasets
    # This would just create noise and potential overfitting

    return df


def categorize_ports(df, verbose=True):
    """
    Convert port numbers to categorical features

    CRITICAL FIX: Instead of using raw port numbers (application-specific),
    we categorize them into:
    - Well-known ports (< 1024)
    - Registered ports (1024-49151)
    - Ephemeral/dynamic ports (49152-65535)
    - Common service types (HTTP, HTTPS, SSH, etc.)

    This reduces overfitting while preserving some network context
    """

    if verbose:
        print("\n  Categorizing port numbers...")

    for port_col in ['src_port', 'dst_port']:
        if port_col not in df.columns:
            continue

        # Port range categories
        df[f'{port_col}_is_well_known'] = (df[port_col] < 1024).astype(int)
        df[f'{port_col}_is_ephemeral'] = (df[port_col] >= 49152).astype(int)
        df[f'{port_col}_is_registered'] = ((df[port_col] >= 1024) & (df[port_col] < 49152)).astype(int)

        # Common services (these might have OS-specific implementations)
        common_ports = {
            80: 'http',
            443: 'https',
            22: 'ssh',
            21: 'ftp',
            25: 'smtp',
            53: 'dns',
            3389: 'rdp',
            445: 'smb'
        }

        for port_num, service in common_ports.items():
            df[f'{port_col}_is_{service}'] = (df[port_col] == port_num).astype(int)

    if verbose:
        print(f"    Created port category features")
        print(f"    - Port range indicators (well-known, ephemeral, registered)")
        print(f"    - Common service flags (HTTP, HTTPS, SSH, etc.)")

    return df


def select_features_fixed(df, verbose=True):
    """
    Select packet-level features for modern OS classification (FIXED VERSION)

    CRITICAL CHANGES:
    1. REMOVED: src_port, dst_port (application-specific)
    2. REMOVED: ip_id (raw values - replaced with behavioral features)
    3. ADDED: Port category features
    4. ADDED: IP ID behavioral features
    """

    # TCP fingerprinting features (core OS identifiers)
    tcp_features = [
        'tcp_window_size',  # CRITICAL: OS-specific default window sizes
        'tcp_window_scale',  # CRITICAL: Window scale option
        'tcp_mss',  # CRITICAL: Maximum Segment Size
        'tcp_options_order',  # CRITICAL: String - needs encoding
        'tcp_flags',
        'tcp_timestamp_val',  # CRITICAL! Timestamp value
        'tcp_timestamp_ecr',  # CRITICAL! Timestamp echo reply
        'tcp_sack_permitted',  # MEDIUM importance
        'tcp_urgent_ptr',  # LOW importance
    ]

    # IP features (FIXED - removed raw ip_id)
    ip_features = [
        'ttl',  # CRITICAL: OS-specific TTL values
        'initial_ttl',  # Estimated initial TTL
        'df_flag',  # Don't Fragment
        'ip_len',  # IP packet length
        'ip_tos',  # Type of Service / DSCP
        # REMOVED: 'ip_id' - replaced with behavioral features below
    ]

    # IP ID behavioral features - REMOVED for single-packet datasets
    # (Cannot determine sequential/random behavior from single packets)
    ip_id_behavioral = [
        # Removed: requires multiple packets per host
    ]

    # Port category features (ADDED - replace raw ports)
    port_features = [
        'src_port_is_well_known',
        'src_port_is_ephemeral',
        'src_port_is_registered',
        'dst_port_is_well_known',
        'dst_port_is_ephemeral',
        'dst_port_is_registered',
        # Common services
        'src_port_is_http', 'src_port_is_https', 'src_port_is_ssh',
        'dst_port_is_http', 'dst_port_is_https', 'dst_port_is_ssh',
        'dst_port_is_rdp', 'dst_port_is_smb',
    ]

    # Protocol (keep this - it's legitimate)
    network_features = [
        'protocol',
    ]

    all_features = tcp_features + ip_features + ip_id_behavioral + port_features + network_features
    available_features = [f for f in all_features if f in df.columns]

    if verbose:
        print(f"\nFeature Selection (FIXED):")
        print(f"  Available: {len(available_features)}")
        print(f"  ✓ REMOVED: src_port, dst_port (application-specific)")
        print(f"  ✓ REMOVED: ip_id (cannot extract behavior from single packets)")
        print(f"  ✓ ADDED: {len([f for f in port_features if f in df.columns])} port category features")
        print(f"\n  Focus: TCP fingerprinting (window, MSS, options, TTL)")

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

    # Only apply median/mean to numeric columns
    if strategy == 'median':
        X_filled = X.fillna(X.select_dtypes(include=['number']).median())
    elif strategy == 'mean':
        X_filled = X.fillna(X.select_dtypes(include=['number']).mean())
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
# MODEL TRAINING (IMPROVED)
# ============================================================================

def train_random_forest_model(X_train, y_train, X_val, y_val,
                              class_weights=None, verbose=True):
    """
    Train Random Forest classifier with improved regularization

    CRITICAL FIXES:
    1. Increased min_samples_split and min_samples_leaf to prevent overfitting
    2. Reduced max_features to 'log2' to prevent single-feature dominance
    3. Added max_leaf_nodes constraint
    """

    if verbose:
        print("\nTraining Random Forest Model (IMPROVED)...")
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

    # Random Forest parameters (IMPROVED with stronger regularization)
    params = {
        'n_estimators': 300,  # More trees for stability
        'max_depth': 12,  # REDUCED from 15 to prevent overfitting
        'min_samples_split': 10,  # INCREASED from 5 to require more samples
        'min_samples_leaf': 5,  # INCREASED from 2 to require more samples per leaf
        'max_features': 'log2',  # CHANGED from 'sqrt' to further reduce feature dominance
        'max_leaf_nodes': 100,  # NEW: Limit tree complexity
        'bootstrap': True,
        'class_weight': class_weights,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }

    if verbose:
        print(f"\n  Random Forest hyperparameters (IMPROVED):")
        for key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'max_leaf_nodes']:
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
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='log2',
        max_leaf_nodes=100,
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
    plt.title('Model 2b (FIXED): Modern OS Classification - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'model2b_confusion_matrix_fixed.png')
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
    plt.title('Model 2b (FIXED): Top 20 Feature Importances')
    plt.tight_layout()

    fi_path = os.path.join(output_dir, 'model2b_feature_importance_fixed.png')
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
        'version': 'FIXED - removed port numbers and raw IP ID',
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

    results_path = os.path.join(output_dir, 'model2b_evaluation_fixed.json')
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
        description='Train Model 2b (FIXED): Modern OS Classifier - removes port/IP ID overfitting'
    )

    parser.add_argument('--input', type=str, default='datasets/cesnet_merged.csv')
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
        print("MODEL 2b (FIXED): MODERN OS CLASSIFIER")
        print("="*70)
        print(f"\n🔧 CRITICAL FIXES APPLIED:")
        print(f"  ✓ Removed src_port, dst_port (application-specific)")
        print(f"  ✓ Removed ip_id (single-packet limitation)")
        print(f"  ✓ Added port category features (well-known, ephemeral, etc.)")
        print(f"  ✓ Increased regularization (max_features='log2', stricter splits)")
        print(f"  ✓ Focus on true OS fingerprints (TCP options, window, TTL)")
        print(f"\nInput:  {args.input}")
        print(f"Output: {args.output_dir}/model2b_modern_os_fixed.pkl")
        print(f"Algorithm: Random Forest (regularized)")
        if args.merge_classes:
            print(f"Class merging: ENABLED")
        if args.use_adasyn:
            print(f"Balancing: ADASYN (threshold: {args.adasyn_threshold})")

    # Load data
    if verbose:
        print(f"\n[1/9] Loading data...")

    if not os.path.exists(args.input):
        print(f"\nERROR: File not found: {args.input}")
        print(f"Run: python preprocess_scripts/cesnet_preprocess.py")
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
            print(f"\n[2/9] Merging similar OS versions...")
        df = merge_similar_os_versions(df, verbose=verbose)
    else:
        if verbose:
            print(f"\n[2/9] Skipping class merging")

    # CRITICAL FIX: Extract behavioral features from IP ID
    if verbose:
        print(f"\n[3/9] Extracting IP ID behavioral features...")
    df = extract_ip_id_behavioral_features(df, verbose=verbose)

    # CRITICAL FIX: Categorize ports
    if verbose:
        print(f"\n[4/9] Categorizing port numbers...")
    df = categorize_ports(df, verbose=verbose)

    # Feature engineering
    if verbose:
        print(f"\n[5/9] Feature engineering...")

    df, tcp_opt_cols = encode_tcp_options(df, max_patterns=15, verbose=verbose)
    feature_columns = select_features_fixed(df, verbose=verbose)

    # Remove string column (tcp_options_order) - it's already encoded as binary features
    if 'tcp_options_order' in feature_columns:
        feature_columns.remove('tcp_options_order')

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
            print(f"\n[6/9] Cross-validation...")
        cv_scores = perform_cross_validation(X, y_encoded, n_splits=5, verbose=verbose)
    else:
        if verbose:
            print(f"\n[6/9] Skipping cross-validation")

    # Split data
    if verbose:
        print(f"\n[7/9] Splitting data...")

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
            print(f"\n[8/9] Applying ADASYN...")
        X_train, y_train = apply_adasyn(X_train, y_train, args.adasyn_threshold, verbose)
    else:
        if verbose:
            print(f"\n[8/9] Skipping ADASYN")

    # Train
    if verbose:
        print(f"\n[9/9] Training model...")

    model, class_weights = train_random_forest_model(X_train, y_train, X_val, y_val, verbose=verbose)

    # Evaluate
    if verbose:
        print(f"\n[10/10] Evaluating...")

    os.makedirs(args.results_dir, exist_ok=True)
    results = evaluate_model(model, X_test, y_test, label_encoder, args.results_dir, verbose)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'model2b_modern_os_fixed.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(args.output_dir, 'model2b_feature_names_fixed.pkl'), 'wb') as f:
        pickle.dump(feature_columns, f)
    with open(os.path.join(args.output_dir, 'model2b_label_encoder_fixed.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(args.output_dir, 'model2b_class_weights_fixed.pkl'), 'wb') as f:
        pickle.dump(class_weights, f)

    if verbose:
        print("\n" + "="*70)
        print("TRAINING COMPLETE (FIXED VERSION)")
        print("="*70)
        print(f"\nModel 2b (FIXED) Performance:")
        print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"  F1-Score: {results['metrics']['f1_score']:.4f}")
        print(f"\n✓ Applied fixes:")
        print(f"  - Removed port number overfitting (application-specific)")
        print(f"  - Removed IP ID (single-packet limitation)")
        print(f"  - Increased model regularization")
        print(f"  - Focus on legitimate OS fingerprints")
        print(f"\n💡 Compare with original model to verify improvement!")
        print(f"\n📊 Expected: TCP options and window size should be top features")


if __name__ == '__main__':
    main()
