#!/usr/bin/env python3
"""
Adaptive Hyperparameter Tuning for Expert Models

Automatically tunes hyperparameters for Linux, Windows, or Android Expert models.
Uses RandomizedSearchCV or GridSearchCV to find optimal XGBoost parameters.

Usage:
    # Auto-detect from input file:
    python train_scripts/tune_hyperparameters.py --input data/processed/linux.csv
    python train_scripts/tune_hyperparameters.py --input data/processed/windows.csv
    python train_scripts/tune_hyperparameters.py --input data/processed/masaryk_android.csv

    # Specify model type explicitly:
    python train_scripts/tune_hyperparameters.py --model-type linux --input data/processed/linux.csv

    # Customize search:
    python train_scripts/tune_hyperparameters.py --input data/processed/linux.csv --n-iter 100
    python train_scripts/tune_hyperparameters.py --input data/processed/linux.csv --search-type grid
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform


def detect_model_type(input_path):
    """Auto-detect model type from filename"""
    filename = os.path.basename(input_path).lower()

    if 'linux' in filename:
        return 'linux'
    elif 'windows' in filename:
        return 'windows'
    elif 'android' in filename:
        return 'android'
    else:
        # Default to checking file contents
        return None


def select_features_for_model(df, model_type):
    """
    Select features based on model type

    All models use the same 25-feature Masaryk structure.
    Android removes some constant/low-variance features.
    """

    if model_type == 'android':
        # Android: Remove constant/low-variance features
        features = [
            'tcp_syn_size', 'tcp_win_size', 'tcp_syn_ttl', 'tcp_flags_a',
            'tcp_option_window_scale_forward', 'tcp_option_maximum_segment_size_forward',
            'ip_tos', 'maximum_ttl_forward', 'ipv4_dont_fragment_forward',
            'src_port', 'packet_total_count_forward', 'packet_total_count_backward', 'total_bytes',
            'tls_ja3_fingerprint', 'tls_cipher_suites', 'tls_extension_types', 'tls_elliptic_curves',
            'tls_handshake_type', 'tls_client_key_length',
            'initial_ttl',
        ]
    else:
        # Linux/Windows: Use all 25 features
        features = [
            # TCP
            'tcp_syn_size', 'tcp_win_size', 'tcp_syn_ttl', 'tcp_flags_a', 'syn_ack_flag',
            'tcp_option_window_scale_forward', 'tcp_option_selective_ack_permitted_forward',
            'tcp_option_maximum_segment_size_forward', 'tcp_option_no_operation_forward',
            # IP
            'l3_proto', 'ip_tos', 'maximum_ttl_forward', 'ipv4_dont_fragment_forward',
            # Flow
            'src_port', 'packet_total_count_forward', 'packet_total_count_backward', 'total_bytes',
            # TLS
            'tls_ja3_fingerprint', 'tls_cipher_suites', 'tls_extension_types', 'tls_elliptic_curves',
            'tls_client_version', 'tls_handshake_type', 'tls_client_key_length',
            # Derived
            'initial_ttl',
        ]

    available = [f for f in features if f in df.columns]
    return available


def load_and_prepare_data(data_path, model_type, verbose=True):
    """Load and prepare dataset for tuning"""

    if verbose:
        print("="*80)
        print(f"LOADING {model_type.upper()} DATASET")
        print("="*80)

    if not os.path.exists(data_path):
        print(f"\n✗ ERROR: File not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)

    if verbose:
        print(f"\nDataset shape: {df.shape}")
        print(f"  Records: {len(df):,}")

    if 'os_label' not in df.columns:
        print(f"\n✗ ERROR: Dataset must contain 'os_label' column")
        sys.exit(1)

    # Select features
    feature_cols = select_features_for_model(df, model_type)

    if verbose:
        print(f"\nFeatures selected: {len(feature_cols)}")

    X = df[feature_cols].copy()
    y = df['os_label'].copy()

    if verbose:
        print(f"\nClass distribution:")
        for label, count in y.value_counts().items():
            pct = (count / len(y)) * 100
            print(f"  {label:<30}: {count:>6,} ({pct:>5.1f}%)")

    # Encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    if verbose and categorical_features:
        print(f"\nEncoding {len(categorical_features)} categorical features...")

    for col in categorical_features:
        le = LabelEncoder()
        X[col] = X[col].fillna('__MISSING__')
        X[col] = le.fit_transform(X[col].astype(str))

    # Impute NaN values
    if verbose:
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            print(f"\nImputing {nan_count} NaN values (median strategy)...")

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    if verbose:
        print(f"\nEncoded classes: {list(target_encoder.classes_)}")

    return X, y_encoded, target_encoder


def get_param_distributions(search_type='random', model_type='android'):
    """
    Get hyperparameter search space

    Optimized for each model type:
    - Android: Similar classes (shared Linux kernel) - needs deeper trees, more regularization
    - Windows: Different TCP stacks - standard search space
    - Linux: Varies by distribution - standard to broader search space
    """

    if search_type == 'random':
        # Randomized search - broader parameter space
        params = {
            'n_estimators': randint(300, 1000),
            'max_depth': randint(8, 20),
            'learning_rate': uniform(0.01, 0.15),
            'subsample': uniform(0.7, 0.3),           # 0.7 to 1.0
            'colsample_bytree': uniform(0.7, 0.3),    # 0.7 to 1.0
            'gamma': uniform(0, 2),
            'min_child_weight': randint(1, 8),
            'reg_alpha': uniform(0, 0.5),
            'reg_lambda': uniform(0.5, 5),
        }

        # Android needs more regularization for similar classes
        if model_type == 'android':
            params['max_depth'] = randint(12, 22)         # Deeper trees
            params['gamma'] = uniform(0.1, 2)             # More pruning
            params['reg_lambda'] = uniform(1.0, 5)        # Stronger L2

    else:
        # Grid search - focused parameter grid
        params = {
            'n_estimators': [300, 500, 700, 1000],
            'max_depth': [8, 10, 12, 15, 18],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.5, 1.0, 1.5],
            'min_child_weight': [1, 2, 3, 5],
            'reg_alpha': [0, 0.1, 0.3, 0.5],
            'reg_lambda': [0.5, 1.0, 2.0, 3.0],
        }

        # Android-specific adjustments
        if model_type == 'android':
            params['max_depth'] = [10, 12, 15, 18, 20]
            params['gamma'] = [0.1, 0.5, 1.0, 1.5, 2.0]

    return params


def tune_hyperparameters(X, y, model_type='android', search_type='random',
                         n_iter=50, cv=5, n_jobs=-1, random_state=42, verbose=True):
    """
    Perform hyperparameter tuning using RandomizedSearchCV or GridSearchCV
    """

    if verbose:
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING")
        print("="*80)
        print(f"\nModel Type: {model_type.upper()}")
        print(f"Search Type: {search_type.upper()}")
        print(f"Search Iterations: {n_iter if search_type == 'random' else 'Full Grid'}")
        print(f"Cross-Validation Folds: {cv}")
        print(f"Scoring Metric: F1-weighted")

    # Get parameter distributions
    param_dist = get_param_distributions(search_type, model_type)

    # Base model
    base_model = xgb.XGBClassifier(
        random_state=random_state,
        n_jobs=n_jobs,
        objective='multi:softmax',
        num_class=len(np.unique(y))
    )

    # Setup search
    if search_type == 'random':
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='f1_weighted',
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
            verbose=1 if verbose else 0,
            random_state=random_state,
            n_jobs=n_jobs,
            return_train_score=False
        )
    else:  # grid
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_dist,
            scoring='f1_weighted',
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
            verbose=1 if verbose else 0,
            n_jobs=n_jobs,
            return_train_score=False
        )

    if verbose:
        print(f"\nStarting search...")
        print("-" * 80)

    # Perform search
    search.fit(X, y)

    if verbose:
        print("\n" + "="*80)
        print("TUNING RESULTS")
        print("="*80)
        print(f"\nBest CV Score (F1-weighted): {search.best_score_:.4f}")
        print(f"\nBest Hyperparameters:")
        print("-" * 80)
        for param, value in sorted(search.best_params_.items()):
            print(f"  {param:<20}: {value}")

    return search.best_params_, search.best_score_, search


def main():
    parser = argparse.ArgumentParser(
        description='Adaptive Hyperparameter Tuning for Expert Models (Linux/Windows/Android)'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to dataset CSV file (linux.csv, windows.csv, or masaryk_android.csv)'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['linux', 'windows', 'android'],
        help='Model type (auto-detected from filename if not specified)'
    )

    parser.add_argument(
        '--search-type',
        type=str,
        choices=['random', 'grid'],
        default='random',
        help='Search strategy: random (RandomizedSearchCV) or grid (GridSearchCV)'
    )

    parser.add_argument(
        '--n-iter',
        type=int,
        default=50,
        help='Number of iterations for RandomizedSearchCV (default: 50)'
    )

    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save tuning results'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 = use all cores)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Auto-detect model type if not specified
    if args.model_type is None:
        args.model_type = detect_model_type(args.input)
        if args.model_type is None:
            print("\n✗ ERROR: Could not auto-detect model type from filename.")
            print("  Please specify --model-type (linux/windows/android)")
            sys.exit(1)

    if not args.quiet:
        print(f"\nAuto-detected model type: {args.model_type.upper()}")

    # Load and prepare data
    X, y, label_encoder = load_and_prepare_data(
        args.input,
        args.model_type,
        verbose=not args.quiet
    )

    # Tune hyperparameters
    best_params, best_score, search = tune_hyperparameters(
        X, y,
        model_type=args.model_type,
        search_type=args.search_type,
        n_iter=args.n_iter,
        cv=args.cv,
        n_jobs=args.n_jobs,
        random_state=42,
        verbose=not args.quiet
    )

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_subdir = f"{args.model_type.capitalize()}Tuning_{timestamp}"
    output_dir = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Save best hyperparameters
    results = {
        'model_type': args.model_type,
        'search_type': args.search_type,
        'best_cv_score': float(best_score),
        'best_params': best_params,
        'classes': list(label_encoder.classes_),
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'cv_folds': args.cv,
        'n_iterations': args.n_iter if args.search_type == 'random' else 'full_grid',
        'timestamp': timestamp
    }

    results_file = os.path.join(output_dir, 'best_hyperparameters.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    if not args.quiet:
        print(f"\n" + "="*80)
        print("RESULTS SAVED")
        print("="*80)
        print(f"\nBest hyperparameters saved to:")
        print(f"  {results_file}")
        print(f"\nTo use these hyperparameters, update train_{args.model_type}_expert.py")
        print(f"with the values from best_hyperparameters.json")

        print(f"\n" + "="*80)
        print(f"EXPECTED IMPROVEMENT")
        print(f"="*80)
        print(f"\nCurrent baseline (default params): ~70-75%")
        print(f"Expected with tuned params: ~{best_score*100:.1f}%")
        print(f"Improvement: ~{(best_score - 0.70)*100:+.1f}%")

    return best_params, best_score


if __name__ == "__main__":
    best_params, best_score = main()
