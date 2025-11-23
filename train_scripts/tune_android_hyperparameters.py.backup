#!/usr/bin/env python3
"""
Hyperparameter Tuning for Android Expert Model

Uses RandomizedSearchCV to find optimal XGBoost hyperparameters for Android version classification.
Optimized for similar classes (Android 7/8/9/10 share Linux kernel TCP stack).

Usage:
    python train_scripts/tune_android_hyperparameters.py
    python train_scripts/tune_android_hyperparameters.py --n-iter 100
    python train_scripts/tune_android_hyperparameters.py --search-type grid
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


def select_android_features(df):
    """Select features for Android (same as train_android_expert.py)"""

    # Exclude constant/low-variance features
    features = [
        'tcp_syn_size', 'tcp_win_size', 'tcp_syn_ttl', 'tcp_flags_a',
        'tcp_option_window_scale_forward', 'tcp_option_maximum_segment_size_forward',
        'ip_tos', 'maximum_ttl_forward', 'ipv4_dont_fragment_forward',
        'src_port', 'packet_total_count_forward', 'packet_total_count_backward', 'total_bytes',
        'tls_ja3_fingerprint', 'tls_cipher_suites', 'tls_extension_types', 'tls_elliptic_curves',
        'tls_handshake_type', 'tls_client_key_length',
        'initial_ttl',
    ]

    available = [f for f in features if f in df.columns]
    return available


def load_and_prepare_data(data_path='data/processed/masaryk_android.csv', verbose=True):
    """Load and prepare Android dataset"""

    if verbose:
        print("="*70)
        print("LOADING ANDROID DATASET")
        print("="*70)

    df = pd.read_csv(data_path)

    if verbose:
        print(f"\nDataset shape: {df.shape}")
        print(f"  Records: {len(df):,}")

    if 'os_label' not in df.columns:
        raise ValueError("Dataset must contain 'os_label' column")

    # Select features
    feature_cols = select_android_features(df)

    if verbose:
        print(f"\nFeatures selected: {len(feature_cols)}")

    X = df[feature_cols].copy()
    y = df['os_label'].copy()

    if verbose:
        print(f"\nClass distribution:")
        print(y.value_counts())

    # Encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    if verbose and categorical_features:
        print(f"\nEncoding {len(categorical_features)} categorical features")

    for col in categorical_features:
        le = LabelEncoder()
        # Handle NaN
        X[col] = X[col].fillna('__MISSING__')
        X[col] = le.fit_transform(X[col].astype(str))

    # Impute NaN values (CRITICAL for ADASYN and tuning)
    if verbose:
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            print(f"\nImputing {nan_count} NaN values (median strategy)")

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


def get_android_param_distributions(search_type='random'):
    """
    Get hyperparameter search space for Android Expert

    Optimized for similar classes (Android versions share kernel TCP stack)
    """

    if search_type == 'random':
        # For RandomizedSearchCV - broader search for Android
        return {
            'n_estimators': randint(300, 1000),       # More trees for subtle patterns
            'max_depth': randint(8, 20),              # Deeper trees
            'learning_rate': uniform(0.01, 0.15),     # Lower learning rates
            'subsample': uniform(0.7, 0.3),           # 0.7 to 1.0
            'colsample_bytree': uniform(0.7, 0.3),    # 0.7 to 1.0
            'gamma': uniform(0, 2),                   # Less regularization
            'min_child_weight': randint(1, 8),        # Allow finer splits
            'reg_alpha': uniform(0, 0.5),             # L1 regularization
            'reg_lambda': uniform(0.5, 5),            # L2 regularization
        }
    else:
        # For GridSearchCV - focused grid
        return {
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


def tune_hyperparameters(X, y, search_type='random', n_iter=50, cv=5,
                         n_jobs=-1, random_state=42, verbose=True):
    """Perform hyperparameter tuning"""

    if verbose:
        print("\n" + "="*70)
        print(f"ANDROID HYPERPARAMETER TUNING - {search_type.upper()} SEARCH")
        print("="*70)

    param_dist = get_android_param_distributions(search_type)

    if verbose:
        print(f"\nSearch space:")
        for param, values in list(param_dist.items())[:5]:
            print(f"  {param}: {values}")
        print(f"  ... and {len(param_dist)-5} more parameters")

    # Base model
    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        tree_method='hist',
        random_state=random_state,
        eval_metric='mlogloss'
    )

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if verbose:
        print(f"\nCross-validation: {cv}-fold StratifiedKFold")

    # Create search
    if search_type == 'random':
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring='f1_weighted',
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=2,
            return_train_score=True
        )
        if verbose:
            print(f"RandomizedSearchCV: Testing {n_iter} parameter combinations")
    else:
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_dist,
            cv=cv_strategy,
            scoring='f1_weighted',
            n_jobs=n_jobs,
            verbose=2,
            return_train_score=True
        )
        total_combinations = 1
        for values in param_dist.values():
            total_combinations *= len(values)
        if verbose:
            print(f"GridSearchCV: Testing all {total_combinations:,} parameter combinations")

    if verbose:
        print(f"\nStarting search at {datetime.now().strftime('%H:%M:%S')}...")
        print(f"This may take a while...")

    search.fit(X, y)

    if verbose:
        print(f"Search completed at {datetime.now().strftime('%H:%M:%S')}")

    return search


def save_results(search, output_dir, verbose=True):
    """Save tuning results"""

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"AndroidTuning_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save best parameters
    best_params_path = os.path.join(run_dir, 'best_hyperparameters.json')
    with open(best_params_path, 'w') as f:
        json.dump(search.best_params_, f, indent=2)

    if verbose:
        print("\n" + "="*70)
        print("BEST HYPERPARAMETERS")
        print("="*70)
        print(json.dumps(search.best_params_, indent=2))
        print(f"\nSaved to: {best_params_path}")

    # Save CV results
    cv_results_df = pd.DataFrame(search.cv_results_)
    cv_results_path = os.path.join(run_dir, 'cv_results.csv')
    cv_results_df.to_csv(cv_results_path, index=False)

    if verbose:
        print(f"\n" + "="*70)
        print("CROSS-VALIDATION RESULTS")
        print("="*70)
        print(f"\nBest CV Score (F1-weighted): {search.best_score_:.4f}")
        print(f"\nTop 5 parameter combinations:")
        top_5 = cv_results_df.nlargest(5, 'mean_test_score')[
            ['mean_test_score', 'std_test_score', 'params']
        ]
        for idx, row in top_5.iterrows():
            print(f"\n  Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
            print(f"  Params: {row['params']}")

        print(f"\nFull CV results saved to: {cv_results_path}")

    # Save summary
    summary = {
        'best_score': float(search.best_score_),
        'best_params': search.best_params_,
        'n_splits': search.n_splits_,
        'total_fits': len(cv_results_df),
        'tuning_date': datetime.now().isoformat(),
    }

    summary_path = os.path.join(run_dir, 'tuning_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nSummary saved to: {summary_path}")

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for Android Expert Model'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/masaryk_android.csv',
        help='Path to Android CSV file'
    )

    parser.add_argument(
        '--search-type',
        type=str,
        default='random',
        choices=['random', 'grid'],
        help='Search type: random (faster) or grid (exhaustive)'
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
        help='Output directory for results'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 = all cores)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Load data
    X, y, target_encoder = load_and_prepare_data(args.data, verbose=verbose)

    # Tune
    search = tune_hyperparameters(
        X, y,
        search_type=args.search_type,
        n_iter=args.n_iter,
        cv=args.cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=verbose
    )

    # Save
    run_dir = save_results(search, args.output_dir, verbose=verbose)

    if verbose:
        print("\n✓ Hyperparameter tuning complete!")
        print(f"\nTo use these hyperparameters:")
        print(f"  1. Copy values from: {run_dir}/best_hyperparameters.json")
        print(f"  2. Update train_scripts/train_android_expert.py with these values")
        print(f"\nExpected improvement: 74% → 77-80% accuracy")


if __name__ == '__main__':
    main()
