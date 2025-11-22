#!/usr/bin/env python3
"""
Hyperparameter Tuning for OS Family Classification

Uses RandomizedSearchCV or GridSearchCV to find optimal XGBoost hyperparameters.
Saves best parameters to JSON for use in training script.

Usage:
    python tune_hyperparameters.py --data data/processed/masaryk_processed.csv --search-type random --n-iter 50
    python tune_hyperparameters.py --data data/processed/masaryk_processed.csv --search-type grid
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
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.stats import randint, uniform


def load_and_prepare_data(data_path, use_adasyn=False, random_state=42, verbose=True):
    """Load data and prepare for training"""

    if verbose:
        print("="*70)
        print("LOADING DATA")
        print("="*70)

    # Load dataset
    df = pd.read_csv(data_path)

    if verbose:
        print(f"\nDataset shape: {df.shape}")
        print(f"  Records: {len(df):,}")
        print(f"  Features: {len(df.columns)}")

    # Separate features and target
    if 'os_family' not in df.columns:
        raise ValueError("Dataset must contain 'os_family' column")

    X = df.drop(columns=['os_family'])
    y = df['os_family']

    if verbose:
        print(f"\nClass distribution:")
        print(y.value_counts())
        print(f"\nClass balance:")
        print((y.value_counts() / len(y) * 100).round(2))

    # Encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    if verbose and categorical_features:
        print(f"\nEncoding {len(categorical_features)} categorical features:")
        for feat in categorical_features:
            print(f"  - {feat}")

    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        # Handle NaN values
        mask = X[col].notna()
        X.loc[mask, col] = le.fit_transform(X.loc[mask, col].astype(str))
        X[col] = X[col].fillna(-1).astype(int)
        label_encoders[col] = le

    # Fill remaining NaN with -1
    X = X.fillna(-1)

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    if verbose:
        print(f"\nEncoded classes: {list(target_encoder.classes_)}")

    # Apply ADASYN if requested
    if use_adasyn:
        try:
            from imblearn.over_sampling import ADASYN

            if verbose:
                print(f"\nApplying ADASYN oversampling...")
                print(f"  Before: {len(X):,} samples")

            adasyn = ADASYN(random_state=random_state, sampling_strategy='minority')
            X, y_encoded = adasyn.fit_resample(X, y_encoded)

            if verbose:
                print(f"  After: {len(X):,} samples")
                unique, counts = np.unique(y_encoded, return_counts=True)
                print(f"\n  Class distribution after ADASYN:")
                for cls, count in zip(unique, counts):
                    cls_name = target_encoder.inverse_transform([cls])[0]
                    print(f"    {cls_name:<15}: {count:>6,}")
        except ImportError:
            print("WARNING: ADASYN requested but imbalanced-learn not installed. Skipping.")

    return X, y_encoded, target_encoder, label_encoders


def get_param_distributions(search_type='random'):
    """
    Get hyperparameter search space for XGBoost

    Args:
        search_type: 'random' for RandomizedSearchCV or 'grid' for GridSearchCV

    Returns:
        Parameter distribution dict
    """

    if search_type == 'random':
        # For RandomizedSearchCV - use scipy distributions
        return {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.3
            'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
            'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
            'gamma': uniform(0, 5),
            'min_child_weight': randint(1, 10),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(1, 10),
        }
    else:
        # For GridSearchCV - use discrete values
        return {
            'n_estimators': [100, 200, 500, 800],
            'max_depth': [3, 5, 7, 10, 12],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 1, 3, 5],
            'min_child_weight': [1, 3, 5, 7],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [1, 2, 5, 10],
        }


def tune_hyperparameters(X, y, search_type='random', n_iter=50, cv=5,
                         n_jobs=-1, random_state=42, verbose=True):
    """
    Perform hyperparameter tuning using RandomizedSearchCV or GridSearchCV

    Args:
        X: Features
        y: Target (encoded)
        search_type: 'random' or 'grid'
        n_iter: Number of iterations for RandomizedSearchCV
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs (-1 = all cores)
        random_state: Random seed
        verbose: Print progress

    Returns:
        Best estimator and search results
    """

    if verbose:
        print("\n" + "="*70)
        print(f"HYPERPARAMETER TUNING - {search_type.upper()} SEARCH")
        print("="*70)

    # Get parameter distributions
    param_dist = get_param_distributions(search_type)

    if verbose:
        print(f"\nSearch space:")
        for param, values in param_dist.items():
            print(f"  {param}: {values}")

    # Base model
    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        tree_method='hist',
        random_state=random_state,
        early_stopping_rounds=20,
        eval_metric='mlogloss'
    )

    # Cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if verbose:
        print(f"\nCross-validation: {cv}-fold StratifiedKFold")

    # Create search object
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
        # Calculate total combinations
        total_combinations = 1
        for values in param_dist.values():
            total_combinations *= len(values)
        if verbose:
            print(f"GridSearchCV: Testing all {total_combinations:,} parameter combinations")

    # Fit search
    if verbose:
        print(f"\nStarting search at {datetime.now().strftime('%H:%M:%S')}...")

    # Create a validation set for early stopping
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    search.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    if verbose:
        print(f"Search completed at {datetime.now().strftime('%H:%M:%S')}")

    return search


def save_results(search, output_dir, target_encoder, verbose=True):
    """Save tuning results and best parameters"""

    os.makedirs(output_dir, exist_ok=True)

    # Save best parameters
    best_params_path = os.path.join(output_dir, 'best_hyperparameters.json')
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
    cv_results_path = os.path.join(output_dir, 'cv_results.csv')
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

    summary_path = os.path.join(output_dir, 'tuning_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for OS family classification'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to processed CSV file'
    )

    parser.add_argument(
        '--search-type',
        type=str,
        default='random',
        choices=['random', 'grid'],
        help='Search type: random (RandomizedSearchCV) or grid (GridSearchCV)'
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
        '--use-adasyn',
        action='store_true',
        help='Use ADASYN oversampling before tuning'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/hyperparameter_tuning',
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
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Load and prepare data
    X, y, target_encoder, label_encoders = load_and_prepare_data(
        args.data,
        use_adasyn=args.use_adasyn,
        random_state=args.random_state,
        verbose=verbose
    )

    # Tune hyperparameters
    search = tune_hyperparameters(
        X, y,
        search_type=args.search_type,
        n_iter=args.n_iter,
        cv=args.cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=verbose
    )

    # Save results
    save_results(
        search,
        args.output_dir,
        target_encoder,
        verbose=verbose
    )

    if verbose:
        print("\n✓ Hyperparameter tuning complete!")
        print(f"\nTo use these hyperparameters in training, copy them from:")
        print(f"  {os.path.join(args.output_dir, 'best_hyperparameters.json')}")
        print(f"\nOr modify train_model1_family.py to load from this file automatically.")


if __name__ == '__main__':
    main()
