#!/usr/bin/env python3
"""
Train Android Expert Ensemble - Multiple models with voting

Trains 3 models with different balancing strategies and combines predictions
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main training script
from train_scripts.train_android_expert import (
    select_features, encode_categorical_features, apply_smote
)

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
except ImportError as e:
    print(f"ERROR: Missing required library: {e}")
    sys.exit(1)


def train_single_model(X_train, y_train, class_weights, strategy_name, balance_strategy, use_borderline):
    """Train a single XGBoost model with given strategy"""

    print(f"\n{'='*80}")
    print(f"Training Model {strategy_name}")
    print(f"{'='*80}")

    # Apply SMOTE with specified strategy
    X_resampled, y_resampled = apply_smote(
        X_train, y_train,
        verbose=True,
        balance_strategy=balance_strategy,
        use_borderline=use_borderline
    )

    # Train XGBoost with tuned hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=708,
        max_depth=19,
        learning_rate=0.070,
        min_child_weight=3,
        gamma=0.475,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.434,
        reg_lambda=1.618,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=None,
        objective='multi:softmax',
        num_class=len(np.unique(y_train))
    )

    # Set class weights
    sample_weights = np.array([class_weights[label] for label in y_resampled])

    print(f"\nTraining {strategy_name}...")
    model.fit(X_resampled, y_resampled, sample_weight=sample_weights, verbose=False)

    return model


def train_android_ensemble(input_path='data/processed/masaryk_android.csv',
                           output_dir='models',
                           results_dir='results'):
    """Train ensemble of Android Expert models"""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"AndroidEnsemble_{timestamp}"

    model_output_dir = os.path.join(output_dir, run_name)
    results_output_dir = os.path.join(results_dir, run_name)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    print("="*80)
    print("ANDROID EXPERT ENSEMBLE - TRAINING")
    print("="*80)
    print(f"\nTask: Android version classification (7, 8, 9, 10)")
    print(f"Strategy: Train 3 models with different balancing, use majority voting")
    print(f"Input: {input_path}")
    print(f"\nOutput directories:")
    print(f"  Models: {model_output_dir}")
    print(f"  Results: {results_output_dir}")

    # Load data
    print(f"\n[1/6] Loading data...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} samples")

    # Feature selection
    print(f"\n[2/6] Selecting features...")
    feature_columns = select_features(df, verbose=True)

    # Prepare X and y
    print(f"\n[3/6] Encoding categorical features...")
    X = df[feature_columns].copy()
    X, encoders = encode_categorical_features(X, verbose=True)

    # Encode target
    y = df['os_label'].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_.tolist()

    print(f"\n  Classes: {class_names}")

    # Train/test split
    print(f"\n[4/6] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    # Compute class weights
    unique_classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weights = dict(zip(unique_classes, class_weights_array))

    # Train 3 models with different strategies
    print(f"\n[5/6] Training ensemble (3 models)...")

    models = {}

    # Model 1: Minimal balancing (like partial ADASYN)
    models['minimal'] = train_single_model(
        X_train, y_train, class_weights,
        strategy_name="Model 1: Minimal Balancing",
        balance_strategy='minimal',
        use_borderline=False
    )

    # Model 2: Moderate balancing with regular SMOTE
    models['moderate'] = train_single_model(
        X_train, y_train, class_weights,
        strategy_name="Model 2: Moderate Balancing",
        balance_strategy='moderate',
        use_borderline=False
    )

    # Model 3: Moderate balancing with BorderlineSMOTE
    models['borderline'] = train_single_model(
        X_train, y_train, class_weights,
        strategy_name="Model 3: BorderlineSMOTE + Moderate",
        balance_strategy='moderate',
        use_borderline=True
    )

    # Evaluate ensemble
    print(f"\n[6/6] Evaluating ensemble...")
    print("="*80)

    # Get predictions from each model
    predictions = {}
    individual_accuracies = {}

    for name, model in models.items():
        pred = model.predict(X_test)
        predictions[name] = pred
        acc = accuracy_score(y_test, pred)
        individual_accuracies[name] = acc
        print(f"\n{name.upper()} Model Accuracy: {acc*100:.2f}%")

    # Majority voting
    print(f"\n\nEnsemble Voting:")
    print("-"*80)

    ensemble_predictions = []
    for i in range(len(y_test)):
        votes = [predictions[name][i] for name in models.keys()]
        # Get majority vote
        vote_counts = Counter(votes)
        majority_vote = vote_counts.most_common(1)[0][0]
        ensemble_predictions.append(majority_vote)

    ensemble_predictions = np.array(ensemble_predictions)
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

    print(f"Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")
    print(f"\nImprovement over best single model: {(ensemble_accuracy - max(individual_accuracies.values()))*100:+.2f}%")

    # Detailed classification report
    print(f"\n\nEnsemble Classification Report:")
    print("="*80)
    print(classification_report(y_test, ensemble_predictions, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, ensemble_predictions)
    print(f"\nConfusion Matrix:")
    print("-"*80)
    header = "Actual \\ Pred"
    print(f"{header:<15}", end="")
    for cls in class_names:
        print(f"{cls:<15}", end="")
    print()
    print("-"*80)
    for i, actual_class in enumerate(class_names):
        print(f"{actual_class:<15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:<15}", end="")
        print()

    # Save ensemble
    ensemble_path = os.path.join(model_output_dir, 'android_ensemble.pkl')
    ensemble_data = {
        'models': models,
        'label_encoder': le,
        'class_names': class_names,
        'feature_columns': feature_columns,
        'encoders': encoders
    }

    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble_data, f)

    print(f"\n\nEnsemble saved to: {ensemble_path}")

    # Save results JSON
    results = {
        'ensemble_accuracy': float(ensemble_accuracy),
        'individual_accuracies': {k: float(v) for k, v in individual_accuracies.items()},
        'classes': class_names,
        'confusion_matrix': cm.tolist(),
        'timestamp': timestamp
    }

    results_path = os.path.join(results_output_dir, 'ensemble_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING COMPLETE")
    print("="*80)

    return ensemble_data, ensemble_accuracy


if __name__ == "__main__":
    ensemble, accuracy = train_android_ensemble()

    if ensemble is None:
        sys.exit(1)
