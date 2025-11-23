#!/usr/bin/env python3
"""
Debug Android Expert Low Accuracy

Investigates why Android Expert model has low accuracy (67%).
Checks data quality, feature distributions, and class separability.

Usage: python debug_android_accuracy.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_android_dataset(csv_path='data/processed/masaryk_android.csv'):
    """Debug Android dataset for low accuracy issues"""

    print("="*80)
    print("ANDROID EXPERT - LOW ACCURACY DEBUGGING")
    print("="*80)

    if not Path(csv_path).exists():
        print(f"\nERROR: File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"\nDataset: {csv_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Check target distribution
    print("\n" + "="*80)
    print("TARGET DISTRIBUTION")
    print("="*80)
    print(f"\n{df['os_label'].value_counts()}")
    print(f"\nPercentages:")
    print(f"{(df['os_label'].value_counts() / len(df) * 100).round(2)}")

    # Check for data quality issues
    print("\n" + "="*80)
    print("DATA QUALITY CHECKS")
    print("="*80)

    # 1. Missing values
    print(f"\n1. MISSING VALUES:")
    null_cols = df.isnull().sum()
    high_nulls = null_cols[null_cols > len(df) * 0.5]
    if len(high_nulls) > 0:
        print(f"   ⚠ {len(high_nulls)} columns with >50% missing:")
        for col, count in high_nulls.items():
            pct = (count / len(df)) * 100
            print(f"     - {col}: {pct:.1f}% missing")
    else:
        print(f"   ✓ No columns with excessive missing values")

    # 2. Feature variance
    print(f"\n2. FEATURE VARIANCE:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'os_label']

    low_variance = []
    for col in numeric_cols:
        var = df[col].var()
        if var < 0.01 and df[col].notna().sum() > 0:
            low_variance.append((col, var))

    if low_variance:
        print(f"   ⚠ {len(low_variance)} features with very low variance:")
        for col, var in sorted(low_variance, key=lambda x: x[1])[:10]:
            print(f"     - {col}: variance={var:.6f}")
    else:
        print(f"   ✓ All features have sufficient variance")

    # 3. Constant features
    print(f"\n3. CONSTANT FEATURES:")
    constant_features = []
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            unique_vals = df[col].nunique()
            if unique_vals == 1:
                constant_features.append(col)

    if constant_features:
        print(f"   ⚠ {len(constant_features)} constant features (all same value):")
        for col in constant_features[:10]:
            val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else 'N/A'
            print(f"     - {col} = {val}")
    else:
        print(f"   ✓ No constant features")

    # 4. Check critical features
    print(f"\n4. CRITICAL FEATURES AVAILABILITY:")
    critical_features = {
        'tcp_win_size': 'HIGH',
        'tcp_syn_size': 'HIGH',
        'initial_ttl': 'HIGH',
        'tcp_option_window_scale_forward': 'HIGH',
        'tls_ja3_fingerprint': 'MEDIUM',
        'tls_cipher_suites': 'MEDIUM',
    }

    for feat, importance in critical_features.items():
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"   {status} {feat:<40} {pct:>5.1f}% ({importance})")
        else:
            print(f"   ✗ {feat:<40} MISSING")

    # 5. Check class separability for key features
    print(f"\n5. CLASS SEPARABILITY (Key Features):")
    key_features = ['tcp_win_size', 'tcp_syn_size', 'initial_ttl', 'tcp_option_window_scale_forward']

    for feat in key_features:
        if feat in df.columns and df[feat].notna().sum() > 0:
            print(f"\n   {feat}:")
            for label in sorted(df['os_label'].unique()):
                label_data = df[df['os_label'] == label][feat].dropna()
                if len(label_data) > 0:
                    print(f"     {label:<12}: mean={label_data.mean():>8.2f}, std={label_data.std():>8.2f}, unique={label_data.nunique():>5}")

    # 6. Check if all Android versions are actually different
    print(f"\n6. VERSION DISCRIMINATION CHECK:")
    print(f"   Checking if Android versions have distinct patterns...")

    # Sample some rows from each version
    for label in sorted(df['os_label'].unique()):
        label_df = df[df['os_label'] == label]
        print(f"\n   {label} ({len(label_df)} samples):")

        # Check tcp_win_size distribution
        if 'tcp_win_size' in df.columns:
            win_sizes = label_df['tcp_win_size'].value_counts().head(3)
            print(f"     Top tcp_win_size values: {dict(win_sizes)}")

        # Check initial_ttl
        if 'initial_ttl' in df.columns:
            ttl_vals = label_df['initial_ttl'].value_counts()
            print(f"     initial_ttl distribution: {dict(ttl_vals)}")

    # 7. Recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print(f"\nBased on analysis:")

    if len(high_nulls) > 0:
        print(f"  1. ⚠ High missing values in {len(high_nulls)} features")
        print(f"     → Consider removing these features or using imputation")

    if len(constant_features) > 0:
        print(f"  2. ⚠ {len(constant_features)} constant features provide no information")
        print(f"     → Remove these features before training")

    if len(low_variance) > 5:
        print(f"  3. ⚠ Many low-variance features")
        print(f"     → Consider feature selection / PCA")

    print(f"\n  Possible reasons for 67% accuracy:")
    print(f"    - Android versions may have very similar network patterns")
    print(f"    - Missing critical discriminative features (TLS ~42% null)")
    print(f"    - Model hyperparameters not optimized for this dataset")
    print(f"    - Small dataset size (7,482 samples) with 4 classes")

    print(f"\n  Next steps:")
    print(f"    1. Run: python debug_android_accuracy.py")
    print(f"    2. Review class separability above")
    print(f"    3. Try hyperparameter tuning if patterns are separable")
    print(f"    4. Consider using only high-quality features (non-null)")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Debug Android Expert low accuracy')
    parser.add_argument('--input', type=str, default='data/processed/masaryk_android.csv',
                       help='Path to Android CSV')

    args = parser.parse_args()

    debug_android_dataset(args.input)
