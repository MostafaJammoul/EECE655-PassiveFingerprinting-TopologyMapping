#!/usr/bin/env python3
"""
Check CESNET dataset class distribution

Reads data/processed/cesnet.csv and displays OS version distribution
to help determine if SMOTE/ADASYN is needed for class balancing.
"""

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed.")
    print("Install with: pip install pandas")
    sys.exit(1)


def check_cesnet_distribution(csv_path='data/processed/cesnet.csv'):
    """
    Check OS version distribution in CESNET dataset

    Args:
        csv_path: Path to processed cesnet CSV file
    """
    csv_file = Path(csv_path)

    if not csv_file.exists():
        print(f"ERROR: File not found: {csv_path}")
        print("\nThe CESNET dataset hasn't been preprocessed yet.")
        print("Run preprocessing first:")
        print("  python preprocess_scripts/cesnet_preprocess.py")
        return None

    print("="*70)
    print("CESNET DATASET - OS VERSION DISTRIBUTION")
    print("="*70)
    print(f"\nReading: {csv_path}")

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return None

    print(f"Total records: {len(df):,}")

    # Check if os_label column exists
    if 'os_label' not in df.columns:
        print("\nERROR: 'os_label' column not found in dataset!")
        print(f"Available columns: {', '.join(df.columns)}")
        return None

    # Get OS version distribution
    print("\n" + "="*70)
    print("OS VERSION DISTRIBUTION")
    print("="*70)

    os_counts = df['os_label'].value_counts()
    total = len(df)

    print(f"\n{'OS Version':<40} {'Count':>10} {'Percentage':>12}")
    print("-"*70)

    for os_version, count in os_counts.items():
        percentage = (count / total) * 100
        print(f"{str(os_version):<40} {count:>10,} {percentage:>11.2f}%")

    print("-"*70)
    print(f"{'TOTAL':<40} {total:>10,} {100.0:>11.2f}%")

    # Imbalance analysis
    print("\n" + "="*70)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*70)

    max_count = os_counts.max()
    min_count = os_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    print(f"\nMost common class:  {os_counts.index[0]} ({max_count:,} samples)")
    print(f"Least common class: {os_counts.index[-1]} ({min_count:,} samples)")
    print(f"Imbalance ratio:    {imbalance_ratio:.2f}:1")

    # Count minority classes (< 10% of dataset)
    minority_threshold = total * 0.10
    minority_classes = os_counts[os_counts < minority_threshold]

    print(f"\nMinority classes (< 10% of data): {len(minority_classes)}")
    if len(minority_classes) > 0:
        print("\nMinority class details:")
        for os_version, count in minority_classes.items():
            percentage = (count / total) * 100
            print(f"  {str(os_version):<38} {count:>8,} ({percentage:.2f}%)")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    if imbalance_ratio > 10:
        print("\n⚠️  SEVERE IMBALANCE DETECTED (>10:1 ratio)")
        print("\nRecommendation: Use SMOTE or ADASYN")
        print("  • SMOTE: Good for general oversampling")
        print("  • ADASYN: Better for focusing on hard-to-learn samples")
        print("\nWithout oversampling, your model may:")
        print("  - Ignore minority classes entirely")
        print("  - Achieve high accuracy but poor per-class performance")
        print("  - Have very low recall for rare OS versions")
    elif imbalance_ratio > 3:
        print("\n⚠️  MODERATE IMBALANCE DETECTED (>3:1 ratio)")
        print("\nRecommendation: Consider using SMOTE/ADASYN or class weights")
        print("  • Option 1: Use SMOTE/ADASYN for oversampling")
        print("  • Option 2: Use class_weight='balanced' in your classifier")
    else:
        print("\n✓ RELATIVELY BALANCED DATASET")
        print("\nRecommendation: SMOTE/ADASYN may not be necessary")
        print("  • Your dataset is reasonably balanced")
        print("  • Standard training should work well")
        print("  • You can still use class_weight='balanced' for slight improvements")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Check CESNET dataset OS version distribution'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/cesnet.csv',
        help='Path to cesnet CSV file (default: data/processed/cesnet.csv)'
    )

    args = parser.parse_args()

    df = check_cesnet_distribution(args.input)

    if df is None:
        sys.exit(1)

    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
