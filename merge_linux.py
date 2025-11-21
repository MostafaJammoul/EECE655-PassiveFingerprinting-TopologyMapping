#!/usr/bin/env python3
"""
Merge CESNET Linux flows with Masaryk dataset

Merges cesnet.csv + masaryk_1.csv → masaryk.csv
Drops NPM timing features (not available in CESNET, not needed for deployment)

Usage:
    python merge_linux.py
    python merge_linux.py --cesnet data/processed/cesnet.csv --masaryk data/processed/masaryk_1.csv --output data/processed/masaryk.csv
"""

import os
import sys
import pandas as pd
import argparse


# NPM columns to drop (not available in CESNET, not needed for deployment)
NPM_COLUMNS = [
    'npm_round_trip_time',
    'npm_tcp_retransmission_a',
    'npm_tcp_retransmission_b',
    'npm_tcp_out_of_order_a',
    'npm_tcp_out_of_order_b',
]


def print_dataset_stats(df, name):
    """Print detailed statistics about a dataset"""
    print(f"\n{'='*70}")
    print(f"{name} - STATISTICS")
    print(f"{'='*70}")

    print(f"\nShape: {df.shape}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # OS distribution
    if 'os_family' in df.columns:
        print(f"\nOS Family Distribution:")
        os_counts = df['os_family'].value_counts()
        print(f"\n  {'OS Family':<15} {'Count':<12} {'Percentage':<12}")
        print(f"  {'-'*40}")
        for os_fam, count in os_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {os_fam:<15} {count:>10,}   {pct:>6.2f}%")

    # Missing values
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / total_cells) * 100

    print(f"\nMissing Values:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Missing: {total_missing:,} ({missing_pct:.2f}%)")

    # Show columns with most missing values
    missing_by_col = df.isnull().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False).head(10)

    if len(cols_with_missing) > 0:
        print(f"\n  Top 10 columns with missing values:")
        for col, count in cols_with_missing.items():
            pct = (count / len(df)) * 100
            print(f"    {col:<45}: {count:>8,} ({pct:>5.1f}%)")

    # Dataset source distribution
    if 'dataset_source' in df.columns:
        print(f"\nDataset Source:")
        for source, count in df['dataset_source'].value_counts().items():
            pct = (count / len(df)) * 100
            print(f"  {source:<15}: {count:>10,} ({pct:>5.1f}%)")


def merge_datasets(cesnet_path, masaryk_path, output_path, verbose=True):
    """Merge CESNET and Masaryk datasets, drop NPM columns"""

    print("="*70)
    print("MERGING DATASETS: CESNET + MASARYK")
    print("="*70)

    # Load datasets
    if verbose:
        print(f"\n[1/5] Loading datasets...")

    if not os.path.exists(cesnet_path):
        print(f"\n✗ ERROR: CESNET file not found: {cesnet_path}")
        sys.exit(1)

    if not os.path.exists(masaryk_path):
        print(f"\n✗ ERROR: Masaryk file not found: {masaryk_path}")
        sys.exit(1)

    print(f"  Loading {cesnet_path}...")
    cesnet_df = pd.read_csv(cesnet_path)
    print(f"  ✓ Loaded {len(cesnet_df):,} rows, {len(cesnet_df.columns)} columns")

    print(f"  Loading {masaryk_path}...")
    masaryk_df = pd.read_csv(masaryk_path)
    print(f"  ✓ Loaded {len(masaryk_df):,} rows, {len(masaryk_df.columns)} columns")

    # Show original stats
    if verbose:
        print_dataset_stats(cesnet_df, "CESNET (Original)")
        print_dataset_stats(masaryk_df, "MASARYK (Original)")

    # Drop NPM columns from both datasets
    if verbose:
        print(f"\n[2/5] Dropping NPM timing columns...")
        print(f"  Columns to drop: {NPM_COLUMNS}")

    cesnet_df = cesnet_df.drop(columns=NPM_COLUMNS, errors='ignore')
    masaryk_df = masaryk_df.drop(columns=NPM_COLUMNS, errors='ignore')

    if verbose:
        print(f"  ✓ CESNET after dropping NPM: {len(cesnet_df.columns)} columns")
        print(f"  ✓ Masaryk after dropping NPM: {len(masaryk_df.columns)} columns")

    # Verify column alignment
    if verbose:
        print(f"\n[3/5] Verifying column alignment...")

    cesnet_cols = set(cesnet_df.columns)
    masaryk_cols = set(masaryk_df.columns)

    common_cols = cesnet_cols & masaryk_cols
    only_in_cesnet = cesnet_cols - masaryk_cols
    only_in_masaryk = masaryk_cols - cesnet_cols

    if verbose:
        print(f"  Common columns: {len(common_cols)}")

        if only_in_cesnet:
            print(f"  ⚠ Columns only in CESNET ({len(only_in_cesnet)}): {sorted(only_in_cesnet)}")

        if only_in_masaryk:
            print(f"  ⚠ Columns only in Masaryk ({len(only_in_masaryk)}): {sorted(only_in_masaryk)}")

        if not only_in_cesnet and not only_in_masaryk:
            print(f"  ✓ All columns match!")

    # Merge datasets
    if verbose:
        print(f"\n[4/5] Merging datasets...")
        print(f"  Concatenating {len(masaryk_df):,} Masaryk + {len(cesnet_df):,} CESNET rows...")

    combined_df = pd.concat([masaryk_df, cesnet_df], ignore_index=True)

    if verbose:
        print(f"  ✓ Merged dataset: {len(combined_df):,} rows, {len(combined_df.columns)} columns")

    # Save merged dataset
    if verbose:
        print(f"\n[5/5] Saving merged dataset...")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    if verbose:
        print(f"  ✓ Saved to: {output_path}")

    # Show final stats
    if verbose:
        print_dataset_stats(combined_df, "MERGED DATASET (Final)")

    # Summary
    print("\n" + "="*70)
    print("MERGE COMPLETE")
    print("="*70)

    original_linux = len(masaryk_df[masaryk_df['os_family'] == 'Linux']) if 'os_family' in masaryk_df.columns else 0
    added_linux = len(cesnet_df[cesnet_df['os_family'] == 'Linux']) if 'os_family' in cesnet_df.columns else 0
    final_linux = len(combined_df[combined_df['os_family'] == 'Linux']) if 'os_family' in combined_df.columns else 0

    print(f"\nLinux samples:")
    print(f"  Original (Masaryk): {original_linux:,}")
    print(f"  Added (CESNET): {added_linux:,}")
    print(f"  Final: {final_linux:,}")

    if original_linux > 0:
        increase_pct = ((final_linux - original_linux) / original_linux) * 100
        print(f"  Increase: +{increase_pct:.1f}%")

    print(f"\nTotal samples:")
    print(f"  Original (Masaryk): {len(masaryk_df):,}")
    print(f"  Added (CESNET): {len(cesnet_df):,}")
    print(f"  Final: {len(combined_df):,}")

    print(f"\nFeatures:")
    print(f"  Original: {len(masaryk_df.columns)} (before dropping NPM)")
    print(f"  Dropped NPM columns: {len(NPM_COLUMNS)}")
    print(f"  Final: {len(combined_df.columns)}")

    print(f"\n✓ Merged dataset ready for training!")
    print(f"  Path: {output_path}")

    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description='Merge CESNET Linux flows with Masaryk dataset'
    )

    parser.add_argument(
        '--cesnet',
        type=str,
        default='data/processed/cesnet.csv',
        help='Path to CESNET processed CSV'
    )

    parser.add_argument(
        '--masaryk',
        type=str,
        default='data/processed/masaryk_1.csv',
        help='Path to Masaryk processed CSV (input)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/masaryk.csv',
        help='Path to save merged dataset'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Run merge
    merge_datasets(
        cesnet_path=args.cesnet,
        masaryk_path=args.masaryk,
        output_path=args.output,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
