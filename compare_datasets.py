#!/usr/bin/env python3
"""
Compare CESNET and Masaryk Datasets for Merging Compatibility

Verifies that cesnet.csv and masaryk.csv have identical structure
and provides detailed statistics for both datasets.

Usage:
    python compare_datasets.py
    python compare_datasets.py --cesnet data/processed/cesnet.csv --masaryk data/processed/masaryk_1.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def compare_shapes(df1, df2, name1, name2):
    """Compare shapes of two dataframes"""
    print("\n" + "="*70)
    print("SHAPE COMPARISON")
    print("="*70)

    print(f"\n{name1}:")
    print(f"  Rows: {len(df1):,}")
    print(f"  Columns: {len(df1.columns)}")

    print(f"\n{name2}:")
    print(f"  Rows: {len(df2):,}")
    print(f"  Columns: {len(df2.columns)}")

    if len(df1.columns) == len(df2.columns):
        print(f"\n✓ Column count matches!")
    else:
        print(f"\n✗ Column count MISMATCH!")
        print(f"  Difference: {abs(len(df1.columns) - len(df2.columns))} columns")


def compare_columns(df1, df2, name1, name2):
    """Compare column names between two dataframes"""
    print("\n" + "="*70)
    print("COLUMN COMPARISON")
    print("="*70)

    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    # Columns in both
    common_cols = cols1 & cols2
    print(f"\n✓ Common columns: {len(common_cols)}")

    # Columns only in df1
    only_in_1 = cols1 - cols2
    if only_in_1:
        print(f"\n✗ Columns only in {name1} ({len(only_in_1)}):")
        for col in sorted(only_in_1):
            print(f"    - {col}")

    # Columns only in df2
    only_in_2 = cols2 - cols1
    if only_in_2:
        print(f"\n✗ Columns only in {name2} ({len(only_in_2)}):")
        for col in sorted(only_in_2):
            print(f"    - {col}")

    if not only_in_1 and not only_in_2:
        print(f"\n✓ All columns match perfectly!")
        return True
    else:
        print(f"\n✗ Column mismatch detected!")
        return False


def compare_dtypes(df1, df2, name1, name2):
    """Compare data types of common columns"""
    print("\n" + "="*70)
    print("DATA TYPE COMPARISON")
    print("="*70)

    common_cols = set(df1.columns) & set(df2.columns)

    mismatches = []
    for col in sorted(common_cols):
        dtype1 = df1[col].dtype
        dtype2 = df2[col].dtype

        if dtype1 != dtype2:
            mismatches.append((col, dtype1, dtype2))

    if mismatches:
        print(f"\n⚠ Data type mismatches found ({len(mismatches)}):")
        print(f"\n{'Column':<45} {name1:<15} {name2:<15}")
        print("-" * 70)
        for col, dtype1, dtype2 in mismatches:
            print(f"{col:<45} {str(dtype1):<15} {str(dtype2):<15}")
    else:
        print(f"\n✓ All common columns have matching data types!")


def analyze_missing_values(df, name):
    """Analyze missing values in dataset"""
    print("\n" + "="*70)
    print(f"MISSING VALUES ANALYSIS - {name}")
    print("="*70)

    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / total_cells) * 100

    print(f"\nOverall:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Missing cells: {total_missing:,} ({missing_pct:.2f}%)")

    # Per-column missing values
    missing_by_col = df.isnull().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False)

    if len(cols_with_missing) > 0:
        print(f"\nColumns with missing values ({len(cols_with_missing)}):")
        print(f"\n{'Column':<45} {'Missing':<12} {'Percentage':<12}")
        print("-" * 70)

        for col, count in cols_with_missing.head(20).items():
            pct = (count / len(df)) * 100
            status = "✗" if pct > 50 else ("⚠" if pct > 20 else "·")
            print(f"{status} {col:<43} {count:>10,} {pct:>10.2f}%")

        if len(cols_with_missing) > 20:
            print(f"\n... and {len(cols_with_missing) - 20} more columns")
    else:
        print(f"\n✓ No missing values found!")


def analyze_ttl_anomalies(df, name):
    """Analyze TTL values and detect anomalies"""
    print("\n" + "="*70)
    print(f"TTL ANOMALY ANALYSIS - {name}")
    print("="*70)

    if 'os_family' not in df.columns or 'initial_ttl' not in df.columns:
        print("\n⚠ Missing required columns (os_family, initial_ttl)")
        return

    # Expected TTL by OS family
    expected_ttl = {
        'Windows': 128,
        'Linux': 64,
        'macOS': 64,
        'Android': 64,
        'iOS': 64,
        'BSD': 64,
    }

    print(f"\nExpected initial TTL values:")
    for os_fam, ttl in expected_ttl.items():
        print(f"  {os_fam}: {ttl}")

    print(f"\nAnomaly detection:")
    print(f"\n{'OS Family':<15} {'Total':<12} {'Anomalies':<12} {'Percentage':<12}")
    print("-" * 70)

    total_anomalies = 0
    total_with_ttl = 0

    for os_fam in sorted(expected_ttl.keys()):
        expected = expected_ttl[os_fam]

        # Filter for this OS family
        os_df = df[df['os_family'] == os_fam].copy()
        os_df_with_ttl = os_df[os_df['initial_ttl'].notna()]

        total_with_ttl += len(os_df_with_ttl)

        if len(os_df_with_ttl) == 0:
            print(f"  {os_fam:<15} {0:<12} {0:<12} {'-':<12}")
            continue

        # Find anomalies
        anomalies = os_df_with_ttl[os_df_with_ttl['initial_ttl'] != expected]
        anomaly_count = len(anomalies)
        anomaly_pct = (anomaly_count / len(os_df_with_ttl)) * 100

        total_anomalies += anomaly_count

        status = "✗" if anomaly_pct > 10 else ("⚠" if anomaly_pct > 5 else "✓")
        print(f"{status} {os_fam:<13} {len(os_df_with_ttl):>10,} {anomaly_count:>10,} {anomaly_pct:>10.2f}%")

        # Show what TTL values were found instead
        if anomaly_count > 0:
            ttl_dist = anomalies['initial_ttl'].value_counts().head(3)
            ttl_str = ", ".join([f"{int(ttl)}({count})" for ttl, count in ttl_dist.items()])
            print(f"    Unexpected TTL values: {ttl_str}")

    overall_anomaly_pct = (total_anomalies / total_with_ttl * 100) if total_with_ttl > 0 else 0
    print(f"\n  Overall anomaly rate: {total_anomalies:,}/{total_with_ttl:,} ({overall_anomaly_pct:.2f}%)")


def analyze_os_distribution(df, name):
    """Analyze OS family distribution"""
    print("\n" + "="*70)
    print(f"OS FAMILY DISTRIBUTION - {name}")
    print("="*70)

    if 'os_family' not in df.columns:
        print("\n⚠ No os_family column found")
        return

    os_counts = df['os_family'].value_counts()

    print(f"\n{'OS Family':<15} {'Count':<12} {'Percentage':<12}")
    print("-" * 70)

    for os_fam, count in os_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {os_fam:<15} {count:>10,} {pct:>10.2f}%")

    print(f"\n  Total: {len(df):,}")


def analyze_feature_completeness(df, name):
    """Analyze completeness of key features"""
    print("\n" + "="*70)
    print(f"KEY FEATURE COMPLETENESS - {name}")
    print("="*70)

    # Define key feature groups
    feature_groups = {
        'TCP Parameters': [
            'tcp_syn_size', 'tcp_win_size', 'tcp_syn_ttl', 'tcp_flags_a', 'syn_ack_flag',
            'tcp_option_window_scale_forward', 'tcp_option_window_scale_backward',
            'tcp_option_maximum_segment_size_forward', 'tcp_option_maximum_segment_size_backward',
        ],
        'IP Parameters': [
            'l3_proto', 'l4_proto', 'ip_tos',
            'maximum_ttl_forward', 'maximum_ttl_backward',
            'ipv4_dont_fragment_forward', 'ipv4_dont_fragment_backward',
        ],
        'Flow Properties': [
            'bytes_a', 'packets_a', 'src_port', 'dst_port',
            'packet_total_count_forward', 'packet_total_count_backward',
        ],
        'NPM Timing': [
            'npm_round_trip_time', 'npm_tcp_retransmission_a', 'npm_tcp_retransmission_b',
            'npm_tcp_out_of_order_a', 'npm_tcp_out_of_order_b',
        ],
        'TLS Features': [
            'tls_ja3_fingerprint', 'tls_client_version', 'tls_cipher_suites',
            'tls_extension_types', 'tls_elliptic_curves',
        ],
        'Derived Features': [
            'initial_ttl', 'total_bytes',
        ],
    }

    for group_name, features in feature_groups.items():
        print(f"\n{group_name}:")

        available_features = [f for f in features if f in df.columns]

        if not available_features:
            print(f"  ⚠ No features from this group found")
            continue

        for feat in available_features:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"  {status} {feat:<45}: {pct:>5.1f}%")


def analyze_value_ranges(df, name):
    """Analyze value ranges for numeric features"""
    print("\n" + "="*70)
    print(f"VALUE RANGES - {name}")
    print("="*70)

    numeric_features = [
        'tcp_syn_size', 'tcp_win_size', 'tcp_syn_ttl', 'initial_ttl',
        'src_port', 'dst_port', 'bytes_a', 'packets_a',
    ]

    available_features = [f for f in numeric_features if f in df.columns]

    if not available_features:
        print("\n⚠ No numeric features found for analysis")
        return

    print(f"\n{'Feature':<30} {'Min':<12} {'Max':<12} {'Mean':<12} {'Median':<12}")
    print("-" * 70)

    for feat in available_features:
        if df[feat].notna().sum() == 0:
            print(f"  {feat:<30} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue

        min_val = df[feat].min()
        max_val = df[feat].max()
        mean_val = df[feat].mean()
        median_val = df[feat].median()

        print(f"  {feat:<30} {min_val:<12.1f} {max_val:<12.1f} {mean_val:<12.1f} {median_val:<12.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare CESNET and Masaryk datasets for merging compatibility'
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
        help='Path to Masaryk processed CSV'
    )

    args = parser.parse_args()

    print("="*70)
    print("DATASET COMPARISON: CESNET vs MASARYK")
    print("="*70)

    # Load datasets
    print(f"\nLoading datasets...")

    if not os.path.exists(args.cesnet):
        print(f"\n✗ ERROR: CESNET file not found: {args.cesnet}")
        sys.exit(1)

    if not os.path.exists(args.masaryk):
        print(f"\n✗ ERROR: Masaryk file not found: {args.masaryk}")
        sys.exit(1)

    print(f"  Loading {args.cesnet}...")
    cesnet_df = pd.read_csv(args.cesnet)
    print(f"  ✓ Loaded {len(cesnet_df):,} rows")

    print(f"  Loading {args.masaryk}...")
    masaryk_df = pd.read_csv(args.masaryk)
    print(f"  ✓ Loaded {len(masaryk_df):,} rows")

    # Perform comparisons
    compare_shapes(cesnet_df, masaryk_df, "CESNET", "MASARYK")
    columns_match = compare_columns(cesnet_df, masaryk_df, "CESNET", "MASARYK")
    compare_dtypes(cesnet_df, masaryk_df, "CESNET", "MASARYK")

    # Detailed statistics for each dataset
    analyze_missing_values(cesnet_df, "CESNET")
    analyze_missing_values(masaryk_df, "MASARYK")

    analyze_os_distribution(cesnet_df, "CESNET")
    analyze_os_distribution(masaryk_df, "MASARYK")

    analyze_ttl_anomalies(cesnet_df, "CESNET")
    analyze_ttl_anomalies(masaryk_df, "MASARYK")

    analyze_feature_completeness(cesnet_df, "CESNET")
    analyze_feature_completeness(masaryk_df, "MASARYK")

    analyze_value_ranges(cesnet_df, "CESNET")
    analyze_value_ranges(masaryk_df, "MASARYK")

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    if columns_match and len(cesnet_df.columns) == len(masaryk_df.columns):
        print("\n✓ Datasets are compatible for merging!")
        print("\nTo merge:")
        print(f"  import pandas as pd")
        print(f"  cesnet = pd.read_csv('{args.cesnet}')")
        print(f"  masaryk = pd.read_csv('{args.masaryk}')")
        print(f"  combined = pd.concat([masaryk, cesnet], ignore_index=True)")
        print(f"  combined.to_csv('data/processed/combined.csv', index=False)")
    else:
        print("\n✗ Datasets have structural differences - merging may fail!")
        print("  Please review the comparison results above.")


if __name__ == '__main__':
    main()
