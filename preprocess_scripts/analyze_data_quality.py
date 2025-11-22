#!/usr/bin/env python3
"""
Comprehensive Data Quality Analysis for Extracted Flow CSV
Shows column statistics, null percentages, per-OS breakdowns, etc.
"""

import pandas as pd
import numpy as np
import sys
import argparse


def analyze_data_quality(csv_path):
    """Comprehensive data quality analysis"""

    print("="*80)
    print("COMPREHENSIVE DATA QUALITY ANALYSIS")
    print("="*80)
    print(f"\nReading CSV: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")
    print(f"Total columns: {len(df.columns)}")

    # ========================================================================
    # COLUMN OVERVIEW
    # ========================================================================
    print("\n" + "="*80)
    print("COLUMN OVERVIEW")
    print("="*80)
    print(f"\nAll columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    # ========================================================================
    # NULL ANALYSIS - ALL COLUMNS
    # ========================================================================
    print("\n" + "="*80)
    print("NULL ANALYSIS - ALL COLUMNS")
    print("="*80)

    null_stats = []
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / total_rows) * 100
        non_null = total_rows - null_count
        dtype = str(df[col].dtype)

        null_stats.append({
            'Column': col,
            'Null Count': null_count,
            'Null %': null_pct,
            'Non-Null': non_null,
            'Data Type': dtype
        })

    null_df = pd.DataFrame(null_stats)
    null_df = null_df.sort_values('Null %', ascending=False)

    print("\nNull Statistics (sorted by Null %):")
    print(f"\n{'Column':<45} {'Null Count':>12} {'Null %':>8} {'Non-Null':>12} {'Type':<10}")
    print("-" * 95)
    for _, row in null_df.iterrows():
        null_indicator = "⚠️ " if row['Null %'] > 50 else "  "
        print(f"{null_indicator}{row['Column']:<43} {row['Null Count']:>12,} {row['Null %']:>7.1f}% {row['Non-Null']:>12,} {row['Data Type']:<10}")

    # Summary
    columns_with_nulls = (null_df['Null %'] > 0).sum()
    columns_mostly_null = (null_df['Null %'] > 50).sum()
    columns_all_null = (null_df['Null %'] == 100).sum()

    print(f"\nSummary:")
    print(f"  Columns with any nulls:    {columns_with_nulls}/{len(df.columns)} ({columns_with_nulls/len(df.columns)*100:.1f}%)")
    print(f"  Columns >50% null:         {columns_mostly_null}")
    print(f"  Columns 100% null:         {columns_all_null}")

    # ========================================================================
    # NULL ANALYSIS - BY OS VERSION
    # ========================================================================
    print("\n" + "="*80)
    print("NULL ANALYSIS - BY OS VERSION")
    print("="*80)

    if 'os_family' in df.columns:
        os_versions = df['os_family'].unique()
        print(f"\nOS Versions found: {', '.join(sorted(os_versions))}")

        # For each column with nulls, show breakdown by OS
        columns_with_nulls_list = null_df[null_df['Null %'] > 0]['Column'].tolist()

        if columns_with_nulls_list:
            print(f"\nNull breakdown by OS for columns with nulls:")

            for col in columns_with_nulls_list:
                if col == 'os_family':
                    continue

                print(f"\n  {col}:")
                print(f"  {'OS Version':<20} {'Total Rows':>12} {'Null Count':>12} {'Null %':>8}")
                print(f"  {'-'*54}")

                for os_ver in sorted(os_versions):
                    os_df = df[df['os_family'] == os_ver]
                    os_total = len(os_df)
                    os_null = os_df[col].isna().sum()
                    os_null_pct = (os_null / os_total * 100) if os_total > 0 else 0

                    null_indicator = "⚠️ " if os_null_pct > 50 else "  "
                    print(f"  {null_indicator}{os_ver:<18} {os_total:>12,} {os_null:>12,} {os_null_pct:>7.1f}%")
        else:
            print("\n✓ No columns have null values!")

    # ========================================================================
    # FEATURE TYPE BREAKDOWN
    # ========================================================================
    print("\n" + "="*80)
    print("FEATURE TYPE BREAKDOWN")
    print("="*80)

    feature_groups = {
        'TCP Features': [
            'tcp_syn_size', 'tcp_win_size', 'tcp_syn_ttl', 'tcp_flags_a',
            'syn_ack_flag', 'tcp_option_window_scale_forward',
            'tcp_option_selective_ack_permitted_forward',
            'tcp_option_maximum_segment_size_forward',
            'tcp_option_no_operation_forward'
        ],
        'IP Features': [
            'l3_proto', 'ip_tos', 'maximum_ttl_forward', 'ipv4_dont_fragment_forward'
        ],
        'Flow Features': [
            'src_port', 'packet_total_count_forward', 'packet_total_count_backward', 'total_bytes'
        ],
        'TLS Features': [
            'tls_ja3_fingerprint', 'tls_cipher_suites', 'tls_extension_types',
            'tls_elliptic_curves', 'tls_client_version', 'tls_handshake_type',
            'tls_client_key_length'
        ],
        'Derived Features': ['initial_ttl'],
        'Target': ['os_family']
    }

    for group_name, features in feature_groups.items():
        print(f"\n{group_name}:")
        existing_features = [f for f in features if f in df.columns]

        for feat in existing_features:
            null_count = df[feat].isna().sum()
            null_pct = (null_count / total_rows) * 100
            completeness = 100 - null_pct

            status = "✓" if completeness == 100 else "⚠️" if completeness > 50 else "❌"
            print(f"  {status} {feat:<50} {completeness:>6.1f}% complete")

    # ========================================================================
    # VALUE DISTRIBUTIONS
    # ========================================================================
    print("\n" + "="*80)
    print("VALUE DISTRIBUTIONS")
    print("="*80)

    # OS Distribution
    if 'os_family' in df.columns:
        print("\nOS Family Distribution:")
        os_counts = df['os_family'].value_counts()
        for os_name, count in os_counts.items():
            pct = (count / total_rows) * 100
            bar = "█" * int(pct / 2)
            print(f"  {os_name:<20} {count:>6,} ({pct:>5.1f}%) {bar}")

    # TCP Flags Distribution
    if 'tcp_flags_a' in df.columns:
        print("\nTop 10 TCP Flag Combinations:")
        flag_counts = df['tcp_flags_a'].value_counts().head(10)
        for flags, count in flag_counts.items():
            pct = (count / total_rows) * 100
            print(f"  {flags:<15} {count:>6,} ({pct:>5.1f}%)")

    # Bidirectional vs Unidirectional
    if 'packet_total_count_backward' in df.columns:
        has_backward = (df['packet_total_count_backward'] > 0).sum()
        no_backward = (df['packet_total_count_backward'] == 0).sum()
        print(f"\nFlow Direction:")
        print(f"  Bidirectional (backward>0): {has_backward:>6,} ({has_backward/total_rows*100:>5.1f}%)")
        print(f"  Unidirectional (backward=0): {no_backward:>6,} ({no_backward/total_rows*100:>5.1f}%)")

    # SYN-ACK Distribution
    if 'syn_ack_flag' in df.columns:
        has_synack = (df['syn_ack_flag'] == 1).sum()
        no_synack = (df['syn_ack_flag'] == 0).sum()
        print(f"\nSYN-ACK Flag:")
        print(f"  Has SYN-ACK (flag=1): {has_synack:>6,} ({has_synack/total_rows*100:>5.1f}%)")
        print(f"  No SYN-ACK (flag=0):  {no_synack:>6,} ({no_synack/total_rows*100:>5.1f}%)")

    # ========================================================================
    # DATA QUALITY ISSUES
    # ========================================================================
    print("\n" + "="*80)
    print("DATA QUALITY ISSUES")
    print("="*80)

    issues = []

    # Check for columns with >50% nulls
    mostly_null_cols = null_df[null_df['Null %'] > 50]['Column'].tolist()
    if mostly_null_cols:
        issues.append(f"⚠️  {len(mostly_null_cols)} columns have >50% null values:")
        for col in mostly_null_cols:
            null_pct = null_df[null_df['Column'] == col]['Null %'].values[0]
            issues.append(f"     - {col}: {null_pct:.1f}% null")

    # Check for impossible values
    if 'packet_total_count_forward' in df.columns and 'packet_total_count_backward' in df.columns:
        both_zero = ((df['packet_total_count_forward'] == 0) &
                     (df['packet_total_count_backward'] == 0)).sum()
        if both_zero > 0:
            issues.append(f"❌ {both_zero} flows have BOTH forward=0 AND backward=0 (impossible!)")

    # Check SYN-ACK consistency
    if 'syn_ack_flag' in df.columns and 'packet_total_count_backward' in df.columns:
        inconsistent = ((df['syn_ack_flag'] == 1) &
                       (df['packet_total_count_backward'] == 0)).sum()
        if inconsistent > 0:
            issues.append(f"❌ {inconsistent} flows have syn_ack_flag=1 with backward=0 (impossible!)")

    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"⚠️  {duplicates} duplicate rows found")

    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        if col != 'os_family':  # Skip target
            unique_count = df[col].nunique()
            if unique_count == 1:
                constant_cols.append(col)

    if constant_cols:
        issues.append(f"⚠️  {len(constant_cols)} columns have only 1 unique value (constant):")
        for col in constant_cols:
            issues.append(f"     - {col}")

    if issues:
        print("\nIssues detected:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No major data quality issues detected!")

    # ========================================================================
    # STATISTICS SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("STATISTICS SUMMARY")
    print("="*80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("\nNumeric Feature Statistics:")
    print(f"\n{'Feature':<50} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10}")
    print("-" * 92)

    for col in numeric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            median_val = df[col].median()
            min_val = df[col].min()
            max_val = df[col].max()

            print(f"{col:<50} {mean_val:>10.1f} {median_val:>10.1f} {min_val:>10.0f} {max_val:>10.0f}")

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    recommendations = []

    # TLS features
    if 'tls_ja3_fingerprint' in df.columns:
        tls_completeness = (df['tls_ja3_fingerprint'].notna().sum() / total_rows) * 100
        if tls_completeness < 30:
            recommendations.append(f"• TLS features only {tls_completeness:.1f}% complete - consider if TLS fingerprinting is critical")
        elif tls_completeness > 30 and tls_completeness < 70:
            recommendations.append(f"• TLS features {tls_completeness:.1f}% complete - good coverage for mixed HTTP/HTTPS traffic")

    # Backward packets
    if 'packet_total_count_backward' in df.columns:
        bidirectional_pct = ((df['packet_total_count_backward'] > 0).sum() / total_rows) * 100
        if bidirectional_pct < 20:
            recommendations.append(f"• Only {bidirectional_pct:.1f}% flows are bidirectional - expected for IDS dataset with scans/failed connections")
            recommendations.append(f"• Consider using forward-only features as primary fingerprints")

    # Constant columns
    if constant_cols:
        recommendations.append(f"• Drop {len(constant_cols)} constant columns before training (no variance)")

    # Class balance
    if 'os_family' in df.columns:
        os_counts = df['os_family'].value_counts()
        min_class = os_counts.min()
        max_class = os_counts.max()
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

        if imbalance_ratio > 3:
            recommendations.append(f"• Class imbalance ratio {imbalance_ratio:.1f}:1 - consider ADASYN/SMOTE for balancing")
        else:
            recommendations.append(f"• Class balance ratio {imbalance_ratio:.1f}:1 - reasonably balanced")

    if recommendations:
        print("\n")
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("\n✓ Data looks ready for training!")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive data quality analysis for flow CSV'
    )

    parser.add_argument(
        'csv',
        type=str,
        help='Path to CSV file with extracted flows'
    )

    args = parser.parse_args()

    analyze_data_quality(args.csv)


if __name__ == '__main__':
    main()
