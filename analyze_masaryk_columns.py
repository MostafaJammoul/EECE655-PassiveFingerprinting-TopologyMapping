#!/usr/bin/env python3
"""
Analyze Masaryk Dataset Column Completeness

This script reads the Masaryk dataset CSV and reports the percentage of non-empty
values for each column/field position.

Usage:
    python analyze_masaryk_columns.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_masaryk_completeness(csv_path='data/raw/masaryk/flows_ground_truth_merged_anonymized.csv'):
    """
    Analyze column completeness in Masaryk dataset

    The Masaryk dataset uses semicolon-separated format.
    This function reads the file, parses it, and reports completeness statistics.
    """

    print("="*80)
    print("MASARYK DATASET COLUMN COMPLETENESS ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing: {csv_path}\n")

    csv_file = Path(csv_path)

    if not csv_file.exists():
        print(f"ERROR: File not found: {csv_path}")
        print("\nPlease ensure the Masaryk dataset is downloaded to:")
        print("  data/raw/masaryk/flows_ground_truth_merged_anonymized.csv")
        return

    # Read header row to get actual column names
    print("[1/3] Reading header row...")
    with open(csv_file, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
        column_names = header_line.split(';')
        num_fields = len(column_names)
        print(f"  Found {num_fields} columns in header")
        print(f"\n  First 10 columns:")
        for i, col in enumerate(column_names[:10]):
            print(f"    {i}: {col}")
        print(f"  ...")
        print(f"  Last 5 columns:")
        for i, col in enumerate(column_names[-5:], start=len(column_names)-5):
            print(f"    {i}: {col}")

    # Read the CSV with proper parsing
    print(f"\n[2/3] Reading CSV data (this may take a while)...")

    # Read line by line and parse (skip header)
    all_rows = []
    total_lines = 0
    error_lines = 0

    with open(csv_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)

        for line_num, line in enumerate(f, 2):  # Start at 2 since row 1 is header
            line = line.strip()
            if not line:
                continue

            fields = line.split(';')

            # Pad with None if fewer fields than expected
            while len(fields) < num_fields:
                fields.append(None)

            # Truncate if more fields than expected
            fields = fields[:num_fields]

            all_rows.append(fields)
            total_lines += 1

            # Progress indicator
            if total_lines % 100000 == 0:
                print(f"  Processed {total_lines:,} data rows...")

            # Limit for quick analysis (remove this for full analysis)
            # if total_lines >= 500000:
            #     print(f"  Stopping at {total_lines:,} rows for quick analysis")
            #     break

    print(f"  Total data rows processed: {total_lines:,}")

    # Convert to DataFrame
    print(f"\n[3/3] Analyzing column completeness...")
    df = pd.DataFrame(all_rows, columns=column_names)

    # Calculate completeness for each column
    print("\n" + "="*80)
    print("COLUMN COMPLETENESS REPORT")
    print("="*80)
    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}\n")

    # Print ALL columns first
    print("\n" + "="*80)
    print("ALL COLUMNS (Complete List)")
    print("="*80)
    print(f"{'Position':<10} {'Column Name':<50} {'Completeness':<15}")
    print("-"*80)

    for i, col in enumerate(column_names):
        non_empty = df[col].apply(lambda x: x is not None and
                                             x != '' and
                                             str(x).lower() != 'none' and
                                             str(x).lower() != 'nan').sum()
        pct_complete = (non_empty / len(df)) * 100
        status = "✓" if pct_complete > 80 else ("⚠" if pct_complete > 50 else "✗")
        print(f"{status} {i:<8} {col:<50} {pct_complete:>6.2f}%")

    print(f"\n{'='*80}")

    # Group columns by category
    tcp_features = [col for col in df.columns if 'tcp' in col.lower()]
    ip_features = [col for col in df.columns if 'ip' in col.lower() or 'ttl' in col.lower()]
    flow_features = [col for col in df.columns if 'flow' in col.lower() or 'packet' in col.lower() or 'bytes' in col.lower()]
    metadata_features = ['flow_id', 'os_label', 'os_version', 'start_timestamp', 'end_timestamp']
    network_features = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'l3_proto', 'l4_proto']

    def analyze_columns(column_list, category_name):
        """Analyze and print completeness for a category of columns"""
        if not column_list:
            return

        print(f"\n{'='*80}")
        print(f"{category_name.upper()}")
        print(f"{'='*80}")
        print(f"{'Column Name':<50} {'Position':<10} {'Completeness':<15}")
        print("-"*80)

        results = []
        for col in column_list:
            if col not in df.columns:
                continue

            # Count non-empty values (not None, not '', not 'None', not NaN)
            non_empty = df[col].apply(lambda x: x is not None and
                                                 x != '' and
                                                 str(x).lower() != 'none' and
                                                 str(x).lower() != 'nan').sum()

            pct_complete = (non_empty / len(df)) * 100

            # Get position (index in column_names list)
            try:
                pos_str = str(column_names.index(col))
            except ValueError:
                pos_str = "?"

            results.append((col, pos_str, pct_complete))

        # Sort by completeness (descending)
        results.sort(key=lambda x: x[2], reverse=True)

        for col, pos, pct in results:
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"{status} {col:<48} {pos:<10} {pct:>6.2f}%")

    # Analyze by category
    analyze_columns(metadata_features, "METADATA & LABELS")
    analyze_columns(network_features, "NETWORK INFO")
    analyze_columns(tcp_features, "TCP FEATURES")
    analyze_columns(ip_features, "IP FEATURES")
    analyze_columns(flow_features, "FLOW STATISTICS")

    # All other fields (not in defined categories)
    categorized = set(metadata_features + network_features + tcp_features + ip_features + flow_features)
    other_fields = [col for col in df.columns if col not in categorized]

    if other_fields:
        print(f"\n{'='*80}")
        print("OTHER FIELDS (Not in defined categories)")
        print(f"{'='*80}")
        print(f"{'Column Name':<50} {'Position':<10} {'Completeness':<15}")
        print("-"*80)

        results = []
        for col in other_fields:
            non_empty = df[col].apply(lambda x: x is not None and
                                                 x != '' and
                                                 str(x).lower() != 'none' and
                                                 str(x).lower() != 'nan').sum()
            pct_complete = (non_empty / len(df)) * 100

            # Get position
            try:
                pos_str = str(column_names.index(col))
            except ValueError:
                pos_str = "?"

            if pct_complete > 10:  # Only show fields with >10% data
                results.append((col, pos_str, pct_complete))

        # Sort by position
        results.sort(key=lambda x: int(x[1]) if x[1].isdigit() else 999)

        for col, pos_str, pct_complete in results:
            status = "✓" if pct_complete > 80 else ("⚠" if pct_complete > 50 else "✗")
            print(f"{status} {col:<48} {pos_str:<10} {pct_complete:>6.2f}%")

    # Summary of critical OS fingerprinting features
    print("\n" + "="*80)
    print("CRITICAL OS FINGERPRINTING FEATURES AVAILABILITY")
    print("="*80)

    critical_features = {
        'tcp_syn_size': 'HIGH',
        'tcp_win_size': 'HIGH',
        'tcp_syn_ttl': 'HIGH',
        'tcp_option_window_scale_forward': 'HIGH',
        'tcp_option_window_scale_backward': 'HIGH',
        'tcp_option_maximum_segment_size_forward': 'HIGH',
        'tcp_option_maximum_segment_size_backward': 'HIGH',
        'tcp_option_selective_ack_permitted_forward': 'MEDIUM',
        'tcp_option_selective_ack_permitted_backward': 'MEDIUM',
        'maximum_ttl_forward': 'HIGH',
        'maximum_ttl_backward': 'HIGH',
        'ipv4_dont_fragment_forward': 'HIGH',
        'ipv4_dont_fragment_backward': 'HIGH',
        'tcp_timestamp_first_packet_forward': 'HIGH',
        'tcp_timestamp_first_packet_backward': 'HIGH',
        'ip_tos': 'MEDIUM',
        'l3_proto': 'MEDIUM',
        'l4_proto': 'HIGH',
        'tcp_flags_a': 'MEDIUM',
    }

    print(f"\n{'Feature':<50} {'Importance':<10} {'Available':<12}")
    print("-"*80)

    for feature, importance in critical_features.items():
        if feature in df.columns:
            non_empty = df[feature].apply(lambda x: x is not None and
                                                     x != '' and
                                                     str(x).lower() != 'none' and
                                                     str(x).lower() != 'nan').sum()
            pct_complete = (non_empty / len(df)) * 100
            status = "✓" if pct_complete > 80 else ("⚠" if pct_complete > 50 else "✗")
            print(f"{status} {feature:<48} {importance:<10} {pct_complete:>6.2f}%")
        else:
            print(f"✗ {feature:<48} {importance:<10} NOT FOUND")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nLegend:")
    print("  ✓ = >80% complete (good)")
    print("  ⚠ = 50-80% complete (usable but incomplete)")
    print("  ✗ = <50% complete (poor)")

    return df

if __name__ == '__main__':
    df = analyze_masaryk_completeness()
