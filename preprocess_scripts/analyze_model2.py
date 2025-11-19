#!/usr/bin/env python3
"""
Analyze model2.csv Dataset

Shows:
1. Row count for each OS (both family and version)
2. Check if there are any packets that are not TCP SYN
"""

import pandas as pd
import sys
from pathlib import Path


def analyze_dataset(csv_path='model2.csv'):
    """Analyze the model2.csv dataset"""

    print("="*70)
    print("MODEL2.CSV DATASET ANALYSIS")
    print("="*70)

    # Check if file exists
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"\nERROR: File not found: {csv_path}")
        print(f"Please provide the correct path to model2.csv")
        return None

    # Load dataset
    print(f"\nLoading: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df):,} rows with {len(df.columns)} columns")
    except Exception as e:
        print(f"\nERROR loading CSV: {e}")
        return None

    # Show first few rows
    print(f"\nFirst 3 rows (sample):")
    print(df.head(3))

    # ========================================================================
    # PART 1: OS DISTRIBUTION
    # ========================================================================

    print("\n" + "="*70)
    print("1. OS DISTRIBUTION")
    print("="*70)

    # Check if OS columns exist
    if 'os_family' in df.columns:
        print(f"\nOS Family Distribution:")
        family_counts = df['os_family'].value_counts()
        print("-" * 40)
        for os_family, count in family_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {os_family:20s}: {count:8,} ({pct:5.2f}%)")
        print("-" * 40)
        print(f"  {'TOTAL':20s}: {len(df):8,}")

        # Check for missing/null values
        null_family = df['os_family'].isna().sum()
        if null_family > 0:
            print(f"\n  WARNING: {null_family:,} rows with missing OS family")
    else:
        print("\n  WARNING: 'os_family' column not found!")

    if 'os_label' in df.columns:
        print(f"\n\nOS Version Distribution (Full Labels):")
        label_counts = df['os_label'].value_counts()
        print("-" * 40)
        for os_label, count in label_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {str(os_label)[:30]:30s}: {count:8,} ({pct:5.2f}%)")
        print("-" * 40)
        print(f"  {'TOTAL':30s}: {len(df):8,}")

        # Check for missing/null values
        null_label = df['os_label'].isna().sum()
        if null_label > 0:
            print(f"\n  WARNING: {null_label:,} rows with missing OS label")
    else:
        print("\n  WARNING: 'os_label' column not found!")

    # ========================================================================
    # PART 2: TCP SYN CHECK
    # ========================================================================

    print("\n" + "="*70)
    print("2. TCP SYN PACKET CHECK")
    print("="*70)

    non_tcp_syn_count = 0
    non_tcp_count = 0
    non_syn_count = 0
    issues_found = []

    # Check protocol
    if 'protocol' in df.columns:
        print(f"\nProtocol Distribution:")
        protocol_counts = df['protocol'].value_counts()
        for proto, count in protocol_counts.items():
            proto_name = "TCP" if proto == 6 else "UDP" if proto == 17 else "Other"
            pct = (count / len(df)) * 100
            print(f"  Protocol {proto} ({proto_name}): {count:,} ({pct:.2f}%)")

        # Count non-TCP packets
        non_tcp_count = (df['protocol'] != 6).sum()
        if non_tcp_count > 0:
            non_tcp_pct = (non_tcp_count / len(df)) * 100
            issues_found.append(f"Found {non_tcp_count:,} non-TCP packets ({non_tcp_pct:.2f}%)")
    else:
        print("\n  WARNING: 'protocol' column not found!")

    # Check TCP flags
    if 'tcp_flags' in df.columns:
        print(f"\nTCP Flags Analysis:")

        # Check for rows with tcp_flags data
        has_flags = df['tcp_flags'].notna()
        print(f"  Rows with tcp_flags: {has_flags.sum():,} / {len(df):,}")

        if has_flags.sum() > 0:
            # For rows with flags, check if SYN bit is set
            # SYN flag is bit 1 (0x02)
            tcp_df = df[has_flags].copy()
            tcp_df['has_syn'] = tcp_df['tcp_flags'].apply(
                lambda x: bool(int(x) & 0x02) if pd.notna(x) else False
            )

            syn_count = tcp_df['has_syn'].sum()
            no_syn_count = (~tcp_df['has_syn']).sum()

            print(f"  Packets WITH SYN flag:    {syn_count:,}")
            print(f"  Packets WITHOUT SYN flag: {no_syn_count:,}")

            if no_syn_count > 0:
                no_syn_pct = (no_syn_count / len(tcp_df)) * 100
                issues_found.append(f"Found {no_syn_count:,} TCP packets without SYN flag ({no_syn_pct:.2f}% of packets with flag data)")

                # Show some examples
                print(f"\n  Sample of non-SYN packets (first 5):")
                non_syn_samples = tcp_df[~tcp_df['has_syn']].head(5)
                for idx, row in non_syn_samples.iterrows():
                    flags = int(row['tcp_flags']) if pd.notna(row['tcp_flags']) else 0
                    flag_str = []
                    if flags & 0x01: flag_str.append("FIN")
                    if flags & 0x02: flag_str.append("SYN")
                    if flags & 0x04: flag_str.append("RST")
                    if flags & 0x08: flag_str.append("PSH")
                    if flags & 0x10: flag_str.append("ACK")
                    if flags & 0x20: flag_str.append("URG")
                    flag_desc = ",".join(flag_str) if flag_str else "None"
                    print(f"    Row {idx}: flags=0x{flags:02x} ({flag_desc})")
        else:
            print(f"  WARNING: No tcp_flags data available!")
            issues_found.append("No TCP flags data available to verify SYN packets")
    else:
        print("\n  WARNING: 'tcp_flags' column not found!")
        issues_found.append("'tcp_flags' column not found")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nTotal rows: {len(df):,}")

    if issues_found:
        print(f"\n⚠ ISSUES FOUND:")
        for issue in issues_found:
            print(f"  - {issue}")
        print(f"\nDataset may contain packets that are NOT TCP SYN!")
    else:
        print(f"\n✓ All packets appear to be TCP SYN packets (as expected)")

    print("\n" + "="*70)

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze model2.csv dataset'
    )
    parser.add_argument(
        'csv_path',
        nargs='?',
        default='model2.csv',
        help='Path to model2.csv file (default: model2.csv in current directory)'
    )

    args = parser.parse_args()

    df = analyze_dataset(args.csv_path)

    if df is None:
        sys.exit(1)
