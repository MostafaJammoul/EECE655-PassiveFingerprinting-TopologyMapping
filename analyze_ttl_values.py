#!/usr/bin/env python3
"""
Analyze TTL Values in Masaryk Dataset

This script analyzes TTL-related features to determine which ones
are most useful for OS fingerprinting:
- TCP SYN TTL (position 20): Captured TTL from SYN packet
- maximumTTLforward (position 92): Max TTL observed in forward direction
- maximumTTLbackward (position 93): Max TTL observed in backward direction
- initial_ttl (derived): Estimated original TTL (32, 64, 128, 255)

Usage:
    python analyze_ttl_values.py [--sample N]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def calculate_initial_ttl(ttl):
    """Estimate original TTL value based on observed TTL"""
    if pd.isna(ttl) or ttl is None:
        return None
    ttl = int(ttl)
    common_ttls = [32, 64, 128, 255]
    for initial in common_ttls:
        if ttl <= initial:
            return initial
    return 255

def extract_os_family(os_label):
    """Extract OS family from label"""
    os_lower = str(os_label).lower()

    if any(w in os_lower for w in ['windows', 'win10', 'win11', 'win7', 'win8', 'microsoft']):
        return 'Windows'
    elif any(w in os_lower for w in ['ubuntu', 'debian', 'fedora', 'centos', 'linux', 'kali', 'mint', 'arch', 'redhat']):
        return 'Linux'
    elif any(w in os_lower for w in ['macos', 'darwin', 'osx', 'mac']):
        return 'macOS'
    elif 'android' in os_lower:
        return 'Android'
    elif any(w in os_lower for w in ['ios', 'iphone', 'ipad']):
        return 'iOS'
    elif 'bsd' in os_lower:
        return 'BSD'
    else:
        return 'Other'

def analyze_ttl_values(csv_path='data/raw/masaryk/flows_ground_truth_merged_anonymized.csv',
                       sample_size=None):
    """
    Analyze TTL values in Masaryk dataset
    """

    print("="*80)
    print("MASARYK DATASET - TTL VALUE ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing: {csv_path}")
    if sample_size:
        print(f"Sample size: {sample_size:,} rows")
    print()

    csv_file = Path(csv_path)

    if not csv_file.exists():
        print(f"ERROR: File not found: {csv_path}")
        return None

    print("[1/3] Reading dataset...")

    # Read CSV and extract TTL columns
    records = []
    total_rows = 0

    with open(csv_file, 'r', encoding='utf-8') as f:
        # Skip header
        header = f.readline()

        for line_num, line in enumerate(f, 2):
            line = line.strip()
            if not line:
                continue

            fields = line.split(';')

            if len(fields) < 93:
                continue

            total_rows += 1

            # Extract OS label
            os_label = fields[1].strip() if len(fields) > 1 else 'Unknown'
            os_family = extract_os_family(os_label)

            # Extract TTL values
            try:
                tcp_syn_ttl = int(fields[20]) if len(fields) > 20 and fields[20] else None
            except (ValueError, TypeError):
                tcp_syn_ttl = None

            try:
                max_ttl_forward = int(fields[92]) if len(fields) > 92 and fields[92] else None
            except (ValueError, TypeError):
                max_ttl_forward = None

            try:
                max_ttl_backward = int(fields[93]) if len(fields) > 93 and fields[93] else None
            except (ValueError, TypeError):
                max_ttl_backward = None

            # Calculate derived initial TTL
            initial_ttl_from_syn = calculate_initial_ttl(tcp_syn_ttl)
            initial_ttl_from_forward = calculate_initial_ttl(max_ttl_forward)

            records.append({
                'os_family': os_family,
                'tcp_syn_ttl': tcp_syn_ttl,
                'max_ttl_forward': max_ttl_forward,
                'max_ttl_backward': max_ttl_backward,
                'initial_ttl_from_syn': initial_ttl_from_syn,
                'initial_ttl_from_forward': initial_ttl_from_forward,
            })

            # Progress
            if total_rows % 100000 == 0:
                print(f"  Processed {total_rows:,} rows...")

            # Sample limit
            if sample_size and total_rows >= sample_size:
                break

    print(f"  Total rows processed: {total_rows:,}\n")

    df = pd.DataFrame(records)

    print("[2/3] Computing statistics...")

    # Overall statistics
    print("\n" + "="*80)
    print("TTL FEATURE AVAILABILITY")
    print("="*80)

    ttl_features = {
        'tcp_syn_ttl': 'TCP SYN TTL (position 20)',
        'max_ttl_forward': 'Maximum TTL Forward (position 92)',
        'max_ttl_backward': 'Maximum TTL Backward (position 93)',
        'initial_ttl_from_syn': 'Initial TTL (derived from SYN)',
        'initial_ttl_from_forward': 'Initial TTL (derived from forward)',
    }

    for feat, desc in ttl_features.items():
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"{status} {desc:<50}: {pct:>6.2f}%")

    # TTL value distributions
    print("\n" + "="*80)
    print("TTL VALUE DISTRIBUTIONS")
    print("="*80)

    print("\n[TCP SYN TTL - Actual Captured Values]")
    if df['tcp_syn_ttl'].notna().any():
        print(df['tcp_syn_ttl'].value_counts().head(20))
        print(f"\nMin: {df['tcp_syn_ttl'].min()}, Max: {df['tcp_syn_ttl'].max()}, Mean: {df['tcp_syn_ttl'].mean():.1f}")

    print("\n[Maximum TTL Forward]")
    if df['max_ttl_forward'].notna().any():
        print(df['max_ttl_forward'].value_counts().head(20))
        print(f"\nMin: {df['max_ttl_forward'].min()}, Max: {df['max_ttl_forward'].max()}, Mean: {df['max_ttl_forward'].mean():.1f}")

    print("\n[Initial TTL - Derived from SYN]")
    if df['initial_ttl_from_syn'].notna().any():
        print(df['initial_ttl_from_syn'].value_counts().sort_index())

    print("\n[Initial TTL - Derived from Forward]")
    if df['initial_ttl_from_forward'].notna().any():
        print(df['initial_ttl_from_forward'].value_counts().sort_index())

    # Per-OS analysis
    print("\n[3/3] Per-OS Family Analysis...")

    print("\n" + "="*80)
    print("TTL VALUES BY OS FAMILY")
    print("="*80)

    for os_fam in ['Windows', 'Linux', 'macOS', 'Android', 'iOS']:
        os_df = df[df['os_family'] == os_fam]
        if len(os_df) == 0:
            continue

        print(f"\n{os_fam} (n={len(os_df):,})")
        print("-" * 80)

        # TCP SYN TTL
        if os_df['tcp_syn_ttl'].notna().any():
            print(f"  TCP SYN TTL:")
            print(f"    Range: {os_df['tcp_syn_ttl'].min()}-{os_df['tcp_syn_ttl'].max()}")
            print(f"    Mean: {os_df['tcp_syn_ttl'].mean():.1f}, Median: {os_df['tcp_syn_ttl'].median():.0f}")
            print(f"    Most common: {os_df['tcp_syn_ttl'].mode().values[0] if len(os_df['tcp_syn_ttl'].mode()) > 0 else 'N/A'}")

        # Maximum TTL Forward
        if os_df['max_ttl_forward'].notna().any():
            print(f"  Maximum TTL Forward:")
            print(f"    Range: {os_df['max_ttl_forward'].min()}-{os_df['max_ttl_forward'].max()}")
            print(f"    Mean: {os_df['max_ttl_forward'].mean():.1f}, Median: {os_df['max_ttl_forward'].median():.0f}")
            print(f"    Most common: {os_df['max_ttl_forward'].mode().values[0] if len(os_df['max_ttl_forward'].mode()) > 0 else 'N/A'}")

        # Initial TTL (derived from SYN)
        if os_df['initial_ttl_from_syn'].notna().any():
            print(f"  Initial TTL (from SYN):")
            print(f"    Distribution: {os_df['initial_ttl_from_syn'].value_counts().to_dict()}")

    # Feature comparison
    print("\n" + "="*80)
    print("FEATURE COMPARISON - Which is better?")
    print("="*80)

    # Check how often they differ
    df_valid = df[df['tcp_syn_ttl'].notna() & df['max_ttl_forward'].notna()].copy()

    if len(df_valid) > 0:
        print(f"\nRows with both tcp_syn_ttl and max_ttl_forward: {len(df_valid):,}")

        # Are they the same?
        same = (df_valid['tcp_syn_ttl'] == df_valid['max_ttl_forward']).sum()
        diff = len(df_valid) - same

        print(f"  Same value: {same:,} ({same/len(df_valid)*100:.1f}%)")
        print(f"  Different value: {diff:,} ({diff/len(df_valid)*100:.1f}%)")

        # When different, which is higher?
        df_diff = df_valid[df_valid['tcp_syn_ttl'] != df_valid['max_ttl_forward']]
        if len(df_diff) > 0:
            forward_higher = (df_diff['max_ttl_forward'] > df_diff['tcp_syn_ttl']).sum()
            syn_higher = (df_diff['tcp_syn_ttl'] > df_diff['max_ttl_forward']).sum()

            print(f"\n  When different:")
            print(f"    max_ttl_forward > tcp_syn_ttl: {forward_higher:,} ({forward_higher/len(df_diff)*100:.1f}%)")
            print(f"    tcp_syn_ttl > max_ttl_forward: {syn_higher:,} ({syn_higher/len(df_diff)*100:.1f}%)")

    # TTL Anomaly Detection
    print("\n" + "="*80)
    print("TTL ANOMALY DETECTION - Mismatches between OS and Expected TTL")
    print("="*80)

    # Expected initial TTL for each OS family
    expected_ttl = {
        'Windows': 128,
        'Linux': 64,
        'macOS': 64,
        'Android': 64,
        'iOS': 64,
        'BSD': 64,
        'Other': None  # No expectation for Other
    }

    print("\nExpected Initial TTL values:")
    print("  Windows  : 128")
    print("  Linux    : 64")
    print("  macOS    : 64")
    print("  Android  : 64")
    print("  iOS      : 64")
    print("  BSD      : 64")

    print("\n" + "-"*80)
    print("Anomalies Found:")
    print("-"*80)

    total_anomalies = 0

    for os_fam in ['Windows', 'Linux', 'macOS', 'Android', 'iOS', 'BSD']:
        os_df = df[df['os_family'] == os_fam].copy()
        if len(os_df) == 0:
            continue

        expected = expected_ttl[os_fam]

        # Check initial_ttl_from_syn
        os_df_with_ttl = os_df[os_df['initial_ttl_from_syn'].notna()]

        if len(os_df_with_ttl) > 0:
            anomalies = os_df_with_ttl[os_df_with_ttl['initial_ttl_from_syn'] != expected]
            anomaly_count = len(anomalies)
            anomaly_pct = (anomaly_count / len(os_df_with_ttl)) * 100

            if anomaly_count > 0:
                total_anomalies += anomaly_count
                print(f"\n{os_fam} (Expected TTL: {expected}):")
                print(f"  Total with TTL: {len(os_df_with_ttl):,}")
                print(f"  Anomalies: {anomaly_count:,} ({anomaly_pct:.2f}%)")

                # Show distribution of anomalous TTL values
                anomaly_ttl_dist = anomalies['initial_ttl_from_syn'].value_counts().sort_index()
                print(f"  Anomalous TTL distribution:")
                for ttl_val, count in anomaly_ttl_dist.items():
                    pct = (count / anomaly_count) * 100
                    print(f"    TTL {int(ttl_val)}: {count:,} flows ({pct:.1f}%)")

                # Show a few example raw TTL values for the most common anomaly
                if len(anomalies) > 0:
                    most_common_anomaly = anomaly_ttl_dist.index[0]
                    examples = anomalies[anomalies['initial_ttl_from_syn'] == most_common_anomaly]['tcp_syn_ttl'].dropna().head(10)
                    if len(examples) > 0:
                        print(f"  Example raw TTL values for initial_ttl={int(most_common_anomaly)}: {examples.tolist()}")

    if total_anomalies == 0:
        print("\n✓ No anomalies detected! All OS families have expected initial TTL values.")
    else:
        print(f"\n⚠ Total anomalies across all OS families: {total_anomalies:,}")
        print("\nPossible explanations for anomalies:")
        print("  1. Misclassified OS (ground truth error)")
        print("  2. Virtual machines / containers (may have different TTL)")
        print("  3. Network appliances / proxies altering packets")
        print("  4. Mobile devices with custom kernels")
        print("  5. Very old/embedded systems with non-standard defaults")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    print("\nBased on the analysis:")
    print("\n1. **tcp_syn_ttl (position 20)** - Raw captured TTL from SYN packet")
    print("   - Pro: Direct measurement, 100% available")
    print("   - Pro: Captures actual TTL after hops (contains distance info)")
    print("   - Con: Varies by network distance")

    print("\n2. **max_ttl_forward (position 92)** - Maximum TTL in forward direction")
    print("   - Pro: May be closer to original TTL")
    print("   - Con: Availability depends on flow data")

    print("\n3. **initial_ttl (derived)** - Estimated original TTL (32/64/128/255)")
    print("   - Pro: Normalizes for network distance")
    print("   - Pro: OS-specific (Windows=128, Linux/Mac=64)")
    print("   - Con: Loses granularity (only 4 values)")

    print("\n4. **Keep both tcp_syn_ttl AND initial_ttl?**")
    print("   - Pro: Model can learn which is more important")
    print("   - Pro: tcp_syn_ttl = raw value, initial_ttl = normalized")
    print("   - Con: Some correlation between them")

    return df

def main():
    parser = argparse.ArgumentParser(
        description='Analyze TTL values in Masaryk dataset for OS fingerprinting'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/masaryk/flows_ground_truth_merged_anonymized.csv',
        help='Path to Masaryk CSV file'
    )

    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample size (for quick analysis)'
    )

    args = parser.parse_args()

    df = analyze_ttl_values(args.input, args.sample)

    if df is not None:
        print("\n✓ Analysis complete!")

if __name__ == '__main__':
    main()
