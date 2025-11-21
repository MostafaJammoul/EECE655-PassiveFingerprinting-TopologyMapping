#!/usr/bin/env python3
"""
Calculate Inter-Packet Timing (IPT) Features

This script takes a preprocessed CSV file with packet-level data and calculates
Inter-Packet Timing (IPT) features, which are critical for behavioral fingerprinting.

IPT Features:
- ipt: Time delta to next packet in same flow (seconds)
- ipt_mean: Mean IPT for the flow
- ipt_std: Standard deviation of IPT for the flow
- ipt_min: Minimum IPT in the flow
- ipt_max: Maximum IPT in the flow

Different OSes have different timing patterns:
- Linux: ~200ms delayed ACK, 1ms clock granularity
- Windows: ~40ms delayed ACK, 100ms clock granularity
- macOS: ~100ms delayed ACK, 10ms clock granularity

Usage:
    python calculate_ipt.py input.csv output.csv
    python calculate_ipt.py data/processed/cesnet.csv data/processed/cesnet_ipt.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def calculate_flow_ipt(df, verbose=True):
    """
    Calculate Inter-Packet Timing (IPT) features for each flow

    Args:
        df: DataFrame with packet-level data (must have timestamp, src_ip, dst_ip, src_port, dst_port)
        verbose: Print progress

    Returns:
        DataFrame with IPT features added
    """

    if verbose:
        print("="*70)
        print("INTER-PACKET TIMING (IPT) CALCULATOR")
        print("="*70)
        print(f"\nInput shape: {df.shape}")
        print(f"  Records: {len(df):,}")

    # Verify required columns
    required_cols = ['timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"\nERROR: Missing required columns: {missing_cols}")
        print(f"Required columns: {required_cols}")
        return None

    # Create flow identifier (5-tuple)
    # For bidirectional flows, normalize the tuple (smaller IP first)
    if verbose:
        print(f"\n[1/4] Creating flow identifiers...")

    def create_flow_id(row):
        """Create normalized 5-tuple flow identifier"""
        # Normalize to ensure bidirectional matching
        # (src_ip, src_port) should always be "smaller" than (dst_ip, dst_port)
        if (row['src_ip'], row['src_port']) < (row['dst_ip'], row['dst_port']):
            return f"{row['src_ip']}:{row['src_port']}-{row['dst_ip']}:{row['dst_port']}"
        else:
            return f"{row['dst_ip']}:{row['dst_port']}-{row['src_ip']}:{row['src_port']}"

    df['flow_id'] = df.apply(create_flow_id, axis=1)

    if verbose:
        num_flows = df['flow_id'].nunique()
        print(f"  Found {num_flows:,} unique flows")
        avg_pkts_per_flow = len(df) / num_flows
        print(f"  Average packets per flow: {avg_pkts_per_flow:.2f}")

    # Sort by flow and timestamp
    if verbose:
        print(f"\n[2/4] Sorting packets by flow and timestamp...")

    df = df.sort_values(['flow_id', 'timestamp'])

    # Calculate IPT (time delta to next packet in same flow)
    if verbose:
        print(f"\n[3/4] Calculating inter-packet timing...")

    # Within each flow, calculate time delta to next packet
    df['ipt'] = df.groupby('flow_id')['timestamp'].diff()

    # Move forward diff (time to NEXT packet, not previous)
    df['ipt_next'] = df.groupby('flow_id')['timestamp'].diff(-1).abs()

    # Statistics: For each packet, calculate flow-level IPT statistics
    # (These will be the same for all packets in a flow, but useful for ML models)
    if verbose:
        print(f"\n[4/4] Calculating flow-level IPT statistics...")

    flow_stats = df.groupby('flow_id')['ipt'].agg([
        ('ipt_mean', 'mean'),
        ('ipt_std', 'std'),
        ('ipt_min', 'min'),
        ('ipt_max', 'max'),
        ('ipt_median', 'median'),
    ]).reset_index()

    # Merge back to original dataframe
    df = df.merge(flow_stats, on='flow_id', how='left')

    # Fill NaN values (first packet in flow has no previous packet)
    df['ipt'].fillna(0, inplace=True)
    df['ipt_next'].fillna(0, inplace=True)
    df['ipt_std'].fillna(0, inplace=True)  # Single-packet flows have 0 std

    # Summary statistics
    if verbose:
        print("\n" + "="*70)
        print("IPT CALCULATION COMPLETE")
        print("="*70)

        print(f"\nNew features added:")
        print(f"  • ipt: Time to previous packet in flow (seconds)")
        print(f"  • ipt_next: Time to next packet in flow (seconds)")
        print(f"  • ipt_mean: Mean IPT for the flow")
        print(f"  • ipt_std: Standard deviation of IPT")
        print(f"  • ipt_min: Minimum IPT in flow")
        print(f"  • ipt_max: Maximum IPT in flow")
        print(f"  • ipt_median: Median IPT in flow")

        print(f"\nIPT Statistics:")
        print(f"  Mean IPT:   {df['ipt'].mean():.6f} seconds ({df['ipt'].mean() * 1000:.2f} ms)")
        print(f"  Median IPT: {df['ipt'].median():.6f} seconds ({df['ipt'].median() * 1000:.2f} ms)")
        print(f"  Std IPT:    {df['ipt'].std():.6f} seconds ({df['ipt'].std() * 1000:.2f} ms)")
        print(f"  Min IPT:    {df['ipt'].min():.6f} seconds ({df['ipt'].min() * 1000:.2f} ms)")
        print(f"  Max IPT:    {df['ipt'].max():.6f} seconds ({df['ipt'].max() * 1000:.2f} ms)")

        # Show distribution of IPT ranges
        print(f"\nIPT Distribution:")
        print(f"  < 1ms:      {(df['ipt'] < 0.001).sum():>10,} packets ({(df['ipt'] < 0.001).sum() / len(df) * 100:>5.1f}%)")
        print(f"  1-10ms:     {((df['ipt'] >= 0.001) & (df['ipt'] < 0.010)).sum():>10,} packets ({((df['ipt'] >= 0.001) & (df['ipt'] < 0.010)).sum() / len(df) * 100:>5.1f}%)")
        print(f"  10-100ms:   {((df['ipt'] >= 0.010) & (df['ipt'] < 0.100)).sum():>10,} packets ({((df['ipt'] >= 0.010) & (df['ipt'] < 0.100)).sum() / len(df) * 100:>5.1f}%)")
        print(f"  100-500ms:  {((df['ipt'] >= 0.100) & (df['ipt'] < 0.500)).sum():>10,} packets ({((df['ipt'] >= 0.100) & (df['ipt'] < 0.500)).sum() / len(df) * 100:>5.1f}%)")
        print(f"  > 500ms:    {(df['ipt'] >= 0.500).sum():>10,} packets ({(df['ipt'] >= 0.500).sum() / len(df) * 100:>5.1f}%)")

        # OS-specific patterns
        print(f"\nExpected OS Patterns:")
        print(f"  Linux:   ~200ms delayed ACK, 1ms clock granularity")
        print(f"  Windows: ~40ms delayed ACK, 100ms clock granularity")
        print(f"  macOS:   ~100ms delayed ACK, 10ms clock granularity")

    # Drop temporary flow_id column
    df = df.drop(columns=['flow_id'])

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Inter-Packet Timing (IPT) features for packet-level datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python calculate_ipt.py input.csv output.csv

  # Process CESNET dataset
  python calculate_ipt.py data/processed/cesnet.csv data/processed/cesnet_ipt.csv

  # Process nprint dataset
  python calculate_ipt.py data/processed/nprint.csv data/processed/nprint_ipt.csv

  # In-place update (overwrite input file)
  python calculate_ipt.py data/processed/cesnet.csv data/processed/cesnet.csv

Output:
  The output CSV will contain all original columns plus:
  - ipt: Time delta to previous packet in flow (seconds)
  - ipt_next: Time delta to next packet in flow (seconds)
  - ipt_mean, ipt_std, ipt_min, ipt_max, ipt_median: Flow-level statistics
        """
    )

    parser.add_argument('input', type=str, help='Input CSV file (preprocessed packet data)')
    parser.add_argument('output', type=str, help='Output CSV file (with IPT features)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print(f"\nPlease run preprocessing first:")
        print(f"  python preprocess_scripts/cesnet_preprocess.py")
        print(f"  python preprocess_scripts/nprint_preprocess.py")
        sys.exit(1)

    # Read input
    if not args.quiet:
        print(f"\nReading input file: {args.input}")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        sys.exit(1)

    # Calculate IPT
    df_ipt = calculate_flow_ipt(df, verbose=not args.quiet)

    if df_ipt is None:
        print("\nERROR: IPT calculation failed!")
        sys.exit(1)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"\nSaving output to: {args.output}")

    df_ipt.to_csv(output_path, index=False)

    if not args.quiet:
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nOutput: {output_path}")
        print(f"Records: {len(df_ipt):,}")
        print(f"Features: {len(df_ipt.columns)} (added 7 IPT features)")

        print(f"\nIPT features are now ready for model training!")
        print(f"\nKey benefits:")
        print(f"  • Behavioral fingerprinting (timing patterns differ by OS)")
        print(f"  • Complements static features (window size, TTL, MSS)")
        print(f"  • Essential for ACK-based fingerprinting (CESNET dataset)")


if __name__ == '__main__':
    main()
