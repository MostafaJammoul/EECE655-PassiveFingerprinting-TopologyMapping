#!/usr/bin/env python3
"""
Analyze extracted flow statistics from nprint PCAP
Provides debugging statistics on bidirectional traffic, TCP flags, etc.
"""

import pandas as pd
import sys
import argparse


def analyze_flow_stats(csv_path):
    """Analyze flow statistics from extracted CSV"""

    print("="*70)
    print("FLOW STATISTICS ANALYSIS")
    print("="*70)
    print(f"\nReading CSV: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    total_flows = len(df)
    print(f"Total flows: {total_flows:,}")

    print("\n" + "="*70)
    print("BIDIRECTIONAL TRAFFIC ANALYSIS")
    print("="*70)

    # Count flows by backward packet status
    zero_backward = (df['packet_total_count_backward'] == 0).sum()
    has_backward = (df['packet_total_count_backward'] > 0).sum()

    print(f"\nBackward packet counts:")
    print(f"  Flows with 0 backward packets:  {zero_backward:,} ({zero_backward/total_flows*100:.1f}%)")
    print(f"  Flows with >0 backward packets: {has_backward:,} ({has_backward/total_flows*100:.1f}%)")

    # Count flows by forward packet status
    zero_forward = (df['packet_total_count_forward'] == 0).sum()
    has_forward = (df['packet_total_count_forward'] > 0).sum()

    print(f"\nForward packet counts:")
    print(f"  Flows with 0 forward packets:  {zero_forward:,} ({zero_forward/total_flows*100:.1f}%)")
    print(f"  Flows with >0 forward packets: {has_forward:,} ({has_forward/total_flows*100:.1f}%)")

    # Count bidirectional vs unidirectional
    bidirectional = ((df['packet_total_count_forward'] > 0) &
                     (df['packet_total_count_backward'] > 0)).sum()
    unidirectional = ((df['packet_total_count_forward'] == 0) |
                      (df['packet_total_count_backward'] == 0)).sum()

    print(f"\nFlow directionality:")
    print(f"  Bidirectional flows (both >0):  {bidirectional:,} ({bidirectional/total_flows*100:.1f}%)")
    print(f"  Unidirectional flows (one = 0): {unidirectional:,} ({unidirectional/total_flows*100:.1f}%)")

    # Show distribution of backward packet counts
    print(f"\nBackward packet distribution:")
    backward_dist = df['packet_total_count_backward'].value_counts().sort_index().head(10)
    for count, freq in backward_dist.items():
        print(f"  {int(count)} packets: {freq:,} flows ({freq/total_flows*100:.1f}%)")

    print("\n" + "="*70)
    print("TCP HANDSHAKE ANALYSIS")
    print("="*70)

    # SYN-ACK flag analysis
    has_synack = (df['syn_ack_flag'] == 1).sum()
    no_synack = (df['syn_ack_flag'] == 0).sum()

    print(f"\nSYN-ACK flag status:")
    print(f"  syn_ack_flag = 1: {has_synack:,} ({has_synack/total_flows*100:.1f}%)")
    print(f"  syn_ack_flag = 0: {no_synack:,} ({no_synack/total_flows*100:.1f}%)")

    # Cross-check: syn_ack_flag vs backward packets
    synack_with_backward = ((df['syn_ack_flag'] == 1) &
                            (df['packet_total_count_backward'] > 0)).sum()
    synack_without_backward = ((df['syn_ack_flag'] == 1) &
                               (df['packet_total_count_backward'] == 0)).sum()
    no_synack_with_backward = ((df['syn_ack_flag'] == 0) &
                               (df['packet_total_count_backward'] > 0)).sum()

    print(f"\nSYN-ACK vs backward packet consistency:")
    print(f"  syn_ack=1 AND backward>0:  {synack_with_backward:,} ✓ CONSISTENT")
    print(f"  syn_ack=1 AND backward=0:  {synack_without_backward:,} ❌ INCONSISTENT!")
    print(f"  syn_ack=0 AND backward>0:  {no_synack_with_backward:,} ⚠️  UNUSUAL")
    print(f"  syn_ack=0 AND backward=0:  {no_synack - no_synack_with_backward:,} ✓ CONSISTENT")

    if synack_without_backward > 0:
        print(f"\n  ⚠️  WARNING: {synack_without_backward} flows claim SYN-ACK but have 0 backward packets!")
        print(f"      This is IMPOSSIBLE - SYN-ACK is a backward packet!")

    print("\n" + "="*70)
    print("TCP FLAGS ANALYSIS")
    print("="*70)

    # Analyze TCP flags patterns
    flag_counts = df['tcp_flags_a'].value_counts().head(10)
    print(f"\nTop 10 TCP flag combinations:")
    for flags, count in flag_counts.items():
        print(f"  {flags}: {count:,} ({count/total_flows*100:.1f}%)")

    # Count specific flag patterns
    syn_only = df['tcp_flags_a'].str.contains('------S-', na=False).sum()
    has_ack = df['tcp_flags_a'].str.contains('A', na=False).sum()
    has_psh = df['tcp_flags_a'].str.contains('P', na=False).sum()
    has_fin = df['tcp_flags_a'].str.contains('F', na=False).sum()

    print(f"\nFlag presence:")
    print(f"  Only SYN (------S-): {syn_only:,} ({syn_only/total_flows*100:.1f}%)")
    print(f"  Has ACK flag:        {has_ack:,} ({has_ack/total_flows*100:.1f}%)")
    print(f"  Has PSH flag:        {has_psh:,} ({has_psh/total_flows*100:.1f}%)")
    print(f"  Has FIN flag:        {has_fin:,} ({has_fin/total_flows*100:.1f}%)")

    print("\n" + "="*70)
    print("OS DISTRIBUTION")
    print("="*70)

    os_dist = df['os_family'].value_counts()
    print(f"\nOS family distribution:")
    for os_name, count in os_dist.items():
        print(f"  {os_name}: {count:,} ({count/total_flows*100:.1f}%)")

    # Per-OS backward packet analysis
    print(f"\nBackward packets by OS:")
    for os_name in os_dist.index:
        os_df = df[df['os_family'] == os_name]
        os_has_backward = (os_df['packet_total_count_backward'] > 0).sum()
        os_total = len(os_df)
        print(f"  {os_name}: {os_has_backward}/{os_total} ({os_has_backward/os_total*100:.1f}%) have backward packets")

    print("\n" + "="*70)
    print("TLS FEATURES")
    print("="*70)

    has_tls = df['tls_ja3_fingerprint'].notna().sum()
    no_tls = df['tls_ja3_fingerprint'].isna().sum()

    print(f"\nTLS fingerprint availability:")
    print(f"  Flows with TLS:    {has_tls:,} ({has_tls/total_flows*100:.1f}%)")
    print(f"  Flows without TLS: {no_tls:,} ({no_tls/total_flows*100:.1f}%)")

    print("\n" + "="*70)
    print("PACKET COUNT STATISTICS")
    print("="*70)

    print(f"\nForward packet statistics:")
    print(f"  Mean:   {df['packet_total_count_forward'].mean():.1f}")
    print(f"  Median: {df['packet_total_count_forward'].median():.1f}")
    print(f"  Min:    {df['packet_total_count_forward'].min()}")
    print(f"  Max:    {df['packet_total_count_forward'].max()}")

    print(f"\nBackward packet statistics:")
    print(f"  Mean:   {df['packet_total_count_backward'].mean():.1f}")
    print(f"  Median: {df['packet_total_count_backward'].median():.1f}")
    print(f"  Min:    {df['packet_total_count_backward'].min()}")
    print(f"  Max:    {df['packet_total_count_backward'].max()}")

    print("\n" + "="*70)
    print("SAMPLE FLOWS")
    print("="*70)

    # Show sample flows with backward packets (if any)
    if has_backward > 0:
        print(f"\nSample flows WITH backward packets:")
        backward_samples = df[df['packet_total_count_backward'] > 0].head(3)
        for idx, row in backward_samples.iterrows():
            print(f"\n  Flow {idx}:")
            print(f"    OS: {row['os_family']}")
            print(f"    TCP flags: {row['tcp_flags_a']}")
            print(f"    syn_ack_flag: {row['syn_ack_flag']}")
            print(f"    Forward packets: {row['packet_total_count_forward']}")
            print(f"    Backward packets: {row['packet_total_count_backward']}")
            print(f"    Total bytes: {row['total_bytes']:.0f}")

    # Show sample flows without backward packets
    if zero_backward > 0:
        print(f"\nSample flows WITHOUT backward packets:")
        no_backward_samples = df[df['packet_total_count_backward'] == 0].head(3)
        for idx, row in no_backward_samples.iterrows():
            print(f"\n  Flow {idx}:")
            print(f"    OS: {row['os_family']}")
            print(f"    TCP flags: {row['tcp_flags_a']}")
            print(f"    syn_ack_flag: {row['syn_ack_flag']}")
            print(f"    Forward packets: {row['packet_total_count_forward']}")
            print(f"    Backward packets: {row['packet_total_count_backward']}")
            print(f"    Total bytes: {row['total_bytes']:.0f}")

    print("\n" + "="*70)
    print("SUMMARY & DIAGNOSIS")
    print("="*70)

    print()

    # Diagnose issues
    issues = []

    if zero_backward / total_flows > 0.5:
        issues.append(f"⚠️  ISSUE: {zero_backward/total_flows*100:.1f}% of flows have NO backward packets")
        issues.append(f"    Possible causes:")
        issues.append(f"    - Flow aggregation bug (reverse packets not grouped)")
        issues.append(f"    - Direction assignment bug (all packets marked forward)")
        issues.append(f"    - Asymmetric capture (PCAP only has outbound traffic)")

    if synack_without_backward > 0:
        issues.append(f"❌ CRITICAL: {synack_without_backward} flows have syn_ack_flag=1 with backward=0")
        issues.append(f"    This is IMPOSSIBLE - indicates a bug in direction assignment")

    if bidirectional / total_flows < 0.5:
        issues.append(f"⚠️  ISSUE: Only {bidirectional/total_flows*100:.1f}% of flows are bidirectional")
        issues.append(f"    TCP requires 3-way handshake - should have backward packets")

    if syn_only / total_flows > 0.1:
        issues.append(f"⚠️  ISSUE: {syn_only/total_flows*100:.1f}% of flows show only SYN flag")
        issues.append(f"    Indicates incomplete flows (no ACK, PSH, FIN)")

    if issues:
        print("ISSUES DETECTED:")
        for issue in issues:
            print(issue)
    else:
        print("✓ No major issues detected!")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze flow statistics from extracted PCAP CSV'
    )

    parser.add_argument(
        'csv',
        type=str,
        help='Path to CSV file with extracted flows'
    )

    args = parser.parse_args()

    analyze_flow_stats(args.csv)


if __name__ == '__main__':
    main()
