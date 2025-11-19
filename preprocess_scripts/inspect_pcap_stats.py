#!/usr/bin/env python3
"""
Diagnostic Script: Inspect PCAP Files Statistics

This script helps diagnose why CESNET idle dataset might have low packet counts.
It provides detailed statistics about PCAP files including:
- Total packets, TCP packets, SYN packets
- OS distribution
- File sizes
- Packet type breakdown
"""

import os
import sys
from pathlib import Path
import argparse

# Scapy for PCAP parsing
try:
    from scapy.all import rdpcap, IP, TCP
except ImportError:
    print("ERROR: Scapy not installed. Install with: pip install scapy")
    sys.exit(1)

# Tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    print("WARNING: tqdm not installed. Install for progress bars: pip install tqdm")
    def tqdm(iterable, **kwargs):
        return iterable


def analyze_pcap_file(pcap_path, quick=False):
    """
    Analyze a single PCAP file and return statistics

    Args:
        pcap_path: Path to PCAP file
        quick: If True, only read first 1000 packets for speed

    Returns:
        Dictionary with statistics
    """
    stats = {
        'file_path': str(pcap_path),
        'file_name': pcap_path.name,
        'file_size_mb': pcap_path.stat().st_size / (1024 * 1024),
        'total_packets': 0,
        'tcp_packets': 0,
        'udp_packets': 0,
        'icmp_packets': 0,
        'other_packets': 0,
        'syn_packets': 0,
        'syn_ack_packets': 0,
        'ack_packets': 0,
        'fin_packets': 0,
        'rst_packets': 0,
        'psh_packets': 0,
        'error': None
    }

    try:
        packets = rdpcap(str(pcap_path))
        stats['total_packets'] = len(packets)

        # Limit for quick analysis
        if quick and len(packets) > 1000:
            packets = packets[:1000]
            stats['sampled'] = True

        for packet in packets:
            # Check for IP layer
            if not packet.haslayer(IP):
                stats['other_packets'] += 1
                continue

            ip_layer = packet[IP]

            # Check protocol
            if packet.haslayer(TCP):
                stats['tcp_packets'] += 1
                tcp_layer = packet[TCP]

                # Analyze TCP flags
                is_syn = (tcp_layer.flags & 0x02) != 0
                is_ack = (tcp_layer.flags & 0x10) != 0
                is_fin = (tcp_layer.flags & 0x01) != 0
                is_rst = (tcp_layer.flags & 0x04) != 0
                is_psh = (tcp_layer.flags & 0x08) != 0

                if is_syn and is_ack:
                    stats['syn_ack_packets'] += 1
                elif is_syn:
                    stats['syn_packets'] += 1

                if is_ack:
                    stats['ack_packets'] += 1
                if is_fin:
                    stats['fin_packets'] += 1
                if is_rst:
                    stats['rst_packets'] += 1
                if is_psh:
                    stats['psh_packets'] += 1

            elif ip_layer.proto == 17:  # UDP
                stats['udp_packets'] += 1
            elif ip_layer.proto == 1:  # ICMP
                stats['icmp_packets'] += 1
            else:
                stats['other_packets'] += 1

    except Exception as e:
        stats['error'] = str(e)

    return stats


def discover_pcap_files(raw_dir):
    """
    Discover all PCAP files in directory structure

    Returns:
        List of (pcap_path, os_label_dir) tuples
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        print(f"ERROR: Directory not found: {raw_dir}")
        return []

    pcap_files = []

    # Look for PCAP files in subdirectories
    for subdir in raw_path.iterdir():
        if not subdir.is_dir():
            continue

        os_label_dir = subdir.name

        # Find all PCAP files in this subdirectory (recursive)
        for pcap_file in subdir.glob('**/*.pcap'):
            pcap_files.append((pcap_file, os_label_dir))

    # Also check for PCAP files directly in raw_dir
    for pcap_file in raw_path.glob('*.pcap'):
        pcap_files.append((pcap_file, 'root'))

    return pcap_files


def print_header(text, char='='):
    """Print a formatted header"""
    print(f"\n{char * 70}")
    print(text)
    print(f"{char * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic tool to inspect PCAP files and understand packet distribution'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/cesnet_idle',
        help='Input directory with PCAP files'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: only analyze first 1000 packets per file'
    )

    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Limit number of files to analyze (for testing)'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show per-file statistics'
    )

    args = parser.parse_args()

    print_header("PCAP FILE STATISTICS ANALYZER")

    print(f"Scanning directory: {args.input}")
    if args.quick:
        print("Mode: Quick (first 1000 packets per file)")
    else:
        print("Mode: Full analysis")

    # Discover files
    print("\n[1/2] Discovering PCAP files...")
    pcap_files = discover_pcap_files(args.input)

    if not pcap_files:
        print(f"\nERROR: No PCAP files found in {args.input}")
        print("\nExpected structure:")
        print("  data/raw/cesnet_idle/")
        print("  ├── os_name_1/")
        print("  │   └── *.pcap")
        print("  └── os_name_2/")
        print("      └── *.pcap")
        return

    print(f"  Found {len(pcap_files)} PCAP files")

    # Show OS distribution
    os_labels = [label for _, label in pcap_files]
    from collections import Counter
    os_counts = Counter(os_labels)
    print(f"\n  OS directory distribution:")
    for os_name, count in sorted(os_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {os_name}: {count} files")
    if len(os_counts) > 10:
        print(f"    ... and {len(os_counts) - 10} more")

    # Limit if requested
    if args.max_files:
        pcap_files = pcap_files[:args.max_files]
        print(f"\n  Limited to {args.max_files} files")

    # Analyze each file
    print("\n[2/2] Analyzing PCAP files...")
    all_stats = []
    total_file_size = 0

    for pcap_path, os_label in tqdm(pcap_files, desc="Analyzing"):
        stats = analyze_pcap_file(pcap_path, quick=args.quick)
        stats['os_label'] = os_label
        all_stats.append(stats)
        total_file_size += stats['file_size_mb']

    # Aggregate statistics
    print_header("AGGREGATE STATISTICS")

    # Summary
    total_files = len(all_stats)
    files_with_errors = sum(1 for s in all_stats if s['error'])
    total_packets = sum(s['total_packets'] for s in all_stats)
    total_tcp = sum(s['tcp_packets'] for s in all_stats)
    total_syn = sum(s['syn_packets'] for s in all_stats)
    total_syn_ack = sum(s['syn_ack_packets'] for s in all_stats)
    total_ack = sum(s['ack_packets'] for s in all_stats)
    total_fin = sum(s['fin_packets'] for s in all_stats)
    total_rst = sum(s['rst_packets'] for s in all_stats)
    total_psh = sum(s['psh_packets'] for s in all_stats)
    total_udp = sum(s['udp_packets'] for s in all_stats)
    total_icmp = sum(s['icmp_packets'] for s in all_stats)
    total_other = sum(s['other_packets'] for s in all_stats)

    print(f"Files analyzed: {total_files}")
    print(f"Total file size: {total_file_size:.2f} MB")
    if files_with_errors > 0:
        print(f"Files with errors: {files_with_errors}")
    if any(s.get('sampled') for s in all_stats):
        print("Note: Some files sampled (quick mode)")

    print(f"\nTotal packets: {total_packets:,}")
    print(f"  TCP packets: {total_tcp:,} ({100*total_tcp/total_packets if total_packets else 0:.1f}%)")
    print(f"  UDP packets: {total_udp:,} ({100*total_udp/total_packets if total_packets else 0:.1f}%)")
    print(f"  ICMP packets: {total_icmp:,} ({100*total_icmp/total_packets if total_packets else 0:.1f}%)")
    print(f"  Other packets: {total_other:,} ({100*total_other/total_packets if total_packets else 0:.1f}%)")

    print(f"\nTCP Flag Distribution:")
    print(f"  SYN packets: {total_syn:,} ({100*total_syn/total_tcp if total_tcp else 0:.1f}% of TCP)")
    print(f"  SYN+ACK packets: {total_syn_ack:,} ({100*total_syn_ack/total_tcp if total_tcp else 0:.1f}% of TCP)")
    print(f"  ACK packets: {total_ack:,} ({100*total_ack/total_tcp if total_tcp else 0:.1f}% of TCP)")
    print(f"  FIN packets: {total_fin:,} ({100*total_fin/total_tcp if total_tcp else 0:.1f}% of TCP)")
    print(f"  RST packets: {total_rst:,} ({100*total_rst/total_tcp if total_tcp else 0:.1f}% of TCP)")
    print(f"  PSH packets: {total_psh:,} ({100*total_psh/total_tcp if total_tcp else 0:.1f}% of TCP)")

    # Expected vs Actual for preprocessing
    total_syn_all = total_syn + total_syn_ack
    print(f"\n{'='*70}")
    print("EXPECTED PREPROCESSING OUTPUT")
    print(f"{'='*70}")
    print(f"\nWith SYN-only filtering (default):")
    print(f"  Expected records: {total_syn_all:,}")
    print(f"  (Includes SYN and SYN+ACK packets)")

    print(f"\nWith --all-tcp flag:")
    print(f"  Expected records: {total_tcp:,}")
    print(f"  (All TCP packets)")

    # Per-OS breakdown
    print(f"\n{'='*70}")
    print("PER-OS STATISTICS")
    print(f"{'='*70}")

    os_stats = {}
    for stat in all_stats:
        os_label = stat['os_label']
        if os_label not in os_stats:
            os_stats[os_label] = {
                'files': 0,
                'total_packets': 0,
                'tcp_packets': 0,
                'syn_packets': 0,
                'syn_ack_packets': 0
            }
        os_stats[os_label]['files'] += 1
        os_stats[os_label]['total_packets'] += stat['total_packets']
        os_stats[os_label]['tcp_packets'] += stat['tcp_packets']
        os_stats[os_label]['syn_packets'] += stat['syn_packets']
        os_stats[os_label]['syn_ack_packets'] += stat['syn_ack_packets']

    # Sort by SYN packets (most important for preprocessing)
    sorted_os = sorted(os_stats.items(),
                      key=lambda x: x[1]['syn_packets'] + x[1]['syn_ack_packets'],
                      reverse=True)

    for os_label, stats in sorted_os[:15]:  # Top 15
        syn_total = stats['syn_packets'] + stats['syn_ack_packets']
        print(f"\n{os_label}:")
        print(f"  Files: {stats['files']}")
        print(f"  Total packets: {stats['total_packets']:,}")
        print(f"  TCP packets: {stats['tcp_packets']:,}")
        print(f"  SYN/SYN+ACK: {syn_total:,} (usable for training)")

    if len(sorted_os) > 15:
        print(f"\n... and {len(sorted_os) - 15} more OS directories")

    # Detailed per-file statistics
    if args.detailed:
        print_header("PER-FILE STATISTICS", char='-')

        # Sort by SYN packets
        sorted_stats = sorted(all_stats,
                             key=lambda x: x['syn_packets'] + x['syn_ack_packets'],
                             reverse=True)

        for stat in sorted_stats[:20]:  # Top 20
            syn_total = stat['syn_packets'] + stat['syn_ack_packets']
            print(f"\n{stat['file_name']}:")
            print(f"  OS: {stat['os_label']}")
            print(f"  Size: {stat['file_size_mb']:.2f} MB")
            print(f"  Total: {stat['total_packets']:,} packets")
            print(f"  TCP: {stat['tcp_packets']:,}")
            print(f"  SYN/SYN+ACK: {syn_total:,}")
            if stat['error']:
                print(f"  ERROR: {stat['error']}")

    # Warnings and recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}\n")

    if total_syn_all < 10000:
        print("⚠️  WARNING: Low SYN packet count!")
        print("   Expected: 50K-200K for a good dataset")
        print(f"   Found: {total_syn_all:,}")
        print("\n   Possible causes:")
        print("   1. Dataset contains mostly established connections (ACK packets)")
        print("   2. PCAP files are incomplete or corrupted")
        print("   3. Not all PCAP files were extracted from the ZIP")
        print("\n   Solutions:")
        print("   1. Use --all-tcp flag to extract all TCP packets")
        print("   2. Verify ZIP file was fully extracted")
        print("   3. Check if more dataset files are available")

    if total_syn_all >= 10000:
        print("✓ SYN packet count looks reasonable")
        print(f"  You should get ~{total_syn_all:,} training records")

    if total_tcp > total_syn_all * 50:
        print("\n💡 TIP: Dataset has many ACK packets")
        print(f"   Using --all-tcp would give you {total_tcp:,} records")
        print("   (But many features will be sparse for non-SYN packets)")

    print("\n" + "="*70)
    print("Run preprocessing with:")
    print(f"  python scripts/preprocess_cesnet_idle.py")
    print(f"  (Expected output: ~{total_syn_all:,} records)")
    print("\nOr for all TCP packets:")
    print(f"  python scripts/preprocess_cesnet_idle.py --all-tcp")
    print(f"  (Expected output: ~{total_tcp:,} records)")
    print("="*70)


if __name__ == '__main__':
    main()
