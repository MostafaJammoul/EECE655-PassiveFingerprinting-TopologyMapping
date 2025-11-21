#!/usr/bin/env python3
"""
Analyze TCP Packet Composition in PCAPNG File

This script analyzes a pcapng file and provides detailed statistics about:
- Total packets and TCP packets
- TCP flag composition (SYN, SYN-ACK, ACK, FIN, RST, PSH, etc.)
- Packet type distribution
- Source/destination statistics

Usage:
    python analyze_pcapng_composition.py <input.pcapng>
    python analyze_pcapng_composition.py <input.pcapng> --detailed
"""

import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict

try:
    from scapy.all import rdpcap, IP, TCP, PcapNgReader
except ImportError:
    print("ERROR: Scapy not installed.")
    print("Install with: pip install scapy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None
    print("WARNING: pandas not installed. Install for enhanced output: pip install pandas")


def analyze_tcp_flags(tcp_layer):
    """
    Analyze TCP flags and return packet type classification

    Returns:
        (packet_type, flags_dict)
    """
    flags_dict = {
        'SYN': bool(tcp_layer.flags & 0x02),
        'ACK': bool(tcp_layer.flags & 0x10),
        'FIN': bool(tcp_layer.flags & 0x01),
        'RST': bool(tcp_layer.flags & 0x04),
        'PSH': bool(tcp_layer.flags & 0x08),
        'URG': bool(tcp_layer.flags & 0x20),
    }

    # Determine packet type
    if flags_dict['SYN'] and not flags_dict['ACK']:
        packet_type = 'SYN'
    elif flags_dict['SYN'] and flags_dict['ACK']:
        packet_type = 'SYN-ACK'
    elif flags_dict['FIN'] and flags_dict['ACK']:
        packet_type = 'FIN-ACK'
    elif flags_dict['RST'] and flags_dict['ACK']:
        packet_type = 'RST-ACK'
    elif flags_dict['RST']:
        packet_type = 'RST'
    elif flags_dict['ACK'] and not any([flags_dict['SYN'], flags_dict['FIN'], flags_dict['RST']]):
        if flags_dict['PSH']:
            packet_type = 'PSH-ACK'
        else:
            packet_type = 'ACK'
    else:
        # Build flag string
        active_flags = [flag for flag, active in flags_dict.items() if active]
        packet_type = '+'.join(active_flags) if active_flags else 'NONE'

    return packet_type, flags_dict


def analyze_pcapng(pcap_path, detailed=False, verbose=True):
    """
    Analyze TCP packet composition in a pcapng file

    Args:
        pcap_path: Path to pcapng file
        detailed: Show detailed per-flag statistics
        verbose: Print progress

    Returns:
        dict with statistics
    """

    if verbose:
        print("="*70)
        print("TCP PACKET COMPOSITION ANALYZER")
        print("="*70)
        print(f"\nAnalyzing: {pcap_path}")

    # Read packets
    try:
        if verbose:
            print(f"\nReading packets...")

        packets = []
        try:
            # Try PcapNgReader first
            with PcapNgReader(str(pcap_path)) as reader:
                packets = [pkt for pkt in reader]
        except Exception as e:
            if verbose:
                print(f"  PcapNgReader failed: {e}")
                print(f"  Trying rdpcap fallback...")
            packets = rdpcap(str(pcap_path))

        if verbose:
            print(f"  Loaded {len(packets):,} packets")

    except Exception as e:
        print(f"ERROR reading {pcap_path}: {e}")
        return None

    # Statistics
    stats = {
        'total_packets': len(packets),
        'ip_packets': 0,
        'tcp_packets': 0,
        'udp_packets': 0,
        'other_packets': 0,
        'packet_types': Counter(),
        'flag_counts': defaultdict(int),
        'src_ips': Counter(),
        'dst_ips': Counter(),
        'src_ports': Counter(),
        'dst_ports': Counter(),
    }

    # Analyze each packet
    if verbose:
        print(f"\nAnalyzing packet composition...")

    for packet in packets:
        # Check if it has IP layer
        if packet.haslayer(IP):
            stats['ip_packets'] += 1
            ip_layer = packet[IP]

            # Check protocol
            if packet.haslayer(TCP):
                stats['tcp_packets'] += 1
                tcp_layer = packet[TCP]

                # Analyze flags
                packet_type, flags_dict = analyze_tcp_flags(tcp_layer)
                stats['packet_types'][packet_type] += 1

                # Count individual flags
                for flag, is_set in flags_dict.items():
                    if is_set:
                        stats['flag_counts'][flag] += 1

                # Track IPs and ports
                stats['src_ips'][ip_layer.src] += 1
                stats['dst_ips'][ip_layer.dst] += 1
                stats['src_ports'][tcp_layer.sport] += 1
                stats['dst_ports'][tcp_layer.dport] += 1

            else:
                # Check for UDP
                try:
                    from scapy.all import UDP
                    if packet.haslayer(UDP):
                        stats['udp_packets'] += 1
                    else:
                        stats['other_packets'] += 1
                except:
                    stats['other_packets'] += 1
        else:
            stats['other_packets'] += 1

    # Print results
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)

    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total packets:     {stats['total_packets']:>10,}")
    print(f"  IP packets:        {stats['ip_packets']:>10,} ({stats['ip_packets']/stats['total_packets']*100:>6.2f}%)")
    print(f"  TCP packets:       {stats['tcp_packets']:>10,} ({stats['tcp_packets']/stats['total_packets']*100:>6.2f}%)")
    print(f"  UDP packets:       {stats['udp_packets']:>10,} ({stats['udp_packets']/stats['total_packets']*100:>6.2f}%)")
    print(f"  Other packets:     {stats['other_packets']:>10,} ({stats['other_packets']/stats['total_packets']*100:>6.2f}%)")

    # TCP packet type distribution
    if stats['tcp_packets'] > 0:
        print(f"\nTCP Packet Type Distribution:")
        print(f"  {'Type':<20} {'Count':>12} {'Percentage':>12}")
        print(f"  {'-'*20} {'-'*12} {'-'*12}")

        # Sort by count
        sorted_types = stats['packet_types'].most_common()
        for pkt_type, count in sorted_types:
            pct = (count / stats['tcp_packets']) * 100
            print(f"  {pkt_type:<20} {count:>12,} {pct:>11.2f}%")

        # TCP flag statistics
        print(f"\nTCP Flag Statistics (individual flags):")
        print(f"  {'Flag':<20} {'Count':>12} {'Percentage':>12}")
        print(f"  {'-'*20} {'-'*12} {'-'*12}")

        for flag in ['SYN', 'ACK', 'FIN', 'RST', 'PSH', 'URG']:
            count = stats['flag_counts'][flag]
            pct = (count / stats['tcp_packets']) * 100 if stats['tcp_packets'] > 0 else 0
            print(f"  {flag:<20} {count:>12,} {pct:>11.2f}%")

    # Detailed statistics
    if detailed:
        print(f"\n" + "="*70)
        print("DETAILED STATISTICS")
        print("="*70)

        # Top source IPs
        print(f"\nTop 10 Source IPs:")
        for ip, count in stats['src_ips'].most_common(10):
            pct = (count / stats['tcp_packets']) * 100 if stats['tcp_packets'] > 0 else 0
            print(f"  {ip:<20} {count:>12,} ({pct:.2f}%)")

        # Top destination IPs
        print(f"\nTop 10 Destination IPs:")
        for ip, count in stats['dst_ips'].most_common(10):
            pct = (count / stats['tcp_packets']) * 100 if stats['tcp_packets'] > 0 else 0
            print(f"  {ip:<20} {count:>12,} ({pct:.2f}%)")

        # Top source ports
        print(f"\nTop 10 Source Ports:")
        for port, count in stats['src_ports'].most_common(10):
            pct = (count / stats['tcp_packets']) * 100 if stats['tcp_packets'] > 0 else 0
            print(f"  {port:<20} {count:>12,} ({pct:.2f}%)")

        # Top destination ports
        print(f"\nTop 10 Destination Ports:")
        for port, count in stats['dst_ports'].most_common(10):
            pct = (count / stats['tcp_packets']) * 100 if stats['tcp_packets'] > 0 else 0
            print(f"  {port:<20} {count:>12,} ({pct:.2f}%)")

    # Summary for inference pipeline
    print(f"\n" + "="*70)
    print("INFERENCE PIPELINE RECOMMENDATIONS")
    print("="*70)

    if stats['tcp_packets'] > 0:
        syn_count = stats['packet_types']['SYN']
        syn_ack_count = stats['packet_types']['SYN-ACK']
        ack_count = stats['packet_types']['ACK']

        syn_pct = (syn_count / stats['tcp_packets']) * 100
        syn_ack_pct = (syn_ack_count / stats['tcp_packets']) * 100
        ack_pct = (ack_count / stats['tcp_packets']) * 100

        print(f"\nDataset Composition:")
        print(f"  SYN packets:       {syn_count:>10,} ({syn_pct:>6.2f}%)")
        print(f"  SYN-ACK packets:   {syn_ack_count:>10,} ({syn_ack_pct:>6.2f}%)")
        print(f"  ACK packets:       {ack_count:>10,} ({ack_pct:>6.2f}%)")

        print(f"\nRecommendations:")
        if syn_pct < 1.0:
            print(f"  ⚠ SYN packets are only {syn_pct:.2f}% of the dataset")
            print(f"    → Consider using ALL TCP packets for better model training")
        else:
            print(f"  ✓ SYN packets are {syn_pct:.2f}% of the dataset")
            print(f"    → Sufficient for SYN-based fingerprinting")

        if ack_pct > 50.0:
            print(f"  ✓ ACK packets dominate ({ack_pct:.2f}%)")
            print(f"    → Good for ACK-based fingerprinting (CESNET idle approach)")

        print(f"\n  Dataset Usage:")
        print(f"    • For OS family classification: Use flow-level features (Masaryk)")
        print(f"    • For OS version (SYN-based): Extract SYN packets only")
        print(f"    • For OS version (ACK-based): Extract ACK packets (3-5 per flow)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Analyze TCP packet composition in PCAPNG files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_pcapng_composition.py input.pcapng

  # Detailed analysis with IP/port statistics
  python analyze_pcapng_composition.py input.pcapng --detailed

  # Quiet mode (minimal output)
  python analyze_pcapng_composition.py input.pcapng --quiet
        """
    )

    parser.add_argument('input', type=str, help='Input PCAPNG file')
    parser.add_argument('--detailed', action='store_true', help='Show detailed statistics (IPs, ports)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print(f"\nPlease provide a valid pcapng file path.")
        print(f"Example: python analyze_pcapng_composition.py data/raw/nprint/dataset.pcapng")
        sys.exit(1)

    # Analyze
    stats = analyze_pcapng(
        pcap_path=input_path,
        detailed=args.detailed,
        verbose=not args.quiet
    )

    if stats is None:
        print("\nERROR: Analysis failed!")
        sys.exit(1)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nUse --detailed flag for more statistics (IP addresses, ports, etc.)")


if __name__ == '__main__':
    main()
