#!/usr/bin/env python3
"""
PCAPNG to CSV Converter - Windows Compatible
Extract TCP/IP Fingerprinting Features from PCAP/PCAPNG Files

This script works on Windows, Linux, and macOS.
It extracts the same features as preprocess_cesnet_idle.py but as a standalone tool.

Usage:
    python pcapng_to_csv_windows.py input.pcapng output.csv
    python pcapng_to_csv_windows.py input.pcapng output.csv --os-label "Windows 11"
    python pcapng_to_csv_windows.py input.pcapng output.csv --syn-only

Installation (Windows):
    pip install scapy pandas

Installation (Windows with Npcap - recommended for better performance):
    1. Download and install Npcap from https://npcap.com/#download
    2. pip install scapy pandas

Features extracted:
    - IP: TTL, initial TTL, DF flag, IP length, src/dst IPs
    - TCP: Window size, MSS, window scale, options order, flags, src/dst ports
    - Labels: OS label (if provided), OS family
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Check for required dependencies
try:
    from scapy.all import rdpcap, IP, TCP
except ImportError:
    print("ERROR: Scapy not installed.")
    print("\nInstallation instructions:")
    print("  Windows: pip install scapy")
    print("  Linux:   pip install scapy")
    print("  macOS:   pip install scapy")
    print("\nFor better performance on Windows, install Npcap:")
    print("  https://npcap.com/#download")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed.")
    print("  Install with: pip install pandas")
    sys.exit(1)


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_tcp_options_order(tcp_packet):
    """
    Extract the ORDER of TCP options - highly discriminative for OS fingerprinting!

    Example: "MSS:NOP:WS:NOP:NOP:SACK" uniquely identifies certain OS versions

    Common option names:
        MSS         - Maximum Segment Size
        NOP         - No Operation (padding)
        WScale      - Window Scale
        SAckOK      - SACK Permitted
        Timestamp   - TCP Timestamp
        EOL         - End of Option List
    """
    if not hasattr(tcp_packet, 'options'):
        return None

    options = []
    for opt in tcp_packet.options:
        if isinstance(opt, tuple) and len(opt) >= 1:
            opt_name = opt[0]
            options.append(str(opt_name))
        elif isinstance(opt, str):
            options.append(opt)

    return ':'.join(options) if options else None


def extract_tcp_mss(tcp_packet):
    """Extract Maximum Segment Size from TCP options"""
    if not hasattr(tcp_packet, 'options'):
        return None

    for opt in tcp_packet.options:
        if isinstance(opt, tuple) and len(opt) >= 2:
            if opt[0] == 'MSS':
                return opt[1]
    return None


def extract_tcp_window_scale(tcp_packet):
    """Extract Window Scale factor from TCP options"""
    if not hasattr(tcp_packet, 'options'):
        return None

    for opt in tcp_packet.options:
        if isinstance(opt, tuple) and len(opt) >= 2:
            if opt[0] == 'WScale':
                return opt[1]
    return None


def calculate_initial_ttl(ttl):
    """
    Estimate the original TTL value based on observed TTL

    Common initial TTLs:
    - 64:  Linux, macOS, Unix, FreeBSD
    - 128: Windows 7/8/10/11, Windows Server
    - 255: Cisco, Solaris, some network devices
    - 32:  Old Windows, some embedded systems
    """
    common_ttls = [32, 64, 128, 255]
    for initial in common_ttls:
        if ttl <= initial:
            return initial
    return 255  # Default to max if ttl > 255 (shouldn't happen)


def extract_os_family(os_label):
    """
    Extract OS family from detailed OS label

    Examples:
    - "Windows 11" → "Windows"
    - "Ubuntu 22.04" → "Linux"
    - "macOS 14" → "macOS"
    """
    if not os_label:
        return 'Unknown'

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
        return 'Unknown'


# ============================================================================
# PCAP PROCESSING
# ============================================================================

def process_pcap(pcap_path, os_label=None, syn_only=True, verbose=True):
    """
    Process PCAP/PCAPNG file and extract TCP/IP fingerprinting features

    Args:
        pcap_path: Path to PCAP or PCAPNG file
        os_label: Optional OS label (e.g., "Windows 11", "Ubuntu 22.04")
        syn_only: If True, only process SYN packets (recommended for fingerprinting)
        verbose: Print progress information

    Returns:
        DataFrame with extracted features
    """

    if verbose:
        print(f"\nProcessing: {pcap_path}")
        print(f"OS Label: {os_label if os_label else 'Not specified'}")
        print(f"Filter: {'SYN packets only' if syn_only else 'All TCP packets'}")

    # Read PCAP file
    try:
        if verbose:
            print(f"\nReading packets from {pcap_path}...")
        packets = rdpcap(str(pcap_path))
        if verbose:
            print(f"  Loaded {len(packets)} packets")
    except Exception as e:
        print(f"ERROR reading {pcap_path}: {e}")
        return None

    # Extract features from packets
    records = []
    tcp_count = 0
    syn_count = 0

    if verbose:
        print(f"\nExtracting features...")

    for pkt_idx, packet in enumerate(packets):
        # Must have IP and TCP layers
        if not (packet.haslayer(IP) and packet.haslayer(TCP)):
            continue

        tcp_count += 1
        ip_layer = packet[IP]
        tcp_layer = packet[TCP]

        # Check if this is a SYN packet
        is_syn = (tcp_layer.flags & 0x02) != 0  # SYN flag
        is_ack = (tcp_layer.flags & 0x10) != 0  # ACK flag

        if is_syn:
            syn_count += 1

        # Skip non-SYN packets if syn_only is True
        if syn_only and not is_syn:
            continue

        # Determine packet type for labeling
        if is_syn and not is_ack:
            packet_type = 'SYN'
        elif is_syn and is_ack:
            packet_type = 'SYN-ACK'
        elif is_ack:
            packet_type = 'ACK'
        else:
            packet_type = 'OTHER'

        # Extract features
        record = {
            # Metadata
            'record_id': f"pkt_{pkt_idx}",
            'timestamp': float(packet.time) if hasattr(packet, 'time') else None,
            'packet_type': packet_type,

            # IP layer
            'src_ip': ip_layer.src,
            'dst_ip': ip_layer.dst,
            'protocol': ip_layer.proto,
            'ttl': ip_layer.ttl,
            'initial_ttl': calculate_initial_ttl(ip_layer.ttl),
            'df_flag': 1 if (ip_layer.flags & 0x2) else 0,  # Don't Fragment
            'ip_len': ip_layer.len if hasattr(ip_layer, 'len') else len(packet),

            # TCP layer
            'src_port': tcp_layer.sport,
            'dst_port': tcp_layer.dport,
            'tcp_window_size': tcp_layer.window,
            'tcp_flags': int(tcp_layer.flags),

            # TCP options (critical for fingerprinting!)
            'tcp_mss': extract_tcp_mss(tcp_layer),
            'tcp_window_scale': extract_tcp_window_scale(tcp_layer),
            'tcp_options_order': extract_tcp_options_order(tcp_layer),

            # Labels
            'os_label': os_label if os_label else 'Unknown',
            'os_family': extract_os_family(os_label) if os_label else 'Unknown',
        }

        records.append(record)

    # Summary
    if verbose:
        print(f"\nPacket statistics:")
        print(f"  Total packets in file: {len(packets)}")
        print(f"  TCP packets: {tcp_count}")
        print(f"  SYN packets: {syn_count}")
        print(f"  Packets extracted: {len(records)}")

    if not records:
        if syn_only:
            print("\nWARNING: No SYN packets found!")
            print("Try running without --syn-only flag to extract all TCP packets.")
        else:
            print("\nWARNING: No TCP packets found!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Show feature statistics
    if verbose:
        print(f"\nFeature completeness:")
        critical_features = ['ttl', 'tcp_window_size', 'tcp_mss', 'tcp_options_order']
        for feat in critical_features:
            if feat in df.columns:
                pct_available = (df[feat].notna().sum() / len(df)) * 100
                print(f"  {feat}: {pct_available:.1f}%")

        if 'tcp_options_order' in df.columns and df['tcp_options_order'].notna().sum() > 0:
            print(f"\nTCP option patterns found:")
            top_patterns = df['tcp_options_order'].value_counts().head(5)
            for pattern, count in top_patterns.items():
                print(f"  {pattern}: {count} packets")

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract TCP/IP fingerprinting features from PCAP/PCAPNG files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python pcapng_to_csv_windows.py capture.pcapng output.csv

  # With OS label
  python pcapng_to_csv_windows.py capture.pcapng output.csv --os-label "Windows 11"

  # Extract all TCP packets (not just SYN)
  python pcapng_to_csv_windows.py capture.pcapng output.csv --all-tcp

  # Quiet mode
  python pcapng_to_csv_windows.py capture.pcapng output.csv --quiet

Features extracted:
  IP Layer:    TTL, initial TTL, DF flag, IP length, src/dst IPs, protocol
  TCP Layer:   Window size, MSS, window scale, options order, flags, src/dst ports
  Metadata:    Timestamp, packet type, OS label (if provided)
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Input PCAP or PCAPNG file'
    )

    parser.add_argument(
        'output',
        type=str,
        help='Output CSV file'
    )

    parser.add_argument(
        '--os-label',
        type=str,
        default=None,
        help='OS label for this capture (e.g., "Windows 11", "Ubuntu 22.04")'
    )

    parser.add_argument(
        '--all-tcp',
        action='store_true',
        help='Extract all TCP packets (default: SYN packets only)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    if not input_path.suffix.lower() in ['.pcap', '.pcapng', '.cap']:
        print(f"WARNING: File extension is {input_path.suffix}")
        print(f"Expected .pcap or .pcapng - will try to process anyway...")

    # Process PCAP
    print("="*70)
    print("PCAP/PCAPNG TO CSV CONVERTER")
    print("="*70)

    df = process_pcap(
        pcap_path=input_path,
        os_label=args.os_label,
        syn_only=not args.all_tcp,
        verbose=not args.quiet
    )

    if df is None:
        print("\nERROR: No data extracted!")
        sys.exit(1)

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"\nSaving to CSV...")

    df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"\nOutput: {output_path}")
    print(f"Records: {len(df):,}")
    print(f"Features: {len(df.columns)}")

    if not args.quiet:
        print(f"\nDataset preview:")
        print(df.head())

        print(f"\nColumn names:")
        for col in df.columns:
            print(f"  - {col}")

    print(f"\nYou can now use this CSV for OS fingerprinting!")
    print(f"The dataset is compatible with the preprocessing pipeline in this project.")


if __name__ == '__main__':
    main()
