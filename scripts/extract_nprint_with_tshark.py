#!/usr/bin/env python3
"""
Extract nPrint PCAPNG with OS Labels Using Tshark

This script uses tshark to extract packet comments from pcapng files,
then uses Scapy to extract TCP/IP features.

Requirements:
    - Wireshark/tshark installed
    - pip install scapy pandas

Usage:
    python extract_nprint_with_tshark.py input.pcapng output.csv
"""

import sys
import subprocess
import json
from pathlib import Path
import argparse

try:
    from scapy.all import rdpcap, IP, TCP
except ImportError:
    print("ERROR: Scapy not installed")
    print("Install with: pip install scapy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed")
    print("Install with: pip install pandas")
    sys.exit(1)


def check_tshark():
    """Check if tshark is installed"""
    try:
        result = subprocess.run(['tshark', '--version'],
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def extract_comments_with_tshark(pcap_path):
    """
    Extract packet comments using tshark

    Returns:
        dict mapping packet_index → comment
    """
    print(f"\n[1/3] Extracting packet comments with tshark...")

    try:
        # Use tshark to export packet comments
        # -T fields: output as fields
        # -e frame.number: packet number
        # -e frame.comment: packet comment
        cmd = [
            'tshark',
            '-r', str(pcap_path),
            '-T', 'fields',
            '-e', 'frame.number',
            '-e', 'frame.comment',
            '-E', 'separator=|',
            '-E', 'quote=d',
            '-E', 'occurrence=f'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"  tshark error: {result.stderr}")
            return {}

        # Parse output
        comments = {}
        for line in result.stdout.strip().split('\n'):
            if not line or '|' not in line:
                continue

            parts = line.split('|', 1)
            if len(parts) == 2:
                frame_num = parts[0].strip()
                comment = parts[1].strip().strip('"')

                if frame_num.isdigit() and comment:
                    # Tshark uses 1-based indexing, convert to 0-based
                    packet_idx = int(frame_num) - 1
                    comments[packet_idx] = comment

        print(f"  Found comments in {len(comments)} packets")
        return comments

    except subprocess.TimeoutExpired:
        print("  ERROR: tshark timeout")
        return {}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}


def parse_os_label_from_comment(comment):
    """
    Parse OS label from nprint-style comment

    Format: "sampleID,os_family_os_version"
    Returns: (os_family, os_version, sample_id)
    """
    if not comment:
        return None, None, None

    parts = comment.split(',')
    if len(parts) < 2:
        return None, None, None

    sample_id = parts[0].strip()
    label_part = parts[1].strip()

    # Split "os_family_os_version"
    label_parts = label_part.split('_', 1)
    if len(label_parts) < 2:
        return None, label_parts[0], sample_id

    os_family_raw = label_parts[0]
    os_version_raw = label_parts[1]

    return os_family_raw, os_version_raw, sample_id


def normalize_os_family(os_family_raw):
    """Normalize to Windows/Linux/macOS/etc."""
    if not os_family_raw:
        return 'Unknown'

    os_lower = str(os_family_raw).lower()

    if 'windows' in os_lower or 'win' in os_lower:
        return 'Windows'
    elif any(w in os_lower for w in ['ubuntu', 'debian', 'linux', 'fedora', 'centos']):
        return 'Linux'
    elif 'mac' in os_lower or 'osx' in os_lower:
        return 'macOS'
    elif 'android' in os_lower:
        return 'Android'
    elif 'ios' in os_lower:
        return 'iOS'
    else:
        return 'Other'


def normalize_os_version(os_version_raw):
    """Clean up OS version string"""
    if not os_version_raw:
        return 'Unknown'

    version = str(os_version_raw).replace('-', ' ').replace('_', ' ')
    version = ' '.join(word.capitalize() for word in version.split())
    version = version.replace('32b', '32-bit').replace('64b', '64-bit')

    return version


def extract_features(pcap_path, comments_map, syn_only=True, verbose=True):
    """Extract TCP/IP features from packets"""

    print(f"\n[2/3] Extracting TCP/IP features with Scapy...")

    try:
        packets = rdpcap(str(pcap_path))
        if verbose:
            print(f"  Loaded {len(packets)} packets")
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    records = []
    tcp_count = 0
    syn_count = 0
    labeled_count = 0

    for pkt_idx, packet in enumerate(packets):
        if not (packet.haslayer(IP) and packet.haslayer(TCP)):
            continue

        tcp_count += 1
        ip = packet[IP]
        tcp = packet[TCP]

        is_syn = (tcp.flags & 0x02) != 0
        is_ack = (tcp.flags & 0x10) != 0

        if is_syn:
            syn_count += 1

        if syn_only and not is_syn:
            continue

        # Determine packet type
        if is_syn and not is_ack:
            packet_type = 'SYN'
        elif is_syn and is_ack:
            packet_type = 'SYN-ACK'
        else:
            packet_type = 'ACK'

        # Get OS label from comments
        comment = comments_map.get(pkt_idx, None)
        os_family_raw, os_version_raw, sample_id = parse_os_label_from_comment(comment)

        if os_family_raw:
            labeled_count += 1

        os_family = normalize_os_family(os_family_raw)
        os_version = normalize_os_version(os_version_raw)

        if os_version_raw:
            os_label = os_version
        elif os_family_raw:
            os_label = os_family
        else:
            os_label = 'Unknown'

        # Extract TCP options
        tcp_options = []
        tcp_mss = None
        tcp_wscale = None

        if hasattr(tcp, 'options'):
            for opt in tcp.options:
                if isinstance(opt, tuple) and len(opt) >= 1:
                    opt_name = str(opt[0])
                    tcp_options.append(opt_name)

                    if opt_name == 'MSS' and len(opt) >= 2:
                        tcp_mss = opt[1]
                    elif opt_name == 'WScale' and len(opt) >= 2:
                        tcp_wscale = opt[1]

        tcp_options_order = ':'.join(tcp_options) if tcp_options else None

        # Calculate initial TTL
        ttl = ip.ttl
        initial_ttl = 255
        for init in [32, 64, 128, 255]:
            if ttl <= init:
                initial_ttl = init
                break

        record = {
            'record_id': f"pkt_{pkt_idx}",
            'sample_id': sample_id if sample_id else None,
            'timestamp': float(packet.time) if hasattr(packet, 'time') else None,
            'packet_type': packet_type,

            'src_ip': ip.src,
            'dst_ip': ip.dst,
            'protocol': ip.proto,
            'ttl': ttl,
            'initial_ttl': initial_ttl,
            'df_flag': 1 if (ip.flags & 0x2) else 0,
            'ip_len': ip.len if hasattr(ip, 'len') else len(packet),

            'src_port': tcp.sport,
            'dst_port': tcp.dport,
            'tcp_window_size': tcp.window,
            'tcp_flags': int(tcp.flags),
            'tcp_mss': tcp_mss,
            'tcp_window_scale': tcp_wscale,
            'tcp_options_order': tcp_options_order,

            'os_label': os_label,
            'os_family': os_family,
        }

        records.append(record)

    if verbose:
        print(f"\n  Statistics:")
        print(f"    Total packets: {len(packets)}")
        print(f"    TCP packets: {tcp_count}")
        print(f"    SYN packets: {syn_count}")
        print(f"    Extracted: {len(records)}")
        print(f"    With OS labels: {labeled_count}")

        if labeled_count == 0:
            print(f"\n  WARNING: No OS labels found!")
            print(f"  Check if the pcapng file has packet comments.")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description='Extract nPrint PCAPNG with OS labels using tshark')
    parser.add_argument('input', help='Input PCAPNG file')
    parser.add_argument('output', help='Output CSV file')
    parser.add_argument('--all-tcp', action='store_true', help='Extract all TCP packets (default: SYN only)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    print("="*70)
    print("NPRINT PCAPNG EXTRACTOR (TSHARK)")
    print("="*70)

    # Check tshark
    if not check_tshark():
        print("\nERROR: tshark not found!")
        print("\nInstallation:")
        print("  Windows: Install Wireshark from https://www.wireshark.org/")
        print("  Linux:   sudo apt install tshark")
        print("  macOS:   brew install wireshark")
        sys.exit(1)

    # Extract comments
    comments_map = extract_comments_with_tshark(input_path)

    # Extract features
    df = extract_features(input_path, comments_map, syn_only=not args.all_tcp, verbose=not args.quiet)

    if df is None or len(df) == 0:
        print("\nERROR: No data extracted!")
        sys.exit(1)

    # Save
    print(f"\n[3/3] Saving to CSV...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"\nOutput: {output_path}")
    print(f"Records: {len(df):,}")

    if not args.quiet:
        print(f"\nOS Label distribution:")
        print(df['os_label'].value_counts().head(15))

        print(f"\nOS Family distribution:")
        print(df['os_family'].value_counts())

        print(f"\nDataset preview:")
        print(df[['src_ip', 'ttl', 'tcp_options_order', 'os_label', 'os_family']].head(10))

    print(f"\n✓ Dataset ready for training!")


if __name__ == '__main__':
    main()
