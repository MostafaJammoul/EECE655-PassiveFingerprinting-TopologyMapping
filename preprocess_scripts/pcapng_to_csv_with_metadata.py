#!/usr/bin/env python3
"""
PCAPNG to CSV Converter with Metadata Extraction
Extracts TCP/IP features AND OS labels from pcapng packet comments

This script specifically handles nprint-style pcapng files where OS labels
are embedded in packet comments as: "sampleID,os_family_os_version"

Usage:
    python pcapng_to_csv_with_metadata.py input.pcapng output.csv
    python pcapng_to_csv_with_metadata.py input.pcapng output.csv --syn-only

Features:
    - Extracts OS labels from packet comments (nprint format)
    - Extracts TCP/IP fingerprinting features
    - Compatible with Windows, Linux, macOS
"""

import sys
import argparse
from pathlib import Path

try:
    from scapy.all import rdpcap, IP, TCP, PcapNgReader
except ImportError:
    print("ERROR: Scapy not installed.")
    print("Install with: pip install scapy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed.")
    print("Install with: pip install pandas")
    sys.exit(1)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_os_label_from_comment(comment):
    """
    Parse OS label from nprint-style packet comment

    Format: "sampleID,os_family_os_version"
    Examples:
        "9574731881961193404,ubuntu_ubuntu-server" → ("ubuntu", "ubuntu-server")
        "15768607979814747063,ubuntu_ubuntu-14.4-32b" → ("ubuntu", "ubuntu-14.4-32b")
        "9957971904990215253,windows_windows-vista" → ("windows", "windows-vista")

    Returns:
        (os_family, os_version) or (None, None) if parsing fails
    """
    if not comment:
        return None, None

    comment = str(comment).strip()

    # Split by comma
    parts = comment.split(',')
    if len(parts) < 2:
        return None, None

    # Second part is "easylabel_hardlabel"
    label_part = parts[1]

    # Split by underscore
    label_parts = label_part.split('_', 1)
    if len(label_parts) < 2:
        # Sometimes there's only one part
        return None, label_parts[0]

    os_family = label_parts[0]
    os_version = label_parts[1]

    return os_family, os_version


def normalize_os_family(os_family_raw):
    """
    Normalize OS family to standard names

    Examples:
        "ubuntu" → "Linux"
        "windows" → "Windows"
        "mac" → "macOS"
    """
    if not os_family_raw:
        return 'Unknown'

    os_lower = str(os_family_raw).lower()

    if 'windows' in os_lower or 'win' in os_lower:
        return 'Windows'
    elif any(w in os_lower for w in ['ubuntu', 'debian', 'linux', 'fedora', 'centos', 'kali']):
        return 'Linux'
    elif 'mac' in os_lower or 'osx' in os_lower or 'darwin' in os_lower:
        return 'macOS'
    elif 'android' in os_lower:
        return 'Android'
    elif 'ios' in os_lower:
        return 'iOS'
    else:
        return 'Other'


def normalize_os_version(os_version_raw):
    """
    Normalize OS version to readable format

    Examples:
        "ubuntu-server" → "Ubuntu Server"
        "ubuntu-14.4-32b" → "Ubuntu 14.4 32-bit"
        "windows-vista" → "Windows Vista"
    """
    if not os_version_raw:
        return 'Unknown'

    version = str(os_version_raw).strip()

    # Replace hyphens and underscores with spaces
    version = version.replace('-', ' ').replace('_', ' ')

    # Capitalize words
    version = ' '.join(word.capitalize() for word in version.split())

    # Fix common abbreviations
    version = version.replace('32b', '32-bit')
    version = version.replace('64b', '64-bit')
    version = version.replace('Osx', 'OSX')

    return version


def extract_tcp_options_order(tcp_packet):
    """Extract TCP options in order (highly discriminative!)"""
    if not hasattr(tcp_packet, 'options'):
        return None

    options = []
    for opt in tcp_packet.options:
        if isinstance(opt, tuple) and len(opt) >= 1:
            options.append(str(opt[0]))
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
    """Estimate original TTL value based on observed TTL"""
    common_ttls = [32, 64, 128, 255]
    for initial in common_ttls:
        if ttl <= initial:
            return initial
    return 255


# ============================================================================
# PCAPNG PROCESSING
# ============================================================================

def process_pcapng_with_metadata(pcap_path, syn_only=True, verbose=True):
    """
    Process PCAPNG file and extract features + OS labels from comments

    Args:
        pcap_path: Path to PCAPNG file
        syn_only: If True, only process SYN packets
        verbose: Print progress

    Returns:
        DataFrame with extracted features and OS labels
    """

    if verbose:
        print(f"\nProcessing: {pcap_path}")
        print(f"Filter: {'SYN packets only' if syn_only else 'All TCP packets'}")

    # Read PCAPNG file
    try:
        if verbose:
            print(f"\nReading packets from {pcap_path}...")

        # Try using PcapNgReader to preserve comments
        packets = []
        comments_map = {}

        try:
            with PcapNgReader(str(pcap_path)) as reader:
                for idx, packet in enumerate(reader):
                    packets.append(packet)

                    # Try to get comment from packet metadata
                    if hasattr(packet, 'comment'):
                        comments_map[idx] = packet.comment
                    elif hasattr(packet, 'time'):
                        # Some versions store it differently
                        if hasattr(packet, 'pkt_comment'):
                            comments_map[idx] = packet.pkt_comment

        except Exception as e:
            if verbose:
                print(f"  PcapNgReader failed: {e}")
                print(f"  Trying rdpcap fallback...")
            packets = rdpcap(str(pcap_path))

        if verbose:
            print(f"  Loaded {len(packets)} packets")
            if comments_map:
                print(f"  Found comments in {len(comments_map)} packets")

    except Exception as e:
        print(f"ERROR reading {pcap_path}: {e}")
        return None

    # Extract features from packets
    records = []
    tcp_count = 0
    syn_count = 0
    labeled_count = 0

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
        is_syn = (tcp_layer.flags & 0x02) != 0
        is_ack = (tcp_layer.flags & 0x10) != 0

        if is_syn:
            syn_count += 1

        # Skip non-SYN packets if syn_only is True
        if syn_only and not is_syn:
            continue

        # Determine packet type
        if is_syn and not is_ack:
            packet_type = 'SYN'
        elif is_syn and is_ack:
            packet_type = 'SYN-ACK'
        elif is_ack:
            packet_type = 'ACK'
        else:
            packet_type = 'OTHER'

        # Try to get OS label from comment
        comment = comments_map.get(pkt_idx, None)
        os_family_raw, os_version_raw = parse_os_label_from_comment(comment)

        if os_family_raw:
            labeled_count += 1

        os_family = normalize_os_family(os_family_raw)
        os_version = normalize_os_version(os_version_raw)

        # Combine for os_label
        if os_version_raw:
            os_label = os_version
        elif os_family_raw:
            os_label = os_family
        else:
            os_label = 'Unknown'

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
            'df_flag': 1 if (ip_layer.flags & 0x2) else 0,
            'ip_len': ip_layer.len if hasattr(ip_layer, 'len') else len(packet),

            # TCP layer
            'src_port': tcp_layer.sport,
            'dst_port': tcp_layer.dport,
            'tcp_window_size': tcp_layer.window,
            'tcp_flags': int(tcp_layer.flags),

            # TCP options
            'tcp_mss': extract_tcp_mss(tcp_layer),
            'tcp_window_scale': extract_tcp_window_scale(tcp_layer),
            'tcp_options_order': extract_tcp_options_order(tcp_layer),

            # Labels (from packet comments!)
            'os_label': os_label,
            'os_family': os_family,

            # Debug info
            'comment_raw': comment if comment else None,
        }

        records.append(record)

    # Summary
    if verbose:
        print(f"\nPacket statistics:")
        print(f"  Total packets in file: {len(packets)}")
        print(f"  TCP packets: {tcp_count}")
        print(f"  SYN packets: {syn_count}")
        print(f"  Packets extracted: {len(records)}")
        print(f"  Packets with OS labels: {labeled_count}")

        if labeled_count == 0:
            print("\n  WARNING: No OS labels found in packet comments!")
            print("  This might not be an nprint-formatted pcapng file.")
            print("  Or Scapy version doesn't support reading pcapng comments.")
            print("\n  Solution: Try using tshark or install pypcapng:")
            print("    pip install pypcapng")

    if not records:
        print("\nERROR: No packets extracted!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Show statistics
    if verbose:
        print(f"\nFeature completeness:")
        critical_features = ['ttl', 'tcp_window_size', 'tcp_mss', 'tcp_options_order']
        for feat in critical_features:
            if feat in df.columns:
                pct_available = (df[feat].notna().sum() / len(df)) * 100
                print(f"  {feat}: {pct_available:.1f}%")

        print(f"\nOS Label distribution:")
        print(df['os_label'].value_counts().head(10))

        print(f"\nOS Family distribution:")
        print(df['os_family'].value_counts())

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract TCP/IP features and OS labels from nprint-style PCAPNG files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python pcapng_to_csv_with_metadata.py nprint_file.pcapng output.csv

  # Extract all TCP packets (not just SYN)
  python pcapng_to_csv_with_metadata.py nprint_file.pcapng output.csv --all-tcp

Features extracted:
  - TCP/IP fingerprinting features (TTL, window size, MSS, options order, etc.)
  - OS labels from packet comments (nprint format)
  - OS family (auto-normalized to Windows/Linux/macOS/etc.)
        """
    )

    parser.add_argument('input', type=str, help='Input PCAPNG file')
    parser.add_argument('output', type=str, help='Output CSV file')
    parser.add_argument('--all-tcp', action='store_true', help='Extract all TCP packets (default: SYN only)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Process
    print("="*70)
    print("PCAPNG TO CSV CONVERTER (WITH METADATA)")
    print("="*70)

    df = process_pcapng_with_metadata(
        pcap_path=input_path,
        syn_only=not args.all_tcp,
        verbose=not args.quiet
    )

    if df is None:
        print("\nERROR: No data extracted!")
        sys.exit(1)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"\nSaving to CSV...")

    # Drop debug column before saving
    if 'comment_raw' in df.columns:
        df_save = df.drop(columns=['comment_raw'])
    else:
        df_save = df

    df_save.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"\nOutput: {output_path}")
    print(f"Records: {len(df):,}")
    print(f"Features: {len(df.columns)}")

    if not args.quiet:
        print(f"\nDataset preview:")
        print(df[['record_id', 'src_ip', 'ttl', 'tcp_window_size', 'tcp_options_order', 'os_label', 'os_family']].head())

    print(f"\nReady for training!")


if __name__ == '__main__':
    main()
