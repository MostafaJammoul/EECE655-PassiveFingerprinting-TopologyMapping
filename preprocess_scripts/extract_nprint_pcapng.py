#!/usr/bin/env python3
"""
Extract nPrint PCAPNG with OS Labels - No tshark Required!

Uses pypcapng library to read packet comments from pcapng files.

Installation:
    pip install pypcapng scapy pandas

Usage:
    python extract_nprint_pcapng.py input.pcapng output.csv
"""

import sys
from pathlib import Path
import argparse

# Check for pypcapng
try:
    import pcapng
    from pcapng.blocks import EnhancedPacket, SimplePacket
except ImportError:
    print("ERROR: pypcapng not installed")
    print("\nInstall with:")
    print("  pip install pypcapng")
    print("\nOr install all dependencies:")
    print("  pip install pypcapng scapy pandas")
    sys.exit(1)

try:
    from scapy.all import Ether, IP, TCP
except ImportError:
    print("ERROR: scapy not installed")
    print("Install with: pip install scapy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed")
    print("Install with: pip install pandas")
    sys.exit(1)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_os_label_from_comment(comment):
    """
    Parse OS label from nprint-style comment

    Format: "sampleID,os_family_os_version"
    Examples:
        "9574731881961193404,ubuntu_ubuntu-server" → ("ubuntu", "ubuntu-server", "9574731881961193404")

    Returns: (os_family, os_version, sample_id)
    """
    if not comment:
        return None, None, None

    comment = str(comment).strip()
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
    """Normalize to standard OS families"""
    if not os_family_raw:
        return 'Unknown'

    os_lower = str(os_family_raw).lower()

    if 'windows' in os_lower or 'win' in os_lower:
        return 'Windows'
    elif any(w in os_lower for w in ['ubuntu', 'debian', 'linux', 'fedora', 'centos', 'kali']):
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

    # Fix common patterns
    version = version.replace('32b', '32-bit')
    version = version.replace('64b', '64-bit')
    version = version.replace('Osx', 'OSX')

    return version


def extract_tcp_options_order(tcp_packet):
    """Extract TCP options in order"""
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
    """Extract MSS from TCP options"""
    if not hasattr(tcp_packet, 'options'):
        return None

    for opt in tcp_packet.options:
        if isinstance(opt, tuple) and len(opt) >= 2:
            if opt[0] == 'MSS':
                return opt[1]
    return None


def extract_tcp_window_scale(tcp_packet):
    """Extract Window Scale from TCP options"""
    if not hasattr(tcp_packet, 'options'):
        return None

    for opt in tcp_packet.options:
        if isinstance(opt, tuple) and len(opt) >= 2:
            if opt[0] == 'WScale':
                return opt[1]
    return None


def calculate_initial_ttl(ttl):
    """Estimate original TTL"""
    for initial in [32, 64, 128, 255]:
        if ttl <= initial:
            return initial
    return 255


# ============================================================================
# PCAPNG PROCESSING
# ============================================================================

def process_pcapng(pcap_path, syn_only=True, verbose=True):
    """
    Process PCAPNG file and extract features + OS labels

    Args:
        pcap_path: Path to PCAPNG file
        syn_only: Only extract SYN packets
        verbose: Print progress

    Returns:
        DataFrame with extracted features
    """

    if verbose:
        print(f"\nProcessing: {pcap_path}")
        print(f"Filter: {'SYN packets only' if syn_only else 'All TCP packets'}")

    records = []
    tcp_count = 0
    syn_count = 0
    labeled_count = 0
    total_packets = 0

    if verbose:
        print(f"\nReading packets with pypcapng...")

    try:
        with open(pcap_path, 'rb') as f:
            scanner = pcapng.FileScanner(f)

            for block_idx, block in enumerate(scanner):
                # Only process Enhanced Packet Blocks (EPB) which can have comments
                if not isinstance(block, (EnhancedPacket, SimplePacket)):
                    continue

                total_packets += 1

                # Get packet data
                packet_data = block.packet_data

                # Parse with Scapy
                try:
                    packet = Ether(packet_data)
                except:
                    # If not Ethernet, try raw IP
                    try:
                        packet = IP(packet_data)
                    except:
                        continue

                # Must have IP and TCP
                if not (packet.haslayer(IP) and packet.haslayer(TCP)):
                    continue

                tcp_count += 1
                ip = packet[IP]
                tcp = packet[TCP]

                # Check if SYN
                is_syn = (tcp.flags & 0x02) != 0
                is_ack = (tcp.flags & 0x10) != 0

                if is_syn:
                    syn_count += 1

                if syn_only and not is_syn:
                    continue

                # Packet type
                if is_syn and not is_ack:
                    packet_type = 'SYN'
                elif is_syn and is_ack:
                    packet_type = 'SYN-ACK'
                elif is_ack:
                    packet_type = 'ACK'
                else:
                    packet_type = 'OTHER'

                # Extract comment from block options
                comment = None
                if hasattr(block, 'options'):
                    for option in block.options:
                        if hasattr(option, 'option_code') and option.option_code == 1:  # opt_comment
                            if hasattr(option, 'option_value'):
                                comment = option.option_value.decode('utf-8', errors='ignore')
                            elif hasattr(option, 'comment'):
                                comment = option.comment
                            break

                # Parse OS label from comment
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

                # Extract features
                record = {
                    'record_id': f"pkt_{block_idx}",
                    'sample_id': sample_id if sample_id else None,
                    'timestamp': float(block.timestamp) if hasattr(block, 'timestamp') else None,
                    'packet_type': packet_type,

                    'src_ip': ip.src,
                    'dst_ip': ip.dst,
                    'protocol': ip.proto,
                    'ttl': ip.ttl,
                    'initial_ttl': calculate_initial_ttl(ip.ttl),
                    'df_flag': 1 if (ip.flags & 0x2) else 0,
                    'ip_len': ip.len if hasattr(ip, 'len') else len(packet),

                    'src_port': tcp.sport,
                    'dst_port': tcp.dport,
                    'tcp_window_size': tcp.window,
                    'tcp_flags': int(tcp.flags),
                    'tcp_mss': extract_tcp_mss(tcp),
                    'tcp_window_scale': extract_tcp_window_scale(tcp),
                    'tcp_options_order': extract_tcp_options_order(tcp),

                    'os_label': os_label,
                    'os_family': os_family,
                }

                records.append(record)

    except Exception as e:
        print(f"\nERROR reading pcapng: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Summary
    if verbose:
        print(f"\nPacket statistics:")
        print(f"  Total packets in file: {total_packets}")
        print(f"  TCP packets: {tcp_count}")
        print(f"  SYN packets: {syn_count}")
        print(f"  Packets extracted: {len(records)}")
        print(f"  Packets with OS labels: {labeled_count}")

        if labeled_count == 0:
            print(f"\n  ⚠️  WARNING: No OS labels found!")
            print(f"  This might not be an nprint-formatted pcapng.")

    if not records:
        print("\nERROR: No packets extracted!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Statistics
    if verbose:
        print(f"\nFeature completeness:")
        for feat in ['ttl', 'tcp_window_size', 'tcp_mss', 'tcp_options_order']:
            if feat in df.columns:
                pct = (df[feat].notna().sum() / len(df)) * 100
                print(f"  {feat}: {pct:.1f}%")

        print(f"\nTCP option patterns found:")
        if 'tcp_options_order' in df.columns:
            for pattern, count in df['tcp_options_order'].value_counts().head(5).items():
                print(f"  {pattern}: {count} packets")

        if labeled_count > 0:
            print(f"\nOS Label distribution:")
            for label, count in df['os_label'].value_counts().head(10).items():
                print(f"  {label}: {count}")

            print(f"\nOS Family distribution:")
            for family, count in df['os_family'].value_counts().items():
                print(f"  {family}: {count}")

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract TCP/IP features and OS labels from nprint PCAPNG files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_nprint_pcapng.py nprint_file.pcapng output.csv
  python extract_nprint_pcapng.py nprint_file.pcapng output.csv --all-tcp

Requirements:
  pip install pypcapng scapy pandas
        """
    )

    parser.add_argument('input', help='Input PCAPNG file')
    parser.add_argument('output', help='Output CSV file')
    parser.add_argument('--all-tcp', action='store_true', help='Extract all TCP (default: SYN only)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Validate
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    # Process
    print("="*70)
    print("NPRINT PCAPNG EXTRACTOR")
    print("="*70)

    df = process_pcapng(
        pcap_path=input_path,
        syn_only=not args.all_tcp,
        verbose=not args.quiet
    )

    if df is None or len(df) == 0:
        print("\nERROR: No data extracted!")
        sys.exit(1)

    # Save
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
        preview_cols = ['record_id', 'src_ip', 'ttl', 'tcp_window_size', 'tcp_options_order', 'os_label', 'os_family']
        available_cols = [c for c in preview_cols if c in df.columns]
        print(df[available_cols].head())

        print(f"\nColumn names:")
        for col in df.columns:
            print(f"  - {col}")

    print(f"\n✓ Dataset ready for training!")


if __name__ == '__main__':
    main()
