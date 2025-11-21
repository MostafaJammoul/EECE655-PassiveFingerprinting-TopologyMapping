#!/usr/bin/env python3
"""
Preprocess CESNET Idle Dataset - Extract Packet-Level Features for OS Fingerprinting

Input:  data/raw/cesnet/*.pcap (organized by OS in subdirectories)
Output: data/processed/cesnet.csv

Extracts TCP/IP fingerprinting features from SYN packets for OS version classification.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# Scapy for PCAP parsing
try:
    from scapy.all import rdpcap, IP, TCP, Ether
    from scapy.layers.inet import TCP
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


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_tcp_options_order(tcp_packet):
    """
    Extract the ORDER of TCP options - highly discriminative for OS fingerprinting!

    Example: "MSS,NOP,WS,NOP,NOP,SACK" uniquely identifies certain OS versions
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


def extract_tcp_timestamp(tcp_packet):
    """
    Extract TCP Timestamp option (CRITICAL for OS fingerprinting!)

    Returns:
        (TSval, TSecr) tuple or (None, None) if not present

    Different OSes use different timestamp granularities:
    - Linux: ~1ms (HZ=1000)
    - Windows: ~100ms
    - macOS: ~10ms
    """
    if not hasattr(tcp_packet, 'options'):
        return None, None

    for opt in tcp_packet.options:
        if isinstance(opt, tuple) and len(opt) >= 2:
            if opt[0] == 'Timestamp':
                # opt[1] is a tuple of (TSval, TSecr)
                if isinstance(opt[1], tuple) and len(opt[1]) >= 2:
                    return opt[1][0], opt[1][1]
                elif isinstance(opt[1], (list, tuple)) and len(opt[1]) >= 1:
                    ts_val = opt[1][0] if len(opt[1]) > 0 else None
                    ts_ecr = opt[1][1] if len(opt[1]) > 1 else None
                    return ts_val, ts_ecr
    return None, None


def extract_tcp_sack_permitted(tcp_packet):
    """
    Check if SACK (Selective Acknowledgment) is permitted

    Modern feature - adoption varies by OS
    """
    if not hasattr(tcp_packet, 'options'):
        return 0

    for opt in tcp_packet.options:
        if isinstance(opt, tuple) and opt[0] == 'SAckOK':
            return 1
        elif isinstance(opt, str) and opt == 'SAckOK':
            return 1
    return 0


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
    
    NOTE: This function is NOT used during preprocessing to avoid data leakage.
    OS family is derived from the target (os_label), which would allow the model
    to trivially predict the OS without learning real fingerprinting patterns.
    
    This function is kept for optional post-hoc analysis only.

    Examples:
    - "Windows 11" → "Windows"
    - "Ubuntu 22.04" → "Linux"
    - "macOS 14" → "macOS"
    """
    os_lower = str(os_label).lower()

    if any(w in os_lower for w in ['windows', 'win10', 'win11', 'win7', 'win8']):
        return 'Windows'
    elif any(w in os_lower for w in ['ubuntu', 'debian', 'fedora', 'centos', 'linux', 'kali', 'mint', 'arch']):
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


# ============================================================================
# PCAP PROCESSING
# ============================================================================

def process_pcap_file(pcap_path, os_label, verbose=False, syn_only=True):
    """
    Process a single PCAP file and extract features from TCP packets

    Args:
        pcap_path: Path to PCAP file
        os_label: OS label (e.g., "Windows 11", "Ubuntu 22.04")
        verbose: Print detailed info
        syn_only: If True, only process SYN packets. If False, process ALL TCP packets.

    Returns:
        List of feature dictionaries (one per packet)
    """
    records = []

    try:
        packets = rdpcap(str(pcap_path))
    except Exception as e:
        if verbose:
            print(f"  ERROR reading {pcap_path.name}: {e}")
        return []

    processed_count = 0
    total_tcp_packets = 0

    for pkt_idx, packet in enumerate(packets):
        # Must have IP and TCP layers
        if not (packet.haslayer(IP) and packet.haslayer(TCP)):
            continue

        total_tcp_packets += 1
        ip_layer = packet[IP]
        tcp_layer = packet[TCP]

        # Check TCP flags
        is_syn = (tcp_layer.flags & 0x02) != 0  # Check SYN flag
        is_ack = (tcp_layer.flags & 0x10) != 0  # Check ACK flag
        is_fin = (tcp_layer.flags & 0x01) != 0  # Check FIN flag
        is_rst = (tcp_layer.flags & 0x04) != 0  # Check RST flag
        is_psh = (tcp_layer.flags & 0x08) != 0  # Check PSH flag

        # Filter based on syn_only parameter
        if syn_only and not is_syn:
            continue

        processed_count += 1

        # Determine packet type for labeling
        packet_type = []
        if is_syn: packet_type.append('SYN')
        if is_ack: packet_type.append('ACK')
        if is_fin: packet_type.append('FIN')
        if is_rst: packet_type.append('RST')
        if is_psh: packet_type.append('PSH')
        packet_type_str = '+'.join(packet_type) if packet_type else 'NONE'

        # Extract TCP timestamp
        tcp_ts_val, tcp_ts_ecr = extract_tcp_timestamp(tcp_layer)

        # Extract TCP sequence and acknowledgment numbers
        tcp_seq = tcp_layer.seq if hasattr(tcp_layer, 'seq') else None
        tcp_ack_num = tcp_layer.ack if hasattr(tcp_layer, 'ack') else None

        # Extract Initial Sequence Number (ISN) for SYN packets
        # ISN is the sequence number of the SYN packet (critical for OS fingerprinting)
        tcp_isn = tcp_seq if is_syn else None

        # Extract features
        record = {
            # Metadata
            'dataset_source': 'cesnet_idle',
            'record_id': f"cesnet_idle_{pcap_path.stem}_pkt{pkt_idx}",
            'timestamp': float(packet.time) if hasattr(packet, 'time') else None,
            'packet_type': packet_type_str,

            # IP layer (ENHANCED with critical features!)
            'src_ip': ip_layer.src,
            'dst_ip': ip_layer.dst,
            'protocol': ip_layer.proto,
            'ttl': ip_layer.ttl,
            'initial_ttl': calculate_initial_ttl(ip_layer.ttl),
            'df_flag': 1 if (ip_layer.flags & 0x2) else 0,  # Don't Fragment
            'ip_len': ip_layer.len if hasattr(ip_layer, 'len') else len(packet),
            'ip_id': ip_layer.id if hasattr(ip_layer, 'id') else None,  # CRITICAL: Windows incremental, Linux random
            'ip_tos': ip_layer.tos if hasattr(ip_layer, 'tos') else None,  # MEDIUM: Some OSes set distinctive values

            # TCP layer
            'src_port': tcp_layer.sport,
            'dst_port': tcp_layer.dport,
            'tcp_window_size': tcp_layer.window,
            'tcp_flags': int(tcp_layer.flags),
            'tcp_urgent_ptr': tcp_layer.urgptr if hasattr(tcp_layer, 'urgptr') else None,  # LOW but easy

            # TCP sequence and acknowledgment (NEW - CRITICAL for ACK-based fingerprinting!)
            'tcp_seq': tcp_seq,  # Sequence number (for all packets)
            'tcp_ack': tcp_ack_num,  # Acknowledgment number (for ACK packets)
            'tcp_isn': tcp_isn,  # Initial Sequence Number (for SYN packets only) - CRITICAL for OS detection

            # TCP options (CRITICAL for fingerprinting!)
            'tcp_mss': extract_tcp_mss(tcp_layer),
            'tcp_window_scale': extract_tcp_window_scale(tcp_layer),
            'tcp_options_order': extract_tcp_options_order(tcp_layer),
            'tcp_timestamp_val': tcp_ts_val,  # CRITICAL: Timestamp value
            'tcp_timestamp_ecr': tcp_ts_ecr,  # CRITICAL: Timestamp echo reply
            'tcp_sack_permitted': extract_tcp_sack_permitted(tcp_layer),  # MEDIUM: Explicit SACK flag

            # Labels
            'os_label': os_label,
            # os_family removed to prevent data leakage - target is specific OS version only
        }

        records.append(record)

    if verbose and processed_count > 0:
        mode_str = "SYN" if syn_only else "TCP"
        print(f"  {pcap_path.name}: extracted {processed_count} {mode_str} packets (out of {total_tcp_packets} TCP packets)")

    return records


def parse_os_label_from_dirname(dirname):
    """
    Parse OS label from CESNET directory naming convention

    Examples:
        "android__android-x86__9.0" → "Android 9.0"
        "linux_mint_21.1-vera" → "Linux Mint 21.1"
        "windows_windows-11_10.0.22631" → "Windows 11"
        "linux_ubuntu_22.04.4-lts" → "Ubuntu 22.04.4 LTS"

    CESNET naming: os-family_os-name_version[-variant]
    """
    # Handle double-underscore separated format first
    if '__' in dirname:
        # e.g., "android__android-x86__9.0" → ["android", "android-x86", "9.0"]
        double_parts = dirname.split('__')
        os_family = double_parts[0].lower()

        # For Android: use family name + last part (version)
        if os_family == 'android':
            version = double_parts[-1] if len(double_parts) > 1 else ""
            return f"Android {version}".strip()

        # For others: reconstruct
        dirname = '_'.join(double_parts)

    # Split by single underscore
    parts = dirname.split('_')

    if len(parts) < 2:
        # Fallback: simple conversion
        return dirname.replace('_', ' ').replace('-', ' ').title()

    # First part is OS family (linux, windows, android, etc.)
    os_family = parts[0].lower()

    # Second part is usually OS name
    os_name = parts[1] if len(parts) > 1 else ""

    # Remaining parts are version
    version_parts = parts[2:] if len(parts) > 2 else []

    # Remove variant suffixes from version (e.g., "-vera", "-victoria")
    # But keep important suffixes like "lts"
    version_str = '_'.join(version_parts)
    if '-' in version_str:
        dash_parts = version_str.split('-')
        # Keep first part (version number) and "lts"
        kept_parts = [dash_parts[0]]
        if any(p.lower() == 'lts' for p in dash_parts[1:]):
            kept_parts.append('lts')
        version_str = '-'.join(kept_parts)

    # Build the label based on OS family
    if os_family == 'windows':
        # "windows_windows-11_10.0.22631" → "Windows 11"
        # Extract version from os_name (e.g., "windows-11" → "11")
        if 'windows' in os_name.lower():
            win_version = os_name.replace('windows-', '').replace('windows', '')
            return f"Windows {win_version}".strip()
        return f"Windows {os_name}".strip()

    elif os_family == 'linux':
        # Handle specific distributions
        os_name_lower = os_name.lower()

        if 'ubuntu' in os_name_lower:
            os_label = f"Ubuntu {version_str.replace('_', '.').replace('-', ' ')}"
        elif 'mint' in os_name_lower:
            os_label = f"Linux Mint {version_str.replace('_', '.').replace('-', ' ')}"
        elif 'debian' in os_name_lower:
            os_label = f"Debian {version_str.replace('_', '.').replace('-', ' ')}"
        elif 'fedora' in os_name_lower:
            os_label = f"Fedora {version_str.replace('_', '.').replace('-', ' ')}"
        elif 'centos' in os_name_lower:
            os_label = f"CentOS {version_str.replace('_', '.').replace('-', ' ')}"
        elif 'openbsd' in os_name_lower:
            os_label = f"OpenBSD {version_str.replace('_', '.')}"
        elif 'freebsd' in os_name_lower:
            os_label = f"FreeBSD {version_str.replace('_', '.')}"
        elif 'opensuse' in os_name_lower or ('suse' in os_name_lower and 'leap' in version_str.lower()):
            # "opensuse-leap_15.2" → "openSUSE Leap 15.2"
            version_str = version_str.replace('leap_', '').replace('leap-', '')
            os_label = f"openSUSE Leap {version_str.replace('_', '.')}"
        elif 'red' in os_name_lower or 'hat' in os_name_lower or 'rhel' in os_name_lower:
            # "red-hat-enterprise-linux_8.9" → "Red Hat Enterprise Linux 8.9"
            full_name = '_'.join(parts[1:])  # Get everything after 'linux'
            full_name = full_name.split('_')[:-1]  # Remove version
            distro_name = '-'.join(full_name).replace('-', ' ').title()
            if 'Red Hat' in distro_name or 'Enterprise' in distro_name:
                os_label = f"{distro_name} {version_str.replace('_', '.')}"
            else:
                os_label = f"RHEL {version_str.replace('_', '.')}"
        elif 'arch' in os_name_lower:
            os_label = f"Arch Linux {version_str.replace('_', '.')}"
        elif 'kali' in os_name_lower:
            os_label = f"Kali Linux {version_str.replace('_', '.')}"
        else:
            # Generic Linux
            os_label = f"{os_name.title()} {version_str.replace('_', '.')}"

        # Capitalize LTS properly
        os_label = os_label.replace(' lts', ' LTS').replace(' Lts', ' LTS')
        return os_label.strip()

    elif os_family == 'android':
        return f"Android {os_name} {version_str.replace('_', '.')}".strip()

    elif os_family in ['macos', 'darwin']:
        return f"macOS {os_name} {version_str.replace('_', '.')}".strip()

    else:
        # Unknown OS family - do best effort
        label = f"{os_name} {version_str}".replace('_', ' ').replace('-', ' ').title()
        return label.strip()


def discover_pcap_files(raw_dir):
    """
    Discover all PCAP files and infer OS labels from directory structure

    Supports nested structure:
        data/raw/cesnet_idle/
        ├── android__android-x86__9.0/
        │   └── 2025-02-04__osboxes.org__2eb9b5/
        │       └── traffic.pcap
        ├── linux_ubuntu_22.04.4-lts/
        │   └── 2025-02-05__*/
        │       └── traffic.pcap
        └── ...

    Returns:
        List of (pcap_path, os_label) tuples
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        print(f"ERROR: Directory not found: {raw_dir}")
        return []

    pcap_files = []

    # Look for PCAP files in subdirectories (supports nested structure)
    for subdir in raw_path.iterdir():
        if not subdir.is_dir():
            continue

        # Use subdirectory name as OS label
        os_label_raw = subdir.name
        os_label = parse_os_label_from_dirname(os_label_raw)

        # Find all PCAP files in this subdirectory (recursive search)
        for pcap_file in subdir.glob('**/*.pcap'):
            pcap_files.append((pcap_file, os_label))

    # Also check for PCAP files directly in raw_dir (flat structure)
    for pcap_file in raw_path.glob('*.pcap'):
        # Try to infer OS from filename
        os_label = parse_os_label_from_dirname(pcap_file.stem)
        pcap_files.append((pcap_file, os_label))

    return pcap_files


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def preprocess_cesnet_idle(raw_dir='data/raw/cesnet',
                           output_dir='data/processed',
                           max_files=None,
                           verbose=True,
                           syn_only=True):
    """
    Main preprocessing pipeline for CESNET Idle dataset

    Args:
        raw_dir: Directory containing PCAP files
        output_dir: Where to save processed CSV
        max_files: Limit number of files (for testing)
        verbose: Print progress
        syn_only: If True, extract only SYN packets. If False, extract ALL TCP packets.

    Returns:
        DataFrame with extracted features
    """

    print("="*70)
    print("CESNET IDLE DATASET PREPROCESSING")
    print("="*70)
    print(f"\nInput:  {raw_dir}")
    print(f"Output: {output_dir}/cesnet.csv")
    print(f"Mode:   {'SYN packets only' if syn_only else 'ALL TCP packets'}")

    # Discover PCAP files
    print(f"\n[1/3] Discovering PCAP files...")
    pcap_files = discover_pcap_files(raw_dir)

    if not pcap_files:
        print(f"\nERROR: No PCAP files found in {raw_dir}")
        print("\nExpected structure (supports nested directories):")
        print("  data/raw/cesnet/")
        print("  ├── android__android-x86__9.0/")
        print("  │   └── 2025-02-04__osboxes.org__*/")
        print("  │       └── traffic.pcap")
        print("  ├── linux_ubuntu_22.04.4-lts/")
        print("  │   └── */")
        print("  │       └── traffic.pcap")
        print("  └── ...")
        return None

    print(f"  Found {len(pcap_files)} PCAP files")

    # Show OS distribution
    os_labels = [label for _, label in pcap_files]
    os_counts = pd.Series(os_labels).value_counts()
    print(f"\n  OS distribution:")
    for os_name, count in os_counts.head(10).items():
        print(f"    {os_name}: {count} files")
    if len(os_counts) > 10:
        print(f"    ... and {len(os_counts) - 10} more")

    # Limit if requested
    if max_files:
        pcap_files = pcap_files[:max_files]
        print(f"\n  Limited to {max_files} files for testing")

    # Process all PCAP files
    mode_desc = "SYN packets" if syn_only else "TCP packets"
    print(f"\n[2/3] Extracting features from PCAP files...")
    all_records = []

    for pcap_path, os_label in tqdm(pcap_files, desc="Processing PCAPs"):
        records = process_pcap_file(pcap_path, os_label, verbose=False, syn_only=syn_only)
        all_records.extend(records)

    if not all_records:
        print(f"\nERROR: No {mode_desc} found in any PCAP files!")
        print("This is unusual - PCAP files should contain TCP packets.")
        print("\nTroubleshooting:")
        print("  1. Verify PCAP files are not corrupted")
        print("  2. Check if files contain TCP traffic (use: tshark -r file.pcap -Y tcp)")
        if syn_only:
            print("  3. Try running with --all-tcp flag to extract all TCP packets")
        return None

    print(f"  Extracted {len(all_records):,} {mode_desc}")

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    print(f"\n[3/3] Saving processed dataset...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cesnet.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nDataset shape: {df.shape}")
    print(f"  Records: {len(df):,}")
    print(f"  Features: {len(df.columns)}")

    print(f"\nOS Version distribution (top 10):")
    print(df['os_label'].value_counts().head(10))

    print(f"\nFeature completeness:")
    critical_features = ['ttl', 'tcp_window_size', 'tcp_mss', 'tcp_options_order',
                        'tcp_timestamp_val', 'tcp_seq', 'tcp_ack', 'tcp_isn',
                        'ip_id', 'ip_tos']
    for feat in critical_features:
        pct_available = (df[feat].notna().sum() / len(df)) * 100
        status = "✓" if pct_available > 80 else "⚠"
        print(f"  {status} {feat}: {pct_available:.1f}%")

    return df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess CESNET Idle dataset - extract packet-level features'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/cesnet',
        help='Input directory with PCAP files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed CSV'
    )

    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Limit number of PCAP files (for testing)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    parser.add_argument(
        '--all-tcp',
        action='store_true',
        help='Extract ALL TCP packets (not just SYN). Increases dataset size 10-100x.'
    )

    args = parser.parse_args()

    # Run preprocessing
    df = preprocess_cesnet_idle(
        raw_dir=args.input,
        output_dir=args.output,
        max_files=args.max_files,
        verbose=not args.quiet,
        syn_only=not args.all_tcp  # If --all-tcp is set, syn_only=False
    )

    if df is None:
        sys.exit(1)

    print("\n✓ Success! Dataset ready for Model 2 training.")
    print(f"\nNext steps:")
    print(f"  1. Preprocess nprint: python scripts/preprocess_nprint.py")
    print(f"  2. Merge datasets: python scripts/merge_packet_datasets.py")


if __name__ == '__main__':
    main()