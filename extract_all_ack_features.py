#!/usr/bin/env python3
"""
UNIFIED ACK PACKET FEATURE EXTRACTOR
=====================================

Extracts ALL features from passive/idle ACK packets in both:
1. nprint dataset (os-100-packet.pcapng) - PCAP with embedded labels
2. CESNET idle dataset - Directory-based PCAP files

Features Extracted (31 total):
- Metadata (4): record_id, dataset_source, timestamp, packet_type
- IP Layer (8): ttl, initial_ttl, df_flag, ip_id, ip_tos, ip_len, src_ip, dst_ip
- TCP Layer (10): window_size, mss, window_scale, options_order, timestamp_val,
                  timestamp_ecr, sack_permitted, tcp_flags, urgent_ptr, payload_len
- TCP Sequence (3): tcp_seq, tcp_ack, tcp_isn
- IPT Features (7): ipt, ipt_next, ipt_mean, ipt_std, ipt_min, ipt_max, ipt_median

Usage:
    # Extract from nprint dataset (ACTIVE traffic, filter for idle ACKs)
    python extract_all_ack_features.py --dataset nprint --input data/raw/nprint/os-100-packet.pcapng --output features_nprint.csv

    # Extract from CESNET dataset (IDLE traffic, all ACKs)
    python extract_all_ack_features.py --dataset cesnet --input data/raw/cesnet --output features_cesnet.csv

    # Extract only idle ACKs (zero payload, no PSH flag)
    python extract_all_ack_features.py --dataset nprint --input data/raw/nprint/os-100-packet.pcapng --output features_idle.csv --idle-only

    # Extract all ACK packets (including PSH-ACK)
    python extract_all_ack_features.py --dataset nprint --input data/raw/nprint/os-100-packet.pcapng --output features_all.csv --all-ack
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

try:
    from scapy.all import rdpcap, TCP, IP, Packet, PcapNgReader
    from scapy.layers.inet import IP, TCP
except ImportError:
    print("ERROR: scapy is required. Install with: pip install scapy")
    sys.exit(1)

import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

# Initial TTL values for common operating systems
INITIAL_TTL_VALUES = [32, 64, 128, 255]

# TCP option kinds
TCP_OPT_EOL = 0
TCP_OPT_NOP = 1
TCP_OPT_MSS = 2
TCP_OPT_WINDOW_SCALE = 3
TCP_OPT_SACK_PERMITTED = 4
TCP_OPT_SACK = 5
TCP_OPT_TIMESTAMP = 8

# Packet type classification
PACKET_TYPE_IDLE_ACK = "idle_ack"
PACKET_TYPE_ACTIVE_ACK = "active_ack"
PACKET_TYPE_PSH_ACK = "psh_ack"
PACKET_TYPE_SYN = "syn"
PACKET_TYPE_SYN_ACK = "syn_ack"
PACKET_TYPE_FIN_ACK = "fin_ack"
PACKET_TYPE_RST = "rst"
PACKET_TYPE_OTHER = "other"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def infer_initial_ttl(ttl: int) -> int:
    """Infer the initial TTL value from the observed TTL."""
    for initial_ttl in INITIAL_TTL_VALUES:
        if ttl <= initial_ttl:
            return initial_ttl
    return 255


def classify_packet_type(packet) -> str:
    """Classify packet type based on TCP flags and payload."""
    if not packet.haslayer(TCP):
        return PACKET_TYPE_OTHER

    tcp = packet[TCP]
    flags = tcp.flags
    payload_len = len(tcp.payload) if tcp.payload else 0

    # Check flags
    has_syn = flags & 0x02
    has_ack = flags & 0x10
    has_fin = flags & 0x01
    has_rst = flags & 0x04
    has_psh = flags & 0x08

    # Classify
    if has_rst:
        return PACKET_TYPE_RST
    elif has_syn and has_ack:
        return PACKET_TYPE_SYN_ACK
    elif has_syn:
        return PACKET_TYPE_SYN
    elif has_fin and has_ack:
        return PACKET_TYPE_FIN_ACK
    elif has_ack:
        if has_psh and payload_len > 0:
            return PACKET_TYPE_PSH_ACK
        elif payload_len > 0:
            return PACKET_TYPE_ACTIVE_ACK
        else:
            return PACKET_TYPE_IDLE_ACK

    return PACKET_TYPE_OTHER


def parse_tcp_options(packet) -> Dict:
    """Parse TCP options and extract all relevant information."""
    options_data = {
        'mss': None,
        'window_scale': None,
        'sack_permitted': 0,
        'timestamp_val': None,
        'timestamp_ecr': None,
        'options_order': ''
    }

    if not packet.haslayer(TCP):
        return options_data

    tcp = packet[TCP]
    if not hasattr(tcp, 'options') or not tcp.options:
        return options_data

    option_kinds = []
    for opt in tcp.options:
        if isinstance(opt, tuple):
            opt_kind = opt[0]
            opt_value = opt[1] if len(opt) > 1 else None

            # Record option kind
            if isinstance(opt_kind, str):
                # Map string names to numeric values
                kind_map = {
                    'MSS': TCP_OPT_MSS,
                    'WScale': TCP_OPT_WINDOW_SCALE,
                    'SAckOK': TCP_OPT_SACK_PERMITTED,
                    'Timestamp': TCP_OPT_TIMESTAMP,
                    'SAck': TCP_OPT_SACK,
                    'NOP': TCP_OPT_NOP,
                    'EOL': TCP_OPT_EOL
                }
                opt_kind_num = kind_map.get(opt_kind, 255)
            else:
                opt_kind_num = opt_kind

            option_kinds.append(str(opt_kind_num))

            # Extract specific option values
            if opt_kind in ['MSS', TCP_OPT_MSS]:
                options_data['mss'] = opt_value
            elif opt_kind in ['WScale', TCP_OPT_WINDOW_SCALE]:
                options_data['window_scale'] = opt_value
            elif opt_kind in ['SAckOK', TCP_OPT_SACK_PERMITTED]:
                options_data['sack_permitted'] = 1
            elif opt_kind in ['Timestamp', TCP_OPT_TIMESTAMP]:
                if isinstance(opt_value, tuple) and len(opt_value) >= 2:
                    options_data['timestamp_val'] = opt_value[0]
                    options_data['timestamp_ecr'] = opt_value[1]

    # Create options order string
    options_data['options_order'] = ','.join(option_kinds) if option_kinds else ''

    return options_data


def extract_packet_features(packet, packet_idx: int, dataset_source: str,
                           flow_data: Optional[Dict] = None, label: Optional[str] = None) -> Dict:
    """Extract ALL features from a single packet."""
    features = {
        # Metadata
        'record_id': packet_idx,
        'dataset_source': dataset_source,
        'timestamp': float(packet.time) if hasattr(packet, 'time') else 0.0,
        'packet_type': classify_packet_type(packet),
        'label': label if label else '',

        # IP Layer
        'ttl': None,
        'initial_ttl': None,
        'df_flag': 0,
        'ip_id': None,
        'ip_tos': None,
        'ip_len': None,
        'src_ip': '',
        'dst_ip': '',

        # TCP Layer
        'window_size': None,
        'mss': None,
        'window_scale': None,
        'options_order': '',
        'timestamp_val': None,
        'timestamp_ecr': None,
        'sack_permitted': 0,
        'tcp_flags': 0,
        'urgent_ptr': 0,
        'payload_len': 0,
        'src_port': None,
        'dst_port': None,

        # TCP Sequence
        'tcp_seq': None,
        'tcp_ack': None,
        'tcp_isn': None,

        # IPT Features (will be calculated later per flow)
        'ipt': None,
        'ipt_next': None,
        'ipt_mean': None,
        'ipt_std': None,
        'ipt_min': None,
        'ipt_max': None,
        'ipt_median': None
    }

    # Extract IP layer features
    if packet.haslayer(IP):
        ip = packet[IP]
        features['ttl'] = ip.ttl
        features['initial_ttl'] = infer_initial_ttl(ip.ttl)
        features['df_flag'] = 1 if (ip.flags & 0x02) else 0
        features['ip_id'] = ip.id
        features['ip_tos'] = ip.tos
        features['ip_len'] = ip.len
        features['src_ip'] = ip.src
        features['dst_ip'] = ip.dst

    # Extract TCP layer features
    if packet.haslayer(TCP):
        tcp = packet[TCP]
        features['window_size'] = tcp.window
        features['tcp_flags'] = int(tcp.flags)
        features['urgent_ptr'] = tcp.urgptr if hasattr(tcp, 'urgptr') else 0
        features['payload_len'] = len(tcp.payload) if tcp.payload else 0
        features['src_port'] = tcp.sport
        features['dst_port'] = tcp.dport

        # TCP sequence numbers
        features['tcp_seq'] = tcp.seq
        features['tcp_ack'] = tcp.ack if tcp.flags & 0x10 else None

        # Parse TCP options
        options = parse_tcp_options(packet)
        features.update(options)

        # ISN (Initial Sequence Number) - get from flow data if available
        if flow_data:
            flow_key = f"{features['src_ip']}:{features['src_port']}-{features['dst_ip']}:{features['dst_port']}"
            if flow_key in flow_data:
                features['tcp_isn'] = flow_data[flow_key].get('isn', None)

    return features


def calculate_ipt_features(packets_by_flow: Dict[str, List[Dict]]) -> None:
    """Calculate IPT (Inter-Packet Time) features for all packets in each flow."""
    for flow_key, packets in packets_by_flow.items():
        if len(packets) < 2:
            continue

        # Sort packets by timestamp
        packets.sort(key=lambda p: p['timestamp'])

        # Calculate IPT for each packet
        ipts = []
        for i in range(len(packets) - 1):
            ipt = packets[i + 1]['timestamp'] - packets[i]['timestamp']
            packets[i]['ipt_next'] = ipt
            ipts.append(ipt)

        # Last packet has no next IPT
        packets[-1]['ipt_next'] = None

        # Calculate IPT statistics for the flow
        if ipts:
            ipt_mean = np.mean(ipts)
            ipt_std = np.std(ipts)
            ipt_min = np.min(ipts)
            ipt_max = np.max(ipts)
            ipt_median = np.median(ipts)

            # Assign flow-level statistics to all packets
            for i, packet in enumerate(packets):
                if i > 0:
                    packet['ipt'] = packets[i]['timestamp'] - packets[i - 1]['timestamp']
                else:
                    packet['ipt'] = None

                packet['ipt_mean'] = ipt_mean
                packet['ipt_std'] = ipt_std
                packet['ipt_min'] = ipt_min
                packet['ipt_max'] = ipt_max
                packet['ipt_median'] = ipt_median


def get_flow_key(packet) -> Optional[str]:
    """Generate a flow key from packet."""
    if not packet.haslayer(IP) or not packet.haslayer(TCP):
        return None

    ip = packet[IP]
    tcp = packet[TCP]

    # Bidirectional flow key (sorted to group both directions)
    src = f"{ip.src}:{tcp.sport}"
    dst = f"{ip.dst}:{tcp.dport}"

    if src < dst:
        return f"{src}-{dst}"
    else:
        return f"{dst}-{src}"


# ============================================================================
# NPRINT DATASET PROCESSING
# ============================================================================

def extract_label_from_pcapng_comment(packet) -> Optional[str]:
    """Extract OS label from PCAPNG packet comment."""
    if hasattr(packet, 'comment') and packet.comment:
        comment = packet.comment.decode('utf-8') if isinstance(packet.comment, bytes) else packet.comment
        # Format: "sampleID,os_family_os_version"
        parts = comment.split(',')
        if len(parts) >= 2:
            return parts[1].strip()
    return None


def process_nprint_dataset(pcap_path: str, output_path: str,
                          idle_only: bool = False, all_ack: bool = False) -> None:
    """Process nprint dataset and extract all ACK features."""
    print("=" * 70)
    print("NPRINT DATASET FEATURE EXTRACTION")
    print("=" * 70)
    print(f"\nInput:  {pcap_path}")
    print(f"Output: {output_path}")
    print(f"Mode:   {'Idle ACKs only' if idle_only else 'All ACKs' if all_ack else 'All TCP packets'}")
    print("\nReading packets...")

    # Read packets
    try:
        packets = rdpcap(pcap_path)
    except:
        print("  Note: Using PcapNgReader for .pcapng format...")
        packets = []
        with PcapNgReader(pcap_path) as reader:
            for packet in reader:
                packets.append(packet)

    print(f"  Loaded {len(packets):,} packets")

    # Track flows and ISNs
    flow_isn = {}  # flow_key -> ISN
    packets_by_flow = defaultdict(list)

    # First pass: identify SYN packets to get ISNs
    print("\nIdentifying flows and ISNs...")
    for packet in packets:
        if not packet.haslayer(TCP):
            continue

        tcp = packet[TCP]
        if tcp.flags & 0x02:  # SYN flag
            flow_key = get_flow_key(packet)
            if flow_key and flow_key not in flow_isn:
                flow_isn[flow_key] = tcp.seq

    print(f"  Found {len(flow_isn):,} flows")

    # Second pass: extract features
    print("\nExtracting features...")
    all_features = []
    packet_idx = 0

    for i, packet in enumerate(packets):
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1:,} packets...")

        # Filter for TCP packets
        if not packet.haslayer(TCP):
            continue

        # Classify packet
        packet_type = classify_packet_type(packet)

        # Apply filters
        if idle_only and packet_type != PACKET_TYPE_IDLE_ACK:
            continue
        elif all_ack and packet_type not in [PACKET_TYPE_IDLE_ACK, PACKET_TYPE_ACTIVE_ACK, PACKET_TYPE_PSH_ACK]:
            continue

        # Extract label from PCAPNG comment
        label = extract_label_from_pcapng_comment(packet)

        # Get flow ISN
        flow_key = get_flow_key(packet)
        flow_data = {}
        if flow_key and flow_key in flow_isn:
            flow_data[flow_key] = {'isn': flow_isn[flow_key]}

        # Extract features
        features = extract_packet_features(packet, packet_idx, 'nprint', flow_data, label)
        all_features.append(features)

        # Group by flow for IPT calculation
        if flow_key:
            packets_by_flow[flow_key].append(features)

        packet_idx += 1

    # Calculate IPT features
    print("\nCalculating IPT features...")
    calculate_ipt_features(packets_by_flow)

    # Write to CSV
    print(f"\nWriting {len(all_features):,} records to CSV...")
    write_features_to_csv(all_features, output_path)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nTotal packets extracted: {len(all_features):,}")
    print(f"Output file: {output_path}")

    # Statistics
    packet_types = defaultdict(int)
    for feat in all_features:
        packet_types[feat['packet_type']] += 1

    print("\nPacket Type Distribution:")
    for ptype, count in sorted(packet_types.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(all_features)
        print(f"  {ptype:20s}: {count:8,} ({pct:5.2f}%)")


# ============================================================================
# CESNET DATASET PROCESSING
# ============================================================================

def parse_cesnet_path(pcap_path: str) -> Optional[str]:
    """Extract OS label from CESNET directory structure."""
    # Expected format: .../os-family_os-name_version/*/traffic.pcap
    parts = Path(pcap_path).parts

    for part in parts:
        # Look for directory matching pattern: os-family_os-name_version
        if part.startswith('os-') or '_' in part:
            # Clean up the label
            label = part.replace('os-', '').replace('_', ' ')
            return label

    return None


def find_cesnet_pcap_files(cesnet_root: str) -> List[Tuple[str, str]]:
    """Find all CESNET PCAP files and their labels."""
    pcap_files = []
    cesnet_path = Path(cesnet_root)

    # Find all traffic.pcap files
    for pcap_file in cesnet_path.rglob('*.pcap'):
        label = parse_cesnet_path(str(pcap_file))
        if label:
            pcap_files.append((str(pcap_file), label))

    return pcap_files


def process_cesnet_dataset(cesnet_root: str, output_path: str,
                          idle_only: bool = False, all_ack: bool = False) -> None:
    """Process CESNET dataset and extract all ACK features."""
    print("=" * 70)
    print("CESNET DATASET FEATURE EXTRACTION")
    print("=" * 70)
    print(f"\nInput:  {cesnet_root}")
    print(f"Output: {output_path}")
    print(f"Mode:   {'Idle ACKs only' if idle_only else 'All ACKs' if all_ack else 'All TCP packets'}")

    # Find all PCAP files
    print("\nScanning for PCAP files...")
    pcap_files = find_cesnet_pcap_files(cesnet_root)

    if not pcap_files:
        print(f"ERROR: No PCAP files found in {cesnet_root}")
        return

    print(f"  Found {len(pcap_files):,} PCAP files")

    # Process all PCAP files
    all_features = []
    packet_idx = 0

    for file_num, (pcap_path, label) in enumerate(pcap_files, 1):
        print(f"\n[{file_num}/{len(pcap_files)}] Processing: {Path(pcap_path).name}")
        print(f"  Label: {label}")

        # Read packets
        try:
            packets = rdpcap(pcap_path)
        except Exception as e:
            print(f"  ERROR reading {pcap_path}: {e}")
            continue

        print(f"  Loaded {len(packets):,} packets")

        # Track flows and ISNs
        flow_isn = {}
        packets_by_flow = defaultdict(list)

        # First pass: identify SYN packets
        for packet in packets:
            if not packet.haslayer(TCP):
                continue

            tcp = packet[TCP]
            if tcp.flags & 0x02:  # SYN flag
                flow_key = get_flow_key(packet)
                if flow_key and flow_key not in flow_isn:
                    flow_isn[flow_key] = tcp.seq

        # Second pass: extract features
        file_features = []
        for packet in packets:
            # Filter for TCP packets
            if not packet.haslayer(TCP):
                continue

            # Classify packet
            packet_type = classify_packet_type(packet)

            # Apply filters
            if idle_only and packet_type != PACKET_TYPE_IDLE_ACK:
                continue
            elif all_ack and packet_type not in [PACKET_TYPE_IDLE_ACK, PACKET_TYPE_ACTIVE_ACK, PACKET_TYPE_PSH_ACK]:
                continue

            # Get flow ISN
            flow_key = get_flow_key(packet)
            flow_data = {}
            if flow_key and flow_key in flow_isn:
                flow_data[flow_key] = {'isn': flow_isn[flow_key]}

            # Extract features
            features = extract_packet_features(packet, packet_idx, 'cesnet', flow_data, label)
            file_features.append(features)

            # Group by flow for IPT calculation
            if flow_key:
                packets_by_flow[flow_key].append(features)

            packet_idx += 1

        # Calculate IPT features for this file
        calculate_ipt_features(packets_by_flow)

        all_features.extend(file_features)
        print(f"  Extracted {len(file_features):,} packets")

    # Write to CSV
    print(f"\n\nWriting {len(all_features):,} records to CSV...")
    write_features_to_csv(all_features, output_path)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nTotal packets extracted: {len(all_features):,}")
    print(f"Total files processed: {len(pcap_files):,}")
    print(f"Output file: {output_path}")

    # Statistics
    packet_types = defaultdict(int)
    labels = defaultdict(int)
    for feat in all_features:
        packet_types[feat['packet_type']] += 1
        if feat['label']:
            labels[feat['label']] += 1

    print("\nPacket Type Distribution:")
    for ptype, count in sorted(packet_types.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(all_features)
        print(f"  {ptype:20s}: {count:8,} ({pct:5.2f}%)")

    print(f"\nOS Labels: {len(labels)} unique labels")
    for label, count in sorted(labels.items(), key=lambda x: -x[1])[:10]:
        pct = 100.0 * count / len(all_features)
        print(f"  {label:30s}: {count:8,} ({pct:5.2f}%)")
    if len(labels) > 10:
        print(f"  ... and {len(labels) - 10} more")


# ============================================================================
# CSV OUTPUT
# ============================================================================

def write_features_to_csv(features: List[Dict], output_path: str) -> None:
    """Write extracted features to CSV file."""
    if not features:
        print("WARNING: No features to write")
        return

    # Define column order
    columns = [
        # Metadata
        'record_id', 'dataset_source', 'timestamp', 'packet_type', 'label',

        # IP Layer
        'src_ip', 'dst_ip', 'ttl', 'initial_ttl', 'df_flag', 'ip_id', 'ip_tos', 'ip_len',

        # TCP Layer
        'src_port', 'dst_port', 'window_size', 'mss', 'window_scale', 'options_order',
        'timestamp_val', 'timestamp_ecr', 'sack_permitted', 'tcp_flags', 'urgent_ptr', 'payload_len',

        # TCP Sequence
        'tcp_seq', 'tcp_ack', 'tcp_isn',

        # IPT Features
        'ipt', 'ipt_next', 'ipt_mean', 'ipt_std', 'ipt_min', 'ipt_max', 'ipt_median'
    ]

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(features)

    print(f"  Wrote {len(features):,} rows × {len(columns)} columns")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract all features from passive ACK packets in nprint and CESNET datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--dataset', required=True, choices=['nprint', 'cesnet'],
                       help='Dataset type to process')
    parser.add_argument('--input', required=True,
                       help='Input path (PCAP file for nprint, directory for CESNET)')
    parser.add_argument('--output', required=True,
                       help='Output CSV file path')

    # Filtering options
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('--idle-only', action='store_true',
                             help='Extract only idle ACK packets (zero payload, no PSH)')
    filter_group.add_argument('--all-ack', action='store_true',
                             help='Extract all ACK packets (idle + active + PSH-ACK)')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input path does not exist: {args.input}")
        sys.exit(1)

    # Process dataset
    if args.dataset == 'nprint':
        process_nprint_dataset(args.input, args.output, args.idle_only, args.all_ack)
    elif args.dataset == 'cesnet':
        process_cesnet_dataset(args.input, args.output, args.idle_only, args.all_ack)


if __name__ == '__main__':
    main()
