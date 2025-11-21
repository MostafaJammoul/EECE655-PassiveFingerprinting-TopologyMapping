#!/usr/bin/env python3
"""
Extract TCP SYN Flows from CESNET Dataset (Linux samples only)

Reads flows.csv files from CESNET directories and extracts TCP SYN flows
to augment the Masaryk dataset with additional Linux samples.

Input:  data/raw/cesnet/**/flows.csv (only from linux* directories)
Output: data/processed/cesnet.csv (in Masaryk format)

Only extracts flows from directories starting with "linux" (e.g., linux__ubuntu__18.04.6)
Filters for TCP flows with SYN flag present.
Maps CESNET features to Masaryk format for seamless merging.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import ipaddress

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# TCP OPTIONS DECODING
# ============================================================================

# TCP Option Bitmask Flags (CESNET encoding)
TCP_OPT_MSS = 0x01          # Maximum Segment Size
TCP_OPT_WS = 0x02           # Window Scale
TCP_OPT_SACK_PERM = 0x04    # SACK Permitted
TCP_OPT_SACK = 0x08         # SACK
TCP_OPT_TIMESTAMP = 0x10    # Timestamps
TCP_OPT_NOP = 0x20          # NOP


def decode_tcp_options(tcp_opt_value):
    """Decode CESNET TCP_OPT bitmask to individual option flags"""
    if tcp_opt_value is None or pd.isna(tcp_opt_value):
        return {
            'window_scale': None,
            'sack_permitted': None,
            'nop': None,
        }

    tcp_opt_value = int(tcp_opt_value)
    return {
        'window_scale': 1 if (tcp_opt_value & TCP_OPT_WS) else 0,
        'sack_permitted': 1 if (tcp_opt_value & TCP_OPT_SACK_PERM) else 0,
        'nop': 1 if (tcp_opt_value & TCP_OPT_NOP) else 0,
    }


def calculate_initial_ttl(ttl):
    """
    Estimate original TTL value based on observed TTL

    Common initial TTLs:
    - 64:  Linux, macOS, Unix
    - 128: Windows
    - 255: Cisco, Solaris
    - 32:  Old systems
    """
    if ttl is None or pd.isna(ttl):
        return None
    ttl = int(ttl)
    common_ttls = [32, 64, 128, 255]
    for initial in common_ttls:
        if ttl <= initial:
            return initial
    return 255


def get_l3_proto(ip_addr):
    """Determine L3 protocol (4=IPv4, 6=IPv6) from IP address string"""
    if not ip_addr or pd.isna(ip_addr):
        return None
    try:
        ip_obj = ipaddress.ip_address(str(ip_addr))
        return 4 if ip_obj.version == 4 else 6
    except:
        return None


def decode_ja3_bytes(ja3_bytes):
    """Decode JA3 fingerprint from bytes to string"""
    if not ja3_bytes or pd.isna(ja3_bytes):
        return None
    try:
        # If it's already a string, return it
        if isinstance(ja3_bytes, str):
            return ja3_bytes
        # If it's bytes, decode
        return ja3_bytes.decode('utf-8')
    except:
        return None


def parse_tls_ext_type(ext_type_array):
    """Parse TLS extension types array to comma-separated string"""
    if not ext_type_array or pd.isna(ext_type_array):
        return None
    try:
        if isinstance(ext_type_array, str):
            return ext_type_array
        # If it's an array, join
        return ','.join(str(x) for x in ext_type_array)
    except:
        return None


def convert_tcp_flags_to_string(tcp_flags_value):
    """
    Convert TCP flags bitmask to Masaryk string format

    Format: C E U A P R S F
    Where:
    - C = CWR (Congestion Window Reduced) - bit 7 (0x80)
    - E = ECE (ECN-Echo) - bit 6 (0x40)
    - U = URG (Urgent) - bit 5 (0x20)
    - A = ACK - bit 4 (0x10)
    - P = PSH (Push) - bit 3 (0x08)
    - R = RST (Reset) - bit 2 (0x04)
    - S = SYN - bit 1 (0x02)
    - F = FIN - bit 0 (0x01)

    Example: 0x12 (SYN+ACK) -> "---A--S-"
    Example: 0x18 (PSH+ACK) -> "---AP---"
    """
    if tcp_flags_value is None or pd.isna(tcp_flags_value):
        return None

    try:
        flags = int(tcp_flags_value)

        result = ""
        result += "C" if (flags & 0x80) else "-"
        result += "E" if (flags & 0x40) else "-"
        result += "U" if (flags & 0x20) else "-"
        result += "A" if (flags & 0x10) else "-"
        result += "P" if (flags & 0x08) else "-"
        result += "R" if (flags & 0x04) else "-"
        result += "S" if (flags & 0x02) else "-"
        result += "F" if (flags & 0x01) else "-"

        return result
    except:
        return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_cesnet_flows(cesnet_dir='data/raw/cesnet',
                         output_dir='data/processed',
                         verbose=True):
    """
    Extract TCP SYN flows from CESNET dataset (Linux directories only)

    Maps CESNET features to Masaryk format for seamless dataset merging.
    Only processes directories starting with "linux".
    """

    print("="*70)
    print("CESNET FLOWS EXTRACTION - LINUX TCP SYN FLOWS ONLY")
    print("="*70)
    print(f"\nInput:  {cesnet_dir}")
    print(f"Output: {output_dir}/cesnet.csv")
    print(f"\nFILTERING:")
    print(f"  - Only Linux directories (linux*)")
    print(f"  - Only TCP flows (PROTOCOL=6)")
    print(f"  - Only flows with SYN flag present")

    cesnet_path = Path(cesnet_dir)

    if not cesnet_path.exists():
        print(f"\nERROR: Directory not found: {cesnet_dir}")
        return None

    # Find all flows.csv files in linux* directories
    print(f"\n[1/3] Searching for flows.csv in Linux directories...")
    flows_files = []

    for root, dirs, files in os.walk(cesnet_path):
        # Get the parent directory name (the OS name like linux__ubuntu__18.04.6)
        root_path = Path(root)
        parent_dir = root_path.parent.name

        # Only process if parent directory starts with "linux" (case-insensitive)
        if parent_dir.lower().startswith('linux') and 'flows.csv' in files:
            flows_csv = root_path / 'flows.csv'
            flows_files.append((flows_csv, parent_dir))
            if verbose:
                print(f"  Found: {parent_dir}/{root_path.name}/flows.csv")

    if not flows_files:
        print(f"\nERROR: No flows.csv files found in linux* directories")
        print(f"Expected structure: {cesnet_dir}/linux__ubuntu__18.04.6/2025-XX-XX__*/flows.csv")
        return None

    print(f"\n  Total Linux directories found: {len(flows_files)}")

    # Process each flows.csv
    print(f"\n[2/3] Processing flows...")

    all_records = []
    total_flows = 0
    total_tcp_syn = 0
    errors = 0

    for flows_file, os_name in tqdm(flows_files, desc="Processing directories"):
        try:
            # Read flows.csv
            df = pd.read_csv(flows_file)

            # Strip type prefixes from column names (e.g., "uint8 PROTOCOL" -> "PROTOCOL")
            df.columns = [col.split()[-1] if ' ' in col else col for col in df.columns]

            total_flows += len(df)

            # Filter for TCP flows with SYN flag
            # PROTOCOL = 6 (TCP)
            # TCP_FLAGS should contain SYN flag (bit 1, value 0x02)
            tcp_flows = df[df['PROTOCOL'] == 6].copy()

            # Check SYN flag in TCP_FLAGS
            tcp_flows['has_syn'] = tcp_flows['TCP_FLAGS'].apply(
                lambda x: bool(int(x) & 0x02) if pd.notna(x) else False
            )
            syn_flows = tcp_flows[tcp_flows['has_syn']].copy()

            total_tcp_syn += len(syn_flows)

            if len(syn_flows) == 0:
                continue

            # Map CESNET features to Masaryk format
            for idx, row in syn_flows.iterrows():
                try:
                    # Decode TCP options
                    tcp_opt_fwd = decode_tcp_options(row.get('TCP_OPT'))
                    tcp_opt_rev = decode_tcp_options(row.get('TCP_OPT_REV'))

                    # Get L3 protocol from IP address
                    l3_proto = get_l3_proto(row.get('SRC_IP'))

                    # Get TTL and calculate initial TTL
                    ip_ttl = row.get('IP_TTL')
                    initial_ttl = calculate_initial_ttl(ip_ttl)

                    # Convert TCP flags to Masaryk string format (e.g., "---AP-S-")
                    tcp_flags_a = convert_tcp_flags_to_string(row.get('TCP_FLAGS'))

                    # Decode TLS features (set to None - not available in CESNET)
                    tls_ja3 = decode_ja3_bytes(row.get('TLS_JA3'))
                    tls_ext_types = parse_tls_ext_type(row.get('TLS_EXT_TYPE'))

                    # Check for SYN-ACK in reverse direction
                    tcp_flags_rev = row.get('TCP_FLAGS_REV')
                    syn_ack_flag = None
                    if pd.notna(tcp_flags_rev):
                        # SYN-ACK = SYN (0x02) + ACK (0x10) = 0x12
                        tcp_flags_rev_int = int(tcp_flags_rev)
                        syn_ack_flag = 1 if (tcp_flags_rev_int & 0x12) == 0x12 else 0

                    # Build record in Masaryk format (convert to float64 to match Masaryk datatypes)
                    record = {
                        # ============================================================
                        # METADATA
                        # ============================================================
                        'dataset_source': 'cesnet',
                        'record_id': f"cesnet_{os_name}_{idx}",

                        # ============================================================
                        # BASIC FLOW PROPERTIES (6 features)
                        # ============================================================
                        'bytes_a': float(row.get('BYTES')) if pd.notna(row.get('BYTES')) else None,
                        'packets_a': float(row.get('PACKETS')) if pd.notna(row.get('PACKETS')) else None,
                        'src_port': row.get('SRC_PORT'),
                        'dst_port': row.get('DST_PORT'),
                        'packet_total_count_forward': row.get('PACKETS'),
                        'packet_total_count_backward': row.get('PACKETS_REV'),

                        # ============================================================
                        # IP PARAMETERS (7 features)
                        # ============================================================
                        'l3_proto': l3_proto,
                        'l4_proto': 6,  # TCP
                        'ip_tos': None,  # NOT AVAILABLE in CESNET
                        'maximum_ttl_forward': row.get('IP_TTL'),
                        'maximum_ttl_backward': row.get('IP_TTL_REV'),
                        'ipv4_dont_fragment_forward': 1 if (pd.notna(row.get('IP_FLG')) and int(row.get('IP_FLG')) & 0x02) else 0,
                        'ipv4_dont_fragment_backward': 1 if (pd.notna(row.get('IP_FLG_REV')) and int(row.get('IP_FLG_REV')) & 0x02) else 0,

                        # ============================================================
                        # TCP PARAMETERS (15 features)
                        # ============================================================
                        'tcp_syn_size': float(row.get('TCP_SYN_SIZE')) if pd.notna(row.get('TCP_SYN_SIZE')) else None,
                        'tcp_win_size': float(row.get('TCP_WIN')) if pd.notna(row.get('TCP_WIN')) else None,
                        'tcp_syn_ttl': ip_ttl,
                        'tcp_flags_a': tcp_flags_a,  # Converted from bitmask to string (e.g., "---AP-S-")
                        'syn_ack_flag': syn_ack_flag,
                        'tcp_option_window_scale_forward': float(tcp_opt_fwd['window_scale']) if tcp_opt_fwd['window_scale'] is not None else None,
                        'tcp_option_window_scale_backward': float(tcp_opt_rev['window_scale']) if tcp_opt_rev['window_scale'] is not None else None,
                        'tcp_option_selective_ack_permitted_forward': tcp_opt_fwd['sack_permitted'],
                        'tcp_option_selective_ack_permitted_backward': tcp_opt_rev['sack_permitted'],
                        'tcp_option_maximum_segment_size_forward': float(row.get('TCP_MSS')) if pd.notna(row.get('TCP_MSS')) else None,
                        'tcp_option_maximum_segment_size_backward': row.get('TCP_MSS_REV'),
                        'tcp_option_no_operation_forward': tcp_opt_fwd['nop'],
                        'tcp_option_no_operation_backward': tcp_opt_rev['nop'],

                        # ============================================================
                        # PACKET TIMINGS / NPM FEATURES (5 features) - NOT AVAILABLE
                        # ============================================================
                        'npm_round_trip_time': None,
                        'npm_tcp_retransmission_a': None,
                        'npm_tcp_retransmission_b': None,
                        'npm_tcp_out_of_order_a': None,
                        'npm_tcp_out_of_order_b': None,

                        # ============================================================
                        # TLS FINGERPRINTING FEATURES (7 features) - PARTIAL
                        # ============================================================
                        'tls_handshake_type': None,  # NOT AVAILABLE
                        'tls_client_version': float(row.get('TLS_VERSION')) if pd.notna(row.get('TLS_VERSION')) else None,
                        'tls_cipher_suites': None,  # NOT AVAILABLE
                        'tls_extension_types': tls_ext_types,
                        'tls_elliptic_curves': None,  # NOT AVAILABLE
                        'tls_client_key_length': None,  # NOT AVAILABLE
                        'tls_ja3_fingerprint': tls_ja3,

                        # ============================================================
                        # DERIVED FEATURES
                        # ============================================================
                        'initial_ttl': float(initial_ttl) if initial_ttl is not None else None,
                        'total_bytes': (row.get('BYTES') or 0) + (row.get('BYTES_REV') or 0),

                        # ============================================================
                        # LABEL (Target)
                        # ============================================================
                        'os_family': 'Linux',  # All CESNET flows are Linux
                    }

                    all_records.append(record)

                except Exception as row_error:
                    errors += 1
                    if errors < 10:
                        if verbose:
                            print(f"\n  Warning: Error processing row in {os_name}: {row_error}")
                    continue

        except Exception as file_error:
            errors += 1
            if verbose:
                print(f"\n  Warning: Error processing {flows_file}: {file_error}")
            continue

    print(f"\n  Total flows across all files: {total_flows:,}")
    print(f"  TCP SYN flows extracted: {total_tcp_syn:,}")
    print(f"  Successfully mapped to Masaryk format: {len(all_records):,}")
    if errors > 0:
        print(f"  Errors encountered: {errors}")

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    if len(df) == 0:
        print(f"\n  WARNING: No records were successfully processed!")
        return df

    # Save
    print(f"\n[3/3] Saving processed dataset...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cesnet.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nDataset shape: {df.shape}")
    print(f"  Records: {len(df):,}")
    print(f"  Features: {len(df.columns)}")

    print(f"\nOS Family distribution:")
    if 'os_family' in df.columns:
        print(df['os_family'].value_counts())

    print(f"\nFeature completeness check:")

    # Check availability of key features
    key_features = [
        'tcp_syn_size', 'tcp_win_size', 'tcp_syn_ttl', 'initial_ttl',
        'src_port', 'dst_port', 'bytes_a', 'packets_a',
        'tcp_option_maximum_segment_size_forward', 'tcp_option_window_scale_forward',
        'maximum_ttl_forward', 'ipv4_dont_fragment_forward',
        'tls_ja3_fingerprint'
    ]

    for feat in key_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"  {status} {feat:<45}: {pct:>5.1f}%")

    return df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract Linux TCP SYN flows from CESNET dataset in Masaryk format'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/cesnet',
        help='Input directory with CESNET data (contains linux* subdirectories)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed CSV'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Run extraction
    df = process_cesnet_flows(
        cesnet_dir=args.input,
        output_dir=args.output,
        verbose=not args.quiet
    )

    if df is None or len(df) == 0:
        sys.exit(1)

    print("\n✓ Success! CESNET Linux flows ready for merging with Masaryk dataset.")
    print(f"\nTo merge with Masaryk:")
    print(f"  import pandas as pd")
    print(f"  masaryk = pd.read_csv('data/processed/masaryk.csv')")
    print(f"  cesnet = pd.read_csv('data/processed/cesnet.csv')")
    print(f"  combined = pd.concat([masaryk, cesnet], ignore_index=True)")
    print(f"  combined.to_csv('data/processed/combined.csv', index=False)")


if __name__ == '__main__':
    main()
