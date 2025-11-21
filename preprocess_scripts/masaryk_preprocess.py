#!/usr/bin/env python3
"""
Preprocess Masaryk Dataset - Extract TCP SYN Flow Features for OS Family Classification

Input:  data/raw/masaryk/*.csv
Output: data/processed/masaryk.csv

Extracts TCP SYN flow-level statistics for OS FAMILY classification (Windows vs Linux vs macOS).
The Masaryk dataset provides coarse OS labels (OS families) with rich TCP fingerprinting features.
FILTERING: Only TCP flows (protocol=6) with SYN flag present are processed.
Flow features like TCP window size, SYN size, TTL, and timing patterns distinguish OS families.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# OS FAMILY EXTRACTION
# ============================================================================

def extract_os_family(os_label):
    """Extract OS family from detailed OS label"""
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
        return 'Other'


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_initial_ttl(ttl):
    """
    Estimate original TTL value based on observed TTL

    Common initial TTLs:
    - 64:  Linux, macOS, Unix
    - 128: Windows
    - 255: Cisco, Solaris
    - 32:  Old systems
    """
    if ttl is None:
        return None
    common_ttls = [32, 64, 128, 255]
    for initial in common_ttls:
        if ttl <= initial:
            return initial
    return 255


def calculate_flow_features(row):
    """
    Calculate additional flow-level features from raw flow data

    These features are specific to flow-level analysis and help distinguish
    OS families based on their network behavior patterns.
    """
    features = {}

    # Helper to get numeric value or 0
    def get_num(key):
        val = row.get(key)
        return val if val is not None else 0

    # Initial TTL estimation (from tcp_syn_ttl only - position 20, 100% availability)
    ttl = row.get('tcp_syn_ttl')
    features['initial_ttl'] = calculate_initial_ttl(ttl) if ttl else None

    # Total bytes
    features['total_bytes'] = get_num('bytes_sent') + get_num('bytes_received')

    return features


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def preprocess_masaryk(raw_dir='data/raw/masaryk',
                       output_dir='data/processed',
                       sample_size=None,
                       chunk_size=100000,
                       verbose=True):
    """
    Main preprocessing pipeline for Masaryk dataset

    The Masaryk dataset contains flow-level features with coarse OS labels.
    NOTE: The dataset format is a single column with semicolon-separated values.
    We parse each row by splitting on semicolons and extracting features by position.

    FILTERING: Only TCP flows (protocol=6) with SYN flag present are extracted.
    This focuses on connection establishment packets which contain the most
    discriminative TCP/IP fingerprinting features.

    Args:
        raw_dir: Directory containing Masaryk CSV files
        output_dir: Where to save processed CSV
        sample_size: If set, randomly sample this many rows (for testing)
        chunk_size: Process CSV in chunks of this size
        verbose: Print progress

    Returns:
        DataFrame with TCP SYN flow-level features
    """

    print("="*70)
    print("MASARYK DATASET PREPROCESSING - TCP SYN FLOWS ONLY")
    print("="*70)
    print(f"\nInput:  {raw_dir}")
    print(f"Output: {output_dir}/masaryk.csv")
    print(f"\nFILTERING: Extracting only TCP flows with SYN flag present")
    print(f"This focuses on connection establishment for better OS fingerprinting")

    raw_path = Path(raw_dir)

    if not raw_path.exists():
        print(f"\nERROR: Directory not found: {raw_dir}")
        return None

    # Find CSV file (could be .csv or .csv.gz)
    csv_files = list(raw_path.glob('*.csv')) + list(raw_path.glob('*.csv.gz'))

    if not csv_files:
        print(f"\nERROR: No CSV files found in {raw_dir}")
        print(f"\nExpected: CSV file(s) with flow data and OS family labels")
        print(f"See DATASET_SETUP_GUIDE.md for download instructions.")
        return None

    csv_file = csv_files[0]
    print(f"\n[1/3] Found: {csv_file.name}")

    # Analyze file structure
    print(f"\n[2/3] Analyzing CSV structure...")
    print(f"  NOTE: Masaryk dataset uses single-column semicolon-separated format")
    print(f"  We parse by splitting each row on semicolons")

    # Read first few lines to verify structure
    with open(csv_file, 'r') as f:
        sample_line = f.readline().strip()
        fields = sample_line.split(';')
        print(f"  First row has {len(fields)} semicolon-separated fields")
        print(f"  Sample: Field[1]={fields[1]}, Field[2]={fields[2]}")

    # Field positions in the semicolon-separated format (CORRECTED positions from analyze_masaryk_columns.py):
    # Position 0: flow_ID
    # Position 1: UA OS family (OS label)
    # Position 2: UA OS major (OS version)
    # Position 6: start (timestamp)
    # Position 7: end (timestamp)
    # Position 8: L3 PROTO ✨ NEW
    # Position 9: L4 PROTO (6=TCP, 17=UDP)
    # Position 10: BYTES A
    # Position 11: PACKETS A
    # Position 12: SRC IP
    # Position 13: DST IP
    # Position 14: TCP flags A ✨ NEW (extract as feature)
    # Position 15: SRC port
    # Position 16: DST port
    # Position 17: ICMP TYPE (unused for TCP)
    # Position 18: TCP SYN Size
    # Position 19: TCP Win Size
    # Position 20: TCP SYN TTL
    # Position 21: IP ToS
    # Position 27: NPM_ROUND_TRIP_TIME ✨ NEW
    # Position 30: NPM_TCP_RETRANSMISSION_A ✨ NEW
    # Position 31: NPM_TCP_RETRANSMISSION_B ✨ NEW
    # Position 32: NPM_TCP_OUT_OF_ORDER_A ✨ NEW
    # Position 33: NPM_TCP_OUT_OF_ORDER_B ✨ NEW
    # Position 65: TLS_HANDSHAKE_TYPE ✨ NEW
    # Position 74: TLS_CLIENT_VERSION ✨ NEW
    # Position 75: TLS_CIPHER_SUITES ✨ NEW
    # Position 78: TLS_EXTENSION_TYPES ✨ NEW
    # Position 80: TLS_ELLIPTIC_CURVES ✨ NEW
    # Position 82: TLS_CLIENT_KEY_LENGTH ✨ NEW
    # Position 91: TLS_JA3_FINGERPRINT ✨ NEW
    # Position 92: maximumTTLforward (CORRECTED from 87)
    # Position 93: maximumTTLbackward (CORRECTED from 88)
    # Position 94: IPv4DontFragmentforward (CORRECTED from 89)
    # Position 95: IPv4DontFragmentbackward (CORRECTED from 90)
    # Position 98: tcpOptionWindowScaleforward (CORRECTED from 93)
    # Position 99: tcpOptionWindowScalebackward (CORRECTED from 94)
    # Position 100: tcpOptionSelectiveAckPermittedforward (CORRECTED from 95)
    # Position 101: tcpOptionSelectiveAckPermittedbackward (CORRECTED from 96)
    # Position 102: tcpOptionMaximumSegmentSizeforward (CORRECTED from 97)
    # Position 103: tcpOptionMaximumSegmentSizebackward (CORRECTED from 98)
    # Position 104: tcpOptionNoOperationforward (CORRECTED from 99)
    # Position 105: tcpOptionNoOperationbackward (CORRECTED from 100)
    # Position 106: packetTotalCountforward (CORRECTED from 101)
    # Position 107: packetTotalCountbackward (CORRECTED from 102)
    # Position 110: synAckFlag ✨ NEW (also extract as feature)

    print(f"\n  Field mapping (38 OS fingerprinting features):")
    print(f"    Position 1-2: OS label + version (target)")
    print(f"    Position 8-21: Basic TCP/IP (L3/L4 proto, bytes, ports, flags, TTL, ToS)")
    print(f"    Position 27, 30-33: NPM timing metrics (RTT, retransmissions, out-of-order)")
    print(f"    Position 65-91: TLS fingerprinting (JA3, ciphers, extensions)")
    print(f"    Position 92-95: IP TTL/DF flags (forward/backward)")
    print(f"    Position 98-107: TCP options bidirectional (WScale, SACK, MSS, NOP)")
    print(f"    Position 110: SYN-ACK flag")

    # Process the file line by line
    print(f"\n[3/3] Processing flow data...")
    if sample_size:
        print(f"  Using sample size: {sample_size:,} rows")

    all_records = []
    total_rows = 0
    errors = 0
    filtered_non_tcp = 0
    filtered_no_syn = 0

    try:
        import gzip
        from datetime import datetime

        # Handle gzipped or regular files
        if csv_file.name.endswith('.gz'):
            file_obj = gzip.open(csv_file, 'rt', encoding='utf-8')
        else:
            file_obj = open(csv_file, 'r', encoding='utf-8')

        try:
            # Skip header row
            header_line = file_obj.readline()
            if verbose:
                print(f"  Skipped header row: {header_line[:100]}...")

            for line_num, line in enumerate(tqdm(file_obj, desc="Processing rows"), 2):  # Start at line 2
                line = line.strip()
                if not line:
                    continue

                # Split by semicolon
                fields = line.split(';')

                if len(fields) < 21:  # Need at least positions 0-20
                    errors += 1
                    continue

                total_rows += 1

                try:
                    # Extract OS label and version
                    os_label = fields[1].strip() if len(fields) > 1 else 'Unknown'
                    os_version = fields[2].strip() if len(fields) > 2 and fields[2].strip() else ''

                    # Combine OS and version for full label
                    if os_version:
                        full_os_label = f"{os_label} {os_version}"
                    else:
                        full_os_label = os_label

                    os_family = extract_os_family(os_label)

                    # Calculate flow duration from timestamps
                    flow_duration = None
                    try:
                        if len(fields) > 7 and fields[6] and fields[7]:
                            start = datetime.fromisoformat(fields[6])
                            end = datetime.fromisoformat(fields[7])
                            flow_duration = (end - start).total_seconds()
                    except (ValueError, TypeError):
                        pass

                    # Extract packet and byte counts
                    # Position 11 appears to be packet count
                    # Position 10 appears to be bytes
                    pkt_count = None
                    total_bytes = None
                    try:
                        if len(fields) > 11 and fields[11]:
                            pkt_count = float(fields[11])
                    except (ValueError, TypeError):
                        pass

                    try:
                        if len(fields) > 10 and fields[10]:
                            total_bytes = float(fields[10])
                    except (ValueError, TypeError):
                        pass

                    # Extract ports
                    src_port = None
                    dst_port = None
                    try:
                        if len(fields) > 15 and fields[15]:
                            src_port = int(fields[15])
                    except (ValueError, TypeError):
                        pass

                    try:
                        if len(fields) > 16 and fields[16]:
                            dst_port = int(fields[16])
                    except (ValueError, TypeError):
                        pass

                    # Extract TTL
                    ttl = None
                    try:
                        if len(fields) > 20 and fields[20]:
                            ttl = int(fields[20])
                    except (ValueError, TypeError):
                        pass

                    # Extract L3 protocol (position 8) - NEW
                    l3_proto = None
                    try:
                        if len(fields) > 8 and fields[8]:
                            l3_proto = int(fields[8])
                    except (ValueError, TypeError):
                        pass

                    # Extract L4 protocol (position 9)
                    protocol = None
                    try:
                        if len(fields) > 9 and fields[9]:
                            protocol = int(fields[9])
                    except (ValueError, TypeError):
                        pass

                    # FILTER: Only process TCP flows (protocol = 6)
                    if protocol != 6:
                        filtered_non_tcp += 1
                        continue

                    # Extract TCP flags (position 14) - NEW: extract as feature, not just for filtering
                    tcp_flags_a = None
                    try:
                        if len(fields) > 14 and fields[14]:
                            tcp_flags_a = int(fields[14])
                    except (ValueError, TypeError):
                        pass

                    # FILTER: Only process flows with SYN flag
                    # Position 110: synAckFlag (indicates SYN or SYN-ACK was observed)
                    has_syn = False
                    syn_ack_flag = None
                    try:
                        if len(fields) > 110 and fields[110]:
                            syn_ack_flag = int(fields[110])
                            has_syn = (syn_ack_flag > 0)  # Non-zero means SYN/SYN-ACK present
                    except (ValueError, TypeError):
                        # If synAckFlag not available, check TCP flags field (position 14)
                        if tcp_flags_a is not None:
                            # SYN flag is bit 1 (0x02)
                            has_syn = (tcp_flags_a & 0x02) != 0

                    if not has_syn:
                        filtered_no_syn += 1
                        continue

                    # Extract additional TCP fingerprinting features for SYN flows
                    tcp_win_size = None
                    tcp_syn_size = None
                    try:
                        if len(fields) > 19 and fields[19]:
                            tcp_win_size = int(fields[19])
                    except (ValueError, TypeError):
                        pass

                    try:
                        if len(fields) > 18 and fields[18]:
                            tcp_syn_size = int(fields[18])
                    except (ValueError, TypeError):
                        pass

                    # Extract IP ToS (position 21)
                    ip_tos = None
                    try:
                        if len(fields) > 21 and fields[21]:
                            ip_tos = int(fields[21])
                    except (ValueError, TypeError):
                        pass

                    # ======================================================================
                    # NPM (Network Performance Monitoring) Features - NEW!
                    # ======================================================================

                    # RTT (position 27)
                    npm_rtt = None
                    try:
                        if len(fields) > 27 and fields[27]:
                            npm_rtt = float(fields[27])
                    except (ValueError, TypeError):
                        pass

                    # Retransmissions (positions 30-31)
                    npm_retrans_a = None
                    npm_retrans_b = None
                    try:
                        if len(fields) > 30 and fields[30]:
                            npm_retrans_a = int(fields[30])
                        if len(fields) > 31 and fields[31]:
                            npm_retrans_b = int(fields[31])
                    except (ValueError, TypeError):
                        pass

                    # Out-of-order packets (positions 32-33)
                    npm_out_of_order_a = None
                    npm_out_of_order_b = None
                    try:
                        if len(fields) > 32 and fields[32]:
                            npm_out_of_order_a = int(fields[32])
                        if len(fields) > 33 and fields[33]:
                            npm_out_of_order_b = int(fields[33])
                    except (ValueError, TypeError):
                        pass

                    # ======================================================================
                    # TLS Fingerprinting Features - NEW!
                    # ======================================================================

                    # TLS Handshake Type (position 65)
                    tls_handshake_type = None
                    try:
                        if len(fields) > 65 and fields[65]:
                            tls_handshake_type = fields[65].strip()
                    except (ValueError, TypeError):
                        pass

                    # TLS Client Version (position 74)
                    tls_client_version = None
                    try:
                        if len(fields) > 74 and fields[74]:
                            tls_client_version = fields[74].strip()
                    except (ValueError, TypeError):
                        pass

                    # TLS Cipher Suites (position 75)
                    tls_cipher_suites = None
                    try:
                        if len(fields) > 75 and fields[75]:
                            tls_cipher_suites = fields[75].strip()
                    except (ValueError, TypeError):
                        pass

                    # TLS Extension Types (position 78)
                    tls_extension_types = None
                    try:
                        if len(fields) > 78 and fields[78]:
                            tls_extension_types = fields[78].strip()
                    except (ValueError, TypeError):
                        pass

                    # TLS Elliptic Curves (position 80)
                    tls_elliptic_curves = None
                    try:
                        if len(fields) > 80 and fields[80]:
                            tls_elliptic_curves = fields[80].strip()
                    except (ValueError, TypeError):
                        pass

                    # TLS Client Key Length (position 82)
                    tls_client_key_length = None
                    try:
                        if len(fields) > 82 and fields[82]:
                            tls_client_key_length = int(fields[82])
                    except (ValueError, TypeError):
                        pass

                    # TLS JA3 Fingerprint (position 91) - CRITICAL!
                    tls_ja3_fingerprint = None
                    try:
                        if len(fields) > 91 and fields[91]:
                            tls_ja3_fingerprint = fields[91].strip()
                    except (ValueError, TypeError):
                        pass

                    # ======================================================================
                    # IP/TCP Bidirectional Features - CORRECTED POSITIONS!
                    # ======================================================================

                    # TTL features (positions 92-93, CORRECTED from 87-88)
                    max_ttl_forward = None
                    max_ttl_backward = None
                    try:
                        if len(fields) > 92 and fields[92]:
                            max_ttl_forward = int(fields[92])
                        if len(fields) > 93 and fields[93]:
                            max_ttl_backward = int(fields[93])
                    except (ValueError, TypeError):
                        pass

                    # Don't Fragment flags (positions 94-95, CORRECTED from 89-90)
                    df_flag_forward = None
                    df_flag_backward = None
                    try:
                        if len(fields) > 94 and fields[94]:
                            df_flag_forward = int(fields[94])
                        if len(fields) > 95 and fields[95]:
                            df_flag_backward = int(fields[95])
                    except (ValueError, TypeError):
                        pass

                    # TCP Window Scale (positions 98-99, CORRECTED from 93-94)
                    tcp_win_scale_forward = None
                    tcp_win_scale_backward = None
                    try:
                        if len(fields) > 98 and fields[98]:
                            tcp_win_scale_forward = int(fields[98])
                        if len(fields) > 99 and fields[99]:
                            tcp_win_scale_backward = int(fields[99])
                    except (ValueError, TypeError):
                        pass

                    # TCP SACK Permitted (positions 100-101, CORRECTED from 95-96)
                    tcp_sack_permitted_forward = None
                    tcp_sack_permitted_backward = None
                    try:
                        if len(fields) > 100 and fields[100]:
                            tcp_sack_permitted_forward = int(fields[100])
                        if len(fields) > 101 and fields[101]:
                            tcp_sack_permitted_backward = int(fields[101])
                    except (ValueError, TypeError):
                        pass

                    # TCP MSS (positions 102-103, CORRECTED from 97-98)
                    tcp_mss_forward = None
                    tcp_mss_backward = None
                    try:
                        if len(fields) > 102 and fields[102]:
                            tcp_mss_forward = int(fields[102])
                        if len(fields) > 103 and fields[103]:
                            tcp_mss_backward = int(fields[103])
                    except (ValueError, TypeError):
                        pass

                    # TCP NOP (positions 104-105, CORRECTED from 99-100)
                    tcp_nop_forward = None
                    tcp_nop_backward = None
                    try:
                        if len(fields) > 104 and fields[104]:
                            tcp_nop_forward = int(fields[104])
                        if len(fields) > 105 and fields[105]:
                            tcp_nop_backward = int(fields[105])
                    except (ValueError, TypeError):
                        pass

                    # Packet counts bidirectional (positions 106-107, CORRECTED from 101-102)
                    pkt_count_forward = None
                    pkt_count_backward = None
                    try:
                        if len(fields) > 106 and fields[106]:
                            pkt_count_forward = int(fields[106])
                        if len(fields) > 107 and fields[107]:
                            pkt_count_backward = int(fields[107])
                    except (ValueError, TypeError):
                        pass

                    record = {
                        # ============================================================
                        # METADATA
                        # ============================================================
                        'dataset_source': 'masaryk',
                        'record_id': f"masaryk_{total_rows}",

                        # ============================================================
                        # BASIC FLOW PROPERTIES (6 features)
                        # ============================================================
                        'bytes_a': total_bytes,  # Position 10
                        'packets_a': pkt_count,  # Position 11
                        'src_port': src_port,  # Position 15
                        'dst_port': dst_port,  # Position 16
                        'packet_total_count_forward': pkt_count_forward,  # Position 106
                        'packet_total_count_backward': pkt_count_backward,  # Position 107

                        # ============================================================
                        # IP PARAMETERS (7 features)
                        # ============================================================
                        'l3_proto': l3_proto,  # Position 8 ✨ NEW
                        'l4_proto': protocol,  # Position 9
                        'ip_tos': ip_tos,  # Position 21
                        'maximum_ttl_forward': max_ttl_forward,  # Position 92 (CORRECTED!)
                        'maximum_ttl_backward': max_ttl_backward,  # Position 93 (CORRECTED!)
                        'ipv4_dont_fragment_forward': df_flag_forward,  # Position 94 (CORRECTED!)
                        'ipv4_dont_fragment_backward': df_flag_backward,  # Position 95 (CORRECTED!)

                        # ============================================================
                        # TCP PARAMETERS (15 features)
                        # ============================================================
                        'tcp_syn_size': tcp_syn_size,  # Position 18
                        'tcp_win_size': tcp_win_size,  # Position 19
                        'tcp_syn_ttl': ttl,  # Position 20
                        'tcp_flags_a': tcp_flags_a,  # Position 14 ✨ NEW
                        'syn_ack_flag': syn_ack_flag,  # Position 110 ✨ NEW
                        'tcp_option_window_scale_forward': tcp_win_scale_forward,  # Position 98 (CORRECTED!)
                        'tcp_option_window_scale_backward': tcp_win_scale_backward,  # Position 99 (CORRECTED!)
                        'tcp_option_selective_ack_permitted_forward': tcp_sack_permitted_forward,  # Position 100 (CORRECTED!)
                        'tcp_option_selective_ack_permitted_backward': tcp_sack_permitted_backward,  # Position 101 (CORRECTED!)
                        'tcp_option_maximum_segment_size_forward': tcp_mss_forward,  # Position 102 (CORRECTED!)
                        'tcp_option_maximum_segment_size_backward': tcp_mss_backward,  # Position 103 (CORRECTED!)
                        'tcp_option_no_operation_forward': tcp_nop_forward,  # Position 104 (CORRECTED!)
                        'tcp_option_no_operation_backward': tcp_nop_backward,  # Position 105 (CORRECTED!)

                        # ============================================================
                        # PACKET TIMINGS / NPM FEATURES (5 features)
                        # ============================================================
                        'npm_round_trip_time': npm_rtt,  # Position 27 ✨ NEW
                        'npm_tcp_retransmission_a': npm_retrans_a,  # Position 30 ✨ NEW
                        'npm_tcp_retransmission_b': npm_retrans_b,  # Position 31 ✨ NEW
                        'npm_tcp_out_of_order_a': npm_out_of_order_a,  # Position 32 ✨ NEW
                        'npm_tcp_out_of_order_b': npm_out_of_order_b,  # Position 33 ✨ NEW

                        # ============================================================
                        # TLS FINGERPRINTING FEATURES (7 features)
                        # ============================================================
                        'tls_handshake_type': tls_handshake_type,  # Position 65 ✨ NEW
                        'tls_client_version': tls_client_version,  # Position 74 ✨ NEW
                        'tls_cipher_suites': tls_cipher_suites,  # Position 75 ✨ NEW
                        'tls_extension_types': tls_extension_types,  # Position 78 ✨ NEW
                        'tls_elliptic_curves': tls_elliptic_curves,  # Position 80 ✨ NEW
                        'tls_client_key_length': tls_client_key_length,  # Position 82 ✨ NEW
                        'tls_ja3_fingerprint': tls_ja3_fingerprint,  # Position 91 ✨ NEW

                        # ============================================================
                        # LABEL (Target)
                        # ============================================================
                        'os_family': os_family,  # Primary target for Model 1 family classification
                    }

                    # Calculate derived features
                    derived = calculate_flow_features(record)
                    record.update(derived)

                    all_records.append(record)

                except Exception as row_error:
                    errors += 1
                    if errors < 10:  # Only print first few errors
                        if verbose:
                            print(f"\n  Warning: Error processing row {line_num}: {row_error}")
                    continue

                # Stop if we've reached sample size
                if sample_size and len(all_records) >= sample_size:
                    break

        finally:
            file_obj.close()

    except Exception as e:
        print(f"\n  ERROR processing CSV: {e}")
        import traceback
        traceback.print_exc()
        if all_records:
            print(f"  Partial data extracted: {len(all_records)} records")
        else:
            return None

    print(f"\n  Total rows processed: {total_rows:,}")
    print(f"  Filtered (non-TCP): {filtered_non_tcp:,}")
    print(f"  Filtered (no SYN flag): {filtered_no_syn:,}")
    print(f"  TCP SYN flows extracted: {len(all_records):,}")
    if errors > 0:
        print(f"  Errors encountered: {errors}")

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    # Save
    print(f"\n  Saving processed dataset...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'masaryk.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nDataset shape: {df.shape}")
    print(f"  Records: {len(df):,}")
    print(f"  Features: {len(df.columns)}")

    if len(df) == 0:
        print("\n  WARNING: No records were successfully processed!")
        print("  Check the input data format and error messages above.")
        return df

    print(f"\nOS Family distribution:")
    if 'os_family' in df.columns:
        print(df['os_family'].value_counts())

    print(f"\nFeature completeness check (38 features extracted):")

    # TCP Parameters (15 features)
    print(f"\n  TCP Parameters (15):")
    tcp_features = [
        'tcp_syn_size', 'tcp_win_size', 'tcp_syn_ttl', 'tcp_flags_a', 'syn_ack_flag',
        'tcp_option_window_scale_forward', 'tcp_option_window_scale_backward',
        'tcp_option_selective_ack_permitted_forward', 'tcp_option_selective_ack_permitted_backward',
        'tcp_option_maximum_segment_size_forward', 'tcp_option_maximum_segment_size_backward',
        'tcp_option_no_operation_forward', 'tcp_option_no_operation_backward'
    ]
    for feat in tcp_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"    {status} {feat:<45}: {pct:>5.1f}%")

    # IP Parameters (7 features)
    print(f"\n  IP Parameters (7):")
    ip_features = [
        'l3_proto', 'l4_proto', 'ip_tos',
        'maximum_ttl_forward', 'maximum_ttl_backward',
        'ipv4_dont_fragment_forward', 'ipv4_dont_fragment_backward'
    ]
    for feat in ip_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"    {status} {feat:<45}: {pct:>5.1f}%")

    # Flow Properties (6 features)
    print(f"\n  Flow Properties (6):")
    flow_features = [
        'bytes_a', 'packets_a', 'src_port', 'dst_port',
        'packet_total_count_forward', 'packet_total_count_backward'
    ]
    for feat in flow_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"    {status} {feat:<45}: {pct:>5.1f}%")

    # NPM Timing Features (5 features)
    print(f"\n  NPM Timing Features (5):")
    npm_features = [
        'npm_round_trip_time', 'npm_tcp_retransmission_a', 'npm_tcp_retransmission_b',
        'npm_tcp_out_of_order_a', 'npm_tcp_out_of_order_b'
    ]
    for feat in npm_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"    {status} {feat:<45}: {pct:>5.1f}%")

    # TLS Features (7 features)
    print(f"\n  TLS Fingerprinting Features (7):")
    tls_features = [
        'tls_handshake_type', 'tls_client_version', 'tls_cipher_suites',
        'tls_extension_types', 'tls_elliptic_curves', 'tls_client_key_length',
        'tls_ja3_fingerprint'
    ]
    for feat in tls_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"    {status} {feat:<45}: {pct:>5.1f}%")

    # Derived Features
    print(f"\n  Derived Features (1):")
    derived_features = ['initial_ttl']
    for feat in derived_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"    {status} {feat:<45}: {pct:>5.1f}%")

    return df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Masaryk dataset - extract TCP SYN flow-level features for OS family classification'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/masaryk',
        help='Input directory with Masaryk CSV files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed CSV'
    )

    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Random sample size (for testing with subset of data)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Process CSV in chunks of this size (default: 100000)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Run preprocessing
    df = preprocess_masaryk(
        raw_dir=args.input,
        output_dir=args.output,
        sample_size=args.sample,
        chunk_size=args.chunk_size,
        verbose=not args.quiet
    )

    if df is None:
        sys.exit(1)

    print("\n✓ Success! Dataset ready for Model 1 (OS family classification) training.")
    print(f"\n" + "="*70)
    print("EXTRACTED FEATURES SUMMARY (38 features total)")
    print("="*70)
    print(f"\n  ✅ TCP Parameters (15):")
    print(f"     - SYN size, window size, TTL, flags")
    print(f"     - Bidirectional TCP options (WScale, SACK, MSS, NOP)")
    print(f"     - SYN-ACK flag")
    print(f"\n  ✅ IP Parameters (7):")
    print(f"     - L3/L4 protocols, ToS")
    print(f"     - Bidirectional TTL and DF flags")
    print(f"\n  ✅ Flow Properties (6):")
    print(f"     - Bytes, packets, ports")
    print(f"     - Bidirectional packet counts")
    print(f"\n  ✅ NPM Timing Features (5):")
    print(f"     - Round-trip time")
    print(f"     - Retransmissions (A/B)")
    print(f"     - Out-of-order packets (A/B)")
    print(f"\n  ✅ TLS Fingerprinting (7):")
    print(f"     - JA3 fingerprint (CRITICAL!)")
    print(f"     - Cipher suites, extensions, elliptic curves")
    print(f"     - Client version, handshake type, key length")
    print(f"\n  📊 Use this for: OS Family prediction (Windows/Linux/macOS/Android/iOS/BSD)")
    print(f"  🎯 Deployment: Flow-based passive OS fingerprinting")
    print(f"  ⚡ Strategy: Wait for SYN → Capture flow → Aggregate features → Classify")


if __name__ == '__main__':
    main()