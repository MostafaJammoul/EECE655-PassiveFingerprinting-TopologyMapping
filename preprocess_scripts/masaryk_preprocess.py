#!/usr/bin/env python3
"""
Preprocess Masaryk Dataset - Extract TCP SYN Flow Features for OS Family Classification

Input:  data/raw/masaryk/flows_ground_truth_merged_anonymized.csv
Output: data/processed/masaryk_processed.csv

Extracts 25 TCP SYN flow-level features for OS family classification.
FILTERING: Only TCP flows (protocol=6) with SYN flag present are processed.
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


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def preprocess_masaryk(raw_dir='data/raw/masaryk',
                       output_dir='data/processed',
                       sample_size=None,
                       verbose=True):
    """
    Main preprocessing pipeline for Masaryk dataset

    Extracts 25 features for OS family classification:
    - TCP fingerprinting: syn_size, win_size, syn_ttl, flags, options
    - IP features: l3_proto, ip_tos, ttl, DF bit
    - TLS fingerprinting: ja3, ciphers, extensions, curves, version
    - Flow metadata: src_port, packet counts, total_bytes

    Args:
        raw_dir: Directory containing Masaryk CSV files
        output_dir: Where to save processed CSV
        sample_size: If set, randomly sample this many rows (for testing)
        verbose: Print progress

    Returns:
        DataFrame with TCP SYN flow-level features
    """

    print("="*70)
    print("MASARYK DATASET PREPROCESSING - TCP SYN FLOWS ONLY")
    print("="*70)
    print(f"\nInput:  {raw_dir}/flows_ground_truth_merged_anonymized.csv")
    print(f"Output: {output_dir}/masaryk_processed.csv")
    print(f"\nFILTERING: Extracting only TCP flows with SYN flag present")
    print(f"FEATURES: 25 core OS fingerprinting features")

    raw_path = Path(raw_dir)

    if not raw_path.exists():
        print(f"\nERROR: Directory not found: {raw_dir}")
        return None

    # Find CSV file
    csv_file = raw_path / 'flows_ground_truth_merged_anonymized.csv'
    if not csv_file.exists():
        print(f"\nERROR: File not found: {csv_file}")
        return None

    print(f"\n[1/3] Found: {csv_file.name}")

    # Analyze file structure
    print(f"\n[2/3] Analyzing CSV structure...")
    print(f"  NOTE: Masaryk dataset uses semicolon-separated format")

    # Read first few lines to verify structure
    with open(csv_file, 'r') as f:
        header_line = f.readline().strip()
        sample_line = f.readline().strip()
        fields = sample_line.split(';')
        print(f"  Data row has {len(fields)} semicolon-separated fields")

    # Field positions in the semicolon-separated format:
    # Position 1: UA OS family (OS label)
    # Position 8: L3 PROTO
    # Position 9: L4 PROTO (6=TCP, 17=UDP)
    # Position 10: BYTES A
    # Position 14: TCP flags A
    # Position 15: SRC port
    # Position 18: TCP SYN Size
    # Position 19: TCP Win Size
    # Position 20: TCP SYN TTL
    # Position 65: TLS_HANDSHAKE_TYPE
    # Position 74: TLS_CLIENT_VERSION
    # Position 75: TLS_CIPHER_SUITES
    # Position 78: TLS_EXTENSION_TYPES
    # Position 80: TLS_ELLIPTIC_CURVES
    # Position 82: TLS_CLIENT_KEY_LENGTH
    # Position 91: TLS_JA3_FINGERPRINT
    # Position 92: maximumTTLforward
    # Position 94: IPv4DontFragmentforward
    # Position 98: tcpOptionWindowScaleforward
    # Position 100: tcpOptionSelectiveAckPermittedforward
    # Position 102: tcpOptionMaximumSegmentSizeforward
    # Position 104: tcpOptionNoOperationforward
    # Position 106: packetTotalCountforward
    # Position 107: packetTotalCountbackward
    # Position 110: synAckFlag

    print(f"\n  Extracting 25 features:")
    print(f"    ✅ TCP: syn_size, win_size, syn_ttl, flags_a, syn_ack_flag, 4 TCP options")
    print(f"    ✅ IP: l3_proto, ip_tos, maximum_ttl_forward, ipv4_dont_fragment_forward")
    print(f"    ✅ TLS: ja3, ciphers, extensions, curves, version, handshake_type, key_length")
    print(f"    ✅ Flow: src_port, packet_count_forward, packet_count_backward, total_bytes")
    print(f"    ✅ Derived: initial_ttl")

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

        # Handle gzipped or regular files
        if csv_file.name.endswith('.gz'):
            file_obj = gzip.open(csv_file, 'rt', encoding='utf-8')
        else:
            file_obj = open(csv_file, 'r', encoding='utf-8')

        try:
            # Skip header row
            header_line = file_obj.readline()
            if verbose:
                print(f"  Skipped header row")

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
                    # Extract OS label
                    os_label = fields[1].strip() if len(fields) > 1 else 'Unknown'
                    os_family = extract_os_family(os_label)

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

                    # Extract TCP flags (position 14) - string format like "---AP-SF"
                    tcp_flags_a = None
                    try:
                        if len(fields) > 14 and fields[14]:
                            tcp_flags_a = fields[14].strip()
                    except (ValueError, TypeError):
                        pass

                    # Extract SYN-ACK flag (position 110)
                    syn_ack_flag = None
                    has_syn = False
                    try:
                        if len(fields) > 110 and fields[110]:
                            syn_ack_flag = int(fields[110])
                            has_syn = (syn_ack_flag > 0)
                    except (ValueError, TypeError):
                        # Fallback: check TCP flags field for 'S'
                        if tcp_flags_a is not None:
                            has_syn = 'S' in tcp_flags_a

                    # FILTER: Only process flows with SYN flag
                    if not has_syn:
                        filtered_no_syn += 1
                        continue

                    # ======================================================================
                    # EXTRACT 24 FEATURES
                    # ======================================================================

                    # TCP fingerprinting features
                    tcp_syn_size = None
                    tcp_win_size = None
                    tcp_syn_ttl = None
                    try:
                        if len(fields) > 18 and fields[18]:
                            tcp_syn_size = int(fields[18])
                        if len(fields) > 19 and fields[19]:
                            tcp_win_size = int(fields[19])
                        if len(fields) > 20 and fields[20]:
                            tcp_syn_ttl = int(fields[20])
                    except (ValueError, TypeError):
                        pass

                    # TCP options (forward direction only)
                    tcp_win_scale_forward = None
                    tcp_sack_permitted_forward = None
                    tcp_mss_forward = None
                    tcp_nop_forward = None
                    try:
                        if len(fields) > 98 and fields[98]:
                            tcp_win_scale_forward = int(fields[98])
                        if len(fields) > 100 and fields[100]:
                            tcp_sack_permitted_forward = int(fields[100])
                        if len(fields) > 102 and fields[102]:
                            tcp_mss_forward = int(fields[102])
                        if len(fields) > 104 and fields[104]:
                            tcp_nop_forward = int(fields[104])
                    except (ValueError, TypeError):
                        pass

                    # IP features
                    l3_proto = None
                    max_ttl_forward = None
                    df_flag_forward = None
                    ip_tos = None
                    try:
                        if len(fields) > 8 and fields[8]:
                            l3_proto = int(fields[8])
                        if len(fields) > 21 and fields[21]:
                            ip_tos = int(fields[21])
                        if len(fields) > 92 and fields[92]:
                            max_ttl_forward = int(fields[92])
                        if len(fields) > 94 and fields[94]:
                            df_flag_forward = int(fields[94])
                    except (ValueError, TypeError):
                        pass

                    # Flow metadata
                    src_port = None
                    pkt_count_forward = None
                    pkt_count_backward = None
                    total_bytes = None
                    try:
                        if len(fields) > 15 and fields[15]:
                            src_port = int(fields[15])
                        if len(fields) > 106 and fields[106]:
                            pkt_count_forward = int(fields[106])
                        if len(fields) > 107 and fields[107]:
                            pkt_count_backward = int(fields[107])
                        if len(fields) > 10 and fields[10]:
                            total_bytes = float(fields[10])
                    except (ValueError, TypeError):
                        pass

                    # TLS fingerprinting features
                    tls_handshake_type = None
                    tls_client_version = None
                    tls_cipher_suites = None
                    tls_extension_types = None
                    tls_elliptic_curves = None
                    tls_client_key_length = None
                    tls_ja3_fingerprint = None
                    try:
                        if len(fields) > 65 and fields[65]:
                            tls_handshake_type = fields[65].strip()
                        if len(fields) > 74 and fields[74]:
                            tls_client_version = fields[74].strip()
                        if len(fields) > 75 and fields[75]:
                            tls_cipher_suites = fields[75].strip()
                        if len(fields) > 78 and fields[78]:
                            tls_extension_types = fields[78].strip()
                        if len(fields) > 80 and fields[80]:
                            tls_elliptic_curves = fields[80].strip()
                        if len(fields) > 82 and fields[82]:
                            tls_client_key_length = int(fields[82])
                        if len(fields) > 91 and fields[91]:
                            tls_ja3_fingerprint = fields[91].strip()
                    except (ValueError, TypeError):
                        pass

                    # Derived feature: initial_ttl
                    initial_ttl = calculate_initial_ttl(tcp_syn_ttl) if tcp_syn_ttl else None

                    # Create record with ONLY the 25 features + os_family
                    record = {
                        # TCP fingerprinting (9 features)
                        'tcp_syn_size': tcp_syn_size,
                        'tcp_win_size': tcp_win_size,
                        'tcp_syn_ttl': tcp_syn_ttl,
                        'tcp_flags_a': tcp_flags_a,
                        'syn_ack_flag': syn_ack_flag,
                        'tcp_option_window_scale_forward': tcp_win_scale_forward,
                        'tcp_option_selective_ack_permitted_forward': tcp_sack_permitted_forward,
                        'tcp_option_maximum_segment_size_forward': tcp_mss_forward,
                        'tcp_option_no_operation_forward': tcp_nop_forward,

                        # IP features (4 features)
                        'l3_proto': l3_proto,
                        'ip_tos': ip_tos,
                        'maximum_ttl_forward': max_ttl_forward,
                        'ipv4_dont_fragment_forward': df_flag_forward,

                        # Flow metadata (4 features)
                        'src_port': src_port,
                        'packet_total_count_forward': pkt_count_forward,
                        'packet_total_count_backward': pkt_count_backward,
                        'total_bytes': total_bytes,

                        # TLS fingerprinting (7 features)
                        'tls_ja3_fingerprint': tls_ja3_fingerprint,
                        'tls_cipher_suites': tls_cipher_suites,
                        'tls_extension_types': tls_extension_types,
                        'tls_elliptic_curves': tls_elliptic_curves,
                        'tls_client_version': tls_client_version,
                        'tls_handshake_type': tls_handshake_type,
                        'tls_client_key_length': tls_client_key_length,

                        # Derived features (1 feature)
                        'initial_ttl': initial_ttl,

                        # Target
                        'os_family': os_family,
                    }

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
    output_path = os.path.join(output_dir, 'try_masaryk.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nDataset shape: {df.shape}")
    print(f"  Records: {len(df):,}")
    print(f"  Features: {len(df.columns) - 1} (+ os_family target)")

    if len(df) == 0:
        print("\n  WARNING: No records were successfully processed!")
        print("  Check the input data format and error messages above.")
        return df

    print(f"\nOS Family distribution:")
    if 'os_family' in df.columns:
        print(df['os_family'].value_counts())

    print(f"\nFeature completeness check (25 features):")

    # TCP features
    print(f"\n  TCP Fingerprinting (9):")
    tcp_features = [
        'tcp_syn_size', 'tcp_win_size', 'tcp_syn_ttl', 'tcp_flags_a', 'syn_ack_flag',
        'tcp_option_window_scale_forward', 'tcp_option_selective_ack_permitted_forward',
        'tcp_option_maximum_segment_size_forward', 'tcp_option_no_operation_forward'
    ]
    for feat in tcp_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"    {status} {feat:<50}: {pct:>5.1f}%")

    # IP features
    print(f"\n  IP Features (4):")
    ip_features = ['l3_proto', 'ip_tos', 'maximum_ttl_forward', 'ipv4_dont_fragment_forward']
    for feat in ip_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"    {status} {feat:<50}: {pct:>5.1f}%")

    # Flow metadata
    print(f"\n  Flow Metadata (4):")
    flow_features = ['src_port', 'packet_total_count_forward', 'packet_total_count_backward', 'total_bytes']
    for feat in flow_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"    {status} {feat:<50}: {pct:>5.1f}%")

    # TLS features
    print(f"\n  TLS Fingerprinting (7):")
    tls_features = [
        'tls_ja3_fingerprint', 'tls_cipher_suites', 'tls_extension_types',
        'tls_elliptic_curves', 'tls_client_version', 'tls_handshake_type',
        'tls_client_key_length'
    ]
    for feat in tls_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"    {status} {feat:<50}: {pct:>5.1f}%")

    # Derived features
    print(f"\n  Derived Features (1):")
    print(f"    ✓ {'initial_ttl':<50}: {(df['initial_ttl'].notna().sum() / len(df)) * 100:>5.1f}%")

    return df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Masaryk dataset - extract 25 TCP SYN flow features for OS family classification'
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
        verbose=not args.quiet
    )

    if df is None:
        sys.exit(1)

    print("\n✓ Success! Dataset ready for Model 1 (OS family classification) training.")
    print(f"\n" + "="*70)
    print("EXTRACTED FEATURES SUMMARY (25 features)")
    print("="*70)
    print(f"\n  ✅ TCP Fingerprinting (9): syn_size, win_size, syn_ttl, flags_a, syn_ack_flag, 4 TCP options")
    print(f"  ✅ IP Features (4): l3_proto, ip_tos, maximum_ttl_forward, ipv4_dont_fragment_forward")
    print(f"  ✅ Flow Metadata (4): src_port, packet counts (fwd/back), total_bytes")
    print(f"  ✅ TLS Fingerprinting (7): ja3, ciphers, extensions, curves, version, handshake, key_length")
    print(f"  ✅ Derived (1): initial_ttl")
    print(f"\n  📊 Use this for: OS Family prediction (Windows/Linux/macOS/Android/iOS/BSD)")
    print(f"  🎯 Deployment: Flow-based passive OS fingerprinting")


if __name__ == '__main__':
    main()
