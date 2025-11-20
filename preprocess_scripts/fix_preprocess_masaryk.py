#!/usr/bin/env python3
"""
Preprocess Masaryk Dataset - Extract TCP SYN Flow Features for OS Family Classification

Input:  data/raw/masaryk/*.csv
Output: data/processed/masaryk_processed.csv

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
    print(f"Output: {output_dir}/masaryk_processed.csv")
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

    # Field positions in the semicolon-separated format (IP/TCP features only):
    # Position 0: Flow ID
    # Position 1: OS (e.g., "Windows", "Linux", "Other")
    # Position 2: OS Version (e.g., "10", or empty)
    # Position 6: Start timestamp
    # Position 7: End timestamp
    # Position 8: L3 PROTO (IP version)
    # Position 9: L4 PROTO (6=TCP, 17=UDP)
    # Position 10: BYTES A
    # Position 11: PACKETS A
    # Position 12: SRC IP
    # Position 13: DST IP
    # Position 14: TCP flags A
    # Position 15: SRC port
    # Position 16: DST port
    # Position 17: ICMP TYPE
    # Position 18: TCP SYN Size
    # Position 19: TCP Win Size
    # Position 20: TCP SYN TTL
    # Position 21: IP ToS
    # Position 87: maximumTTLforward
    # Position 88: maximumTTLbackward
    # Position 89: IPv4DontFragmentforward
    # Position 90: IPv4DontFragmentbackward
    # Position 91: tcpTimestampFirstPacketforward
    # Position 92: tcpTimestampFirstPacketbackward
    # Position 93: tcpOptionWindowScaleforward
    # Position 94: tcpOptionWindowScalebackward
    # Position 95: tcpOptionSelectiveAckPermittedforward
    # Position 96: tcpOptionSelectiveAckPermittedbackward
    # Position 97: tcpOptionMaximumSegmentSizeforward
    # Position 98: tcpOptionMaximumSegmentSizebackward
    # Position 99: tcpOptionNoOperationforward
    # Position 100: tcpOptionNoOperationbackward
    # Position 101: packetTotalCountforward
    # Position 102: packetTotalCountbackward
    # Position 103: flowDirection
    # Position 104: flowEndReason
    # Position 105: synAckFlag

    print(f"\n  Field mapping (IP/TCP features only):")
    print(f"    Position 1-2: OS label + version (target)")
    print(f"    Position 8-9: L3/L4 protocols")
    print(f"    Position 10-11: Bytes/Packets")
    print(f"    Position 12-13: SRC/DST IP")
    print(f"    Position 14: TCP flags")
    print(f"    Position 15-16: Src/Dst ports")
    print(f"    Position 18-21: TCP SYN size, Win size, TTL, IP ToS")
    print(f"    Position 87-106: Advanced TCP/IP features (TTL, DF, TCP options)")

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
            for line_num, line in enumerate(tqdm(file_obj, desc="Processing rows"), 1):
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

                    # Extract protocol
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

                    # FILTER: Only process flows with SYN flag
                    # Position 105: synAckFlag (indicates SYN or SYN-ACK was observed)
                    has_syn = False
                    try:
                        if len(fields) > 105 and fields[105]:
                            syn_ack_flag = int(fields[105])
                            has_syn = (syn_ack_flag > 0)  # Non-zero means SYN/SYN-ACK present
                    except (ValueError, TypeError):
                        # If synAckFlag not available, check TCP flags field (position 14)
                        try:
                            if len(fields) > 14 and fields[14]:
                                tcp_flags = int(fields[14])
                                # SYN flag is bit 1 (0x02)
                                has_syn = (tcp_flags & 0x02) != 0
                        except (ValueError, TypeError):
                            pass

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

                    # Extract CRITICAL TCP options (positions 87-100) - essential for OS fingerprinting!
                    max_ttl_fwd = None
                    max_ttl_bwd = None
                    try:
                        if len(fields) > 87 and fields[87]:
                            max_ttl_fwd = int(fields[87])
                    except (ValueError, TypeError):
                        pass

                    try:
                        if len(fields) > 88 and fields[88]:
                            max_ttl_bwd = int(fields[88])
                    except (ValueError, TypeError):
                        pass

                    # DF flag (Don't Fragment) - important for OS identification
                    df_flag_fwd = None
                    df_flag_bwd = None
                    try:
                        if len(fields) > 89 and fields[89]:
                            df_flag_fwd = int(fields[89])
                    except (ValueError, TypeError):
                        pass

                    try:
                        if len(fields) > 90 and fields[90]:
                            df_flag_bwd = int(fields[90])
                    except (ValueError, TypeError):
                        pass

                    # TCP Window Scale option - CRITICAL for OS fingerprinting
                    tcp_win_scale_fwd = None
                    tcp_win_scale_bwd = None
                    try:
                        if len(fields) > 93 and fields[93]:
                            tcp_win_scale_fwd = int(fields[93])
                    except (ValueError, TypeError):
                        pass

                    try:
                        if len(fields) > 94 and fields[94]:
                            tcp_win_scale_bwd = int(fields[94])
                    except (ValueError, TypeError):
                        pass

                    # SACK permitted option
                    tcp_sack_permitted_fwd = None
                    tcp_sack_permitted_bwd = None
                    try:
                        if len(fields) > 95 and fields[95]:
                            tcp_sack_permitted_fwd = int(fields[95])
                    except (ValueError, TypeError):
                        pass

                    try:
                        if len(fields) > 96 and fields[96]:
                            tcp_sack_permitted_bwd = int(fields[96])
                    except (ValueError, TypeError):
                        pass

                    # TCP MSS (Maximum Segment Size) - EXTREMELY important for OS fingerprinting
                    tcp_mss_fwd = None
                    tcp_mss_bwd = None
                    try:
                        if len(fields) > 97 and fields[97]:
                            tcp_mss_fwd = int(fields[97])
                    except (ValueError, TypeError):
                        pass

                    try:
                        if len(fields) > 98 and fields[98]:
                            tcp_mss_bwd = int(fields[98])
                    except (ValueError, TypeError):
                        pass

                    # TCP NOP (No Operation) option count
                    tcp_nop_fwd = None
                    tcp_nop_bwd = None
                    try:
                        if len(fields) > 99 and fields[99]:
                            tcp_nop_fwd = int(fields[99])
                    except (ValueError, TypeError):
                        pass

                    try:
                        if len(fields) > 100 and fields[100]:
                            tcp_nop_bwd = int(fields[100])
                    except (ValueError, TypeError):
                        pass

                    record = {
                        # Metadata
                        'dataset_source': 'masaryk',
                        'record_id': f"masaryk_{total_rows}",

                        # Flow-level features
                        'pkt_count': pkt_count,
                        'flow_duration': flow_duration,
                        'bytes_sent': total_bytes if total_bytes else None,  # Using total as "sent" estimate
                        'bytes_received': None,  # Not available in this format

                        # Network info
                        'src_port': src_port,
                        'dst_port': dst_port,
                        'protocol': protocol,
                        'ttl': ttl,

                        # TCP fingerprinting features (from SYN flows)
                        'tcp_win_size': tcp_win_size,
                        'tcp_syn_size': tcp_syn_size,

                        # CRITICAL TCP OPTIONS - Essential for OS fingerprinting!
                        'max_ttl_forward': max_ttl_fwd,
                        'max_ttl_backward': max_ttl_bwd,
                        'df_flag_forward': df_flag_fwd,
                        'df_flag_backward': df_flag_bwd,
                        'tcp_win_scale_forward': tcp_win_scale_fwd,
                        'tcp_win_scale_backward': tcp_win_scale_bwd,
                        'tcp_sack_permitted_forward': tcp_sack_permitted_fwd,
                        'tcp_sack_permitted_backward': tcp_sack_permitted_bwd,
                        'tcp_mss_forward': tcp_mss_fwd,
                        'tcp_mss_backward': tcp_mss_bwd,
                        'tcp_nop_forward': tcp_nop_fwd,
                        'tcp_nop_backward': tcp_nop_bwd,

                        # Labels (os_label removed to prevent data leakage)
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
    output_path = os.path.join(output_dir, 'masaryk_processed.csv')
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

    print(f"\nTCP fingerprinting features (basic):")
    tcp_features = ['tcp_win_size', 'tcp_syn_size', 'ttl']
    available_tcp_features = [f for f in tcp_features if f in df.columns]
    if available_tcp_features:
        print(df[available_tcp_features].describe())

    print(f"\nCRITICAL TCP Options (essential for OS fingerprinting):")
    tcp_option_features = [
        'tcp_mss_forward', 'tcp_mss_backward',
        'tcp_win_scale_forward', 'tcp_win_scale_backward',
        'df_flag_forward', 'df_flag_backward',
        'tcp_sack_permitted_forward', 'tcp_sack_permitted_backward',
        'max_ttl_forward', 'max_ttl_backward'
    ]
    available_tcp_options = [f for f in tcp_option_features if f in df.columns]
    if available_tcp_options:
        for feat in available_tcp_options:
            pct_available = (df[feat].notna().sum() / len(df)) * 100
            print(f"  {feat}: {pct_available:.1f}% available")
        print("\nTCP Options statistics:")
        print(df[available_tcp_options].describe())

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

    print("\n✓ Success! Dataset ready for Model 1 (family classification) training.")
    print(f"\nThis dataset contains TCP SYN FLOW-level features with coarse OS labels:")
    print(f"  - FILTERED: Only TCP flows with SYN flag present")
    print(f"  - TCP fingerprinting features (window size, SYN size, TTL)")
    print(f"  - Flow statistics (packet counts, duration, byte rates)")
    print(f"  - Use this for predicting OS FAMILY (Windows/Linux/macOS)")
    print(f"  - Masaryk provides rich TCP/IP features from connection establishment!")


if __name__ == '__main__':
    main()