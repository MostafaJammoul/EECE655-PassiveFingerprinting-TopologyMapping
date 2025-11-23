#!/usr/bin/env python3
"""
Extract Android Flows from Masaryk Dataset

Extracts Android traffic flows (versions 7, 8, 9, 10) from Masaryk dataset
and creates a CSV with the same 25 features used in Model 1 (Family Classifier).

Android version mapping:
  - Android.7.x.x → Android 7
  - Android.8.x.x → Android 8
  - Android.9.x.x → Android 9
  - Android.10.x.x → Android 10

Input:  data/raw/masaryk/flows_ground_truth_merged_anonymized.csv
Output: data/processed/masaryk_android.csv
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
# UTILITY FUNCTIONS
# ============================================================================

def calculate_initial_ttl(ttl):
    """Estimate original TTL value based on observed TTL"""
    if ttl is None:
        return None
    common_ttls = [32, 64, 128, 255]
    for initial in common_ttls:
        if ttl <= initial:
            return initial
    return 255


def parse_android_version(os_family, os_major, os_minor, os_patch, os_patch_minor):
    """
    Parse Android version and return major version label

    Examples:
        Android, 7, 0, None, None → Android 7
        Android, 8, 1, 0, None → Android 8
        Android, 9, None, None, None → Android 9
        Android, 10, 0, 0, None → Android 10

    Returns:
        "Android 7", "Android 8", "Android 9", "Android 10", or None if not valid
    """
    # Check if OS family is Android
    if not os_family or 'android' not in os_family.lower():
        return None

    # Extract major version
    if not os_major:
        return None

    try:
        major_version = int(os_major)
    except (ValueError, TypeError):
        return None

    # Only keep Android 7, 8, 9, 10
    if major_version in [7, 8, 9, 10]:
        return f"Android {major_version}"

    return None


# ============================================================================
# MAIN EXTRACTION
# ============================================================================

def extract_android_flows(raw_dir='data/raw/masaryk',
                          output_dir='data/processed',
                          sample_size=None,
                          verbose=True):
    """
    Extract Android flows from Masaryk dataset

    Extracts same 25 features as masaryk_preprocess.py but filtered to
    Android versions 7, 8, 9, 10 only.
    """

    print("="*80)
    print("MASARYK ANDROID EXTRACTION - ANDROID 7/8/9/10 ONLY")
    print("="*80)
    print(f"\nInput:  {raw_dir}/flows_ground_truth_merged_anonymized.csv")
    print(f"Output: {output_dir}/masaryk_android.csv")
    print(f"\nFiltering: Android 7, 8, 9, 10 flows only")
    print(f"Features: 25 core OS fingerprinting features")

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

    # Process the file line by line
    print(f"\n[2/3] Processing flow data...")
    if sample_size:
        print(f"  Using sample size: {sample_size:,} rows")

    all_records = []
    total_rows = 0
    errors = 0
    filtered_non_tcp = 0
    filtered_no_syn = 0
    filtered_non_android = 0
    filtered_wrong_android_version = 0

    # Field positions in semicolon-separated format (same as masaryk_preprocess.py)
    # Position 1: UA OS family
    # Position 2: UA OS major
    # Position 3: UA OS minor
    # Position 4: UA OS patch
    # Position 5: UA OS patch minor
    # Position 8: L3 PROTO
    # Position 9: L4 PROTO (6=TCP, 17=UDP)
    # Position 15: SRC port
    # Position 18: TCP SYN Size
    # Position 19: TCP Win Size
    # Position 20: TCP SYN TTL
    # ... (same positions as masaryk_preprocess.py)

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

            for line_num, line in enumerate(tqdm(file_obj, desc="Processing rows"), 2):
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
                    # Extract OS fields (positions 1-5)
                    os_family = fields[1].strip() if len(fields) > 1 else ''
                    os_major = fields[2].strip() if len(fields) > 2 else ''
                    os_minor = fields[3].strip() if len(fields) > 3 else ''
                    os_patch = fields[4].strip() if len(fields) > 4 else ''
                    os_patch_minor = fields[5].strip() if len(fields) > 5 else ''

                    # Parse Android version
                    android_label = parse_android_version(os_family, os_major, os_minor, os_patch, os_patch_minor)

                    if not android_label:
                        # Not Android or not valid Android version
                        if os_family and 'android' in os_family.lower():
                            filtered_wrong_android_version += 1
                        else:
                            filtered_non_android += 1
                        continue

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

                    # Extract TCP flags (position 14)
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
                    # EXTRACT 25 FEATURES (same as masaryk_preprocess.py)
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

                    # Create record with 25 features + os_label
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

                        # Target (Android version label)
                        'os_label': android_label,
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
    print(f"  Filtered (non-Android): {filtered_non_android:,}")
    print(f"  Filtered (wrong Android version): {filtered_wrong_android_version:,}")
    print(f"  Filtered (non-TCP): {filtered_non_tcp:,}")
    print(f"  Filtered (no SYN flag): {filtered_no_syn:,}")
    print(f"  Android flows extracted: {len(all_records):,}")
    if errors > 0:
        print(f"  Errors encountered: {errors}")

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    if len(df) == 0:
        print("\n  WARNING: No Android flows extracted!")
        return df

    # Save
    print(f"\n[3/3] Saving processed dataset...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'masaryk_android.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nDataset shape: {df.shape}")
    print(f"  Records: {len(df):,}")
    print(f"  Features: {len(df.columns) - 1} (+ os_label target)")

    print(f"\nAndroid version distribution:")
    if 'os_label' in df.columns:
        version_counts = df['os_label'].value_counts()
        for version, count in version_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {version}: {count:,} ({pct:.2f}%)")

    print(f"\nFeature completeness check:")

    # TCP features
    print(f"\n  TCP Fingerprinting:")
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
    print(f"\n  IP Features:")
    ip_features = ['l3_proto', 'ip_tos', 'maximum_ttl_forward', 'ipv4_dont_fragment_forward']
    for feat in ip_features:
        if feat in df.columns:
            pct = (df[feat].notna().sum() / len(df)) * 100
            status = "✓" if pct > 80 else ("⚠" if pct > 50 else "✗")
            print(f"    {status} {feat:<50}: {pct:>5.1f}%")

    # TLS features
    print(f"\n  TLS Fingerprinting:")
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

    print(f"\n✓ Success! Android dataset ready for training.")
    print(f"\n  Use this for: Android version prediction (7, 8, 9, 10)")
    print(f"  Training: python train_scripts/train_android_expert.py")

    return df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract Android flows from Masaryk dataset'
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

    # Run extraction
    df = extract_android_flows(
        raw_dir=args.input,
        output_dir=args.output,
        sample_size=args.sample,
        verbose=not args.quiet
    )

    if df is None or len(df) == 0:
        sys.exit(1)

    print(f"\n" + "="*80)


if __name__ == '__main__':
    main()
