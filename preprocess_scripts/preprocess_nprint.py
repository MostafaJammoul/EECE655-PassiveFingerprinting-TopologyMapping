#!/usr/bin/env python3
"""
Preprocess nPrint Dataset - Extract Packet-Level Features for OS Fingerprinting

Input:  data/raw/nprint/*.csv (or *.npz or *.pcap)
Output: data/processed/nprint_packets.csv

The nPrint dataset can come in various formats. This script handles:
- Pre-extracted CSV files with features
- NPZ files (numpy arrays)
- PCAP files (if using raw nPrint captures)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Try importing optional dependencies
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
# CSV PARSING
# ============================================================================

def parse_nprint_csv(csv_path, verbose=True):
    """
    Parse nPrint CSV file and map to our unified schema

    nPrint CSV files may have different column names depending on version.
    We'll try to map common column names to our schema.
    """
    if verbose:
        print(f"  Parsing CSV: {csv_path.name}")

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"  ERROR reading {csv_path}: {e}")
        return None

    if verbose:
        print(f"    Loaded {len(df)} records, {len(df.columns)} columns")
        print(f"    Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")

    # Initialize output records
    records = []

    # Detect column mapping (nPrint datasets vary)
    # Common column name patterns:
    label_col = None
    ttl_col = None
    window_col = None
    mss_col = None

    # Try to find label column
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['label', 'os_label', 'os', 'operating_system', 'class']:
            label_col = col
            break

    # Try to find network feature columns
    for col in df.columns:
        col_lower = col.lower()
        if 'ttl' in col_lower:
            ttl_col = col
        elif any(w in col_lower for w in ['window', 'win_size', 'window_size']):
            window_col = col
        elif 'mss' in col_lower:
            mss_col = col

    if not label_col:
        print(f"  WARNING: No label column found in {csv_path.name}")
        print(f"  Available columns: {list(df.columns)}")
        return None

    # Process each row
    for idx, row in df.iterrows():
        os_label = row[label_col]

        record = {
            # Metadata
            'dataset_source': 'nprint',
            'record_id': f"nprint_{csv_path.stem}_{idx}",
            'timestamp': None,

            # Network info (may not be available)
            'src_ip': row.get('src_ip', None),
            'dst_ip': row.get('dst_ip', None),
            'protocol': row.get('protocol', None),
            'src_port': row.get('src_port', None),
            'dst_port': row.get('dst_port', None),

            # IP layer features
            'ttl': row[ttl_col] if ttl_col and ttl_col in row else None,
            'initial_ttl': None,  # Will calculate if TTL available
            'df_flag': row.get('df_flag', None),
            'ip_len': row.get('ip_len', None),

            # TCP layer features
            'tcp_window_size': row[window_col] if window_col and window_col in row else None,
            'tcp_window_scale': row.get('tcp_window_scale', None),
            'tcp_mss': row[mss_col] if mss_col and mss_col in row else None,
            'tcp_options_order': row.get('tcp_options_order', None),
            'tcp_flags': row.get('tcp_flags', None),

            # Labels
            'os_label': str(os_label),
            'os_family': extract_os_family(os_label),
        }

        # Calculate initial TTL if available
        if record['ttl'] is not None:
            ttl_val = record['ttl']
            for initial in [32, 64, 128, 255]:
                if ttl_val <= initial:
                    record['initial_ttl'] = initial
                    break
            if record['initial_ttl'] is None:
                record['initial_ttl'] = 255

        records.append(record)

    df_out = pd.DataFrame(records)

    if verbose:
        print(f"    Extracted {len(df_out)} records")
        print(f"    OS labels: {df_out['os_label'].nunique()} unique")

    return df_out


# ============================================================================
# NPZ PARSING (if nPrint provides numpy arrays)
# ============================================================================

def parse_nprint_npz(npz_path, verbose=True):
    """
    Parse nPrint NPZ file (numpy compressed arrays)

    NPZ files typically contain:
    - X: feature matrix (N x F)
    - y: labels (N,)
    - feature_names: list of feature names

    We'll try to map these to our schema.
    """
    if verbose:
        print(f"  Parsing NPZ: {npz_path.name}")

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  ERROR reading {npz_path}: {e}")
        return None

    if verbose:
        print(f"    Keys: {list(data.keys())}")

    # Try to find labels and features
    labels = None
    features = None
    feature_names = None

    # Common key patterns
    if 'y' in data:
        labels = data['y']
    elif 'labels' in data:
        labels = data['labels']

    if 'X' in data:
        features = data['X']
    elif 'features' in data:
        features = data['features']

    if 'feature_names' in data:
        feature_names = data['feature_names']

    if labels is None:
        print(f"  WARNING: No labels found in {npz_path.name}")
        return None

    if features is None:
        print(f"  WARNING: No features found in {npz_path.name}")
        # Can still process labels only
        records = []
        for idx, label in enumerate(labels):
            records.append({
                'dataset_source': 'nprint',
                'record_id': f"nprint_{npz_path.stem}_{idx}",
                'os_label': str(label),
                'os_family': extract_os_family(label),
                # All other fields None
                **{k: None for k in ['timestamp', 'src_ip', 'dst_ip', 'protocol',
                                     'src_port', 'dst_port', 'ttl', 'initial_ttl',
                                     'df_flag', 'ip_len', 'tcp_window_size',
                                     'tcp_window_scale', 'tcp_mss', 'tcp_options_order', 'tcp_flags']}
            })
        return pd.DataFrame(records)

    # If we have features, try to map them
    records = []
    for idx in range(len(labels)):
        feature_vec = features[idx]
        label = labels[idx]

        record = {
            'dataset_source': 'nprint',
            'record_id': f"nprint_{npz_path.stem}_{idx}",
            'os_label': str(label),
            'os_family': extract_os_family(label),
            # Try to extract known features if feature_names available
            # Otherwise all None
        }

        # If feature names available, try to map
        if feature_names is not None:
            for feat_idx, feat_name in enumerate(feature_names):
                feat_lower = str(feat_name).lower()
                if 'ttl' in feat_lower:
                    record['ttl'] = feature_vec[feat_idx]
                elif 'window' in feat_lower:
                    record['tcp_window_size'] = feature_vec[feat_idx]
                elif 'mss' in feat_lower:
                    record['tcp_mss'] = feature_vec[feat_idx]
                # Add more mappings as needed

        records.append(record)

    df_out = pd.DataFrame(records)

    if verbose:
        print(f"    Extracted {len(df_out)} records")

    return df_out


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def preprocess_nprint(raw_dir='data/raw/nprint',
                      output_dir='data/processed',
                      verbose=True):
    """
    Main preprocessing pipeline for nPrint dataset

    Handles CSV, NPZ, and potentially PCAP formats

    Args:
        raw_dir: Directory containing nPrint data files
        output_dir: Where to save processed CSV
        verbose: Print progress

    Returns:
        DataFrame with extracted features
    """

    print("="*70)
    print("NPRINT DATASET PREPROCESSING")
    print("="*70)
    print(f"\nInput:  {raw_dir}")
    print(f"Output: {output_dir}/nprint_packets.csv")

    raw_path = Path(raw_dir)

    if not raw_path.exists():
        print(f"\nERROR: Directory not found: {raw_dir}")
        print("\nPlease download nPrint dataset first.")
        print("See DATASET_SETUP_GUIDE.md for instructions.")
        return None

    # Discover data files
    print(f"\n[1/3] Discovering data files...")
    csv_files = list(raw_path.glob('**/*.csv'))
    npz_files = list(raw_path.glob('**/*.npz'))

    print(f"  Found {len(csv_files)} CSV files")
    print(f"  Found {len(npz_files)} NPZ files")

    if not csv_files and not npz_files:
        print(f"\nERROR: No supported files found in {raw_dir}")
        print(f"Expected: .csv or .npz files")
        print(f"\nMake sure you downloaded and extracted the nPrint dataset.")
        return None

    # Process files
    print(f"\n[2/3] Parsing data files...")
    all_dfs = []

    # Process CSV files
    if csv_files:
        for csv_file in tqdm(csv_files, desc="Processing CSVs"):
            df_chunk = parse_nprint_csv(csv_file, verbose=False)
            if df_chunk is not None:
                all_dfs.append(df_chunk)

    # Process NPZ files
    if npz_files:
        for npz_file in tqdm(npz_files, desc="Processing NPZs"):
            df_chunk = parse_nprint_npz(npz_file, verbose=False)
            if df_chunk is not None:
                all_dfs.append(df_chunk)

    if not all_dfs:
        print("\nERROR: No records extracted from any files!")
        return None

    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)

    print(f"  Total records extracted: {len(df):,}")

    # Save
    print(f"\n[3/3] Saving processed dataset...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'nprint_packets.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nDataset shape: {df.shape}")
    print(f"  Records: {len(df):,}")
    print(f"  Features: {len(df.columns)}")

    print(f"\nOS Family distribution:")
    print(df['os_family'].value_counts())

    print(f"\nOS Version distribution (top 10):")
    print(df['os_label'].value_counts().head(10))

    print(f"\nFeature availability:")
    critical_features = ['ttl', 'tcp_window_size', 'tcp_mss', 'tcp_options_order']
    for feat in critical_features:
        if feat in df.columns:
            pct_available = (df[feat].notna().sum() / len(df)) * 100
            print(f"  {feat}: {pct_available:.1f}%")
        else:
            print(f"  {feat}: not available")

    return df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess nPrint dataset - extract packet-level features'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/nprint',
        help='Input directory with nPrint data files'
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

    # Run preprocessing
    df = preprocess_nprint(
        raw_dir=args.input,
        output_dir=args.output,
        verbose=not args.quiet
    )

    if df is None:
        sys.exit(1)

    print("\n✓ Success! Dataset ready for merging.")
    print(f"\nNext steps:")
    print(f"  1. Merge with CESNET idle: python scripts/merge_packet_datasets.py")


if __name__ == '__main__':
    main()
