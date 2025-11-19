#!/usr/bin/env python3
"""
Merge Packet-Level Datasets for Model 2 (OS Version Classification)

Combines:
- CESNET idle packets (data/processed/cesnet_idle_packets.csv)
- nPrint packets (data/processed/nprint_packets.csv)

Output: data/processed/packet_level_merged.csv

This merged dataset is used to train Model 2, which predicts OS VERSION
given packet-level features + predicted OS family from Model 1.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


# ============================================================================
# SCHEMA UNIFICATION
# ============================================================================

# Unified schema for packet-level features
PACKET_SCHEMA = [
    # Metadata
    'dataset_source',
    'record_id',
    'timestamp',

    # Network info
    'src_ip',
    'dst_ip',
    'protocol',
    'src_port',
    'dst_port',

    # IP layer features (CRITICAL for OS fingerprinting)
    'ttl',
    'initial_ttl',
    'df_flag',
    'ip_len',

    # TCP layer features (CRITICAL for OS fingerprinting)
    'tcp_window_size',
    'tcp_window_scale',
    'tcp_mss',
    'tcp_options_order',
    'tcp_flags',

    # Labels
    'os_label',
    'os_family',
]


def ensure_schema(df, verbose=True):
    """
    Ensure DataFrame has all columns from unified schema

    Adds missing columns as None
    Reorders columns to match schema
    """
    # Add missing columns
    for col in PACKET_SCHEMA:
        if col not in df.columns:
            df[col] = None
            if verbose:
                print(f"    Added missing column: {col}")

    # Reorder columns
    df = df[PACKET_SCHEMA]

    return df


# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

def validate_dataset(df, dataset_name, verbose=True):
    """
    Validate dataset quality and report issues

    Returns: (is_valid, warnings)
    """
    warnings = []

    # Check if empty
    if len(df) == 0:
        warnings.append(f"{dataset_name}: Empty dataset!")
        return False, warnings

    # Check for OS labels
    if df['os_label'].isna().all():
        warnings.append(f"{dataset_name}: No OS labels found!")
        return False, warnings

    # Check for OS family
    if df['os_family'].isna().all():
        warnings.append(f"{dataset_name}: No OS family labels!")
        return False, warnings

    # Check for critical features
    critical_features = ['ttl', 'tcp_window_size', 'tcp_mss']
    for feat in critical_features:
        if feat in df.columns:
            pct_available = (df[feat].notna().sum() / len(df)) * 100
            if pct_available < 50:
                warnings.append(f"{dataset_name}: {feat} only {pct_available:.1f}% available")

    # Check for duplicate records
    dup_count = df.duplicated(subset=['record_id']).sum()
    if dup_count > 0:
        warnings.append(f"{dataset_name}: {dup_count} duplicate record IDs found")

    if verbose:
        print(f"\n  Validation for {dataset_name}:")
        print(f"    Records: {len(df):,}")
        print(f"    OS families: {df['os_family'].nunique()}")
        print(f"    OS versions: {df['os_label'].nunique()}")

        if warnings:
            print(f"    Warnings:")
            for w in warnings:
                print(f"      - {w}")
        else:
            print(f"    ✓ No issues")

    return True, warnings


# ============================================================================
# FEATURE COMPLETENESS
# ============================================================================

def analyze_feature_completeness(df, verbose=True):
    """Analyze which features are available across datasets"""

    if verbose:
        print(f"\n  Feature availability:")

    critical_features = [
        'ttl',
        'initial_ttl',
        'tcp_window_size',
        'tcp_mss',
        'df_flag',
        'tcp_window_scale',
        'tcp_options_order',
    ]

    availability = {}
    for feat in critical_features:
        pct = (df[feat].notna().sum() / len(df)) * 100
        availability[feat] = pct
        if verbose:
            status = "✓" if pct >= 75 else "⚠" if pct >= 25 else "✗"
            print(f"    {status} {feat}: {pct:.1f}%")

    return availability


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def merge_packet_datasets(cesnet_idle_path='data/processed/cesnet_idle_packets.csv',
                          nprint_path='data/processed/nprint_packets.csv',
                          output_dir='data/processed',
                          verbose=True):
    """
    Merge CESNET idle and nPrint datasets into unified packet-level dataset

    Args:
        cesnet_idle_path: Path to processed CESNET idle CSV
        nprint_path: Path to processed nPrint CSV
        output_dir: Where to save merged CSV
        verbose: Print detailed info

    Returns:
        Merged DataFrame
    """

    print("="*70)
    print("MERGE PACKET-LEVEL DATASETS (for Model 2)")
    print("="*70)
    print(f"\nInputs:")
    print(f"  - {cesnet_idle_path}")
    print(f"  - {nprint_path}")
    print(f"\nOutput:")
    print(f"  - {output_dir}/packet_level_merged.csv")

    # Load datasets
    print(f"\n[1/4] Loading datasets...")

    dfs_to_merge = []
    all_warnings = []

    # Load CESNET idle
    if Path(cesnet_idle_path).exists():
        print(f"  Loading CESNET idle...")
        df_cesnet = pd.read_csv(cesnet_idle_path)
        print(f"    Loaded {len(df_cesnet):,} records")

        # Ensure schema
        df_cesnet = ensure_schema(df_cesnet, verbose=False)

        # Validate
        is_valid, warnings = validate_dataset(df_cesnet, "CESNET idle", verbose=verbose)
        if is_valid:
            dfs_to_merge.append(df_cesnet)
        all_warnings.extend(warnings)
    else:
        print(f"  WARNING: CESNET idle not found at {cesnet_idle_path}")
        print(f"  Run: python scripts/preprocess_cesnet_idle.py")

    # Load nPrint
    if Path(nprint_path).exists():
        print(f"  Loading nPrint...")
        df_nprint = pd.read_csv(nprint_path)
        print(f"    Loaded {len(df_nprint):,} records")

        # Ensure schema
        df_nprint = ensure_schema(df_nprint, verbose=False)

        # Validate
        is_valid, warnings = validate_dataset(df_nprint, "nPrint", verbose=verbose)
        if is_valid:
            dfs_to_merge.append(df_nprint)
        all_warnings.extend(warnings)
    else:
        print(f"  WARNING: nPrint not found at {nprint_path}")
        print(f"  Run: python scripts/preprocess_nprint.py")

    if not dfs_to_merge:
        print(f"\nERROR: No datasets available to merge!")
        print(f"\nPlease preprocess datasets first:")
        print(f"  python scripts/preprocess_cesnet_idle.py")
        print(f"  python scripts/preprocess_nprint.py")
        return None

    # Merge
    print(f"\n[2/4] Merging {len(dfs_to_merge)} datasets...")
    df_merged = pd.concat(dfs_to_merge, ignore_index=True)
    print(f"  Total records: {len(df_merged):,}")

    # Dataset distribution
    print(f"\n  Records per dataset:")
    for source, count in df_merged['dataset_source'].value_counts().items():
        pct = (count / len(df_merged)) * 100
        print(f"    {source}: {count:,} ({pct:.1f}%)")

    # OS distribution
    print(f"\n  OS Family distribution:")
    for family, count in df_merged['os_family'].value_counts().items():
        print(f"    {family}: {count:,}")

    print(f"\n  OS Version distribution (top 10):")
    for os_ver, count in df_merged['os_label'].value_counts().head(10).items():
        print(f"    {os_ver}: {count:,}")

    # Feature analysis
    print(f"\n[3/4] Analyzing feature completeness...")
    availability = analyze_feature_completeness(df_merged, verbose=verbose)

    # Check for minimum viable features
    min_required = ['ttl', 'tcp_window_size']
    missing_critical = [f for f in min_required if availability.get(f, 0) < 25]

    if missing_critical:
        print(f"\n  ⚠ WARNING: Critical features have <25% availability:")
        for feat in missing_critical:
            print(f"    - {feat}: {availability.get(feat, 0):.1f}%")
        print(f"  Model performance may be degraded.")

    # Save
    print(f"\n[4/4] Saving merged dataset...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'packet_level_merged.csv')
    df_merged.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # Final summary
    print("\n" + "="*70)
    print("MERGE COMPLETE")
    print("="*70)
    print(f"\nMerged dataset:")
    print(f"  Records: {len(df_merged):,}")
    print(f"  OS Families: {df_merged['os_family'].nunique()}")
    print(f"  OS Versions: {df_merged['os_label'].nunique()}")
    print(f"  Features: {len(df_merged.columns)}")

    if all_warnings:
        print(f"\n⚠ Total warnings: {len(all_warnings)}")
        print(f"  (See details above)")

    print(f"\nThis dataset is ready for Model 2 training!")
    print(f"  - Input: Packet-level features")
    print(f"  - Output: OS VERSION prediction")
    print(f"  - Will also receive predicted OS FAMILY from Model 1")

    return df_merged


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Merge packet-level datasets for Model 2 training'
    )

    parser.add_argument(
        '--cesnet-idle',
        type=str,
        default='data/processed/cesnet_idle_packets.csv',
        help='Path to processed CESNET idle CSV'
    )

    parser.add_argument(
        '--nprint',
        type=str,
        default='data/processed/nprint_packets.csv',
        help='Path to processed nPrint CSV'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for merged CSV'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Run merge
    df = merge_packet_datasets(
        cesnet_idle_path=args.cesnet_idle,
        nprint_path=args.nprint,
        output_dir=args.output,
        verbose=not args.quiet
    )

    if df is None:
        sys.exit(1)

    print("\n✓ Success! Merged dataset ready for Model 2 training.")
    print(f"\nNext step: Train the two-model pipeline:")
    print(f"  python scripts/train_two_model_pipeline.py")


if __name__ == '__main__':
    main()
