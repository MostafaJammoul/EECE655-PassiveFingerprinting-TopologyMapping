#!/usr/bin/env python3
"""
Master Preprocessing Pipeline - Prepare All Datasets for Two-Model Architecture

Runs all preprocessing steps in the correct order:
1. Preprocess CESNET idle (packet-level for Model 2)
2. Preprocess nPrint (packet-level for Model 2)
3. Merge packet-level datasets (CESNET idle + nPrint) → Model 2 training data
4. Preprocess Masaryk (flow-level for Model 1) → Model 1 training data

Final outputs:
- data/processed/packet_level_merged.csv  (for Model 2: OS version prediction)
- data/processed/masaryk_processed.csv (for Model 1: OS family prediction)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================================
# UTILITIES
# ============================================================================

def print_header(text):
    """Print a nice header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def run_script(script_name, args=None, verbose=True):
    """
    Run a preprocessing script and return success/failure

    Args:
        script_name: Name of script (e.g., "preprocess_cesnet_idle.py")
        args: List of additional arguments
        verbose: Print output

    Returns:
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=not verbose,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Script failed with exit code {e.returncode}")
        if not verbose and e.output:
            print(e.output)
        return False


def check_raw_data_exists(verbose=True):
    """
    Check if raw data exists before preprocessing

    Returns: dict of {dataset_name: exists}
    """
    raw_dir = Path("data/raw")

    datasets = {
        'cesnet_idle': raw_dir / 'cesnet_idle',
        'nprint': raw_dir / 'nprint',
        'masaryk': raw_dir / 'masaryk',
    }

    status = {}

    for name, path in datasets.items():
        if not path.exists():
            status[name] = False
        elif name == 'cesnet_idle':
            # Check for PCAP files
            pcap_count = len(list(path.glob('**/*.pcap')))
            status[name] = pcap_count > 0
        elif name == 'nprint':
            # Check for CSV or NPZ files
            file_count = len(list(path.glob('*.csv'))) + len(list(path.glob('*.npz')))
            status[name] = file_count > 0
        elif name == 'masaryk':
            # Check for CSV file
            csv_count = len(list(path.glob('*.csv'))) + len(list(path.glob('*.csv.gz')))
            status[name] = csv_count > 0
        else:
            status[name] = False

    if verbose:
        print("Raw data availability:")
        for name, exists in status.items():
            status_str = "✓ Found" if exists else "✗ Not found"
            print(f"  {status_str} - {name}")

    return status


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def preprocess_all(sample=None, skip_flow=False, verbose=True):
    """
    Run complete preprocessing pipeline

    Args:
        sample: Sample size for testing (applies to CESNET flow)
        skip_flow: Skip CESNET flow preprocessing (it's large!)
        verbose: Print detailed output

    Returns:
        True if all steps succeeded
    """

    start_time = datetime.now()

    print_header("MASTER PREPROCESSING PIPELINE")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if raw data exists
    print_header("Step 0: Checking Raw Data")
    data_status = check_raw_data_exists(verbose=True)

    missing = [name for name, exists in data_status.items() if not exists]
    if missing:
        print(f"\n⚠ WARNING: Missing raw data for: {', '.join(missing)}")
        print(f"\nPlease download datasets first!")
        print(f"See DATASET_SETUP_GUIDE.md for instructions.\n")

        # Ask if user wants to continue anyway
        response = input("Continue with available datasets? (y/n): ")
        if response.lower() != 'y':
            return False

    # Step 1: CESNET idle
    if data_status.get('cesnet_idle'):
        print_header("Step 1: Preprocess CESNET Idle (Packet-Level)")
        success = run_script('preprocess_cesnet_idle.py', verbose=verbose)
        if not success:
            print("\n✗ CESNET idle preprocessing failed!")
            return False
        print("\n✓ CESNET idle preprocessing complete")
    else:
        print_header("Step 1: CESNET Idle - SKIPPED (data not found)")

    # Step 2: nPrint
    if data_status.get('nprint'):
        print_header("Step 2: Preprocess nPrint (Packet-Level)")
        success = run_script('preprocess_nprint.py', verbose=verbose)
        if not success:
            print("\n✗ nPrint preprocessing failed!")
            return False
        print("\n✓ nPrint preprocessing complete")
    else:
        print_header("Step 2: nPrint - SKIPPED (data not found)")

    # Step 3: Merge packet-level datasets
    print_header("Step 3: Merge Packet-Level Datasets (for Model 2)")
    success = run_script('merge_packet_datasets.py', verbose=verbose)
    if not success:
        print("\n✗ Packet-level merge failed!")
        return False
    print("\n✓ Packet-level merge complete")

    # Step 4: Masaryk
    if not skip_flow and data_status.get('masaryk'):
        print_header("Step 4: Preprocess Masaryk (Flow-Level for Model 1)")

        args = []
        if sample:
            args.extend(['--sample', str(sample)])
            print(f"Using sample size: {sample:,} (for testing)")

        success = run_script('preprocess_masaryk.py', args=args, verbose=verbose)
        if not success:
            print("\n✗ Masaryk preprocessing failed!")
            return False
        print("\n✓ Masaryk preprocessing complete")
    else:
        print_header("Step 4: Masaryk - SKIPPED")
        if skip_flow:
            print("(User requested skip)")
        else:
            print("(Data not found)")

    # Done!
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_header("PREPROCESSING COMPLETE")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    # Summary of outputs
    print("\nGenerated datasets:")

    output_files = {
        'packet_level_merged.csv': 'Model 2 training data (OS version from packets)',
        'masaryk_processed.csv': 'Model 1 training data (OS family from flows)',
        'cesnet_idle_packets.csv': 'CESNET idle intermediate',
        'nprint_packets.csv': 'nPrint intermediate',
    }

    for filename, description in output_files.items():
        filepath = Path('data/processed') / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename}")
            print(f"    {description}")
            print(f"    Size: {size_mb:.1f} MB")
        else:
            print(f"  ✗ {filename} (not generated)")

    print("\nNext steps:")
    print("  1. Train Model 1 (family): python scripts/train_model1_family.py")
    print("  2. Train Model 2 (version): python scripts/train_model2_version.py")
    print("  3. Or train both: python scripts/train_two_model_pipeline.py")

    return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Master preprocessing pipeline - prepare all datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full preprocessing (all datasets)
  python scripts/preprocess_all.py

  # Quick test with sampled Masaryk
  python scripts/preprocess_all.py --sample 100000

  # Skip Masaryk (if only testing packet-level)
  python scripts/preprocess_all.py --skip-flow

  # Quiet mode
  python scripts/preprocess_all.py --quiet
        """
    )

    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample size for Masaryk (for testing, e.g., 100000)'
    )

    parser.add_argument(
        '--skip-flow',
        action='store_true',
        help='Skip Masaryk preprocessing (saves time if only testing packet-level)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output from scripts'
    )

    args = parser.parse_args()

    # Run pipeline
    success = preprocess_all(
        sample=args.sample,
        skip_flow=args.skip_flow,
        verbose=not args.quiet
    )

    if not success:
        print("\n✗ Preprocessing pipeline failed!")
        sys.exit(1)

    print("\n✓ All preprocessing complete!")
    sys.exit(0)


if __name__ == '__main__':
    main()
