#!/usr/bin/env python3
"""
Merge Multiple CSV Files
Combine multiple extracted CSV files into a single dataset

Usage:
    python merge_csvs.py processed/*.csv -o merged_dataset.csv
    python merge_csvs.py *.csv -o combined.csv
    python merge_csvs.py data/processed/ -o final_dataset.csv
"""

import argparse
import pandas as pd
import glob
from pathlib import Path
import sys


def merge_csv_files(input_pattern, output_file, verbose=True):
    """
    Merge multiple CSV files into one

    Args:
        input_pattern: Glob pattern or directory (e.g., "processed/*.csv" or "processed/")
        output_file: Output CSV filename
        verbose: Print progress
    """

    # Handle directory input
    input_path = Path(input_pattern)
    if input_path.is_dir():
        pattern = str(input_path / "*.csv")
        csv_files = glob.glob(pattern)
    else:
        csv_files = glob.glob(input_pattern)

    if not csv_files:
        print(f"ERROR: No CSV files found matching: {input_pattern}")
        return None

    if verbose:
        print("="*70)
        print("CSV MERGER")
        print("="*70)
        print(f"\nFound {len(csv_files)} CSV files:")
        for f in csv_files[:10]:
            print(f"  - {Path(f).name}")
        if len(csv_files) > 10:
            print(f"  ... and {len(csv_files) - 10} more")

    # Read all CSVs
    if verbose:
        print(f"\nReading CSV files...")

    dfs = []
    total_rows = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            total_rows += len(df)
            if verbose:
                print(f"  {Path(csv_file).name}: {len(df):,} rows")
        except Exception as e:
            print(f"  WARNING: Failed to read {csv_file}: {e}")

    if not dfs:
        print("ERROR: No CSV files could be read!")
        return None

    # Merge
    if verbose:
        print(f"\nMerging {len(dfs)} dataframes...")

    merged = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"  Total rows: {len(merged):,}")
        print(f"  Total columns: {len(merged.columns)}")

    # Show dataset statistics
    if verbose:
        print(f"\nDataset statistics:")

        if 'os_family' in merged.columns:
            print(f"\n  OS Family distribution:")
            for os_fam, count in merged['os_family'].value_counts().items():
                print(f"    {os_fam}: {count:,} packets")

        if 'os_label' in merged.columns:
            print(f"\n  OS Version distribution (top 10):")
            for os_ver, count in merged['os_label'].value_counts().head(10).items():
                print(f"    {os_ver}: {count:,} packets")

        if 'packet_type' in merged.columns:
            print(f"\n  Packet type distribution:")
            for pkt_type, count in merged['packet_type'].value_counts().items():
                print(f"    {pkt_type}: {count:,} packets")

    # Save
    if verbose:
        print(f"\nSaving to {output_file}...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"\nOutput: {output_path}")
    print(f"Total records: {len(merged):,}")
    print(f"Features: {len(merged.columns)}")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple CSV files into a single dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all CSVs in a directory
  python merge_csvs.py processed/ -o merged.csv

  # Merge specific pattern
  python merge_csvs.py "data/processed/*.csv" -o combined.csv

  # Merge specific files
  python merge_csvs.py file1.csv file2.csv file3.csv -o merged.csv

  # Quiet mode
  python merge_csvs.py processed/ -o merged.csv --quiet
        """
    )

    parser.add_argument(
        'input',
        nargs='+',
        help='Input CSV files (glob pattern, directory, or list of files)'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output merged CSV file'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Handle multiple inputs
    if len(args.input) == 1:
        # Single pattern or directory
        input_pattern = args.input[0]
    else:
        # Multiple files specified
        csv_files = args.input
        if not args.quiet:
            print("="*70)
            print("CSV MERGER")
            print("="*70)
            print(f"\nMerging {len(csv_files)} specified files:")
            for f in csv_files:
                print(f"  - {f}")

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
                if not args.quiet:
                    print(f"  {Path(csv_file).name}: {len(df):,} rows")
            except Exception as e:
                print(f"  WARNING: Failed to read {csv_file}: {e}")

        if not dfs:
            print("ERROR: No CSV files could be read!")
            sys.exit(1)

        merged = pd.concat(dfs, ignore_index=True)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)

        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nOutput: {output_path}")
        print(f"Total records: {len(merged):,}")
        print(f"Features: {len(merged.columns)}")
        return

    # Single pattern/directory
    df = merge_csv_files(
        input_pattern=input_pattern,
        output_file=args.output,
        verbose=not args.quiet
    )

    if df is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
