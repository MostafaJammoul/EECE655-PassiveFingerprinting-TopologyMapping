#!/usr/bin/env python3
"""
Clean OS Labels in model2.csv

Removes trailing apostrophes from OS labels (caused by nprint dataset).
Creates a cleaned version: model2_cleaned.csv
"""

import pandas as pd
import sys
from pathlib import Path


def clean_os_labels(input_path, output_path=None, verbose=True):
    """
    Clean OS labels by removing trailing apostrophes

    Args:
        input_path: Path to model2.csv
        output_path: Where to save cleaned CSV (default: model2_cleaned.csv)
        verbose: Print progress
    """

    if verbose:
        print("="*70)
        print("CLEAN OS LABELS IN MODEL2.CSV")
        print("="*70)

    # Load dataset
    if verbose:
        print(f"\n[1/3] Loading dataset: {input_path}")

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"ERROR: File not found: {input_path}")
        return None

    try:
        df = pd.read_csv(input_path, low_memory=False)
        if verbose:
            print(f"  Loaded {len(df):,} rows with {len(df.columns)} columns")
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        return None

    # Clean os_label
    if verbose:
        print(f"\n[2/3] Cleaning os_label column...")

    if 'os_label' not in df.columns:
        print(f"ERROR: 'os_label' column not found!")
        return None

    # Count labels with trailing apostrophes
    has_apostrophe = df['os_label'].astype(str).str.endswith("'")
    apostrophe_count = has_apostrophe.sum()

    if verbose:
        print(f"  Found {apostrophe_count:,} labels with trailing apostrophe")

        # Show examples before cleaning
        if apostrophe_count > 0:
            print(f"\n  Examples BEFORE cleaning:")
            examples = df[has_apostrophe]['os_label'].head(5).tolist()
            for ex in examples:
                print(f"    '{ex}'")

    # Remove trailing apostrophes
    df['os_label'] = df['os_label'].astype(str).str.rstrip("'")

    if verbose and apostrophe_count > 0:
        print(f"\n  Examples AFTER cleaning:")
        examples_after = df['os_label'].head(5).tolist()
        for ex in examples_after:
            print(f"    '{ex}'")

    # Check if there are any remaining apostrophes
    still_has_apostrophe = df['os_label'].astype(str).str.endswith("'")
    remaining_count = still_has_apostrophe.sum()

    if remaining_count > 0:
        print(f"\n  WARNING: {remaining_count:,} labels still have trailing apostrophe!")
    else:
        if verbose:
            print(f"\n  ✓ All trailing apostrophes removed!")

    # Save cleaned dataset
    if output_path is None:
        # Create output path based on input
        output_path = input_file.parent / f"{input_file.stem}_cleaned.csv"

    if verbose:
        print(f"\n[3/3] Saving cleaned dataset...")
        print(f"  Output: {output_path}")

    try:
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"  ✓ Saved {len(df):,} rows")
    except Exception as e:
        print(f"ERROR saving CSV: {e}")
        return None

    # Summary
    if verbose:
        print("\n" + "="*70)
        print("CLEANING COMPLETE")
        print("="*70)
        print(f"\nCleaned dataset:")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Rows:   {len(df):,}")
        print(f"  Fixed:  {apostrophe_count:,} labels")

        print(f"\n✓ Success! Use {output_path.name} for training.")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Clean OS labels in model2.csv (remove trailing apostrophes)'
    )

    parser.add_argument(
        'input',
        help='Path to model2.csv file'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output path (default: input_cleaned.csv)',
        default=None
    )

    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Overwrite the input file (use with caution!)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Determine output path
    if args.in_place:
        output = args.input
        print("WARNING: Overwriting input file in-place!")
    else:
        output = args.output

    # Clean the dataset
    df = clean_os_labels(
        input_path=args.input,
        output_path=output,
        verbose=not args.quiet
    )

    if df is None:
        sys.exit(1)
