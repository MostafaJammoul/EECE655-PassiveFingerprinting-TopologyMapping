#!/usr/bin/env python3
"""
Comprehensive CSV Statistics Analyzer

Provides detailed statistics for any CSV file, including:
- All column names and data types
- OS distribution (if os_family column exists)
- Null values in all columns
- Null values per OS version
- Value ranges for numeric columns
- Sample data

Usage:
    python analyze_csv_statistics.py <csv_file>
    python analyze_csv_statistics.py data/processed/nprint_windows_flows.csv
    python analyze_csv_statistics.py data/processed/nprint_windows_flows.csv --detailed
"""

import sys
import argparse
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy are required.")
    print("Install with: pip install pandas numpy")
    sys.exit(1)


def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def analyze_basic_info(df, csv_path):
    """Show basic dataset information"""
    print_section_header("BASIC DATASET INFORMATION")

    print(f"\nFile: {csv_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Total cells: {df.shape[0] * df.shape[1]:,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")


def analyze_columns(df):
    """Show all column names and data types"""
    print_section_header("ALL COLUMNS")

    print(f"\n{'#':<5} {'Column Name':<50} {'Data Type':<15} {'Non-Null':<12} {'Null %':<10}")
    print("-"*95)

    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100

        status = "✓" if null_pct < 5 else ("⚠" if null_pct < 20 else "✗")
        print(f"{status} {i:<3} {col:<50} {dtype:<15} {non_null:>10,} {null_pct:>8.2f}%")


def analyze_os_distribution(df):
    """Analyze OS distribution if os_family or os_label column exists"""
    print_section_header("OS DISTRIBUTION")

    # Try to find OS column
    os_column = None
    if 'os_family' in df.columns:
        os_column = 'os_family'
    elif 'os_label' in df.columns:
        os_column = 'os_label'

    if not os_column:
        print("\n⚠ No OS column found (os_family or os_label)")
        return None

    os_counts = df[os_column].value_counts()
    total = len(df)

    print(f"\n{'OS Version':<40} {'Count':<12} {'Percentage':<12}")
    print("-"*65)

    for os_name, count in os_counts.items():
        pct = (count / total) * 100
        print(f"  {str(os_name):<38} {count:>10,} {pct:>10.2f}%")

    print("-"*65)
    print(f"  {'TOTAL':<38} {total:>10,} {100.0:>10.2f}%")

    # Imbalance analysis
    if len(os_counts) > 1:
        max_count = os_counts.max()
        min_count = os_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        print(f"\nImbalance Analysis:")
        print(f"  Most common:  {os_counts.index[0]} ({max_count:,} samples)")
        print(f"  Least common: {os_counts.index[-1]} ({min_count:,} samples)")
        print(f"  Ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 10:
            print(f"  ⚠ SEVERE imbalance detected (>10:1)")
        elif imbalance_ratio > 3:
            print(f"  ⚠ MODERATE imbalance detected (>3:1)")
        else:
            print(f"  ✓ Relatively balanced")

    return os_column


def analyze_null_values(df):
    """Analyze null values across all columns"""
    print_section_header("NULL VALUES ANALYSIS (ALL COLUMNS)")

    total_cells = df.shape[0] * df.shape[1]
    total_nulls = df.isnull().sum().sum()
    null_pct = (total_nulls / total_cells) * 100

    print(f"\nOverall:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Null cells: {total_nulls:,} ({null_pct:.2f}%)")

    # Per-column null values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].sort_values(ascending=False)

    if len(cols_with_nulls) > 0:
        print(f"\nColumns with null values ({len(cols_with_nulls)} out of {len(df.columns)}):")
        print(f"\n{'Column':<50} {'Null Count':<12} {'Null %':<10}")
        print("-"*72)

        for col, count in cols_with_nulls.items():
            pct = (count / len(df)) * 100
            status = "✗" if pct > 50 else ("⚠" if pct > 20 else "·")
            print(f"{status} {col:<48} {count:>10,} {pct:>8.2f}%")
    else:
        print(f"\n✓ No null values found in any column!")


def analyze_null_values_per_os(df, os_column):
    """Analyze null values broken down by OS version"""
    print_section_header("NULL VALUES PER OS VERSION")

    if not os_column:
        print("\n⚠ Cannot analyze - no OS column found")
        return

    os_versions = df[os_column].unique()

    # Get columns with any nulls
    cols_with_nulls = df.columns[df.isnull().any()].tolist()

    if not cols_with_nulls:
        print("\n✓ No null values in any column for any OS version!")
        return

    print(f"\nAnalyzing {len(cols_with_nulls)} columns with null values across {len(os_versions)} OS versions")

    for os_name in sorted(os_versions):
        os_df = df[df[os_column] == os_name]
        os_total = len(os_df)

        # Find columns with nulls for this OS
        os_nulls = os_df.isnull().sum()
        os_cols_with_nulls = os_nulls[os_nulls > 0].sort_values(ascending=False)

        if len(os_cols_with_nulls) > 0:
            print(f"\n{os_name} ({os_total:,} samples):")
            print(f"  {'Column':<48} {'Nulls':<10} {'% of OS':<10}")
            print("  " + "-"*68)

            # Show top 10 columns with nulls for this OS
            for col, count in os_cols_with_nulls.head(10).items():
                pct = (count / os_total) * 100
                status = "✗" if pct > 50 else ("⚠" if pct > 20 else "·")
                print(f"  {status} {col:<46} {count:>8,} {pct:>8.2f}%")

            if len(os_cols_with_nulls) > 10:
                print(f"  ... and {len(os_cols_with_nulls) - 10} more columns with nulls")
        else:
            print(f"\n{os_name} ({os_total:,} samples): ✓ No null values")


def analyze_value_ranges(df):
    """Analyze value ranges for numeric columns"""
    print_section_header("VALUE RANGES (NUMERIC COLUMNS)")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print("\n⚠ No numeric columns found")
        return

    print(f"\nFound {len(numeric_cols)} numeric columns")
    print(f"\n{'Column':<45} {'Min':<12} {'Max':<12} {'Mean':<12} {'Median':<12}")
    print("-"*95)

    for col in numeric_cols:
        if df[col].notna().sum() == 0:
            print(f"  {col:<45} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue

        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        median_val = df[col].median()

        # Format based on value range
        if max_val < 100:
            fmt = ".2f"
        elif max_val < 10000:
            fmt = ".1f"
        else:
            fmt = ".0f"

        print(f"  {col:<45} {min_val:<12{fmt}} {max_val:<12{fmt}} {mean_val:<12{fmt}} {median_val:<12{fmt}}")


def analyze_categorical_columns(df):
    """Analyze categorical/string columns"""
    print_section_header("CATEGORICAL COLUMNS")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not categorical_cols:
        print("\n⚠ No categorical columns found")
        return

    print(f"\nFound {len(categorical_cols)} categorical columns")
    print(f"\n{'Column':<45} {'Unique Values':<15} {'Most Common':<20}")
    print("-"*85)

    for col in categorical_cols:
        unique_count = df[col].nunique()

        # Get most common value
        if df[col].notna().sum() > 0:
            most_common = df[col].value_counts().index[0]
            most_common_str = str(most_common)[:18]  # Truncate if too long
        else:
            most_common_str = "N/A"

        print(f"  {col:<45} {unique_count:>13,} {most_common_str:<20}")


def show_sample_data(df, n=5):
    """Show sample rows from the dataset"""
    print_section_header(f"SAMPLE DATA (First {n} rows)")

    print(f"\n{df.head(n).to_string()}")


def analyze_feature_groups(df):
    """Analyze features by logical groups"""
    print_section_header("FEATURE COMPLETENESS BY GROUP")

    # Define feature groups based on common naming patterns
    groups = {
        'TCP Features': [col for col in df.columns if 'tcp' in col.lower()],
        'IP Features': [col for col in df.columns if 'ip' in col.lower() or 'ttl' in col.lower()],
        'TLS Features': [col for col in df.columns if 'tls' in col.lower()],
        'Flow Stats': [col for col in df.columns if any(x in col.lower() for x in ['packet', 'bytes', 'flow'])],
        'Port Features': [col for col in df.columns if 'port' in col.lower()],
    }

    for group_name, cols in groups.items():
        if not cols:
            continue

        print(f"\n{group_name} ({len(cols)} features):")

        for col in sorted(cols):
            if col not in df.columns:
                continue

            non_null_pct = (df[col].notna().sum() / len(df)) * 100
            status = "✓" if non_null_pct > 80 else ("⚠" if non_null_pct > 50 else "✗")
            print(f"  {status} {col:<50} {non_null_pct:>6.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive CSV statistics analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_csv_statistics.py data/processed/nprint_windows_flows.csv

  # Detailed analysis with sample data
  python analyze_csv_statistics.py data/processed/nprint_windows_flows.csv --detailed
        """
    )

    parser.add_argument('csv_file', type=str, help='Path to CSV file')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis including sample data')
    parser.add_argument('--sample-rows', type=int, default=5, help='Number of sample rows to show (default: 5)')

    args = parser.parse_args()

    # Validate input
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"ERROR: File not found: {args.csv_file}")
        sys.exit(1)

    # Read CSV
    print("="*80)
    print("COMPREHENSIVE CSV STATISTICS ANALYZER")
    print("="*80)
    print(f"\nReading CSV file...")

    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded {len(df):,} rows")
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        sys.exit(1)

    # Run all analyses
    analyze_basic_info(df, csv_path)
    analyze_columns(df)
    os_column = analyze_os_distribution(df)
    analyze_null_values(df)
    analyze_null_values_per_os(df, os_column)
    analyze_value_ranges(df)
    analyze_categorical_columns(df)
    analyze_feature_groups(df)

    if args.detailed:
        show_sample_data(df, n=args.sample_rows)

    # Final summary
    print_section_header("SUMMARY")

    print(f"\nDataset: {csv_path.name}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    if os_column:
        os_count = df[os_column].nunique()
        print(f"  OS Versions: {os_count}")

    total_nulls = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    null_pct = (total_nulls / total_cells) * 100
    print(f"  Null cells: {total_nulls:,} ({null_pct:.2f}%)")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
