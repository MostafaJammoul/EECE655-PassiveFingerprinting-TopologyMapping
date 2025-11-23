#!/usr/bin/env python3
"""
Analyze Masaryk Dataset OS Versions

Extracts and counts all unique OS versions from the Masaryk dataset.
OS fields are at positions 1-5 (semicolon-separated):
  1: UA OS family
  2: UA OS major
  3: UA OS minor
  4: UA OS patch
  5: UA OS patch minor
"""

import sys
from pathlib import Path
from collections import Counter

def analyze_masaryk_os_versions(csv_path='data/raw/masaryk/flows_ground_truth_merged_anonymized.csv'):
    """Extract OS version distribution from Masaryk dataset"""

    csv_file = Path(csv_path)

    if not csv_file.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    print("="*80)
    print("MASARYK DATASET - OS VERSION DISTRIBUTION")
    print("="*80)
    print(f"\nAnalyzing: {csv_path}\n")

    # Counters for different levels of detail
    os_family_counts = Counter()
    os_full_version_counts = Counter()

    total_rows = 0

    with open(csv_file, 'r', encoding='utf-8') as f:
        # Skip header
        header = f.readline()

        print("Processing rows...")
        for line in f:
            line = line.strip()
            if not line:
                continue

            fields = line.split(';')

            if len(fields) < 6:
                continue

            total_rows += 1

            # Extract OS fields (positions 1-5)
            os_family = fields[1].strip() if len(fields) > 1 else ''
            os_major = fields[2].strip() if len(fields) > 2 else ''
            os_minor = fields[3].strip() if len(fields) > 3 else ''
            os_patch = fields[4].strip() if len(fields) > 4 else ''
            os_patch_minor = fields[5].strip() if len(fields) > 5 else ''

            # Count family
            if os_family:
                os_family_counts[os_family] += 1

            # Build full version string
            version_parts = [p for p in [os_family, os_major, os_minor, os_patch, os_patch_minor] if p]
            full_version = '.'.join(version_parts) if version_parts else 'Unknown'
            os_full_version_counts[full_version] += 1

            # Progress indicator
            if total_rows % 100000 == 0:
                print(f"  Processed {total_rows:,} rows...")

    print(f"\nTotal rows: {total_rows:,}")

    # Print OS Family distribution
    print("\n" + "="*80)
    print("OS FAMILY DISTRIBUTION")
    print("="*80)
    print(f"\n{'OS Family':<40} {'Count':>12} {'Percentage':>12}")
    print("-"*65)

    for os_family, count in os_family_counts.most_common():
        pct = (count / total_rows) * 100
        print(f"{os_family:<40} {count:>12,} {pct:>11.2f}%")

    print("-"*65)
    print(f"{'TOTAL':<40} {total_rows:>12,} {100.0:>11.2f}%")

    # Print Full OS Version distribution
    print("\n" + "="*80)
    print("FULL OS VERSION DISTRIBUTION (Top 50)")
    print("="*80)
    print(f"\n{'OS Version':<60} {'Count':>12} {'Percentage':>12}")
    print("-"*85)

    for os_version, count in os_full_version_counts.most_common(50):
        pct = (count / total_rows) * 100
        print(f"{os_version:<60} {count:>12,} {pct:>11.2f}%")

    if len(os_full_version_counts) > 50:
        print(f"\n... and {len(os_full_version_counts) - 50} more versions")

    print("\n" + "="*80)
    print(f"Unique OS families: {len(os_family_counts)}")
    print(f"Unique OS versions: {len(os_full_version_counts)}")
    print("="*80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Masaryk OS version distribution')
    parser.add_argument('--input', type=str,
                       default='data/raw/masaryk/flows_ground_truth_merged_anonymized.csv',
                       help='Path to Masaryk CSV file')

    args = parser.parse_args()

    analyze_masaryk_os_versions(args.input)
