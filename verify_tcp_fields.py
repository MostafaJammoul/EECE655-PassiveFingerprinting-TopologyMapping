#!/usr/bin/env python3
"""Quick verification of TCP SYN TTL and TCP flags A extraction"""

import sys

csv_file = 'data/raw/masaryk/flows_ground_truth_merged_anonymized.csv'

print("="*70)
print("VERIFYING TCP FIELD EXTRACTION")
print("="*70)

# Read header
with open(csv_file, 'r', encoding='utf-8') as f:
    header = f.readline().strip().split(';')

    print(f"\n[Position 14] Column name: {header[14]}")
    print(f"[Position 20] Column name: {header[20]}")

    print(f"\n[Sample values from first 10 data rows]")
    print(f"{'Row':<6} {'Pos 14 (TCP flags A)':<25} {'Pos 20 (TCP SYN TTL)':<25}")
    print("-"*70)

    for i in range(10):
        line = f.readline().strip()
        if not line:
            break
        fields = line.split(';')

        pos14_val = fields[14] if len(fields) > 14 else 'N/A'
        pos20_val = fields[20] if len(fields) > 20 else 'N/A'

        print(f"{i+1:<6} {pos14_val:<25} {pos20_val:<25}")

print("\n[Checking if position 14 has non-empty values]")
with open(csv_file, 'r', encoding='utf-8') as f:
    next(f)  # Skip header

    total = 0
    pos14_non_empty = 0
    pos20_non_empty = 0

    for i, line in enumerate(f):
        if i >= 1000:  # Check first 1000 rows
            break
        fields = line.strip().split(';')
        total += 1

        if len(fields) > 14 and fields[14].strip():
            pos14_non_empty += 1
        if len(fields) > 20 and fields[20].strip():
            pos20_non_empty += 1

    print(f"  Position 14 (TCP flags A): {pos14_non_empty}/{total} rows have values ({pos14_non_empty/total*100:.1f}%)")
    print(f"  Position 20 (TCP SYN TTL): {pos20_non_empty}/{total} rows have values ({pos20_non_empty/total*100:.1f}%)")

print("\n" + "="*70)
