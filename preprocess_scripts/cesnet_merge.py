#!/usr/bin/env python3
"""
CESNET Dataset OS Version Merging Script

This script merges and removes specific OS versions from the CESNET dataset
to create better class balance and consolidate similar OS versions.

Operations:
- Merge Debian 10 and Debian 11 → "Debian 10/11"
- Delete Ubuntu 14.04.6 LTS
- Merge Fedora 28 and Fedora 40 → "Fedora 28/40"
- Delete Amazon-Linux 2
- Merge OpenBSD 6.8, 7.0, 7.2, 7.4 → "OpenBSD 7"
- Merge Red Hat Enterprise Linux 8.9 and 9.3 → "Red Hat Enterprise Linux 9"
- Merge Ubuntu 20.04.6 LTS and Ubuntu 22.04.4 LTS → "Ubuntu 20/22 LTS"
- Merge Ubuntu 16.04.6 LTS and Ubuntu 18.04.6 → "Ubuntu 16/18 LTS"
- Merge openSUSE Leap 15.3, 15.2, 15.4 → "openSUSE Leap 15"
- Merge Linux Mint 21.2 and 21.1 → "Linux Mint 21"
- Merge Manjaro 22.0 and 21.0 → "Manjaro 21/22"
- Keep all other OS versions unchanged

Input:  data/processed/cesnet.csv
Output: data/processed/iter2_cesnet.csv
"""

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed.")
    print("Install with: pip install pandas")
    sys.exit(1)


def merge_cesnet_os_versions(input_path='data/processed/cesnet.csv',
                              output_path='data/processed/iter2_cesnet.csv'):
    """
    Merge and remove specific OS versions from CESNET dataset

    Args:
        input_path: Path to original cesnet.csv
        output_path: Path to save merged dataset

    Returns:
        DataFrame with merged OS versions
    """

    print("="*70)
    print("CESNET DATASET - OS VERSION MERGING")
    print("="*70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")

    # Read the dataset
    csv_file = Path(input_path)
    if not csv_file.exists():
        print(f"\nERROR: File not found: {input_path}")
        print("Run preprocessing first:")
        print("  python preprocess_scripts/cesnet_preprocess.py")
        return None

    print(f"\n[1/4] Reading dataset...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return None

    print(f"  Total records: {len(df):,}")

    # Check if os_label column exists
    if 'os_label' not in df.columns:
        print("\nERROR: 'os_label' column not found in dataset!")
        print(f"Available columns: {', '.join(df.columns)}")
        return None

    # Show original distribution
    print("\n[2/4] Original OS version distribution:")
    original_counts = df['os_label'].value_counts()
    print(f"  Total unique OS versions: {len(original_counts)}")
    print(f"  Total records: {len(df):,}")

    # Create a copy for merging
    df_merged = df.copy()

    print("\n[3/4] Applying merges and deletions...")

    # Track changes
    merge_operations = []
    delete_operations = []

    # === MERGES ===

    # 1. Merge Debian 10 and Debian 11 → "Debian 10/11"
    debian_versions = ['Debian 10', 'Debian 11']
    debian_mask = df_merged['os_label'].isin(debian_versions)
    debian_count = debian_mask.sum()
    if debian_count > 0:
        df_merged.loc[debian_mask, 'os_label'] = 'Debian 10/11'
        merge_operations.append(f"  ✓ Merged {debian_versions} → 'Debian 10/11' ({debian_count:,} records)")

    # 2. Merge Fedora 28 and Fedora 40 → "Fedora 28/40"
    fedora_versions = ['Fedora 28', 'Fedora 40']
    fedora_mask = df_merged['os_label'].isin(fedora_versions)
    fedora_count = fedora_mask.sum()
    if fedora_count > 0:
        df_merged.loc[fedora_mask, 'os_label'] = 'Fedora 28/40'
        merge_operations.append(f"  ✓ Merged {fedora_versions} → 'Fedora 28/40' ({fedora_count:,} records)")

    # 3. Merge OpenBSD 6.8, 7.0, 7.2, 7.4 → "OpenBSD 7"
    openbsd_versions = ['OpenBSD 6.8', 'OpenBSD 7.0', 'OpenBSD 7.2', 'OpenBSD 7.4']
    openbsd_mask = df_merged['os_label'].isin(openbsd_versions)
    openbsd_count = openbsd_mask.sum()
    if openbsd_count > 0:
        df_merged.loc[openbsd_mask, 'os_label'] = 'OpenBSD 7'
        merge_operations.append(f"  ✓ Merged {openbsd_versions} → 'OpenBSD 7' ({openbsd_count:,} records)")

    # 4. Merge Red Hat Enterprise Linux 8.9 and 9.3 → "Red Hat Enterprise Linux 9"
    rhel_versions = ['Red Hat Enterprise Linux 8.9', 'Red Hat Enterprise Linux 9.3']
    rhel_mask = df_merged['os_label'].isin(rhel_versions)
    rhel_count = rhel_mask.sum()
    if rhel_count > 0:
        df_merged.loc[rhel_mask, 'os_label'] = 'Red Hat Enterprise Linux 9'
        merge_operations.append(f"  ✓ Merged {rhel_versions} → 'Red Hat Enterprise Linux 9' ({rhel_count:,} records)")

    # 5. Merge Ubuntu 20.04.6 LTS and Ubuntu 22.04.4 LTS → "Ubuntu 20/22 LTS"
    # Note: Some Ubuntu versions might appear with or without "LTS" suffix
    ubuntu_20_22_versions = ['Ubuntu 20.04.6 LTS', 'Ubuntu 20.04.6', 'Ubuntu 22.04.4 LTS', 'Ubuntu 22.04.4']
    ubuntu_20_22_mask = df_merged['os_label'].isin(ubuntu_20_22_versions)
    ubuntu_20_22_count = ubuntu_20_22_mask.sum()
    if ubuntu_20_22_count > 0:
        df_merged.loc[ubuntu_20_22_mask, 'os_label'] = 'Ubuntu 20/22 LTS'
        merge_operations.append(f"  ✓ Merged {ubuntu_20_22_versions} → 'Ubuntu 20/22 LTS' ({ubuntu_20_22_count:,} records)")

    # 6. Merge Ubuntu 16.04.6 LTS and Ubuntu 18.04.6 → "Ubuntu 16/18 LTS"
    # Note: Ubuntu 18.04.6 might appear with or without "LTS" suffix
    ubuntu_16_18_versions = ['Ubuntu 16.04.6 LTS', 'Ubuntu 18.04.6 LTS', 'Ubuntu 18.04.6']
    ubuntu_16_18_mask = df_merged['os_label'].isin(ubuntu_16_18_versions)
    ubuntu_16_18_count = ubuntu_16_18_mask.sum()
    if ubuntu_16_18_count > 0:
        df_merged.loc[ubuntu_16_18_mask, 'os_label'] = 'Ubuntu 16/18 LTS'
        merge_operations.append(f"  ✓ Merged {ubuntu_16_18_versions} → 'Ubuntu 16/18 LTS' ({ubuntu_16_18_count:,} records)")

    # 7. Merge openSUSE Leap 15.3, 15.2, 15.4 → "openSUSE Leap 15"
    opensuse_versions = ['openSUSE Leap 15.3', 'openSUSE Leap 15.2', 'openSUSE Leap 15.4']
    opensuse_mask = df_merged['os_label'].isin(opensuse_versions)
    opensuse_count = opensuse_mask.sum()
    if opensuse_count > 0:
        df_merged.loc[opensuse_mask, 'os_label'] = 'openSUSE Leap 15'
        merge_operations.append(f"  ✓ Merged {opensuse_versions} → 'openSUSE Leap 15' ({opensuse_count:,} records)")

    # 8. Merge Linux Mint 21.2 and 21.1 → "Linux Mint 21"
    mint_versions = ['Linux Mint 21.2', 'Linux Mint 21.1']
    mint_mask = df_merged['os_label'].isin(mint_versions)
    mint_count = mint_mask.sum()
    if mint_count > 0:
        df_merged.loc[mint_mask, 'os_label'] = 'Linux Mint 21'
        merge_operations.append(f"  ✓ Merged {mint_versions} → 'Linux Mint 21' ({mint_count:,} records)")

    # 9. Merge Manjaro 22.0 and 21.0 → "Manjaro 21/22"
    manjaro_versions = ['Manjaro 22.0', 'Manjaro 21.0']
    manjaro_mask = df_merged['os_label'].isin(manjaro_versions)
    manjaro_count = manjaro_mask.sum()
    if manjaro_count > 0:
        df_merged.loc[manjaro_mask, 'os_label'] = 'Manjaro 21/22'
        merge_operations.append(f"  ✓ Merged {manjaro_versions} → 'Manjaro 21/22' ({manjaro_count:,} records)")

    # === DELETIONS ===

    # 1. Delete Ubuntu 14.04.6 LTS
    ubuntu_delete = 'Ubuntu 14.04.6 LTS'
    ubuntu_mask = df_merged['os_label'] == ubuntu_delete
    ubuntu_count = ubuntu_mask.sum()
    if ubuntu_count > 0:
        df_merged = df_merged[~ubuntu_mask]
        delete_operations.append(f"  ✓ Deleted '{ubuntu_delete}' ({ubuntu_count:,} records)")

    # 2. Delete Amazon-Linux 2
    amazon_delete = 'Amazon-Linux 2'
    amazon_mask = df_merged['os_label'] == amazon_delete
    amazon_count = amazon_mask.sum()
    if amazon_count > 0:
        df_merged = df_merged[~amazon_mask]
        delete_operations.append(f"  ✓ Deleted '{amazon_delete}' ({amazon_count:,} records)")

    # Print operations
    if merge_operations:
        print("\n  Merge operations:")
        for op in merge_operations:
            print(op)
    else:
        print("\n  No merge operations performed (OS versions not found)")

    if delete_operations:
        print("\n  Delete operations:")
        for op in delete_operations:
            print(op)
    else:
        print("\n  No delete operations performed (OS versions not found)")

    # Show final distribution
    print("\n[4/4] Final OS version distribution:")
    final_counts = df_merged['os_label'].value_counts()
    print(f"  Total unique OS versions: {len(final_counts)}")
    print(f"  Total records: {len(df_merged):,}")

    print(f"\n{'OS Version':<50} {'Count':>10} {'Percentage':>12}")
    print("-"*75)

    total = len(df_merged)
    for os_version, count in final_counts.items():
        percentage = (count / total) * 100
        print(f"{str(os_version):<50} {count:>10,} {percentage:>11.2f}%")

    print("-"*75)
    print(f"{'TOTAL':<50} {total:>10,} {100.0:>11.2f}%")

    # Save merged dataset
    print(f"\n[5/5] Saving merged dataset...")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_merged.to_csv(output_path, index=False)
        print(f"  ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"ERROR saving file: {e}")
        return None

    # Summary
    print("\n" + "="*70)
    print("MERGING COMPLETE")
    print("="*70)
    print(f"\nRecords before: {len(df):,}")
    print(f"Records after:  {len(df_merged):,}")
    print(f"Records removed: {len(df) - len(df_merged):,}")
    print(f"\nOS versions before: {len(original_counts)}")
    print(f"OS versions after:  {len(final_counts)}")
    print(f"OS versions reduced: {len(original_counts) - len(final_counts)}")

    return df_merged


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge and remove specific OS versions from CESNET dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/cesnet.csv',
        help='Input CESNET CSV file (default: data/processed/cesnet.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/iter2_cesnet.csv',
        help='Output merged CSV file (default: data/processed/iter2_cesnet.csv)'
    )

    args = parser.parse_args()

    df = merge_cesnet_os_versions(args.input, args.output)

    if df is None:
        sys.exit(1)

    print("\n✓ Success! Merged dataset ready for training.")
    print(f"\nNext steps:")
    print(f"  1. Use {args.output} for Model 2 training")
    print(f"  2. Update training scripts to use merged dataset")


if __name__ == '__main__':
    main()
