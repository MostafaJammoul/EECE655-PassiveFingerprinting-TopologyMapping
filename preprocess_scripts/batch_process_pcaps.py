#!/usr/bin/env python3
"""
Batch Process Multiple PCAP Files
Process multiple PCAP/PCAPNG files and extract features to CSV

Usage:
    python batch_process_pcaps.py captures/ -o processed/
    python batch_process_pcaps.py captures/*.pcapng -o processed/
    python batch_process_pcaps.py -i captures/ -o processed/ --merge
"""

import argparse
import subprocess
import glob
from pathlib import Path
import sys


def process_pcaps_batch(input_pattern, output_dir, merge_output=None, syn_only=True, verbose=True):
    """
    Process multiple PCAP files in batch

    Args:
        input_pattern: Glob pattern or directory (e.g., "captures/*.pcapng" or "captures/")
        output_dir: Output directory for individual CSVs
        merge_output: If provided, merge all CSVs into this file
        syn_only: Extract SYN packets only
        verbose: Print progress
    """

    # Handle directory input
    input_path = Path(input_pattern)
    if input_path.is_dir():
        # Find all PCAP files in directory
        pcap_files = list(input_path.glob("*.pcap")) + list(input_path.glob("*.pcapng"))
    else:
        # Use glob pattern
        pcap_files = [Path(f) for f in glob.glob(input_pattern)]

    if not pcap_files:
        print(f"ERROR: No PCAP files found matching: {input_pattern}")
        return False

    print("="*70)
    print("BATCH PCAP PROCESSOR")
    print("="*70)
    print(f"\nFound {len(pcap_files)} PCAP files:")
    for f in pcap_files[:10]:
        print(f"  - {f.name}")
    if len(pcap_files) > 10:
        print(f"  ... and {len(pcap_files) - 10} more")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing files...")
    print(f"Output directory: {output_path}")

    successful = 0
    failed = 0

    for i, pcap_file in enumerate(pcap_files, 1):
        # Generate output filename
        output_csv = output_path / f"{pcap_file.stem}.csv"

        # Try to infer OS label from filename
        # Common patterns: "windows11_capture.pcapng", "ubuntu_22_04.pcap"
        filename = pcap_file.stem.lower()
        os_label = None

        # Check for OS names in filename
        if 'windows' in filename or 'win' in filename:
            if '11' in filename:
                os_label = 'Windows 11'
            elif '10' in filename:
                os_label = 'Windows 10'
            elif '7' in filename:
                os_label = 'Windows 7'
            else:
                os_label = 'Windows'
        elif 'ubuntu' in filename:
            # Try to extract version
            if '22' in filename:
                os_label = 'Ubuntu 22.04'
            elif '20' in filename:
                os_label = 'Ubuntu 20.04'
            else:
                os_label = 'Ubuntu'
        elif 'debian' in filename:
            os_label = 'Debian'
        elif 'macos' in filename or 'mac' in filename:
            os_label = 'macOS'
        elif 'linux' in filename:
            os_label = 'Linux'

        print(f"\n[{i}/{len(pcap_files)}] {pcap_file.name}")
        if os_label:
            print(f"  Detected OS: {os_label}")

        # Build command
        cmd = [
            sys.executable,  # Use same Python interpreter
            'scripts/pcapng_to_csv_windows.py',
            str(pcap_file),
            str(output_csv)
        ]

        if os_label:
            cmd.extend(['--os-label', os_label])

        if not syn_only:
            cmd.append('--all-tcp')

        if not verbose:
            cmd.append('--quiet')

        # Run the extraction script
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if verbose and result.stdout:
                print(result.stdout)
            successful += 1
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Failed to process {pcap_file.name}")
            if e.stderr:
                print(f"  {e.stderr}")
            failed += 1

    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"\nSuccessful: {successful}/{len(pcap_files)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(pcap_files)}")

    # Merge if requested
    if merge_output and successful > 0:
        print(f"\nMerging all CSV files...")

        csv_files = list(output_path.glob("*.csv"))
        if csv_files:
            cmd = [
                sys.executable,
                'scripts/merge_csvs.py',
                str(output_path),
                '-o', merge_output
            ]
            if not verbose:
                cmd.append('--quiet')

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print("  WARNING: Merge failed")

    return successful > 0


def main():
    parser = argparse.ArgumentParser(
        description='Batch process multiple PCAP/PCAPNG files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PCAPs in a directory
  python batch_process_pcaps.py captures/ -o processed/

  # Process specific pattern
  python batch_process_pcaps.py "captures/*.pcapng" -o processed/

  # Process and merge into single CSV
  python batch_process_pcaps.py captures/ -o processed/ --merge merged.csv

  # Extract all TCP packets (not just SYN)
  python batch_process_pcaps.py captures/ -o processed/ --all-tcp

Note: The script will try to detect OS labels from filenames.
For best results, name your files like:
  - windows11_capture.pcapng
  - ubuntu_22_04.pcap
  - macos14_traffic.pcap
        """
    )

    parser.add_argument(
        'input',
        help='Input directory or glob pattern (e.g., "captures/" or "*.pcapng")'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for CSV files'
    )

    parser.add_argument(
        '--merge',
        type=str,
        default=None,
        help='Merge all CSVs into this file'
    )

    parser.add_argument(
        '--all-tcp',
        action='store_true',
        help='Extract all TCP packets (default: SYN packets only)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    success = process_pcaps_batch(
        input_pattern=args.input,
        output_dir=args.output,
        merge_output=args.merge,
        syn_only=not args.all_tcp,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
