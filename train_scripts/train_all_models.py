#!/usr/bin/env python3
"""
Train All Models - Convenience Script

Trains all three models (Model 1, 2a, 2b) with recommended settings.

Usage:
    python scripts/train_all_models.py                    # Basic training
    python scripts/train_all_models.py --recommended      # Use all recommended flags
    python scripts/train_all_models.py --quick            # Fast training for testing
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time


def run_command(cmd, description, verbose=True):
    """Run a shell command and handle errors"""

    if verbose:
        print("\n" + "="*70)
        print(description)
        print("="*70)
        print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True, capture_output=not verbose)
        elapsed = time.time() - start_time

        if verbose:
            print(f"\n✓ {description} completed in {elapsed:.1f} seconds")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} FAILED after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        return False


def check_prerequisites(verbose=True):
    """Check if preprocessed data exists"""

    # Use absolute paths for dataset files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')

    required_files = [
        os.path.join(data_dir, 'masaryk.csv'),
        os.path.join(data_dir, 'nprint.csv'),
        os.path.join(data_dir, 'cesnet.csv'),
    ]

    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)

    if missing:
        print("\n" + "="*70)
        print("ERROR: Missing preprocessed data files!")
        print("="*70)
        print("\nMissing files:")
        for file in missing:
            print(f"  - {file}")

        print("\nPlease run preprocessing first:")
        print("  python preprocess_scripts/preprocess_all.py")
        print("\nOr run individual preprocessing scripts:")
        if 'masaryk' in str(missing):
            print("  python preprocess_scripts/masaryk_preprocess.py")
        if 'nprint' in str(missing):
            print("  python preprocess_scripts/nprint_preprocess.py")
        if 'cesnet' in str(missing):
            print("  python preprocess_scripts/cesnet_preprocess.py")

        return False

    if verbose:
        print("\n✓ All prerequisite files found")

    return True


def train_model1(args, verbose=True):
    """Train Model 1: OS Family Classifier"""

    # Use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(base_dir, 'train_scripts', 'train_model1_family.py')
    input_path = os.path.join(base_dir, 'data', 'processed', 'masaryk.csv')

    cmd = [
        sys.executable,
        script_path,
        '--input', input_path,
        '--output-dir', args.output_dir,
        '--results-dir', args.results_dir,
    ]

    if args.quiet:
        cmd.append('--quiet')

    return run_command(
        cmd,
        "Training Model 1: OS Family Classifier (Masaryk dataset)",
        verbose=verbose
    )


def train_model2a(args, verbose=True):
    """Train Model 2a: Legacy OS Classifier"""

    # Use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(base_dir, 'train_scripts', 'train_model2a_legacy.py')
    input_path = os.path.join(base_dir, 'data', 'processed', 'nprint.csv')

    cmd = [
        sys.executable,
        script_path,
        '--input', input_path,
        '--output-dir', args.output_dir,
        '--results-dir', args.results_dir,
    ]

    # Recommended: Use SMOTE for minority classes
    if args.recommended or args.use_smote:
        cmd.append('--use-smote')
        if args.smote_threshold:
            cmd.extend(['--smote-threshold', str(args.smote_threshold)])

    if args.quiet:
        cmd.append('--quiet')

    return run_command(
        cmd,
        "Training Model 2a: Legacy OS Classifier (nPrint dataset)",
        verbose=verbose
    )


def train_model2b(args, verbose=True):
    """Train Model 2b: Modern OS Classifier"""

    # Use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(base_dir, 'train_scripts', 'train_model2b_modern.py')
    input_path = os.path.join(base_dir, 'data', 'processed', 'cesnet.csv')

    cmd = [
        sys.executable,
        script_path,
        '--input', input_path,
        '--output-dir', args.output_dir,
        '--results-dir', args.results_dir,
    ]

    # Recommended flags for small dataset
    if args.recommended or args.merge_classes:
        cmd.append('--merge-classes')

    if args.recommended or args.use_adasyn:
        cmd.append('--use-adasyn')
        if args.adasyn_threshold:
            cmd.extend(['--adasyn-threshold', str(args.adasyn_threshold)])

    if args.recommended or args.cross_validate:
        cmd.append('--cross-validate')

    if args.quiet:
        cmd.append('--quiet')

    return run_command(
        cmd,
        "Training Model 2b: Modern OS Classifier (CESNET dataset)",
        verbose=verbose
    )


def print_summary(results, total_time):
    """Print training summary"""

    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    print("\nResults:")
    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {model_name}: {status}")

    all_success = all(results.values())

    if all_success:
        print("\n" + "="*70)
        print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*70)

        print("\nNext steps:")
        print("  1. Review results in results/ directory")
        print("  2. Check confusion matrices and feature importance plots")
        print("  3. Test ensemble: python scripts/predict_ensemble.py")

        print("\nModel files saved in:")
        print("  models/model1_os_family.pkl")
        print("  models/model2a_legacy_os.pkl")
        print("  models/model2b_modern_os.pkl")

    else:
        print("\n" + "="*70)
        print("⚠️  SOME MODELS FAILED TO TRAIN")
        print("="*70)

        print("\nPlease check error messages above and:")
        print("  1. Verify preprocessed data exists")
        print("  2. Check for missing dependencies")
        print("  3. Review error logs")

        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Train all three OS fingerprinting models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (no SMOTE/ADASYN)
  python scripts/train_all_models.py

  # Recommended settings (best accuracy)
  python scripts/train_all_models.py --recommended

  # Quick training for testing
  python scripts/train_all_models.py --quick

  # Custom configuration
  python scripts/train_all_models.py --use-smote --merge-classes --cross-validate
        """
    )

    # Presets
    parser.add_argument(
        '--recommended',
        action='store_true',
        help='Use all recommended settings: SMOTE (Model 2a) + ADASYN (Model 2b) + class merging (Model 2b) + cross-validation (Model 2b). WARNING: --merge-classes defeats fine-grained version classification!'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training for testing (skip SMOTE/ADASYN/CV)'
    )

    # Output directories
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for trained models (default: models/)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results/)'
    )

    # Model-specific options
    parser.add_argument(
        '--use-smote',
        action='store_true',
        help='Use SMOTE for Model 2a (legacy OS)'
    )

    parser.add_argument(
        '--smote-threshold',
        type=int,
        default=100,
        help='SMOTE threshold for Model 2a (default: 100)'
    )

    parser.add_argument(
        '--merge-classes',
        action='store_true',
        help='Merge similar OS classes in Model 2b (WARNING: defeats fine-grained version classification!)'
    )

    parser.add_argument(
        '--use-adasyn',
        action='store_true',
        help='Use ADASYN for Model 2b (HIGHLY RECOMMENDED)'
    )

    parser.add_argument(
        '--adasyn-threshold',
        type=int,
        default=50,
        help='ADASYN threshold for Model 2b (default: 50)'
    )

    parser.add_argument(
        '--cross-validate',
        action='store_true',
        help='Perform cross-validation for Model 2b (RECOMMENDED)'
    )

    # Training control
    parser.add_argument(
        '--skip-model1',
        action='store_true',
        help='Skip training Model 1'
    )

    parser.add_argument(
        '--skip-model2a',
        action='store_true',
        help='Skip training Model 2a'
    )

    parser.add_argument(
        '--skip-model2b',
        action='store_true',
        help='Skip training Model 2b'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output from training scripts'
    )

    args = parser.parse_args()

    # Handle presets
    if args.quick:
        # Quick mode: no SMOTE/ADASYN, no CV, no merging
        args.use_smote = False
        args.use_adasyn = False
        args.merge_classes = False
        args.cross_validate = False
        print("\n⚡ QUICK MODE: Training with basic settings")

    # Print configuration
    print("="*70)
    print("TRAIN ALL MODELS")
    print("="*70)

    print("\nConfiguration:")
    if args.recommended:
        print("  Mode: RECOMMENDED (best accuracy)")
    elif args.quick:
        print("  Mode: QUICK (fast testing)")
    else:
        print("  Mode: CUSTOM")

    print(f"\nOutput directories:")
    print(f"  Models: {args.output_dir}/")
    print(f"  Results: {args.results_dir}/")

    print(f"\nModel 2a (Legacy OS) settings:")
    print(f"  SMOTE: {'Enabled' if args.use_smote or args.recommended else 'Disabled'}")
    if args.use_smote or args.recommended:
        print(f"  SMOTE threshold: {args.smote_threshold}")

    print(f"\nModel 2b (Modern OS) settings:")
    print(f"  Class merging: {'Enabled' if args.merge_classes or args.recommended else 'Disabled'}")
    print(f"  ADASYN: {'Enabled' if args.use_adasyn or args.recommended else 'Disabled'}")
    if args.use_adasyn or args.recommended:
        print(f"  ADASYN threshold: {args.adasyn_threshold}")
    print(f"  Cross-validation: {'Enabled' if args.cross_validate or args.recommended else 'Disabled'}")

    # Check prerequisites
    if not check_prerequisites(verbose=not args.quiet):
        sys.exit(1)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Train models
    results = {}
    start_time = time.time()

    if not args.skip_model1:
        results['Model 1 (OS Family)'] = train_model1(args, verbose=not args.quiet)
    else:
        print("\nSkipping Model 1 (--skip-model1)")

    if not args.skip_model2a:
        results['Model 2a (Legacy OS)'] = train_model2a(args, verbose=not args.quiet)
    else:
        print("\nSkipping Model 2a (--skip-model2a)")

    if not args.skip_model2b:
        results['Model 2b (Modern OS)'] = train_model2b(args, verbose=not args.quiet)
    else:
        print("\nSkipping Model 2b (--skip-model2b)")

    total_time = time.time() - start_time

    # Print summary
    print_summary(results, total_time)


if __name__ == '__main__':
    main()
