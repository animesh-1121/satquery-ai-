"""
Data Pipeline CLI for SatQueryAI

This module provides command-line interface for dataset operations including
validation, preparation, and inspection.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.adapters import BigEarthNetDataset, VRSBenchDataset, RSVQADataset, CDVQADataset
from data.validation import DataValidator
from data.compatibility import PairCompatibilityChecker
from data.preprocessing import ImagePreprocessor, PreprocessingConfig
from data.manifest import DatasetManifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_dataset(args):
    """Validate a dataset."""
    logger.info(f"Validating dataset: {args.dataset}")
    
    # Create dataset based on name
    if args.dataset == "bigearthnet":
        dataset = BigEarthNetDataset(
            root=args.root,
            subset_size=args.limit,
            enabled=True
        )
    elif args.dataset == "vrsbench":
        dataset = VRSBenchDataset(
            root=args.root,
            subset_size=args.limit,
            enabled=True
        )
    elif args.dataset == "rsvqa":
        dataset = RSVQADataset(
            root=args.root,
            subset_size=args.limit,
            enabled=True
        )
    elif args.dataset == "cdvqa":
        dataset = CDVQADataset(
            root=args.root,
            subset_size=args.limit,
            enabled=True
        )
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        return 1
    
    # Load dataset
    dataset.load()
    
    if not dataset.is_loaded():
        logger.error("Failed to load dataset")
        return 1
    
    # Validate paths
    result = dataset.validate_paths()
    
    logger.info(f"Validation Results:")
    logger.info(f"  Total samples: {result.total_samples}")
    logger.info(f"  Valid samples: {result.valid_samples}")
    logger.info(f"  Invalid samples: {result.invalid_samples}")
    
    if result.errors:
        logger.error(f"  Errors ({len(result.errors)}):")
        for error in result.errors[:10]:  # Show first 10 errors
            logger.error(f"    - {error}")
        if len(result.errors) > 10:
            logger.error(f"    ... and {len(result.errors) - 10} more errors")
    
    if result.warnings:
        logger.warning(f"  Warnings ({len(result.warnings)}):")
        for warning in result.warnings[:10]:
            logger.warning(f"    - {warning}")
        if len(result.warnings) > 10:
            logger.warning(f"    ... and {len(result.warnings) - 10} more warnings")
    
    return 0 if result.valid else 1


def prepare_dataset(args):
    """Prepare a dataset (generate manifest)."""
    logger.info(f"Preparing dataset: {args.dataset}")
    
    # Create dataset
    if args.dataset == "bigearthnet":
        dataset = BigEarthNetDataset(
            root=args.root,
            subset_size=args.limit,
            enabled=True
        )
    elif args.dataset == "vrsbench":
        dataset = VRSBenchDataset(
            root=args.root,
            subset_size=args.limit,
            enabled=True
        )
    elif args.dataset == "rsvqa":
        dataset = RSVQADataset(
            root=args.root,
            subset_size=args.limit,
            enabled=True
        )
    elif args.dataset == "cdvqa":
        dataset = CDVQADataset(
            root=args.root,
            subset_size=args.limit,
            enabled=True
        )
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        return 1
    
    # Load dataset
    dataset.load()
    
    if not dataset.is_loaded():
        logger.error("Failed to load dataset")
        return 1
    
    # Create manifest
    manifest = DatasetManifest(manifest_path=args.output)
    manifest.add_samples(dataset._samples)
    manifest.save()
    
    # Print statistics
    stats = manifest.get_statistics()
    logger.info(f"Manifest Statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Datasets: {stats['datasets']}")
    logger.info(f"  Samples per dataset: {stats['samples_per_dataset']}")
    logger.info(f"  Samples per modality: {stats['samples_per_modality']}")
    logger.info(f"  Bi-temporal pairs: {stats['bi_temporal_count']}")
    logger.info(f"  Optical-SAR pairs: {stats['optical_sar_count']}")
    
    return 0


def inspect_sample(args):
    """Inspect a specific sample."""
    logger.info(f"Inspecting sample: {args.sample_id}")
    
    # Load manifest
    manifest = DatasetManifest(manifest_path=args.manifest)
    
    sample = manifest.get_sample(args.sample_id)
    
    if not sample:
        logger.error(f"Sample not found: {args.sample_id}")
        return 1
    
    logger.info(f"Sample Information:")
    for key, value in sample.items():
        logger.info(f"  {key}: {value}")
    
    return 0


def test_compatibility(args):
    """Test pair compatibility."""
    logger.info("Testing pair compatibility")
    
    checker = PairCompatibilityChecker()
    
    if args.type == "temporal":
        result = checker.check_temporal_pair(args.image1, args.image2)
    elif args.type == "optical_sar":
        result = checker.check_optical_sar_pair(args.image1, args.image2)
    else:
        logger.error(f"Unknown compatibility type: {args.type}")
        return 1
    
    logger.info(f"Compatibility Result: {result.compatible}")
    
    if result.errors:
        logger.error(f"Errors:")
        for error in result.errors:
            logger.error(f"  - {error}")
    
    if result.warnings:
        logger.warning(f"Warnings:")
        for warning in result.warnings:
            logger.warning(f"  - {warning}")
    
    logger.info(f"Checks:")
    for check_name, check_value in result.checks.items():
        logger.info(f"  {check_name}: {check_value}")
    
    return 0 if result.compatible else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="SatQueryAI Data Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a dataset")
    validate_parser.add_argument("--dataset", required=True, help="Dataset name (bigearthnet, vrsbench, rsvqa, cdvqa)")
    validate_parser.add_argument("--root", required=True, help="Dataset root directory")
    validate_parser.add_argument("--limit", type=int, help="Limit number of samples")
    
    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare a dataset (generate manifest)")
    prepare_parser.add_argument("--dataset", required=True, help="Dataset name")
    prepare_parser.add_argument("--root", required=True, help="Dataset root directory")
    prepare_parser.add_argument("--limit", type=int, help="Limit number of samples")
    prepare_parser.add_argument("--output", default="data_manifest.json", help="Output manifest path")
    
    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a specific sample")
    inspect_parser.add_argument("--sample-id", required=True, help="Sample ID")
    inspect_parser.add_argument("--manifest", default="data_manifest.json", help="Manifest path")
    
    # Compatibility command
    compat_parser = subparsers.add_parser("compatibility", help="Test pair compatibility")
    compat_parser.add_argument("--type", required=True, choices=["temporal", "optical_sar"], help="Compatibility type")
    compat_parser.add_argument("--image1", required=True, help="First image path")
    compat_parser.add_argument("--image2", required=True, help="Second image path")
    
    args = parser.parse_args()
    
    if args.command == "validate":
        return validate_dataset(args)
    elif args.command == "prepare":
        return prepare_dataset(args)
    elif args.command == "inspect":
        return inspect_sample(args)
    elif args.command == "compatibility":
        return test_compatibility(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
