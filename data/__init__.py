"""
SatQueryAI Data Package

This package provides a modular, extensible data pipeline for remote sensing datasets.
It supports multiple dataset types, geo-spatial preprocessing, and compatibility checking
for bi-temporal and optical-SAR pairs.

Architecture:
    BaseRemoteSensingDataset (abstract interface)
        ├── BigEarthNetDataset
        ├── VRSBenchDataset
        ├── RSVQADataset
        └── CDVQADataset

    Preprocessing Pipeline:
        Raw Satellite Data → Format Detection → Metadata Inspection → 
        Band Validation → Spatial Validation → Normalization → 
        Resize/Crop/Tile → Tensor Conversion → Model Input

    Compatibility Checking:
        - Bi-temporal pair compatibility
        - Optical-SAR pair compatibility
        - Spatial alignment verification
"""

from .base import BaseRemoteSensingDataset, Sample, Modality
from .preprocessing import ImagePreprocessor, PreprocessingConfig
from .validation import DataValidator, ValidationResult
from .compatibility import PairCompatibilityChecker, CompatibilityResult
from .manifest import DatasetManifest
from .splits import DataSplitter, create_train_val_test_split
from .remoteclip_preprocessing import RemoteCLIPPreprocessor, prepare_image_for_remoteclip
from .adapters import BigEarthNetDataset, VRSBenchDataset, RSVQADataset, CDVQADataset
from .vqa_dataset import VQADataset, SyntheticVQADataset, create_vqa_dataloader, collate_fn
from .rsvqa_parser import RSVQAAnnotationParser, create_rsvqa_parser
from .real_vqa_dataset import RealVQADataset, create_real_vqa_dataloader
from .change_detection_dataset import ChangeDetectionDataset, ChangeDetectionSample, create_change_detection_dataloader
from .optical_sar_dataset import OpticalSARDataset, OpticalSARSample, create_optical_sar_dataloader

__all__ = [
    "BaseRemoteSensingDataset",
    "Sample",
    "Modality",
    "ImagePreprocessor",
    "PreprocessingConfig",
    "DataValidator",
    "ValidationResult",
    "PairCompatibilityChecker",
    "CompatibilityResult",
    "DatasetManifest",
    "DataSplitter",
    "create_train_val_test_split",
    "RemoteCLIPPreprocessor",
    "prepare_image_for_remoteclip",
    "BigEarthNetDataset",
    "VRSBenchDataset",
    "RSVQADataset",
    "CDVQADataset",
    "VQADataset",
    "SyntheticVQADataset",
    "create_vqa_dataloader",
    "collate_fn",
    "RSVQAAnnotationParser",
    "create_rsvqa_parser",
    "RealVQADataset",
    "create_real_vqa_dataloader",
    "ChangeDetectionDataset",
    "ChangeDetectionSample",
    "create_change_detection_dataloader",
    "OpticalSARDataset",
    "OpticalSARSample",
    "create_optical_sar_dataloader",
]
