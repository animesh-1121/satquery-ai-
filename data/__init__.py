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
]
