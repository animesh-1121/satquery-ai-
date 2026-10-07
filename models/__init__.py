"""
SatQueryAI Model Layer

This package contains machine learning models and inference interfaces
for remote sensing AI tasks.
"""

__version__ = "0.1.0"

from .base import ModelType, RemoteSensingModel, VQAModel, ChangeDetectionModel
from .vqa.multimodal_vqa import MultimodalVQA, VQAConfig, create_multimodal_vqa
from .change_detection import SiameseChangeDetection, ChangeDetectionConfig, create_change_detection
from .optical_sar.optical_sar_fusion import OpticalSARFusion, OpticalSARConfig, create_optical_sar_fusion

__all__ = [
    "ModelType",
    "RemoteSensingModel",
    "VQAModel",
    "ChangeDetectionModel",
    "MultimodalVQA",
    "VQAConfig",
    "create_multimodal_vqa",
    "SiameseChangeDetection",
    "ChangeDetectionConfig",
    "create_change_detection",
    "OpticalSARFusion",
    "OpticalSARConfig",
    "create_optical_sar_fusion"
]
