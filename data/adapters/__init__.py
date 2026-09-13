"""
Dataset Adapters for SatQueryAI

This package contains dataset-specific adapters for various remote sensing datasets.
"""

from .bigearthnet import BigEarthNetDataset
from .vrsbench import VRSBenchDataset
from .rsvqa import RSVQADataset
from .cdvqa import CDVQADataset

__all__ = [
    "BigEarthNetDataset",
    "VRSBenchDataset",
    "RSVQADataset",
    "CDVQADataset",
]
