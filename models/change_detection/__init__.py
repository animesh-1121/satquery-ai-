"""
Change Detection Module for SatQueryAI Phase 4

This module implements bi-temporal change detection using a Siamese architecture
with shared RemoteCLIP encoder.

Architecture:
    T1 Image          T2 Image
       ↓                 ↓
    Shared RemoteCLIP ViT-B/32 (same weights)
       ↓                 ↓
    F1                  F2
       ↓                 ↓
    [F1, F2, |F1-F2|]  →  Change Detection Head  →  Change Map/Classification
"""

from .siamese_change_detection import SiameseChangeDetection, ChangeDetectionConfig, create_change_detection

__all__ = [
    'SiameseChangeDetection',
    'ChangeDetectionConfig',
    'create_change_detection'
]
