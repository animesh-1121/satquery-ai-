"""
Change Detection Training Module for SatQueryAI Phase 4

This module provides training infrastructure for the Siamese change detection model.
"""

from .change_detection_trainer import ChangeDetectionTrainer, ChangeDetectionTrainingConfig

__all__ = [
    'ChangeDetectionTrainer',
    'ChangeDetectionTrainingConfig'
]
