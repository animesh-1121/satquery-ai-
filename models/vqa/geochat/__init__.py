"""
GeoChat VQA Module for SatQueryAI

This module provides inference capabilities for remote sensing visual question answering
using the GeoChat model (CVPR 2024).
"""

from .inference import GeoChatInference, create_inference_engine

__all__ = ['GeoChatInference', 'create_inference_engine']
