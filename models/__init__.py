"""
SatQueryAI Model Layer

This package contains machine learning models and inference interfaces
for remote sensing AI tasks.
"""

__version__ = "0.1.0"

from .vqa.multimodal_vqa import MultimodalVQA, VQAConfig, create_multimodal_vqa

__all__ = [
    "MultimodalVQA",
    "VQAConfig",
    "create_multimodal_vqa"
]
