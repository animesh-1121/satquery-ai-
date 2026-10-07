"""
Optical-SAR Cross-Modal Fusion Module for SatQueryAI Phase 5

This module implements a genuine optical-SAR cross-modal fusion system using
separate modality encoders and cross-attention for fusion.

Architecture:
    Optical Image      SAR Image
          ↓               ↓
    RemoteCLIP CNN     SAR Encoder
          ↓               ↓
    Optical Features  SAR Features
          ↓               ↓
    Optical Proj      SAR Proj
          ↓               ↓
          └── Cross-Attention ───┘
                    ↓
              Fused Features
                    ↓
              Fusion Head
                    ↓
              Prediction
"""

from .optical_sar_fusion import OpticalSARFusion, OpticalSARConfig, create_optical_sar_fusion

__all__ = [
    'OpticalSARFusion',
    'OpticalSARConfig',
    'create_optical_sar_fusion'
]
