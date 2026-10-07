"""
Optical-SAR Cross-Modal Fusion Model for SatQueryAI Phase 5

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

Key Design Decisions:
- Reuse existing RemoteCLIP ViT-B/32 for optical encoder (frozen)
- Lightweight CNN for SAR encoder (frozen, no pretrained weights available)
- Cross-modal attention for genuine fusion (not simple concatenation)
- Lightweight architecture for RTX 2050 4GB VRAM
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from models.base import RemoteSensingModel, ModelType

logger = logging.getLogger(__name__)


@dataclass
class OpticalSARConfig:
    """Configuration for optical-SAR fusion model."""
    # Model
    remoteclip_model: str = "ViT-B-32"
    optical_feature_dim: int = 512  # RemoteCLIP embedding dimension
    sar_feature_dim: int = 256  # SAR encoder output dimension
    fusion_dim: int = 256  # Common fusion dimension
    num_classes: int = 10  # Number of classification classes
    num_attention_heads: int = 4  # Number of attention heads
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 1  # Small for 4GB VRAM
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Hardware
    device: str = "cuda"
    freeze_optical_encoder: bool = True
    freeze_sar_encoder: bool = True
    mixed_precision: bool = False  # Disabled for compatibility


class LightweightSAREncoder(nn.Module):
    """
    Lightweight CNN encoder for SAR imagery.
    
    Since SSL4EO-S12 is not available and would be too heavy for RTX 2050 4GB,
    we implement a lightweight CNN suitable for SAR feature extraction.
    
    Architecture:
        Input (3 channels for simplicity, typically SAR is 1-2 channels)
        → Conv Block 1
        → Conv Block 2
        → Conv Block 3
        → Global Average Pooling
        → FC Layer
        → SAR Features
    """
    
    def __init__(self, input_channels: int = 3, output_dim: int = 256):
        super().__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SAR encoder.
        
        Args:
            x: [batch, channels, height, width]
            
        Returns:
            sar_features: [batch, output_dim]
        """
        x = self.conv1(x)  # [batch, 64, H/2, W/2]
        x = self.conv2(x)  # [batch, 128, H/4, W/4]
        x = self.conv3(x)  # [batch, 256, H/8, W/8]
        x = self.global_pool(x)  # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 256]
        x = self.fc(x)  # [batch, output_dim]
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention module for optical-SAR fusion.
    
    Implements bidirectional cross-attention:
    1. Optical → Query, SAR → Key/Value
    2. SAR → Query, Optical → Key/Value
    3. Combine both directions
    
    This ensures both modalities influence the fused representation.
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        # Multi-head attention
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, optical_features: torch.Tensor, sar_features: torch.Tensor) -> torch.Tensor:
        """
        Cross-modal attention forward pass.
        
        Args:
            optical_features: [batch, feature_dim]
            sar_features: [batch, feature_dim]
            
        Returns:
            fused_features: [batch, feature_dim]
        """
        # Add sequence dimension for attention
        optical_seq = optical_features.unsqueeze(1)  # [batch, 1, feature_dim]
        sar_seq = sar_features.unsqueeze(1)  # [batch, 1, feature_dim]
        
        # Direction 1: Optical as query, SAR as key/value
        attn_out_1, _ = self.cross_attn_1(
            query=optical_seq,
            key=sar_seq,
            value=sar_seq
        )
        attn_out_1 = self.norm1(attn_out_1 + optical_seq)  # Residual + norm
        attn_out_1 = attn_out_1.squeeze(1)  # [batch, feature_dim]
        
        # Direction 2: SAR as query, Optical as key/value
        attn_out_2, _ = self.cross_attn_2(
            query=sar_seq,
            key=optical_seq,
            value=optical_seq
        )
        attn_out_2 = self.norm2(attn_out_2 + sar_seq)  # Residual + norm
        attn_out_2 = attn_out_2.squeeze(1)  # [batch, feature_dim]
        
        # Combine both directions
        combined = torch.cat([attn_out_1, attn_out_2], dim=-1)  # [batch, feature_dim * 2]
        fused = self.fusion(combined)  # [batch, feature_dim]
        
        return fused


class OpticalSARFusion(RemoteSensingModel):
    """
    Optical-SAR cross-modal fusion model.
    
    This model uses separate encoders for optical and SAR imagery,
    then fuses them using cross-modal attention.
    """
    
    def __init__(
        self,
        config: OpticalSARConfig,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize optical-SAR fusion model.
        
        Args:
            config: Optical-SAR configuration
            checkpoint_path: Path to trained checkpoint
        """
        super().__init__(
            model_name=f"RemoteCLIP-{config.remoteclip_model}-SARFusion",
            device=config.device
        )
        
        self.config = config
        self.num_classes = config.num_classes
        
        # Model components
        self.remoteclip_model = None
        self.remoteclip_preprocess = None
        self.sar_encoder = None
        self.optical_projection = None
        self.sar_projection = None
        self.cross_attention = None
        self.fusion_head = None
        
        self._loaded = False
        self.checkpoint_path = checkpoint_path
    
    def load(self) -> None:
        """Load RemoteCLIP and initialize optical-SAR components."""
        if not OPENCLIP_AVAILABLE:
            raise ImportError("open-clip-torch is required")
        
        logger.info(f"Loading RemoteCLIP-{self.config.remoteclip_model} for optical encoder...")
        
        # Load RemoteCLIP model (reused from Phase 3/4)
        self.remoteclip_model, self.remoteclip_preprocess, _ = open_clip.create_model_and_transforms(
            self.config.remoteclip_model,
            pretrained="openai"
        )
        
        # Get optical embedding dimensions
        try:
            self.optical_dim = self.remoteclip_model.visual.output_dim
        except AttributeError:
            self.optical_dim = self.remoteclip_model.visual.proj.shape[0]
        
        # Load RemoteCLIP checkpoint if available
        if self.config.freeze_optical_encoder:
            if HF_HUB_AVAILABLE:
                try:
                    checkpoint_path = hf_hub_download(
                        repo_id="chendelong/RemoteCLIP",
                        filename=f"RemoteCLIP-{self.config.remoteclip_model}.pt"
                    )
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    self.remoteclip_model.load_state_dict(checkpoint)
                    logger.info("RemoteCLIP weights loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load RemoteCLIP checkpoint: {e}")
        
        # Freeze RemoteCLIP if configured
        if self.config.freeze_optical_encoder:
            logger.info("Freezing RemoteCLIP optical encoder parameters...")
            for param in self.remoteclip_model.parameters():
                param.requires_grad = False
        
        # Initialize SAR encoder (lightweight CNN, no pretrained weights)
        logger.info("Initializing lightweight SAR encoder...")
        self.sar_encoder = LightweightSAREncoder(
            input_channels=3,  # Assuming 3-channel SAR for simplicity
            output_dim=self.config.sar_feature_dim
        ).to(self.config.device)
        
        # Freeze SAR encoder if configured
        if self.config.freeze_sar_encoder:
            logger.info("Freezing SAR encoder parameters...")
            for param in self.sar_encoder.parameters():
                param.requires_grad = False
        
        # Initialize projections to common fusion dimension
        self.optical_projection = nn.Sequential(
            nn.Linear(self.optical_dim, self.config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(self.config.device)
        
        self.sar_projection = nn.Sequential(
            nn.Linear(self.config.sar_feature_dim, self.config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(self.config.device)
        
        # Initialize cross-modal attention
        self.cross_attention = CrossModalAttention(
            feature_dim=self.config.fusion_dim,
            num_heads=self.config.num_attention_heads
        ).to(self.config.device)
        
        # Initialize fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(self.config.fusion_dim, self.config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.fusion_dim, self.num_classes)
        ).to(self.config.device)
        
        # Load checkpoint if available
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            self._load_checkpoint(self.checkpoint_path)
        
        # Move RemoteCLIP to device
        self.remoteclip_model = self.remoteclip_model.to(self.config.device).eval()
        
        self._loaded = True
        logger.info(f"Optical-SAR fusion model loaded successfully")
        
        # Report parameters
        optical_params = sum(p.numel() for p in self.remoteclip_model.parameters())
        sar_params = sum(p.numel() for p in self.sar_encoder.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_params())
        logger.info(f"Parameters: Optical ({optical_params:,}, frozen), SAR ({sar_params:,}, frozen), Trainable ({trainable_params:,})")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load optical-SAR checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        if 'sar_encoder' in checkpoint:
            self.sar_encoder.load_state_dict(checkpoint['sar_encoder'])
        if 'optical_projection' in checkpoint:
            self.optical_projection.load_state_dict(checkpoint['optical_projection'])
        if 'sar_projection' in checkpoint:
            self.sar_projection.load_state_dict(checkpoint['sar_projection'])
        if 'cross_attention' in checkpoint:
            self.cross_attention.load_state_dict(checkpoint['cross_attention'])
        if 'fusion_head' in checkpoint:
            self.fusion_head.load_state_dict(checkpoint['fusion_head'])
        
        logger.info(f"Optical-SAR checkpoint loaded from {checkpoint_path}")
    
    def _encode_optical(self, image: Image.Image) -> torch.Tensor:
        """Encode optical image using RemoteCLIP."""
        image_input = self.remoteclip_preprocess(image).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            optical_features = self.remoteclip_model.encode_image(image_input)
            # Normalize features
            optical_features = optical_features / optical_features.norm(dim=-1, keepdim=True)
        
        return optical_features  # [1, optical_dim]
    
    def _encode_sar(self, sar_image: torch.Tensor) -> torch.Tensor:
        """Encode SAR image using SAR encoder."""
        # sar_image is already a tensor [batch, channels, height, width]
        sar_image = sar_image.to(self.config.device)
        
        with torch.no_grad():
            sar_features = self.sar_encoder(sar_image)
        
        return sar_features  # [batch, sar_feature_dim]
    
    def predict(
        self,
        optical_path: str,
        sar_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run optical-SAR fusion inference.
        
        Args:
            optical_path: Path to optical image
            sar_path: Path to SAR image
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with prediction and metadata
        """
        try:
            if not self._loaded:
                self.load()
            
            # Validate inputs
            if not os.path.exists(optical_path):
                raise FileNotFoundError(f"Optical image not found: {optical_path}")
            if not os.path.exists(sar_path):
                raise FileNotFoundError(f"SAR image not found: {sar_path}")
            
            start_time = time.time()
            
            # Load optical image
            try:
                optical_image = Image.open(optical_path)
                if optical_image.mode != "RGB":
                    optical_image = optical_image.convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to load optical image: {e}")
            
            # Load SAR image (as tensor)
            try:
                sar_image = Image.open(sar_path)
                if sar_image.mode != "RGB":
                    sar_image = sar_image.convert("RGB")
                # Convert to tensor and normalize
                from torchvision import transforms
                sar_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                sar_tensor = sar_transform(sar_image).unsqueeze(0)  # [1, 3, 224, 224]
            except Exception as e:
                raise ValueError(f"Failed to load SAR image: {e}")
            
            # Encode optical image
            optical_features = self._encode_optical(optical_image)  # [1, optical_dim]
            
            # Encode SAR image
            sar_features = self._encode_sar(sar_tensor)  # [1, sar_feature_dim]
            
            # Project to common dimension
            optical_proj = self.optical_projection(optical_features)  # [1, fusion_dim]
            sar_proj = self.sar_projection(sar_features)  # [1, fusion_dim]
            
            # Cross-modal attention fusion
            fused_features = self.cross_attention(optical_proj, sar_proj)  # [1, fusion_dim]
            
            # Classification
            with torch.no_grad():
                logits = self.fusion_head(fused_features)  # [1, num_classes]
                probabilities = F.softmax(logits, dim=-1)  # [1, num_classes]
            
            # Get prediction
            prob_cpu = probabilities.cpu().squeeze(0)
            top_idx = prob_cpu.argmax().item()
            confidence = prob_cpu[top_idx].item()
            predicted_class = str(top_idx)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Get memory stats
            memory_stats = {}
            if torch.cuda.is_available():
                memory_stats["gpu_allocated_mb"] = round(
                    torch.cuda.memory_allocated(self.config.device) / (1024 * 1024), 1
                )
                memory_stats["gpu_reserved_mb"] = round(
                    torch.cuda.memory_reserved(self.config.device) / (1024 * 1024), 1
                )
            
            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "model": self.model_name,
                "task": "optical_sar_fusion",
                "optical_input": optical_path,
                "sar_input": sar_path,
                "execution_time_ms": round(inference_time, 2),
                "device": self.config.device,
                "top_k_classes": [
                    {"class": str(idx), "confidence": round(prob_cpu[idx].item(), 4)}
                    for idx in prob_cpu.topk(5).indices.tolist()
                ],
                "memory": memory_stats,
                "status": "success"
            }
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return {
                "prediction": None,
                "confidence": 0.0,
                "model": self.model_name,
                "task": "optical_sar_fusion",
                "optical_input": optical_path,
                "sar_input": sar_path,
                "execution_time_ms": 0.0,
                "device": self.config.device,
                "error": str(e),
                "status": "error"
            }
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            return {
                "prediction": None,
                "confidence": 0.0,
                "model": self.model_name,
                "task": "optical_sar_fusion",
                "optical_input": optical_path,
                "sar_input": sar_path,
                "execution_time_ms": 0.0,
                "device": self.config.device,
                "error": str(e),
                "status": "error"
            }
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "prediction": None,
                "confidence": 0.0,
                "model": self.model_name,
                "task": "optical_sar_fusion",
                "optical_input": optical_path,
                "sar_input": sar_path,
                "execution_time_ms": 0.0,
                "device": self.config.device,
                "error": str(e),
                "status": "error"
            }
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.VQA  # Using VQA type for compatibility
    
    def get_trainable_params(self):
        """Get trainable parameters (projections, attention, fusion head only)."""
        return list(self.optical_projection.parameters()) + \
               list(self.sar_projection.parameters()) + \
               list(self.cross_attention.parameters()) + \
               list(self.fusion_head.parameters())
    
    def save_checkpoint(self, save_path: str, epoch: int, metrics: Dict[str, float], training_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save optical-SAR checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'sar_encoder': self.sar_encoder.state_dict(),
            'optical_projection': self.optical_projection.state_dict(),
            'sar_projection': self.sar_projection.state_dict(),
            'cross_attention': self.cross_attention.state_dict(),
            'fusion_head': self.fusion_head.state_dict(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
            'num_classes': self.num_classes,
            'metrics': metrics,
            'training_metadata': training_metadata or {}
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Optical-SAR checkpoint saved to {save_path}")


def create_optical_sar_fusion(
    config: Optional[OpticalSARConfig] = None,
    checkpoint_path: Optional[str] = None
) -> OpticalSARFusion:
    """
    Factory function to create optical-SAR fusion model.
    
    Args:
        config: Optical-SAR configuration (uses defaults if None)
        checkpoint_path: Path to trained checkpoint
        
    Returns:
        OpticalSARFusion: Initialized model
    """
    if config is None:
        config = OpticalSARConfig()
    
    model = OpticalSARFusion(
        config=config,
        checkpoint_path=checkpoint_path
    )
    model.load()
    return model
