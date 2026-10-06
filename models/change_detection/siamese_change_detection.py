"""
Siamese Change Detection Model for SatQueryAI Phase 4

This module implements a learned bi-temporal change detection system using a
Siamese architecture with shared RemoteCLIP encoder.

Architecture:
    T1 Image          T2 Image
       ↓                 ↓
    Shared RemoteCLIP ViT-B/32 (same weights)
       ↓                 ↓
    F1                  F2
       ↓                 ↓
    [F1, F2, |F1-F2|]  →  Change Detection Head  →  Change Map/Classification

Key Design Decisions:
- Single shared RemoteCLIP encoder for both temporal images
- Temporal features: [F1, F2, |F1-F2|] for change modeling
- Lightweight change head for 4GB VRAM compatibility
- RemoteCLIP base weights frozen
- Only change head is trained
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

from ..base import ChangeDetectionModel, ModelType

logger = logging.getLogger(__name__)


@dataclass
class ChangeDetectionConfig:
    """Configuration for change detection model."""
    # Model
    remoteclip_model: str = "ViT-B-32"
    embedding_dim: int = 512  # RemoteCLIP embedding dimension
    change_head_dim: int = 256  # Change head hidden dimension
    num_change_classes: int = 2  # Binary: no-change, change
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 1  # Small for 4GB VRAM
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Hardware
    device: str = "cuda"
    freeze_remoteclip: bool = True
    mixed_precision: bool = False  # Disabled for compatibility


class TemporalFeatureFusion(nn.Module):
    """
    Temporal feature fusion module.
    
    Combines features from T1 and T2 images:
    - F1: Features from T1
    - F2: Features from T2
    - |F1 - F2|: Absolute difference
    
    Output: [F1, F2, |F1-F2|] concatenated and projected
    """
    
    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        
        # Project concatenated temporal features
        self.temporal_proj = nn.Sequential(
            nn.Linear(feature_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Fuse temporal features.
        
        Args:
            f1: [batch, feature_dim] - T1 features
            f2: [batch, feature_dim] - T2 features
            
        Returns:
            temporal_features: [batch, output_dim]
        """
        # Compute absolute difference
        diff = torch.abs(f1 - f2)  # [batch, feature_dim]
        
        # Concatenate [F1, F2, |F1-F2|]
        concat = torch.cat([f1, f2, diff], dim=-1)  # [batch, feature_dim * 3]
        
        # Project to output dimension
        temporal_features = self.temporal_proj(concat)  # [batch, output_dim]
        
        return temporal_features


class ChangeDetectionHead(nn.Module):
    """
    Change detection head.
    
    Takes temporal features and predicts:
    - Binary change classification (no-change vs change)
    - Optional: dense change map (if spatial resolution allows)
    """
    
    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dim: int = 512):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through change detection head.
        
        Args:
            temporal_features: [batch, input_dim]
            
        Returns:
            logits: [batch, num_classes]
        """
        return self.classifier(temporal_features)


class SiameseChangeDetection(ChangeDetectionModel):
    """
    Siamese change detection model for bi-temporal remote sensing images.
    
    This model uses a shared RemoteCLIP encoder to extract features from both
    temporal images, then models their feature-level differences to detect changes.
    """
    
    def __init__(
        self,
        config: ChangeDetectionConfig,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize Siamese change detection model.
        
        Args:
            config: Change detection configuration
            checkpoint_path: Path to trained change detection checkpoint
        """
        super().__init__(
            model_name=f"RemoteCLIP-{config.remoteclip_model}-SiameseChangeDetection",
            device=config.device
        )
        
        self.config = config
        self.num_change_classes = config.num_change_classes
        
        # Model components
        self.remoteclip_model = None
        self.remoteclip_preprocess = None
        self.temporal_fusion = None
        self.change_head = None
        
        self._loaded = False
        self.checkpoint_path = checkpoint_path
    
    def load(self) -> None:
        """Load RemoteCLIP and initialize change detection components."""
        if not OPENCLIP_AVAILABLE:
            raise ImportError("open-clip-torch is required")
        
        logger.info(f"Loading RemoteCLIP-{self.config.remoteclip_model} for Siamese change detection...")
        
        # Load RemoteCLIP model (SHARED for both T1 and T2)
        self.remoteclip_model, self.remoteclip_preprocess, _ = open_clip.create_model_and_transforms(
            self.config.remoteclip_model,
            pretrained="openai"
        )
        
        # Get embedding dimensions
        try:
            self.visual_dim = self.remoteclip_model.visual.output_dim
        except AttributeError:
            self.visual_dim = self.remoteclip_model.visual.proj.shape[0]
        
        # Load RemoteCLIP checkpoint if available
        if self.config.freeze_remoteclip:
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
        
        # Freeze RemoteCLIP if configured (SHARED encoder)
        if self.config.freeze_remoteclip:
            logger.info("Freezing shared RemoteCLIP encoder parameters...")
            for param in self.remoteclip_model.parameters():
                param.requires_grad = False
        
        # Initialize temporal fusion module
        self.temporal_fusion = TemporalFeatureFusion(
            feature_dim=self.visual_dim,
            output_dim=self.config.change_head_dim
        ).to(self.config.device)
        
        # Initialize change detection head
        self.change_head = ChangeDetectionHead(
            input_dim=self.config.change_head_dim,
            num_classes=self.num_change_classes,
            hidden_dim=self.config.change_head_dim
        ).to(self.config.device)
        
        # Load change detection checkpoint if available
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            self._load_checkpoint(self.checkpoint_path)
        
        # Move RemoteCLIP to device
        self.remoteclip_model = self.remoteclip_model.to(self.config.device).eval()
        
        self._loaded = True
        logger.info(f"Siamese change detection model loaded successfully")
        logger.info(f"Parameters: RemoteCLIP (frozen, shared), Temporal Fusion ({sum(p.numel() for p in self.temporal_fusion.parameters())}), Change Head ({sum(p.numel() for p in self.change_head.parameters())})")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load change detection checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        if 'temporal_fusion' in checkpoint:
            self.temporal_fusion.load_state_dict(checkpoint['temporal_fusion'])
        if 'change_head' in checkpoint:
            self.change_head.load_state_dict(checkpoint['change_head'])
        
        logger.info(f"Change detection checkpoint loaded from {checkpoint_path}")
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image using shared RemoteCLIP visual encoder."""
        image_input = self.remoteclip_preprocess(image).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            visual_features = self.remoteclip_model.encode_image(image_input)
            # Normalize features
            visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        
        return visual_features  # [1, visual_dim]
    
    def predict(
        self,
        image_t1_path: str,
        image_t2_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run change detection inference.
        
        Args:
            image_t1_path: Path to T1 image (earlier time)
            image_t2_path: Path to T2 image (later time)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with change detection results and metadata
        """
        try:
            if not self._loaded:
                self.load()
            
            # Validate inputs
            if not os.path.exists(image_t1_path):
                raise FileNotFoundError(f"T1 image not found: {image_t1_path}")
            if not os.path.exists(image_t2_path):
                raise FileNotFoundError(f"T2 image not found: {image_t2_path}")
            
            start_time = time.time()
            
            # Load and preprocess images
            try:
                image_t1 = Image.open(image_t1_path)
                if image_t1.mode != "RGB":
                    image_t1 = image_t1.convert("RGB")
                
                image_t2 = Image.open(image_t2_path)
                if image_t2.mode != "RGB":
                    image_t2 = image_t2.convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to load images: {e}")
            
            # Validate image compatibility
            if image_t1.size != image_t2.size:
                logger.warning(f"Image size mismatch: T1={image_t1.size}, T2={image_t2.size}")
            
            # Encode both images using SHARED encoder
            f1 = self._encode_image(image_t1)  # [1, visual_dim]
            f2 = self._encode_image(image_t2)  # [1, visual_dim]
            
            # Temporal feature fusion
            temporal_features = self.temporal_fusion(f1, f2)  # [1, change_head_dim]
            
            # Change detection prediction
            with torch.no_grad():
                logits = self.change_head(temporal_features)  # [1, num_classes]
                probabilities = F.softmax(logits, dim=-1)  # [1, num_classes]
            
            # Get prediction
            prob_cpu = probabilities.cpu().squeeze(0)
            top_idx = prob_cpu.argmax().item()
            confidence = prob_cpu[top_idx].item()
            
            # Determine change detected
            change_detected = (top_idx == 1)  # Assuming class 1 = change
            change_label = "change" if change_detected else "no_change"
            
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
                "change_detected": change_detected,
                "change_label": change_label,
                "confidence": confidence,
                "class_probabilities": {
                    "no_change": round(prob_cpu[0].item(), 4),
                    "change": round(prob_cpu[1].item(), 4)
                },
                "model": self.model_name,
                "task": "bi_temporal_change_detection",
                "image_t1": image_t1_path,
                "image_t2": image_t2_path,
                "execution_time_ms": round(inference_time, 2),
                "device": self.config.device,
                "memory": memory_stats,
                "status": "success"
            }
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return {
                "change_detected": False,
                "change_label": "error",
                "confidence": 0.0,
                "model": self.model_name,
                "task": "bi_temporal_change_detection",
                "image_t1": image_t1_path,
                "image_t2": image_t2_path,
                "execution_time_ms": 0.0,
                "device": self.config.device,
                "error": str(e),
                "status": "error"
            }
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            return {
                "change_detected": False,
                "change_label": "error",
                "confidence": 0.0,
                "model": self.model_name,
                "task": "bi_temporal_change_detection",
                "image_t1": image_t1_path,
                "image_t2": image_t2_path,
                "execution_time_ms": 0.0,
                "device": self.config.device,
                "error": str(e),
                "status": "error"
            }
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "change_detected": False,
                "change_label": "error",
                "confidence": 0.0,
                "model": self.model_name,
                "task": "bi_temporal_change_detection",
                "image_t1": image_t1_path,
                "image_t2": image_t2_path,
                "execution_time_ms": 0.0,
                "device": self.config.device,
                "error": str(e),
                "status": "error"
            }
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.CHANGE_DETECTION
    
    def get_trainable_params(self):
        """Get trainable parameters (temporal fusion + change head only)."""
        return list(self.temporal_fusion.parameters()) + list(self.change_head.parameters())
    
    def save_checkpoint(self, save_path: str, epoch: int, metrics: Dict[str, float], training_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save change detection checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'temporal_fusion': self.temporal_fusion.state_dict(),
            'change_head': self.change_head.state_dict(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
            'num_change_classes': self.num_change_classes,
            'metrics': metrics,
            'training_metadata': training_metadata or {}
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Change detection checkpoint saved to {save_path}")


def create_change_detection(
    config: Optional[ChangeDetectionConfig] = None,
    checkpoint_path: Optional[str] = None
) -> SiameseChangeDetection:
    """
    Factory function to create Siamese change detection model.
    
    Args:
        config: Change detection configuration (uses defaults if None)
        checkpoint_path: Path to trained checkpoint
        
    Returns:
        SiameseChangeDetection: Initialized model
    """
    if config is None:
        config = ChangeDetectionConfig()
    
    model = SiameseChangeDetection(
        config=config,
        checkpoint_path=checkpoint_path
    )
    model.load()
    return model
