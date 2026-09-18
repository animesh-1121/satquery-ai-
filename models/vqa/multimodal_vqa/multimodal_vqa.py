"""
Multimodal VQA Model for SatQueryAI Phase 3

This module implements a true learned single-image VQA system using RemoteCLIP
as the remote-sensing visual encoder, with proper multimodal fusion between
visual and textual features.

Architecture:
    Satellite Image + Question
           ↓
    RemoteCLIP ViT-B/32 (visual + text encoders)
           ↓
    Visual embedding + Text embedding
           ↓
    Multimodal fusion (element-wise + attention)
           ↓
    VQA classification head
           ↓
    Answer + confidence

Key Design Decisions:
- Use RemoteCLIP's existing text encoder (no additional LLM needed)
- Lightweight fusion for 4GB VRAM compatibility
- Answer classification for efficiency and reliability
- RemoteCLIP base weights frozen
- Only fusion and VQA head are trained
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Any, List
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

from ...base import VQAModel, ModelType

logger = logging.getLogger(__name__)


@dataclass
class VQAConfig:
    """Configuration for multimodal VQA model."""
    # Model
    remoteclip_model: str = "ViT-B-32"
    embedding_dim: int = 512  # RemoteCLIP embedding dimension
    fusion_dim: int = 256  # Fusion layer dimension
    num_answers: int = 100  # Number of possible answers (will be set from dataset)
    
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
    
    # Dataset
    max_question_length: int = 77  # RemoteCLIP max text length


class MultimodalFusion(nn.Module):
    """
    Lightweight multimodal fusion module.
    
    Combines visual and text embeddings using:
    1. Element-wise multiplication
    2. Learned projection
    3. Optional attention mechanism
    """
    
    def __init__(self, visual_dim: int, text_dim: int, fusion_dim: int, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        
        # Project visual and text to common dimension
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        
        # Fusion layers
        if use_attention:
            # Cross-modal attention
            self.attention = nn.MultiheadAttention(fusion_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(fusion_dim)
        else:
            # Simple element-wise fusion
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim)
            )
    
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse visual and text features.
        
        Args:
            visual_features: [batch, visual_dim]
            text_features: [batch, text_dim]
            
        Returns:
            fused_features: [batch, fusion_dim]
        """
        # Project to common dimension
        visual_proj = self.visual_proj(visual_features)  # [batch, fusion_dim]
        text_proj = self.text_proj(text_features)  # [batch, fusion_dim]
        
        if self.use_attention:
            # Cross-modal attention
            visual_proj = visual_proj.unsqueeze(1)  # [batch, 1, fusion_dim]
            text_proj = text_proj.unsqueeze(1)  # [batch, 1, fusion_dim]
            
            # Visual query text key
            fused, _ = self.attention(visual_proj, text_proj, text_proj)
            fused = self.norm(fused.squeeze(1))  # [batch, fusion_dim]
        else:
            # Element-wise multiplication + concatenation
            element_wise = visual_proj * text_proj  # [batch, fusion_dim]
            concat = torch.cat([visual_proj, text_proj], dim=-1)  # [batch, fusion_dim * 2]
            fused = self.fusion(concat)  # [batch, fusion_dim]
        
        return fused


class VQAHead(nn.Module):
    """
    VQA classification head.
    
    Takes fused multimodal features and predicts the answer.
    """
    
    def __init__(self, fusion_dim: int, num_answers: int, hidden_dim: int = 512):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_answers)
        )
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VQA head.
        
        Args:
            fused_features: [batch, fusion_dim]
            
        Returns:
            logits: [batch, num_answers]
        """
        return self.classifier(fused_features)


class MultimodalVQA(VQAModel):
    """
    Multimodal VQA model for remote sensing images.
    
    This model implements true learned VQA with proper multimodal fusion
    between RemoteCLIP visual features and text features.
    """
    
    def __init__(
        self,
        config: VQAConfig,
        answer_vocab: List[str],
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize multimodal VQA model.
        
        Args:
            config: VQA configuration
            answer_vocab: List of possible answers
            checkpoint_path: Path to trained VQA checkpoint
        """
        super().__init__(
            model_name=f"RemoteCLIP-{config.remoteclip_model}-MultimodalVQA",
            device=config.device
        )
        
        self.config = config
        self.answer_vocab = answer_vocab
        self.num_answers = len(answer_vocab)
        self.answer_to_idx = {ans: idx for idx, ans in enumerate(answer_vocab)}
        self.idx_to_answer = {idx: ans for idx, ans in enumerate(answer_vocab)}
        
        # Model components
        self.remoteclip_model = None
        self.remoteclip_preprocess = None
        self.remoteclip_tokenizer = None
        self.fusion = None
        self.vqa_head = None
        
        self._loaded = False
        self.checkpoint_path = checkpoint_path
    
    def load(self) -> None:
        """Load RemoteCLIP and initialize VQA components."""
        if not OPENCLIP_AVAILABLE:
            raise ImportError("open-clip-torch is required")
        
        logger.info(f"Loading RemoteCLIP-{self.config.remoteclip_model}...")
        
        # Load RemoteCLIP model
        self.remoteclip_model, self.remoteclip_preprocess, _ = open_clip.create_model_and_transforms(
            self.config.remoteclip_model,
            pretrained="openai"
        )
        self.remoteclip_tokenizer = open_clip.get_tokenizer(self.config.remoteclip_model)
        
        # Get embedding dimensions (handle different open_clip versions)
        try:
            self.visual_dim = self.remoteclip_model.visual.output_dim
        except AttributeError:
            # Fallback for newer open_clip versions
            self.visual_dim = self.remoteclip_model.visual.proj.shape[0]
        
        try:
            self.text_dim = self.remoteclip_model.text_projection.output_dim
        except AttributeError:
            # Fallback for newer open_clip versions
            self.text_dim = self.remoteclip_model.text_projection.shape[0]
        
        # Load RemoteCLIP checkpoint if available
        if self.config.freeze_remoteclip:
            # Download checkpoint if needed
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
        if self.config.freeze_remoteclip:
            logger.info("Freezing RemoteCLIP parameters...")
            for param in self.remoteclip_model.parameters():
                param.requires_grad = False
        
        # Initialize fusion module
        self.fusion = MultimodalFusion(
            visual_dim=self.visual_dim,
            text_dim=self.text_dim,
            fusion_dim=self.config.fusion_dim,
            use_attention=False  # Disabled for memory efficiency
        ).to(self.config.device)
        
        # Initialize VQA head
        self.vqa_head = VQAHead(
            fusion_dim=self.config.fusion_dim,
            num_answers=self.num_answers,
            hidden_dim=self.config.fusion_dim
        ).to(self.config.device)
        
        # Load VQA checkpoint if available
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            self._load_checkpoint(self.checkpoint_path)
        
        # Move RemoteCLIP to device
        self.remoteclip_model = self.remoteclip_model.to(self.config.device).eval()
        
        self._loaded = True
        logger.info(f"Multimodal VQA model loaded successfully")
        logger.info(f"Parameters: RemoteCLIP (frozen), Fusion ({sum(p.numel() for p in self.fusion.parameters())}), VQA Head ({sum(p.numel() for p in self.vqa_head.parameters())})")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load VQA checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        if 'fusion' in checkpoint:
            self.fusion.load_state_dict(checkpoint['fusion'])
        if 'vqa_head' in checkpoint:
            self.vqa_head.load_state_dict(checkpoint['vqa_head'])
        
        # Load answer vocabulary if available
        if 'answer_vocab' in checkpoint:
            self.answer_vocab = checkpoint['answer_vocab']
            self.answer_to_idx = {ans: idx for idx, ans in enumerate(self.answer_vocab)}
            self.idx_to_answer = {idx: ans for idx, ans in enumerate(self.answer_vocab)}
            self.num_answers = len(self.answer_vocab)
            # Update VQA head if vocabulary size changed
            if self.vqa_head.classifier[-1].out_features != self.num_answers:
                logger.warning(f"VQA head output size mismatch. Reinitializing for {self.num_answers} classes.")
                self.vqa_head = VQAHead(
                    fusion_dim=self.config.fusion_dim,
                    num_answers=self.num_answers,
                    hidden_dim=self.config.fusion_dim
                ).to(self.config.device)
                self.vqa_head.load_state_dict(checkpoint['vqa_head'])
        
        logger.info(f"VQA checkpoint loaded from {checkpoint_path}")
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image using RemoteCLIP visual encoder."""
        image_input = self.remoteclip_preprocess(image).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            visual_features = self.remoteclip_model.encode_image(image_input)
            # Normalize features
            visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        
        return visual_features  # [1, visual_dim]
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode question using RemoteCLIP text encoder."""
        text_tokens = self.remoteclip_tokenizer(text)
        text_input = text_tokens.to(self.config.device)
        
        with torch.no_grad():
            text_features = self.remoteclip_model.encode_text(text_input)
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features  # [1, text_dim]
    
    def predict(
        self,
        image_path: str,
        question: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run VQA inference.
        
        Args:
            image_path: Path to satellite image
            question: Natural language question
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with answer, confidence, and metadata
        """
        try:
            if not self._loaded:
                self.load()
            
            # Validate inputs
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            if not question or not question.strip():
                raise ValueError("Question cannot be empty")
            
            start_time = time.time()
            
            # Load and preprocess image
            try:
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to load image: {e}")
            
            # Encode image and question
            visual_features = self._encode_image(image)  # [1, visual_dim]
            text_features = self._encode_text(question)  # [1, text_dim]
            
            # Multimodal fusion
            fused_features = self.fusion(visual_features, text_features)  # [1, fusion_dim]
            
            # VQA prediction
            with torch.no_grad():
                logits = self.vqa_head(fused_features)  # [1, num_answers]
                probabilities = F.softmax(logits, dim=-1)  # [1, num_answers]
            
            # Get prediction
            prob_cpu = probabilities.cpu().squeeze(0)
            top_idx = prob_cpu.argmax().item()
            confidence = prob_cpu[top_idx].item()
            predicted_answer = self.idx_to_answer[top_idx]
            
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
                "answer": predicted_answer,
                "confidence": confidence,
                "model": self.model_name,
                "task": "single_image_vqa",
                "question": question,
                "execution_time_ms": round(inference_time, 2),
                "device": self.config.device,
                "top_k_answers": [
                    {"answer": self.idx_to_answer[idx], "confidence": round(prob_cpu[idx].item(), 4)}
                    for idx in prob_cpu.topk(5).indices.tolist()
                ],
                "memory": memory_stats,
                "status": "success"
            }
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return {
                "answer": None,
                "confidence": 0.0,
                "model": self.model_name,
                "task": "single_image_vqa",
                "question": question,
                "execution_time_ms": 0.0,
                "device": self.config.device,
                "error": str(e),
                "status": "error"
            }
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            return {
                "answer": None,
                "confidence": 0.0,
                "model": self.model_name,
                "task": "single_image_vqa",
                "question": question,
                "execution_time_ms": 0.0,
                "device": self.config.device,
                "error": str(e),
                "status": "error"
            }
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "answer": None,
                "confidence": 0.0,
                "model": self.model_name,
                "task": "single_image_vqa",
                "question": question,
                "execution_time_ms": 0.0,
                "device": self.config.device,
                "error": str(e),
                "status": "error"
            }
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.VQA
    
    def get_trainable_params(self):
        """Get trainable parameters (fusion + VQA head only)."""
        return list(self.fusion.parameters()) + list(self.vqa_head.parameters())
    
    def save_checkpoint(self, save_path: str, epoch: int, metrics: Dict[str, float], training_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save VQA checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'fusion': self.fusion.state_dict(),
            'vqa_head': self.vqa_head.state_dict(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
            'answer_vocab': self.answer_vocab,
            'num_answers': self.num_answers,
            'metrics': metrics,
            'training_metadata': training_metadata or {}
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")


def create_multimodal_vqa(
    answer_vocab: List[str],
    config: Optional[VQAConfig] = None,
    checkpoint_path: Optional[str] = None
) -> MultimodalVQA:
    """
    Factory function to create multimodal VQA model.
    
    Args:
        answer_vocab: List of possible answers
        config: VQA configuration (uses defaults if None)
        checkpoint_path: Path to trained checkpoint
        
    Returns:
        MultimodalVQA: Initialized model
    """
    if config is None:
        config = VQAConfig()
    
    model = MultimodalVQA(
        config=config,
        answer_vocab=answer_vocab,
        checkpoint_path=checkpoint_path
    )
    model.load()
    return model
