"""
Training Pipeline for SatQueryAI Phase 3 VQA

This module implements the training pipeline for the multimodal VQA model,
including loss functions, optimizers, and training loops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import json
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.vqa.multimodal_vqa.multimodal_vqa import MultimodalVQA, VQAConfig
from data.vqa_dataset import VQADataset, SyntheticVQADataset, create_vqa_dataloader, collate_fn
from data.remoteclip_preprocessing import RemoteCLIPPreprocessor

logger = logging.getLogger(__name__)


class VQATrainer:
    """
    Trainer for multimodal VQA model.
    
    Handles training, validation, checkpointing, and evaluation.
    """
    
    def __init__(
        self,
        model: MultimodalVQA,
        config: VQAConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints/vqa"
    ):
        """
        Initialize VQA trainer.
        
        Args:
            model: MultimodalVQA model
            config: VQA configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.get_trainable_params(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_accuracy = 0.0
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.accumulated_steps = 0
        
        # Mixed precision (if enabled)
        self.scaler = None
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"VQA Trainer initialized with {len(train_loader)} training batches")
        if val_loader:
            logger.info(f"and {len(val_loader)} validation batches")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch["images"]
            questions = batch["questions"]
            answer_indices = batch["answer_indices"]
            
            # Preprocess images
            preprocessed_images = []
            for image in images:
                # Apply RemoteCLIP preprocessing
                import numpy as np
                img_array = np.array(image)
                if img_array.shape[2] == 3:
                    img_array = np.transpose(img_array, (2, 0, 1))
                img_tensor = torch.from_numpy(img_array).float() / 255.0
                if img_tensor.shape[0] == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                preprocessed_images.append(img_tensor)
            
            # Stack images
            if len(preprocessed_images) > 0:
                image_batch = torch.cat(preprocessed_images, dim=0)
            else:
                continue
            
            # Move to device
            image_batch = image_batch.to(self.config.device)
            answer_indices = answer_indices.to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Encode image and text
            with torch.set_grad_enabled(True):
                # Use model's internal encoding
                visual_features = self.model.remoteclip_model.encode_image(image_batch)
                visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
                
                # Encode questions
                text_features_list = []
                for question in questions:
                    text_tokens = self.model.remoteclip_tokenizer(question)
                    text_input = text_tokens.to(self.config.device)
                    text_feat = self.model.remoteclip_model.encode_text(text_input)
                    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                    text_features_list.append(text_feat)
                
                text_features = torch.cat(text_features_list, dim=0)
                
                # Fusion
                fused_features = self.model.fusion(visual_features, text_features)
                
                # VQA head
                logits = self.model.vqa_head(fused_features)
                
                # Loss
                loss = self.criterion(logits, answer_indices)
                
                # Gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                self.accumulated_steps += 1
            
            # Update weights
            if self.accumulated_steps >= self.gradient_accumulation_steps:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_params(),
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.accumulated_steps = 0
                self.global_step += 1
            
            # Metrics
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == answer_indices).sum().item()
                total += answer_indices.size(0)
                total_loss += loss.item() * answer_indices.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch["images"]
                questions = batch["questions"]
                answer_indices = batch["answer_indices"]
                
                # Preprocess images (simplified for validation)
                preprocessed_images = []
                for image in images:
                    import numpy as np
                    img_array = np.array(image)
                    if img_array.shape[2] == 3:
                        img_array = np.transpose(img_array, (2, 0, 1))
                    img_tensor = torch.from_numpy(img_array).float() / 255.0
                    if img_tensor.shape[0] == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    preprocessed_images.append(img_tensor)
                
                if len(preprocessed_images) > 0:
                    image_batch = torch.cat(preprocessed_images, dim=0)
                else:
                    continue
                
                image_batch = image_batch.to(self.config.device)
                answer_indices = answer_indices.to(self.config.device)
                
                # Forward pass
                visual_features = self.model.remoteclip_model.encode_image(image_batch)
                visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
                
                text_features_list = []
                for question in questions:
                    text_tokens = self.model.remoteclip_tokenizer(question)
                    text_input = text_tokens.to(self.config.device)
                    text_feat = self.model.remoteclip_model.encode_text(text_input)
                    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                    text_features_list.append(text_feat)
                
                text_features = torch.cat(text_features_list, dim=0)
                
                fused_features = self.model.fusion(visual_features, text_features)
                logits = self.model.vqa_head(fused_features)
                
                loss = self.criterion(logits, answer_indices)
                
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == answer_indices).sum().item()
                total += answer_indices.size(0)
                total_loss += loss.item() * answer_indices.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }
    
    def train(self, num_epochs: int, save_every: int = 1, val_every: int = 1) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            val_every: Validate every N epochs
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_accuracy'].append(train_metrics['train_accuracy'])
            
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_accuracy']:.4f}")
            
            # Validate
            if epoch % val_every == 0:
                val_metrics = self.validate()
                history['val_loss'].append(val_metrics.get('val_loss', 0))
                history['val_accuracy'].append(val_metrics.get('val_accuracy', 0))
                
                logger.info(f"Epoch {epoch}: Val Loss: {val_metrics.get('val_loss', 0):.4f}, Val Acc: {val_metrics.get('val_accuracy', 0):.4f}")
                
                # Save best model
                if val_metrics.get('val_accuracy', 0) > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics.get('val_accuracy', 0)
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, train_metrics)
        
        return history
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        
        self.model.save_checkpoint(str(checkpoint_path), epoch, metrics)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        self.model.checkpoint_path = checkpoint_path
        self.model.load()
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


def create_trainer(
    model: MultimodalVQA,
    config: VQAConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None
) -> VQATrainer:
    """
    Factory function to create VQA trainer.
    
    Args:
        model: MultimodalVQA model
        config: VQA configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        
    Returns:
        VQATrainer instance
    """
    return VQATrainer(model, config, train_loader, val_loader)
