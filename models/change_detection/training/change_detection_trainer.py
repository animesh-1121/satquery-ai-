"""
Change Detection Trainer for SatQueryAI Phase 4

This module provides training infrastructure for the Siamese change detection model.
It handles training loops, validation, checkpointing, and parameter update verification.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List
from pathlib import Path
from datetime import datetime
import logging
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from models.change_detection.siamese_change_detection import SiameseChangeDetection, ChangeDetectionConfig
from data.change_detection_dataset import ChangeDetectionDataset, create_change_detection_dataloader

logger = logging.getLogger(__name__)


class ChangeDetectionTrainingConfig:
    """Configuration for change detection training."""
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 1,
        epochs: int = 10,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        mixed_precision: bool = False,
        checkpoint_dir: str = "models/change_detection",
        checkpoint_interval: int = 5,
        validation_interval: int = 1
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval


class ChangeDetectionTrainer:
    """
    Trainer for Siamese change detection model.
    
    Handles training loops, validation, checkpointing, and parameter update verification.
    """
    
    def __init__(
        self,
        model: SiameseChangeDetection,
        train_dataset: ChangeDetectionDataset,
        val_dataset: Optional[ChangeDetectionDataset] = None,
        config: Optional[ChangeDetectionTrainingConfig] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: SiameseChangeDetection model
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            config: Training configuration
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or ChangeDetectionTrainingConfig()
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.get_trainable_params(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Loss function (binary cross entropy for change detection)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # For parameter verification
        self.initial_params = self._get_trainable_param_snapshot()
    
    def _get_trainable_param_snapshot(self) -> Dict[str, float]:
        """Get snapshot of trainable parameter values for verification."""
        snapshot = {}
        for name, param in self.model.get_trainable_params():
            snapshot[name] = param.data.clone().cpu()
        return snapshot
    
    def _verify_parameter_updates(self) -> Dict[str, bool]:
        """
        Verify that trainable parameters have been updated.
        
        Returns:
            Dictionary mapping parameter names to whether they changed
        """
        current_params = self._get_trainable_param_snapshot()
        updates = {}
        
        for name in self.initial_params:
            if name in current_params:
                changed = not torch.equal(self.initial_params[name], current_params[name])
                updates[name] = changed
        
        return updates
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.remoteclip_model.eval()  # Keep RemoteCLIP frozen
        self.model.temporal_fusion.train()
        self.model.change_head.train()
        
        train_loader = create_change_detection_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract batch data
            image_t1_list = batch['image_t1']
            image_t2_list = batch['image_t2']
            change_labels = batch['change_label']
            
            # Skip samples without labels
            valid_indices = [i for i, label in enumerate(change_labels) if label is not None]
            if not valid_indices:
                continue
            
            # Filter to valid samples
            image_t1_list = [image_t1_list[i] for i in valid_indices]
            image_t2_list = [image_t2_list[i] for i in valid_indices]
            change_labels = [change_labels[i] for i in valid_indices]
            
            # Move to device
            if self.config.device == "cuda" and torch.cuda.is_available():
                change_labels = torch.tensor(change_labels).cuda()
            else:
                change_labels = torch.tensor(change_labels)
            
            # Encode images
            with torch.no_grad():
                f1_list = []
                f2_list = []
                for img_t1, img_t2 in zip(image_t1_list, image_t2_list):
                    f1 = self.model._encode_image(img_t1)
                    f2 = self.model._encode_image(img_t2)
                    f1_list.append(f1)
                    f2_list.append(f2)
                
                f1_batch = torch.cat(f1_list, dim=0)
                f2_batch = torch.cat(f2_list, dim=0)
            
            # Temporal fusion
            temporal_features = self.model.temporal_fusion(f1_batch, f2_batch)
            
            # Change detection
            logits = self.model.change_head(temporal_features)
            
            # Compute loss
            loss = self.criterion(logits, change_labels)
            
            # Gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Predictions
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == change_labels).sum().item()
            total_samples += len(change_labels)
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_params(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
        
        # Compute metrics
        avg_loss = total_loss / max(len(train_loader), 1)
        accuracy = total_correct / max(total_samples, 1)
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'total_samples': total_samples
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_dataset is None:
            return {}
        
        self.model.remoteclip_model.eval()
        self.model.temporal_fusion.eval()
        self.model.change_head.eval()
        
        val_loader = create_change_detection_dataloader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                image_t1_list = batch['image_t1']
                image_t2_list = batch['image_t2']
                change_labels = batch['change_label']
                
                # Skip samples without labels
                valid_indices = [i for i, label in enumerate(change_labels) if label is not None]
                if not valid_indices:
                    continue
                
                # Filter to valid samples
                image_t1_list = [image_t1_list[i] for i in valid_indices]
                image_t2_list = [image_t2_list[i] for i in valid_indices]
                change_labels = [change_labels[i] for i in valid_indices]
                
                # Move to device
                if self.config.device == "cuda" and torch.cuda.is_available():
                    change_labels = torch.tensor(change_labels).cuda()
                else:
                    change_labels = torch.tensor(change_labels)
                
                # Encode images
                f1_list = []
                f2_list = []
                for img_t1, img_t2 in zip(image_t1_list, image_t2_list):
                    f1 = self.model._encode_image(img_t1)
                    f2 = self.model._encode_image(img_t2)
                    f1_list.append(f1)
                    f2_list.append(f2)
                
                f1_batch = torch.cat(f1_list, dim=0)
                f2_batch = torch.cat(f2_list, dim=0)
                
                # Temporal fusion
                temporal_features = self.model.temporal_fusion(f1_batch, f2_batch)
                
                # Change detection
                logits = self.model.change_head(temporal_features)
                
                # Compute loss
                loss = self.criterion(logits, change_labels)
                total_loss += loss.item()
                
                # Predictions
                predictions = logits.argmax(dim=-1)
                total_correct += (predictions == change_labels).sum().item()
                total_samples += len(change_labels)
        
        # Compute metrics
        avg_loss = total_loss / max(len(val_loader), 1)
        accuracy = total_correct / max(total_samples, 1)
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_samples': total_samples
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting change detection training for {self.config.epochs} epochs")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.get_trainable_params())}")
        
        training_history = []
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}, Accuracy: {train_metrics['train_accuracy']:.4f}")
            
            # Validate
            val_metrics = {}
            if self.val_dataset is not None and (epoch + 1) % self.config.validation_interval == 0:
                val_metrics = self.validate()
                logger.info(f"Val Loss: {val_metrics.get('val_loss', 0):.4f}, Accuracy: {val_metrics.get('val_accuracy', 0):.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self.model.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch + 1,
                    metrics={**train_metrics, **val_metrics}
                )
            
            # Save best model
            if val_metrics and val_metrics.get('val_loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                best_checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                self.model.save_checkpoint(
                    best_checkpoint_path,
                    epoch=epoch + 1,
                    metrics={**train_metrics, **val_metrics}
                )
                logger.info(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
            
            training_history.append({
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics
            })
        
        # Verify parameter updates
        parameter_updates = self._verify_parameter_updates()
        logger.info(f"Parameter updates verified: {sum(parameter_updates.values())}/{len(parameter_updates)} parameters changed")
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(self.config.checkpoint_dir, "final_model.pt")
        self.model.save_checkpoint(
            final_checkpoint_path,
            epoch=self.config.epochs,
            metrics=training_history[-1] if training_history else {}
        )
        
        return {
            'training_history': training_history,
            'parameter_updates': parameter_updates,
            'best_val_loss': self.best_val_loss
        }
