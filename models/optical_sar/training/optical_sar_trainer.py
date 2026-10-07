"""
Optical-SAR Trainer for SatQueryAI Phase 5
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
from datetime import datetime
import logging

from ..optical_sar_fusion import OpticalSARFusion, OpticalSARConfig
from data.optical_sar_dataset import OpticalSARDataset, create_optical_sar_dataloader

logger = logging.getLogger(__name__)


class OpticalSARTrainingConfig:
    """Configuration for optical-SAR training."""
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 1,
        epochs: int = 10,
        gradient_accumulation_steps: int = 4,
        device: str = "cuda",
        checkpoint_dir: str = "models/optical_sar"
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.checkpoint_dir = checkpoint_dir


class OpticalSARTrainer:
    """Trainer for optical-SAR fusion model."""
    
    def __init__(
        self,
        model: OpticalSARFusion,
        train_dataset: OpticalSARDataset,
        val_dataset: Optional[OpticalSARDataset] = None,
        config: Optional[OpticalSARTrainingConfig] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or OpticalSARTrainingConfig()
        
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        self.optimizer = optim.AdamW(
            self.model.get_trainable_params(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.initial_params = self._get_trainable_param_snapshot()
    
    def _get_trainable_param_snapshot(self):
        snapshot = {}
        for name, param in enumerate(self.model.get_trainable_params()):
            snapshot[name] = param.data.clone().cpu()
        return snapshot
    
    def _verify_parameter_updates(self):
        current_params = self._get_trainable_param_snapshot()
        updates = {}
        for name in self.initial_params:
            if name in current_params:
                changed = not torch.equal(self.initial_params[name], current_params[name])
                updates[name] = changed
        return updates
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.remoteclip_model.eval()
        self.model.sar_encoder.eval()
        self.model.optical_projection.train()
        self.model.sar_projection.train()
        self.model.cross_attention.train()
        self.model.fusion_head.train()
        
        train_loader = create_optical_sar_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            optical_list = batch['optical']
            sar_list = batch['sar']
            labels = batch['label']
            
            valid_indices = [i for i, label in enumerate(labels) if label is not None]
            if not valid_indices:
                continue
            
            optical_list = [optical_list[i] for i in valid_indices]
            sar_list = [sar_list[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            
            if self.config.device == "cuda" and torch.cuda.is_available():
                labels = torch.tensor(labels).cuda()
            else:
                labels = torch.tensor(labels)
            
            # Encode optical
            with torch.no_grad():
                optical_features_list = []
                for img in optical_list:
                    optical_features_list.append(self.model._encode_optical(img))
                optical_features = torch.cat(optical_features_list, dim=0)
            
            # Encode SAR
            with torch.no_grad():
                from torchvision import transforms
                sar_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                sar_tensors = []
                for img in sar_list:
                    sar_tensors.append(sar_transform(img).unsqueeze(0))
                sar_batch = torch.cat(sar_tensors, dim=0)
                sar_features = self.model._encode_sar(sar_batch)
            
            # Project and fuse
            optical_proj = self.model.optical_projection(optical_features)
            sar_proj = self.model.sar_projection(sar_features)
            fused = self.model.cross_attention(optical_proj, sar_proj)
            logits = self.model.fusion_head(fused)
            
            loss = self.criterion(logits, labels)
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_params(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        avg_loss = total_loss / max(len(train_loader), 1)
        accuracy = total_correct / max(total_samples, 1)
        
        return {'train_loss': avg_loss, 'train_accuracy': accuracy, 'total_samples': total_samples}
    
    def train(self) -> Dict[str, Any]:
        logger.info(f"Starting optical-SAR training for {self.config.epochs} epochs")
        
        training_history = []
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}, Accuracy: {train_metrics['train_accuracy']:.4f}")
            
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self.model.save_checkpoint(checkpoint_path, epoch + 1, train_metrics)
            
            if train_metrics['train_loss'] < self.best_val_loss:
                self.best_val_loss = train_metrics['train_loss']
                best_checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                self.model.save_checkpoint(best_checkpoint_path, epoch + 1, train_metrics)
                logger.info(f"Saved best model with loss: {self.best_val_loss:.4f}")
            
            training_history.append({'epoch': epoch + 1, **train_metrics})
        
        parameter_updates = self._verify_parameter_updates()
        logger.info(f"Parameter updates verified: {sum(parameter_updates.values())}/{len(parameter_updates)} parameters changed")
        
        final_checkpoint_path = os.path.join(self.config.checkpoint_dir, "final_model.pt")
        self.model.save_checkpoint(final_checkpoint_path, self.config.epochs, training_history[-1] if training_history else {})
        
        return {
            'training_history': training_history,
            'parameter_updates': parameter_updates,
            'best_val_loss': self.best_val_loss
        }
