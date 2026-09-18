"""
Checkpoint Manager for SatQueryAI Phase 3.1

This module handles saving and loading VQA model checkpoints with metadata.
"""

import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manager for VQA model checkpoints.
    
    Handles saving and loading of model checkpoints with associated metadata.
    """
    
    def __init__(self, checkpoint_dir: str = "models/vqa"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for saving checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model,
        answer_vocab: list,
        config: Dict[str, Any],
        training_metadata: Dict[str, Any],
        metrics: Dict[str, float],
        epoch: int,
        is_best: bool = False
    ) -> str:
        """
        Save model checkpoint with metadata.
        
        Args:
            model: MultimodalVQA model
            answer_vocab: Answer vocabulary list
            config: Model configuration
            training_metadata: Training metadata
            metrics: Current metrics
            epoch: Current epoch
            is_best: Whether this is the best model
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().isoformat()
        
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'timestamp': timestamp,
            'fusion': model.fusion.state_dict(),
            'vqa_head': model.vqa_head.state_dict(),
            'answer_vocab': answer_vocab,
            'num_answers': len(answer_vocab),
            'config': config,
            'training_metadata': training_metadata,
            'metrics': metrics
        }
        
        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save answer vocabulary separately
        vocab_path = self.checkpoint_dir / "answer_vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump({
                'vocabulary': answer_vocab,
                'timestamp': timestamp,
                'num_answers': len(answer_vocab)
            }, f, indent=2)
        
        # Save training metadata
        metadata_path = self.checkpoint_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        # Save config
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str, model) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: MultimodalVQA model to load into
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model weights
        model.fusion.load_state_dict(checkpoint['fusion'])
        model.vqa_head.load_state_dict(checkpoint['vqa_head'])
        
        # Update answer vocabulary
        if 'answer_vocab' in checkpoint:
            model.answer_vocab = checkpoint['answer_vocab']
            model.answer_to_idx = {ans: idx for idx, ans in enumerate(model.answer_vocab)}
            model.idx_to_answer = {idx: ans for idx, ans in enumerate(model.answer_vocab)}
            model.num_answers = len(model.answer_vocab)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'metrics': checkpoint.get('metrics', {}),
            'training_metadata': checkpoint.get('training_metadata', {})
        }
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            return str(max(checkpoints, key=lambda p: p.stat().st_mtime))
        return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return str(best_path)
        return None


def create_checkpoint_manager(checkpoint_dir: str = "models/vqa") -> CheckpointManager:
    """
    Factory function to create checkpoint manager.
    
    Args:
        checkpoint_dir: Directory for saving checkpoints
        
    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(checkpoint_dir)
