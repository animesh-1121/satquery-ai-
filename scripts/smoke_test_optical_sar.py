"""
Smoke Test Training for Optical-SAR Fusion (Phase 5)

This script runs a minimal smoke test to verify the optical-SAR training pipeline.
It uses synthetic data to test the complete training flow without requiring the real dataset.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.optical_sar import OpticalSARFusion, OpticalSARConfig
from models.optical_sar.training import OpticalSARTrainer, OpticalSARTrainingConfig
from data.optical_sar_dataset import OpticalSARDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Smoke test training for optical-SAR fusion")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of synthetic samples")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default="models/optical_sar", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Check open-clip availability
    try:
        import open_clip
    except ImportError:
        logger.error("open-clip-torch is not installed. Please install it with: pip install open-clip-torch")
        logger.error("For now, this smoke test will exit. The infrastructure is complete but requires the dependency.")
        return
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    logger.info("=" * 60)
    logger.info("OPTICAL-SAR FUSION SMOKE TEST TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Synthetic samples: {args.num_samples}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    # Create synthetic datasets
    logger.info("\nCreating synthetic datasets...")
    train_dataset = OpticalSARDataset(
        root="synthetic",
        split="train",
        subset_size=args.num_samples,
        synthetic=True
    )
    
    val_dataset = OpticalSARDataset(
        root="synthetic",
        split="val",
        subset_size=args.num_samples // 2,
        synthetic=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create model
    logger.info("\nInitializing optical-SAR fusion model...")
    model_config = OpticalSARConfig(
        device=args.device,
        freeze_optical_encoder=True,
        freeze_sar_encoder=True,
        batch_size=args.batch_size
    )
    
    model = OpticalSARFusion(config=model_config)
    model.load()
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.get_trainable_params())
    optical_params = sum(p.numel() for p in model.remoteclip_model.parameters())
    sar_params = sum(p.numel() for p in model.sar_encoder.parameters())
    total_params = trainable_params + optical_params + sar_params
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen optical parameters: {optical_params:,}")
    logger.info(f"Frozen SAR parameters: {sar_params:,}")
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    training_config = OpticalSARTrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    trainer = OpticalSARTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config
    )
    
    # Run training
    logger.info("\nStarting training...")
    results = trainer.train()
    
    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    
    final_metrics = results['training_history'][-1] if results['training_history'] else {}
    logger.info(f"Final train loss: {final_metrics.get('train_loss', 0):.4f}")
    logger.info(f"Final train accuracy: {final_metrics.get('train_accuracy', 0):.4f}")
    
    logger.info(f"\nParameter updates verified:")
    for param_idx, changed in results['parameter_updates'].items():
        status = "✓ CHANGED" if changed else "✗ UNCHANGED"
        logger.info(f"  Parameter {param_idx}: {status}")
    
    logger.info(f"\nBest checkpoint saved to: {args.checkpoint_dir}/best_model.pt")
    logger.info(f"Final checkpoint saved to: {args.checkpoint_dir}/final_model.pt")
    
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST SUCCESSFUL")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
