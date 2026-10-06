"""
Change Detection Inference Script for SatQueryAI Phase 4

This script runs change detection inference on a pair of satellite images
from different times (T1 and T2).

Usage:
    python scripts/run_change_detection.py \
        --image-t1 path/to/t1.jpg \
        --image-t2 path/to/t2.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.change_detection import SiameseChangeDetection, ChangeDetectionConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run change detection inference")
    parser.add_argument("--image-t1", type=str, required=True, help="Path to T1 image (earlier time)")
    parser.add_argument("--image-t2", type=str, required=True, help="Path to T2 image (later time)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Check open-clip availability
    try:
        import open_clip
    except ImportError:
        logger.error("open-clip-torch is not installed. Please install it with: pip install open-clip-torch")
        return
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    logger.info("=" * 60)
    logger.info("BI-TEMPORAL CHANGE DETECTION")
    logger.info("=" * 60)
    logger.info(f"T1 Image: {args.image_t1}")
    logger.info(f"T2 Image: {args.image_t2}")
    logger.info(f"Device: {args.device}")
    if args.checkpoint:
        logger.info(f"Checkpoint: {args.checkpoint}")
    else:
        logger.info("Checkpoint: None (using randomly initialized weights)")
    
    # Create model
    logger.info("\nInitializing Siamese change detection model...")
    model_config = ChangeDetectionConfig(
        device=args.device,
        freeze_remoteclip=True
    )
    
    model = SiameseChangeDetection(
        config=model_config,
        checkpoint_path=args.checkpoint
    )
    model.load()
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.get_trainable_params())
    frozen_params = sum(p.numel() for p in model.remoteclip_model.parameters())
    total_params = trainable_params + frozen_params
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters (RemoteCLIP): {frozen_params:,}")
    
    # Run inference
    logger.info("\nRunning change detection...")
    result = model.predict(
        image_t1_path=args.image_t1,
        image_t2_path=args.image_t2
    )
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    if result['status'] == 'success':
        logger.info(f"Task: {result['task']}")
        logger.info(f"Change Detected: {result['change_detected']}")
        logger.info(f"Change Label: {result['change_label']}")
        logger.info(f"Confidence: {result['confidence']:.4f}")
        logger.info(f"\nClass Probabilities:")
        logger.info(f"  No Change: {result['class_probabilities']['no_change']:.4f}")
        logger.info(f"  Change: {result['class_probabilities']['change']:.4f}")
        logger.info(f"\nModel: {result['model']}")
        logger.info(f"Execution Time: {result['execution_time_ms']:.2f} ms")
        logger.info(f"Device: {result['device']}")
        
        if result.get('memory'):
            logger.info(f"\nMemory Usage:")
            logger.info(f"  GPU Allocated: {result['memory'].get('gpu_allocated_mb', 0)} MB")
            logger.info(f"  GPU Reserved: {result['memory'].get('gpu_reserved_mb', 0)} MB")
    else:
        logger.error(f"Error: {result.get('error', 'Unknown error')}")
    
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
