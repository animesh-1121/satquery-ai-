"""
Optical-SAR Fusion Inference Script for SatQueryAI Phase 5

This script runs optical-SAR fusion inference on paired optical and SAR images.

Usage:
    python scripts/run_optical_sar.py \
        --optical path/to/optical.jpg \
        --sar path/to/sar.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.optical_sar import OpticalSARFusion, OpticalSARConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run optical-SAR fusion inference")
    parser.add_argument("--optical", type=str, required=True, help="Path to optical image")
    parser.add_argument("--sar", type=str, required=True, help="Path to SAR image")
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
    logger.info("OPTICAL-SAR FUSION INFERENCE")
    logger.info("=" * 60)
    logger.info(f"Optical Image: {args.optical}")
    logger.info(f"SAR Image: {args.sar}")
    logger.info(f"Device: {args.device}")
    if args.checkpoint:
        logger.info(f"Checkpoint: {args.checkpoint}")
    else:
        logger.info("Checkpoint: None (using randomly initialized weights)")
    
    # Create model
    logger.info("\nInitializing optical-SAR fusion model...")
    model_config = OpticalSARConfig(
        device=args.device,
        freeze_optical_encoder=True,
        freeze_sar_encoder=True
    )
    
    model = OpticalSARFusion(
        config=model_config,
        checkpoint_path=args.checkpoint
    )
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
    
    # Run inference
    logger.info("\nRunning optical-SAR fusion...")
    result = model.predict(
        optical_path=args.optical,
        sar_path=args.sar
    )
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    if result['status'] == 'success':
        logger.info(f"Task: {result['task']}")
        logger.info(f"Prediction: {result['prediction']}")
        logger.info(f"Confidence: {result['confidence']:.4f}")
        logger.info(f"\nTop K Classes:")
        for item in result['top_k_classes']:
            logger.info(f"  Class {item['class']}: {item['confidence']:.4f}")
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
