"""
Multimodal VQA Inference Script for SatQueryAI Phase 3

This script provides command-line interface for running multimodal VQA inference
on satellite images with natural language questions.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqa.multimodal_vqa.multimodal_vqa import MultimodalVQA, VQAConfig, create_multimodal_vqa
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Multimodal VQA Inference for SatQueryAI")
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to satellite image"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Natural language question about the image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/vqa/best_model.pt",
        help="Path to trained VQA checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)"
    )
    parser.add_argument(
        "--num-answers",
        type=int,
        default=10,
        help="Number of possible answers in vocabulary (only used with --synthetic)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic vocabulary instead of trained checkpoint"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return 1
    
    if not args.question or not args.question.strip():
        logger.error("Question cannot be empty")
        return 1
    
    # Load answer vocabulary from checkpoint if available (unless --synthetic flag)
    answer_vocab = None
    if not args.synthetic and args.checkpoint and os.path.exists(args.checkpoint):
        try:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            if 'answer_vocab' in checkpoint:
                answer_vocab = checkpoint['answer_vocab']
                logger.info(f"Loaded answer vocabulary from checkpoint: {len(answer_vocab)} answers")
        except Exception as e:
            logger.warning(f"Failed to load answer vocabulary from checkpoint: {e}")
    
    # Use synthetic vocabulary if --synthetic flag or checkpoint not available
    if answer_vocab is None:
        if args.synthetic:
            logger.info("Using synthetic vocabulary (--synthetic flag)")
        else:
            logger.warning("Using synthetic vocabulary (checkpoint not available)")
        answer_vocab = [
            "agricultural",
            "forest", 
            "urban",
            "water",
            "barren",
            "grassland",
            "industrial",
            "residential",
            "wetland",
            "mixed"
        ][:args.num_answers]
    
    # Create VQA configuration
    config = VQAConfig(
        remoteclip_model="ViT-B-32",
        embedding_dim=512,
        fusion_dim=256,
        num_answers=len(answer_vocab),
        device=args.device,
        freeze_remoteclip=True,
        mixed_precision=False
    )
    
    # Update num_answers to match actual vocabulary
    args.num_answers = len(answer_vocab)
    
    # Create model
    logger.info("Loading multimodal VQA model...")
    try:
        model = create_multimodal_vqa(
            answer_vocab=answer_vocab,
            config=config,
            checkpoint_path=args.checkpoint
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Run inference
    logger.info(f"Running VQA inference on {args.image}")
    logger.info(f"Question: {args.question}")
    
    try:
        result = model.predict(
            image_path=args.image,
            question=args.question
        )
        
        # Print results
        print("\n" + "="*60)
        print("VQA INFERENCE RESULT")
        print("="*60)
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Model: {result['model']}")
        print(f"Task: {result['task']}")
        print(f"Execution Time: {result['execution_time_ms']:.2f} ms")
        print(f"Device: {result['device']}")
        
        if result['memory']:
            print(f"GPU Allocated: {result['memory'].get('gpu_allocated_mb', 'N/A')} MB")
            print(f"GPU Reserved: {result['memory'].get('gpu_reserved_mb', 'N/A')} MB")
        
        print("\nTop 5 Answers:")
        for i, pred in enumerate(result['top_k_answers'][:5], 1):
            print(f"  {i}. {pred['answer']}: {pred['confidence']:.4f}")
        
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
