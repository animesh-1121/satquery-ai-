"""
Interactive Multimodal VQA Script for SatQueryAI Phase 3

This script provides an interactive interface for running multimodal VQA inference
on satellite images with natural language questions.
"""

import sys
import os
import logging
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqa.multimodal_vqa.multimodal_vqa import MultimodalVQA, VQAConfig, create_multimodal_vqa
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*60)
    print("SatQueryAI - Multimodal VQA Interactive Mode")
    print("="*60)
    print("Ask questions about satellite images using RemoteCLIP")
    print("Type 'quit' to exit")
    print("="*60 + "\n")


def get_user_input(prompt: str) -> str:
    """Get user input with error handling."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nExiting...")
        return "quit"


def main():
    """Main interactive function."""
    print_banner()
    
    # Load answer vocabulary from checkpoint if available
    checkpoint_path = "models/vqa/best_model.pt"
    answer_vocab = None
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'answer_vocab' in checkpoint:
                answer_vocab = checkpoint['answer_vocab']
                print(f"Loaded answer vocabulary from checkpoint: {len(answer_vocab)} answers")
        except Exception as e:
            logger.warning(f"Failed to load answer vocabulary from checkpoint: {e}")
    
    # Fallback to synthetic vocabulary (with warning)
    if answer_vocab is None:
        print("WARNING: Using synthetic vocabulary (trained checkpoint not available)")
        print("For best results, train the model with real VQA data first.")
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
        ]
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create VQA configuration
    config = VQAConfig(
        remoteclip_model="ViT-B-32",
        embedding_dim=512,
        fusion_dim=256,
        num_answers=len(answer_vocab),
        device=device,
        freeze_remoteclip=True,
        mixed_precision=False
    )
    
    # Create model
    use_checkpoint = os.path.exists(checkpoint_path)
    print("Loading multimodal VQA model...")
    if use_checkpoint:
        print(f"Using trained checkpoint: {checkpoint_path}")
    else:
        print("Using randomly initialized weights (synthetic mode)")
    
    try:
        model = create_multimodal_vqa(
            answer_vocab=answer_vocab,
            config=config,
            checkpoint_path=checkpoint_path if use_checkpoint else None
        )
        print("Model loaded successfully!\n")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Interactive loop
    while True:
        # Get image path
        image_path = get_user_input("Enter image path (or 'quit'): ")
        
        if image_path.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}\n")
            continue
        
        # Get question
        question = get_user_input("Enter your question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not question:
            print("[ERROR] Question cannot be empty\n")
            continue
        
        # Run inference
        print(f"\nAnalyzing: {question}")
        print(f"Image: {image_path}\n")
        
        try:
            result = model.predict(
                image_path=image_path,
                question=question
            )
            
            # Print results
            print("="*60)
            print("VQA RESULT")
            print("="*60)
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Execution Time: {result['execution_time_ms']:.2f} ms")
            
            if result['memory']:
                print(f"GPU Memory: {result['memory'].get('gpu_allocated_mb', 'N/A')} MB")
            
            print("\nTop Answers:")
            for i, pred in enumerate(result['top_k_answers'][:3], 1):
                print(f"  {i}. {pred['answer']}: {pred['confidence']:.4f}")
            
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}\n")
            import traceback
            traceback.print_exc()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
