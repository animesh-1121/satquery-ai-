"""
Simple VQA Inference for SatQuery AI

Run RemoteCLIP VQA on your image with a question.

Usage:
    python scripts/run_vqa_simple.py --image path/to/image.jpg --question "What is in this image?"
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_vqa_inference(image_path: str, question: str, device: str = "cpu"):
    """Run VQA inference on an image."""
    
    print(f"\n{'='*70}")
    print(f"  SatQuery AI - RemoteCLIP VQA Inference")
    print(f"{'='*70}")
    print(f"\nImage: {image_path}")
    print(f"Question: {question}")
    print(f"Device: {device}\n")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        return None
    
    try:
        from models.vqa.remoteclip_vqa import create_inference_engine
        
        print("Loading RemoteCLIP VQA model...")
        engine = create_inference_engine(
            model_name="ViT-B-32",
            device=device,
            freeze_encoder=True
        )
        
        print("Running inference...")
        result = engine.predict(
            image_path=image_path,
            question=question
        )
        
        print(f"\n{'='*70}")
        print(f"  VQA Result")
        print(f"{'='*70}")
        print(f"\nAnswer: {result['answer']}")
        print(f"Model: {result['model']}")
        print(f"Task: {result['task']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference time: {result['inference_time_s']}s")
        print(f"Device: {result['device']}")
        
        if 'all_probabilities' in result:
            print(f"\nClass Probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
        
        if 'memory' in result:
            print(f"\nMemory Usage:")
            for key, value in result['memory'].items():
                print(f"  {key}: {value}")
        
        print(f"\n{'='*70}\n")
        
        return result
        
    except ImportError as e:
        print(f"ERROR: Required dependencies not installed: {e}")
        print("Please install: pip install open-clip-torch")
        return None
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Run VQA inference on an image")
    parser.add_argument("--image", required=True, help="Path to your image file")
    parser.add_argument("--question", required=True, help="Question about the image")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run inference on")
    
    args = parser.parse_args()
    
    result = run_vqa_inference(args.image, args.question, args.device)
    
    if result:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
