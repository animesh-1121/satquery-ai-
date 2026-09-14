"""
Interactive VQA for SatQuery AI

Ask questions about your images interactively.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def interactive_vqa():
    """Run interactive VQA session."""
    
    print_header("SatQuery AI - Interactive VQA")
    
    try:
        from models.vqa.remoteclip_vqa import create_inference_engine
    except ImportError as e:
        print(f"ERROR: Required dependencies not installed: {e}")
        print("Please install: pip install open-clip-torch")
        return
    
    # Load model once
    print("Loading RemoteCLIP VQA model (this may take a moment)...")
    try:
        engine = create_inference_engine(
            model_name="ViT-B-32",
            device="cpu",  # Use CPU by default, change to "cuda" if available
            freeze_encoder=True
        )
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    # Interactive loop
    while True:
        print("="*70)
        
        # Get image path
        image_path = input("Enter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("\nExiting...")
            break
        
        if not image_path:
            print("Please enter a valid path.")
            continue
        
        if not Path(image_path).exists():
            print(f"File not found: {image_path}")
            continue
        
        # Get question
        question = input("Enter your question about the image: ").strip()
        
        if not question:
            print("Please enter a question.")
            continue
        
        # Run inference
        print(f"\nAnalyzing image: {image_path}")
        print(f"Question: {question}\n")
        
        try:
            result = engine.predict(
                image_path=image_path,
                question=question
            )
            
            print(f"Answer: {result['answer']}")
            print(f"Task: {result['task']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Inference time: {result['inference_time_s']}s")
            
            if 'all_probabilities' in result:
                print(f"\nTop predictions:")
                # Sort by probability
                sorted_probs = sorted(
                    result['all_probabilities'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                for class_name, prob in sorted_probs[:3]:
                    print(f"  {class_name}: {prob:.4f}")
            
        except Exception as e:
            print(f"ERROR: Inference failed: {e}")
        
        print()  # Add spacing


def main():
    """Main entry point."""
    try:
        interactive_vqa()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
