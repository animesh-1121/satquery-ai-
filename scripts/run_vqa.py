#!/usr/bin/env python3
"""
CLI script for running GeoChat VQA inference.

This script provides a command-line interface for running visual question answering
on remote sensing images using the GeoChat model.

Usage:
    python scripts/run_vqa.py --image <image_path> --question "What type of land cover is visible in this image?"
"""

import argparse
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqa.geochat import create_inference_engine


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GeoChat VQA inference on remote sensing images"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input RGB satellite image"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Natural language question about the image"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="MBZUAI/geochat-7B",
        help="Path or HuggingFace ID to GeoChat model (default: MBZUAI/geochat-7B)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)"
    )
    
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load model in 8-bit mode for memory efficiency"
    )
    
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help="Maximum number of tokens to generate (default: 300)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save output as JSON file"
    )
    
    return parser.parse_args()


def main():
    """Main function to run VQA inference."""
    args = parse_args()
    
    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    print("=" * 60)
    print("GeoChat VQA Inference")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Question: {args.question}")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"8-bit mode: {args.load_8bit}")
    print("=" * 60)
    
    try:
        # Create inference engine
        print("\nLoading GeoChat model...")
        engine = create_inference_engine(
            model_path=args.model_path,
            device=args.device,
            load_8bit=args.load_8bit
        )
        
        # Run inference
        print("Running inference...")
        result = engine.predict(
            image_path=args.image,
            question=args.question,
            max_new_tokens=args.max_new_tokens
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Answer: {result['answer']}")
        print(f"Model: {result['model']}")
        print(f"Confidence: {result['confidence']}")
        print("=" * 60)
        
        # Save to JSON if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
