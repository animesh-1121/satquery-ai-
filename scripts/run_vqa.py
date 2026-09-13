#!/usr/bin/env python3
"""
CLI script for running VQA inference on remote sensing images.

This script provides a command-line interface for running visual question answering
using SatQueryAI's pluggable model architecture with RemoteCLIP.

Usage:
    python scripts/run_vqa.py --image <image_path> --question "What type of land cover is visible in this image?"
"""

import argparse
import json
import sys
import os
import time
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqa.remoteclip_vqa import create_inference_engine


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VQA inference on remote sensing images using RemoteCLIP"
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
        "--model-name",
        type=str,
        default="ViT-B-32",
        choices=["RN50", "ViT-B-32", "ViT-L-14"],
        help="RemoteCLIP model variant (default: ViT-B-32)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)"
    )
    
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Local path to RemoteCLIP checkpoint (if None, downloads from HuggingFace)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching downloaded checkpoints"
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
    print("RemoteCLIP VQA Inference")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Question: {args.question}")
    print(f"Model: RemoteCLIP-{args.model_name}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    try:
        # Create inference engine
        print("\nLoading RemoteCLIP model...")
        load_start = time.time()
        engine = create_inference_engine(
            model_name=args.model_name,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            cache_dir=args.cache_dir
        )
        load_elapsed = time.time() - load_start
        
        # Get image info
        from PIL import Image
        img = Image.open(args.image)
        img_size = img.size
        img_mode = img.mode
        
        # Run inference
        print("Running inference...")
        result = engine.predict(
            image_path=args.image,
            question=args.question
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Answer: {result['answer']}")
        print(f"Model: {result['model']}")
        print(f"Task: {result['task']}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Device: {result['device']}")
        print(f"Inference time: {result['inference_time_s']}s")
        print(f"Model load time: {load_elapsed:.1f}s")
        print(f"Image: {img_size[0]}x{img_size[1]} {img_mode}")
        
        if "all_probabilities" in result:
            print("\nClass probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
        
        if "memory" in result and result["memory"]:
            print(f"\nMemory: {json.dumps(result['memory'])}")
        
        print("=" * 60)
        
        # Build full output
        full_result = {
            **result,
            "model_load_time_s": round(load_elapsed, 2),
            "image_path": os.path.abspath(args.image),
            "image_dimensions": f"{img_size[0]}x{img_size[1]}",
            "image_mode": img_mode,
            "question": args.question,
        }
        
        # Save to JSON if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(full_result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except MemoryError as e:
        print(f"\n✗ Hardware insufficient: {e}")
        return 1
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n✗ Out of memory: {e}")
            print("Try: --device cpu")
        else:
            print(f"\n✗ Runtime error: {e}")
            traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
