#!/usr/bin/env python3
"""
Test script for RemoteCLIP loading and inference.

This script tests RemoteCLIP model loading, image encoding, and basic functionality
without requiring the full VQA pipeline.
"""

import sys
import os
import time
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import open_clip
    from huggingface_hub import hf_hub_download
    from PIL import Image
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install open-clip-torch huggingface_hub")
    sys.exit(1)


def test_remoteclip_loading():
    """Test RemoteCLIP model loading and basic inference."""
    
    print("=" * 60)
    print("RemoteCLIP Loading Test")
    print("=" * 60)
    
    # Model configuration
    model_name = "ViT-B-32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nConfiguration:")
    print(f"  Model: RemoteCLIP-{model_name}")
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Download checkpoint
    print(f"\nDownloading RemoteCLIP-{model_name} checkpoint...")
    download_start = time.time()
    try:
        checkpoint_path = hf_hub_download(
            repo_id="chendelong/RemoteCLIP",
            filename=f"RemoteCLIP-{model_name}.pt",
            cache_dir="checkpoints"
        )
        download_time = time.time() - download_start
        print(f"  Checkpoint downloaded in {download_time:.2f}s")
        print(f"  Path: {checkpoint_path}")
    except Exception as e:
        print(f"  ERROR downloading checkpoint: {e}")
        return False
    
    # Load OpenCLIP model
    print(f"\nLoading OpenCLIP model...")
    load_start = time.time()
    try:
        model, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained="openai")
        tokenizer = open_clip.get_tokenizer(model_name)
        load_time = time.time() - load_start
        print(f"  OpenCLIP model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"  ERROR loading OpenCLIP: {e}")
        return False
    
    # Load RemoteCLIP weights
    print(f"\nLoading RemoteCLIP weights...")
    load_start = time.time()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        load_time = time.time() - load_start
        print(f"  RemoteCLIP weights loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"  ERROR loading RemoteCLIP weights: {e}")
        return False
    
    # Move to device
    print(f"\nMoving model to {device}...")
    model = model.to(device).eval()
    
    # Get model info
    embedding_dim = model.visual.output_dim
    print(f"\nModel information:")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with a dummy image
    print(f"\nTesting image encoding...")
    try:
        # Create a dummy RGB image
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        image_input = preprocess(dummy_image).unsqueeze(0).to(device)
        
        # Encode
        with torch.no_grad():
            encode_start = time.time()
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            encode_time = time.time() - encode_start
        
        print(f"  Image encoded successfully")
        print(f"  Encoding time: {encode_time*1000:.2f}ms")
        print(f"  Embedding shape: {embedding.shape}")
        
        # Memory stats
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated(device) / (1024**2)
            gpu_reserved = torch.cuda.memory_reserved(device) / (1024**2)
            print(f"  GPU allocated: {gpu_allocated:.1f} MB")
            print(f"  GPU reserved: {gpu_reserved:.1f} MB")
    except Exception as e:
        print(f"  ERROR encoding image: {e}")
        return False
    
    # Test text encoding
    print(f"\nTesting text encoding...")
    try:
        text_queries = ["A satellite image of agricultural land.", "Urban area from above."]
        text = tokenizer(text_queries).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        print(f"  Text encoded successfully")
        print(f"  Text features shape: {text_features.shape}")
    except Exception as e:
        print(f"  ERROR encoding text: {e}")
        return False
    
    # Compute similarity
    print(f"\nComputing image-text similarity...")
    try:
        with torch.no_grad():
            similarity = (100.0 * embedding @ text_features.T).softmax(dim=-1)
        
        print(f"  Similarity scores:")
        for query, score in zip(text_queries, similarity[0].cpu().numpy()):
            print(f"    {query}: {score:.4f}")
    except Exception as e:
        print(f"  ERROR computing similarity: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    
    # Return test results
    return {
        "model": f"RemoteCLIP-{model_name}",
        "device": device,
        "embedding_dimension": str(embedding_dim),
        "parameters": sum(p.numel() for p in model.parameters()),
        "inference_time_seconds": f"{encode_time:.4f}",
        "peak_vram_mb": f"{gpu_allocated:.1f}" if torch.cuda.is_available() else "N/A",
        "checkpoint_size_mb": f"{os.path.getsize(checkpoint_path) / (1024**2):.1f}"
    }


if __name__ == "__main__":
    result = test_remoteclip_loading()
    if result:
        print("\nTest Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        sys.exit(0)
    else:
        print("\nTest failed!")
        sys.exit(1)
