"""
Quick Image Test - Just provide your image path and see results

Usage:
    python scripts/quick_test.py "path/to/your/image.jpg"
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.validation import DataValidator
from data.remoteclip_preprocessing import prepare_image_for_remoteclip
from data.preprocessing import ImagePreprocessor, PreprocessingConfig


def quick_test(image_path):
    """Quick test of an image with all outputs."""
    
    print(f"\n{'='*70}")
    print(f"  SatQuery AI - Quick Image Test")
    print(f"{'='*70}")
    print(f"\nImage: {image_path}\n")
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"ERROR: File not found: {image_path}")
        return False
    
    # Step 1: Validation
    print("Step 1: Validation")
    print("-" * 40)
    result = DataValidator.validate_image_path(image_path)
    print(f"Status: {'PASS' if result.valid else 'FAIL'}")
    if result.errors:
        for error in result.errors:
            print(f"  Error: {error}")
    if result.warnings:
        for warning in result.warnings:
            print(f"  Warning: {warning}")
    
    if not result.valid:
        return False
    
    # Step 2: RemoteCLIP Preprocessing
    print("\nStep 2: RemoteCLIP Preprocessing")
    print("-" * 40)
    try:
        processed_image = prepare_image_for_remoteclip(image_path)
        print(f"Status: PASS")
        print(f"  Shape: {processed_image.shape}")
        print(f"  Dtype: {processed_image.dtype}")
        print(f"  Range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        print(f"  Mean: {processed_image.mean():.3f}")
        print(f"  Std: {processed_image.std():.3f}")
    except Exception as e:
        print(f"Status: FAIL - {e}")
        return False
    
    # Step 3: General Preprocessing
    print("\nStep 3: Image Metadata")
    print("-" * 40)
    try:
        config = PreprocessingConfig(target_size=(224, 224))
        preprocessor = ImagePreprocessor(config)
        image_data, metadata = preprocessor.load_image(image_path)
        
        print(f"Format: {metadata.get('format')}")
        print(f"Size: {metadata.get('size')}")
        print(f"Bands: {metadata.get('bands')}")
        print(f"Modality: {metadata.get('modality')}")
        
        if metadata.get('crs'):
            print(f"CRS: {metadata.get('crs')}")
        else:
            print(f"CRS: None (not a GeoTIFF)")
            
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    print(f"\n{'='*70}")
    print(f"SUCCESS: Image ready for SatQuery AI!")
    print(f"{'='*70}\n")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/quick_test.py <image_path>")
        print("Example: python scripts/quick_test.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = quick_test(image_path)
    sys.exit(0 if success else 1)
