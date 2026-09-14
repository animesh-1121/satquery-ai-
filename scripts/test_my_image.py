"""
Test your own image with SatQuery AI Phase 2 data pipeline.

Usage:
    python scripts/test_my_image.py --image path/to/your/image.jpg
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.validation import DataValidator
from data.remoteclip_preprocessing import prepare_image_for_remoteclip
from data.preprocessing import ImagePreprocessor, PreprocessingConfig


def test_image(image_path: str):
    """Test an image through the SatQuery AI pipeline."""
    
    print(f"\n{'='*60}")
    print(f"Testing Image: {image_path}")
    print(f"{'='*60}\n")
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"[ERROR] Image not found at {image_path}")
        return False
    
    # 1. Validate the image
    print("Step 1: Validating image...")
    validation_result = DataValidator.validate_image_path(image_path)
    
    if validation_result.valid:
        print(f"[PASS] Image validation passed")
    else:
        print(f"[FAIL] Image validation failed")
        for error in validation_result.errors:
            print(f"   Error: {error}")
        for warning in validation_result.warnings:
            print(f"   Warning: {warning}")
        return False
    
    # 2. Preprocess for RemoteCLIP
    print("\nStep 2: Preprocessing for RemoteCLIP...")
    try:
        processed_image = prepare_image_for_remoteclip(image_path)
        print(f"[PASS] RemoteCLIP preprocessing successful")
        print(f"   Shape: {processed_image.shape}")
        print(f"   Dtype: {processed_image.dtype}")
        print(f"   Range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    except Exception as e:
        print(f"[FAIL] Preprocessing failed: {e}")
        return False
    
    # 3. Test general preprocessing
    print("\nStep 3: Testing general preprocessing pipeline...")
    try:
        config = PreprocessingConfig(target_size=(224, 224))
        preprocessor = ImagePreprocessor(config)
        
        image_data, img_metadata = preprocessor.load_image(image_path)
        print(f"[PASS] General preprocessing successful")
        print(f"   Format: {img_metadata.get('format')}")
        print(f"   Size: {img_metadata.get('size')}")
        print(f"   Bands: {img_metadata.get('bands')}")
        print(f"   Modality: {img_metadata.get('modality')}")
        
        if img_metadata.get('crs'):
            print(f"   CRS: {img_metadata.get('crs')}")
        
    except Exception as e:
        print(f"[FAIL] General preprocessing failed: {e}")
        return False
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS] All tests passed! Your image is ready for SatQuery AI")
    print(f"{'='*60}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test your image with SatQuery AI")
    parser.add_argument("--image", required=True, help="Path to your image file")
    
    args = parser.parse_args()
    
    success = test_image(args.image)
    
    if success:
        print("\nNext steps:")
        print("1. Use this image with RemoteCLIP VQA model")
        print("2. Add it to a dataset for training/evaluation")
        print("3. Test compatibility with other images for change detection")
        return 0
    else:
        print("\nPlease check the errors above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
