"""
Interactive Image Testing for SatQuery AI

This script provides an interactive interface to:
1. Input your image path
2. Validate the image
3. Preprocess for RemoteCLIP
4. See detailed results
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.validation import DataValidator
from data.remoteclip_preprocessing import prepare_image_for_remoteclip
from data.preprocessing import ImagePreprocessor, PreprocessingConfig


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_section(title):
    """Print a section header."""
    print(f"\n--- {title} ---")


def get_user_image():
    """Get image path from user input."""
    print("SatQuery AI - Interactive Image Testing")
    print("Enter the path to your image (or 'quit' to exit):")
    
    while True:
        image_path = input("\nImage path: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            return None
        
        if not image_path:
            print("Please enter a valid path.")
            continue
        
        # Check if file exists
        if Path(image_path).exists():
            return image_path
        else:
            print(f"File not found: {image_path}")
            print("Please try again or enter 'quit' to exit.")


def validate_image(image_path):
    """Validate the image and show results."""
    print_section("STEP 1: Image Validation")
    
    result = DataValidator.validate_image_path(image_path)
    
    print(f"File: {image_path}")
    print(f"Status: {'PASS' if result.valid else 'FAIL'}")
    print(f"Total samples: {result.total_samples}")
    print(f"Valid samples: {result.valid_samples}")
    print(f"Invalid samples: {result.invalid_samples}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    return result.valid


def preprocess_remoteclip(image_path):
    """Preprocess image for RemoteCLIP and show results."""
    print_section("STEP 2: RemoteCLIP Preprocessing")
    
    try:
        processed_image = prepare_image_for_remoteclip(image_path)
        
        print(f"Status: PASS")
        print(f"Output shape: {processed_image.shape}")
        print(f"Data type: {processed_image.dtype}")
        print(f"Value range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        print(f"Mean: {processed_image.mean():.3f}")
        print(f"Std: {processed_image.std():.3f}")
        
        return True, processed_image
    except Exception as e:
        print(f"Status: FAIL")
        print(f"Error: {e}")
        return False, None


def preprocess_general(image_path):
    """Run general preprocessing and show metadata."""
    print_section("STEP 3: General Preprocessing & Metadata")
    
    try:
        config = PreprocessingConfig(target_size=(224, 224))
        preprocessor = ImagePreprocessor(config)
        
        image_data, metadata = preprocessor.load_image(image_path)
        
        print(f"Status: PASS")
        print(f"\nFile Information:")
        print(f"  Format: {metadata.get('format')}")
        print(f"  Size: {metadata.get('size')}")
        print(f"  Bands: {metadata.get('bands')}")
        print(f"  Modality: {metadata.get('modality')}")
        
        if metadata.get('crs'):
            print(f"  CRS: {metadata.get('crs')}")
        else:
            print(f"  CRS: None (not a GeoTIFF)")
        
        if metadata.get('transform'):
            print(f"  Transform: Available")
        else:
            print(f"  Transform: None")
        
        return True
    except Exception as e:
        print(f"Status: FAIL")
        print(f"Error: {e}")
        return False


def show_summary(all_passed):
    """Show final summary."""
    print_section("FINAL SUMMARY")
    
    if all_passed:
        print("SUCCESS: All tests passed!")
        print("\nYour image is ready for:")
        print("  - RemoteCLIP VQA inference")
        print("  - Dataset integration")
        print("  - Change detection (with paired images)")
        print("  - Model training/evaluation")
    else:
        print("FAILED: Some tests did not pass.")
        print("Please check the errors above and try with a different image.")


def interactive_test():
    """Run interactive image testing."""
    print_header("SatQuery AI - Interactive Image Testing")
    
    while True:
        # Get image from user
        image_path = get_user_image()
        
        if image_path is None:
            print("\nExiting...")
            break
        
        # Run tests
        print_header(f"Testing: {image_path}")
        
        # Step 1: Validation
        validation_passed = validate_image(image_path)
        
        if not validation_passed:
            show_summary(False)
            continue
        
        # Step 2: RemoteCLIP preprocessing
        remoteclip_passed, processed_image = preprocess_remoteclip(image_path)
        
        # Step 3: General preprocessing
        general_passed = preprocess_general(image_path)
        
        # Show summary
        all_passed = validation_passed and remoteclip_passed and general_passed
        show_summary(all_passed)
        
        # Ask if user wants to test another image
        print("\n" + "="*70)
        another = input("Test another image? (y/n): ").strip().lower()
        if another not in ['y', 'yes']:
            print("\nExiting...")
            break


def main():
    """Main entry point."""
    try:
        interactive_test()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
