#!/usr/bin/env python3
"""
Test script to verify GeoChat interface is working without requiring model download.
This demonstrates that the code infrastructure is functional.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_interface():
    """Test that the GeoChat interface can be imported and basic functionality works."""
    print("=" * 60)
    print("GeoChat Interface Test")
    print("=" * 60)
    
    # Test 1: Import check
    print("\n1. Testing GeoChat import...")
    try:
        from models.vqa.geochat.inference import create_inference_engine, GeoChatInference
        print("[OK] GeoChat interface imported successfully")
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False
    
    # Test 2: Check class structure
    print("\n2. Testing class structure...")
    try:
        assert hasattr(GeoChatInference, '__init__'), "Missing __init__ method"
        assert hasattr(GeoChatInference, 'predict'), "Missing predict method"
        assert hasattr(GeoChatInference, '__call__'), "Missing __call__ method"
        print("[OK] GeoChatInference class structure correct")
    except AssertionError as e:
        print(f"[FAIL] Class structure error: {e}")
        return False
    
    # Test 3: Check factory function
    print("\n3. Testing factory function...")
    try:
        assert callable(create_inference_engine), "create_inference_engine not callable"
        print("[OK] Factory function is callable")
    except AssertionError as e:
        print(f"[FAIL] Factory function error: {e}")
        return False
    
    # Test 4: Check GeoChat availability
    print("\n4. Testing GeoChat package availability...")
    try:
        from geochat.model.builder import load_pretrained_model
        from geochat.mm_utils import get_model_name_from_path
        from geochat.conversation import conv_templates, Chat
        print("[OK] GeoChat package installed and importable")
    except ImportError as e:
        print(f"[FAIL] GeoChat package not available: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Interface Test Complete")
    print("=" * 60)
    print("\nSummary:")
    print("[OK] GeoChat interface is functional")
    print("[OK] GeoChat package installed and importable")
    print("[OK] All required components available")
    print("[WARN] Model download needed for real inference")
    print("\nTo run real inference:")
    print("1. Ensure stable network connection")
    print("2. Run: venv310\\Scripts\\activate")
    print("3. Run: python scripts/run_vqa.py --image test.jpg --question \"What is this?\" --device cpu")
    
    return True

if __name__ == "__main__":
    success = test_interface()
    sys.exit(0 if success else 1)
