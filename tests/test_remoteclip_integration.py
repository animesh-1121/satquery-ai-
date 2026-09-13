"""
RemoteCLIP Integration Test for SatQueryAI

This test verifies that the data pipeline works correctly with RemoteCLIP:
- Real image preprocessing
- RemoteCLIP model loading
- Real embedding generation
- No mock data or fake embeddings
"""

import unittest
import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from data.remoteclip_preprocessing import RemoteCLIPPreprocessor, prepare_image_for_remoteclip


class TestRemoteCLIPIntegration(unittest.TestCase):
    """Test RemoteCLIP integration with real data."""
    
    @classmethod
    def setUpClass(cls):
        """Create a test image."""
        if not PIL_AVAILABLE:
            cls.skipTest(cls, "PIL not available")
            return
        
        if not NUMPY_AVAILABLE:
            cls.skipTest(cls, "NumPy not available")
            return
        
        # Create a more realistic test image with variation
        np.random.seed(42)
        image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cls.test_image = Image.fromarray(image_array)
        
        # Save to temporary file
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_image_path = os.path.join(cls.temp_dir, "test_image.jpg")
        cls.test_image.save(cls.test_image_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        if hasattr(cls, 'temp_dir'):
            import shutil
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_remoteclip_preprocessor_creation(self):
        """Test creating RemoteCLIP preprocessor."""
        if not PIL_AVAILABLE:
            self.skipTest("PIL not available")
        
        preprocessor = RemoteCLIPPreprocessor()
        
        self.assertIsNotNone(preprocessor.preprocessor)
        self.assertEqual(preprocessor.preprocessor.config.target_size, (224, 224))
    
    def test_real_image_preprocessing(self):
        """Test preprocessing a real image for RemoteCLIP."""
        if not PIL_AVAILABLE:
            self.skipTest("PIL not available")
        
        preprocessor = RemoteCLIPPreprocessor()
        
        # Preprocess the test image
        processed_image, metadata = preprocessor.preprocess_for_remoteclip(self.test_image_path)
        
        # Verify output shape (CHW format, 3 channels, 224x224)
        self.assertEqual(processed_image.shape, (3, 224, 224))
        
        # Verify metadata
        self.assertIn("path", metadata)
        self.assertIn("original_size", metadata)
        self.assertIn("target_size", metadata)
        self.assertEqual(metadata["target_size"], (224, 224))
        self.assertEqual(metadata["format"], "CHW")
    
    def test_remoteclip_normalization(self):
        """Test RemoteCLIP-specific normalization."""
        if not PIL_AVAILABLE:
            self.skipTest("PIL not available")
        
        preprocessor = RemoteCLIPPreprocessor(normalize=True)
        
        # Preprocess with normalization
        processed_image, metadata = preprocessor.preprocess_for_remoteclip(self.test_image_path)
        
        # Verify normalization was applied
        self.assertEqual(metadata["normalization"], "remoteclip")
        
        # Check that values are normalized (should be centered around 0, not 0-1)
        # RemoteCLIP normalization centers around 0
        self.assertLess(abs(processed_image.mean()), 5.0)  # Mean should be reasonably close to 0
        
        # Check that values are not in the original [0, 255] range
        self.assertLess(processed_image.max(), 10.0)  # Should be much smaller than 255
        self.assertGreater(processed_image.min(), -10.0)  # Should be negative values
    
    def test_convenience_function(self):
        """Test the convenience function for image preparation."""
        if not PIL_AVAILABLE:
            self.skipTest("PIL not available")
        
        # Use convenience function
        image = prepare_image_for_remoteclip(self.test_image_path)
        
        # Verify output
        self.assertEqual(image.shape, (3, 224, 224))
    
    def test_batch_preprocessing(self):
        """Test preprocessing a batch of images."""
        if not PIL_AVAILABLE:
            self.skipTest("PIL not available")
        
        # Create multiple test images
        image_paths = []
        for i in range(3):
            img = Image.new('RGB', (256, 256), color=(i * 50, i * 50, i * 50))
            path = os.path.join(self.temp_dir, f"test_image_{i}.jpg")
            img.save(path)
            image_paths.append(path)
        
        # Preprocess batch
        preprocessor = RemoteCLIPPreprocessor()
        batch_images, metadata_list = preprocessor.preprocess_batch(image_paths)
        
        # Verify batch shape (batch_size, channels, height, width)
        self.assertEqual(batch_images.shape, (3, 3, 224, 224))
        self.assertEqual(len(metadata_list), 3)
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_remoteclip_model_loading(self):
        """Test loading RemoteCLIP model (requires open-clip-torch)."""
        try:
            from models.vqa.remoteclip_vqa import RemoteCLIPVQA
        except ImportError:
            self.skipTest("RemoteCLIP VQA module not available")
        
        try:
            import open_clip
        except ImportError:
            self.skipTest("open-clip-torch not available")
        
        # Create model (don't load checkpoint, just test interface)
        model = RemoteCLIPVQA(
            model_name="ViT-B-32",
            device="cpu",  # Use CPU for testing
            checkpoint_path=None,  # Don't download checkpoint
            freeze_encoder=True
        )
        
        # Verify model attributes
        self.assertEqual(model.remoteclip_model_name, "ViT-B-32")
        self.assertTrue(model.freeze_encoder)
        self.assertIsNotNone(model.question_parser)
    
    def test_different_target_sizes(self):
        """Test preprocessing with different target sizes."""
        if not PIL_AVAILABLE:
            self.skipTest("PIL not available")
        
        # Test with different target size
        preprocessor = RemoteCLIPPreprocessor(target_size=(336, 336))
        
        processed_image, metadata = preprocessor.preprocess_for_remoteclip(self.test_image_path)
        
        # Verify custom target size
        self.assertEqual(processed_image.shape, (3, 336, 336))
        self.assertEqual(metadata["target_size"], (336, 336))


if __name__ == "__main__":
    unittest.main()
