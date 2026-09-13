"""
Unit tests for RemoteCLIP VQA model.

Tests cover:
- Model loading
- Image preprocessing
- Question parsing
- VQA head initialization
- Prediction interface
- Error handling
"""

import unittest
import sys
import os
import tempfile
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.vqa.remoteclip_vqa import (
        RemoteCLIPVQA,
        VQATask,
        QuestionParser,
        VQAHead,
        create_inference_engine,
    )
    REMOTECLIP_AVAILABLE = True
except ImportError:
    REMOTECLIP_AVAILABLE = False


class TestQuestionParser(unittest.TestCase):
    """Test question parsing logic."""
    
    def test_land_cover_parsing(self):
        """Test parsing of land cover questions."""
        questions = [
            "What type of land cover is visible?",
            "What is the dominant land cover?",
            "Land use classification",
        ]
        for q in questions:
            task = QuestionParser.parse(q)
            self.assertEqual(task, VQATask.LAND_COVER)
    
    def test_water_presence_parsing(self):
        """Test parsing of water presence questions."""
        questions = [
            "Is there water present?",
            "Does this image contain water?",
            "Is there a lake or river?",
        ]
        for q in questions:
            task = QuestionParser.parse(q)
            self.assertEqual(task, VQATask.WATER_PRESENCE)
    
    def test_scene_type_parsing(self):
        """Test parsing of scene type questions."""
        # Test with exact keyword match
        task = QuestionParser.parse("What type of scene is shown?")
        self.assertEqual(task, VQATask.SCENE_TYPE)
    
    def test_urban_rural_parsing(self):
        """Test parsing of urban/rural questions."""
        # Test with exact keyword match
        task = QuestionParser.parse("Is the area urban or rural?")
        self.assertEqual(task, VQATask.URBAN_RURAL)
    
    def test_default_parsing(self):
        """Test default task for unrecognized questions."""
        task = QuestionParser.parse("What is the color of the sky?")
        self.assertEqual(task, VQATask.LAND_COVER)  # Default


class TestVQAHead(unittest.TestCase):
    """Test VQA classification head."""
    
    def test_head_initialization(self):
        """Test VQA head initialization."""
        embedding_dim = 512
        num_classes = 7
        head = VQAHead(embedding_dim, num_classes)
        
        # Check output dimension
        import torch
        dummy_input = torch.randn(2, embedding_dim)
        output = head(dummy_input)
        self.assertEqual(output.shape, (2, num_classes))
    
    def test_head_forward(self):
        """Test forward pass through VQA head."""
        import torch
        embedding_dim = 512
        num_classes = 3
        head = VQAHead(embedding_dim, num_classes)
        
        dummy_embedding = torch.randn(1, embedding_dim)
        logits = head(dummy_embedding)
        
        self.assertEqual(logits.shape, (1, num_classes))


@unittest.skipIf(not REMOTECLIP_AVAILABLE, "RemoteCLIP not available")
class TestRemoteCLIPVQA(unittest.TestCase):
    """Test RemoteCLIP VQA model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image = os.path.join(self.temp_dir, "test.jpg")
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(self.test_image)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_initialization(self):
        """Test model initialization without loading."""
        model = RemoteCLIPVQA(model_name="ViT-B-32", device="cpu")
        self.assertEqual(model.model_name, "RemoteCLIP-ViT-B-32")
        self.assertEqual(model.device, "cpu")
        self.assertFalse(model.is_loaded())
    
    def test_missing_image_error(self):
        """Test error handling for missing image."""
        model = RemoteCLIPVQA(model_name="ViT-B-32", device="cpu")
        model.load()
        
        with self.assertRaises(FileNotFoundError):
            model.predict(
                image_path="/nonexistent/image.jpg",
                question="What is this?"
            )
    
    def test_prediction_structure(self):
        """Test that prediction returns correct structure."""
        model = RemoteCLIPVQA(model_name="ViT-B-32", device="cpu")
        model.load()
        
        result = model.predict(
            image_path=self.test_image,
            question="What type of land cover is visible?"
        )
        
        # Check required fields
        self.assertIn("answer", result)
        self.assertIn("model", result)
        self.assertIn("confidence", result)
        self.assertIn("device", result)
        self.assertIn("inference_time_s", result)
        self.assertIn("task", result)
        self.assertIn("predicted_class", result)
        
        # Check types
        self.assertIsInstance(result["answer"], str)
        self.assertIsInstance(result["model"], str)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["device"], str)
        self.assertIsInstance(result["inference_time_s"], float)
        self.assertIsInstance(result["task"], str)
    
    def test_task_mapping(self):
        """Test that different questions map to different tasks."""
        model = RemoteCLIPVQA(model_name="ViT-B-32", device="cpu")
        model.load()
        
        result1 = model.predict(
            image_path=self.test_image,
            question="What type of land cover is visible?"
        )
        self.assertEqual(result1["task"], "land_cover")
        
        result2 = model.predict(
            image_path=self.test_image,
            question="Is there water present?"
        )
        self.assertEqual(result2["task"], "water_presence")


class TestInterface(unittest.TestCase):
    """Test public interface functions."""
    
    def test_factory_function_exists(self):
        """Test that factory function is callable."""
        if REMOTECLIP_AVAILABLE:
            self.assertTrue(callable(create_inference_engine))
    
    @unittest.skipIf(not REMOTECLIP_AVAILABLE, "RemoteCLIP not available")
    def test_factory_creates_model(self):
        """Test that factory function creates a working model."""
        model = create_inference_engine(model_name="ViT-B-32", device="cpu")
        self.assertIsInstance(model, RemoteCLIPVQA)
        self.assertTrue(model.is_loaded())


if __name__ == "__main__":
    unittest.main()
