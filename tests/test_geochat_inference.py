"""
Unit tests for GeoChat inference interface.

These tests cover error handling and basic functionality of the GeoChatInference class.
Note: Actual model inference tests require GPU and are not included in unit tests.
"""

import unittest
import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock
import PIL.Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only when GeoChat is available
try:
    from models.vqa.geochat.inference import GeoChatInference, create_inference_engine, GEOCHAT_AVAILABLE
    GEOCHAT_IMPORTED = True
except ImportError:
    GEOCHAT_IMPORTED = False
    GEOCHAT_AVAILABLE = False


class TestGeoChatInference(unittest.TestCase):
    """Test cases for GeoChatInference class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.valid_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        
        # Create a valid test image
        test_image = PIL.Image.new('RGB', (100, 100), color='red')
        test_image.save(self.valid_image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_import_error_when_geochat_not_available(self):
        """Test that ImportError is raised when GeoChat is not installed."""
        if not GEOCHAT_IMPORTED:
            # If GeoChat couldn't be imported, that's expected
            self.assertTrue(True)
            return
        
        # Test with mocked unavailable flag
        with patch('models.vqa.geochat.inference.GEOCHAT_AVAILABLE', False):
            with self.assertRaises(ImportError) as context:
                GeoChatInference(model_path="test")
            
            self.assertIn("GeoChat package is not installed", str(context.exception))
    
    @unittest.skipIf(not GEOCHAT_AVAILABLE, "GeoChat not installed - requires full installation")
    def test_missing_image_raises_file_not_found(self):
        """Test that FileNotFoundError is raised for missing image."""
        # Skip this test if GeoChat is not available
        if not GEOCHAT_AVAILABLE:
            self.skipTest("GeoChat not installed")
        
        # Mock the model loading
        with patch('models.vqa.geochat.inference.load_pretrained_model') as mock_load_model, \
             patch('models.vqa.geochat.inference.get_model_name_from_path') as mock_get_model_name:
            
            mock_get_model_name.return_value = "geochat-7b"
            mock_load_model.return_value = (Mock(), Mock(), Mock(), 2048)
            
            engine = GeoChatInference(model_path="test")
            
            with self.assertRaises(FileNotFoundError) as context:
                engine.predict(
                    image_path="/nonexistent/path/image.jpg",
                    question="What is this?"
                )
            
            self.assertIn("Image file not found", str(context.exception))
    
    @unittest.skipIf(not GEOCHAT_AVAILABLE, "GeoChat not installed - requires full installation")
    def test_invalid_image_raises_value_error(self):
        """Test that ValueError is raised for invalid image file."""
        if not GEOCHAT_AVAILABLE:
            self.skipTest("GeoChat not installed")
        
        # Create an invalid image file
        invalid_image_path = os.path.join(self.temp_dir, "invalid.jpg")
        with open(invalid_image_path, 'w') as f:
            f.write("This is not an image")
        
        # Mock the model loading
        with patch('models.vqa.geochat.inference.load_pretrained_model') as mock_load_model, \
             patch('models.vqa.geochat.inference.get_model_name_from_path') as mock_get_model_name:
            
            mock_get_model_name.return_value = "geochat-7b"
            mock_load_model.return_value = (Mock(), Mock(), Mock(), 2048)
            
            engine = GeoChatInference(model_path="test")
            
            with self.assertRaises(ValueError) as context:
                engine.predict(
                    image_path=invalid_image_path,
                    question="What is this?"
                )
            
            self.assertIn("Failed to load image", str(context.exception))
    
    @unittest.skipIf(not GEOCHAT_AVAILABLE, "GeoChat not installed - requires full installation")
    def test_missing_question_raises_value_error(self):
        """Test that ValueError is raised for empty question."""
        if not GEOCHAT_AVAILABLE:
            self.skipTest("GeoChat not installed")
        
        # Mock the model loading
        with patch('models.vqa.geochat.inference.load_pretrained_model') as mock_load_model, \
             patch('models.vqa.geochat.inference.get_model_name_from_path') as mock_get_model_name:
            
            mock_get_model_name.return_value = "geochat-7b"
            mock_load_model.return_value = (Mock(), Mock(), Mock(), 2048)
            
            engine = GeoChatInference(model_path="test")
            
            # Test empty string
            with self.assertRaises(ValueError) as context:
                engine.predict(
                    image_path=self.valid_image_path,
                    question=""
                )
            self.assertIn("Question cannot be empty", str(context.exception))
            
            # Test whitespace-only string
            with self.assertRaises(ValueError) as context:
                engine.predict(
                    image_path=self.valid_image_path,
                    question="   "
                )
            self.assertIn("Question cannot be empty", str(context.exception))
    
    @unittest.skipIf(not GEOCHAT_AVAILABLE, "GeoChat not installed - requires full installation")
    def test_successful_inference_returns_correct_format(self):
        """Test that successful inference returns correct dictionary format."""
        if not GEOCHAT_AVAILABLE:
            self.skipTest("GeoChat not installed")
        
        # Mock the model loading
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "USER: <image>\nWhat is this?\nASSISTANT: This is a test response"
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_image_processor = Mock()
        mock_image_processor.preprocess.return_value = {'pixel_values': [Mock()]}
        
        with patch('models.vqa.geochat.inference.load_pretrained_model') as mock_load_model, \
             patch('models.vqa.geochat.inference.get_model_name_from_path') as mock_get_model_name, \
             patch('models.vqa.geochat.inference.Chat') as mock_chat_class:
            
            mock_get_model_name.return_value = "geochat-7b"
            mock_load_model.return_value = (mock_tokenizer, mock_model, mock_image_processor, 2048)
            
            mock_chat = Mock()
            mock_chat.generate.return_value = Mock()
            mock_chat_class.return_value = mock_chat
            
            engine = GeoChatInference(model_path="test")
            
            result = engine.predict(
                image_path=self.valid_image_path,
                question="What is this?"
            )
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn('answer', result)
            self.assertIn('model', result)
            self.assertIn('confidence', result)
            
            # Verify values
            self.assertEqual(result['model'], 'GeoChat')
            self.assertIsNone(result['confidence'])
            self.assertIsInstance(result['answer'], str)
    
    @unittest.skipIf(not GEOCHAT_AVAILABLE, "GeoChat not installed - requires full installation")
    def test_call_method_alias(self):
        """Test that __call__ method works as alias for predict."""
        if not GEOCHAT_AVAILABLE:
            self.skipTest("GeoChat not installed")
        
        # Mock the model loading
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "USER: <image>\nWhat is this?\nASSISTANT: Test response"
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_image_processor = Mock()
        mock_image_processor.preprocess.return_value = {'pixel_values': [Mock()]}
        
        with patch('models.vqa.geochat.inference.load_pretrained_model') as mock_load_model, \
             patch('models.vqa.geochat.inference.get_model_name_from_path') as mock_get_model_name, \
             patch('models.vqa.geochat.inference.Chat') as mock_chat_class:
            
            mock_get_model_name.return_value = "geochat-7b"
            mock_load_model.return_value = (mock_tokenizer, mock_model, mock_image_processor, 2048)
            
            mock_chat = Mock()
            mock_chat.generate.return_value = Mock()
            mock_chat_class.return_value = mock_chat
            
            engine = GeoChatInference(model_path="test")
            
            # Test __call__ method
            result = engine(
                image_path=self.valid_image_path,
                question="What is this?"
            )
            
            # Verify it returns same format as predict
            self.assertIsInstance(result, dict)
            self.assertIn('answer', result)
            self.assertEqual(result['model'], 'GeoChat')


class TestCreateInferenceEngine(unittest.TestCase):
    """Test cases for create_inference_engine factory function."""
    
    @unittest.skipIf(not GEOCHAT_IMPORTED, "GeoChat not installed - requires full installation")
    def test_create_inference_engine_default_params(self):
        """Test factory function with default parameters."""
        if not GEOCHAT_IMPORTED:
            self.skipTest("GeoChat not installed")
        
        with patch('models.vqa.geochat.inference.GeoChatInference') as mock_geochat_class:
            mock_instance = Mock()
            mock_geochat_class.return_value = mock_instance
            
            result = create_inference_engine()
            
            mock_geochat_class.assert_called_once_with(
                model_path="MBZUAI/geochat-7B",
                device="cuda",
                load_8bit=False
            )
            self.assertEqual(result, mock_instance)
    
    @unittest.skipIf(not GEOCHAT_IMPORTED, "GeoChat not installed - requires full installation")
    def test_create_inference_engine_custom_params(self):
        """Test factory function with custom parameters."""
        if not GEOCHAT_IMPORTED:
            self.skipTest("GeoChat not installed")
        
        with patch('models.vqa.geochat.inference.GeoChatInference') as mock_geochat_class:
            mock_instance = Mock()
            mock_geochat_class.return_value = mock_instance
            
            result = create_inference_engine(
                model_path="custom/path",
                device="cpu",
                load_8bit=True
            )
            
            mock_geochat_class.assert_called_once_with(
                model_path="custom/path",
                device="cpu",
                load_8bit=True
            )
            self.assertEqual(result, mock_instance)


if __name__ == '__main__':
    unittest.main()
