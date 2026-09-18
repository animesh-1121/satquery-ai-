"""
Tests for Multimodal VQA Model - SatQueryAI Phase 3
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqa.multimodal_vqa.multimodal_vqa import MultimodalVQA, VQAConfig, create_multimodal_vqa, MultimodalFusion, VQAHead


class TestVQAConfig(unittest.TestCase):
    """Test VQA configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = VQAConfig()
        self.assertEqual(config.remoteclip_model, "ViT-B-32")
        self.assertEqual(config.embedding_dim, 512)
        self.assertEqual(config.fusion_dim, 256)
        self.assertEqual(config.freeze_remoteclip, True)
        self.assertEqual(config.batch_size, 1)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = VQAConfig(
            remoteclip_model="ViT-B-16",
            embedding_dim=768,
            fusion_dim=512,
            learning_rate=5e-5
        )
        self.assertEqual(config.remoteclip_model, "ViT-B-16")
        self.assertEqual(config.embedding_dim, 768)
        self.assertEqual(config.fusion_dim, 512)
        self.assertEqual(config.learning_rate, 5e-5)


class TestMultimodalFusion(unittest.TestCase):
    """Test multimodal fusion module."""
    
    def test_fusion_initialization(self):
        """Test fusion module initialization."""
        fusion = MultimodalFusion(
            visual_dim=512,
            text_dim=512,
            fusion_dim=256,
            use_attention=False
        )
        self.assertIsInstance(fusion, MultimodalFusion)
        self.assertEqual(fusion.use_attention, False)
    
    def test_fusion_forward(self):
        """Test fusion forward pass."""
        fusion = MultimodalFusion(
            visual_dim=512,
            text_dim=512,
            fusion_dim=256,
            use_attention=False
        )
        
        # Create dummy inputs
        visual_features = torch.randn(2, 512)
        text_features = torch.randn(2, 512)
        
        # Forward pass
        fused = fusion(visual_features, text_features)
        
        # Check output shape
        self.assertEqual(fused.shape, (2, 256))
    
    def test_fusion_with_attention(self):
        """Test fusion with attention."""
        fusion = MultimodalFusion(
            visual_dim=512,
            text_dim=512,
            fusion_dim=256,
            use_attention=True
        )
        
        # Create dummy inputs
        visual_features = torch.randn(2, 512)
        text_features = torch.randn(2, 512)
        
        # Forward pass
        fused = fusion(visual_features, text_features)
        
        # Check output shape
        self.assertEqual(fused.shape, (2, 256))


class TestVQAHead(unittest.TestCase):
    """Test VQA head module."""
    
    def test_head_initialization(self):
        """Test VQA head initialization."""
        head = VQAHead(
            fusion_dim=256,
            num_answers=10,
            hidden_dim=512
        )
        self.assertIsInstance(head, VQAHead)
    
    def test_head_forward(self):
        """Test VQA head forward pass."""
        head = VQAHead(
            fusion_dim=256,
            num_answers=10,
            hidden_dim=512
        )
        
        # Create dummy input
        fused_features = torch.randn(4, 256)
        
        # Forward pass
        logits = head(fused_features)
        
        # Check output shape
        self.assertEqual(logits.shape, (4, 10))


class TestMultimodalVQA(unittest.TestCase):
    """Test multimodal VQA model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.answer_vocab = [
            "agricultural",
            "forest", 
            "urban",
            "water",
            "barren",
            "grassland",
            "industrial",
            "residential",
            "wetland",
            "mixed"
        ]
        
        self.config = VQAConfig(
            remoteclip_model="ViT-B-32",
            embedding_dim=512,
            fusion_dim=256,
            num_answers=len(self.answer_vocab),
            device="cpu",  # Use CPU for testing
            freeze_remoteclip=True,
            mixed_precision=False
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = MultimodalVQA(
            config=self.config,
            answer_vocab=self.answer_vocab
        )
        self.assertIsInstance(model, MultimodalVQA)
        self.assertEqual(model.num_answers, len(self.answer_vocab))
        self.assertFalse(model._loaded)
    
    def test_model_loading(self):
        """Test model loading."""
        model = MultimodalVQA(
            config=self.config,
            answer_vocab=self.answer_vocab
        )
        model.load()
        self.assertTrue(model._loaded)
        self.assertIsNotNone(model.remoteclip_model)
        self.assertIsNotNone(model.fusion)
        self.assertIsNotNone(model.vqa_head)
    
    def test_answer_vocab_mapping(self):
        """Test answer vocabulary mapping."""
        model = MultimodalVQA(
            config=self.config,
            answer_vocab=self.answer_vocab
        )
        
        # Check answer to index mapping
        self.assertEqual(model.answer_to_idx["agricultural"], 0)
        self.assertEqual(model.answer_to_idx["mixed"], 9)
        
        # Check index to answer mapping
        self.assertEqual(model.idx_to_answer[0], "agricultural")
        self.assertEqual(model.idx_to_answer[9], "mixed")
    
    def test_get_trainable_params(self):
        """Test getting trainable parameters."""
        model = MultimodalVQA(
            config=self.config,
            answer_vocab=self.answer_vocab
        )
        model.load()
        
        # Get trainable parameters
        trainable_params = model.get_trainable_params()
        
        # Should have parameters from fusion and VQA head only
        self.assertGreater(len(trainable_params), 0)
        
        # RemoteCLIP parameters should be frozen
        for param in model.remoteclip_model.parameters():
            self.assertFalse(param.requires_grad)
    
    def test_predict_with_invalid_image(self):
        """Test prediction with invalid image path."""
        model = MultimodalVQA(
            config=self.config,
            answer_vocab=self.answer_vocab
        )
        model.load()
        
        # Test with non-existent image
        result = model.predict(
            image_path="nonexistent.jpg",
            question="What is this?"
        )
        
        # Should return error status
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["answer"])
        self.assertIsNotNone(result["error"])
    
    def test_predict_with_empty_question(self):
        """Test prediction with empty question."""
        model = MultimodalVQA(
            config=self.config,
            answer_vocab=self.answer_vocab
        )
        model.load()
        
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
            # Create a minimal valid image
            from PIL import Image
            img = Image.new('RGB', (224, 224), color='red')
            img.save(temp_path)
        
        try:
            # Test with empty question
            result = model.predict(
                image_path=temp_path,
                question=""
            )
            
            # Should return error status
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["answer"])
            self.assertIsNotNone(result["error"])
        finally:
            # Clean up
            os.unlink(temp_path)


class TestCreateMultimodalVQA(unittest.TestCase):
    """Test factory function for creating multimodal VQA."""
    
    def test_factory_function(self):
        """Test factory function."""
        answer_vocab = ["answer1", "answer2", "answer3"]
        
        model = create_multimodal_vqa(
            answer_vocab=answer_vocab,
            config=VQAConfig(
                device="cpu",
                freeze_remoteclip=True
            )
        )
        
        self.assertIsInstance(model, MultimodalVQA)
        self.assertTrue(model._loaded)


if __name__ == "__main__":
    unittest.main()
