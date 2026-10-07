"""
Tests for Optical-SAR Fusion Model (Phase 5)

This module contains comprehensive tests for the optical-SAR fusion model,
including:
- Model initialization
- Encoder tests
- Cross-attention tests
- Modality ablation tests (critical)
- Parameter verification
- Inference tests
"""

import pytest
import torch
import tempfile
from pathlib import Path
from PIL import Image

from models.optical_sar import OpticalSARFusion, OpticalSARConfig
from data.optical_sar_dataset import OpticalSARDataset


class TestModelInitialization:
    """Tests for model initialization."""
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        assert model._loaded
        assert model.remoteclip_model is not None
        assert model.sar_encoder is not None
        assert model.optical_projection is not None
        assert model.sar_projection is not None
        assert model.cross_attention is not None
        assert model.fusion_head is not None


class TestEncoders:
    """Tests for optical and SAR encoders."""
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_optical_encoder_frozen(self):
        """Test that optical encoder is frozen."""
        config = OpticalSARConfig(device="cpu", freeze_optical_encoder=True)
        model = OpticalSARFusion(config=config)
        model.load()
        
        for param in model.remoteclip_model.parameters():
            assert not param.requires_grad, "Optical encoder should be frozen"
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_sar_encoder_frozen(self):
        """Test that SAR encoder is frozen."""
        config = OpticalSARConfig(device="cpu", freeze_sar_encoder=True)
        model = OpticalSARFusion(config=config)
        model.load()
        
        for param in model.sar_encoder.parameters():
            assert not param.requires_grad, "SAR encoder should be frozen"
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_optical_encoder_output_dim(self):
        """Test optical encoder output dimension."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        test_image = Image.new('RGB', (224, 224), color=(100, 100, 100))
        features = model._encode_optical(test_image)
        
        assert features.shape == (1, 512), f"Expected (1, 512), got {features.shape}"
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_sar_encoder_output_dim(self):
        """Test SAR encoder output dimension."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        sar_tensor = torch.randn(1, 3, 224, 224)
        features = model._encode_sar(sar_tensor)
        
        assert features.shape == (1, 256), f"Expected (1, 256), got {features.shape}"


class TestProjections:
    """Tests for feature projections."""
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_optical_projection_dim(self):
        """Test optical projection to fusion dimension."""
        config = OpticalSARConfig(device="cpu", fusion_dim=256)
        model = OpticalSARFusion(config=config)
        model.load()
        
        optical_features = torch.randn(1, 512)
        projected = model.optical_projection(optical_features)
        
        assert projected.shape == (1, 256), f"Expected (1, 256), got {projected.shape}"
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_sar_projection_dim(self):
        """Test SAR projection to fusion dimension."""
        config = OpticalSARConfig(device="cpu", fusion_dim=256)
        model = OpticalSARFusion(config=config)
        model.load()
        
        sar_features = torch.randn(1, 256)
        projected = model.sar_projection(sar_features)
        
        assert projected.shape == (1, 256), f"Expected (1, 256), got {projected.shape}"


class TestCrossAttention:
    """Tests for cross-modal attention."""
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_cross_attention_exists(self):
        """Test that cross-attention module exists."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        assert model.cross_attention is not None
        assert hasattr(model.cross_attention, 'cross_attn_1')
        assert hasattr(model.cross_attention, 'cross_attn_2')
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_cross_attention_output_dim(self):
        """Test cross-attention output dimension."""
        config = OpticalSARConfig(device="cpu", fusion_dim=256)
        model = OpticalSARFusion(config=config)
        model.load()
        
        optical_features = torch.randn(1, 256)
        sar_features = torch.randn(1, 256)
        
        fused = model.cross_attention(optical_features, sar_features)
        
        assert fused.shape == (1, 256), f"Expected (1, 256), got {fused.shape}"
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_cross_attention_changes_with_sar(self):
        """Test that changing SAR features changes attention output."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        optical_features = torch.randn(1, 256)
        sar_features_1 = torch.randn(1, 256)
        sar_features_2 = torch.randn(1, 256)
        
        fused_1 = model.cross_attention(optical_features, sar_features_1)
        fused_2 = model.cross_attention(optical_features, sar_features_2)
        
        # Outputs should be different
        assert not torch.allclose(fused_1, fused_2), "Changing SAR should change output"
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_cross_attention_changes_with_optical(self):
        """Test that changing optical features changes attention output."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        optical_features_1 = torch.randn(1, 256)
        optical_features_2 = torch.randn(1, 256)
        sar_features = torch.randn(1, 256)
        
        fused_1 = model.cross_attention(optical_features_1, sar_features)
        fused_2 = model.cross_attention(optical_features_2, sar_features)
        
        # Outputs should be different
        assert not torch.allclose(fused_1, fused_2), "Changing optical should change output"


class TestModalityAblation:
    """Critical modality ablation tests to prove fusion is genuine."""
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_modality_ablation_optical_sar_vs_optical_zero_sar(self):
        """Test that zeroing SAR changes the output."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        optical_features = torch.randn(1, 512)
        sar_features = torch.randn(1, 256)
        
        # Case A: Optical + SAR
        optical_proj = model.optical_projection(optical_features)
        sar_proj = model.sar_projection(sar_features)
        fused_normal = model.cross_attention(optical_proj, sar_proj)
        
        # Case B: Optical + zeroed SAR
        sar_proj_zeroed = torch.zeros_like(sar_proj)
        fused_zeroed = model.cross_attention(optical_proj, sar_proj_zeroed)
        
        # Outputs should be different
        assert not torch.allclose(fused_normal, fused_zeroed), "Zeroing SAR should change output"
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_modality_ablation_optical_sar_vs_zero_optical_sar(self):
        """Test that zeroing optical changes the output."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        optical_features = torch.randn(1, 512)
        sar_features = torch.randn(1, 256)
        
        # Case A: Optical + SAR
        optical_proj = model.optical_projection(optical_features)
        sar_proj = model.sar_projection(sar_features)
        fused_normal = model.cross_attention(optical_proj, sar_proj)
        
        # Case C: Zeroed optical + SAR
        optical_proj_zeroed = torch.zeros_like(optical_proj)
        fused_zeroed = model.cross_attention(optical_proj_zeroed, sar_proj)
        
        # Outputs should be different
        assert not torch.allclose(fused_normal, fused_zeroed), "Zeroing optical should change output"
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_modality_ablation_optical_sar_vs_optical_shuffled_sar(self):
        """Test that shuffling SAR changes the output."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        optical_features = torch.randn(1, 512)
        sar_features = torch.randn(1, 256)
        
        # Case A: Optical + SAR
        optical_proj = model.optical_projection(optical_features)
        sar_proj = model.sar_projection(sar_features)
        fused_normal = model.cross_attention(optical_proj, sar_proj)
        
        # Case D: Optical + shuffled SAR
        sar_proj_shuffled = sar_proj[:, torch.randperm(sar_proj.size(1))]
        fused_shuffled = model.cross_attention(optical_proj, sar_proj_shuffled)
        
        # Outputs should be different
        assert not torch.allclose(fused_normal, fused_shuffled), "Shuffling SAR should change output"


class TestParameterVerification:
    """Tests for parameter counting and trainability."""
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_trainable_parameter_count(self):
        """Test trainable parameter count."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        trainable_params = sum(p.numel() for p in model.get_trainable_params())
        
        assert trainable_params > 0, "Should have trainable parameters"
        assert trainable_params < 1_000_000, "Trainable params should be lightweight (<1M)"
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_frozen_parameter_count(self):
        """Test frozen parameter count."""
        config = OpticalSARConfig(device="cpu", freeze_optical_encoder=True, freeze_sar_encoder=True)
        model = OpticalSARFusion(config=config)
        model.load()
        
        frozen_params = sum(p.numel() for p in model.remoteclip_model.parameters()) + \
                       sum(p.numel() for p in model.sar_encoder.parameters())
        
        assert frozen_params > 0, "Should have frozen parameters"
        assert frozen_params > 100_000_000, "Frozen params should include large encoders"


class TestInference:
    """Tests for inference."""
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_inference_missing_optical(self):
        """Test inference with missing optical image."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        result = model.predict(
            optical_path="nonexistent_optical.jpg",
            sar_path="nonexistent_sar.jpg"
        )
        
        assert result['status'] == 'error'
        assert 'error' in result
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_inference_with_real_images(self):
        """Test inference with real images."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        # Create test images
        optical_image = Image.new('RGB', (224, 224), color=(100, 100, 100))
        sar_image = Image.new('RGB', (224, 224), color=(150, 150, 150))
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f2:
            optical_image.save(f1.name)
            sar_image.save(f2.name)
            
            result = model.predict(optical_path=f1.name, sar_path=f2.name)
            
            assert result['status'] == 'success'
            assert 'prediction' in result
            assert 'confidence' in result
            assert 'execution_time_ms' in result
            
            Path(f1.name).unlink()
            Path(f2.name).unlink()


class TestDataset:
    """Tests for optical-SAR dataset."""
    
    def test_synthetic_dataset_loading(self):
        """Test synthetic dataset loading."""
        dataset = OpticalSARDataset(
            root="synthetic",
            split="train",
            subset_size=10,
            synthetic=True
        )
        
        assert len(dataset) == 10
        assert dataset._loaded
    
    def test_synthetic_sample_structure(self):
        """Test synthetic sample structure."""
        dataset = OpticalSARDataset(
            root="synthetic",
            split="train",
            subset_size=5,
            synthetic=True
        )
        
        sample = dataset[0]
        
        assert 'sample_id' in sample
        assert 'optical' in sample
        assert 'sar' in sample
        assert 'label' in sample
        assert 'metadata' in sample


class TestCheckpoint:
    """Tests for checkpoint save/load."""
    
    @pytest.mark.skipif(not pytest.importorskip("open_clip", reason="open-clip not available"), reason="Requires open-clip")
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        config = OpticalSARConfig(device="cpu")
        model = OpticalSARFusion(config=config)
        model.load()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            
            # Save checkpoint
            model.save_checkpoint(
                str(checkpoint_path),
                epoch=1,
                metrics={'train_loss': 0.5, 'train_accuracy': 0.8}
            )
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            model2 = OpticalSARFusion(config=config, checkpoint_path=str(checkpoint_path))
            model2.load()
            
            assert model2._loaded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
