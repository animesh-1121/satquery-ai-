"""
Unit tests for SatQueryAI data layer.

Tests cover:
- Dataset adapters
- Validation utilities
- Compatibility checkers
- Preprocessing pipeline
- Manifest functionality
"""

import unittest
import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.base import BaseRemoteSensingDataset, Sample, Modality
from data.validation import DataValidator, ValidationResult
from data.compatibility import PairCompatibilityChecker, CompatibilityResult
from data.preprocessing import ImagePreprocessor, PreprocessingConfig
from data.manifest import DatasetManifest
from data.splits import DataSplitter, create_train_val_test_split
from data.remoteclip_preprocessing import RemoteCLIPPreprocessor
from data.adapters import BigEarthNetDataset, VRSBenchDataset, RSVQADataset, CDVQADataset


class TestSample(unittest.TestCase):
    """Test Sample dataclass."""
    
    def test_sample_creation(self):
        """Test creating a sample."""
        sample = Sample(
            sample_id="test_001",
            dataset="TestDataset",
            image_path="/path/to/image.jpg",
            modality=Modality.OPTICAL
        )
        
        self.assertEqual(sample.sample_id, "test_001")
        self.assertEqual(sample.dataset, "TestDataset")
        self.assertEqual(sample.modality, Modality.OPTICAL)
    
    def test_sample_to_dict(self):
        """Test converting sample to dictionary."""
        sample = Sample(
            sample_id="test_001",
            dataset="TestDataset",
            image_path="/path/to/image.jpg",
            question="What is this?",
            answer="Test answer",
            modality=Modality.OPTICAL
        )
        
        result = sample.to_dict()
        
        self.assertEqual(result["sample_id"], "test_001")
        self.assertEqual(result["dataset"], "TestDataset")
        self.assertEqual(result["image_path"], "/path/to/image.jpg")
        self.assertEqual(result["question"], "What is this?")
        self.assertEqual(result["answer"], "Test answer")
        self.assertEqual(result["modality"], "optical")


class TestDataValidator(unittest.TestCase):
    """Test data validation utilities."""
    
    def test_validate_missing_file(self):
        """Test validation of missing file."""
        result = DataValidator.validate_image_path("/nonexistent/file.jpg")
        
        self.assertFalse(result.valid)
        self.assertEqual(result.total_samples, 1)
        self.assertEqual(result.valid_samples, 0)
        self.assertEqual(result.invalid_samples, 1)
        self.assertTrue(any("not found" in error for error in result.errors))
    
    def test_validate_empty_file(self):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        
        try:
            result = DataValidator.validate_image_path(temp_path)
            
            self.assertFalse(result.valid)
            self.assertTrue(any("empty" in error for error in result.errors))
        finally:
            os.unlink(temp_path)
    
    def test_validate_sample_missing_image(self):
        """Test validation of sample with missing image."""
        sample = Sample(
            sample_id="test_001",
            dataset="TestDataset",
            image_path="/nonexistent/image.jpg"
        )
        
        result = DataValidator.validate_sample(sample)
        
        self.assertFalse(result.valid)
        # The error will be "File not found" from the image path validation
        self.assertTrue(any("not found" in error.lower() for error in result.errors))


class TestPairCompatibilityChecker(unittest.TestCase):
    """Test pair compatibility checker."""
    
    def test_temporal_pair_missing_files(self):
        """Test temporal pair with missing files."""
        checker = PairCompatibilityChecker()
        
        result = checker.check_temporal_pair("/nonexistent/t1.jpg", "/nonexistent/t2.jpg")
        
        self.assertFalse(result.compatible)
        self.assertTrue(any("not found" in error for error in result.errors))
    
    def test_optical_sar_pair_missing_files(self):
        """Test optical-SAR pair with missing files."""
        checker = PairCompatibilityChecker()
        
        result = checker.check_optical_sar_pair("/nonexistent/optical.jpg", "/nonexistent/sar.jpg")
        
        self.assertFalse(result.compatible)
        self.assertTrue(any("not found" in error for error in result.errors))


class TestImagePreprocessor(unittest.TestCase):
    """Test image preprocessing."""
    
    def test_config_creation(self):
        """Test creating preprocessing config."""
        config = PreprocessingConfig(
            target_size=(224, 224),
            normalize=True
        )
        
        self.assertEqual(config.target_size, (224, 224))
        self.assertTrue(config.normalize)
    
    def test_preprocessor_creation(self):
        """Test creating preprocessor."""
        config = PreprocessingConfig()
        preprocessor = ImagePreprocessor(config)
        
        self.assertIsNotNone(preprocessor.config)


class TestDatasetManifest(unittest.TestCase):
    """Test dataset manifest."""
    
    def test_manifest_creation(self):
        """Test creating manifest."""
        manifest = DatasetManifest()
        
        self.assertEqual(len(manifest.samples), 0)
        self.assertIsNotNone(manifest.metadata)
    
    def test_add_sample(self):
        """Test adding sample to manifest."""
        manifest = DatasetManifest()
        sample = Sample(
            sample_id="test_001",
            dataset="TestDataset",
            image_path="/path/to/image.jpg"
        )
        
        manifest.add_sample(sample)
        
        self.assertEqual(len(manifest.samples), 1)
        self.assertIn("test_001", manifest.samples)
    
    def test_get_sample(self):
        """Test getting sample from manifest."""
        manifest = DatasetManifest()
        sample = Sample(
            sample_id="test_001",
            dataset="TestDataset",
            image_path="/path/to/image.jpg"
        )
        
        manifest.add_sample(sample)
        retrieved = manifest.get_sample("test_001")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["sample_id"], "test_001")
    
    def test_get_statistics(self):
        """Test getting manifest statistics."""
        manifest = DatasetManifest()
        
        # Add samples
        for i in range(3):
            sample = Sample(
                sample_id=f"test_{i:03d}",
                dataset="TestDataset",
                image_path=f"/path/to/image_{i}.jpg"
            )
            manifest.add_sample(sample)
        
        stats = manifest.get_statistics()
        
        self.assertEqual(stats["total_samples"], 3)
        self.assertEqual(stats["samples_per_dataset"]["TestDataset"], 3)


class TestBigEarthNetDataset(unittest.TestCase):
    """Test BigEarthNet dataset adapter."""
    
    def test_dataset_creation(self):
        """Test creating BigEarthNet dataset."""
        dataset = BigEarthNetDataset(
            root="/fake/path",
            subset_size=10,
            enabled=True,
            split="s2"  # Specify Sentinel-2 for optical
        )
        
        self.assertEqual(dataset.dataset_name, "BigEarthNet-S2")
        self.assertEqual(dataset.subset_size, 10)
        self.assertFalse(dataset.supports_vqa)
        self.assertFalse(dataset.supports_change_detection)
    
    def test_load_nonexistent_root(self):
        """Test loading dataset with nonexistent root."""
        dataset = BigEarthNetDataset(
            root="/nonexistent/path",
            enabled=True
        )
        
        dataset.load()
        
        # Should load without error but with 0 samples
        self.assertTrue(dataset.is_loaded())
        self.assertEqual(len(dataset), 0)


class TestVRSBenchDataset(unittest.TestCase):
    """Test VRSBench dataset adapter."""
    
    def test_dataset_creation(self):
        """Test creating VRSBench dataset."""
        dataset = VRSBenchDataset(
            root="/fake/path",
            subset_size=10,
            enabled=True
        )
        
        self.assertEqual(dataset.dataset_name, "VRSBench")
        self.assertTrue(dataset.supports_vqa)
        self.assertTrue(dataset.supports_captioning)
        self.assertTrue(dataset.supports_grounding)


class TestRSVQADataset(unittest.TestCase):
    """Test RSVQA dataset adapter."""
    
    def test_dataset_creation(self):
        """Test creating RSVQA dataset."""
        dataset = RSVQADataset(
            root="/fake/path",
            subset_size=10,
            enabled=True
        )
        
        self.assertEqual(dataset.dataset_name, "RSVQA")
        self.assertTrue(dataset.supports_vqa)
        self.assertFalse(dataset.supports_captioning)


class TestCDVQADataset(unittest.TestCase):
    """Test CDVQA dataset adapter."""
    
    def test_dataset_creation(self):
        """Test creating CDVQA dataset."""
        dataset = CDVQADataset(
            root="/fake/path",
            subset_size=10,
            enabled=True
        )
        
        self.assertEqual(dataset.dataset_name, "CDVQA")
        self.assertTrue(dataset.supports_vqa)
        self.assertTrue(dataset.supports_change_detection)


class TestDataSplitter(unittest.TestCase):
    """Test data splitting functionality."""
    
    def test_splitter_creation(self):
        """Test creating data splitter."""
        splitter = DataSplitter(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42
        )
        
        self.assertEqual(splitter.train_ratio, 0.8)
        self.assertEqual(splitter.val_ratio, 0.1)
        self.assertEqual(splitter.test_ratio, 0.1)
    
    def test_random_split(self):
        """Test creating random split."""
        splitter = DataSplitter(random_seed=42)
        
        # Create fake samples
        samples = [
            Sample(sample_id=f"sample_{i}", dataset="TestDataset")
            for i in range(100)
        ]
        
        train, val, test = splitter.split_samples(samples, "TestDataset")
        
        self.assertEqual(len(train) + len(val) + len(test), 100)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(test), 10)
    
    def test_invalid_ratios(self):
        """Test that invalid ratios raise error."""
        with self.assertRaises(ValueError):
            DataSplitter(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
    
    def test_reproducibility(self):
        """Test that splits are reproducible with same seed."""
        samples = [
            Sample(sample_id=f"sample_{i}", dataset="TestDataset")
            for i in range(50)
        ]
        
        # Create two splitters with same seed
        splitter1 = DataSplitter(random_seed=42)
        splitter2 = DataSplitter(random_seed=42)
        
        train1, val1, test1 = splitter1.split_samples(samples, "TestDataset")
        train2, val2, test2 = splitter2.split_samples(samples, "TestDataset")
        
        # Sample IDs should be identical
        train_ids1 = [s.sample_id for s in train1]
        train_ids2 = [s.sample_id for s in train2]
        self.assertEqual(train_ids1, train_ids2)


class TestRemoteCLIPPreprocessor(unittest.TestCase):
    """Test RemoteCLIP preprocessing."""
    
    def test_preprocessor_creation(self):
        """Test creating RemoteCLIP preprocessor."""
        preprocessor = RemoteCLIPPreprocessor()
        
        self.assertIsNotNone(preprocessor.preprocessor)
        self.assertEqual(preprocessor.preprocessor.config.target_size, (224, 224))
    
    def test_custom_target_size(self):
        """Test RemoteCLIP preprocessor with custom target size."""
        preprocessor = RemoteCLIPPreprocessor(target_size=(336, 336))
        
        self.assertEqual(preprocessor.preprocessor.config.target_size, (336, 336))


class TestImageTiling(unittest.TestCase):
    """Test image tiling functionality."""
    
    def test_tiling_config(self):
        """Test tiling configuration."""
        config = PreprocessingConfig(
            tile_size=(256, 256),
            overlap=32,
            enable_tiling=True
        )
        
        self.assertTrue(config.enable_tiling)
        self.assertEqual(config.tile_size, (256, 256))
        self.assertEqual(config.overlap, 32)
    
    def test_tiling_disabled_by_default(self):
        """Test that tiling is disabled by default."""
        config = PreprocessingConfig()
        
        self.assertFalse(config.enable_tiling)
        self.assertIsNone(config.tile_size)


if __name__ == "__main__":
    unittest.main()
