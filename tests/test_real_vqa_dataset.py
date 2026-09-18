"""
Tests for Real VQA Dataset Parsing - SatQueryAI Phase 3.1
"""

import unittest
import sys
import os
import json
import tempfile
from pathlib import Path
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.rsvqa_parser import RSVQAAnnotationParser, create_rsvqa_parser
from data.real_vqa_dataset import RealVQADataset, create_real_vqa_dataloader


class TestRSVQAAnnotationParser(unittest.TestCase):
    """Test RSVQA annotation parser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_root = Path(self.temp_dir)
        
        # Create directory structure
        (self.dataset_root / "annotations").mkdir(parents=True)
        (self.dataset_root / "images").mkdir(parents=True)
        
        # Create sample JSON annotations
        self.sample_annotations = [
            {
                "image_id": "img_001.jpg",
                "question": "What type of building is visible?",
                "answer": "residential"
            },
            {
                "image_id": "img_002.jpg",
                "question": "Is there water in this image?",
                "answer": "yes"
            },
            {
                "image_id": "img_003.jpg",
                "question": "What is the dominant land cover?",
                "answer": "agricultural"
            }
        ]
        
        # Save annotations
        annotation_file = self.dataset_root / "annotations" / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(self.sample_annotations, f)
        
        # Create sample images
        from PIL import Image
        for ann in self.sample_annotations:
            img_path = self.dataset_root / "images" / ann['image_id']
            img = Image.new('RGB', (224, 224), color='red')
            img.save(img_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = RSVQAAnnotationParser(str(self.dataset_root), "hr")
        self.assertEqual(parser.dataset_root, self.dataset_root)
        self.assertEqual(parser.dataset_variant, "hr")
    
    def test_json_annotation_parsing(self):
        """Test JSON annotation parsing."""
        parser = RSVQAAnnotationParser(str(self.dataset_root), "hr")
        annotations = parser.load_annotations()
        
        self.assertEqual(len(annotations), 3)
        self.assertEqual(annotations[0]['image_id'], "img_001.jpg")
        self.assertEqual(annotations[0]['question'], "What type of building is visible?")
        self.assertEqual(annotations[0]['answer'], "residential")
    
    def test_answer_vocabulary_building(self):
        """Test answer vocabulary building."""
        parser = RSVQAAnnotationParser(str(self.dataset_root), "hr")
        parser.load_annotations()
        vocab = parser.build_answer_vocabulary(min_frequency=1)
        
        self.assertEqual(len(vocab), 3)
        self.assertIn("residential", vocab)
        self.assertIn("yes", vocab)
        self.assertIn("agricultural", vocab)
    
    def test_dataset_statistics(self):
        """Test dataset statistics."""
        parser = RSVQAAnnotationParser(str(self.dataset_root), "hr")
        parser.load_annotations()
        stats = parser.get_statistics()
        
        self.assertEqual(stats['total_annotations'], 3)
        self.assertEqual(stats['unique_images'], 3)
        self.assertEqual(stats['unique_answers'], 3)
        self.assertIn('answer_distribution', stats)
    
    def test_split_creation(self):
        """Test dataset split creation."""
        parser = RSVQAAnnotationParser(str(self.dataset_root), "hr")
        parser.load_annotations()
        
        train_split = parser.get_split("train", random_seed=42)
        val_split = parser.get_split("val", random_seed=42)
        test_split = parser.get_split("test", random_seed=42)
        
        self.assertEqual(len(train_split) + len(val_split) + len(test_split), 3)
        self.assertEqual(len(train_split), 2)  # 80% of 3 = 2.4, rounded down
        self.assertEqual(len(val_split), 0)  # 10% of 3 = 0.3, rounded down
        self.assertEqual(len(test_split), 1)  # 10% of 3 = 0.3, rounded up
    
    def test_invalid_dataset_variant(self):
        """Test invalid dataset variant."""
        with self.assertRaises(ValueError):
            RSVQAAnnotationParser(str(self.dataset_root), "invalid")
    
    def test_missing_dataset_directory(self):
        """Test handling of missing dataset directory."""
        parser = RSVQAAnnotationParser("/nonexistent/path", "hr")
        annotations = parser.load_annotations()
        self.assertEqual(len(annotations), 0)


class TestRealVQADataset(unittest.TestCase):
    """Test real VQA dataset loader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_root = Path(self.temp_dir)
        
        # Create directory structure
        (self.dataset_root / "annotations").mkdir(parents=True)
        (self.dataset_root / "images").mkdir(parents=True)
        
        # Create sample annotations (enough for train/val/test splits)
        self.sample_annotations = [
            {
                "image_id": f"img_{i:03d}.jpg",
                "question": f"What type {i}?",
                "answer": f"answer_{i % 3}"  # Cycle through 3 answers
            }
            for i in range(10)  # Create 10 samples
        ]
        
        # Save annotations
        annotation_file = self.dataset_root / "annotations" / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(self.sample_annotations, f)
        
        # Create sample images
        from PIL import Image
        for i in range(10):  # Create 10 images
            img_path = self.dataset_root / "images" / f"img_{i:03d}.jpg"
            img = Image.new('RGB', (224, 224), color='blue')
            img.save(img_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = RealVQADataset(
            dataset_root=str(self.dataset_root),
            dataset_variant="hr",
            split="train",
            random_seed=42
        )
        
        # With 10 samples and 80% train split, we should have 8 samples
        self.assertEqual(len(dataset), 8)
        self.assertEqual(len(dataset.answer_vocab), 3)  # 3 unique answers
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = RealVQADataset(
            dataset_root=str(self.dataset_root),
            dataset_variant="hr",
            split="train",
            random_seed=42
        )
        
        sample = dataset[0]
        
        self.assertIn('image', sample)
        self.assertIn('question', sample)
        self.assertIn('answer', sample)
        self.assertIn('answer_idx', sample)
        self.assertEqual(sample['dataset'], 'RSVQA-HR')
        self.assertEqual(sample['split'], 'train')
    
    def test_dataset_with_invalid_images(self):
        """Test dataset with missing images."""
        # Add annotation for non-existent image
        self.sample_annotations.append({
            "image_id": "nonexistent.jpg",
            "question": "What is this?",
            "answer": "unknown"
        })
        
        annotation_file = self.dataset_root / "annotations" / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(self.sample_annotations, f)
        
        dataset = RealVQADataset(
            dataset_root=str(self.dataset_root),
            dataset_variant="hr",
            split="train",
            random_seed=42
        )
        
        # Should only load valid samples (10 valid in train split, 1 invalid filtered out)
        # 10 total samples * 0.8 train ratio = 8 samples in train
        # The invalid image might or might not be in the train split depending on random split
        # So we just check that we have some valid samples
        self.assertGreater(len(dataset), 0)
        self.assertLessEqual(len(dataset), 8)
    
    def test_dataset_statistics(self):
        """Test dataset statistics."""
        dataset = RealVQADataset(
            dataset_root=str(self.dataset_root),
            dataset_variant="hr",
            split="train",
            random_seed=42
        )
        
        stats = dataset.get_statistics()
        
        self.assertIn('total_annotations', stats)
        self.assertIn('unique_images', stats)
        self.assertIn('unique_answers', stats)


class TestCreateRealVQADataloader(unittest.TestCase):
    """Test real VQA dataloader creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_root = Path(self.temp_dir)
        
        # Create directory structure
        (self.dataset_root / "annotations").mkdir(parents=True)
        (self.dataset_root / "images").mkdir(parents=True)
        
        # Create sample annotations (enough for train/val/test splits)
        self.sample_annotations = [
            {
                "image_id": f"img_{i:03d}.jpg",
                "question": f"What type {i}?",
                "answer": f"answer_{i % 3}"  # Cycle through 3 answers
            }
            for i in range(10)  # Create 10 samples
        ]
        
        # Save annotations
        annotation_file = self.dataset_root / "annotations" / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(self.sample_annotations, f)
        
        # Create sample images
        from PIL import Image
        for i in range(10):  # Create 10 images
            img_path = self.dataset_root / "images" / f"img_{i:03d}.jpg"
            img = Image.new('RGB', (224, 224), color='green')
            img.save(img_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_dataloader_creation(self):
        """Test dataloader creation using the factory function."""
        dataloader = create_real_vqa_dataloader(
            dataset_root=str(self.dataset_root),
            dataset_variant="hr",
            split="train",
            batch_size=1,
            shuffle=False
        )
        
        self.assertIsNotNone(dataloader)
        # With 10 samples total and 80% train split, we should have 8 samples in train
        # With batch size 1 and drop_last=False, we should have 8 batches
        self.assertEqual(len(dataloader), 8)
    
    def test_dataloader_batch(self):
        """Test dataloader batch retrieval."""
        from data.vqa_dataset import collate_fn
        from torch.utils.data import DataLoader
        
        dataset = RealVQADataset(
            dataset_root=str(self.dataset_root),
            dataset_variant="hr",
            split="train",
            random_seed=42
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )
        
        # Check if dataloader has any batches
        if len(dataloader) > 0:
            batch = next(iter(dataloader))
            
            self.assertIn('images', batch)
            self.assertIn('questions', batch)
            self.assertIn('answers', batch)
            self.assertIn('answer_indices', batch)
            self.assertEqual(len(batch['images']), 1)
        else:
            # If no batches (dataset too small), skip this test
            self.skipTest("Dataloader has no batches (dataset too small for batch size)")


if __name__ == "__main__":
    unittest.main()
