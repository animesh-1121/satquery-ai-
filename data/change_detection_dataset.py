"""
Change Detection Dataset for SatQueryAI Phase 4

This module provides a dataset loader for bi-temporal change detection data.
It handles T1/T2 image pairs with validation and compatibility checking.

Dataset Format:
    root/
        t1/
            image_001_t1.jpg
            ...
        t2/
            image_001_t2.jpg
            ...
        annotations/
            change_labels.json  # Optional: binary change labels
            change_masks/       # Optional: dense change masks
"""

import json
import random
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ChangeDetectionSample:
    """A single change detection sample with T1/T2 pair."""
    
    def __init__(
        self,
        sample_id: str,
        image_t1_path: str,
        image_t2_path: str,
        change_label: Optional[int] = None,  # 0 = no change, 1 = change
        change_mask_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.sample_id = sample_id
        self.image_t1_path = image_t1_path
        self.image_t2_path = image_t2_path
        self.change_label = change_label
        self.change_mask_path = change_mask_path
        self.metadata = metadata or {}


class ChangeDetectionDataset(Dataset):
    """
    Dataset for bi-temporal change detection.
    
    Handles T1/T2 image pairs with validation and compatibility checking.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        subset_size: Optional[int] = None,
        random_seed: int = 42,
        synthetic: bool = False
    ):
        """
        Initialize change detection dataset.
        
        Args:
            root: Root directory of dataset
            split: One of 'train', 'val', 'test'
            subset_size: Optional limit on number of samples
            random_seed: Random seed for reproducible splits
            synthetic: If True, use synthetic data for testing
        """
        self.root = Path(root)
        self.split = split
        self.subset_size = subset_size
        self.random_seed = random_seed
        self.synthetic = synthetic
        
        self.samples: List[ChangeDetectionSample] = []
        self._loaded = False
        
        self.load()
    
    def load(self) -> None:
        """Load dataset samples."""
        if self.synthetic:
            self._load_synthetic()
        else:
            self._load_real()
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} change detection samples for {self.split} split")
    
    def _load_synthetic(self) -> None:
        """Load synthetic data for testing."""
        # Create synthetic samples for smoke testing
        self.samples = []
        for i in range(10):
            sample = ChangeDetectionSample(
                sample_id=f"synthetic_{i:05d}",
                image_t1_path="",  # Will be generated on-the-fly
                image_t2_path="",  # Will be generated on-the-fly
                change_label=i % 2,  # Alternate between change and no-change
                metadata={"synthetic": True}
            )
            self.samples.append(sample)
    
    def _load_real(self) -> None:
        """Load real dataset from disk."""
        # Check if root exists
        if not self.root.exists():
            logger.warning(f"Dataset root not found: {self.root}")
            return
        
        # Scan for T1 and T2 images
        t1_dir = self.root / "t1"
        t2_dir = self.root / "t2"
        
        if not t1_dir.exists() or not t2_dir.exists():
            logger.warning(f"T1 or T2 directory not found: {t1_dir}, {t2_dir}")
            return
        
        t1_files = list(t1_dir.glob("*.jpg")) + list(t1_dir.glob("*.png"))
        
        # Create samples with bi-temporal pairs
        self.samples = []
        
        for idx, t1_file in enumerate(t1_files):
            # Look for corresponding T2 file
            base_name = t1_file.stem.replace("_t1", "")
            t2_file = t2_dir / f"{base_name}_t2{t1_file.suffix}"
            
            if not t2_file.exists():
                # Try alternative naming
                t2_file = t2_dir / t1_file.name.replace("_t1", "_t2")
            
            if not t2_file.exists():
                logger.warning(f"T2 file not found for {t1_file.name}")
                continue
            
            # Try to load change label from annotations
            change_label = None
            annotations_file = self.root / "annotations" / "change_labels.json"
            if annotations_file.exists():
                try:
                    with open(annotations_file, 'r') as f:
                        annotations = json.load(f)
                        change_label = annotations.get(base_name, {}).get("change_label")
                except Exception as e:
                    logger.warning(f"Failed to load annotations: {e}")
            
            sample = ChangeDetectionSample(
                sample_id=f"cd_{idx:05d}",
                image_t1_path=str(t1_file),
                image_t2_path=str(t2_file),
                change_label=change_label,
                metadata={"source": "CDVQA"}
            )
            self.samples.append(sample)
        
        # Apply subset size if configured
        if self.subset_size and len(self.samples) > self.subset_size:
            random.seed(self.random_seed)
            self.samples = random.sample(self.samples, self.subset_size)
            logger.info(f"Using subset of {self.subset_size} samples from {len(self.samples)} total")
    
    def validate_sample(self, sample: ChangeDetectionSample) -> bool:
        """
        Validate a change detection sample.
        
        Args:
            sample: ChangeDetectionSample to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check if both images exist
        if not Path(sample.image_t1_path).exists():
            logger.warning(f"T1 image not found: {sample.image_t1_path}")
            return False
        
        if not Path(sample.image_t2_path).exists():
            logger.warning(f"T2 image not found: {sample.image_t2_path}")
            return False
        
        # Try to load and validate images
        try:
            image_t1 = Image.open(sample.image_t1_path)
            image_t2 = Image.open(sample.image_t2_path)
            
            # Check compatibility
            if image_t1.size != image_t2.size:
                logger.warning(f"Image size mismatch for {sample.sample_id}: T1={image_t1.size}, T2={image_t2.size}")
                # Still accept but warn
            
            if image_t1.mode != image_t2.mode:
                logger.warning(f"Image mode mismatch for {sample.sample_id}: T1={image_t1.mode}, T2={image_t2.mode}")
            
            image_t1.close()
            image_t2.close()
            
        except Exception as e:
            logger.warning(f"Failed to validate images for {sample.sample_id}: {e}")
            return False
        
        return True
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample data
        """
        sample = self.samples[idx]
        
        if self.synthetic:
            # Generate synthetic images on-the-fly
            image_t1 = Image.new('RGB', (224, 224), color=(100, 100, 100))
            image_t2 = Image.new('RGB', (224, 224), color=(150, 150, 150) if sample.change_label == 1 else (100, 100, 100))
        else:
            # Load real images
            image_t1 = Image.open(sample.image_t1_path).convert('RGB')
            image_t2 = Image.open(sample.image_t2_path).convert('RGB')
        
        return {
            'sample_id': sample.sample_id,
            'image_t1': image_t1,
            'image_t2': image_t2,
            'change_label': sample.change_label,
            'metadata': sample.metadata
        }


def create_change_detection_dataloader(
    dataset: ChangeDetectionDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for change detection dataset.
    
    Args:
        dataset: ChangeDetectionDataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader
    """
    def collate_fn(batch):
        """Custom collate function to handle PIL images."""
        image_t1_list = [item['image_t1'] for item in batch]
        image_t2_list = [item['image_t2'] for item in batch]
        change_labels = [item['change_label'] for item in batch]
        sample_ids = [item['sample_id'] for item in batch]
        
        return {
            'image_t1': image_t1_list,
            'image_t2': image_t2_list,
            'change_label': change_labels,
            'sample_id': sample_ids
        }
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
