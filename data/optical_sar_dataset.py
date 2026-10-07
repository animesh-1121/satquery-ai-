"""
Optical-SAR Dataset for SatQueryAI Phase 5

This module provides a dataset loader for optical-SAR paired data.
It handles optical and SAR image pairs with validation.
"""

import json
import random
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class OpticalSARSample:
    """A single optical-SAR sample with paired images."""
    
    def __init__(
        self,
        sample_id: str,
        optical_path: str,
        sar_path: str,
        label: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.sample_id = sample_id
        self.optical_path = optical_path
        self.sar_path = sar_path
        self.label = label
        self.metadata = metadata or {}


class OpticalSARDataset(Dataset):
    """
    Dataset for optical-SAR paired data.
    
    Handles optical and SAR image pairs with validation.
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
        Initialize optical-SAR dataset.
        
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
        
        self.samples: List[OpticalSARSample] = []
        self._loaded = False
        
        self.load()
    
    def load(self) -> None:
        """Load dataset samples."""
        if self.synthetic:
            self._load_synthetic()
        else:
            self._load_real()
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} optical-SAR samples for {self.split} split")
    
    def _load_synthetic(self) -> None:
        """Load synthetic data for testing."""
        # Create synthetic samples for smoke testing
        self.samples = []
        for i in range(10):
            sample = OpticalSARSample(
                sample_id=f"synthetic_{i:05d}",
                optical_path="",  # Will be generated on-the-fly
                sar_path="",  # Will be generated on-the-fly
                label=i % 10,  # Cycle through 10 classes
                metadata={"synthetic": True}
            )
            self.samples.append(sample)
    
    def _load_real(self) -> None:
        """Load real dataset from disk."""
        # Check if root exists
        if not self.root.exists():
            logger.warning(f"Dataset root not found: {self.root}")
            return
        
        # Scan for optical and SAR images
        optical_dir = self.root / "optical"
        sar_dir = self.root / "sar"
        
        if not optical_dir.exists() or not sar_dir.exists():
            logger.warning(f"Optical or SAR directory not found: {optical_dir}, {sar_dir}")
            return
        
        optical_files = list(optical_dir.glob("*.jpg")) + list(optical_dir.glob("*.png"))
        
        # Create samples with optical-SAR pairs
        self.samples = []
        
        for idx, optical_file in enumerate(optical_files):
            # Look for corresponding SAR file
            base_name = optical_file.stem
            sar_file = sar_dir / f"{base_name}{optical_file.suffix}"
            
            if not sar_file.exists():
                # Try alternative naming
                sar_file = sar_dir / optical_file.name
            
            if not sar_file.exists():
                logger.warning(f"SAR file not found for {optical_file.name}")
                continue
            
            # Try to load label from annotations
            label = None
            annotations_file = self.root / "annotations" / "labels.json"
            if annotations_file.exists():
                try:
                    with open(annotations_file, 'r') as f:
                        annotations = json.load(f)
                        label = annotations.get(base_name, {}).get("label")
                except Exception as e:
                    logger.warning(f"Failed to load annotations: {e}")
            
            sample = OpticalSARSample(
                sample_id=f"os_{idx:05d}",
                optical_path=str(optical_file),
                sar_path=str(sar_file),
                label=label,
                metadata={"source": "OpticalSAR"}
            )
            self.samples.append(sample)
        
        # Apply subset size if configured
        if self.subset_size and len(self.samples) > self.subset_size:
            random.seed(self.random_seed)
            self.samples = random.sample(self.samples, self.subset_size)
            logger.info(f"Using subset of {self.subset_size} samples from {len(self.samples)} total")
    
    def validate_sample(self, sample: OpticalSARSample) -> bool:
        """
        Validate an optical-SAR sample.
        
        Args:
            sample: OpticalSARSample to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check if both images exist
        if not Path(sample.optical_path).exists():
            logger.warning(f"Optical image not found: {sample.optical_path}")
            return False
        
        if not Path(sample.sar_path).exists():
            logger.warning(f"SAR image not found: {sample.sar_path}")
            return False
        
        # Try to load and validate images
        try:
            optical_image = Image.open(sample.optical_path)
            sar_image = Image.open(sample.sar_path)
            
            # Check compatibility
            if optical_image.size != sar_image.size:
                logger.warning(f"Image size mismatch for {sample.sample_id}: Optical={optical_image.size}, SAR={sar_image.size}")
            
            optical_image.close()
            sar_image.close()
            
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
            optical_image = Image.new('RGB', (224, 224), color=(100, 100, 100))
            sar_image = Image.new('RGB', (224, 224), color=(150, 150, 150))
        else:
            # Load real images
            optical_image = Image.open(sample.optical_path).convert('RGB')
            sar_image = Image.open(sample.sar_path).convert('RGB')
        
        return {
            'sample_id': sample.sample_id,
            'optical': optical_image,
            'sar': sar_image,
            'label': sample.label,
            'metadata': sample.metadata
        }


def create_optical_sar_dataloader(
    dataset: OpticalSARDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for optical-SAR dataset.
    
    Args:
        dataset: OpticalSARDataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader
    """
    def collate_fn(batch):
        """Custom collate function to handle PIL images."""
        optical_list = [item['optical'] for item in batch]
        sar_list = [item['sar'] for item in batch]
        labels = [item['label'] for item in batch]
        sample_ids = [item['sample_id'] for item in batch]
        
        return {
            'optical': optical_list,
            'sar': sar_list,
            'label': labels,
            'sample_id': sample_ids
        }
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
