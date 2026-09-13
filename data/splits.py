"""
Data Split Utilities for SatQueryAI

This module provides reproducible dataset splitting functionality with support
for official splits and deterministic random splits.
"""

import random
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json
import logging
from datetime import datetime

from .base import Sample

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Utility for creating reproducible dataset splits.
    
    Supports:
    - Deterministic random splits with fixed seeds
    - Official splits from dataset metadata
    - Custom split ratios
    - Split persistence for reproducibility
    """
    
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        use_official_splits: bool = True
    ):
        """
        Initialize the data splitter.
        
        Args:
            train_ratio: Ratio of training data (0.0-1.0)
            val_ratio: Ratio of validation data (0.0-1.0)
            test_ratio: Ratio of test data (0.0-1.0)
            random_seed: Random seed for reproducibility
            use_official_splits: Whether to use official splits if available
        """
        # Validate ratios
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.use_official_splits = use_official_splits
    
    def split_samples(
        self,
        samples: List[Sample],
        dataset_name: str
    ) -> Tuple[List[Sample], List[Sample], List[Sample]]:
        """
        Split samples into train, validation, and test sets.
        
        Args:
            samples: List of Sample objects
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        if not samples:
            logger.warning(f"No samples to split for {dataset_name}")
            return [], [], []
        
        # Check if samples have official split information
        if self.use_official_splits:
            official_splits = self._extract_official_splits(samples)
            if official_splits:
                logger.info(f"Using official splits for {dataset_name}")
                return official_splits
        
        # Create deterministic random split
        logger.info(f"Creating random split for {dataset_name} (seed={self.random_seed})")
        return self._create_random_split(samples)
    
    def _extract_official_splits(
        self,
        samples: List[Sample]
    ) -> Optional[Tuple[List[Sample], List[Sample], List[Sample]]]:
        """
        Extract official splits from sample metadata.
        
        Args:
            samples: List of Sample objects
            
        Returns:
            Tuple of (train, val, test) if official splits exist, else None
        """
        train_samples = []
        val_samples = []
        test_samples = []
        
        for sample in samples:
            split = sample.metadata.get("split")
            if split == "train":
                train_samples.append(sample)
            elif split == "val":
                train_samples.append(sample)  # Some datasets use "val" as part of train
            elif split == "validation":
                val_samples.append(sample)
            elif split == "test":
                test_samples.append(sample)
        
        # Only return if we have all three splits
        if train_samples and val_samples and test_samples:
            return train_samples, val_samples, test_samples
        
        return None
    
    def _create_random_split(
        self,
        samples: List[Sample]
    ) -> Tuple[List[Sample], List[Sample], List[Sample]]:
        """
        Create a deterministic random split.
        
        Args:
            samples: List of Sample objects
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Shuffle samples
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        # Calculate split indices
        n_samples = len(shuffled_samples)
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)
        
        # Split samples
        train_samples = shuffled_samples[:n_train]
        val_samples = shuffled_samples[n_train:n_train + n_val]
        test_samples = shuffled_samples[n_train + n_val:]
        
        logger.info(
            f"Split {n_samples} samples into "
            f"{len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test"
        )
        
        return train_samples, val_samples, test_samples
    
    def save_split_info(
        self,
        train_samples: List[Sample],
        val_samples: List[Sample],
        test_samples: List[Sample],
        output_path: str
    ) -> None:
        """
        Save split information to disk for reproducibility.
        
        Args:
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples
            output_path: Path to save split information
        """
        split_info = {
            "created_at": datetime.now().isoformat(),
            "random_seed": self.random_seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "train_samples": [s.sample_id for s in train_samples],
            "val_samples": [s.sample_id for s in val_samples],
            "test_samples": [s.sample_id for s in test_samples],
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "test_count": len(test_samples)
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Saved split information to {output_path}")
    
    def load_split_info(
        self,
        samples: List[Sample],
        split_info_path: str
    ) -> Tuple[List[Sample], List[Sample], List[Sample]]:
        """
        Load split information from disk and apply to samples.
        
        Args:
            samples: List of Sample objects
            split_info_path: Path to split information file
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
        
        # Create sample ID to sample mapping
        sample_map = {s.sample_id: s for s in samples}
        
        # Reconstruct splits
        train_samples = [sample_map[sid] for sid in split_info["train_samples"] if sid in sample_map]
        val_samples = [sample_map[sid] for sid in split_info["val_samples"] if sid in sample_map]
        test_samples = [sample_map[sid] for sid in split_info["test_samples"] if sid in sample_map]
        
        logger.info(
            f"Loaded split from {split_info_path}: "
            f"{len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test"
        )
        
        return train_samples, val_samples, test_samples


def create_train_val_test_split(
    samples: List[Sample],
    dataset_name: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    use_official_splits: bool = True
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Convenience function to create train/val/test split.
    
    Args:
        samples: List of Sample objects
        dataset_name: Name of the dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility
        use_official_splits: Whether to use official splits if available
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    splitter = DataSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        use_official_splits=use_official_splits
    )
    
    return splitter.split_samples(samples, dataset_name)
