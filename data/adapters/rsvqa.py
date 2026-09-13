"""
RSVQA Dataset Adapter for SatQueryAI

RSVQA is a remote sensing visual question answering dataset.
This adapter provides an interface for loading RSVQA data.

Dataset Information:
- Remote sensing VQA dataset
- Contains questions and answers for satellite images
- Source: https://github.com/...

Usage:
    dataset = RSVQADataset(
        root="/path/to/rsvqa",
        subset_size=50,
        random_seed=42
    )
    dataset.load()
"""

import random
from typing import List, Optional
from pathlib import Path
import logging

from ..base import BaseRemoteSensingDataset, Sample, Modality

logger = logging.getLogger(__name__)


class RSVQADataset(BaseRemoteSensingDataset):
    """
    Adapter for RSVQA dataset.
    
    RSVQA supports:
    - Visual Question Answering (VQA)
    - Question type/category information
    
    Note: This adapter does NOT automatically download the dataset.
    Users must obtain RSVQA from the official source and place it in the root directory.
    """
    
    def __init__(
        self,
        root: str,
        subset_size: Optional[int] = None,
        random_seed: int = 42,
        enabled: bool = True
    ):
        """
        Initialize RSVQA dataset.
        
        Args:
            root: Root directory of RSVQA dataset
            subset_size: Optional limit on number of samples (for development)
            random_seed: Random seed for reproducible splits
            enabled: Whether this dataset is enabled
        """
        super().__init__(root, subset_size, random_seed, enabled)
    
    def load(self) -> None:
        """
        Load RSVQA dataset samples.
        
        Expected directory structure:
            root/
                images/
                    image_001.jpg
                    ...
                annotations/
                    rsvqa_annotations.json
        """
        root_path = Path(self.root)
        
        # Check if root exists
        if not root_path.exists():
            logger.warning(f"RSVQA root directory not found: {self.root}")
            logger.info("Please obtain RSVQA from the official source")
            self._loaded = True
            return
        
        # Scan for images
        images_dir = root_path / "images"
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            self._loaded = True
            return
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        # Create basic samples without full annotation parsing
        self._samples = []
        
        for idx, image_file in enumerate(image_files):
            sample = Sample(
                sample_id=f"rsvqa_{idx:05d}",
                dataset=self.dataset_name,
                image_path=str(image_file),
                modality=Modality.OPTICAL,
                question=None,  # Would be loaded from RSVQA annotations
                answer=None,     # Would be loaded from RSVQA annotations
                metadata={
                    "source": "RSVQA",
                    "question_type": None,  # Would preserve question type/category
                    "annotations_loaded": False
                }
            )
            self._samples.append(sample)
        
        # Apply subset size if configured
        if self.subset_size and len(self._samples) > self.subset_size:
            random.seed(self.random_seed)
            self._samples = random.sample(self._samples, self.subset_size)
            logger.info(f"Using subset of {self.subset_size} samples from {len(self._samples)} total")
        
        self._loaded = True
        logger.info(f"Loaded {len(self._samples)} RSVQA samples (annotation parsing not implemented)")
    
    def get_split(self, split: str) -> List[Sample]:
        """
        Get samples for a specific split.
        
        Args:
            split: One of 'train', 'val', 'test'
            
        Returns:
            List of Sample objects for the requested split
        """
        return self._samples
    
    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "RSVQA"
    
    @property
    def supports_vqa(self) -> bool:
        """RSVQA supports VQA tasks."""
        return True
    
    @property
    def supports_captioning(self) -> bool:
        """RSVQA does not support captioning tasks."""
        return False
    
    @property
    def supports_grounding(self) -> bool:
        """RSVQA does not support grounding tasks."""
        return False
    
    @property
    def supports_change_detection(self) -> bool:
        """RSVQA does not support change detection tasks."""
        return False
    
    @property
    def supports_optical_sar_fusion(self) -> bool:
        """RSVQA does not support optical-SAR fusion tasks."""
        return False
