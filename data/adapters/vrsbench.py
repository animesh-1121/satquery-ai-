"""
VRSBench Dataset Adapter for SatQueryAI

VRSBench is a benchmark for remote sensing visual question answering,
captioning, and grounding tasks.

Dataset Information:
- Contains remote sensing images with VQA, captioning, and grounding annotations
- Supports multiple remote sensing tasks
- Source: https://github.com/...

Usage:
    dataset = VRSBenchDataset(
        root="/path/to/vrsbench",
        subset_size=50,  # Use only 50 samples for development
        random_seed=42
    )
    dataset.load()
"""

import os
import random
from typing import List, Optional
from pathlib import Path
import logging

from ..base import BaseRemoteSensingDataset, Sample, Modality

logger = logging.getLogger(__name__)


class VRSBenchDataset(BaseRemoteSensingDataset):
    """
    Adapter for VRSBench dataset.
    
    VRSBench supports:
    - Visual Question Answering (VQA)
    - Image Captioning
    - Grounding (where applicable)
    
    Note: This adapter does NOT automatically download the dataset.
    Users must obtain VRSBench from the official source and place it in the root directory.
    """
    
    def __init__(
        self,
        root: str,
        subset_size: Optional[int] = None,
        random_seed: int = 42,
        enabled: bool = True
    ):
        """
        Initialize VRSBench dataset.
        
        Args:
            root: Root directory of VRSBench dataset
            subset_size: Optional limit on number of samples (for development)
            random_seed: Random seed for reproducible splits
            enabled: Whether this dataset is enabled
        """
        super().__init__(root, subset_size, random_seed, enabled)
    
    def load(self) -> None:
        """
        Load VRSBench dataset samples.
        
        Expected directory structure:
            root/
                images/
                    image_001.jpg
                    image_002.jpg
                    ...
                vqa/
                    vqa_annotations.json
                captions/
                    caption_annotations.json
                grounding/
                    grounding_annotations.json
        """
        root_path = Path(self.root)
        
        # Check if root exists
        if not root_path.exists():
            logger.warning(f"VRSBench root directory not found: {self.root}")
            logger.info("Please obtain VRSBench from the official source")
            self._loaded = True
            return
        
        # Scan for images
        images_dir = root_path / "images"
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            self._loaded = True
            return
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        # For now, create basic samples without full annotation parsing
        # Full implementation would parse the JSON annotation files
        self._samples = []
        
        for idx, image_file in enumerate(image_files):
            sample = Sample(
                sample_id=f"vrsbench_{idx:05d}",
                dataset=self.dataset_name,
                image_path=str(image_file),
                modality=Modality.OPTICAL,  # VRSBench typically uses optical imagery
                question=None,  # Would be loaded from VQA annotations
                answer=None,     # Would be loaded from VQA annotations
                caption=None,    # Would be loaded from caption annotations
                metadata={
                    "source": "VRSBench",
                    "annotations_loaded": False  # Full annotation parsing not implemented
                }
            )
            self._samples.append(sample)
        
        # Apply subset size if configured
        if self.subset_size and len(self._samples) > self.subset_size:
            random.seed(self.random_seed)
            self._samples = random.sample(self._samples, self.subset_size)
            logger.info(f"Using subset of {self.subset_size} samples from {len(self._samples)} total")
        
        self._loaded = True
        logger.info(f"Loaded {len(self._samples)} VRSBench samples (annotation parsing not implemented)")
    
    def get_split(self, split: str) -> List[Sample]:
        """
        Get samples for a specific split.
        
        Note: VRSBench may not have official splits. This implementation
        returns all samples for any split. Full implementation would
        parse the official split if available.
        
        Args:
            split: One of 'train', 'val', 'test'
            
        Returns:
            List of Sample objects for the requested split
        """
        # For now, return all samples regardless of split
        # Full implementation would respect official splits
        return self._samples
    
    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "VRSBench"
    
    @property
    def supports_vqa(self) -> bool:
        """VRSBench supports VQA tasks."""
        return True
    
    @property
    def supports_captioning(self) -> bool:
        """VRSBench supports captioning tasks."""
        return True
    
    @property
    def supports_grounding(self) -> bool:
        """VRSBench supports grounding tasks (where applicable)."""
        return True
    
    @property
    def supports_change_detection(self) -> bool:
        """VRSBench does not support change detection tasks."""
        return False
    
    @property
    def supports_optical_sar_fusion(self) -> bool:
        """VRSBench does not support optical-SAR fusion tasks."""
        return False
