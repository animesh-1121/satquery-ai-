"""
CDVQA Dataset Adapter for SatQueryAI

CDVQA is a change detection visual question answering dataset.
This adapter provides an interface for loading bi-temporal change detection data.

Dataset Information:
- Bi-temporal image pairs (T1, T2)
- Change detection questions and answers
- Source: https://github.com/...

Usage:
    dataset = CDVQADataset(
        root="/path/to/cdvqa",
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


class CDVQADataset(BaseRemoteSensingDataset):
    """
    Adapter for CDVQA dataset.
    
    CDVQA supports:
    - Bi-temporal change detection
    - Change VQA (questions about changes between T1 and T2)
    
    Note: This adapter does NOT automatically download the dataset.
    Users must obtain CDVQA from the official source and place it in the root directory.
    """
    
    def __init__(
        self,
        root: str,
        subset_size: Optional[int] = None,
        random_seed: int = 42,
        enabled: bool = True
    ):
        """
        Initialize CDVQA dataset.
        
        Args:
            root: Root directory of CDVQA dataset
            subset_size: Optional limit on number of samples (for development)
            random_seed: Random seed for reproducible splits
            enabled: Whether this dataset is enabled
        """
        super().__init__(root, subset_size, random_seed, enabled)
    
    def load(self) -> None:
        """
        Load CDVQA dataset samples.
        
        Expected directory structure:
            root/
                t1/
                    image_001_t1.jpg
                    ...
                t2/
                    image_001_t2.jpg
                    ...
                annotations/
                    cdvqa_annotations.json
        """
        root_path = Path(self.root)
        
        # Check if root exists
        if not root_path.exists():
            logger.warning(f"CDVQA root directory not found: {self.root}")
            logger.info("Please obtain CDVQA from the official source")
            self._loaded = True
            return
        
        # Scan for T1 and T2 images
        t1_dir = root_path / "t1"
        t2_dir = root_path / "t2"
        
        if not t1_dir.exists() or not t2_dir.exists():
            logger.warning(f"T1 or T2 directory not found: {t1_dir}, {t2_dir}")
            self._loaded = True
            return
        
        t1_files = list(t1_dir.glob("*.jpg")) + list(t1_dir.glob("*.png"))
        
        # Create samples with bi-temporal pairs
        self._samples = []
        
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
            
            sample = Sample(
                sample_id=f"cdvqa_{idx:05d}",
                dataset=self.dataset_name,
                image_t1_path=str(t1_file),
                image_t2_path=str(t2_file),
                modality=Modality.OPTICAL,  # Assuming optical for change detection
                question=None,  # Would be loaded from CDVQA annotations
                answer=None,     # Would be loaded from CDVQA annotations
                metadata={
                    "source": "CDVQA",
                    "change_type": None,  # Would be loaded from annotations
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
        logger.info(f"Loaded {len(self._samples)} CDVQA bi-temporal samples (annotation parsing not implemented)")
    
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
        return "CDVQA"
    
    @property
    def supports_vqa(self) -> bool:
        """CDVQA supports change VQA tasks."""
        return True
    
    @property
    def supports_captioning(self) -> bool:
        """CDVQA does not support captioning tasks."""
        return False
    
    @property
    def supports_grounding(self) -> bool:
        """CDVQA does not support grounding tasks."""
        return False
    
    @property
    def supports_change_detection(self) -> bool:
        """CDVQA supports change detection tasks."""
        return True
    
    @property
    def supports_optical_sar_fusion(self) -> bool:
        """CDVQA does not support optical-SAR fusion tasks."""
        return False
