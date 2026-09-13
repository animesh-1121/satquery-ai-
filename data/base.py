"""
Base Dataset Interface for SatQueryAI

This module defines the abstract interface for all remote sensing datasets
in the SatQueryAI system. This enables pluggable dataset adapters that
can be used for training, evaluation, and inference.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Union, Tuple, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import logging

if TYPE_CHECKING:
    from .validation import ValidationResult

logger = logging.getLogger(__name__)


class Modality(Enum):
    """Image modality types."""
    OPTICAL = "optical"
    SAR = "sar"
    MULTIMODAL_OPTICAL_SAR = "multimodal_optical_sar"
    UNKNOWN = "unknown"


@dataclass
class Sample:
    """
    Normalized sample representation for remote sensing data.
    
    This structure provides a consistent interface across different datasets.
    Only fields that actually exist for a particular dataset are populated.
    """
    sample_id: str
    dataset: str
    
    # Image paths
    image_path: Optional[str] = None
    image_t1_path: Optional[str] = None  # For bi-temporal datasets
    image_t2_path: Optional[str] = None  # For bi-temporal datasets
    optical_path: Optional[str] = None   # For optical-SAR pairs
    sar_path: Optional[str] = None       # For optical-SAR pairs
    
    # Annotations
    question: Optional[str] = None
    answer: Optional[str] = None
    caption: Optional[str] = None
    labels: Optional[List[str]] = None
    labels_multi_hot: Optional[List[int]] = None  # Multi-label encoding
    
    # Metadata
    modality: Modality = Modality.UNKNOWN
    sensor: Optional[str] = None
    bands: Optional[List[str]] = None
    resolution: Optional[float] = None  # Ground sample distance in meters
    crs: Optional[str] = None  # Coordinate reference system
    width: Optional[int] = None
    height: Optional[int] = None
    original_width: Optional[int] = None
    original_height: Optional[int] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        result = {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
        }
        
        # Add image paths
        if self.image_path:
            result["image_path"] = self.image_path
        if self.image_t1_path:
            result["image_t1"] = self.image_t1_path
        if self.image_t2_path:
            result["image_t2"] = self.image_t2_path
        if self.optical_path:
            result["optical"] = self.optical_path
        if self.sar_path:
            result["sar"] = self.sar_path
        
        # Add annotations
        if self.question:
            result["question"] = self.question
        if self.answer:
            result["answer"] = self.answer
        if self.caption:
            result["caption"] = self.caption
        if self.labels:
            result["labels"] = self.labels
        if self.labels_multi_hot:
            result["labels_multi_hot"] = self.labels_multi_hot
        
        # Add metadata
        result["modality"] = self.modality.value
        if self.sensor:
            result["sensor"] = self.sensor
        if self.bands:
            result["bands"] = self.bands
        if self.resolution:
            result["resolution"] = self.resolution
        if self.crs:
            result["crs"] = self.crs
        if self.width:
            result["width"] = self.width
        if self.height:
            result["height"] = self.height
        if self.original_width:
            result["original_width"] = self.original_width
        if self.original_height:
            result["original_height"] = self.original_height
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result


class BaseRemoteSensingDataset(ABC):
    """
    Abstract base class for all remote sensing datasets in SatQueryAI.
    
    All dataset adapters must implement this interface to ensure
    consistent behavior and enable downstream processing.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        subset_size: Optional[int] = None,
        random_seed: int = 42,
        enabled: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory of the dataset
            subset_size: Optional limit on number of samples (for development/testing)
            random_seed: Random seed for reproducible splits
            enabled: Whether this dataset is enabled
        """
        self.root = Path(root)
        self.subset_size = subset_size
        self.random_seed = random_seed
        self.enabled = enabled
        self._samples: List[Sample] = []
        self._loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the dataset samples.
        
        This method should populate self._samples with Sample objects.
        """
        pass
    
    @abstractmethod
    def get_split(self, split: str) -> List[Sample]:
        """
        Get samples for a specific split.
        
        Args:
            split: One of 'train', 'val', 'test'
            
        Returns:
            List of Sample objects for the requested split
        """
        pass
    
    def create_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        use_official_splits: bool = True
    ) -> Tuple[List[Sample], List[Sample], List[Sample]]:
        """
        Create train/val/test splits for the dataset.
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            use_official_splits: Whether to use official splits if available
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        # Import here to avoid circular dependency
        try:
            from .splits import DataSplitter
        except ImportError:
            # If splits module is not available, return empty splits
            logger.warning("DataSplitter not available, returning empty splits")
            return [], [], []
        
        splitter = DataSplitter(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=self.random_seed,
            use_official_splits=use_official_splits
        )
        
        return splitter.split_samples(self._samples, self.dataset_name)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> Sample:
        """Get a sample by index."""
        return self._samples[idx]
    
    def is_loaded(self) -> bool:
        """Check if the dataset is loaded."""
        return self._loaded
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the dataset name."""
        pass
    
    @property
    @abstractmethod
    def supports_vqa(self) -> bool:
        """Whether this dataset supports VQA tasks."""
        pass
    
    @property
    @abstractmethod
    def supports_captioning(self) -> bool:
        """Whether this dataset supports captioning tasks."""
        pass
    
    @property
    @abstractmethod
    def supports_grounding(self) -> bool:
        """Whether this dataset supports grounding tasks."""
        pass
    
    @property
    @abstractmethod
    def supports_change_detection(self) -> bool:
        """Whether this dataset supports change detection tasks."""
        pass
    
    @property
    @abstractmethod
    def supports_optical_sar_fusion(self) -> bool:
        """Whether this dataset supports optical-SAR fusion tasks."""
        pass
    
    def validate_paths(self) -> 'ValidationResult':
        """
        Validate that all image paths in the dataset exist.
        
        Returns:
            ValidationResult with validation status and any errors
        """
        from .validation import ValidationResult as VR
        
        errors = []
        warnings = []
        valid_count = 0
        total_count = 0
        
        for sample in self._samples:
            total_count += 1
            
            # Check single image
            if sample.image_path and not Path(sample.image_path).exists():
                errors.append(f"Sample {sample.sample_id}: Image not found: {sample.image_path}")
                continue
            
            # Check bi-temporal images
            if sample.image_t1_path and not Path(sample.image_t1_path).exists():
                errors.append(f"Sample {sample.sample_id}: T1 image not found: {sample.image_t1_path}")
                continue
            
            if sample.image_t2_path and not Path(sample.image_t2_path).exists():
                errors.append(f"Sample {sample.sample_id}: T2 image not found: {sample.image_t2_path}")
                continue
            
            # Check optical-SAR images
            if sample.optical_path and not Path(sample.optical_path).exists():
                errors.append(f"Sample {sample.sample_id}: Optical image not found: {sample.optical_path}")
                continue
            
            if sample.sar_path and not Path(sample.sar_path).exists():
                errors.append(f"Sample {sample.sample_id}: SAR image not found: {sample.sar_path}")
                continue
            
            valid_count += 1
        
        return VR(
            valid=total_count > 0 and len(errors) == 0,
            total_samples=total_count,
            valid_samples=valid_count,
            invalid_samples=len(errors),
            errors=errors,
            warnings=warnings
        )
