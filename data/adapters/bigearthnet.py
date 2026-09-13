"""
BigEarthNet Dataset Adapter for SatQueryAI

BigEarthNet is a large-scale dataset for remote sensing image classification.
This adapter provides a configurable interface for loading BigEarthNet data
without requiring automatic download of the full dataset.

Dataset Information:
- Paper: "BigEarthNet: A Large-Scale Benchmark Archive of Remote Sensing Images"
- Contains ~590,000 Sentinel-2 image patches
- Multi-label classification with 43 land cover classes
- Sentinel-2 data with 13 spectral bands
- Source: https://bigearthnet.github.io/

Usage:
    dataset = BigEarthNetDataset(
        root="/path/to/bigearthnet",
        subset_size=100,  # Use only 100 samples for development
        random_seed=42
    )
    dataset.load()
    train_samples = dataset.get_split("train")
"""

import os
import random
from typing import List, Optional
from pathlib import Path
import logging

from ..base import BaseRemoteSensingDataset, Sample, Modality

logger = logging.getLogger(__name__)


class BigEarthNetDataset(BaseRemoteSensingDataset):
    """
    Adapter for BigEarthNet dataset.
    
    BigEarthNet is a multi-label remote sensing dataset with Sentinel-2 imagery.
    This adapter supports:
    - Configurable subset sizes for development
    - Multi-label classification
    - Spectral band preservation
    - Sentinel-2 data (13 bands)
    
    Note: This adapter does NOT automatically download the dataset.
    Users must obtain BigEarthNet from the official source and place it in the root directory.
    """
    
    # BigEarthNet multi-label classes (43 classes)
    LABEL_CLASSES = [
        "Urban fabric", "Industrial or commercial units", "Road and rail networks",
        "Sparse residential", "Dense residential", "Clouds", "Cultivated land",
        "Bare rock", "Herbaceous vegetation", "Non-irrigated arable land",
        "Water bodies", "Peatbogs", "Salt marshes", "Intermittent water",
        "Coniferous forest", "Mixed forest", "Moors and heathland",
        "Natural grassland", "Continuous urban fabric", "Discontinuous urban fabric",
        "Dump sites", "Construction sites", "Green urban areas",
        "Sport and leisure facilities", "Airports", "Mineral extraction sites",
        "Beaches, dunes, sands", "Estuaries", "Marine waters",
        "Rice fields", "Agricultural with natural vegetation", "Port areas",
        "Olive groves", "Annual crops associated with permanent crops",
        "Broad-leaved forest", "Industrial or commercial units with public facilities",
        "Sclerophyllous vegetation", "Transitional woodland/shrub",
        "Water courses", "Sea and ocean", "Coastal lagoons", "Wetlands"
    ]
    
    def __init__(
        self,
        root: str,
        subset_size: Optional[int] = None,
        random_seed: int = 42,
        enabled: bool = True,
        split: str = "s1"  # "s1" for Sentinel-1 (SAR) or "s2" for Sentinel-2 (optical)
    ):
        """
        Initialize BigEarthNet dataset.
        
        Args:
            root: Root directory of BigEarthNet dataset
            subset_size: Optional limit on number of samples (for development)
            random_seed: Random seed for reproducible splits
            enabled: Whether this dataset is enabled
            split: "s1" for Sentinel-1 (SAR) or "s2" for Sentinel-2 (optical)
        """
        super().__init__(root, subset_size, random_seed, enabled)
        self.split = split  # s1 or s2
        self.modality = Modality.SAR if split == "s1" else Modality.OPTICAL
    
    def load(self) -> None:
        """
        Load BigEarthNet dataset samples.
        
        This method scans the BigEarthNet directory structure and creates Sample objects.
        It does NOT download the dataset automatically.
        
        Expected directory structure:
            root/
                BigEarthNet-S1/  # or BigEarthNet-S2/
                    train/
                        patch_1/
                            patch_1.tif
                            patch_1_labels.csv
                        patch_2/
                            ...
                    val/
                    test/
        """
        root_path = Path(self.root)
        
        # Check if root exists
        if not root_path.exists():
            logger.warning(f"BigEarthNet root directory not found: {self.root}")
            logger.info("Please download BigEarthNet from: https://bigearthnet.github.io/")
            self._loaded = True
            return
        
        # Determine dataset name based on split
        dataset_name = f"BigEarthNet-{self.split.upper()}"
        dataset_path = root_path / dataset_name
        
        if not dataset_path.exists():
            logger.warning(f"BigEarthNet {self.split.upper()} directory not found: {dataset_path}")
            self._loaded = True
            return
        
        # Scan for samples
        self._samples = []
        
        for split_name in ["train", "val", "test"]:
            split_path = dataset_path / split_name
            if not split_path.exists():
                logger.warning(f"Split directory not found: {split_path}")
                continue
            
            # Scan for patch directories
            patch_dirs = [d for d in split_path.iterdir() if d.is_dir()]
            
            for patch_dir in patch_dirs:
                # Look for image file
                tif_files = list(patch_dir.glob("*.tif"))
                if not tif_files:
                    continue
                
                image_path = str(tif_files[0])
                
                # Look for labels file
                label_files = list(patch_dir.glob("*labels.csv"))
                labels = None
                labels_multi_hot = None
                
                if label_files:
                    labels_file = label_files[0]
                    try:
                        labels, labels_multi_hot = self._parse_labels(labels_file)
                    except Exception as e:
                        logger.warning(f"Failed to parse labels for {patch_dir.name}: {e}")
                
                # Create sample
                sample = Sample(
                    sample_id=patch_dir.name,
                    dataset=self.dataset_name,
                    image_path=image_path,
                    labels=labels,
                    labels_multi_hot=labels_multi_hot,
                    modality=self.modality,
                    sensor="Sentinel-1" if self.split == "s1" else "Sentinel-2",
                    bands=self._get_bands_for_split(),
                    resolution=10.0 if self.split == "s2" else None,  # Sentinel-2 resolution
                    metadata={"split": split_name}  # Store split in metadata
                )
                
                self._samples.append(sample)
        
        # Apply subset size if configured
        if self.subset_size and len(self._samples) > self.subset_size:
            random.seed(self.random_seed)
            self._samples = random.sample(self._samples, self.subset_size)
            logger.info(f"Using subset of {self.subset_size} samples from {len(self._samples)} total")
        
        self._loaded = True
        logger.info(f"Loaded {len(self._samples)} BigEarthNet-{self.split.upper()} samples")
    
    def get_split(self, split: str) -> List[Sample]:
        """
        Get samples for a specific split.
        
        Args:
            split: One of 'train', 'val', 'test'
            
        Returns:
            List of Sample objects for the requested split
        """
        # BigEarthNet has official splits in metadata
        return [s for s in self._samples if s.metadata.get("split") == split]
    
    def _parse_labels(self, labels_file: Path) -> tuple:
        """
        Parse BigEarthNet labels file.
        
        Args:
            labels_file: Path to labels CSV file
            
        Returns:
            Tuple of (label_names, multi_hot_encoding)
        """
        import csv
        
        label_names = []
        multi_hot = [0] * len(self.LABEL_CLASSES)
        
        with open(labels_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    class_name = row[0]
                    class_index = int(row[1])
                    
                    if 0 <= class_index < len(self.LABEL_CLASSES):
                        label_names.append(self.LABEL_CLASSES[class_index])
                        multi_hot[class_index] = 1
        
        return label_names, multi_hot
    
    def _get_bands_for_split(self) -> List[str]:
        """
        Get band names for the current split.
        
        Returns:
            List of band names
        """
        if self.split == "s1":
            # Sentinel-1 bands (VV, VH)
            return ["VV", "VH"]
        else:
            # Sentinel-2 bands (13 bands)
            return [
                "B1", "B2", "B3", "B4", "B5", "B6", "B7",
                "B8", "B8A", "B9", "B10", "B11", "B12"
            ]
    
    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return f"BigEarthNet-{self.split.upper()}"
    
    @property
    def supports_vqa(self) -> bool:
        """BigEarthNet does not support VQA tasks."""
        return False
    
    @property
    def supports_captioning(self) -> bool:
        """BigEarthNet does not support captioning tasks."""
        return False
    
    @property
    def supports_grounding(self) -> bool:
        """BigEarthNet does not support grounding tasks."""
        return False
    
    @property
    def supports_change_detection(self) -> bool:
        """BigEarthNet does not support change detection tasks."""
        return False
    
    @property
    def supports_optical_sar_fusion(self) -> bool:
        """BigEarthNet does not support optical-SAR fusion tasks (separate splits)."""
        return False
