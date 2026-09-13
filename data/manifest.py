"""
Dataset Manifest for SatQueryAI

This module provides functionality for creating and managing a normalized
manifest/index of processed samples across all datasets.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

from .base import Sample

logger = logging.getLogger(__name__)


class DatasetManifest:
    """
    Manifest for dataset samples.
    
    Provides a normalized index of all samples across datasets,
    enabling efficient discovery and querying.
    """
    
    def __init__(self, manifest_path: Optional[str] = None):
        """
        Initialize the manifest.
        
        Args:
            manifest_path: Path to manifest file (if loading existing)
        """
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.samples: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "datasets": []
        }
        
        if self.manifest_path and self.manifest_path.exists():
            self.load()
    
    def add_sample(self, sample: Sample) -> None:
        """
        Add a sample to the manifest.
        
        Args:
            sample: Sample object to add
        """
        self.samples[sample.sample_id] = sample.to_dict()
        
        # Track datasets
        if sample.dataset not in self.metadata["datasets"]:
            self.metadata["datasets"].append(sample.dataset)
    
    def add_samples(self, samples: List[Sample]) -> None:
        """
        Add multiple samples to the manifest.
        
        Args:
            samples: List of Sample objects to add
        """
        for sample in samples:
            self.add_sample(sample)
    
    def get_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a sample by ID.
        
        Args:
            sample_id: Sample ID
            
        Returns:
            Sample dictionary or None if not found
        """
        return self.samples.get(sample_id)
    
    def get_samples_by_dataset(self, dataset: str) -> List[Dict[str, Any]]:
        """
        Get all samples from a specific dataset.
        
        Args:
            dataset: Dataset name
            
        Returns:
            List of sample dictionaries
        """
        return [
            sample for sample in self.samples.values()
            if sample.get("dataset") == dataset
        ]
    
    def get_samples_by_modality(self, modality: str) -> List[Dict[str, Any]]:
        """
        Get all samples with a specific modality.
        
        Args:
            modality: Modality (optical, sar, multimodal_optical_sar)
            
        Returns:
            List of sample dictionaries
        """
        return [
            sample for sample in self.samples.values()
            if sample.get("modality") == modality
        ]
    
    def get_bi_temporal_samples(self) -> List[Dict[str, Any]]:
        """
        Get all bi-temporal samples (with T1 and T2 images).
        
        Returns:
            List of sample dictionaries
        """
        return [
            sample for sample in self.samples.values()
            if sample.get("image_t1") and sample.get("image_t2")
        ]
    
    def get_optical_sar_samples(self) -> List[Dict[str, Any]]:
        """
        Get all optical-SAR pairs.
        
        Returns:
            List of sample dictionaries
        """
        return [
            sample for sample in self.samples.values()
            if sample.get("optical") and sample.get("sar")
        ]
    
    def save(self, manifest_path: Optional[str] = None) -> None:
        """
        Save the manifest to disk.
        
        Args:
            manifest_path: Path to save manifest (uses default if None)
        """
        save_path = Path(manifest_path) if manifest_path else self.manifest_path
        
        if not save_path:
            raise ValueError("No manifest path specified")
        
        self.metadata["updated_at"] = datetime.now().isoformat()
        self.metadata["total_samples"] = len(self.samples)
        
        manifest_data = {
            "metadata": self.metadata,
            "samples": self.samples
        }
        
        with open(save_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        logger.info(f"Saved manifest with {len(self.samples)} samples to {save_path}")
    
    def load(self) -> None:
        """Load manifest from disk."""
        if not self.manifest_path or not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        with open(self.manifest_path, 'r') as f:
            manifest_data = json.load(f)
        
        self.metadata = manifest_data.get("metadata", {})
        self.samples = manifest_data.get("samples", {})
        
        logger.info(f"Loaded manifest with {len(self.samples)} samples from {self.manifest_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the manifest.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": len(self.samples),
            "datasets": self.metadata.get("datasets", []),
            "samples_per_dataset": {},
            "samples_per_modality": {},
            "bi_temporal_count": 0,
            "optical_sar_count": 0
        }
        
        # Count samples per dataset
        for sample in self.samples.values():
            dataset = sample.get("dataset", "unknown")
            stats["samples_per_dataset"][dataset] = stats["samples_per_dataset"].get(dataset, 0) + 1
            
            modality = sample.get("modality", "unknown")
            stats["samples_per_modality"][modality] = stats["samples_per_modality"].get(modality, 0) + 1
            
            if sample.get("image_t1") and sample.get("image_t2"):
                stats["bi_temporal_count"] += 1
            
            if sample.get("optical") and sample.get("sar"):
                stats["optical_sar_count"] += 1
        
        return stats
