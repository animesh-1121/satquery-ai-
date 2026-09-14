# Adding New Datasets to SatQuery AI

This guide explains how to add new datasets to the SatQuery AI system.

## Dataset Architecture

The system uses a modular architecture where each dataset implements the `BaseRemoteSensingDataset` interface:

```
BaseRemoteSensingDataset (abstract interface)
        ├── BigEarthNetDataset
        ├── VRSBenchDataset
        ├── RSVQADataset
        ├── CDVQADataset
        └── YourNewDataset  # Add your dataset here
```

## Step-by-Step Guide

### Step 1: Create Dataset Adapter

Create a new file in `data/adapters/your_dataset.py`:

```python
"""
Your Dataset Adapter for SatQueryAI

Dataset Information:
- Describe your dataset here
- Source: Link to dataset website
"""

from typing import List, Optional
from pathlib import Path
import logging

from ..base import BaseRemoteSensingDataset, Sample, Modality

logger = logging.getLogger(__name__)


class YourDataset(BaseRemoteSensingDataset):
    """
    Adapter for your dataset.
    
    This adapter supports:
    - List the capabilities here
    """
    
    def __init__(
        self,
        root: str,
        subset_size: Optional[int] = None,
        random_seed: int = 42,
        enabled: bool = True
    ):
        """
        Initialize your dataset.
        
        Args:
            root: Root directory of the dataset
            subset_size: Optional limit on number of samples
            random_seed: Random seed for reproducible splits
            enabled: Whether this dataset is enabled
        """
        super().__init__(root, subset_size, random_seed, enabled)
    
    def load(self) -> None:
        """
        Load your dataset samples.
        
        Expected directory structure:
            root/
                images/
                    image_001.jpg
                    ...
                annotations/
                    annotations.json
        """
        root_path = Path(self.root)
        
        if not root_path.exists():
            logger.warning(f"Dataset root not found: {self.root}")
            self._loaded = True
            return
        
        # Scan for samples
        self._samples = []
        
        # Add your dataset loading logic here
        # Create Sample objects for each data point
        
        # Apply subset size if configured
        if self.subset_size and len(self._samples) > self.subset_size:
            import random
            random.seed(self.random_seed)
            self._samples = random.sample(self._samples, self.subset_size)
        
        self._loaded = True
        logger.info(f"Loaded {len(self._samples)} samples")
    
    def get_split(self, split: str) -> List[Sample]:
        """
        Get samples for a specific split.
        
        Args:
            split: One of 'train', 'val', 'test'
            
        Returns:
            List of Sample objects for the requested split
        """
        # Return samples based on split
        return self._samples
    
    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "YourDataset"
    
    @property
    def supports_vqa(self) -> bool:
        """Whether this dataset supports VQA tasks."""
        return True  # or False
    
    @property
    def supports_captioning(self) -> bool:
        """Whether this dataset supports captioning tasks."""
        return False  # or True
    
    @property
    def supports_grounding(self) -> bool:
        """Whether this dataset supports grounding tasks."""
        return False  # or True
    
    @property
    def supports_change_detection(self) -> bool:
        """Whether this dataset supports change detection tasks."""
        return False  # or True
    
    @property
    def supports_optical_sar_fusion(self) -> bool:
        """Whether this dataset supports optical-SAR fusion tasks."""
        return False  # or True
```

### Step 2: Register Dataset Adapter

Add your dataset to `data/adapters/__init__.py`:

```python
from .your_dataset import YourDataset

__all__ = [
    "BigEarthNetDataset",
    "VRSBenchDataset",
    "RSVQADataset",
    "CDVQADataset",
    "YourDataset",  # Add this
]
```

Also add to `data/__init__.py`:

```python
from .adapters import BigEarthNetDataset, VRSBenchDataset, RSVQADataset, CDVQADataset, YourDataset

__all__ = [
    # ... existing exports ...
    "YourDataset",  # Add this
]
```

### Step 3: Add Dataset Configuration

Add your dataset to `configs/datasets.yaml`:

```yaml
datasets:
  # ... existing datasets ...
  
  your_dataset:
    root: ${YOUR_DATASET_ROOT:-/path/to/your_dataset}
    enabled: true
    subset_size: 50  # Adjust based on your needs
    random_seed: 42
    description: "Your dataset description"
```

### Step 4: Update CLI (Optional)

If you want CLI support, update `data/cli.py`:

```python
def validate_dataset(args):
    # ... existing code ...
    
    elif args.dataset == "your_dataset":
        dataset = YourDataset(
            root=args.root,
            subset_size=args.limit,
            enabled=True
        )
    # ... rest of code ...
```

And add the CLI argument:

```python
validate_parser.add_argument("--dataset", required=True, 
    choices=["bigearthnet", "vrsbench", "rsvqa", "cdvqa", "your_dataset"])
```

### Step 5: Add Tests

Create tests in `tests/test_your_dataset.py`:

```python
import unittest
from data.adapters import YourDataset

class TestYourDataset(unittest.TestCase):
    def test_dataset_creation(self):
        dataset = YourDataset(root="/fake/path")
        self.assertEqual(dataset.dataset_name, "YourDataset")
    
    def test_load_nonexistent_root(self):
        dataset = YourDataset(root="/nonexistent/path")
        dataset.load()
        self.assertTrue(dataset.is_loaded())

if __name__ == "__main__":
    unittest.main()
```

## Popular Remote Sensing Datasets to Consider

### Single-Image Datasets
- **xView**: Object detection in satellite imagery
- **DOTA**: Object detection in aerial images
- **UCMerce21**: Land use classification
- **AID**: Aerial image classification

### VQA Datasets
- **RSVQA X**: Enhanced RSVQA dataset
- **RSIVQA**: Remote sensing image VQA
- **RSVQA_LR**: Low-resolution RSVQA

### Change Detection Datasets
- **LEVIR-CD**: Building change detection
- **DSIFN-CD": Multi-scale change detection
- **SYSU-CD": Building change detection

### Optical-SAR Fusion Datasets
- **SEN12MS**: Sentinel-1/2 fusion dataset
- **OpenSARShip**: SAR ship detection
- **SAR-Ship-Dataset**: SAR ship detection

### Multi-Modal Datasets
- **SSL4EO-S12**: Self-supervised learning dataset
- **TorchGeo Datasets**: Various geospatial datasets

## Sample Dataset Implementations

### Simple Image Classification Dataset

```python
class SimpleClassificationDataset(BaseRemoteSensingDataset):
    def load(self):
        root_path = Path(self.root)
        images_dir = root_path / "images"
        
        for image_file in images_dir.glob("*.jpg"):
            sample = Sample(
                sample_id=image_file.stem,
                dataset=self.dataset_name,
                image_path=str(image_file),
                modality=Modality.OPTICAL,
                labels=["class_name"]  # Add your labels
            )
            self._samples.append(sample)
```

### VQA Dataset

```python
class VQADataset(BaseRemoteSensingDataset):
    def load(self):
        # Load VQA annotations
        annotations = self._load_annotations()
        
        for item in annotations:
            sample = Sample(
                sample_id=item["id"],
                dataset=self.dataset_name,
                image_path=item["image_path"],
                question=item["question"],
                answer=item["answer"],
                modality=Modality.OPTICAL
            )
            self._samples.append(sample)
```

### Bi-Temporal Change Detection Dataset

```python
class ChangeDetectionDataset(BaseRemoteSensingDataset):
    def load(self):
        root_path = Path(self.root)
        t1_dir = root_path / "t1"
        t2_dir = root_path / "t2"
        
        for t1_file in t1_dir.glob("*.jpg"):
            t2_file = t2_dir / t1_file.name.replace("_t1", "_t2")
            
            if t2_file.exists():
                sample = Sample(
                    sample_id=t1_file.stem,
                    dataset=self.dataset_name,
                    image_t1_path=str(t1_file),
                    image_t2_path=str(t2_file),
                    modality=Modality.OPTICAL
                )
                self._samples.append(sample)
```

## Testing Your New Dataset

### Test the Adapter

```python
from data.adapters import YourDataset

dataset = YourDataset(
    root="/path/to/your/dataset",
    subset_size=10
)
dataset.load()

print(f"Loaded {len(dataset)} samples")
print(f"Dataset name: {dataset.dataset_name}")
print(f"Supports VQA: {dataset.supports_vqa}")
```

### Test with CLI

```bash
python -m data.cli validate --dataset your_dataset --root /path/to/dataset
```

### Test Preprocessing

```python
from data.preprocessing import ImagePreprocessor, PreprocessingConfig

config = PreprocessingConfig(target_size=(224, 224))
preprocessor = ImagePreprocessor(config)

# Test with a sample image
image_data, metadata = preprocessor.load_image("path/to/image.jpg")
```

## Best Practices

1. **Start Small**: Begin with a small subset_size during development
2. **Lazy Loading**: Implement lazy loading for large datasets
3. **Error Handling**: Handle missing files gracefully
4. **Documentation**: Document dataset structure and requirements
5. **Testing**: Write unit tests for your adapter
6. **Configuration**: Use environment variables for paths
7. **Validation**: Implement validation for your specific data format

## Hardware Considerations

For your hardware (8GB RAM / 4GB VRAM):

- **Small Datasets (<1000 samples)**: Can load entirely
- **Medium Datasets (1000-10000 samples)**: Use subset_size
- **Large Datasets (>10000 samples)**: Use streaming/lazy loading
- **Full BigEarthNet (590K samples)**: Requires streaming and batch_size=1

## Example: Adding SSL4EO-S12

SSL4EO-S12 is a great candidate for future addition:

```python
class SSL4EOS12Dataset(BaseRemoteSensingDataset):
    """SSL4EO-S12 self-supervised learning dataset."""
    
    def __init__(self, root: str, subset_size: Optional[int] = None, ...):
        super().__init__(root, subset_size, ...)
        self.regions = ["america_south", "america_north", ...]
    
    def load(self):
        for region in self.regions:
            # Load Sentinel-2 patches for each region
            # Implement SSL4EO-S12 specific loading logic
            pass
```

## Summary

Adding new datasets to SatQuery AI is straightforward:

1. ✅ Create adapter implementing `BaseRemoteSensingDataset`
2. ✅ Register in `__init__.py` files
3. ✅ Add configuration to `datasets.yaml`
4. ✅ Optionally update CLI
5. ✅ Add tests
6. ✅ Test integration

The modular architecture makes it easy to add unlimited datasets without affecting existing functionality.
