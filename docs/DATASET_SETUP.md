# Dataset Setup Guide for SatQueryAI

This guide explains how to set up and configure the datasets required for SatQueryAI Phase 2.

## Overview

SatQueryAI Phase 2 supports the following remote sensing datasets:

- **BigEarthNet**: Primary adaptation dataset for remote-sensing domain-specific training
- **VRSBench**: Single-image evaluation dataset (VQA, captioning, grounding)
- **RSVQA**: Remote sensing visual question answering dataset
- **CDVQA**: Change detection visual question answering dataset (bi-temporal)

## Dataset Configuration

Dataset paths and settings are configured in `configs/datasets.yaml`. The configuration uses environment variables for flexibility:

```yaml
datasets:
  bigearthnet:
    root: ${BIGEARTHNET_ROOT:-/path/to/bigearthnet}
    enabled: true
    subset_size: 100  # Use only 100 samples for development
    split: "s2"  # Sentinel-2 (optical) or Sentinel-1 (SAR)
```

### Setting Environment Variables

You can set dataset paths via environment variables:

```bash
# Windows (Command Prompt)
set BIGEARTHNET_ROOT=D:\datasets\BigEarthNet
set VRSBENCH_ROOT=D:\datasets\VRSBench
set RSVQA_ROOT=D:\datasets\RSVQA
set CDVQA_ROOT=D:\datasets\CDVQA

# Windows (PowerShell)
$env:BIGEARTHNET_ROOT="D:\datasets\BigEarthNet"
$env:VRSBENCH_ROOT="D:\datasets\VRSBench"
$env:RSVQA_ROOT="D:\datasets\RSVQA"
$env:CDVQA_ROOT="D:\datasets\CDVQA"

# Linux/Mac
export BIGEARTHNET_ROOT=/path/to/bigearthnet
export VRSBENCH_ROOT=/path/to/vrsbench
export RSVQA_ROOT=/path/to/rsvqa
export CDVQA_ROOT=/path/to/cdvqa
```

## BigEarthNet

### Dataset Information

- **Paper**: "BigEarthNet: A Large-Scale Benchmark Archive of Remote Sensing Images"
- **Size**: ~590,000 Sentinel-2 image patches
- **Task**: Multi-label classification with 43 land cover classes
- **Modality**: Sentinel-2 (optical, 13 spectral bands) or Sentinel-1 (SAR)
- **Resolution**: 10-60m (Sentinel-2)

### How to Obtain

1. Visit the official BigEarthNet website: https://bigearthnet.github.io/
2. Download the dataset (requires registration)
3. Choose between:
   - **BigEarthNet-S2**: Sentinel-2 optical imagery
   - **BigEarthNet-S1**: Sentinel-1 SAR imagery

### Expected Directory Structure

```
BigEarthNet-S2/
├── train/
│   ├── patch_1/
│   │   ├── patch_1.tif
│   │   └── patch_1_labels.csv
│   ├── patch_2/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### Configuration

```yaml
datasets:
  bigearthnet:
    root: ${BIGEARTHNET_ROOT:-/path/to/bigearthnet}
    enabled: true
    subset_size: 100  # Set to null for full dataset
    split: "s2"  # "s1" for SAR, "s2" for optical
    modality: "optical"
```

### Usage

```python
from data.adapters import BigEarthNetDataset

dataset = BigEarthNetDataset(
    root="/path/to/BigEarthNet-S2",
    subset_size=100,  # Use 100 samples for development
    split="s2"
)
dataset.load()

# Get train/val/test splits
train_samples = dataset.get_split("train")
val_samples = dataset.get_split("val")
test_samples = dataset.get_split("test")
```

## VRSBench

### Dataset Information

- **Task**: Remote sensing VQA, captioning, and grounding
- **Modality**: Optical satellite imagery
- **Annotations**: Questions, answers, captions, bounding boxes

### How to Obtain

VRSBench is available from the official repository. Contact the authors or check the official GitHub repository for download instructions.

### Expected Directory Structure

```
VRSBench/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── vqa/
│   └── vqa_annotations.json
├── captions/
│   └── caption_annotations.json
└── grounding/
    └── grounding_annotations.json
```

### Configuration

```yaml
datasets:
  vrsbench:
    root: ${VRSBENCH_ROOT:-/path/to/vrsbench}
    enabled: true
    subset_size: 50  # Set to null for full dataset
```

### Usage

```python
from data.adapters import VRSBenchDataset

dataset = VRSBenchDataset(
    root="/path/to/VRSBench",
    subset_size=50
)
dataset.load()
```

## RSVQA

### Dataset Information

- **Task**: Remote sensing visual question answering
- **Modality**: Optical satellite imagery
- **Features**: Question type/category information preserved

### How to Obtain

RSVQA is available from the official repository. Check the official GitHub repository for download instructions.

### Expected Directory Structure

```
RSVQA/
├── images/
│   ├── image_001.jpg
│   └── ...
└── annotations/
    └── rsvqa_annotations.json
```

### Configuration

```yaml
datasets:
  rsvqa:
    root: ${RSVQA_ROOT:-/path/to/rsvqa}
    enabled: true
    subset_size: 50
```

### Usage

```python
from data.adapters import RSVQADataset

dataset = RSVQADataset(
    root="/path/to/RSVQA",
    subset_size=50
)
dataset.load()
```

## CDVQA

### Dataset Information

- **Task**: Change detection visual question answering
- **Modality**: Bi-temporal optical imagery
- **Features**: T1/T2 image pairs with change annotations

### How to Obtain

CDVQA is available from the official repository. Check the official GitHub repository for download instructions.

### Expected Directory Structure

```
CDVQA/
├── t1/
│   ├── image_001_t1.jpg
│   └── ...
├── t2/
│   ├── image_001_t2.jpg
│   └── ...
└── annotations/
    └── cdvqa_annotations.json
```

### Configuration

```yaml
datasets:
  cdvqa:
    root: ${CDVQA_ROOT:-/path/to/cdvqa}
    enabled: true
    subset_size: 50
```

### Usage

```python
from data.adapters import CDVQADataset

dataset = CDVQADataset(
    root="/path/to/CDVQA",
    subset_size=50
)
dataset.load()
```

## Preprocessing Configuration

Preprocessing settings are configured in `configs/datasets.yaml`:

```yaml
preprocessing:
  target_size: [224, 224]  # Target size for RemoteCLIP
  tile_size: null  # Set to [width, height] to enable tiling
  overlap: 0  # Overlap between tiles in pixels
  max_size: [1024, 1024]  # Maximum size for large images
  normalize: true  # Normalize pixel values to [0, 1]
  preserve_bands: true  # Preserve spectral bands
  resize_method: "resize"  # 'crop', 'pad', or 'resize'
  band_selection: null  # Specific bands to select
```

## Data Splits

### Reproducible Splits

The system supports reproducible data splits with fixed random seeds:

```yaml
splits:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  use_official_splits: true  # Use official splits if available
```

### Creating Splits

```python
from data.splits import create_train_val_test_split

train, val, test = create_train_val_test_split(
    samples=dataset._samples,
    dataset_name="BigEarthNet",
    random_seed=42
)
```

## Validation

### Dataset Validation

Use the CLI to validate datasets:

```bash
python -m data.cli validate --dataset bigearthnet --root /path/to/bigearthnet
python -m data.cli validate --dataset vrsbench --root /path/to/vrsbench
python -m data.cli validate --dataset rsvqa --root /path/to/rsvqa
python -m data.cli validate --dataset cdvqa --root /path/to/cdvqa
```

### Pair Compatibility

Test temporal and optical-SAR pair compatibility:

```bash
# Test bi-temporal pair
python -m data.cli compatibility --type temporal --image1 /path/to/t1.jpg --image2 /path/to/t2.jpg

# Test optical-SAR pair
python -m data.cli compatibility --type optical_sar --image1 /path/to/optical.jpg --image2 /path/to/sar.tif
```

## Data Preparation

### Generate Manifest

Create a normalized manifest/index of processed samples:

```bash
python -m data.cli prepare --dataset bigearthnet --root /path/to/bigearthnet --limit 100 --output manifest.json
```

### Inspect Sample

Inspect a specific sample from the manifest:

```bash
python -m data.cli inspect --sample-id sample_00001 --manifest manifest.json
```

## Hardware Considerations

### Memory Constraints

For development machines with limited RAM (8 GB) and GPU (4 GB VRAM):

- Use small subset sizes (50-100 samples)
- Set batch size to 1
- Enable streaming/lazy loading
- Use CPU-safe preprocessing

```yaml
hardware:
  batch_size: 1
  num_workers: 0  # Disable multiprocessing on Windows
  pin_memory: false
```

### Subset Configuration

For development/testing:

```yaml
datasets:
  bigearthnet:
    subset_size: 100  # Use only 100 samples
  vrsbench:
    subset_size: 50
  rsvqa:
    subset_size: 50
  cdvqa:
    subset_size: 50
```

For production/training:

```yaml
datasets:
  bigearthnet:
    subset_size: null  # Use full dataset
```

## Compatibility Checking

### Bi-Temporal Pairs

The system automatically checks bi-temporal image pairs for:

- Dimension compatibility
- CRS/georeferencing alignment
- Spatial coverage overlap
- Transform/geospatial alignment

### Optical-SAR Pairs

The system checks optical-SAR pairs for:

- Modality identification
- Dimension compatibility
- Spatial alignment
- CRS consistency

### Error Handling

Incompatible pairs are either rejected or flagged with detailed error messages:

```
ERROR: Bi-temporal images have incompatible spatial dimensions.
T1: 512 × 512
T2: 640 × 512

Change detection execution blocked.
```

## Troubleshooting

### Dataset Not Found

If you see "Dataset root directory not found":

1. Check that the environment variable is set correctly
2. Verify the dataset path exists
3. Ensure the directory structure matches expectations

### Import Errors

If you see import errors for numpy/PIL:

```bash
# Install required dependencies
pip install numpy pillow rasterio
```

### Memory Issues

If you encounter memory issues:

1. Reduce subset_size in configuration
2. Enable streaming/lazy loading
3. Reduce batch_size to 1
4. Disable multiprocessing (num_workers: 0)

### Compatibility Issues

If pairs are flagged as incompatible:

1. Check image dimensions match
2. Verify CRS is consistent
3. Ensure spatial alignment
4. Review compatibility checker output

## Next Steps

After setting up datasets:

1. Run dataset validation: `python -m data.cli validate --dataset <name> --root <path>`
2. Generate manifest: `python -m data.cli prepare --dataset <name> --root <path>`
3. Test preprocessing with RemoteCLIP: `python -m unittest tests.test_remoteclip_integration`
4. Run full test suite: `python -m unittest discover tests`

## Additional Resources

- BigEarthNet: https://bigearthnet.github.io/
- TorchGeo: https://torchgeo.readthedocs.io/
- RemoteCLIP: https://github.com/ChenDelong1997/RemoteCLIP
