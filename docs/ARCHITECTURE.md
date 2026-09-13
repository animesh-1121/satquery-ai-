# SatQueryAI Architecture Documentation

This document describes the architecture of SatQueryAI, focusing on the data layer and model integration.

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Layer                        │
│                     (React / Web UI)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       API Layer                              │
│                      (FastAPI)                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Agentic Controller                           │
│          (Task routing and orchestration)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       Model Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ RemoteCLIP   │  │   GeoChat    │  │ Future       │     │
│  │ VQA          │  │ VQA          │  │ Models       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Dataset      │  │ Preprocess-  │  │ Validation   │     │
│  │ Adapters     │  │ ing          │  │ & Compat-    │     │
│  └──────────────┘  └──────────────┘  │ ibility      │     │
│                                        └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Data Layer Architecture

### Design Principles

1. **Modularity**: Each dataset adapter is independent and pluggable
2. **Extensibility**: New datasets can be added by implementing the base interface
3. **Reproducibility**: Deterministic splits and configurations
4. **Independence**: Data layer does not depend on models, API, or frontend
5. **Performance**: Streaming/lazy loading for large datasets

### Core Components

#### 1. Base Dataset Interface

```python
class BaseRemoteSensingDataset(ABC):
    """Abstract base class for all remote sensing datasets."""
    
    @abstractmethod
    def load(self) -> None:
        """Load the dataset samples."""
        pass
    
    @abstractmethod
    def get_split(self, split: str) -> List[Sample]:
        """Get samples for a specific split."""
        pass
    
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the dataset name."""
        pass
    
    @abstractmethod
    def supports_vqa(self) -> bool:
        """Whether this dataset supports VQA tasks."""
        pass
```

#### 2. Normalized Sample Representation

```python
@dataclass
class Sample:
    """Normalized sample representation for remote sensing data."""
    
    sample_id: str
    dataset: str
    
    # Image paths
    image_path: Optional[str] = None
    image_t1_path: Optional[str] = None  # Bi-temporal
    image_t2_path: Optional[str] = None  # Bi-temporal
    optical_path: Optional[str] = None   # Optical-SAR
    sar_path: Optional[str] = None       # Optical-SAR
    
    # Annotations
    question: Optional[str] = None
    answer: Optional[str] = None
    caption: Optional[str] = None
    labels: Optional[List[str]] = None
    
    # Metadata
    modality: Modality = Modality.UNKNOWN
    sensor: Optional[str] = None
    bands: Optional[List[str]] = None
    resolution: Optional[float] = None
    crs: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
```

#### 3. Dataset Adapters

Each dataset implements the base interface:

```
BaseRemoteSensingDataset
        │
        ├── BigEarthNetDataset
        ├── VRSBenchDataset
        ├── RSVQADataset
        └── CDVQADataset
```

**BigEarthNet**:
- Primary adaptation dataset
- Multi-label classification (43 classes)
- Sentinel-2 (optical) or Sentinel-1 (SAR)
- Official train/val/test splits

**VRSBench**:
- Single-image evaluation
- VQA, captioning, grounding
- Optical imagery

**RSVQA**:
- Remote sensing VQA
- Question type preservation
- Optical imagery

**CDVQA**:
- Bi-temporal change detection
- Change VQA
- Optical imagery

### Preprocessing Pipeline

#### Pipeline Stages

```
Raw Satellite Data
       ↓
Format Detection (TIFF, GeoTIFF, PNG, JPG)
       ↓
Metadata Inspection (CRS, transform, bands)
       ↓
Band Validation (spectral band preservation)
       ↓
Spatial Validation (dimensions, coverage)
       ↓
Normalization (pixel value scaling)
       ↓
Resize / Crop / Tile (image preparation)
       ↓
Tensor Conversion (CHW format)
       ↓
Model Input
```

#### Preprocessing Configuration

```yaml
preprocessing:
  target_size: [224, 224]  # RemoteCLIP input size
  tile_size: null  # Enable tiling for large images
  overlap: 0  # Tile overlap in pixels
  max_size: [1024, 1024]  # Maximum image size
  normalize: true  # Normalize to [0, 1]
  preserve_bands: true  # Preserve spectral bands
  resize_method: "resize"  # 'crop', 'pad', 'resize'
  band_selection: null  # Specific band selection
```

#### Image Tiling

For large satellite imagery, the system supports tiling:

```python
config = PreprocessingConfig(
    tile_size=(256, 256),
    overlap=32,
    enable_tiling=True
)

preprocessor = ImagePreprocessor(config)
tiles, metadata = preprocessor.tile_image(image_data, metadata)
```

Tiling metadata includes:
- Number of tiles
- Tile dimensions
- Overlap information
- Tile spatial coordinates

### Data Validation

#### Validation Checks

- **File Existence**: Verify image files exist
- **File Integrity**: Check for corrupt/empty files
- **Format Validation**: Supported formats (TIFF, GeoTIFF, PNG, JPG)
- **Dimension Validation**: Valid image dimensions
- **Metadata Validation**: Required metadata present
- **Annotation Validation**: Complete annotations
- **Pair Validation**: T1/T2 or optical/SAR pairs complete

#### Validation Results

```python
@dataclass
class ValidationResult:
    valid: bool
    total_samples: int
    valid_samples: int
    invalid_samples: int
    errors: List[str]
    warnings: List[str]
```

### Compatibility Checking

#### Bi-Temporal Pairs

Checks compatibility between T1 and T2 images:

- **Dimensions**: Width and height match
- **CRS**: Coordinate reference systems match
- **Transform**: Geospatial transforms match
- **Spatial Alignment**: Bounds overlap
- **Misalignment**: Pixel-level offset within tolerance

```python
result = checker.check_temporal_pair(
    image_t1_path="t1.tif",
    image_t2_path="t2.tif",
    check_spatial=True
)
```

#### Optical-SAR Pairs

Checks compatibility between optical and SAR images:

- **Modality**: Correct modality identification
- **Dimensions**: Width and height match
- **CRS**: Coordinate reference systems match
- **Spatial Alignment**: Bounds overlap

```python
result = checker.check_optical_sar_pair(
    optical_path="optical.tif",
    sar_path="sar.tif",
    check_spatial=True
)
```

#### Compatibility Results

```python
@dataclass
class CompatibilityResult:
    compatible: bool
    errors: List[str]
    warnings: List[str]
    checks: Dict[str, Any]  # Individual check results
```

### Data Splits

#### Reproducible Splits

Deterministic splits with fixed random seeds:

```python
splitter = DataSplitter(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
    use_official_splits=True
)

train, val, test = splitter.split_samples(samples, "BigEarthNet")
```

#### Official Splits

If datasets provide official splits, they are preserved:

```python
# BigEarthNet has official splits
train_samples = dataset.get_split("train")
val_samples = dataset.get_split("val")
test_samples = dataset.get_split("test")
```

#### Split Persistence

Split information can be saved for reproducibility:

```python
splitter.save_split_info(
    train_samples=train,
    val_samples=val,
    test_samples=test,
    output_path="splits/bigearthnet_split.json"
)
```

### Dataset Manifest

#### Purpose

The manifest provides a normalized index of all samples across datasets:

```python
manifest = DatasetManifest(manifest_path="manifest.json")
manifest.add_samples(dataset._samples)
manifest.save()
```

#### Manifest Structure

```json
{
  "metadata": {
    "created_at": "2024-01-01T00:00:00",
    "version": "1.0",
    "datasets": ["BigEarthNet", "VRSBench"],
    "total_samples": 150
  },
  "samples": {
    "sample_00001": {
      "sample_id": "sample_00001",
      "dataset": "BigEarthNet",
      "image_path": "/path/to/image.tif",
      "modality": "optical",
      "labels": ["forest", "water"],
      "metadata": {...}
    }
  }
}
```

#### Querying

```python
# Get samples by dataset
bigearthnet_samples = manifest.get_samples_by_dataset("BigEarthNet")

# Get samples by modality
optical_samples = manifest.get_samples_by_modality("optical")

# Get bi-temporal samples
temporal_samples = manifest.get_bi_temporal_samples()

# Get optical-SAR pairs
fusion_samples = manifest.get_optical_sar_samples()
```

## Model Layer Architecture

### Base Model Interface

```python
class RemoteSensingModel(ABC):
    """Abstract base class for all remote sensing models."""
    
    @abstractmethod
    def load(self) -> None:
        """Load the model weights."""
        pass
    
    @abstractmethod
    def predict(self, **kwargs) -> Dict[str, Any]:
        """Run inference."""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Return the model type."""
        pass
```

### Model Types

```python
class ModelType(Enum):
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
```

### RemoteCLIP VQA Architecture

```
Question
   ↓
Question Parser (rule-based task mapping)
   ↓
RemoteCLIP Encoder (RS-adapted vision encoder)
   ↓
Image Embedding
   ↓
Task-Specific MLP Head
   ↓
Classification Probabilities
   ↓
Answer Formatter
   ↓
Natural Language Answer
```

#### Encoder Freezing

For limited GPU environments, the encoder can be frozen:

```python
model = RemoteCLIPVQA(
    model_name="ViT-B-32",
    device="cuda",
    freeze_encoder=True  # Freeze RemoteCLIP encoder
)
```

Configuration:

```yaml
models:
  remoteclip_vqa:
    freeze_encoder: true
    freeze_encoder_bn: true
```

#### Task-Specific Heads

Each VQA task has a dedicated classification head:

```python
TASK_CONFIGS = {
    VQATask.LAND_COVER: {
        "classes": ["water", "forest", "agricultural", ...],
        "templates": ["A satellite image of {class}.", ...]
    },
    VQATask.WATER_PRESENCE: {
        "classes": ["no water", "water present"],
        "templates": [...]
    }
}
```

### RemoteCLIP Preprocessing Integration

Specialized preprocessing for RemoteCLIP:

```python
preprocessor = RemoteCLIPPreprocessor(
    target_size=(224, 224),
    normalize=True
)

image, metadata = preprocessor.preprocess_for_remoteclip(image_path)
```

Preprocessing steps:
1. Load image with PIL
2. Convert to RGB
3. Resize to target size
4. Apply RemoteCLIP normalization (ImageNet stats)
5. Convert to CHW format
6. Return tensor ready for RemoteCLIP

## Dependency Flow

### Correct Dependency Direction

```
Frontend (React)
    ↓
API (FastAPI)
    ↓
Agentic Controller
    ↓
Model Layer
    ↓
Data Layer
```

### Anti-Patterns to Avoid

❌ **Don't do this:**
```
Data Layer → FastAPI
Data Layer → React
Data Layer → Agentic Controller
```

✅ **Do this instead:**
```
Frontend → API → Controller → Model → Data
```

This ensures:
- Data layer is reusable for training, evaluation, and inference
- No circular dependencies
- Easy testing of data layer in isolation
- Models can be swapped without affecting data layer

## Configuration Management

### Dataset Configuration

```yaml
# configs/datasets.yaml
datasets:
  bigearthnet:
    root: ${BIGEARTHNET_ROOT:-/path/to/bigearthnet}
    enabled: true
    subset_size: 100
    split: "s2"
```

### Model Configuration

```yaml
# configs/model_config.yaml
models:
  remoteclip_vqa:
    enabled: true
    model_name: "ViT-B-32"
    device: "cuda"
    freeze_encoder: true
```

### Environment Variables

Configuration supports environment variable substitution:

```bash
export BIGEARTHNET_ROOT=/path/to/bigearthnet
export REMOTECLIP_CACHE_DIR=/path/to/cache
```

## Hardware Constraints

### Memory Management

For 8 GB RAM / 4 GB VRAM:

```yaml
hardware:
  batch_size: 1
  num_workers: 0  # Disable multiprocessing
  pin_memory: false
  gradient_accumulation_steps: 4  # Effective batch size = 4
```

### Subset Configuration

```yaml
datasets:
  bigearthnet:
    subset_size: 100  # Use only 100 samples
```

### Streaming

Dataset adapters use lazy loading to avoid loading entire datasets into RAM:

```python
# Samples are loaded on-demand
for sample in dataset:
    process(sample)  # Only current sample in memory
```

## Testing Strategy

### Unit Tests

- Dataset adapter tests (mock data)
- Validation tests (edge cases)
- Compatibility tests (pair checking)
- Preprocessing tests (image operations)

### Integration Tests

- RemoteCLIP preprocessing with real images
- Model inference with real data
- End-to-end pipeline tests

### Test Structure

```
tests/
├── test_data_layer.py              # Data layer unit tests
├── test_remoteclip_integration.py  # RemoteCLIP integration tests
├── test_geochat_inference.py       # GeoChat tests
└── test_remoteclip_vqa.py         # RemoteCLIP VQA tests
```

## Future Extensions

### Planned Models

- **Change Detection**: Siamese architecture for bi-temporal analysis
- **Optical-SAR Fusion**: Cross-modal attention for multi-sensor analysis
- **Captioning**: Image captioning for remote sensing
- **Grounding**: Visual grounding for remote sensing objects

### Planned Datasets

- **SSL4EO-S12**: Self-supervised learning dataset
- **xView**: Object detection dataset
- **DOTA**: Object detection in aerial images

### Planned Features

- **Active Learning**: Sample selection for efficient training
- **Data Augmentation**: Remote sensing-specific augmentations
- **Multi-modal Fusion**: Advanced fusion strategies
- **Explainability**: Model interpretation tools
