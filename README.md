# SatQueryAI - Remote Sensing AI Assistant

SatQueryAI is a remote-sensing AI assistant designed to support multiple vision-language tasks for satellite imagery analysis.

## Project Status

**Phase 1: GeoChat VQA Baseline** ✅

This phase established a baseline for single-image remote sensing visual question answering using the GeoChat model (CVPR 2024).

**Phase 2: Dataset Collection & Preparation** ✅

This phase implements a comprehensive data pipeline for remote sensing datasets, supporting:
- Multiple dataset adapters (BigEarthNet, VRSBench, RSVQA, CDVQA)
- GeoTIFF/TIFF preprocessing pipeline
- Reproducible data splits
- Pair compatibility checking (bi-temporal, optical-SAR)
- RemoteCLIP preprocessing integration
- Image tiling for large satellite imagery
- Model configuration for encoder freezing

**Current Implementation Status:**
- ✅ Repository structure created
- ✅ Environment configuration (environment.yaml)
- ✅ GeoChat inference interface implemented
- ✅ RemoteCLIP VQA model implemented (lightweight alternative)
- ✅ Dataset adapters for BigEarthNet, VRSBench, RSVQA, CDVQA
- ✅ Preprocessing pipeline with TIFF/GeoTIFF support
- ✅ Data validation and compatibility checking
- ✅ Reproducible data splits
- ✅ RemoteCLIP preprocessing integration
- ✅ Image tiling strategy
- ✅ Model configuration with encoder freezing
- ✅ CLI for dataset operations
- ✅ Comprehensive unit tests
- ✅ Dataset setup documentation
- ⚠️ GeoChat package installation compatibility issues with Python 3.12+ (Phase 1)
- ⚠️ Real dataset testing pending dataset download

## GeoChat Installation Compatibility Issues

**Status**: ✅ **RESOLVED** - Python 3.10 environment successfully set up

**Original Problem**: GeoChat has strict dependency requirements that conflict with modern Python environments:

- **Required**: Python 3.10, PyTorch 2.0.1, transformers 4.31.0
- **Current Environment**: Python 3.12/3.13, newer library versions
- **Issue**: Old dependencies lack pre-built wheels for Python 3.12+, and source compilation fails on Windows

**Solution Implemented**: ✅ Python 3.10 Environment Setup

### Completed Steps:
1. ✅ Installed Python 3.10.11 via winget
2. ✅ Created dedicated Python 3.10 virtual environment (`venv310/`)
3. ✅ Installed PyTorch 2.0.1 and torchvision 0.15.2
4. ✅ Installed transformers 4.31.0 and tokenizers 0.13.3
5. ✅ Installed GeoChat dependencies (accelerate 0.21.0, peft 0.4.0, bitsandbytes 0.41.0, etc.)
6. ✅ Installed GeoChat package with modified dependencies (removed deepspeed, markdown2[all])
7. ✅ Fixed NumPy compatibility (downgraded to numpy<2)
8. ✅ Updated inference interface to use GeoChat's image processing
9. ✅ Tested GeoChat import and CLI interface - **WORKING**

### How to Use the Working Environment:

```bash
# Activate the Python 3.10 environment
cd D:\projects\satquery-ai
venv310\Scripts\activate

# Run inference
python scripts/run_vqa.py --image test_image.jpg --question "What type of land cover is visible in this image?" --device cpu
```

### Current Status:
- ✅ GeoChat package successfully installed and importable
- ✅ CLI interface working (help command successful)
- ✅ All dependencies properly configured
- ⚠️ Model download failed due to network connection (retriable)
- ⚠️ Bitsandbytes compiled without GPU support (CPU inference only)

### Remaining Steps:
1. **Model Download**: Retry model download when network is stable
2. **GPU Support**: Install CUDA version of bitsandbytes for GPU inference
3. **Testing**: Run real inference test once model is downloaded

### Alternative Solutions (if network issues persist):

### Option 2: Use Docker Container
```bash
# Use Python 3.10 base image
docker run -it python:3.10 bash
# Follow GeoChat installation instructions
```

### Option 3: Alternative Remote Sensing VLMs
Consider more modern alternatives with better Python 3.12+ support:
- BLIP-2, InstructBLIP (Salesforce)
- OpenCLIP, CLIP models
- Recent HuggingFace vision-language models

## Project Structure

```
satquery-ai/
├── models/
│   ├── base.py                    # Base model interface
│   ├── vqa/
│   │   ├── geochat/               # GeoChat VQA (Phase 1)
│   │   │   ├── __init__.py
│   │   │   ├── inference.py
│   │   │   └── README.md
│   │   └── remoteclip_vqa/        # RemoteCLIP VQA (Phase 2)
│   │       ├── __init__.py
│   │       └── remoteclip_vqa.py
│   └── __init__.py
├── data/                          # Data pipeline (Phase 2)
│   ├── base.py                    # Base dataset interface
│   ├── adapters/                  # Dataset adapters
│   │   ├── bigearthnet.py         # BigEarthNet adapter
│   │   ├── vrsbench.py            # VRSBench adapter
│   │   ├── rsvqa.py               # RSVQA adapter
│   │   └── cdvqa.py               # CDVQA adapter
│   ├── preprocessing.py           # Image preprocessing pipeline
│   ├── remoteclip_preprocessing.py # RemoteCLIP-specific preprocessing
│   ├── validation.py              # Data validation utilities
│   ├── compatibility.py           # Pair compatibility checker
│   ├── splits.py                  # Data splitting utilities
│   ├── manifest.py                # Dataset manifest/index
│   └── cli.py                     # Data pipeline CLI
├── scripts/
│   ├── run_vqa.py                 # VQA inference script
│   ├── test_interface.py          # Interface testing
│   └── test_remoteclip.py         # RemoteCLIP testing
├── tests/
│   ├── test_data_layer.py         # Data layer tests
│   ├── test_remoteclip_integration.py # RemoteCLIP integration tests
│   ├── test_geochat_inference.py  # GeoChat tests
│   └── test_remoteclip_vqa.py    # RemoteCLIP VQA tests
├── configs/
│   ├── datasets.yaml              # Dataset configuration
│   ├── model_config.yaml          # Model configuration
│   └── geochat_config.yaml        # GeoChat configuration
├── docs/
│   └── DATASET_SETUP.md           # Dataset setup guide
├── environment.yaml
└── README.md
```

## Current Capabilities

### Data Pipeline (Phase 2)

- **Datasets**: BigEarthNet, VRSBench, RSVQA, CDVQA
- **Preprocessing**: TIFF/GeoTIFF handling, normalization, tiling
- **Validation**: Image validation, format checking, data integrity
- **Compatibility**: Bi-temporal and optical-SAR pair checking
- **Splits**: Reproducible train/val/test splits
- **RemoteCLIP Integration**: Specialized preprocessing for RemoteCLIP
- **Status**: ✅ Implemented and tested

### RemoteCLIP VQA (Phase 2)

- **Model**: RemoteCLIP ViT-B/32 (RS-adapted CLIP)
- **Task**: Lightweight single-image VQA
- **Input**: RGB satellite images + natural language questions
- **Output**: Text answers with confidence scores
- **Features**: Encoder freezing for limited GPU, task-specific heads
- **Status**: ✅ Implemented and tested

### GeoChat VQA (Phase 1)

- **Model**: GeoChat-7B (Vicuna-13B-v1.3 + CLIP-L-336px)
- **Task**: Single-image visual question answering
- **Input**: RGB satellite images + natural language questions
- **Output**: Text answers (no confidence scores)
- **Status**: ✅ Implemented (requires Python 3.10 environment)

## Planned Capabilities

Future phases will add:

- Single-image grounding/captioning
- Bi-temporal change detection / change VQA
- Optical + SAR cross-modal fusion
- Agentic routing between specialist models
- Custom fine-tuning with BigEarthNet

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate satquery-ai
```

### 2. Dataset Setup

Configure dataset paths in `configs/datasets.yaml` or via environment variables:

```bash
# Set dataset paths
export BIGEARTHNET_ROOT=/path/to/bigearthnet
export VRSBENCH_ROOT=/path/to/vrsbench
export RSVQA_ROOT=/path/to/rsvqa
export CDVQA_ROOT=/path/to/cdvqa
```

See [docs/DATASET_SETUP.md](docs/DATASET_SETUP.md) for detailed dataset setup instructions.

### 3. Data Pipeline Operations

```bash
# Validate a dataset
python -m data.cli validate --dataset bigearthnet --root /path/to/bigearthnet

# Prepare dataset (generate manifest)
python -m data.cli prepare --dataset bigearthnet --root /path/to/bigearthnet --limit 100

# Test pair compatibility
python -m data.cli compatibility --type temporal --image1 /path/to/t1.jpg --image2 /path/to/t2.jpg
```

### 4. Run Inference

**RemoteCLIP VQA (Recommended for limited GPU):**

```bash
python scripts/test_remoteclip.py
```

**GeoChat VQA (Requires Python 3.10):**

```bash
# Activate Python 3.10 environment
venv310\Scripts\activate

python scripts/run_vqa.py \
    --image path/to/satellite_image.jpg \
    --question "What type of land cover is visible in this image?"
```

### 5. Python API

**RemoteCLIP VQA:**

```python
from models.vqa.remoteclip_vqa import create_inference_engine

engine = create_inference_engine(
    model_name="ViT-B-32",
    device="cuda",
    freeze_encoder=True
)
result = engine.predict(
    image_path="path/to/image.jpg",
    question="What is in this image?"
)
print(result["answer"])
```

**Data Pipeline:**

```python
from data.adapters import BigEarthNetDataset
from data.preprocessing import ImagePreprocessor, PreprocessingConfig

# Load dataset
dataset = BigEarthNetDataset(
    root="/path/to/bigearthnet",
    subset_size=100
)
dataset.load()

# Preprocess images
config = PreprocessingConfig(target_size=(224, 224))
preprocessor = ImagePreprocessor(config)
```

## Hardware Requirements

### RemoteCLIP VQA (Phase 2 - Recommended)

- **GPU**: 4GB+ VRAM (RTX 2050 4GB sufficient)
- **RAM**: 8GB+ (configurable with subset sizes)
- **Storage**: 1GB+ for RemoteCLIP model weights
- **Suitable for**: Development, limited GPU environments

### GeoChat VQA (Phase 1)

- **GPU**: 16GB+ VRAM (24GB+ recommended)
- **RAM**: 32GB+ (64GB+ recommended)
- **Storage**: 20GB+ for model weights
- **Requires**: Python 3.10 environment

### Data Pipeline

- **RAM**: 8GB+ (configurable with subset sizes and streaming)
- **Storage**: Varies by dataset size
- **CPU**: Modern multi-core processor recommended

See [models/vqa/geochat/README.md](models/vqa/geochat/README.md) for detailed GeoChat requirements.

## Testing

```bash
# Run data layer tests
python -m unittest tests.test_data_layer -v

# Run RemoteCLIP integration tests
python -m unittest tests.test_remoteclip_integration -v

# Run GeoChat tests (requires Python 3.10 environment)
venv310\Scripts\activate
python -m unittest tests.test_geochat_inference -v

# Run all tests
python -m unittest discover tests -v
```

**Note**: 
- Data layer tests are designed to run without datasets installed
- RemoteCLIP integration tests require PIL and numpy
- GeoChat tests require Python 3.10 environment with GeoChat installed

## Documentation

- [Dataset Setup Guide](docs/DATASET_SETUP.md) - Comprehensive dataset configuration and setup
- [GeoChat Integration Guide](models/vqa/geochat/README.md)
- [GeoChat Official Repository](https://github.com/mbzuai-oryx/GeoChat)
- [GeoChat Paper](https://arxiv.org/abs/2311.15826)
- [RemoteCLIP Paper](https://arxiv.org/abs/2307.15926)

## Implementation Strategy

This project follows an **ADAPT & ORCHESTRATE** approach:

- Leverage pretrained remote-sensing models (RemoteCLIP, GeoChat, SSL4EO-S12)
- Use existing datasets (BigEarthNet, VRSBench, RSVQA, CDVQA)
- Build custom modules for change detection and optical-SAR fusion
- Implement agentic routing between specialist models

## Architecture

### Data Layer (Phase 2)

The data layer provides a modular, extensible pipeline for remote sensing datasets:

```
BaseRemoteSensingDataset (abstract interface)
    ├── BigEarthNetDataset
    ├── VRSBenchDataset
    ├── RSVQADataset
    └── CDVQADataset

Preprocessing Pipeline:
    Raw Satellite Data → Format Detection → Metadata Inspection → 
    Band Validation → Spatial Validation → Normalization → 
    Resize/Crop/Tile → Tensor Conversion → Model Input

Compatibility Checking:
    - Bi-temporal pair compatibility
    - Optical-SAR pair compatibility
    - Spatial alignment verification
```

### Model Layer

The model layer supports multiple specialist models:

```
RemoteSensingModel (abstract interface)
    ├── VQAModel
    │   ├── GeoChat (Phase 1 baseline)
    │   └── RemoteCLIP VQA (Phase 2 lightweight)
    ├── CaptioningModel (future)
    ├── GroundingModel (future)
    ├── ChangeDetectionModel (future)
    └── OpticalSARFusionModel (future)
```

### Dependency Direction

The architecture follows a strict dependency hierarchy:

```
Frontend → FastAPI → Agentic Controller → Model → Data/Preprocessing Layer
```

The data layer is independent from:
- FastAPI
- React
- Agentic Controller
- Individual ML models

This ensures the data layer can be reused for training, evaluation, and inference.

## License

This project uses GeoChat (Apache 2.0) and other open-source models. See individual model licenses for details.

## Citation

If you use this code, please cite the relevant papers:

```bibtex
@inproceedings{kuckreja2024geochat,
  title={GeoChat: Grounded Large Vision-Language Model for Remote Sensing},
  author={Kuckreja, Kartik and Danish, Muhammad Sohail and Naseer, Muzammal and Das, Abhijit and Khan, Salman and Khan, Fahad S},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## Contact

For questions about this project, please open an issue on the repository.
