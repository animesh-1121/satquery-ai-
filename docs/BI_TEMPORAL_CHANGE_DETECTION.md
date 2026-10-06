# Bi-Temporal Change Detection (Phase 4)

## Overview

SatQueryAI Phase 4 implements a learned bi-temporal change detection system that accepts two satellite images of the same geographic area at different times and detects meaningful changes between them.

### Problem Statement

Given two satellite images of the same location acquired at different times (T1 and T2), the system must:

1. Determine whether meaningful change occurred
2. Classify the change (binary: no-change vs change)
3. Provide confidence in the prediction
4. Optionally, generate a change mask (future enhancement)

### Key Innovation

Instead of comparing raw pixels, we encode both temporal images using the **same remote-sensing encoder**. Because the encoder shares weights, both images are represented in the same feature space. We then model their feature-level differences to detect meaningful changes.

## Architecture

### Siamese Architecture with Shared Encoder

```
    T1 IMAGE          T2 IMAGE
       ↓                 ↓
    Shared RemoteCLIP ViT-B/32 (same weights)
       ↓                 ↓
    F1                  F2
       ↓                 ↓
    [F1, F2, |F1-F2|]  →  Temporal Feature Fusion  →  Change Detection Head  →  Change Classification
```

### Why Siamese Architecture?

1. **Shared Weights**: The same RemoteCLIP encoder processes both T1 and T2, ensuring they are represented in the same feature space
2. **Efficiency**: Only one encoder to train (frozen), reducing memory usage
3. **Consistency**: Same feature extraction for both temporal images
4. **Hardware Optimization**: Suitable for RTX 2050 4GB VRAM constraint

### Component Details

#### 1. Shared RemoteCLIP Encoder

- **Model**: RemoteCLIP ViT-B/32
- **Status**: Frozen (pretrained weights)
- **Output**: 512-dimensional visual features
- **Purpose**: Extract remote-sensing visual representations from both temporal images

#### 2. Temporal Feature Fusion

- **Input**: F1 (T1 features), F2 (T2 features)
- **Computation**: [F1, F2, |F1-F2|] concatenated
- **Output**: Projected temporal features (256 dimensions)
- **Purpose**: Model feature-level differences between temporal images

**Temporal Features**:
- **F1**: Features from T1 image
- **F2**: Features from T2 image
- **|F1-F2|**: Absolute difference (captures change information)

#### 3. Change Detection Head

- **Architecture**: Lightweight MLP classifier
- **Input**: Temporal features (256 dimensions)
- **Output**: Binary classification (no-change vs change)
- **Parameters**: ~200K trainable parameters
- **Purpose**: Predict whether meaningful change occurred

## Dataset

### CDVQA Dataset

The system is designed to work with the CDVQA (Change Detection Visual Question Answering) dataset.

**Dataset Information**:
- **Source**: Based on SECOND dataset
- **Image Pairs**: 2,968 bi-temporal aerial image pairs
- **Resolution**: 512×512 pixels, 0.5-3m spatial resolution
- **Locations**: Shanghai, Hangzhou, Chengdu (China)
- **Semantic Change Maps**: Pixel-level annotations for 7 land-cover classes
- **Format**: RGB optical images

**Expected Directory Structure**:
```
root/
    t1/
        image_001_t1.jpg
        ...
    t2/
        image_001_t2.jpg
        ...
    annotations/
        change_labels.json  # Optional: binary change labels
        change_masks/       # Optional: dense change masks
```

### Current Status

The CDVQA dataset is **not currently available** in the project repository. The infrastructure is complete and tested with synthetic data, but real training requires downloading the dataset separately.

## Training

### Training Pipeline

The training infrastructure includes:

1. **Dataset Loading**: T1/T2 image pairs with change labels
2. **Shared Encoding**: Both images pass through the same RemoteCLIP encoder
3. **Temporal Fusion**: [F1, F2, |F1-F2|] feature combination
4. **Change Prediction**: Binary classification
5. **Loss Computation**: CrossEntropyLoss
6. **Backpropagation**: Only temporal fusion and change head are updated
7. **Checkpointing**: Model state and metadata saved

### Training Configuration

```python
ChangeDetectionTrainingConfig(
    learning_rate=1e-4,
    weight_decay=0.01,
    batch_size=1,  # Small for 4GB VRAM
    epochs=10,
    gradient_accumulation_steps=4,
    device="cuda"  # or "cpu"
)
```

### Loss Function

**Binary Cross Entropy Loss** (CrossEntropyLoss):
- Suitable for binary change classification
- Penalizes incorrect change/no-change predictions
- Standard for classification tasks

### Optimizer

**AdamW**:
- Learning rate: 1e-4
- Weight decay: 0.01
- Gradient clipping: max_norm=1.0

### Parameter Update Verification

The training pipeline verifies that:
- Temporal fusion parameters **change** during training
- Change head parameters **change** during training
- RemoteCLIP encoder parameters **remain frozen**

### Smoke Test

A smoke test script is provided to verify the training pipeline without requiring the real dataset:

```bash
python scripts/smoke_test_change_detection.py \
    --device cpu \
    --num-samples 10 \
    --epochs 2
```

**Note**: Requires `open-clip-torch` to be installed.

## Inference

### Command-Line Inference

```bash
python scripts/run_change_detection.py \
    --image-t1 path/to/t1.jpg \
    --image-t2 path/to/t2.jpg \
    --checkpoint models/change_detection/best_model.pt
```

**Output**:
```
Task: bi_temporal_change_detection
Change Detected: True
Change Label: change
Confidence: 0.8234

Class Probabilities:
  No Change: 0.1766
  Change: 0.8234

Model: RemoteCLIP-ViT-B-32-SiameseChangeDetection
Execution Time: 245.67 ms
Device: cuda
```

### API Inference

**Endpoint**: `POST /change-detection`

**Request**:
- `image_t1`: T1 image file (earlier time)
- `image_t2`: T2 image file (later time)

**Response**:
```json
{
    "change_detected": true,
    "change_label": "change",
    "confidence": 0.8234,
    "class_probabilities": {
        "no_change": 0.1766,
        "change": 0.8234
    },
    "model": "RemoteCLIP-ViT-B-32-SiameseChangeDetection",
    "task": "bi_temporal_change_detection",
    "image_t1": "path/to/t1.jpg",
    "image_t2": "path/to/t2.jpg",
    "execution_time_ms": 245.67,
    "device": "cuda",
    "status": "success"
}
```

**Start API Server**:
```bash
python api/change_detection_api.py
```

Server runs on `http://0.0.0.0:8001`

## Hardware Optimization

### Target Hardware

- **GPU**: RTX 2050 4GB VRAM
- **RAM**: 8GB

### Optimization Strategies

1. **Frozen Encoder**: RemoteCLIP weights are frozen, reducing memory usage
2. **Small Batch Size**: Batch size of 1 for 4GB VRAM compatibility
3. **Gradient Accumulation**: Accumulate gradients over 4 steps for effective larger batch
4. **Lightweight Head**: Only ~200K trainable parameters
5. **Mixed Precision**: Optional (disabled for compatibility)

### Memory Usage Estimates

- **RemoteCLIP (frozen)**: ~600 MB
- **Temporal Fusion**: ~1 MB
- **Change Head**: ~1 MB
- **Total Model**: ~602 MB
- **Training VRAM**: ~2-3 GB (estimated for RTX 2050)
- **Inference VRAM**: ~1-2 GB (estimated)

## Evaluation

### Metrics

For binary change classification:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for change class
- **Recall**: Recall for change class
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (if applicable)

### Validation

The training pipeline includes:
- Validation after each epoch
- Best model checkpointing based on validation loss
- Training history tracking

### Current Status

Real evaluation metrics are **not yet available** because:
1. The CDVQA dataset is not currently in the repository
2. Training has only been verified with synthetic smoke-test data

## Temporal Sanity Tests

### Test 1: Identical Images (T1 == T2)

**Expected**: Low change probability

Rationale: If the same image is provided as both T1 and T2, the model should predict no change.

### Test 2: Different Images (T1 != T2)

**Expected**: Higher change response

Rationale: If significantly different images are provided, the model should detect change.

These are qualitative sanity tests to verify the model behaves reasonably.

## Limitations

### Current Limitations

1. **Dataset Unavailable**: CDVQA dataset not in repository; infrastructure tested with synthetic data only
2. **Binary Classification**: Currently only supports binary change/no-change classification
3. **No Dense Masks**: Spatial change masks not yet implemented (RemoteCLIP features lack spatial resolution)
4. **No Textual Description**: Change description generation not yet implemented
5. **CPU Testing Only**: Training tested on CPU; GPU training not yet verified
6. **Random Initialization**: Current weights are randomly initialized (no real training yet)

### Technical Limitations

1. **Spatial Resolution**: RemoteCLIP outputs global features, not pixel-level features, limiting dense change mask generation
2. **Temporal Scope**: Only two-time-point comparison (T1 and T2)
3. **Label Format**: Designed for binary labels; multi-label or dense masks require modifications

## Future Enhancements

### Potential Improvements

1. **Dense Change Masks**: Implement spatial decoder for pixel-level change detection
2. **Multi-Class Change**: Support multiple change types (building, vegetation, water, etc.)
3. **Textual Change Description**: Add natural language explanation of detected changes
4. **Multi-Temporal Analysis**: Support more than two time points
5. **SAR Change Detection**: Extend to SAR imagery (separate phase)

### Integration with VQA

After Phase 4, the change detection system can be integrated with the Phase 3 VQA system to enable:

- "What changed between these two images?"
- "Where did the building appear?"
- "How much vegetation was lost?"

## Files and Structure

### Model Files

- `models/change_detection/__init__.py` - Module initialization
- `models/change_detection/siamese_change_detection.py` - Main model implementation
- `models/change_detection/training/__init__.py` - Training module
- `models/change_detection/training/change_detection_trainer.py` - Training pipeline

### Data Files

- `data/change_detection_dataset.py` - Dataset loader for T1/T2 pairs
- `data/adapters/cdvqa.py` - CDVQA dataset adapter (existing)

### Script Files

- `scripts/run_change_detection.py` - Command-line inference
- `scripts/smoke_test_change_detection.py` - Smoke test training

### API Files

- `api/change_detection_api.py` - FastAPI endpoint

### Test Files

- `tests/test_change_detection.py` - Comprehensive tests

### Documentation

- `docs/BI_TEMPORAL_CHANGE_DETECTION.md` - This document

## Usage Examples

### Example 1: Command-Line Inference

```bash
python scripts/run_change_detection.py \
    --image-t1 data/samples/t1_2024.jpg \
    --image-t2 data/samples/t2_2026.jpg
```

### Example 2: Python API

```python
from models.change_detection import SiameseChangeDetection, ChangeDetectionConfig

config = ChangeDetectionConfig(device="cuda")
model = SiameseChangeDetection(config=config)
model.load()

result = model.predict(
    image_t1_path="t1.jpg",
    image_t2_path="t2.jpg"
)

print(f"Change detected: {result['change_detected']}")
print(f"Confidence: {result['confidence']}")
```

### Example 3: Training with Real Data

```bash
python scripts/smoke_test_change_detection.py \
    --dataset-root /path/to/cdvqa \
    --device cuda \
    --epochs 10 \
    --batch-size 1
```

## Requirements

### Dependencies

- Python 3.10+
- PyTorch
- open-clip-torch
- FastAPI (for API)
- uvicorn (for API server)
- Pillow
- huggingface-hub (optional, for RemoteCLIP checkpoint)

### Installation

```bash
pip install torch open-clip-torch fastapi uvicorn pillow huggingface-hub
```

## Summary

Phase 4 implements a complete bi-temporal change detection system with:

- ✅ Siamese architecture with shared RemoteCLIP encoder
- ✅ Temporal feature fusion ([F1, F2, |F1-F2|])
- ✅ Lightweight change detection head
- ✅ Training infrastructure with parameter verification
- ✅ Inference scripts and API
- ✅ Comprehensive tests
- ✅ T1/T2 input validation
- ✅ Hardware optimization for RTX 2050 4GB VRAM

The infrastructure is complete and tested with synthetic data. Real training and evaluation require the CDVQA dataset to be downloaded and configured.
