# GeoChat Integration for SatQueryAI

## Overview

This module provides an interface for [GeoChat](https://github.com/mbzuai-oryx/GeoChat), a grounded Large Vision-Language Model specifically designed for remote sensing scenarios. GeoChat was published at CVPR 2024 and excels at handling high-resolution remote sensing imagery with region-level reasoning capabilities.

### What is GeoChat?

GeoChat is the first grounded Large Vision Language Model tailored for Remote Sensing (RS) scenarios. Unlike general-domain VLMs, GeoChat:

- Handles high-resolution RS imagery with region-level reasoning
- Provides zero-shot performance across various RS tasks
- Supports visual question answering, image/region captioning, scene classification, and visually grounded conversations
- Uses the LLaVA-1.5 architecture fine-tuned on a 318k RS multimodal instruction dataset

### Paper and Citation

```bibtex
@inproceedings{kuckreja2024geochat,
  title={GeoChat: Grounded Large Vision-Language Model for Remote Sensing},
  author={Kuckreja, Kartik and Danish, Muhammad Sohail and Naseer, Muzammal and Das, Abhijit and Khan, Salman and Khan, Fahad S},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## Installation

### 1. Set up the environment

```bash
# Create conda environment from the project environment.yaml
conda env create -f environment.yaml
conda activate satquery-ai
```

### 2. Install GeoChat package

```bash
# Clone the official GeoChat repository
git clone https://github.com/mbzuai-oryx/GeoChat.git
cd GeoChat

# Install GeoChat package
pip install -e .
```

### 3. Verify installation

```bash
python -c "from geochat.model.builder import load_pretrained_model; print('GeoChat installed successfully')"
```

## Model Checkpoint

The model checkpoint is available on HuggingFace:

- **Model ID**: `MBZUAI/geochat-7B`
- **Base LLM**: Vicuna-13B-v1.3
- **Vision Encoder**: CLIP-L-336px (extended to 504x504)
- **License**: Apache 2.0

The model will be automatically downloaded from HuggingFace on first use.

## Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 16GB VRAM (for 7B model in 8-bit mode)
- **CPU**: Modern multi-core processor
- **RAM**: 32GB system RAM
- **Storage**: 20GB free disk space (for model weights)

### Recommended Requirements
- **GPU**: NVIDIA GPU with 24GB+ VRAM (A100, RTX 3090/4090, or similar)
- **CPU**: Modern multi-core processor
- **RAM**: 64GB system RAM
- **Storage**: 50GB free disk space

### Memory Optimization Options

For systems with limited GPU memory, use 8-bit quantization:

```python
engine = create_inference_engine(load_8bit=True)
```

Or 4-bit quantization (experimental):

```python
engine = create_inference_engine(load_4bit=True)
```

## Input/Output Format

### Input

- **Image**: RGB satellite image file (JPG, PNG, etc.)
- **Resolution**: Automatically padded to square, processed at 504x504
- **Question**: Natural language question about the image

### Output

```python
{
    "answer": "The image shows an urban area with residential buildings and roads.",
    "model": "GeoChat",
    "confidence": null
}
```

**Note**: GeoChat does not provide confidence scores. The `confidence` field is always `null`.

## Usage

### Python API

```python
from models.vqa.geochat import create_inference_engine

# Create inference engine
engine = create_inference_engine(
    model_path="MBZUAI/geochat-7B",
    device="cuda",
    load_8bit=False
)

# Run inference
result = engine.predict(
    image_path="path/to/satellite_image.jpg",
    question="What type of land cover is visible in this image?"
)

print(result["answer"])
```

### Command Line Interface

```bash
python scripts/run_vqa.py \
    --image path/to/satellite_image.jpg \
    --question "What type of land cover is visible in this image?" \
    --model-path MBZUAI/geochat-7B \
    --device cuda \
    --output results.json
```

### CLI Options

- `--image`: Path to input image (required)
- `--question`: Natural language question (required)
- `--model-path`: Model path or HuggingFace ID (default: MBZUAI/geochat-7B)
- `--device`: Device to use (cuda/cpu, default: cuda)
- `--load-8bit`: Enable 8-bit quantization for memory efficiency
- `--max-new-tokens`: Maximum tokens to generate (default: 300)
- `--output`: Optional path to save results as JSON

## Supported Tasks

GeoChat supports multiple remote sensing tasks in a unified framework:

1. **Visual Question Answering (VQA)**: Answer questions about images
2. **Image Captioning**: Generate descriptions of images
3. **Region Captioning**: Describe specific regions in images
4. **Scene Classification**: Classify the scene type
5. **Visually Grounded Conversations**: Generate responses with object locations
6. **Referring Object Detection**: Detect objects based on descriptions

This integration currently focuses on image-level VQA. Region-level tasks can be added in future iterations.

## Limitations

1. **No Confidence Scores**: GeoChat does not provide confidence scores for its predictions
2. **Optical Only**: Currently supports only optical RGB imagery (no SAR support)
3. **Memory Intensive**: The 7B model requires significant GPU memory
4. **Inference Speed**: Generation can be slow on consumer hardware
5. **Language**: Primarily optimized for English questions
6. **Resolution**: Fixed 504x504 processing resolution (images are padded)

## Example Questions

Here are some example questions you can ask GeoChat:

- "What type of land cover is visible in this image?"
- "Describe the buildings in this image."
- "What are the main features of this landscape?"
- "Is there any water body visible in this image?"
- "What type of vegetation is present?"
- "Describe the transportation infrastructure visible."

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:

1. Use 8-bit quantization: `--load-8bit`
2. Reduce `max_new_tokens` parameter
3. Use a smaller batch size (if running batch inference)
4. Close other GPU-intensive applications

### Installation Issues

If GeoChat installation fails:

1. Ensure Python 3.10 is installed
2. Update pip: `pip install --upgrade pip`
3. Install torch separately first: `pip install torch==2.0.1 torchvision==0.15.2`
4. Check CUDA version compatibility

### Model Download Issues

If HuggingFace download fails:

1. Check internet connection
2. Set HuggingFace cache directory: `export HF_HOME=/path/to/cache`
3. Use a mirror if in a region with restricted access

## Testing

### Run Unit Tests

```bash
python -m pytest tests/test_geochat_inference.py -v
```

### Run Inference Test

```bash
# Download a sample satellite image (or use your own)
python scripts/run_vqa.py \
    --image path/to/test_image.jpg \
    --question "What is in this image?"
```

## Future Enhancements

Planned improvements for this integration:

1. Add region-level VQA support
2. Implement batch inference
4. Add SAR modality support (when available in GeoChat)
5. Add optical-SAR fusion capabilities
6. Implement model quantization for faster inference
7. Add streaming responses for long generations

## References

- [GeoChat GitHub Repository](https://github.com/mbzuai-oryx/GeoChat)
- [GeoChat Project Website](https://mbzuai-oryx.github.io/GeoChat)
- [Paper on arXiv](https://arxiv.org/abs/2311.15826)
- [Model on HuggingFace](https://huggingface.co/MBZUAI/geochat-7B)
- [GeoChat Dataset](https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct)

## License

This integration uses GeoChat, which is licensed under the Apache 2.0 License. See the [GeoChat repository](https://github.com/mbzuai-oryx/GeoChat) for the full license text.
