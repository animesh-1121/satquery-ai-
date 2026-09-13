# Phase 1 Model Decision: GeoChat → RemoteCLIP

## Executive Summary

After evaluating GeoChat-7B for local deployment on the development hardware, we determined it is not suitable for reliable local inference. This document explains the decision to pivot to a lightweight RemoteCLIP-based architecture.

## Original Candidate: GeoChat-7B

### Model Specifications
- **Model**: MBZUAI/geochat-7B
- **Architecture**: GeoChatLlamaForCausalLM (7B parameters)
- **Vision Encoder**: CLIP-L-336px (interpolated to 504x504)
- **Base LLM**: Vicuna-7B-v1.5 (LLaMA-based)
- **Model Size**: ~13.2 GB (FP16)
- **License**: Apache 2.0

### Hardware Requirements
Based on official documentation and testing:
- **Training**: 3× A100 40GB GPUs
- **Inference (FP16)**: 14+ GB VRAM
- **Inference (INT8)**: 8+ GB VRAM
- **Inference (INT4)**: 5+ GB VRAM
- **CPU Inference**: 28+ GB RAM

### Why GeoChat Was Rejected for Local Deployment

#### Hardware Constraints
The development machine has:
- **GPU**: NVIDIA RTX 2050 with 4 GB VRAM
- **System RAM**: 8 GB
- **OS**: Windows

**Analysis**:
- GeoChat-7B requires minimum 5 GB VRAM even with aggressive INT4 quantization
- Available VRAM (4 GB) is below the minimum threshold
- CPU inference requires 28 GB RAM, but only 8 GB is available
- Even with 8-bit/4-bit quantization, the model would exceed available memory

#### Practical Issues
1. **Model Size**: 13.2 GB model weights exceed practical storage for local development
2. **Download Time**: Multiple gigabytes required over network connections
3. **Inference Speed**: 7B parameter model would be impractically slow on CPU
4. **Development Workflow**: Large model size impedes rapid iteration
5. **Dependency Complexity**: GeoChat requires specific older dependency versions (Python 3.10, Torch 2.0.1)

#### Compatibility Issues Encountered
- GeoChat requires Python 3.10 (not the system default)
- Dependencies (torch==2.0.1, transformers==4.31.0) lack pre-built wheels for Python 3.12+
- Source compilation failed on Windows due to C++/Rust toolchain requirements
- Isolated Python 3.10 environment was set up but model download timed out repeatedly

### GeoChat Status in SatQueryAI

**Decision**: GeoChat is retained as a **reference candidate** for future cloud deployment but is **not used for local Phase 1 development**.

**Rationale**:
- GeoChat is a valid research model with strong remote-sensing VQA capabilities
- It may be suitable for cloud deployment with sufficient GPU resources
- The SatQueryAI architecture supports pluggable specialist models
- GeoChat can be added back as an option when cloud GPU access is available

**Documentation**:
- GeoChat integration code remains in `models/vqa/geochat/`
- GeoChat is documented as an evaluated but locally-deployed candidate
- Future cloud deployment can re-enable GeoChat as a specialist model

## Selected Model: RemoteCLIP

### Model Specifications
- **Model**: RemoteCLIP ViT-B/32
- **Architecture**: Vision Transformer (ViT-B/32) with remote-sensing adaptation
- **Parameters**: 151.3 million
- **Checkpoint Size**: 577 MB
- **Embedding Dimension**: 512
- **License**: Apache 2.0
- **Source**: https://github.com/ChenDelong1999/RemoteCLIP
- **Checkpoint**: https://huggingface.co/chendelong/RemoteCLIP

### Why RemoteCLIP Was Selected

#### 1. Remote-Sensing Adapted
- **Trained on remote sensing data**: RemoteCLIP is specifically trained on remote sensing imagery
- **Outperforms generic CLIP**: 9.14% mean recall improvement on RSICD dataset
- **Task-optimized**: Designed for remote sensing tasks (classification, retrieval, etc.)

#### 2. Lightweight and Hardware-Suitable
- **VRAM Requirement**: ~1-2 GB for ViT-B/32 (vs 14+ GB for GeoChat)
- **Fits in 4 GB VRAM**: Well within RTX 2050 constraints
- **RAM Requirement**: ~4-8 GB (fits in available 8 GB)
- **Fast Inference**: 300ms encoding time on CPU (vs minutes for 7B model)

#### 3. Suitable as Foundation Encoder
- **Architecture Flexibility**: RemoteCLIP provides image embeddings that can be used with various downstream heads
- **Task-Specific Heads**: Lightweight MLP heads can be added for VQA, classification, etc.
- **Fine-Tuning Support**: Can be fine-tuned on BigEarthNet or other RS datasets
- **Pluggable Design**: Fits SatQueryAI's "adapt & orchestrate" strategy

#### 4. Practical for Development
- **Fast Download**: 577 MB checkpoint downloads quickly
- **Rapid Iteration**: Small model size enables quick testing
- **Modern Dependencies**: Compatible with modern PyTorch and Python versions
- **Easy Integration**: OpenCLIP format for straightforward loading

### RemoteCLIP Limitations

**Important**: RemoteCLIP is **not itself a conversational VQA model**.

- **No Language Generation**: RemoteCLIP is a vision-language encoder, not a generative model
- **No Free-form Answers**: Cannot generate arbitrary natural language responses
- **Task-Specific**: Requires task-specific heads for different VQA tasks

**Solution**: SatQueryAI uses:
```
RemoteCLIP Encoder (RS-adapted)
    ↓
Image Embedding
    ↓
Task-Specific VQA Head (Lightweight MLP)
    ↓
Classification Prediction
    ↓
Formatted Answer
```

This architecture provides:
- Remote-sensing-adapted visual understanding
- Lightweight inference suitable for local hardware
- Task-specific answers (land cover, water presence, etc.)
- Extensible design for future tasks

## Architecture Comparison

### GeoChat Architecture (Rejected for Local)
```
Image → CLIP Vision Encoder → LLM (7B params) → Text Generation → Answer
              ↓
         Heavy GPU/VRAM requirements
```

### RemoteCLIP Architecture (Selected for Local)
```
Image → RemoteCLIP Encoder (151M params) → Image Embedding
                                           ↓
Question → Question Parser → Task Type
                                           ↓
                    Task-Specific MLP Head → Classification → Formatted Answer
                                           ↓
                               Lightweight GPU/CPU inference
```

## Future Adaptation Plans

### BigEarthNet Fine-Tuning
The RemoteCLIP-based architecture is designed to support fine-tuning:

**Training Pipeline**:
```
BigEarthNet Image → RemoteCLIP Encoder → Embedding → Task Head → Prediction → Loss
```

**Strategy**:
- **Initial Phase**: Freeze RemoteCLIP encoder, train only lightweight task head locally
- **Future Phase**: Fine-tune RemoteCLIP encoder on cloud GPU with full BigEarthNet dataset
- **Benefit**: Leverages remote-sensing adaptation while keeping local development feasible

### Model Expansion
The pluggable architecture supports future models:
- **Captioning**: Add captioning head to RemoteCLIP encoder
- **Grounding**: Add grounding/detection head
- **Change Detection**: Add Siamese architecture for bi-temporal analysis
- **Optical-SAR Fusion**: Add cross-modal fusion modules

### Cloud Deployment
When cloud GPU access is available:
- Re-enable GeoChat as a specialist model for conversational VQA
- Use RemoteCLIP for lightweight tasks, GeoChat for complex reasoning
- Implement agentic routing between specialist models

## Conclusion

**Decision**: Replace GeoChat-7B with RemoteCLIP-based architecture for Phase 1 local development.

**Rationale**:
- Hardware constraints (4 GB VRAM, 8 GB RAM) make GeoChat impractical for local use
- RemoteCLIP provides remote-sensing-adapted visual understanding at suitable scale
- Lightweight architecture enables rapid development and iteration
- Pluggable design allows future expansion and cloud deployment

**Status**:
- ✅ RemoteCLIP ViT-B/32 successfully loaded and tested
- ✅ End-to-end VQA inference working on local hardware
- ✅ Real satellite image inference completed
- ⏳ VQA head currently untrained (random initialization)
- ⏳ BigEarthNet fine-tuning planned for future phase

**Next Steps**:
1. Complete Phase 1 verification documentation
2. Set up BigEarthNet data structure for fine-tuning
3. Implement comprehensive tests
4. Commit Phase 1 completion
