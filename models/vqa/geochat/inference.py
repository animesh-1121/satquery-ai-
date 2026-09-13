"""
GeoChat Inference Interface for SatQueryAI

This module provides a clean interface for running GeoChat inference
on remote sensing images for visual question answering tasks.
"""

import os
import torch
from PIL import Image
from typing import Dict, Optional, Union
import warnings

# Import GeoChat modules
# Note: These require the GeoChat package to be installed
try:
    from geochat.model.builder import load_pretrained_model
    from geochat.mm_utils import get_model_name_from_path
    from geochat.conversation import conv_templates, Chat
    GEOCHAT_AVAILABLE = True
except ImportError:
    GEOCHAT_AVAILABLE = False
    warnings.warn("GeoChat package not installed. Install with: pip install -e <geochat_repo_path>")


class GeoChatInference:
    """
    Interface for GeoChat remote sensing VQA inference.
    
    This class loads the GeoChat model and provides a simple predict interface
    for answering questions about remote sensing images.
    
    Attributes:
        model_path: Path to GeoChat model checkpoint
        model_base: Path to base model (if using LoRA weights)
        device: Device to run inference on ('cuda' or 'cpu')
        load_8bit: Whether to load model in 8-bit mode
        load_4bit: Whether to load model in 4-bit mode
        tokenizer: Loaded tokenizer
        model: Loaded GeoChat model
        image_processor: Image processor for preprocessing
    """
    
    def __init__(
        self,
        model_path: str = "MBZUAI/geochat-7B",
        model_base: Optional[str] = None,
        device: str = "cuda",
        load_8bit: bool = False,
        load_4bit: bool = False,
        gpu_id: int = 0
    ):
        """
        Initialize GeoChat inference.
        
        Args:
            model_path: Path or HuggingFace ID to GeoChat model checkpoint
            model_base: Path to base model (required for LoRA weights)
            device: Device to run inference on ('cuda' or 'cpu')
            load_8bit: Whether to load model in 8-bit mode for memory efficiency
            load_4bit: Whether to load model in 4-bit mode for memory efficiency
            gpu_id: GPU device ID when using CUDA
        """
        if not GEOCHAT_AVAILABLE:
            raise ImportError(
                "GeoChat package is not installed. "
                "Please install it from: https://github.com/mbzuai-oryx/GeoChat"
            )
        
        self.model_path = model_path
        self.model_base = model_base
        self.device = device
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.gpu_id = gpu_id
        
        # Model components (loaded on first use)
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.conv_mode = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the GeoChat model and associated components."""
        print(f"Loading GeoChat model from {self.model_path}...")
        
        # Get model name from path
        model_name = get_model_name_from_path(self.model_path)
        
        # Load pretrained model
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(
                self.model_path,
                self.model_base,
                model_name,
                self.load_8bit,
                self.load_4bit,
                device_map="auto",
                device=self.device
            )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Determine conversation mode
        if 'llava' in model_name.lower():
            self.conv_mode = "llava_v1"
        elif 'v1' in model_name.lower():
            self.conv_mode = "llava_v1"
        else:
            self.conv_mode = "llava_v1"
        
        print(f"GeoChat model loaded successfully on {self.device}")
        print(f"Conversation mode: {self.conv_mode}")
    
    def predict(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 300
    ) -> Dict[str, Union[str, None]]:
        """
        Run GeoChat inference on an image with a question.
        
        Args:
            image_path: Path to the input image file (RGB satellite image)
            question: Natural language question about the image
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing:
                - answer: The model's response to the question
                - model: Name of the model used
                - confidence: None (GeoChat does not provide confidence scores)
        
        Raises:
            FileNotFoundError: If image file does not exist
            ValueError: If image cannot be loaded or question is empty
        """
        # Validate inputs
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        # Load and validate image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
        
        # Prepare conversation
        conv = conv_templates[self.conv_mode].copy()
        inp = f"USER: <image>\n{question}\nASSISTANT:"
        
        # Process image
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_tensor.unsqueeze(0).to(self.device, dtype=torch.float16)
        
        # Tokenize input
        input_ids = self.tokenizer(inp, return_tensors='pt').input_ids.to(self.device)
        
        # Prepare chat input
        chat = Chat(self.model, self.tokenizer, self.image_processor, self.context_len)
        
        # Generate response
        with torch.no_grad():
            output_ids = chat.generate(
                image_tensor,
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False  # Use greedy decoding for reproducibility
            )
        
        # Decode output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "ASSISTANT:" in output_text:
            answer = output_text.split("ASSISTANT:")[-1].strip()
        else:
            answer = output_text.strip()
        
        return {
            "answer": answer,
            "model": "GeoChat",
            "confidence": None  # GeoChat does not provide confidence scores
        }
    
    def __call__(self, image_path: str, question: str, max_new_tokens: int = 300) -> Dict[str, Union[str, None]]:
        """
        Alias for predict method for convenience.
        
        Args:
            image_path: Path to the input image file
            question: Natural language question about the image
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with answer, model, and confidence (None)
        """
        return self.predict(image_path, question, max_new_tokens)


def create_inference_engine(
    model_path: str = "MBZUAI/geochat-7B",
    device: str = "cuda",
    load_8bit: bool = False
) -> GeoChatInference:
    """
    Factory function to create a GeoChat inference engine.
    
    Args:
        model_path: Path or HuggingFace ID to GeoChat model
        device: Device to run inference on
        load_8bit: Whether to use 8-bit quantization
        
    Returns:
        Initialized GeoChatInference instance
    """
    return GeoChatInference(
        model_path=model_path,
        device=device,
        load_8bit=load_8bit
    )
