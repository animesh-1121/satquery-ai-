"""
Base Model Interface for SatQueryAI

This module defines the abstract interface for all remote sensing models
in the SatQueryAI system. This enables pluggable specialist models that
can be selected by an agentic controller.

Supported model types:
- VQA (Visual Question Answering)
- Captioning
- Grounding
- Change Detection
- Optical-SAR Fusion
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from enum import Enum


class ModelType(Enum):
    """Supported model types in SatQueryAI."""
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"


class RemoteSensingModel(ABC):
    """
    Abstract base class for all remote sensing models in SatQueryAI.
    
    All specialist models must implement this interface to ensure
    consistent behavior and enable agentic routing.
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize the model.
        
        Args:
            model_name: Human-readable name of the model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model weights and initialize components.
        This method should be called before running inference.
        """
        pass
    
    @abstractmethod
    def predict(self, **kwargs) -> Dict[str, Any]:
        """
        Run inference and return structured results.
        
        Returns:
            Dictionary containing at minimum:
            - answer: str (model output)
            - model: str (model identifier)
            - confidence: Optional[float] (prediction confidence, or None)
            Additional fields may include:
            - device: str
            - inference_time_s: float
            - task: str
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Return the model type."""
        pass
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class VQAModel(RemoteSensingModel):
    """
    Abstract base class for VQA (Visual Question Answering) models.
    
    VQA models take an image and a natural language question as input
    and produce a textual answer.
    """
    
    @abstractmethod
    def predict(self, image_path: str, question: str, **kwargs) -> Dict[str, Any]:
        """
        Run VQA inference.
        
        Args:
            image_path: Path to the input image
            question: Natural language question about the image
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing:
            - answer: str (answer to the question)
            - model: str (model identifier)
            - confidence: Optional[float] (prediction confidence)
            - device: str
            - inference_time_s: float
            - task: str (VQA task type)
        """
        pass
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.VQA
