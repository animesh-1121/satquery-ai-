"""
RemoteCLIP-based VQA Model for SatQueryAI

This module implements a lightweight VQA architecture using RemoteCLIP as the
remote-sensing-adapted vision encoder, combined with a task-specific classification head.

Architecture:
    Question
    ↓
    Question parser (maps to task type)
    ↓
    RemoteCLIP encoder (RS-adapted vision encoder)
    ↓
    Image embedding
    ↓
    Task-specific MLP head
    ↓
    Classification probabilities
    ↓
    Answer formatter

Model:
    - RemoteCLIP ViT-B/32 (605 MB checkpoint, RS-adapted)
    - Lightweight MLP classification head
    - Support for multiple VQA tasks (land cover, water presence, etc.)

Hardware Requirements:
    - VRAM: ~1-2 GB for ViT-B/32 + head
    - RAM: ~4-8 GB
    - Suitable for RTX 2050 4GB VRAM
"""

import os
import time
import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Optional, Any, List
from enum import Enum
import warnings

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    warnings.warn("open-clip-torch not installed. Install with: pip install open-clip-torch")

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    warnings.warn("huggingface_hub not installed. Install with: pip install huggingface_hub")

from ...base import VQAModel, ModelType


class VQATask(Enum):
    """Supported VQA task types."""
    LAND_COVER = "land_cover"
    WATER_PRESENCE = "water_presence"
    SCENE_TYPE = "scene_type"
    URBAN_RURAL = "urban_rural"


class QuestionParser:
    """
    Lightweight rule-based question parser for mapping questions to VQA tasks.
    
    This avoids requiring a large LLM just to parse questions, keeping the
    architecture lightweight and suitable for local deployment.
    """
    
    TASK_KEYWORDS = {
        VQATask.LAND_COVER: [
            "land cover", "land use", "type of land", "what type of",
            "dominant land", "land type", "classification"
        ],
        VQATask.WATER_PRESENCE: [
            "water", "water present", "is there water", "contains water",
            "lake", "river", "ocean", "sea"
        ],
        VQATask.SCENE_TYPE: [
            "scene", "what is this", "what type of scene", "location type",
            "what is this image", "scene classification"
        ],
        VQATask.URBAN_RURAL: [
            "urban", "rural", "city", "countryside", "built-up", "developed",
            "urban or rural"
        ]
    }
    
    @classmethod
    def parse(cls, question: str) -> VQATask:
        """
        Parse a natural language question into a VQA task type.
        
        Args:
            question: Natural language question
            
        Returns:
            VQATask: The mapped task type
        """
        question_lower = question.lower()
        
        # Check for specific task keywords in order of specificity
        # Water presence is most specific
        if any(keyword in question_lower for keyword in cls.TASK_KEYWORDS[VQATask.WATER_PRESENCE]):
            return VQATask.WATER_PRESENCE
        
        # Urban/rural is specific
        if any(keyword in question_lower for keyword in cls.TASK_KEYWORDS[VQATask.URBAN_RURAL]):
            return VQATask.URBAN_RURAL
        
        # Scene type
        if any(keyword in question_lower for keyword in cls.TASK_KEYWORDS[VQATask.SCENE_TYPE]):
            return VQATask.SCENE_TYPE
        
        # Land cover is most general
        if any(keyword in question_lower for keyword in cls.TASK_KEYWORDS[VQATask.LAND_COVER]):
            return VQATask.LAND_COVER
        
        # Default to land cover if no specific task detected
        return VQATask.LAND_COVER


class VQAHead(nn.Module):
    """
    Lightweight classification head for VQA tasks.
    
    This is a simple MLP that takes RemoteCLIP image embeddings and
    produces task-specific predictions. It can be fine-tuned on
    BigEarthNet or other remote sensing datasets.
    """
    
    def __init__(self, embedding_dim: int, num_classes: int, hidden_dim: int = 512):
        """
        Initialize the VQA head.
        
        Args:
            embedding_dim: Dimension of RemoteCLIP image embeddings
            num_classes: Number of output classes for the task
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, image_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.
        
        Args:
            image_embedding: Image embedding from RemoteCLIP [batch, embedding_dim]
            
        Returns:
            logits: Class logits [batch, num_classes]
        """
        return self.classifier(image_embedding)


class RemoteCLIPVQA(VQAModel):
    """
    RemoteCLIP-based VQA model for remote sensing images.
    
    This model uses RemoteCLIP (RS-adapted CLIP) as the vision encoder
    and a lightweight classification head for VQA tasks.
    """
    
    # Land cover classes (BigEarthNet-like)
    LAND_COVER_CLASSES = [
        "water", "forest", "agricultural", "urban/built-up",
        "barren", "grassland", "mixed"
    ]
    
    # Task configurations
    TASK_CONFIGS = {
        VQATask.LAND_COVER: {
            "classes": LAND_COVER_CLASSES,
            "templates": [
                "A satellite image of {class}.",
                "A photo of {class} from above.",
                "Remote sensing view of {class}."
            ]
        },
        VQATask.WATER_PRESENCE: {
            "classes": ["no water", "water present"],
            "templates": [
                "A satellite image with {class}.",
                "A photo {class} from above."
            ]
        },
        VQATask.SCENE_TYPE: {
            "classes": LAND_COVER_CLASSES,
            "templates": [
                "A satellite image of {class}.",
                "A remote sensing view of {class}."
            ]
        },
        VQATask.URBAN_RURAL: {
            "classes": ["rural", "urban"],
            "templates": [
                "A satellite image of {class} area.",
                "A photo of {class} landscape from above."
            ]
        }
    }
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        freeze_encoder: bool = True
    ):
        """
        Initialize RemoteCLIP VQA model.
        
        Args:
            model_name: RemoteCLIP model variant ('RN50', 'ViT-B-32', 'ViT-L-14')
            device: Device to run inference on ('cuda' or 'cpu')
            checkpoint_path: Local path to RemoteCLIP checkpoint (if None, downloads from HF)
            cache_dir: Directory for caching downloaded checkpoints
            freeze_encoder: Whether to freeze the RemoteCLIP encoder (for lightweight adaptation)
        """
        super().__init__(
            model_name=f"RemoteCLIP-{model_name}",
            device=device
        )
        
        self.remoteclip_model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.cache_dir = cache_dir
        self.freeze_encoder = freeze_encoder
        
        # Model components
        self.remoteclip_model = None
        self.preprocess = None
        self.tokenizer = None
        self.vqa_heads = {}  # Task-specific heads
        self.embedding_dim = None
        
        # Question parser
        self.question_parser = QuestionParser()
    
    def load(self) -> None:
        """Load RemoteCLIP model and initialize VQA heads."""
        if not OPENCLIP_AVAILABLE:
            raise ImportError("open-clip-torch is required. Install with: pip install open-clip-torch")
        
        print(f"Loading RemoteCLIP-{self.remoteclip_model_name}...")
        
        # Create OpenCLIP model
        self.remoteclip_model, self.preprocess, _ = open_clip.create_model_and_transforms(
            self.remoteclip_model_name,
            pretrained="openai"  # Load OpenAI weights as base
        )
        self.tokenizer = open_clip.get_tokenizer(self.remoteclip_model_name)
        
        # Get embedding dimension
        self.embedding_dim = self.remoteclip_model.visual.output_dim
        
        # Load RemoteCLIP checkpoint
        if self.checkpoint_path is None:
            # Download from HuggingFace
            if not HF_HUB_AVAILABLE:
                raise ImportError("huggingface_hub is required for auto-download. Install with: pip install huggingface_hub")
            
            print(f"Downloading RemoteCLIP-{self.remoteclip_model_name} from HuggingFace...")
            self.checkpoint_path = hf_hub_download(
                repo_id="chendelong/RemoteCLIP",
                filename=f"RemoteCLIP-{self.remoteclip_model_name}.pt",
                cache_dir=self.cache_dir
            )
            print(f"Checkpoint downloaded to: {self.checkpoint_path}")
        
        # Load RemoteCLIP weights
        print(f"Loading RemoteCLIP weights from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.remoteclip_model.load_state_dict(checkpoint)
        print("RemoteCLIP weights loaded successfully")
        
        # Move to device and set to eval mode
        self.remoteclip_model = self.remoteclip_model.to(self.device).eval()
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            print("Freezing RemoteCLIP encoder...")
            for param in self.remoteclip_model.visual.parameters():
                param.requires_grad = False
            print("RemoteCLIP encoder frozen")
        
        # Initialize VQA heads for each task
        print("Initializing VQA heads...")
        for task, config in self.TASK_CONFIGS.items():
            num_classes = len(config["classes"])
            head = VQAHead(self.embedding_dim, num_classes).to(self.device)
            self.vqa_heads[task] = head
            print(f"  {task.value}: {num_classes} classes")
            
            # Freeze heads by default (unfreeze for training)
            for param in head.parameters():
                param.requires_grad = False
        
        self._loaded = True
        print("RemoteCLIP VQA model loaded successfully")
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image using RemoteCLIP.
        
        Args:
            image: PIL Image
            
        Returns:
            embedding: Normalized image embedding [1, embedding_dim]
        """
        # Preprocess image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode with RemoteCLIP
        with torch.no_grad():
            embedding = self.remoteclip_model.encode_image(image_input)
            # Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding
    
    def _classify_image(
        self,
        image_embedding: torch.Tensor,
        task: VQATask
    ) -> tuple[torch.Tensor, List[str]]:
        """
        Classify image embedding using task-specific head.
        
        Args:
            image_embedding: Image embedding from RemoteCLIP
            task: VQA task type
            
        Returns:
            probabilities: Class probabilities [1, num_classes]
            class_names: List of class names
        """
        if task not in self.vqa_heads:
            raise ValueError(f"Task {task} not supported")
        
        head = self.vqa_heads[task]
        class_names = self.TASK_CONFIGS[task]["classes"]
        
        # Get logits from head
        with torch.no_grad():
            logits = head(image_embedding)
            probabilities = torch.softmax(logits, dim=-1)
        
        return probabilities, class_names
    
    def _format_answer(
        self,
        task: VQATask,
        class_name: str,
        confidence: float
    ) -> str:
        """
        Format the answer based on task type and prediction.
        
        Args:
            task: VQA task type
            class_name: Predicted class name
            confidence: Prediction confidence
            
        Returns:
            answer: Formatted natural language answer
        """
        if task == VQATask.LAND_COVER:
            return f"The image is predominantly {class_name} land."
        elif task == VQATask.WATER_PRESENCE:
            if class_name == "water present":
                return f"Yes, water is visible in the image."
            else:
                return f"No water is visible in the image."
        elif task == VQATask.SCENE_TYPE:
            return f"The image shows a {class_name} scene."
        elif task == VQATask.URBAN_RURAL:
            return f"The area appears to be {class_name}."
        else:
            return f"The image is classified as {class_name}."
    
    def predict(
        self,
        image_path: str,
        question: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run VQA inference on an image with a question.
        
        Args:
            image_path: Path to the input image
            question: Natural language question about the image
            **kwargs: Additional parameters (unused currently)
            
        Returns:
            Dictionary containing:
            - answer: str (answer to the question)
            - model: str (model identifier)
            - confidence: float (prediction confidence)
            - device: str
            - inference_time_s: float
            - task: str (VQA task type)
        """
        if not self._loaded:
            self.load()
        
        # Validate image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Parse question to task
        task = self.question_parser.parse(question)
        
        # Start timing
        start_time = time.time()
        
        # Encode image
        image_embedding = self._encode_image(image)
        
        # Classify with task-specific head
        probabilities, class_names = self._classify_image(image_embedding, task)
        
        # Get top prediction
        prob_cpu = probabilities.cpu().squeeze(0)
        top_idx = prob_cpu.argmax().item()
        confidence = prob_cpu[top_idx].item()
        predicted_class = class_names[top_idx]
        
        # Format answer
        answer = self._format_answer(task, predicted_class, confidence)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Get memory stats
        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats["gpu_allocated_mb"] = round(
                torch.cuda.memory_allocated(self.device) / (1024 * 1024), 1
            )
            memory_stats["gpu_reserved_mb"] = round(
                torch.cuda.memory_reserved(self.device) / (1024 * 1024), 1
            )
        
        return {
            "answer": answer,
            "model": self.model_name,
            "confidence": confidence,
            "device": self.device,
            "inference_time_s": round(inference_time, 3),
            "task": task.value,
            "predicted_class": predicted_class,
            "all_probabilities": {
                class_names[i]: round(prob_cpu[i].item(), 4)
                for i in range(len(class_names))
            },
            "memory": memory_stats
        }


def create_inference_engine(
    model_name: str = "ViT-B-32",
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    freeze_encoder: bool = True
) -> RemoteCLIPVQA:
    """
    Factory function to create a RemoteCLIP VQA inference engine.
    
    Args:
        model_name: RemoteCLIP model variant ('RN50', 'ViT-B-32', 'ViT-L-14')
        device: Device to run inference on ('cuda' or 'cpu')
        checkpoint_path: Local path to RemoteCLIP checkpoint
        cache_dir: Directory for caching downloaded checkpoints
        freeze_encoder: Whether to freeze the RemoteCLIP encoder
        
    Returns:
        RemoteCLIPVQA: Initialized inference engine
    """
    engine = RemoteCLIPVQA(
        model_name=model_name,
        device=device,
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        freeze_encoder=freeze_encoder
    )
    engine.load()
    return engine
