"""
RemoteCLIP Preprocessing Integration for SatQueryAI

This module provides preprocessing utilities specifically for RemoteCLIP,
integrating the general preprocessing pipeline with RemoteCLIP's specific requirements.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .preprocessing import ImagePreprocessor, PreprocessingConfig

logger = logging.getLogger(__name__)


class RemoteCLIPPreprocessor:
    """
    Preprocessor specifically for RemoteCLIP models.
    
    This class integrates the general preprocessing pipeline with RemoteCLIP's
    specific requirements (224x224 input, RGB format, normalization).
    """
    
    # RemoteCLIP standard input size
    REMOTECLIP_INPUT_SIZE = (224, 224)
    
    # RemoteCLIP normalization (ImageNet stats)
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]
    
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ):
        """
        Initialize RemoteCLIP preprocessor.
        
        Args:
            target_size: Target size for images (defaults to 224x224 for RemoteCLIP)
            normalize: Whether to apply RemoteCLIP normalization
        """
        if target_size is None:
            target_size = self.REMOTECLIP_INPUT_SIZE
        
        # Create preprocessing config for RemoteCLIP
        config = PreprocessingConfig(
            target_size=target_size,
            normalize=False,  # We'll do custom normalization
            preserve_bands=False,  # Convert to RGB
            resize_method="resize"
        )
        
        self.preprocessor = ImagePreprocessor(config)
        self.normalize = normalize
    
    def preprocess_for_remoteclip(
        self,
        image_path: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess an image for RemoteCLIP input.
        
        This method:
        1. Loads the image
        2. Converts to RGB if needed
        3. Resizes to target size
        4. Applies RemoteCLIP normalization
        5. Converts to tensor format (CHW)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (preprocessed_image, metadata)
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for RemoteCLIP preprocessing")
        
        # Load image
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load with PIL
        with Image.open(path) as img:
            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Convert to numpy array
            image_data = np.array(img)
            
            # Extract basic metadata
            metadata = {
                "path": str(path),
                "original_size": img.size,
                "original_mode": img.mode,
                "target_size": self.preprocessor.config.target_size
            }
        
        # Apply general preprocessing
        processed_image, processed_metadata = self.preprocessor.preprocess(image_data, metadata)
        
        # Apply RemoteCLIP-specific normalization
        if self.normalize:
            processed_image = self._apply_remoteclip_normalization(processed_image)
        
        # Convert to CHW format (RemoteCLIP expects CHW)
        if len(processed_image.shape) == 3:
            # HWC to CHW
            processed_image = np.transpose(processed_image, (2, 0, 1))
        
        # Update metadata
        processed_metadata["normalization"] = "remoteclip" if self.normalize else "none"
        processed_metadata["format"] = "CHW"
        
        logger.info(f"Preprocessed image for RemoteCLIP: {image_path} -> {processed_image.shape}")
        
        return processed_image, processed_metadata
    
    def _apply_remoteclip_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply RemoteCLIP normalization (ImageNet stats).
        
        Args:
            image: Image array in [0, 255] or [0, 1] range (HWC format)
            
        Returns:
            Normalized image array
        """
        # Ensure float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalize to [0, 1] if in [0, 255] range
        if image.max() > 1.0:
            image = image / 255.0
        
        # Apply normalization per channel
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.MEAN[i]) / self.STD[i]
        
        return image
    
    def preprocess_batch(
        self,
        image_paths: list
    ) -> Tuple[np.ndarray, list]:
        """
        Preprocess a batch of images for RemoteCLIP.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Tuple of (batch_images, metadata_list)
        """
        batch_images = []
        metadata_list = []
        
        for image_path in image_paths:
            try:
                image, metadata = self.preprocess_for_remoteclip(image_path)
                batch_images.append(image)
                metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to preprocess {image_path}: {e}")
                continue
        
        if not batch_images:
            raise ValueError("No images were successfully preprocessed")
        
        # Stack into batch
        batch_array = np.stack(batch_images, axis=0)
        
        logger.info(f"Preprocessed batch of {len(batch_images)} images: {batch_array.shape}")
        
        return batch_array, metadata_list


def prepare_image_for_remoteclip(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Convenience function to prepare an image for RemoteCLIP.
    
    Args:
        image_path: Path to the image file
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed image tensor (CHW format)
    """
    preprocessor = RemoteCLIPPreprocessor(target_size=target_size)
    image, _ = preprocessor.preprocess_for_remoteclip(image_path)
    return image
