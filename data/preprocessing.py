"""
Image Preprocessing Pipeline for SatQueryAI

This module provides a standardized preprocessing pipeline for remote sensing images,
supporting TIFF/GeoTIFF handling, band preservation, and configurable image preparation.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """
    Configuration for image preprocessing.
    
    Attributes:
        target_size: Target size for resizing (width, height) or None for no resize
        tile_size: Size for tiling (width, height) or None for no tiling
        overlap: Overlap between tiles in pixels
        max_size: Maximum size for large images (width, height)
        normalize: Whether to normalize pixel values
        preserve_bands: Whether to preserve spectral bands (for multispectral)
        resize_method: Resize method ('crop', 'pad', 'resize')
        band_selection: List of bands to select (e.g., ['R', 'G', 'B'])
        enable_tiling: Whether to enable image tiling
    """
    target_size: Optional[Tuple[int, int]] = None
    tile_size: Optional[Tuple[int, int]] = None
    overlap: int = 0
    max_size: Optional[Tuple[int, int]] = None
    normalize: bool = True
    preserve_bands: bool = True
    resize_method: str = "resize"  # 'crop', 'pad', 'resize'
    band_selection: Optional[list] = None  # e.g., ['R', 'G', 'B'] for RGB
    enable_tiling: bool = False


class ImagePreprocessor:
    """
    Preprocessor for remote sensing images.
    
    Handles:
    - Format detection (TIFF, GeoTIFF, common image formats)
    - Metadata inspection
    - Band validation
    - Spatial validation
    - Normalization
    - Resize/crop/tile operations
    - Tensor conversion
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
    
    def load_image(self, image_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load an image and extract metadata.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (image_data, metadata)
        """
        path = Path(image_path)
        
        metadata = {
            "path": str(path),
            "format": path.suffix.lower(),
            "size": None,
            "bands": None,
            "crs": None,
            "transform": None,
            "modality": "optical"  # Default assumption
        }
        
        # Try GeoTIFF first
        if RASTERIO_AVAILABLE and path.suffix.lower() in {'.tif', '.tiff'}:
            try:
                with rasterio.open(path) as src:
                    # Read all bands
                    image_data = src.read()
                    
                    # Extract metadata
                    metadata["size"] = (src.width, src.height)
                    metadata["bands"] = [src.descriptions[i] if src.descriptions[i] else f"Band_{i+1}" 
                                        for i in range(src.count)]
                    metadata["crs"] = str(src.crs) if src.crs else None
                    metadata["transform"] = str(src.transform)
                    metadata["modality"] = self._detect_modality_from_bands(metadata["bands"])
                    
                    logger.info(f"Loaded GeoTIFF: {path} ({src.width}x{src.height}, {src.count} bands)")
                    
                    return image_data, metadata
                    
            except Exception as e:
                logger.warning(f"Failed to load as GeoTIFF, trying PIL: {e}")
        
        # Fall back to PIL
        if PIL_AVAILABLE:
            try:
                with Image.open(path) as img:
                    image_data = np.array(img)
                    
                    # Extract metadata
                    metadata["size"] = img.size
                    if len(image_data.shape) == 3:
                        # Assume RGB
                        metadata["bands"] = ['R', 'G', 'B']
                    else:
                        metadata["bands"] = ['Gray']
                    
                    logger.info(f"Loaded image with PIL: {path} {img.size}")
                    
                    return image_data, metadata
                    
            except Exception as e:
                raise ValueError(f"Failed to load image with PIL: {e}")
        
        raise ValueError(f"Cannot load image: {image_path} (PIL and rasterio not available)")
    
    def _detect_modality_from_bands(self, bands: list) -> str:
        """
        Detect image modality from band names.
        
        Args:
            bands: List of band names
            
        Returns:
            Modality string ('optical', 'sar', 'multimodal_optical_sar', 'unknown')
        """
        band_str = " ".join(bands).lower()
        
        if any(x in band_str for x in ['vh', 'vv', 'hh', 'hv', 'sar', 'radar']):
            return 'sar'
        elif any(x in band_str for x in ['red', 'green', 'blue', 'nir', 'swir']):
            return 'optical'
        else:
            return 'unknown'
    
    def preprocess(self, image_data: Any, metadata: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Apply preprocessing to image data.
        
        Args:
            image_data: Raw image data
            metadata: Image metadata
            
        Returns:
            Tuple of (processed_image, updated_metadata)
        """
        processed_metadata = metadata.copy()
        
        # Store original dimensions
        if len(image_data.shape) == 3:
            original_height, original_width = image_data.shape[1], image_data.shape[2]
        else:
            original_height, original_width = image_data.shape[0], image_data.shape[1]
        
        processed_metadata["original_width"] = original_width
        processed_metadata["original_height"] = original_height
        
        # Apply size limit if configured
        if self.config.max_size:
            image_data = self._apply_size_limit(image_data, self.config.max_size)
        
        # Apply target size if configured
        if self.config.target_size:
            image_data = self._apply_target_size(image_data, self.config.target_size, self.config.resize_method)
        
        # Update metadata with new dimensions
        if len(image_data.shape) == 3:
            processed_metadata["width"] = image_data.shape[2]
            processed_metadata["height"] = image_data.shape[1]
        else:
            processed_metadata["width"] = image_data.shape[1]
            processed_metadata["height"] = image_data.shape[0]
        
        # Normalize if configured
        if self.config.normalize:
            image_data = self._normalize(image_data)
        
        return image_data, processed_metadata
    
    def _apply_size_limit(self, image_data: Any, max_size: Tuple[int, int]) -> Any:
        """
        Apply maximum size limit to image.
        
        Args:
            image_data: Image data
            max_size: Maximum (width, height)
            
        Returns:
            Resized image data
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available, cannot resize")
            return image_data
        
        max_width, max_height = max_size
        
        # Get current dimensions
        if len(image_data.shape) == 3:
            height, width = image_data.shape[1], image_data.shape[2]
            # Convert to PIL (transpose from CHW to HWC)
            if image_data.shape[0] == 3 or image_data.shape[0] == 4:
                image_data = np.transpose(image_data, (1, 2, 0))
        else:
            height, width = image_data.shape
        
        # Check if resize needed
        if width <= max_width and height <= max_height:
            return image_data
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Convert to PIL, resize, convert back
        if len(image_data.shape) == 3:
            img = Image.fromarray(image_data.astype(np.uint8))
            img = img.resize((new_width, new_height), Image.LANCZOS)
            return np.array(img)
        else:
            img = Image.fromarray(image_data.astype(np.uint8))
            img = img.resize((new_width, new_height), Image.LANCZOS)
            return np.array(img)
    
    def _apply_target_size(self, image_data: Any, target_size: Tuple[int, int], method: str) -> Any:
        """
        Apply target size to image using specified method.
        
        Args:
            image_data: Image data
            target_size: Target (width, height)
            method: Resize method ('crop', 'pad', 'resize')
            
        Returns:
            Resized image data
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available, cannot resize")
            return image_data
        
        target_width, target_height = target_size
        
        # Get current dimensions
        if len(image_data.shape) == 3:
            height, width = image_data.shape[1], image_data.shape[2]
            # Convert to PIL (transpose from CHW to HWC)
            if image_data.shape[0] == 3 or image_data.shape[0] == 4:
                image_data = np.transpose(image_data, (1, 2, 0))
        else:
            height, width = image_data.shape
        
        # Convert to PIL
        img = Image.fromarray(image_data.astype(np.uint8))
        
        if method == "resize":
            img = img.resize((target_width, target_height), Image.LANCZOS)
        elif method == "crop":
            # Center crop
            left = (width - target_width) // 2
            top = (height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            img = img.crop((left, top, right, bottom))
        elif method == "pad":
            # Pad to target size
            img = self._pad_image(img, target_width, target_height)
        
        return np.array(img)
    
    def _pad_image(self, img: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """
        Pad image to target size.
        
        Args:
            img: PIL Image
            target_width: Target width
            target_height: Target height
            
        Returns:
            Padded image
        """
        width, height = img.size
        
        # Calculate padding
        pad_left = (target_width - width) // 2
        pad_top = (target_height - height) // 2
        pad_right = target_width - width - pad_left
        pad_bottom = target_height - height - pad_top
        
        # Create new image with padding
        padded = Image.new(img.mode, (target_width, target_height), (0, 0, 0))
        padded.paste(img, (pad_left, pad_top))
        
        return padded
    
    def _normalize(self, image_data: Any) -> Any:
        """
        Normalize image data to [0, 1] range.
        
        Args:
            image_data: Image data
            
        Returns:
            Normalized image data
        """
        if image_data.dtype == np.uint8:
            return image_data.astype(np.float32) / 255.0
        else:
            # Assume already float, just clip to [0, 1]
            return np.clip(image_data, 0.0, 1.0)
    
    def to_tensor(self, image_data: Any) -> Any:
        """
        Convert image data to tensor format.
        
        Args:
            image_data: Image data (numpy array)
            
        Returns:
            Tensor data (CHW format for PyTorch)
        """
        # Convert to CHW format if needed
        if len(image_data.shape) == 3 and image_data.shape[2] <= 4:
            # HWC to CHW
            image_data = np.transpose(image_data, (2, 0, 1))
        
        return image_data
    
    def tile_image(self, image_data: Any, metadata: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Tile a large image into smaller patches.
        
        Args:
            image_data: Image data (numpy array)
            metadata: Image metadata
            
        Returns:
            Tuple of (list of tiles, updated metadata with tiling info)
        """
        if not self.config.enable_tiling or self.config.tile_size is None:
            return [image_data], metadata
        
        tile_width, tile_height = self.config.tile_size
        overlap = self.config.overlap
        
        # Get current dimensions
        if len(image_data.shape) == 3:
            # HWC format
            height, width = image_data.shape[:2]
        else:
            # Grayscale
            height, width = image_data.shape
        
        # Calculate stride
        stride_x = tile_width - overlap
        stride_y = tile_height - overlap
        
        tiles = []
        tile_info = []
        
        # Generate tiles
        for y in range(0, height, stride_y):
            for x in range(0, width, stride_x):
                # Calculate tile boundaries
                x_end = min(x + tile_width, width)
                y_end = min(y + tile_height, height)
                
                # Adjust start if at the end
                x_start = max(0, x_end - tile_width)
                y_start = max(0, y_end - tile_height)
                
                # Extract tile
                if len(image_data.shape) == 3:
                    tile = image_data[y_start:y_end, x_start:x_end, :]
                else:
                    tile = image_data[y_start:y_end, x_start:x_end]
                
                tiles.append(tile)
                tile_info.append({
                    "tile_index": len(tiles) - 1,
                    "x_start": x_start,
                    "y_start": y_start,
                    "x_end": x_end,
                    "y_end": y_end,
                    "width": x_end - x_start,
                    "height": y_end - y_start
                })
        
        # Update metadata
        updated_metadata = metadata.copy()
        updated_metadata["tiling"] = {
            "enabled": True,
            "tile_size": self.config.tile_size,
            "overlap": overlap,
            "num_tiles": len(tiles),
            "tiles": tile_info
        }
        
        logger.info(f"Tiled image into {len(tiles)} tiles ({tile_width}x{tile_height} with {overlap}px overlap)")
        
        return tiles, updated_metadata
