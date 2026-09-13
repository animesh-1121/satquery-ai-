"""
Data Validation Utilities for SatQueryAI

This module provides validation utilities for remote sensing datasets,
including image validation, format checking, and data integrity verification.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of data validation.
    
    Attributes:
        valid: Whether the validation passed
        total_samples: Total number of samples validated
        valid_samples: Number of valid samples
        invalid_samples: Number of invalid samples
        errors: List of error messages
        warnings: List of warning messages
    """
    valid: bool
    total_samples: int
    valid_samples: int
    invalid_samples: int
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "errors": self.errors,
            "warnings": self.warnings
        }


class DataValidator:
    """
    Validator for remote sensing data.
    
    Validates images, formats, dimensions, and metadata integrity.
    """
    
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    SUPPORTED_GEOTIFF_FORMATS = {'.tif', '.tiff'}
    
    @staticmethod
    def validate_image_path(image_path: str) -> ValidationResult:
        """
        Validate a single image path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        path = Path(image_path)
        
        # Check if file exists
        if not path.exists():
            return ValidationResult(
                valid=False,
                total_samples=1,
                valid_samples=0,
                invalid_samples=1,
                errors=[f"File not found: {image_path}"],
                warnings=warnings
            )
        
        # Check if file is empty
        if path.stat().st_size == 0:
            errors.append(f"File is empty: {image_path}")
            return ValidationResult(
                valid=False,
                total_samples=1,
                valid_samples=0,
                invalid_samples=1,
                errors=errors,
                warnings=warnings
            )
        
        # Check file extension
        if path.suffix.lower() not in DataValidator.SUPPORTED_IMAGE_FORMATS:
            warnings.append(f"Unsupported image format: {path.suffix}")
        
        # Try to open and validate the image
        if PIL_AVAILABLE:
            try:
                with Image.open(path) as img:
                    # Check if image can be loaded
                    img.verify()
                    
                    # Re-open for further checks (verify closes the file)
                    with Image.open(path) as img:
                        # Check dimensions
                        if img.width < 1 or img.height < 1:
                            errors.append(f"Invalid dimensions: {img.width}x{img.height}")
                        
                        # Check for reasonable dimensions
                        if img.width > 100000 or img.height > 100000:
                            warnings.append(f"Unusually large dimensions: {img.width}x{img.height}")
                        
            except Exception as e:
                errors.append(f"Failed to open/validate image: {e}")
        
        # Check if it's a GeoTIFF
        if RASTERIO_AVAILABLE and path.suffix.lower() in DataValidator.SUPPORTED_GEOTIFF_FORMATS:
            try:
                with rasterio.open(path) as src:
                    # Check CRS
                    if src.crs is None:
                        warnings.append(f"No CRS defined for GeoTIFF: {image_path}")
                    
                    # Check transform
                    if src.transform is None:
                        warnings.append(f"No transform defined for GeoTIFF: {image_path}")
                    
                    # Check band count
                    if src.count == 0:
                        errors.append(f"No bands in GeoTIFF: {image_path}")
                    
            except Exception as e:
                errors.append(f"Failed to validate GeoTIFF: {e}")
        
        valid = len(errors) == 0
        
        return ValidationResult(
            valid=valid,
            total_samples=1,
            valid_samples=1 if valid else 0,
            invalid_samples=0 if valid else 1,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def validate_sample(sample) -> ValidationResult:
        """
        Validate a dataset sample.
        
        Args:
            sample: Sample object to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        # Check that at least one image path is provided
        has_image = (
            sample.image_path is not None or
            sample.image_t1_path is not None or
            sample.image_t2_path is not None or
            sample.optical_path is not None or
            sample.sar_path is not None
        )
        
        if not has_image:
            errors.append(f"Sample {sample.sample_id}: No image path provided")
        
        # Validate image paths
        if sample.image_path:
            result = DataValidator.validate_image_path(sample.image_path)
            if not result.valid:
                errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        if sample.image_t1_path:
            result = DataValidator.validate_image_path(sample.image_t1_path)
            if not result.valid:
                errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        if sample.image_t2_path:
            result = DataValidator.validate_image_path(sample.image_t2_path)
            if not result.valid:
                errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        if sample.optical_path:
            result = DataValidator.validate_image_path(sample.optical_path)
            if not result.valid:
                errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        if sample.sar_path:
            result = DataValidator.validate_image_path(sample.sar_path)
            if not result.valid:
                errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        # Check annotations
        if sample.question is None and sample.answer is not None:
            warnings.append(f"Sample {sample.sample_id}: Answer provided without question")
        
        if sample.question is not None and sample.answer is None:
            warnings.append(f"Sample {sample.sample_id}: Question provided without answer")
        
        # Check labels
        if sample.labels is not None and len(sample.labels) == 0:
            warnings.append(f"Sample {sample.sample_id}: Empty labels list")
        
        valid = len(errors) == 0
        
        return ValidationResult(
            valid=valid,
            total_samples=1,
            valid_samples=1 if valid else 0,
            invalid_samples=0 if valid else 1,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def validate_dimensions_consistency(image1_path: str, image2_path: str) -> ValidationResult:
        """
        Validate that two images have compatible dimensions.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            ValidationResult with compatibility status
        """
        errors = []
        warnings = []
        
        if not PIL_AVAILABLE:
            warnings.append("PIL not available, cannot validate dimensions")
            return ValidationResult(
                valid=True,  # Assume valid if we can't check
                total_samples=1,
                valid_samples=1,
                invalid_samples=0,
                errors=errors,
                warnings=warnings
            )
        
        try:
            with Image.open(image1_path) as img1:
                with Image.open(image2_path) as img2:
                    size1 = img1.size
                    size2 = img2.size
                    
                    if size1 != size2:
                        errors.append(
                            f"Dimension mismatch: {image1_path} ({size1[0]}x{size1[1]}) "
                            f"vs {image2_path} ({size2[0]}x{size2[1]})"
                        )
                    
        except Exception as e:
            errors.append(f"Failed to compare dimensions: {e}")
        
        valid = len(errors) == 0
        
        return ValidationResult(
            valid=valid,
            total_samples=1,
            valid_samples=1 if valid else 0,
            invalid_samples=0 if valid else 1,
            errors=errors,
            warnings=warnings
        )
