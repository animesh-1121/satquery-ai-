"""
Pair Compatibility Checker for SatQueryAI

This module provides compatibility checking for bi-temporal and optical-SAR
image pairs, ensuring spatial alignment and metadata consistency.
"""

from typing import Dict, Any, Optional, List
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

from .base import Modality
from .validation import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityResult:
    """
    Result of pair compatibility checking.
    
    Attributes:
        compatible: Whether the pair is compatible
        errors: List of error messages
        warnings: List of warning messages
        checks: Dictionary of individual check results
    """
    compatible: bool
    errors: List[str]
    warnings: List[str]
    checks: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compatible": self.compatible,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks
        }


class PairCompatibilityChecker:
    """
    Checker for image pair compatibility.
    
    Checks bi-temporal pairs and optical-SAR pairs for:
    - Dimension compatibility
    - CRS/georeferencing alignment
    - Spatial coverage overlap
    - Modality consistency
    - Metadata consistency
    """
    
    @staticmethod
    def check_temporal_pair(
        image_t1_path: str,
        image_t2_path: str,
        check_spatial: bool = True
    ) -> CompatibilityResult:
        """
        Check compatibility of bi-temporal image pair.
        
        Args:
            image_t1_path: Path to T1 image
            image_t2_path: Path to T2 image
            check_spatial: Whether to perform spatial checks
            
        Returns:
            CompatibilityResult with compatibility status
        """
        errors = []
        warnings = []
        checks = {}
        
        # Check if files exist
        if not Path(image_t1_path).exists():
            errors.append(f"T1 image not found: {image_t1_path}")
            return CompatibilityResult(
                compatible=False,
                errors=errors,
                warnings=warnings,
                checks=checks
            )
        
        if not Path(image_t2_path).exists():
            errors.append(f"T2 image not found: {image_t2_path}")
            return CompatibilityResult(
                compatible=False,
                errors=errors,
                warnings=warnings,
                checks=checks
            )
        
        # Check dimensions
        if PIL_AVAILABLE:
            try:
                with Image.open(image_t1_path) as img1:
                    with Image.open(image_t2_path) as img2:
                        size1 = img1.size
                        size2 = img2.size
                        
                        checks["dimensions"] = {
                            "t1": f"{size1[0]}x{size1[1]}",
                            "t2": f"{size2[0]}x{size2[1]}",
                            "compatible": size1 == size2
                        }
                        
                        if size1 != size2:
                            errors.append(
                                f"Bi-temporal images have incompatible spatial dimensions. "
                                f"T1: {size1[0]}x{size1[1]}, T2: {size2[0]}x{size2[1]}"
                            )
                        
            except Exception as e:
                errors.append(f"Failed to check dimensions: {e}")
                checks["dimensions"] = {"error": str(e)}
        
        # Check spatial alignment if requested and GeoTIFF
        if check_spatial and RASTERIO_AVAILABLE:
            try:
                with rasterio.open(image_t1_path) as src1:
                    with rasterio.open(image_t2_path) as src2:
                        # Check CRS
                        crs1 = src1.crs
                        crs2 = src2.crs
                        
                        checks["crs"] = {
                            "t1": str(crs1) if crs1 else None,
                            "t2": str(crs2) if crs2 else None,
                            "compatible": crs1 == crs2
                        }
                        
                        if crs1 != crs2:
                            warnings.append(
                                f"CRS mismatch: T1 uses {crs1}, T2 uses {crs2}"
                            )
                        
                        # Check transform/alignment
                        transform1 = src1.transform
                        transform2 = src2.transform
                        
                        checks["transform"] = {
                            "t1": str(transform1),
                            "t2": str(transform2),
                            "compatible": transform1 == transform2
                        }
                        
                        if transform1 != transform2:
                            warnings.append(
                                f"Transform mismatch: images may not be spatially aligned"
                            )
                        
                        # Check bounds overlap
                        bounds1 = src1.bounds
                        bounds2 = src2.bounds
                        
                        # Simple overlap check
                        x_overlap = not (bounds1.right < bounds2.left or bounds2.right < bounds1.left)
                        y_overlap = not (bounds1.bottom < bounds2.top or bounds2.bottom < bounds1.top)
                        spatial_overlap = x_overlap and y_overlap
                        
                        checks["spatial_alignment"] = {
                            "t1_bounds": str(bounds1),
                            "t2_bounds": str(bounds2),
                            "overlap": spatial_overlap
                        }
                        
                        if not spatial_overlap:
                            errors.append("Images have no spatial overlap")
                        
            except Exception as e:
                warnings.append(f"Failed to check spatial alignment: {e}")
                checks["spatial_alignment"] = {"error": str(e)}
        
        compatible = len(errors) == 0
        
        return CompatibilityResult(
            compatible=compatible,
            errors=errors,
            warnings=warnings,
            checks=checks
        )
    
    @staticmethod
    def check_optical_sar_pair(
        optical_path: str,
        sar_path: str,
        check_spatial: bool = True
    ) -> CompatibilityResult:
        """
        Check compatibility of optical-SAR image pair.
        
        Args:
            optical_path: Path to optical image
            sar_path: Path to SAR image
            check_spatial: Whether to perform spatial checks
            
        Returns:
            CompatibilityResult with compatibility status
        """
        errors = []
        warnings = []
        checks = {}
        
        # Check if files exist
        if not Path(optical_path).exists():
            errors.append(f"Optical image not found: {optical_path}")
            return CompatibilityResult(
                compatible=False,
                errors=errors,
                warnings=warnings,
                checks=checks
            )
        
        if not Path(sar_path).exists():
            errors.append(f"SAR image not found: {sar_path}")
            return CompatibilityResult(
                compatible=False,
                errors=errors,
                warnings=warnings,
                checks=checks
            )
        
        # Check modality
        checks["modality"] = {
            "optical": Modality.OPTICAL.value,
            "sar": Modality.SAR.value,
            "compatible": True  # Optical-SAR is a valid fusion pair
        }
        
        # Check dimensions
        if PIL_AVAILABLE:
            try:
                with Image.open(optical_path) as img_opt:
                    with Image.open(sar_path) as img_sar:
                        size_opt = img_opt.size
                        size_sar = img_sar.size
                        
                        checks["dimensions"] = {
                            "optical": f"{size_opt[0]}x{size_opt[1]}",
                            "sar": f"{size_sar[0]}x{size_sar[1]}",
                            "compatible": size_opt == size_sar
                        }
                        
                        if size_opt != size_sar:
                            warnings.append(
                                f"Optical and SAR images have different dimensions. "
                                f"Optical: {size_opt[0]}x{size_opt[1]}, SAR: {size_sar[0]}x{size_sar[1]}"
                            )
                        else:
                            logger.info("Optical and SAR dimensions are compatible")
                        
            except Exception as e:
                errors.append(f"Failed to check dimensions: {e}")
                checks["dimensions"] = {"error": str(e)}
        
        # Check spatial alignment if requested and GeoTIFF
        if check_spatial and RASTERIO_AVAILABLE:
            try:
                with rasterio.open(optical_path) as src_opt:
                    with rasterio.open(sar_path) as src_sar:
                        # Check CRS
                        crs_opt = src_opt.crs
                        crs_sar = src_sar.crs
                        
                        checks["crs"] = {
                            "optical": str(crs_opt) if crs_opt else None,
                            "sar": str(crs_sar) if crs_sar else None,
                            "compatible": crs_opt == crs_sar
                        }
                        
                        if crs_opt != crs_sar:
                            warnings.append(
                                f"CRS mismatch: Optical uses {crs_opt}, SAR uses {crs_sar}"
                            )
                        
                        # Check bounds overlap
                        bounds_opt = src_opt.bounds
                        bounds_sar = src_sar.bounds
                        
                        x_overlap = not (bounds_opt.right < bounds_sar.left or bounds_sar.right < bounds_opt.left)
                        y_overlap = not (bounds_opt.bottom < bounds_sar.top or bounds_sar.bottom < bounds_opt.top)
                        spatial_overlap = x_overlap and y_overlap
                        
                        checks["spatial_alignment"] = {
                            "optical_bounds": str(bounds_opt),
                            "sar_bounds": str(bounds_sar),
                            "overlap": spatial_overlap
                        }
                        
                        if not spatial_overlap:
                            errors.append("Optical and SAR images have no spatial overlap")
                        
            except Exception as e:
                warnings.append(f"Failed to check spatial alignment: {e}")
                checks["spatial_alignment"] = {"error": str(e)}
        
        compatible = len(errors) == 0
        
        return CompatibilityResult(
            compatible=compatible,
            errors=errors,
            warnings=warnings,
            checks=checks
        )
    
    @staticmethod
    def check_misalignment(
        image1_path: str,
        image2_path: str,
        tolerance_pixels: int = 5
    ) -> CompatibilityResult:
        """
        Check for spatial misalignment between two images.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            tolerance_pixels: Maximum allowed misalignment in pixels
            
        Returns:
            CompatibilityResult with misalignment status
        """
        errors = []
        warnings = []
        checks = {}
        
        if not RASTERIO_AVAILABLE:
            warnings.append("Rasterio not available, cannot check misalignment")
            return CompatibilityResult(
                compatible=True,  # Assume aligned if we can't check
                errors=errors,
                warnings=warnings,
                checks=checks
            )
        
        try:
            with rasterio.open(image1_path) as src1:
                with rasterio.open(image2_path) as src2:
                    transform1 = src1.transform
                    transform2 = src2.bounds
                    
                    # Calculate offset in pixels
                    offset_x = abs(transform1[2] - transform2[0])
                    offset_y = abs(transform1[5] - transform2[3])
                    
                    # Convert to approximate pixel offset (using resolution)
                    pixel_offset_x = offset_x / abs(transform1[0])
                    pixel_offset_y = offset_y / abs(transform1[4])
                    
                    checks["misalignment"] = {
                        "offset_x_pixels": pixel_offset_x,
                        "offset_y_pixels": pixel_offset_y,
                        "tolerance": tolerance_pixels,
                        "aligned": (pixel_offset_x <= tolerance_pixels and 
                                   pixel_offset_y <= tolerance_pixels)
                    }
                    
                    if pixel_offset_x > tolerance_pixels or pixel_offset_y > tolerance_pixels:
                        errors.append(
                            f"Images are misaligned by {pixel_offset_x:.1f}x{pixel_offset_y:.1f} pixels "
                            f"(tolerance: {tolerance_pixels} pixels)"
                        )
                    else:
                        logger.info(f"Images are aligned within tolerance")
                        
        except Exception as e:
            warnings.append(f"Failed to check misalignment: {e}")
            checks["misalignment"] = {"error": str(e)}
        
        compatible = len(errors) == 0
        
        return CompatibilityResult(
            compatible=compatible,
            errors=errors,
            warnings=warnings,
            checks=checks
        )
