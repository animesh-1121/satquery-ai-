"""
Change Detection API for SatQueryAI Phase 4

This module provides a FastAPI endpoint for bi-temporal change detection.
It accepts T1 and T2 images and returns change detection results.

Usage:
    python api/change_detection_api.py

API Endpoint:
    POST /change-detection
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
import tempfile
import os

from models.change_detection import SiameseChangeDetection, ChangeDetectionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SatQueryAI Change Detection API", version="0.1.0")

# Global model instance
model: Optional[SiameseChangeDetection] = None


@app.on_event("startup")
async def startup_event():
    """Load the change detection model on startup."""
    global model
    
    try:
        import open_clip
    except ImportError:
        logger.error("open-clip-torch is not installed. API will not function properly.")
        return
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading change detection model on {device}...")
    
    config = ChangeDetectionConfig(
        device=device,
        freeze_remoteclip=True
    )
    
    # Try to load trained checkpoint if available
    checkpoint_path = "models/change_detection/best_model.pt"
    if not Path(checkpoint_path).exists():
        checkpoint_path = None
        logger.warning("No trained checkpoint found. Using randomly initialized weights.")
    
    model = SiameseChangeDetection(
        config=config,
        checkpoint_path=checkpoint_path
    )
    model.load()
    
    logger.info("Change detection model loaded successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "SatQueryAI Change Detection API",
        "version": "0.1.0",
        "status": "ready" if model is not None else "model_not_loaded"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/change-detection")
async def change_detection(
    image_t1: UploadFile = File(...),
    image_t2: UploadFile = File(...)
):
    """
    Change detection endpoint.
    
    Args:
        image_t1: T1 image (earlier time)
        image_t2: T2 image (later time)
        
    Returns:
        Change detection results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
        f1.write(await image_t1.read())
        f2.write(await image_t2.read())
        t1_path = f1.name
        t2_path = f2.name
    
    try:
        # Run inference
        result = model.predict(
            image_t1_path=t1_path,
            image_t2_path=t2_path
        )
        
        # Clean up temporary files
        os.unlink(t1_path)
        os.unlink(t2_path)
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result.get('error', 'Inference failed'))
        
        return JSONResponse(content=result)
    
    except Exception as e:
        # Clean up temporary files on error
        if os.path.exists(t1_path):
            os.unlink(t1_path)
        if os.path.exists(t2_path):
            os.unlink(t2_path)
        
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
