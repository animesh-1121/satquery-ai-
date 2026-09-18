"""
FastAPI Integration for SatQueryAI Phase 3 VQA

This module provides FastAPI endpoints for the multimodal VQA system.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import uvicorn

from models.vqa.multimodal_vqa.multimodal_vqa import MultimodalVQA, VQAConfig, create_multimodal_vqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
vqa_model: Optional[MultimodalVQA] = None

# Answer vocabulary
ANSWER_VOCAB = [
    "agricultural",
    "forest", 
    "urban",
    "water",
    "barren",
    "grassland",
    "industrial",
    "residential",
    "wetland",
    "mixed"
]


class VQARequest(BaseModel):
    """VQA request model."""
    question: str
    image_path: Optional[str] = None


class VQAResponse(BaseModel):
    """VQA response model."""
    answer: Optional[str]
    confidence: float
    model: str
    task: str
    question: str
    execution_time_ms: float
    device: str
    top_k_answers: list
    memory: dict
    status: str
    error: Optional[str] = None


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="SatQueryAI VQA API",
        description="Multimodal VQA for remote sensing images",
        version="0.1.0"
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize VQA model on startup."""
        global vqa_model
        logger.info("Initializing VQA model...")
        
        # Try to load answer vocabulary from checkpoint
        checkpoint_path = "models/vqa/best_model.pt"
        answer_vocab = ANSWER_VOCAB.copy()
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'answer_vocab' in checkpoint:
                    answer_vocab = checkpoint['answer_vocab']
                    logger.info(f"Loaded answer vocabulary from checkpoint: {len(answer_vocab)} answers")
            except Exception as e:
                logger.warning(f"Failed to load answer vocabulary from checkpoint: {e}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = VQAConfig(
            remoteclip_model="ViT-B-32",
            embedding_dim=512,
            fusion_dim=256,
            num_answers=len(answer_vocab),
            device=device,
            freeze_remoteclip=True,
            mixed_precision=False
        )
        
        try:
            vqa_model = create_multimodal_vqa(
                answer_vocab=answer_vocab,
                config=config,
                checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None
            )
            logger.info("VQA model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VQA model: {e}")
            vqa_model = None
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "SatQueryAI VQA API",
            "status": "ready" if vqa_model else "error",
            "model": vqa_model.model_name if vqa_model else None
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy" if vqa_model else "unhealthy",
            "model_loaded": vqa_model is not None
        }
    
    @app.post("/vqa", response_model=VQAResponse)
    async def vqa_inference(request: VQARequest):
        """
        VQA inference endpoint.
        
        Args:
            request: VQA request with question and optional image path
            
        Returns:
            VQA response with answer and confidence
        """
        if vqa_model is None:
            raise HTTPException(status_code=503, detail="VQA model not initialized")
        
        if not request.image_path:
            raise HTTPException(status_code=400, detail="image_path is required")
        
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
        
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        try:
            result = vqa_model.predict(
                image_path=request.image_path,
                question=request.question
            )
            
            return VQAResponse(**result)
            
        except Exception as e:
            logger.error(f"VQA inference failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/vqa/upload")
    async def vqa_upload(
        image: UploadFile = File(...),
        question: str = Form(...)
    ):
        """
        VQA inference with uploaded image.
        
        Args:
            image: Uploaded image file
            question: Natural language question
            
        Returns:
            VQA response with answer and confidence
        """
        if vqa_model is None:
            raise HTTPException(status_code=503, detail="VQA model not initialized")
        
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        try:
            # Save uploaded image temporarily
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / f"upload_{image.filename}"
            
            with open(temp_path, "wb") as f:
                f.write(await image.read())
            
            # Run inference
            result = vqa_model.predict(
                image_path=str(temp_path),
                question=question
            )
            
            # Clean up
            temp_path.unlink()
            
            return VQAResponse(**result)
            
        except Exception as e:
            logger.error(f"VQA inference failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api()
