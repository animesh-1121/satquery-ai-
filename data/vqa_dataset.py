"""
VQA Dataset Loader for SatQueryAI Phase 3

This module provides dataset loaders for VQA training, handling the
conversion of dataset samples into batches suitable for multimodal VQA training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging

from data.base import Sample
from data.remoteclip_preprocessing import RemoteCLIPPreprocessor
from PIL import Image

logger = logging.getLogger(__name__)


class VQADataset(Dataset):
    """
    PyTorch Dataset for VQA training.
    
    Handles loading of images, questions, and answers for multimodal VQA.
    """
    
    def __init__(
        self,
        samples: List[Sample],
        answer_vocab: Dict[str, int],
        max_question_length: int = 77,
        transform=None
    ):
        """
        Initialize VQA dataset.
        
        Args:
            samples: List of Sample objects with question and answer
            answer_vocab: Dictionary mapping answers to indices
            max_question_length: Maximum question length
            transform: Optional image transform
        """
        self.samples = samples
        self.answer_vocab = answer_vocab
        self.max_question_length = max_question_length
        self.transform = transform
        
        # Filter samples that have question and answer
        self.valid_samples = [
            s for s in samples 
            if s.question is not None and s.answer is not None
        ]
        
        logger.info(f"VQA Dataset: {len(self.valid_samples)} valid samples from {len(samples)} total")
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single VQA sample.
        
        Returns:
            Dictionary with image, question, answer, and metadata
        """
        sample = self.valid_samples[idx]
        
        # Load image
        try:
            image = Image.open(sample.image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {sample.image_path}: {e}")
            # Return a dummy sample
            return self._get_dummy_sample()
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Get answer index
        answer = sample.answer
        if answer not in self.answer_vocab:
            # Use a default answer if not in vocab
            answer = list(self.answer_vocab.keys())[0]
        
        answer_idx = self.answer_vocab[answer]
        
        return {
            "image": image,
            "question": sample.question,
            "answer": answer,
            "answer_idx": answer_idx,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset
        }
    
    def _get_dummy_sample(self) -> Dict[str, Any]:
        """Return a dummy sample for error cases."""
        return {
            "image": Image.new('RGB', (224, 224)),
            "question": "What is this?",
            "answer": list(self.answer_vocab.keys())[0],
            "answer_idx": 0,
            "sample_id": "dummy",
            "dataset": "dummy"
        }


class SyntheticVQADataset(Dataset):
    """
    Synthetic VQA dataset for demonstration and training.
    
    Since the existing dataset adapters don't have full annotation parsing,
    this creates synthetic VQA samples based on land cover classification.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        num_samples: int = 100,
        num_answers: int = 10,
        max_question_length: int = 77
    ):
        """
        Initialize synthetic VQA dataset.
        
        Args:
            image_paths: List of image file paths
            num_samples: Number of synthetic samples to generate
            num_answers: Number of possible answers
            max_question_length: Maximum question length
        """
        self.image_paths = image_paths
        self.num_samples = min(num_samples, len(image_paths) * 10)  # Multiple questions per image
        self.num_answers = num_answers
        self.max_question_length = max_question_length
        
        # Generate synthetic questions and answers
        self.questions = [
            "What type of land cover is visible?",
            "What is the dominant land cover?",
            "What is the main land use?",
            "What type of terrain is this?",
            "What is the landscape type?",
            "What is the primary land cover?",
            "What kind of land cover is shown?",
            "What is the area's land cover?",
            "What is the surface cover?",
            "What type of environment is this?"
        ]
        
        # Generate synthetic answers
        self.answers = [
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
        ][:num_answers]
        
        # Create answer vocabulary
        self.answer_vocab = {ans: idx for idx, ans in enumerate(self.answers)}
        self.idx_to_answer = {idx: ans for idx, ans in enumerate(self.answers)}
        
        # Generate samples
        self.samples = []
        for i in range(self.num_samples):
            image_path = image_paths[i % len(image_paths)]
            question = self.questions[i % len(self.questions)]
            answer = self.answers[i % len(self.answers)]
            
            self.samples.append({
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "answer_idx": self.answer_vocab[answer]
            })
        
        logger.info(f"Synthetic VQA Dataset: {len(self.samples)} samples, {len(self.answers)} answers")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a synthetic VQA sample."""
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample["image_path"])
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {sample['image_path']}: {e}")
            image = Image.new('RGB', (224, 224))
        
        return {
            "image": image,
            "question": sample["question"],
            "answer": sample["answer"],
            "answer_idx": sample["answer_idx"],
            "sample_id": f"synthetic_{idx:05d}",
            "dataset": "synthetic"
        }


def create_vqa_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    Create a DataLoader for VQA training.
    
    Args:
        dataset: VQA dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for VQA batch.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    answer_indices = torch.tensor([item["answer_idx"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]
    datasets = [item["dataset"] for item in batch]
    
    return {
        "images": images,
        "questions": questions,
        "answers": answers,
        "answer_indices": answer_indices,
        "sample_ids": sample_ids,
        "datasets": datasets
    }
