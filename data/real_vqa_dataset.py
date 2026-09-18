"""
Real VQA Dataset Loader for SatQueryAI Phase 3.1

This module provides dataset loaders for VQA training using real RSVQA annotations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from PIL import Image

from data.base import Sample, Modality
from data.rsvqa_parser import RSVQAAnnotationParser, create_rsvqa_parser

logger = logging.getLogger(__name__)


class RealVQADataset(Dataset):
    """
    PyTorch Dataset for VQA training using real RSVQA annotations.
    
    Handles loading of images, questions, and answers from actual VQA datasets.
    """
    
    def __init__(
        self,
        dataset_root: str,
        dataset_variant: str = "hr",
        split: str = "train",
        random_seed: int = 42,
        min_answer_frequency: int = 1,
        transform=None
    ):
        """
        Initialize real VQA dataset.
        
        Args:
            dataset_root: Root directory of RSVQA dataset
            dataset_variant: "hr" (high resolution) or "lr" (low resolution)
            split: "train", "val", or "test"
            random_seed: Random seed for reproducible splits
            min_answer_frequency: Minimum frequency for answer in vocabulary
            transform: Optional image transform
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_variant = dataset_variant
        self.split = split
        self.random_seed = random_seed
        self.min_answer_frequency = min_answer_frequency
        self.transform = transform
        
        # Initialize annotation parser
        self.parser = create_rsvqa_parser(dataset_root, dataset_variant)
        
        # Load annotations
        self.annotations = self.parser.load_annotations()
        
        # Get split
        self.annotations = self.parser.get_split(split, random_seed)
        
        # Build answer vocabulary
        self.answer_vocab = self.parser.build_answer_vocabulary(min_answer_frequency)
        self.answer_to_idx = self.parser._answer_vocab
        self.idx_to_answer = {idx: ans for ans, idx in self.answer_to_idx.items()}
        
        # Validate samples
        self.valid_samples = self._validate_samples()
        
        logger.info(f"Real VQA Dataset: {len(self.valid_samples)} valid samples from {len(self.annotations)} total")
        logger.info(f"Answer vocabulary size: {len(self.answer_vocab)}")
    
    def _validate_samples(self) -> List[Dict[str, Any]]:
        """Validate samples and filter out invalid ones."""
        valid_samples = []
        
        for ann in self.annotations:
            # Check if image exists
            image_path = self.dataset_root / "images" / ann['image_id']
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Check if answer is in vocabulary
            answer = ann['answer'].strip().lower()
            if answer not in self.answer_vocab:
                logger.warning(f"Answer not in vocabulary: {answer}")
                continue
            
            valid_samples.append(ann)
        
        return valid_samples
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single VQA sample.
        
        Returns:
            Dictionary with image, question, answer, and metadata
        """
        ann = self.valid_samples[idx]
        
        # Load image
        image_path = self.dataset_root / "images" / ann['image_id']
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a dummy sample
            return self._get_dummy_sample()
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Get answer
        answer = ann['answer'].strip().lower()
        answer_idx = self.answer_to_idx[answer]
        
        return {
            "image": image,
            "question": ann['question'],
            "answer": answer,
            "answer_idx": answer_idx,
            "sample_id": ann['image_id'],
            "dataset": f"RSVQA-{self.dataset_variant.upper()}",
            "split": self.split
        }
    
    def _get_dummy_sample(self) -> Dict[str, Any]:
        """Return a dummy sample for error cases."""
        first_answer = list(self.answer_vocab.keys())[0] if self.answer_vocab else "unknown"
        return {
            "image": Image.new('RGB', (224, 224)),
            "question": "What is this?",
            "answer": first_answer,
            "answer_idx": self.answer_vocab.get(first_answer, 0),
            "sample_id": "dummy",
            "dataset": "dummy",
            "split": self.split
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.parser.get_statistics()


def create_real_vqa_dataloader(
    dataset_root: str,
    dataset_variant: str = "hr",
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    random_seed: int = 42,
    min_answer_frequency: int = 1
) -> DataLoader:
    """
    Create a DataLoader for real VQA training.
    
    Args:
        dataset_root: Root directory of RSVQA dataset
        dataset_variant: "hr" or "lr"
        split: "train", "val", or "test"
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        random_seed: Random seed for reproducibility
        min_answer_frequency: Minimum answer frequency
        
    Returns:
        DataLoader
    """
    dataset = RealVQADataset(
        dataset_root=dataset_root,
        dataset_variant=dataset_variant,
        split=split,
        random_seed=random_seed,
        min_answer_frequency=min_answer_frequency
    )
    
    from data.vqa_dataset import collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False  # Changed to False for small datasets
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
    splits = [item["split"] for item in batch]
    
    return {
        "images": images,
        "questions": questions,
        "answers": answers,
        "answer_indices": answer_indices,
        "sample_ids": sample_ids,
        "datasets": datasets,
        "splits": splits
    }
