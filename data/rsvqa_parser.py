"""
Real RSVQA Annotation Parser for SatQueryAI Phase 3.1

This module implements proper parsing for RSVQA dataset annotations.
Based on the RSVQA dataset format: https://rsvqa.sylvainlobry.com/

RSVQA Format:
- High Resolution (HR): USGS orthophotos (15.24cm)
- Low Resolution (LR): Sentinel-2 imagery
- Question-Answer pairs generated from OpenStreetMap (OSM) data
- Each image has multiple question-answer pairs
- Answers are text strings (not pre-enumerated classes)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import random

logger = logging.getLogger(__name__)


class RSVQAAnnotationParser:
    """
    Parser for RSVQA dataset annotations.
    
    RSVQA datasets typically contain:
    - images/ directory with image files
    - annotations/ directory with JSON or CSV files
    - Each annotation file contains question-answer pairs for images
    """
    
    def __init__(self, dataset_root: str, dataset_variant: str = "hr"):
        """
        Initialize RSVQA annotation parser.
        
        Args:
            dataset_root: Root directory of RSVQA dataset
            dataset_variant: "hr" (high resolution) or "lr" (low resolution)
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_variant = dataset_variant.lower()
        self._annotations = None
        self._answer_vocab = None
        self._stats = None
        
        # Validate dataset variant
        if self.dataset_variant not in ["hr", "lr"]:
            raise ValueError(f"Invalid dataset variant: {dataset_variant}. Must be 'hr' or 'lr'")
    
    def _detect_annotation_format(self) -> str:
        """
        Detect the annotation file format.
        
        Returns:
            Format type: "json", "csv", or "unknown"
        """
        annotations_dir = self.dataset_root / "annotations"
        
        if not annotations_dir.exists():
            logger.warning(f"Annotations directory not found: {annotations_dir}")
            return "unknown"
        
        # Check for JSON files
        json_files = list(annotations_dir.glob("*.json"))
        if json_files:
            return "json"
        
        # Check for CSV files
        csv_files = list(annotations_dir.glob("*.csv"))
        if csv_files:
            return "csv"
        
        logger.warning("No JSON or CSV annotation files found")
        return "unknown"
    
    def _parse_json_annotations(self) -> List[Dict[str, Any]]:
        """
        Parse JSON format annotations.
        
        Expected format (based on RSVQA paper):
        [
            {
                "image_id": "img_001.jpg",
                "question": "What type of building is visible?",
                "answer": "residential"
            },
            ...
        ]
        
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        annotations_dir = self.dataset_root / "annotations"
        
        # Find all JSON files
        json_files = list(annotations_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    annotations.extend(data)
                elif isinstance(data, dict):
                    # If it's a dict, look for common keys
                    if 'annotations' in data:
                        annotations.extend(data['annotations'])
                    elif 'qa_pairs' in data:
                        annotations.extend(data['qa_pairs'])
                    else:
                        # Assume the dict values are the annotations
                        for key, value in data.items():
                            if isinstance(value, list):
                                annotations.extend(value)
                
                logger.info(f"Parsed {len(annotations)} annotations from {json_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to parse {json_file}: {e}")
        
        return annotations
    
    def _parse_csv_annotations(self) -> List[Dict[str, Any]]:
        """
        Parse CSV format annotations.
        
        Expected format:
        image_id,question,answer
        img_001.jpg,What type of building?,residential
        ...
        
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        annotations_dir = self.dataset_root / "annotations"
        
        # Find all CSV files
        csv_files = list(annotations_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                import csv
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        annotations.append(dict(row))
                
                logger.info(f"Parsed {len(annotations)} annotations from {csv_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to parse {csv_file}: {e}")
        
        return annotations
    
    def load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load RSVQA annotations.
        
        Returns:
            List of annotation dictionaries with keys:
            - image_id: Image filename
            - question: Natural language question
            - answer: Answer text
        """
        if self._annotations is not None:
            return self._annotations
        
        # Check if dataset exists
        if not self.dataset_root.exists():
            logger.error(f"Dataset root not found: {self.dataset_root}")
            self._annotations = []
            return self._annotations
        
        # Detect annotation format
        format_type = self._detect_annotation_format()
        
        if format_type == "json":
            self._annotations = self._parse_json_annotations()
        elif format_type == "csv":
            self._annotations = self._parse_csv_annotations()
        else:
            logger.warning("Unknown annotation format, returning empty list")
            self._annotations = []
        
        # Validate annotations
        self._validate_annotations()
        
        return self._annotations
    
    def _validate_annotations(self) -> None:
        """Validate loaded annotations."""
        if not self._annotations:
            return
        
        valid_count = 0
        invalid_count = 0
        
        for ann in self._annotations:
            # Check required fields
            if 'image_id' not in ann or 'question' not in ann or 'answer' not in ann:
                invalid_count += 1
                continue
            
            # Check for empty fields
            if not ann['image_id'] or not ann['question'] or not ann['answer']:
                invalid_count += 1
                continue
            
            valid_count += 1
        
        logger.info(f"Validation: {valid_count} valid, {invalid_count} invalid annotations")
        
        # Remove invalid annotations
        self._annotations = [ann for ann in self._annotations 
                             if 'image_id' in ann and 'question' in ann and 'answer' in ann
                             and ann['image_id'] and ann['question'] and ann['answer']]
    
    def build_answer_vocabulary(self, min_frequency: int = 1) -> Dict[str, int]:
        """
        Build answer vocabulary from annotations.
        
        Args:
            min_frequency: Minimum frequency for an answer to be included
            
        Returns:
            Dictionary mapping answers to indices
        """
        if self._annotations is None:
            self.load_annotations()
        
        if not self._annotations:
            logger.warning("No annotations available for vocabulary building")
            self._answer_vocab = {}
            return self._answer_vocab
        
        # Count answer frequencies
        answer_counter = Counter()
        for ann in self._annotations:
            answer = ann['answer'].strip().lower()
            answer_counter[answer] += 1
        
        # Filter by minimum frequency
        filtered_answers = [ans for ans, count in answer_counter.items() 
                           if count >= min_frequency]
        
        # Sort by frequency (most common first)
        sorted_answers = sorted(filtered_answers, 
                               key=lambda x: answer_counter[x], 
                               reverse=True)
        
        # Build vocabulary
        self._answer_vocab = {ans: idx for idx, ans in enumerate(sorted_answers)}
        
        logger.info(f"Built vocabulary with {len(self._answer_vocab)} answers")
        logger.info(f"Most common answers: {sorted_answers[:10]}")
        
        return self._answer_vocab
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self._annotations is None:
            self.load_annotations()
        
        if self._answer_vocab is None:
            self.build_answer_vocabulary()
        
        stats = {
            "dataset": "RSVQA",
            "variant": self.dataset_variant,
            "total_annotations": len(self._annotations),
            "unique_images": len(set(ann['image_id'] for ann in self._annotations)),
            "unique_questions": len(set(ann['question'] for ann in self._annotations)),
            "unique_answers": len(self._answer_vocab),
            "avg_questions_per_image": len(self._annotations) / len(set(ann['image_id'] for ann in self._annotations)) if self._annotations else 0,
            "answer_distribution": self._get_answer_distribution()
        }
        
        self._stats = stats
        return stats
    
    def _get_answer_distribution(self) -> Dict[str, int]:
        """Get answer frequency distribution."""
        if not self._annotations:
            return {}
        
        answer_counter = Counter()
        for ann in self._annotations:
            answer = ann['answer'].strip().lower()
            answer_counter[answer] += 1
        
        return dict(answer_counter.most_common(20))
    
    def get_split(self, split: str = "train", random_seed: int = 42) -> List[Dict[str, Any]]:
        """
        Get annotations for a specific split.
        
        Note: RSVQA doesn't have official splits in the basic format.
        This method creates reproducible splits.
        
        Args:
            split: "train", "val", or "test"
            random_seed: Random seed for reproducibility
            
        Returns:
            List of annotations for the requested split
        """
        if self._annotations is None:
            self.load_annotations()
        
        if not self._annotations:
            return []
        
        # Create reproducible splits
        random.seed(random_seed)
        random.shuffle(self._annotations)
        
        total = len(self._annotations)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)
        
        if split == "train":
            return self._annotations[:train_end]
        elif split == "val":
            return self._annotations[train_end:val_end]
        elif split == "test":
            return self._annotations[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}")


def create_rsvqa_parser(dataset_root: str, dataset_variant: str = "hr") -> RSVQAAnnotationParser:
    """
    Factory function to create RSVQA annotation parser.
    
    Args:
        dataset_root: Root directory of RSVQA dataset
        dataset_variant: "hr" (high resolution) or "lr" (low resolution)
        
    Returns:
        RSVQAAnnotationParser instance
    """
    return RSVQAAnnotationParser(dataset_root, dataset_variant)
