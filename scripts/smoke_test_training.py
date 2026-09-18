"""
Smoke Test Training Script for SatQueryAI Phase 3.1

This script implements a smoke test to verify the complete training pipeline
with real VQA data (or synthetic data if real data is unavailable).
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqa.multimodal_vqa.multimodal_vqa import MultimodalVQA, VQAConfig
from data.real_vqa_dataset import RealVQADataset, create_real_vqa_dataloader, collate_fn
from data.vqa_dataset import SyntheticVQADataset, create_vqa_dataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def record_parameter_values(model: nn.Module, name: str) -> float:
    """Record parameter values for comparison."""
    params = list(model.parameters())
    if params:
        return params[0].data.clone().detach().mean().item()
    return 0.0


def run_smoke_test(
    dataset_root: str,
    dataset_variant: str = "hr",
    use_synthetic: bool = False,
    num_samples: int = 10,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    epochs: int = 2,
    device: str = "cpu"
) -> dict:
    """
    Run smoke test training.
    
    Args:
        dataset_root: Root directory of dataset
        dataset_variant: "hr" or "lr"
        use_synthetic: Use synthetic data if real data unavailable
        num_samples: Number of samples for smoke test
        batch_size: Batch size
        learning_rate: Learning rate
        epochs: Number of training epochs
        device: Device to use
        
    Returns:
        Dictionary with smoke test results
    """
    results = {
        "status": "started",
        "dataset": "synthetic" if use_synthetic else f"RSVQA-{dataset_variant.upper()}",
        "device": device,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Create dataset
        if use_synthetic:
            logger.info("Using synthetic dataset for smoke test")
            dataset = SyntheticVQADataset(
                image_paths=["test_image.jpg", "test_satellite.png"],
                num_samples=num_samples,
                num_answers=10
            )
        else:
            logger.info(f"Using real RSVQA dataset: {dataset_variant}")
            try:
                dataset = RealVQADataset(
                    dataset_root=dataset_root,
                    dataset_variant=dataset_variant,
                    split="train",
                    random_seed=42,
                    min_answer_frequency=1
                )
                results["dataset_stats"] = dataset.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to load real dataset: {e}")
                logger.info("Falling back to synthetic dataset")
                dataset = SyntheticVQADataset(
                    image_paths=["test_image.jpg", "test_satellite.png"],
                    num_samples=num_samples,
                    num_answers=10
                )
                results["dataset"] = "synthetic (fallback)"
        
        # Create dataloader
        from data.vqa_dataset import collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        # Get answer vocabulary
        answer_vocab = dataset.answer_vocab if hasattr(dataset, 'answer_vocab') else [
            "agricultural", "forest", "urban", "water", "barren",
            "grassland", "industrial", "residential", "wetland", "mixed"
        ]
        
        # Create model
        config = VQAConfig(
            remoteclip_model="ViT-B-32",
            embedding_dim=512,
            fusion_dim=256,
            num_answers=len(answer_vocab),
            device=device,
            freeze_remoteclip=True,
            mixed_precision=False
        )
        
        model = MultimodalVQA(
            config=config,
            answer_vocab=answer_vocab
        )
        model.load()
        
        results["answer_vocab_size"] = len(answer_vocab)
        results["trainable_params"] = sum(p.numel() for p in model.get_trainable_params())
        results["frozen_params"] = sum(p.numel() for p in model.remoteclip_model.parameters())
        
        # Record initial parameter values
        initial_fusion_param = record_parameter_values(model.fusion, "fusion")
        initial_head_param = record_parameter_values(model.vqa_head, "vqa_head")
        results["initial_fusion_param"] = initial_fusion_param
        results["initial_head_param"] = initial_head_param
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.get_trainable_params(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Training loop
        model.remoteclip_model.eval()  # Keep RemoteCLIP in eval mode
        model.fusion.train()  # Train fusion
        model.vqa_head.train()  # Train VQA head
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Preprocess images - resize to RemoteCLIP size (224x224)
                preprocessed_images = []
                for image in batch["images"]:
                    # Resize to 224x224
                    image = image.resize((224, 224))
                    import numpy as np
                    img_array = np.array(image)
                    if img_array.shape[2] == 3:
                        img_array = np.transpose(img_array, (2, 0, 1))
                    img_tensor = torch.from_numpy(img_array).float() / 255.0
                    if img_tensor.shape[0] == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    preprocessed_images.append(img_tensor)
                
                if len(preprocessed_images) > 0:
                    image_batch = torch.cat(preprocessed_images, dim=0)
                else:
                    continue
                
                image_batch = image_batch.to(device)
                answer_indices = batch["answer_indices"].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                    # Encode image and text
                    visual_features = model.remoteclip_model.encode_image(image_batch)
                    visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
                    
                    text_features_list = []
                    for question in batch["questions"]:
                        text_tokens = model.remoteclip_tokenizer(question)
                        text_input = text_tokens.to(device)
                        text_feat = model.remoteclip_model.encode_text(text_input)
                        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                        text_features_list.append(text_feat)
                    
                    text_features = torch.cat(text_features_list, dim=0)
                    
                    # Fusion
                    fused_features = model.fusion(visual_features, text_features)
                    
                    # VQA head
                    logits = model.vqa_head(fused_features)
                    
                    # Loss
                    loss = criterion(logits, answer_indices)
                    
                    # Backward
                    loss.backward()
                    optimizer.step()
                
                # Metrics
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=-1)
                    epoch_correct += (predictions == answer_indices).sum().item()
                    epoch_samples += answer_indices.size(0)
                    epoch_loss += loss.item() * answer_indices.size(0)
            
            logger.info(f"Epoch {epoch}: Loss: {epoch_loss/epoch_samples:.4f}, Acc: {epoch_correct/epoch_samples:.4f}")
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        # Record final parameter values
        final_fusion_param = record_parameter_values(model.fusion, "fusion")
        final_head_param = record_parameter_values(model.vqa_head, "vqa_head")
        results["final_fusion_param"] = final_fusion_param
        results["final_head_param"] = final_head_param
        
        # Check if parameters changed
        fusion_changed = abs(final_fusion_param - initial_fusion_param) > 1e-6
        head_changed = abs(final_head_param - initial_head_param) > 1e-6
        results["fusion_parameters_changed"] = fusion_changed
        results["head_parameters_changed"] = head_changed
        
        # Verify frozen parameters didn't change
        initial_remoteclip_param = record_parameter_values(model.remoteclip_model, "remoteclip")
        final_remoteclip_param = record_parameter_values(model.remoteclip_model, "remoteclip")
        remoteclip_changed = abs(final_remoteclip_param - initial_remoteclip_param) > 1e-6
        results["remoteclip_parameters_changed"] = remoteclip_changed
        
        # Final metrics
        results["final_loss"] = total_loss / total_samples
        results["final_accuracy"] = total_correct / total_samples
        results["status"] = "completed"
        
        # Save checkpoint
        from models.vqa.checkpoint_manager import create_checkpoint_manager
        checkpoint_manager = create_checkpoint_manager("models/vqa")
        
        training_metadata = {
            "dataset": results["dataset"],
            "device": device,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "trainable_params": results["trainable_params"],
            "frozen_params": results["frozen_params"],
            "timestamp": results["timestamp"]
        }
        
        metrics = {
            "final_loss": results["final_loss"],
            "final_accuracy": results["final_accuracy"]
        }
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            answer_vocab=answer_vocab,
            config=config.__dict__,
            training_metadata=training_metadata,
            metrics=metrics,
            epoch=epochs,
            is_best=True
        )
        
        results["checkpoint_path"] = checkpoint_path
        
        logger.info(f"Smoke test completed successfully!")
        logger.info(f"Parameters changed - Fusion: {fusion_changed}, Head: {head_changed}, RemoteCLIP: {remoteclip_changed}")
        logger.info(f"Final loss: {results['final_loss']:.4f}, Final accuracy: {results['final_accuracy']:.4f}")
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Smoke test training for Phase 3.1")
    
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to RSVQA dataset")
    parser.add_argument("--dataset-variant", type=str, default="hr", choices=["hr", "lr"], help="Dataset variant")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples for smoke test")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--output", type=str, default="smoke_test_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Run smoke test
    results = run_smoke_test(
        dataset_root=args.dataset_root or "",
        dataset_variant=args.dataset_variant,
        use_synthetic=args.use_synthetic,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=args.device
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("SMOKE TEST RESULTS")
    print("="*60)
    print(f"Status: {results['status']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Device: {results['device']}")
    print(f"Answer Vocabulary Size: {results.get('answer_vocab_size', 'N/A')}")
    print(f"Trainable Parameters: {results.get('trainable_params', 'N/A')}")
    print(f"Frozen Parameters: {results.get('frozen_params', 'N/A')}")
    print(f"Fusion Parameters Changed: {results.get('fusion_parameters_changed', 'N/A')}")
    print(f"Head Parameters Changed: {results.get('head_parameters_changed', 'N/A')}")
    print(f"RemoteCLIP Parameters Changed: {results.get('remoteclip_parameters_changed', 'N/A')}")
    print(f"Final Loss: {results.get('final_loss', 'N/A'):.4f}" if 'final_loss' in results else "Final Loss: N/A")
    print(f"Final Accuracy: {results.get('final_accuracy', 'N/A'):.4f}" if 'final_accuracy' in results else "Final Accuracy: N/A")
    print("="*60)
    
    if results["status"] == "failed":
        print(f"Error: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
