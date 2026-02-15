#!/usr/bin/env python
"""
Convergence Check Script
------------------------

This script checks if a model is converging by training on a small subset of the dataset
for a limited number of epochs and visualizing the output segmentation.

Usage:
    python src/check_convergence.py --model_name TrackNetV4_TypeA --dataset tennis_game_level_split \
        --epochs 10 --subset_fraction 0.1 --output_dir ./convergence_check

The script will:
1. Train the model on 10% of the dataset for 10 epochs
2. Save sample output visualizations to check if segmentation is working
3. Plot training loss curves
4. Report if the model appears to be converging
"""

import argparse
import os
import sys
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util import get_model
from dataset_lazy import get_lazy_dataset
from constants import HEIGHT, WIDTH


def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def visualize_prediction(model, sample_input, sample_label, device, output_path, epoch):
    """
    Visualize model predictions vs ground truth.
    
    Args:
        model: The trained model
        sample_input: Input tensor (9, H, W)
        sample_label: Ground truth tensor (3, 2, H, W)
        device: torch device
        output_path: Directory to save visualizations
        epoch: Current epoch number
    """
    model.eval()
    with torch.no_grad():
        input_batch = sample_input.unsqueeze(0).to(device)
        output = model(input_batch)
        
        # Handle different model output formats
        if isinstance(output, tuple):
            pred = output[0]  # Ball predictions
        else:
            pred = output
        
        pred = pred.cpu().numpy()[0]  # (3, H, W)
        gt = sample_label.cpu().numpy()  # (3, 2, H, W) -> use ball channel
        
        if gt.ndim == 4 and gt.shape[1] == 2:
            gt = gt[:, 0, :, :]  # Ball heatmaps
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Epoch {epoch} - Convergence Check', fontsize=16)
        
        for i in range(3):
            # Input frame (reconstruct from channels)
            frame = sample_input[i*3:(i+1)*3].cpu().numpy()  # RGB channels
            frame = np.transpose(frame, (1, 2, 0))  # H, W, C
            frame = np.clip(frame, 0, 1)
            
            # Ground truth heatmap
            gt_heatmap = gt[i]
            
            # Predicted heatmap
            pred_heatmap = pred[i]
            
            # Plot
            axes[i, 0].imshow(frame)
            axes[i, 0].set_title(f'Frame {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt_heatmap, cmap='hot', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Ground Truth (max={gt_heatmap.max():.3f})')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_heatmap, cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Prediction (max={pred_heatmap.max():.3f})')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'visualization_epoch_{epoch}.png'), dpi=150)
        plt.close()
        
        # Clear memory
        del input_batch, output, pred
        clear_memory()


def check_segmentation_quality(pred, threshold=0.3):
    """
    Check if the prediction shows meaningful segmentation.
    
    Returns:
        bool: True if segmentation appears meaningful
        dict: Quality metrics
    """
    # Binary mask
    binary = (pred > threshold).astype(np.float32)
    
    # Calculate metrics
    max_val = np.max(pred)
    mean_val = np.mean(pred)
    num_positive = np.sum(binary)
    total_pixels = pred.size
    coverage = num_positive / total_pixels
    
    # Check for meaningful segmentation
    is_meaningful = (
        max_val > 0.3 and
        coverage > 0.0001 and  # At least some activation
        coverage < 0.1  # Not too much (ball should be small)
    )
    
    metrics = {
        'max_value': float(max_val),
        'mean_value': float(mean_val),
        'positive_pixels': int(num_positive),
        'coverage': float(coverage),
        'is_meaningful': is_meaningful
    }
    
    return is_meaningful, metrics


def main(args):
    """Main convergence checking function."""
    # Setup
    model_name = args.model_name
    dataset_name = args.dataset
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    subset_fraction = args.subset_fraction
    output_dir = args.output_dir
    height = args.height
    width = args.width
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CONVERGENCE CHECK")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Subset fraction: {subset_fraction * 100:.1f}%")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Device setup
    if hasattr(args, 'device') and args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    
    # Load subset of dataset (per-sample, not per-clip)
    print(f"\nLoading {subset_fraction * 100:.1f}% of dataset (per-sample mode)...")
    train_dataset, val_dataset = get_lazy_dataset(
        dataset_name, 
        height=height, 
        width=width, 
        subset_fraction=subset_fraction
    )
    
    # DataLoader with batch_size > 1 since each sample is just 3 frames
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = get_model(model_name, height, width)
    if args.model_path:
        print(f"Loading weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # AMP scaler for mixed precision (reduces memory significantly)
    use_amp = torch.cuda.is_available() and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for memory efficiency")
    
    # Training tracking
    train_losses = []
    val_losses = []
    
    # Get a sample for visualization
    sample_input, sample_label = None, None
    for inputs, labels in train_loader:
        sample_input = inputs[0]  # First sample (9, H, W)
        sample_label = labels[0]  # First label (3, 2, H, W)
        break
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # inputs: (B, 9, H, W), labels: (B, 3, 2, H, W)
            inputs = inputs.float().to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                
                # Compute loss outside autocast (BCELoss with sigmoid outputs)
                if isinstance(outputs, tuple):
                    outputs, motion_loss = outputs[0], outputs[1]
                    ball_labels = labels[:, :, 0, :, :]  # (B, 3, H, W)
                    loss = criterion(outputs.float(), ball_labels.float())
                    if isinstance(motion_loss, torch.Tensor) and motion_loss.numel() > 0:
                        loss = loss + motion_loss.mean()
                else:
                    if labels.dim() == 5 and labels.shape[2] == 2:
                        ball_labels = labels[:, :, 0, :, :]
                    else:
                        ball_labels = labels
                    loss = criterion(outputs.float(), ball_labels.float())
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                outputs = model(inputs)
                
                if isinstance(outputs, tuple):
                    outputs, motion_loss = outputs[0], outputs[1]
                    ball_labels = labels[:, :, 0, :, :]  # (B, 3, H, W)
                    loss = criterion(outputs, ball_labels)
                    if isinstance(motion_loss, torch.Tensor) and motion_loss.numel() > 0:
                        loss = loss + motion_loss.mean()
                else:
                    if labels.dim() == 5 and labels.shape[2] == 2:
                        ball_labels = labels[:, :, 0, :, :]
                    else:
                        ball_labels = labels
                    loss = criterion(outputs, ball_labels)
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Progress update every 10 batches
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Periodic memory cleanup
            if (batch_idx + 1) % 50 == 0:
                clear_memory()
        
        epoch_time = time.time() - start_time
        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)
        
        # Clear memory before validation
        clear_memory()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.float().to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                
                if use_amp:
                    with autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if labels.dim() == 5 and labels.shape[2] == 2:
                    ball_labels = labels[:, :, 0, :, :]
                else:
                    ball_labels = labels
                
                val_loss += criterion(outputs.float(), ball_labels.float()).item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Visualize at certain epochs
        if (epoch + 1) in [1, epochs // 2, epochs]:
            visualize_prediction(model, sample_input, sample_label, device, output_dir, epoch + 1)
        
        # Memory cleanup
        clear_memory()
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convergence Check - Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
    plt.close()
    
    # Final evaluation - check segmentation quality
    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS")
    print("=" * 60)
    
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        count = 0
        for inputs, labels in val_loader:
            if count >= 5:  # Check first 5 batches
                break
            
            inputs = inputs.float().to(device)
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Check each sample in the batch
            for i in range(min(outputs.size(0), 3)):  # Up to 3 samples per batch
                pred = outputs[i].cpu().numpy()  # (3, H, W)
                for j in range(3):  # 3 frames per sample
                    is_meaningful, metrics = check_segmentation_quality(pred[j])
                    all_metrics.append(metrics)
            
            count += 1
    
    # Analyze results
    if all_metrics:
        avg_max = np.mean([m['max_value'] for m in all_metrics])
        avg_coverage = np.mean([m['coverage'] for m in all_metrics])
        meaningful_count = sum([m['is_meaningful'] for m in all_metrics])
        total_checked = len(all_metrics)
        
        print(f"\nSegmentation Quality Metrics:")
        print(f"  Average max activation: {avg_max:.4f}")
        print(f"  Average coverage: {avg_coverage:.6f}")
        print(f"  Meaningful predictions: {meaningful_count}/{total_checked} ({100*meaningful_count/total_checked:.1f}%)")
        
        # Convergence assessment
        print("\n" + "-" * 40)
        
        loss_decreased = train_losses[-1] < train_losses[0] * 0.8
        val_reasonable = val_losses[-1] < 1.0
        segmentation_ok = meaningful_count > total_checked * 0.3
        
        if loss_decreased and val_reasonable and segmentation_ok:
            print("✓ CONVERGENCE: Model appears to be converging!")
            print("  - Loss is decreasing")
            print("  - Validation loss is reasonable")
            print("  - Segmentation shows meaningful regions")
            convergence_status = "CONVERGING"
        elif loss_decreased:
            print("⚠ PARTIAL CONVERGENCE: Loss is decreasing but segmentation may need more training")
            convergence_status = "PARTIAL"
        else:
            print("✗ NOT CONVERGING: Model may have issues")
            print("  Suggestions:")
            print("  - Check learning rate (try lower values)")
            print("  - Check data loading")
            print("  - Check model architecture")
            convergence_status = "NOT_CONVERGING"
        
        # Save summary
        summary = {
            'model_name': model_name,
            'dataset': dataset_name,
            'subset_fraction': subset_fraction,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1]),
            'loss_decrease_ratio': float(train_losses[-1] / train_losses[0]) if train_losses[0] > 0 else 1.0,
            'avg_max_activation': float(avg_max),
            'avg_coverage': float(avg_coverage),
            'meaningful_predictions_ratio': float(meaningful_count / total_checked),
            'convergence_status': convergence_status,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
    else:
        convergence_status = "UNKNOWN"
        summary = {
            'model_name': model_name,
            'dataset': dataset_name,
            'convergence_status': convergence_status,
            'error': 'No validation metrics collected'
        }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)
    
    return convergence_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check model convergence on a subset of data")
    
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['Baseline_TrackNetV2', 'TrackNetV4_TypeA', 'TrackNetV4_TypeB',
                                 'TrackNetV4_EfficientNet_B0', 'TrackNetV4_EfficientNet_B1', 
                                 'TrackNetV4_EfficientNet_Lite'],
                        help="Name of the model to test")
    parser.add_argument('--dataset', type=str, default='tennis_game_level_split',
                        choices=['tennis_game_level_split', 'tennis_clip_level_split'],
                        help="Dataset to use")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epochs for convergence check (default: 10)")
    parser.add_argument('--subset_fraction', type=float, default=0.1,
                        help="Fraction of dataset to use (default: 0.1 = 10%%)")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument('--height', type=int, default=288,
                        help="Image height (default: 288)")
    parser.add_argument('--width', type=int, default=512,
                        help="Image width (default: 512)")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Optional: Path to pretrained weights to continue from")
    parser.add_argument('--output_dir', type=str, default='./convergence_check',
                        help="Output directory for results")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use (e.g., cuda:0, cuda:1, cpu). Auto-detects if not specified.")
    
    args = parser.parse_args()
    main(args)
