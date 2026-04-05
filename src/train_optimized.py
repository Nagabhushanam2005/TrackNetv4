#!/usr/bin/env python
"""
Optimized Training Script
-------------------------

This script is optimized for training TrackNet models on systems with limited RAM and SSD.

Key optimizations:
1. Per-sample lazy-loading dataset - loads only 3 frames at a time (no preprocessing required)
2. Memory-efficient training with gradient accumulation
3. Automatic mixed precision (AMP) for faster training and lower memory
4. Gradient checkpointing for larger batch sizes
5. Periodic garbage collection to prevent memory leaks
6. Resume training from checkpoints

Dataset format (per-sample):
- Input: (batch_size, 9, H, W) - 3 consecutive RGB frames stacked
- Labels: (batch_size, 3, 2, H, W) - heatmaps for each frame (ball + player channels)

Usage:
    python src/train_optimized.py --model_name TrackNetV4_TypeA --dataset tennis_game_level_split \
        --batch_size 12 --learning_rate 0.005 --epochs 200 --work_dir ./models

Example (matching your current command):
    python src/train_optimized.py --model_name TrackNetV4_TypeA \
        --dataset tennis_game_level_split \
        --batch_size 12 \
        --learning_rate 0.005 \
        --epochs 200 \
        --work_dir ./models-pytorch-2 \
        --model_path models-pytorch-2/TrackNetV4_TypeA_epoch_47.pth \
        --start_epoch 48
"""

import argparse
import datetime
import os
import sys
import time
import gc
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from torchvision.transforms import ToTensor

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util import get_model, outcome
from dataset_lazy import get_lazy_dataset
from constants import HEIGHT, WIDTH


def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_validation_clip_video(model, val_dataset, device, work_dir, epoch, height, width, use_amp=True, num_samples=30, fps=30):
    """
    Save a random validation clip as a video with model predictions overlaid.
    
    Args:
        model: The trained model
        val_dataset: Validation dataset
        device: Device to run inference on
        work_dir: Directory to save the video
        epoch: Current epoch number
        height: Input height for model
        width: Input width for model
        use_amp: Whether to use automatic mixed precision
        num_samples: Number of consecutive samples to include in the video (default 30 for ~1 second)
        fps: Frames per second for output video
    """
    model.eval()
    
    # Pick a random starting point in the validation dataset
    max_start = max(0, len(val_dataset) - num_samples)
    if max_start == 0:
        start_idx = 0
        num_samples = len(val_dataset)
    else:
        start_idx = random.randint(0, max_start)
    
    # Try to find consecutive samples from the same clip
    # Get the initial sample to find game/clip
    first_sample_info = val_dataset.samples[start_idx]
    target_game = first_sample_info['game']
    target_clip = first_sample_info['clip']
    
    # Find consecutive samples from the same clip
    clip_indices = []
    for i in range(start_idx, min(start_idx + num_samples * 3, len(val_dataset))):
        sample_info = val_dataset.samples[i]
        if sample_info['game'] == target_game and sample_info['clip'] == target_clip:
            clip_indices.append(i)
        if len(clip_indices) >= num_samples:
            break
    
    if len(clip_indices) < 3:
        print(f"  Warning: Could not find enough consecutive samples for video")
        return None
    
    # Set up video writer
    video_path = os.path.join(work_dir, f"val_clip_epoch_{epoch}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    transform = ToTensor()
    
    with torch.no_grad():
        for idx in clip_indices:
            sample_info = val_dataset.samples[idx]
            frame_paths = sample_info['frame_paths']
            ball_coords = sample_info['ball_coords']
            
            # Load frames for inference
            frames_sequence = []
            original_frames = []
            for frame_path in frame_paths:
                img = Image.open(frame_path)
                img = img.resize((width, height), Image.BILINEAR)
                original_frames.append(np.array(img))
                img_tensor = transform(img)
                frames_sequence.append(img_tensor)
            
            # Stack frames: (3, 3, H, W) -> (9, H, W)
            x_tensor = torch.stack(frames_sequence, dim=0)
            x_tensor = x_tensor.view(-1, height, width).unsqueeze(0).float().to(device)
            
            # Run inference
            if use_amp:
                with autocast():
                    outputs = model(x_tensor)
            else:
                outputs = model(x_tensor)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Get predictions as numpy
            preds = outputs.cpu().squeeze(0).numpy()  # (3, H, W)
            
            # Only process the middle frame to avoid duplicate frames in video
            frame_idx = 1  # Middle frame
            frame = original_frames[frame_idx].copy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            pred_heatmap = preds[frame_idx]
            gt_coord = ball_coords[frame_idx]
            
            # Find predicted ball position
            binary_pred = (pred_heatmap > 0.5).astype(np.uint8) * 255
            if np.max(binary_pred) > 0:
                contours, _ = cv2.findContours(binary_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest)
                    pred_x = int(x + w / 2)
                    pred_y = int(y + h / 2)
                    # Draw predicted position (green circle)
                    cv2.circle(frame_bgr, (pred_x, pred_y), 5, (0, 255, 0), -1)
                    cv2.circle(frame_bgr, (pred_x, pred_y), 8, (0, 255, 0), 2)
            
            # Draw ground truth position (red circle)
            if gt_coord['visible']:
                gt_x, gt_y = gt_coord['x'], gt_coord['y']
                cv2.circle(frame_bgr, (gt_x, gt_y), 5, (0, 0, 255), -1)
                cv2.circle(frame_bgr, (gt_x, gt_y), 8, (0, 0, 255), 2)
            
            # Add epoch label
            cv2.putText(frame_bgr, f"Epoch {epoch}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, "Green: Pred, Red: GT", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            video_writer.write(frame_bgr)
    
    video_writer.release()
    print(f"  Validation clip saved: {video_path}")
    return video_path


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, train_losses, val_losses, work_dir, model_name):
    """Save a full checkpoint for resuming training."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    checkpoint_path = os.path.join(work_dir, f"{model_name}_checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # Also save model weights only
    model_path = os.path.join(work_dir, f"{model_name}_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_path)
    
    return model_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load a checkpoint and return the epoch to resume from."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
    
    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load scaler state: {e}")
    
    epoch = checkpoint.get('epoch', 0)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    
    return epoch, train_losses, val_losses


def main(args):
    """Main training function with optimizations for limited resources."""
    
    # Configuration
    model_name = args.model_name
    dataset_name = args.dataset
    batch_size = args.batch_size
    height = args.height
    width = args.width
    learning_rate = args.learning_rate
    epochs = args.epochs
    tol = args.tol
    save_freq = args.save_freq
    model_path = args.model_path
    work_dir = args.work_dir
    start_epoch = args.start_epoch
    use_amp = args.use_amp
    gradient_accumulation_steps = args.gradient_accumulation
    num_workers = args.num_workers
    
    # Create unique work directory if using default
    if work_dir == "./models":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(work_dir, timestamp)
    
    os.makedirs(work_dir, exist_ok=True)
    
    # Print configuration
    config = {
        "model_name": model_name,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "height": height,
        "width": width,
        "epochs": epochs,
        "tol": tol,
        "model_path": model_path,
        "work_dir": work_dir,
        "save_freq": save_freq,
        "start_epoch": start_epoch,
        "use_amp": use_amp,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_workers": num_workers,
    }
    
    print("=" * 60)
    print("OPTIMIZED TRAINING")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Save config
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Memory info
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    
    # Load dataset with lazy loading
    print("\nLoading dataset (lazy mode - per-sample loading)...")
    train_dataset, val_dataset = get_lazy_dataset(dataset_name, height, width)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Per-sample dataset: each item is (9, H, W) input, (3, 2, H, W) label
    # Use actual batch_size since dataset returns individual samples
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,  # Drop incomplete batches for consistent training
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = get_model(model_name, height, width)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load pretrained weights if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from: {model_path}")
        if model_path.endswith('_checkpoint.pth'):
            # Full checkpoint
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            scaler = GradScaler() if use_amp else None
            start_epoch, train_losses, val_losses = load_checkpoint(
                model_path, model, optimizer, scheduler, scaler
            )
            print(f"Resuming from epoch {start_epoch}")
        else:
            # Just weights
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            scaler = GradScaler() if use_amp else None
            train_losses, val_losses = [], []
    else:
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scaler = GradScaler() if use_amp else None
        train_losses, val_losses = [], []
    
    # Loss functions
    criterion = nn.BCELoss()
    logits_criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    outputs_are_logits = None

    for epoch in range(start_epoch, epochs):
        logits_detected_batches = 0
        epoch_start_time = time.time()
        model.train()
        
        running_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Per-sample dataset: inputs (B, 9, H, W), labels (B, 3, 2, H, W)
            inputs = inputs.float().to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            # Mixed precision training
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                
                # Compute loss outside autocast (BCELoss with sigmoid outputs)
                if isinstance(outputs, tuple):
                    outputs, motion_loss = outputs[0], outputs[1]
                    # Handle label format
                    if labels.dim() == 4 and labels.shape[1] == 3:
                        target_labels = labels
                    elif labels.dim() == 5 and labels.shape[2] == 2:
                        target_labels = labels[:, :, 0, :, :]  # Ball channel
                    else:
                        target_labels = labels
                    
                    if outputs_are_logits is None:
                        with torch.no_grad():
                            outputs_min = outputs.min().item()
                            outputs_max = outputs.max().item()
                        outputs_are_logits = (outputs_min < 0.0) or (outputs_max > 1.0)
                        if outputs_are_logits:
                            logits_detected_batches += 1

                    if outputs_are_logits:
                        loss = logits_criterion(outputs.float(), target_labels.float())
                    else:
                        loss = criterion(outputs.float(), target_labels.float())
                    if isinstance(motion_loss, torch.Tensor) and motion_loss.numel() > 0:
                        loss = loss + motion_loss.mean()
                else:
                    if labels.dim() == 5 and labels.shape[2] == 2:
                        target_labels = labels[:, :, 0, :, :]
                    else:
                        target_labels = labels
                    if outputs_are_logits is None:
                        with torch.no_grad():
                            outputs_min = outputs.min().item()
                            outputs_max = outputs.max().item()
                        outputs_are_logits = (outputs_min < 0.0) or (outputs_max > 1.0)
                        if outputs_are_logits:
                            logits_detected_batches += 1

                    if outputs_are_logits:
                        loss = logits_criterion(outputs.float(), target_labels.float())
                    else:
                        loss = criterion(outputs.float(), target_labels.float())
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs, motion_loss = outputs[0], outputs[1]
                    if labels.dim() == 4 and labels.shape[1] == 3:
                        target_labels = labels
                    elif labels.dim() == 5 and labels.shape[2] == 2:
                        target_labels = labels[:, :, 0, :, :]
                    else:
                        target_labels = labels
                    
                    loss = criterion(outputs, target_labels)
                    if isinstance(motion_loss, torch.Tensor) and motion_loss.numel() > 0:
                        loss = loss + motion_loss.mean()
                else:
                    if labels.dim() == 5 and labels.shape[2] == 2:
                        target_labels = labels[:, :, 0, :, :]
                    else:
                        target_labels = labels
                    loss = criterion(outputs, target_labels)
                
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            running_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            # Update weights after gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Progress update
            if (i + 1) % 50 == 0:
                avg_loss = running_loss / num_batches
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], "
                      f"Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Periodic memory cleanup
            if (i + 1) % 200 == 0:
                clear_memory()
        
        # Final optimizer step for remaining gradients
        if (num_batches % gradient_accumulation_steps) != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Update learning rate
        scheduler.step()
        
        avg_train_loss = running_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        
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
                
                if labels.dim() == 4 and labels.shape[1] == 3:
                    target_labels = labels
                elif labels.dim() == 5 and labels.shape[2] == 2:
                    target_labels = labels[:, :, 0, :, :]
                else:
                    target_labels = labels
                
                if outputs_are_logits is None:
                    with torch.no_grad():
                        outputs_min = outputs.min().item()
                        outputs_max = outputs.max().item()
                    outputs_are_logits = (outputs_min < 0.0) or (outputs_max > 1.0)
                    if outputs_are_logits:
                        logits_detected_batches += 1

                if outputs_are_logits:
                    val_loss += logits_criterion(outputs.float(), target_labels.float()).item()
                    pred_probs = torch.sigmoid(outputs)
                else:
                    val_loss += criterion(outputs.float(), target_labels.float()).item()
                    pred_probs = outputs
                val_batches += 1
                
                # Calculate metrics
                preds = (pred_probs > 0.5).cpu()
                t, n, p, f = outcome(target_labels.cpu(), preds, tol)
                tp += t
                tn += n
                fp += p
                fn += f
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # Calculate metrics
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        if logits_detected_batches > 0:
            print(f"  Logits-detected batches: {logits_detected_batches}")
        print()
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            model_save_path = save_checkpoint(
                model, optimizer, scheduler, scaler, 
                epoch + 1, train_losses, val_losses, 
                work_dir, model_name
            )
            print(f"Checkpoint saved: {model_save_path}")
            
            # Save validation clip video for visual inspection
            try:
                save_validation_clip_video(
                    model, val_dataset, device, work_dir, 
                    epoch + 1, height, width, use_amp
                )
            except Exception as e:
                print(f"  Warning: Could not save validation video: {e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(work_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved: {best_path}")
        
        # Memory cleanup after each epoch
        clear_memory()
    
    # Final save
    final_path = os.path.join(work_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved: {final_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    with open(os.path.join(work_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized TrackNet training for limited resources")
    
    # Model and dataset
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['Baseline_TrackNetV2', 'TrackNetV4_TypeA', 'TrackNetV4_TypeB',
                                 'TrackNetV4_EfficientNet_B0', 'TrackNetV4_EfficientNet_B1', 'TrackNetV4_EfficientNet_Lite',
                                 'TrackNetV4Plus_Lite', 'TrackNetV4Plus_Standard', 'TrackNetV4Plus_Large'],
                        help="Name of the model to use")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['tennis_game_level_split', 'tennis_clip_level_split'],
                        help="Name of the dataset to use")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size for training (default: 12)")
    parser.add_argument("--learning_rate", type=float, default=0.005,
                        help="Learning rate (default: 0.005)")
    parser.add_argument("--height", type=int, default=288,
                        help="Target image height (default: 288)")
    parser.add_argument("--width", type=int, default=512,
                        help="Target image width (default: 512)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs (default: 200)")
    parser.add_argument("--tol", type=int, default=4,
                        help="Tolerance for outcome evaluation (default: 4)")
    
    # Checkpointing
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pretrained model or checkpoint to resume from")
    parser.add_argument("--work_dir", type=str, default="./models",
                        help="Directory to save models (default: ./models)")
    parser.add_argument("--save_freq", type=int, default=1,
                        help="Save checkpoint every N epochs (default: 1)")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Starting epoch (for resuming training)")
    
    # Optimization flags
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision (default: True)")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp",
                        help="Disable automatic mixed precision")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loading workers (default: 2)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., cuda:0, cuda:2, cpu)")
    
    args = parser.parse_args()
    main(args)
