"""
Utility functions for dataset generation, training utilities, and model/dataset management.
"""

import math
import numpy as np
import cv2
import torch

# Handle both package and direct imports
try:
    from .constants import BADMINTON_DATASET_ROOT, TENNIS_DATASET_ROOT, NEW_TENNIS_DATASET_ROOT, WIDTH, HEIGHT
except ImportError:
    from constants import BADMINTON_DATASET_ROOT, TENNIS_DATASET_ROOT, NEW_TENNIS_DATASET_ROOT, WIDTH, HEIGHT

# Lazy load models to avoid import issues
TrackNetV2_pt = None
TrackNetV4_pt = None
TrackNetV4_EfficientUNet = None

def _load_models():
    """Lazy load model classes."""
    global TrackNetV2_pt, TrackNetV4_pt, TrackNetV4_EfficientUNet
    
    if TrackNetV2_pt is None:
        try:
            from .models.TrackNetV2_pt import TrackNetV2 as _TrackNetV2_pt
            TrackNetV2_pt = _TrackNetV2_pt
        except ImportError:
            from models.TrackNetV2_pt import TrackNetV2 as _TrackNetV2_pt
            TrackNetV2_pt = _TrackNetV2_pt
    
    if TrackNetV4_pt is None:
        try:
            from .models.TrackNetV4_pt import TrackNetV4 as _TrackNetV4_pt
            TrackNetV4_pt = _TrackNetV4_pt
        except ImportError:
            from models.TrackNetV4_pt import TrackNetV4 as _TrackNetV4_pt
            TrackNetV4_pt = _TrackNetV4_pt
    
    if TrackNetV4_EfficientUNet is None:
        try:
            from .models.TrackNetv4_EfficientNet import TrackNetV4_EfficientUNet as _TrackNetV4_EfficientUNet
            TrackNetV4_EfficientUNet = _TrackNetV4_EfficientUNet
        except ImportError:
            from models.TrackNetv4_EfficientNet import TrackNetV4_EfficientUNet as _TrackNetV4_EfficientUNet
            TrackNetV4_EfficientUNet = _TrackNetV4_EfficientUNet
####################################
# Dataset related helper functions #
####################################

def genHeatMap(w, h, cx, cy, r, mag):
    """
    Generate a heatmap with a circular region set to a specified magnitude.
    If the center coordinates (cx, cy) are negative, the function returns a zero-filled heatmap.
    """
    if cx < 0 or cy < 0:
        return np.zeros((h, w))

    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap * mag

##############################
# Training related functions #
##############################

def outcome(y_pred, y_true, tol):
    """
    Calculate the outcomes (TP, TN, FP, FN) for a batch of predicted heatmaps versus ground truth.
    """
    n = y_pred.shape[0]
    i = 0
    TP = TN = FP = FN = 0
    while i < n:
        for j in range(3):
            pred_max = torch.max(y_pred[i][j])
            true_max = torch.max(y_true[i][j])

            if pred_max == 0 and true_max == 0:
                TN += 1
            elif pred_max > 0 and true_max == 0:
                FP += 1
            elif pred_max == 0 and true_max > 0:
                FN += 1
            elif pred_max > 0 and true_max > 0:
                h_pred = (y_pred[i][j] * 255).byte().numpy()
                h_true = (y_true[i][j] * 255).byte().numpy()
                
                (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for k in range(len(rects)):
                    area = rects[k][2] * rects[k][3]
                    if area > max_area:
                        max_area_idx = k
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

                (cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for k in range(len(rects)):
                    area = rects[k][2] * rects[k][3]
                    if area > max_area:
                        max_area_idx = k
                        max_area = area
                target = rects[max_area_idx]
                (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
                
                dist = math.sqrt(pow(cx_pred - cx_true, 2) + pow(cy_pred - cy_true, 2))
                if dist > tol:
                    FP += 1
                else:
                    TP += 1
        i += 1
    return (TP, TN, FP, FN)


###################################
# Model/dataset related functions #
###################################

def get_model(model_name, height=HEIGHT, width=WIDTH):
    """
    Returns the specified model.
    """
    global TrackNetV2_pt, TrackNetV4_pt, TrackNetV4_EfficientUNet
    _load_models()
    
    if model_name == 'Baseline_TrackNetV2':
        return TrackNetV2_pt(height, width)
    elif model_name == 'TrackNetV4_TypeA':
        return TrackNetV4_pt(height, width, fusion_layer_type='TypeA')
    elif model_name == 'TrackNetV4_TypeB':
        return TrackNetV4_pt(height, width, fusion_layer_type='TypeB')
    elif model_name == 'TrackNetV4_EfficientNet_B0':
        return TrackNetV4_EfficientUNet(height, width, fusion_layer_type='TypeA', width_mult=1.0, depth_mult=1.0)
    elif model_name == 'TrackNetV4_EfficientNet_B1':
        return TrackNetV4_EfficientUNet(height, width, fusion_layer_type='TypeA', width_mult=1.0, depth_mult=1.1)
    elif model_name == 'TrackNetV4_EfficientNet_Lite':
        return TrackNetV4_EfficientUNet(height, width, fusion_layer_type='TypeA', width_mult=0.75, depth_mult=0.75)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_dataset(dataset_name, height=HEIGHT, width=WIDTH, subset_fraction=1.0, use_lazy=True):
    """
    Returns the specified dataset.
    
    Args:
        dataset_name: Name of the dataset
        height: Target image height
        width: Target image width
        subset_fraction: Fraction of dataset to use (for convergence testing)
        use_lazy: If True, use lazy-loading dataset (recommended for limited RAM/SSD)
    """
    if use_lazy:
        try:
            from .dataset_lazy import get_lazy_dataset
        except ImportError:
            from dataset_lazy import get_lazy_dataset
        return get_lazy_dataset(dataset_name, height, width, subset_fraction)
    
    # Legacy: Use preprocessed dataset (requires preprocessing first)
    try:
        from .dataset import TennisDataset
    except ImportError:
        from dataset import TennisDataset

    if dataset_name == 'tennis_game_level_split':
        train_ds = TennisDataset(root_dir=TENNIS_DATASET_ROOT, mode='train', split_type='game_level', target_img_height=height, target_img_width=width)
        val_ds = TennisDataset(root_dir=TENNIS_DATASET_ROOT, mode='val', split_type='game_level', target_img_height=height, target_img_width=width)
        return train_ds, val_ds
    elif dataset_name == 'tennis_clip_level_split':
        train_ds = TennisDataset(root_dir=TENNIS_DATASET_ROOT, mode='train', split_type='clip_level', target_img_height=height, target_img_width=width)
        val_ds = TennisDataset(root_dir=TENNIS_DATASET_ROOT, mode='val', split_type='clip_level', target_img_height=height, target_img_width=width)
        return train_ds, val_ds
    elif dataset_name == 'badminton':
        # Assuming BadmintonDataset is defined elsewhere
        raise NotImplementedError("Badminton dataset not implemented for PyTorch yet.")
    elif dataset_name == 'new_tennis':
        # Assuming NewTennisDataset is defined elsewhere
        raise NotImplementedError("New tennis dataset not implemented for PyTorch yet.")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
