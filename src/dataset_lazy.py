"""
Per-sample lazy-loading dataset implementation for TrackNet training.

This dataset loads individual samples (3 consecutive frames) directly from disk on-demand,
using a JSON index file for fast sample lookup. Memory efficient for systems with limited RAM.
"""

import os
import json
import numpy as np
import cv2
import pandas as pd
from glob import glob
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image

try:
    from .util import genHeatMap
    from .constants import (
        SIGMA,
        MAG,
        TENNIS_DATASET_CLIP_PLAYER_CSV_DIR,
        TENNIS_DATASET_SELECT_CLIP,
        WIDTH,
        HEIGHT,
        TENNIS_DATASET_GAME_LEVEL_SPLIT_CSV,
        TENNIS_DATASET_CLIP_LEVEL_SPLIT_CSV,
        TENNIS_DATASET_ROOT,
    )
except ImportError:
    from util import genHeatMap
    from constants import (
        SIGMA,
        MAG,
        TENNIS_DATASET_CLIP_PLAYER_CSV_DIR,
        TENNIS_DATASET_SELECT_CLIP,
        WIDTH,
        HEIGHT,
        TENNIS_DATASET_GAME_LEVEL_SPLIT_CSV,
        TENNIS_DATASET_CLIP_LEVEL_SPLIT_CSV,
        TENNIS_DATASET_ROOT,
    )


class PerSampleDataset(Dataset):
    """
    A per-sample lazy-loading dataset that:
    - Returns individual samples (3 consecutive frames) 
    - Loads images directly from disk on-demand
    - Generates heatmaps on-the-fly
    - Uses a JSON index file for fast sample lookup
    - Memory efficient: only loads 3 frames at a time
    
    Each sample consists of:
    - x: (9, H, W) tensor - 3 RGB frames concatenated
    - y: (3, 2, H, W) tensor - ball and player heatmaps for each frame
    """
    
    def __init__(
        self,
        root_dir,
        mode,
        split_type='game_level',
        target_img_height=HEIGHT,
        target_img_width=WIDTH,
        sequence_dim=(3, 3),
        mag=MAG,
        sigma=SIGMA,
        subset_fraction=1.0,
        use_augmentation=False,
        shuffle_index=False,
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.split_type = split_type
        self.target_img_height = target_img_height
        self.target_img_width = target_img_width
        self.sequence_dim = sequence_dim
        self.mag = mag
        self.sigma = sigma
        self.subset_fraction = subset_fraction
        self.use_augmentation = use_augmentation
        
        self.transform = ToTensor()
        
        # Build or load index of samples
        self.index_dir = os.path.join(root_dir, "sample_index")
        os.makedirs(self.index_dir, exist_ok=True)
        
        self.index_path = os.path.join(self.index_dir, f"{split_type}_{mode}_samples.json")
        self.samples = self._build_or_load_index()
        
        # Apply subset fraction if specified
        if self.subset_fraction < 1.0:
            num_samples = int(len(self.samples) * self.subset_fraction)
            self.samples = self.samples[:max(1, num_samples)]
        
        # Double the index: each original sample is followed by its flipped twin.
        # A sample is a flip if its index in the doubled list is >= len(original).
        # We store the original length and derive flip flag from __getitem__ idx.
        if self.use_augmentation:
            self._base_len = len(self.samples)
            self.samples = self.samples + self.samples  # duplicate list
        else:
            self._base_len = len(self.samples)
        # Optionally shuffle the index
        if shuffle_index:
            import random
            random.shuffle(self.samples)
    
    def _build_or_load_index(self):
        """Build or load the sample index from disk."""
        if os.path.exists(self.index_path):
            print(f"Loading existing index from {self.index_path}")
            with open(self.index_path, 'r') as f:
                return json.load(f)
        
        return self._build_index()
    
    def _build_index(self):
        """Build an index of all individual samples in the dataset."""
        print(f"Building per-sample index for {self.split_type}/{self.mode}...")
        
        # Load valid clips
        with open(TENNIS_DATASET_SELECT_CLIP, 'r') as f:
            selected_clips = f.read().splitlines()
        valid_clips = set()
        for clip in selected_clips:
            if clip.strip():
                parts = clip.strip().split('/')
                if len(parts) == 2:
                    valid_clips.add((parts[0], parts[1]))
        
        # Get split information
        if self.split_type == "game_level":
            train_games, test_games = self._get_game_level_split()
            if self.mode == 'train':
                games = train_games
            else:
                games = test_games
            clips_to_process = None
        else:
            train_clips, test_clips = self._get_clip_level_split()
            if self.mode == 'train':
                clips_to_process = train_clips
            else:
                clips_to_process = test_clips
            games = None
        
        samples = []
        
        if clips_to_process is not None:
            # Clip level split
            for game, clips in clips_to_process.items():
                for clip in clips:
                    if (game, clip) not in valid_clips:
                        continue
                    clip_samples = self._index_clip(game, clip)
                    samples.extend(clip_samples)
        else:
            # Game level split
            for game in games:
                game_folder = os.path.join(self.root_dir, "Dataset", game)
                if not os.path.exists(game_folder):
                    continue
                clips = sorted([d for d in os.listdir(game_folder) 
                               if os.path.isdir(os.path.join(game_folder, d))])
                for clip in clips:
                    if (game, clip) not in valid_clips:
                        continue
                    clip_samples = self._index_clip(game, clip)
                    samples.extend(clip_samples)
        
        # Save index
        with open(self.index_path, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"Index built with {len(samples)} individual samples")
        return samples
    
    def _index_clip(self, game, clip):
        """
        Index a single clip, returning list of sample definitions.
        Each sample contains paths to 3 consecutive frames.
        """
        clip_folder = os.path.join(self.root_dir, "Dataset", game, clip)
        label_path = os.path.join(clip_folder, 'Label.csv')
        player_label_path = os.path.join(TENNIS_DATASET_CLIP_PLAYER_CSV_DIR, f'{game}_{clip}_players.csv')
        
        if not os.path.exists(label_path):
            return []
        
        has_player_data = os.path.exists(player_label_path)
        
        # Load label data
        label_data = pd.read_csv(label_path)
        file_names = label_data['file name'].values.tolist()
        visibility = label_data['visibility'].values.tolist()
        x_coords = label_data['x-coordinate'].values.tolist()
        y_coords = label_data['y-coordinate'].values.tolist()
        num_frames = len(file_names)
        
        if num_frames < self.sequence_dim[0]:
            return []
        
        # Get original image dimensions for ratio calculation
        sample_image_path = os.path.join(clip_folder, str(file_names[0]))
        if not os.path.exists(sample_image_path):
            return []
        sample_image = Image.open(sample_image_path)
        orig_height = sample_image.height
        ratio = orig_height / self.target_img_height
        
        samples = []
        for i in range(num_frames - (self.sequence_dim[0] - 1)):
            # Create sample with paths to 3 consecutive frames
            frame_paths = []
            ball_coords = []
            
            for j in range(self.sequence_dim[0]):
                idx = i + j
                frame_path = os.path.join(clip_folder, str(file_names[idx]))
                frame_paths.append(frame_path)
                
                # Store ball coordinates (scaled)
                if visibility[idx] == 0:
                    ball_coords.append({'visible': False, 'x': -1, 'y': -1})
                else:
                    ball_coords.append({
                        'visible': True,
                        'x': int(x_coords[idx] / ratio),
                        'y': int(y_coords[idx] / ratio)
                    })
            
            sample = {
                'game': game,
                'clip': clip,
                'start_idx': i,
                'frame_paths': frame_paths,
                'ball_coords': ball_coords,
                'has_player_data': has_player_data,
                'ratio': ratio,
            }
            samples.append(sample)
        
        return samples
    
    def _get_game_level_split(self):
        """Get game level train/test split."""
        df = pd.read_csv(TENNIS_DATASET_GAME_LEVEL_SPLIT_CSV)
        train_games = df[df['set'] == 'train']['game'].tolist()
        test_games = df[df['set'] == 'test']['game'].tolist()
        return train_games, test_games
    
    def _get_clip_level_split(self):
        """Get clip level train/test split."""
        df = pd.read_csv(TENNIS_DATASET_CLIP_LEVEL_SPLIT_CSV)
        train_dict = defaultdict(list)
        test_dict = defaultdict(list)
        
        for _, row in df.iterrows():
            if row['set'] == 'train':
                train_dict[row['game']].append(row['clip'])
            elif row['set'] == 'test':
                test_dict[row['game']].append(row['clip'])
        
        return dict(train_dict), dict(test_dict)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load a single sample (3 consecutive frames) on-demand."""
        sample_info = self.samples[idx]
        # Second half of the doubled list => apply horizontal flip
        do_hflip = self.use_augmentation and (idx >= self._base_len)
        return self._load_sample(sample_info, do_hflip=do_hflip)

    
    def _load_sample(self, sample_info, do_hflip=False):
        """Load a single sample from disk."""
        frame_paths = sample_info['frame_paths']
        ball_coords = sample_info['ball_coords']
        has_player_data = sample_info.get('has_player_data', False)
        ratio = sample_info.get('ratio', 1.0)
        game = sample_info['game']
        clip = sample_info['clip']
        start_idx = sample_info['start_idx']
        
        # Load the 3 frames
        frames_sequence = []
        for frame_path in frame_paths:
            img = Image.open(frame_path)
            img = img.resize((self.target_img_width, self.target_img_height), Image.BILINEAR)
            img_tensor = self.transform(img)
            if do_hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            frames_sequence.append(img_tensor)
        
        # Stack frames: (3, 3, H, W) -> (9, H, W)
        x_tensor = torch.stack(frames_sequence, dim=0)
        x_tensor = x_tensor.view(-1, self.target_img_height, self.target_img_width)
        
        # Generate ball heatmaps
        heatmap_sequence = []
        for coord in ball_coords:
            if not coord['visible']:
                heatmap = genHeatMap(self.target_img_width, self.target_img_height, -1, -1, self.sigma, self.mag)
            else:
                x = (self.target_img_width - 1 - coord['x']) if do_hflip else coord['x']
                heatmap = genHeatMap(
                    self.target_img_width, self.target_img_height,
                    x, coord['y'],
                    self.sigma, self.mag
                )
            heatmap_sequence.append(heatmap)
        heatmap_sequence = np.stack(heatmap_sequence, axis=0)  # (3, H, W)
        
        # Generate player heatmaps
        player_heatmap_sequence = []
        if has_player_data:
            player_label_path = os.path.join(TENNIS_DATASET_CLIP_PLAYER_CSV_DIR, f'{game}_{clip}_players.csv')
            if os.path.exists(player_label_path):
                player_data = pd.read_csv(player_label_path)
                
                for j in range(self.sequence_dim[1]):
                    frame_idx = start_idx + j
                    player_heatmap = np.zeros((self.target_img_height, self.target_img_width), dtype=np.float32)
                    
                    # Get frame name from path
                    frame_name = os.path.basename(frame_paths[j])
                    try:
                        frame_num = int(os.path.splitext(frame_name)[0])
                        frame_players = player_data[player_data['frame'] == frame_num]
                        for _, prow in frame_players.iterrows():
                            x1 = int(prow['x1'] / ratio)
                            y1 = int(prow['y1'] / ratio)
                            x2 = int(prow['x2'] / ratio)
                            y2 = int(prow['y2'] / ratio)
                            # Clamp coordinates
                            x1 = max(0, min(x1, self.target_img_width - 1))
                            y1 = max(0, min(y1, self.target_img_height - 1))
                            x2 = max(0, min(x2, self.target_img_width - 1))
                            y2 = max(0, min(y2, self.target_img_height - 1))
                            cv2.rectangle(player_heatmap, (x1, y1), (x2, y2), 1.0, thickness=-1)
                    except (ValueError, KeyError):
                        pass

                    if do_hflip:
                        player_heatmap = np.fliplr(player_heatmap)
                    
                    player_heatmap_sequence.append(player_heatmap)
            else:
                for _ in range(self.sequence_dim[1]):
                    player_heatmap_sequence.append(np.zeros((self.target_img_height, self.target_img_width), dtype=np.float32))
        else:
            for _ in range(self.sequence_dim[1]):
                player_heatmap_sequence.append(np.zeros((self.target_img_height, self.target_img_width), dtype=np.float32))
        
        player_heatmap_sequence = np.stack(player_heatmap_sequence, axis=0)  # (3, H, W)
        
        # Combine heatmaps: (3, 2, H, W)
        y_data = np.stack([heatmap_sequence, player_heatmap_sequence], axis=1)
        y_tensor = torch.from_numpy(y_data).float()
        
        return x_tensor, y_tensor
    
    def rebuild_index(self):
        """Force rebuild of the sample index."""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        self.samples = self._build_index()
    
    def get_sample_info(self, idx):
        """Get metadata about a sample without loading it."""
        return self.samples[idx]


def get_per_sample_dataset(dataset_name, height=HEIGHT, width=WIDTH, subset_fraction=1.0):
    """
    Get per-sample lazy-loading dataset instances.
    
    Args:
        dataset_name: Name of the dataset ('tennis_game_level_split' or 'tennis_clip_level_split')
        height: Target image height
        width: Target image width
        subset_fraction: Fraction of dataset to use (for convergence testing)
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if dataset_name == 'tennis_game_level_split':
        train_ds = PerSampleDataset(
            root_dir=TENNIS_DATASET_ROOT,
            mode='train',
            split_type='game_level',
            target_img_height=height,
            target_img_width=width,
            subset_fraction=subset_fraction,
            use_augmentation=True,
        )
        val_ds = PerSampleDataset(
            root_dir=TENNIS_DATASET_ROOT,
            mode='val',
            split_type='game_level',
            target_img_height=height,
            target_img_width=width,
            subset_fraction=subset_fraction,
            use_augmentation=False,
        )
        return train_ds, val_ds
    elif dataset_name == 'tennis_clip_level_split':
        train_ds = PerSampleDataset(
            root_dir=TENNIS_DATASET_ROOT,
            mode='train',
            split_type='clip_level',
            target_img_height=height,
            target_img_width=width,
            subset_fraction=subset_fraction,
            use_augmentation=True,
        )
        val_ds = PerSampleDataset(
            root_dir=TENNIS_DATASET_ROOT,
            mode='val',
            split_type='clip_level',
            target_img_height=height,
            target_img_width=width,
            subset_fraction=subset_fraction,
            use_augmentation=False,
        )
        return train_ds, val_ds
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


# Keep backward compatibility with old name
def get_lazy_dataset(dataset_name, height=HEIGHT, width=WIDTH, subset_fraction=1.0, use_disk_cache=True):
    """Backward compatible function - now uses per-sample dataset."""
    return get_per_sample_dataset(dataset_name, height, width, subset_fraction)
