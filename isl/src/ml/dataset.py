"""
PyTorch Dataset for ISL landmark sequences.

Handles:
  - Loading pre-extracted landmark .npy files
  - Padding/truncating sequences to fixed length
  - Data augmentation (time stretch, noise, mirroring, frame drop)
  - Train/val splitting with stratification
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

from src.ml.extract_landmarks import FEATURE_DIM


@dataclass
class AugmentationConfig:
    """Landmark augmentation settings used only for the training split."""

    enabled: bool = True
    noise_std: float = 0.012
    time_stretch_min: float = 0.85
    time_stretch_max: float = 1.18
    frame_drop_prob: float = 0.08
    mirror_prob: float = 0.0
    jitter_shift: float = 0.025
    temporal_crop_prob: float = 0.25
    temporal_crop_min_ratio: float = 0.82


# ─────────────────────────────── Augmentation ─────────────────────────────────


def augment_sequence(
    seq: np.ndarray,
    noise_std: float = 0.02,
    time_stretch_range: Tuple[float, float] = (0.8, 1.2),
    frame_drop_prob: float = 0.1,
    mirror_prob: float = 0.5,
    jitter_shift: float = 0.05,
    temporal_crop_prob: float = 0.0,
    temporal_crop_min_ratio: float = 0.85,
) -> np.ndarray:
    """
    Apply data augmentation to a landmark sequence.
    
    Args:
        seq: (T, FEATURE_DIM) landmark sequence
        noise_std: Gaussian noise standard deviation
        time_stretch_range: (min, max) time stretch factors
        frame_drop_prob: Probability of dropping each frame
        mirror_prob: Probability of left-right mirroring
        jitter_shift: Max random spatial shift
        
    Returns:
        Augmented sequence (possibly different length)
    """
    seq = seq.copy()
    T, D = seq.shape

    # Optional random temporal crop keeps signs centered but varies boundaries.
    if random.random() < temporal_crop_prob and T > 8:
        crop_ratio = random.uniform(temporal_crop_min_ratio, 1.0)
        crop_len = max(5, int(T * crop_ratio))
        if crop_len < T:
            start = random.randint(0, T - crop_len)
            seq = seq[start:start + crop_len]
            T = len(seq)
    
    # 1. Time stretching — resample to different speed
    if random.random() < 0.6:
        stretch = random.uniform(*time_stretch_range)
        new_T = max(3, int(T * stretch))
        indices = np.linspace(0, T - 1, new_T).astype(np.float32)
        # Interpolate each feature
        new_seq = np.zeros((new_T, D), dtype=np.float32)
        for d in range(D):
            new_seq[:, d] = np.interp(indices, np.arange(T), seq[:, d])
        seq = new_seq
        T = new_T
    
    # 2. Random frame dropping
    if random.random() < 0.4 and T > 5:
        keep = [i for i in range(T) if random.random() > frame_drop_prob]
        if len(keep) >= 3:
            seq = seq[keep]
            T = len(seq)
    
    # 3. Gaussian noise on landmark coordinates (not on flags)
    if random.random() < 0.7:
        noise = np.random.randn(T, D).astype(np.float32) * noise_std
        # Don't add noise to presence flags (last 2 features)
        noise[:, -2:] = 0
        seq = seq + noise
    
    # 4. Spatial jitter (small random translation)
    if random.random() < 0.5:
        shift = np.random.uniform(-jitter_shift, jitter_shift, size=3).astype(np.float32)
        # Apply to all xyz coordinates (groups of 3)
        for start in range(0, D - 2, 3):
            seq[:, start:start + 3] += shift
    
    # 5. Left-right mirroring (swap hands, negate x)
    if random.random() < mirror_prob:
        # Swap right hand (0:63) and left hand (63:126)
        right = seq[:, :63].copy()
        left = seq[:, 63:126].copy()
        seq[:, :63] = left
        seq[:, 63:126] = right
        
        # Negate x coordinates (every 3rd starting from 0)
        for start in range(0, D - 2, 3):
            seq[:, start] = -seq[:, start]
        
        # Swap presence flags
        seq[:, -2], seq[:, -1] = seq[:, -1].copy(), seq[:, -2].copy()
    
    return seq


# ─────────────────────────── Dataset Classes ──────────────────────────────────


class ISLDataset(Dataset):
    """
    Dataset for ISL landmark sequences.
    
    Loads pre-extracted .npy files and handles padding/truncation
    to a fixed sequence length.
    """
    
    def __init__(
        self,
        data_dir: str,
        seq_length: int = 30,
        augment: bool = False,
        augmentation_config: Optional[AugmentationConfig] = None,
        class_names: Optional[List[str]] = None,
        class_map: Optional[Dict[str, int]] = None,
        return_length: bool = False,
    ):
        """
        Args:
            data_dir: Path to extracted_data/ directory
            seq_length: Fixed sequence length (pad or truncate)
            augment: Whether to apply data augmentation
            class_names: List of class names (loaded from metadata if None)
            class_map: Dict mapping class name -> class index
        """
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.augment = augment
        self.augmentation_config = augmentation_config or AugmentationConfig(enabled=augment)
        self.return_length = return_length
        
        # Load metadata
        meta_path = self.data_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        
        if class_names is None:
            self.class_names = meta["class_names"]
            self.class_map = meta["class_map"]
        else:
            self.class_names = class_names
            self.class_map = class_map
        
        self.num_classes = len(self.class_names)
        
        # Collect all samples
        self.samples: List[Tuple[str, int]] = []  # (npy_path, class_idx)
        landmarks_dir = self.data_dir / "landmarks"
        
        for class_name, class_idx in self.class_map.items():
            safe_name = class_name.replace(" ", "_")
            class_dir = landmarks_dir / safe_name
            if not class_dir.exists():
                continue
            for npy_file in sorted(class_dir.glob("*.npy")):
                self.samples.append((str(npy_file), class_idx))
        
        # Class weights for balanced sampling
        class_counts = [0] * self.num_classes
        for _, idx in self.samples:
            class_counts[idx] += 1
        
        total = len(self.samples)
        self.class_weights = [
            total / (self.num_classes * max(c, 1)) for c in class_counts
        ]
        self.sample_weights = [self.class_weights[idx] for _, idx in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        seq = np.load(npy_path)  # (T, FEATURE_DIM)

        # Data augmentation
        if self.augment and self.augmentation_config.enabled:
            cfg = self.augmentation_config
            seq = augment_sequence(
                seq,
                noise_std=cfg.noise_std,
                time_stretch_range=(cfg.time_stretch_min, cfg.time_stretch_max),
                frame_drop_prob=cfg.frame_drop_prob,
                mirror_prob=cfg.mirror_prob,
                jitter_shift=cfg.jitter_shift,
                temporal_crop_prob=cfg.temporal_crop_prob,
                temporal_crop_min_ratio=cfg.temporal_crop_min_ratio,
            )

        effective_length = min(len(seq), self.seq_length)

        # Pad or truncate to fixed length
        seq = self._pad_or_truncate(seq)

        seq_tensor = torch.FloatTensor(seq)
        label_tensor = torch.LongTensor([label])[0]

        if self.return_length:
            length_tensor = torch.LongTensor([effective_length])[0]
            return seq_tensor, label_tensor, length_tensor

        return seq_tensor, label_tensor

    def _pad_or_truncate(self, seq: np.ndarray) -> np.ndarray:
        """Ensure sequence is exactly self.seq_length frames."""
        T = len(seq)
        
        if T == self.seq_length:
            return seq
        elif T > self.seq_length:
            # Uniformly sample seq_length frames (preserves temporal structure)
            indices = np.linspace(0, T - 1, self.seq_length, dtype=int)
            return seq[indices]
        else:
            # Pad with zeros at the end
            pad = np.zeros((self.seq_length - T, seq.shape[1]), dtype=np.float32)
            return np.concatenate([seq, pad], axis=0)


def create_data_loaders(
    data_dir: str,
    seq_length: int = 30,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    augmentation_config: Optional[AugmentationConfig] = None,
    augment_repeats: int = 1,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, List[str], Dict]:
    """
    Create train and validation data loaders with stratified split.
    
    Returns:
        (train_loader, val_loader, class_names, metadata)
    """
    # Load metadata
    meta_path = Path(data_dir) / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    
    class_names = meta["class_names"]
    class_map = meta["class_map"]
    
    # Create full dataset (without augmentation) for splitting
    full_dataset = ISLDataset(
        data_dir, seq_length=seq_length, augment=False,
        class_names=class_names, class_map=class_map,
    )
    
    # Stratified split
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Group by class
    class_indices: Dict[int, List[int]] = {}
    for i, (_, label) in enumerate(full_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    train_indices = []
    val_indices = []
    
    for cls, indices in class_indices.items():
        np_rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    
    # Create separate datasets for train (with augmentation) and val
    train_dataset = ISLDataset(
        data_dir, seq_length=seq_length, augment=True,
        augmentation_config=augmentation_config,
        class_names=class_names, class_map=class_map, return_length=True,
    )
    val_dataset = ISLDataset(
        data_dir, seq_length=seq_length, augment=False,
        class_names=class_names, class_map=class_map, return_length=True,
    )
    
    # Subset
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Weighted sampler for balanced classes
    train_weights = [train_dataset.sample_weights[i] for i in train_indices]
    samples_per_epoch = len(train_weights) * max(1, augment_repeats)
    sampler = WeightedRandomSampler(train_weights, samples_per_epoch, replacement=True)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
    print(f"Classes: {len(class_names)}")
    
    return train_loader, val_loader, class_names, meta
