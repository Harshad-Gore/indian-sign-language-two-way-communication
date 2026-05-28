"""
ML-based sign recognizer for real-time inference.

Replaces the rule-based finger_spelling + gesture_recognizer with
a trained LSTM model that classifies landmark sequences.

Works with a sliding window of recent frames, making predictions
on the accumulated sequence.
"""

import os
import json
import time
import numpy as np
import torch
import logging
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List

from src.ml.model import (
    ISLModel,
    ISLModelHybrid,
    ISLModelLite,
    ISLModelTCN,
    ISLModelTransformer,
)
from src.ml.extract_landmarks import (
    _normalize_landmarks,
    FEATURE_DIM,
    POSE_INDICES,
)

logger = logging.getLogger(__name__)


class MLRecognizer:
    """
    Real-time sign language recognizer using trained ML model.
    
    Maintains a sliding window of landmark frames and runs
    the model periodically to classify the current sign.
    """
    
    def __init__(
        self,
        model_path: str = "models/best_model.pt",
        window_size: int = 30,
        min_frames: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        predict_interval: int = 2,
        margin_threshold: Optional[float] = None,
        max_entropy: Optional[float] = None,
        min_active_ratio: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to saved model checkpoint
            window_size: Number of frames in the sliding window (must match seq_length used in training)
            min_frames: Minimum frames before making predictions
            confidence_threshold: Minimum softmax probability to accept a prediction
            predict_interval: Run model every N frames (for efficiency)
            device: 'cpu' or 'cuda' (auto-detected if None)
        """
        self.window_size = window_size
        self.min_frames = min_frames
        self.confidence_threshold = confidence_threshold
        self.predict_interval = predict_interval
        self.margin_threshold = margin_threshold
        self.max_entropy = max_entropy
        self.min_active_ratio = min_active_ratio
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self._load_model(model_path)
        
        # Frame buffer
        self.frame_buffer: deque = deque(maxlen=self.window_size)
        self.frame_count = 0
        
        # Cache last prediction
        self._last_prediction: Optional[str] = None
        self._last_confidence: float = 0.0
        self._last_predict_time: float = 0.0
        self._last_topk: List[Tuple[str, float]] = []
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Train first: python -m src.ml.train"
            )
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.class_names = checkpoint["class_names"]
        self.num_classes = checkpoint["num_classes"]
        model_type = checkpoint.get("model_type", "full")
        hidden_dim = checkpoint.get("hidden_dim", 256)
        num_layers = checkpoint.get("num_layers", 2)
        dropout = checkpoint.get("dropout", 0.4)
        input_dim = checkpoint.get("input_dim", FEATURE_DIM)
        self.seq_length = checkpoint.get("seq_length", 30)
        recommended_min_frames = checkpoint.get("recommended_min_frames", max(10, self.seq_length // 2))
        recommended_confidence = checkpoint.get("recommended_confidence_threshold", 0.60)
        recommended_margin = checkpoint.get("recommended_margin_threshold", 0.08)
        recommended_max_entropy = checkpoint.get("recommended_max_entropy", 0.55)

        if self.min_frames is None:
            self.min_frames = recommended_min_frames
        if self.confidence_threshold is None:
            self.confidence_threshold = recommended_confidence
        if self.margin_threshold is None:
            self.margin_threshold = recommended_margin
        if self.max_entropy is None:
            self.max_entropy = recommended_max_entropy

        # Update window size to match model
        if self.window_size != self.seq_length:
            logger.info(f"Adjusting window_size from {self.window_size} to {self.seq_length}")
            self.window_size = self.seq_length
            self.frame_buffer = deque(maxlen=self.window_size)
        
        # Create model
        if model_type == "lite":
            self.model = ISLModelLite(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                dropout=dropout,
            )
        elif model_type == "hybrid":
            self.model = ISLModelHybrid(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=self.num_classes,
                dropout=dropout,
            )
        elif model_type == "transformer":
            self.model = ISLModelTransformer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=self.num_classes,
                dropout=dropout,
            )
        elif model_type == "tcn":
            self.model = ISLModelTCN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=self.num_classes,
                dropout=dropout,
            )
        else:
            self.model = ISLModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=self.num_classes,
                dropout=dropout,
            )
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        val_acc = checkpoint.get("val_acc", 0)
        logger.info(
            f"Loaded {model_type} model: {self.num_classes} classes, "
            f"val_acc={val_acc:.1%}, device={self.device}"
        )

    def _build_sequence_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a padded sequence tensor and its true length."""
        seq = np.array(list(self.frame_buffer), dtype=np.float32)
        seq_length = min(len(seq), self.seq_length)

        if len(seq) < self.seq_length:
            pad = np.zeros((self.seq_length - len(seq), FEATURE_DIM), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        elif len(seq) > self.seq_length:
            indices = np.linspace(0, len(seq) - 1, self.seq_length, dtype=int)
            seq = seq[indices]
            seq_length = self.seq_length

        x = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        lengths = torch.LongTensor([seq_length]).to(self.device)
        return x, lengths

    def _infer_probabilities(self) -> Optional[torch.Tensor]:
        """Run the classifier on the current buffer and return probabilities."""
        if len(self.frame_buffer) < self.min_frames:
            return None

        x, lengths = self._build_sequence_tensor()
        mask = torch.arange(self.seq_length, device=self.device).unsqueeze(0) >= lengths.unsqueeze(1)

        with torch.no_grad():
            logits = self.model(x, mask=mask, lengths=lengths)
            return torch.softmax(logits, dim=1)[0]
    
    def add_frame(
        self,
        right_hand: Optional[np.ndarray],
        left_hand: Optional[np.ndarray],
        pose: Optional[np.ndarray],
    ):
        """
        Add a frame's landmarks to the sliding window.
        
        Args:
            right_hand: (21, 3) right hand landmarks or None
            left_hand: (21, 3) left hand landmarks or None
            pose: (33, 4) pose landmarks or None
        """
        features = _normalize_landmarks(right_hand, left_hand, pose)
        self.frame_buffer.append(features)
        self.frame_count += 1
    
    def predict(self, force: bool = False) -> Tuple[Optional[str], float]:
        """
        Get the current prediction.
        
        Args:
            force: If True, always run inference (ignore predict_interval)
            
        Returns:
            (sign_name, confidence) or (None, 0.0) if no prediction
        """
        # Not enough frames
        if len(self.frame_buffer) < self.min_frames:
            return None, 0.0
        
        # Rate limiting — don't run model every frame
        if not force and self.frame_count % self.predict_interval != 0:
            return self._last_prediction, self._last_confidence
        
        # Check that we actually have hand data in recent frames
        recent = list(self.frame_buffer)[-min(8, len(self.frame_buffer)):]
        hands_present = sum(1 for f in recent if f[-2] > 0.5 or f[-1] > 0.5)
        required_active = max(3, int(np.ceil(len(recent) * self.min_active_ratio)))
        if hands_present < required_active:
            self._last_prediction = None
            self._last_confidence = 0.0
            return None, 0.0

        probs = self._infer_probabilities()
        if probs is None:
            return None, 0.0

        conf, pred_idx = probs.max(dim=0)
        conf = conf.item()
        pred_idx = pred_idx.item()

        sorted_probs, sorted_indices = probs.sort(descending=True)
        margin = sorted_probs[0].item() - sorted_probs[1].item()
        entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-8))).item()
        normalized_entropy = entropy / float(np.log(max(self.num_classes, 2)))
        self._last_topk = [
            (self.class_names[idx.item()], prob.item())
            for prob, idx in zip(sorted_probs[:5], sorted_indices[:5])
        ]

        if (
            conf >= self.confidence_threshold
            and margin >= self.margin_threshold
            and normalized_entropy <= self.max_entropy
        ):
            sign_name = self.class_names[pred_idx]
            self._last_prediction = sign_name
            self._last_confidence = conf
        else:
            self._last_prediction = None
            self._last_confidence = conf
        
        return self._last_prediction, self._last_confidence
    
    def get_top_k(self, k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k predictions with probabilities."""
        if self._last_topk:
            return self._last_topk[:k]

        probs = self._infer_probabilities()
        if probs is None:
            return []

        topk = torch.topk(probs, k=min(k, len(probs)))
        return [
            (self.class_names[idx.item()], prob.item())
            for prob, idx in zip(topk.values, topk.indices)
        ]
    
    def reset(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()
        self.frame_count = 0
        self._last_prediction = None
        self._last_confidence = 0.0
        self._last_topk = []
