"""
Models for ISL sign recognition from landmark sequences.

The project supports five variants:
  - ISLModel: baseline BiGRU + attention model
  - ISLModelLite: smaller CNN + BiGRU model for low-latency use
  - ISLModelHybrid: stronger hand-aware temporal model with velocity cues
  - ISLModelTransformer: self-attention model for longer temporal context
  - ISLModelTCN: dilated temporal convolution model for fast benchmarking

All models accept optional padding masks and sequence lengths so padded frames
do not contaminate temporal pooling or recurrent state.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


RIGHT_HAND_DIM = 63
LEFT_HAND_DIM = 63
POSE_DIM = 30
FLAG_DIM = 2
TOTAL_INPUT_DIM = RIGHT_HAND_DIM + LEFT_HAND_DIM + POSE_DIM + FLAG_DIM


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Build a boolean padding mask from per-sequence lengths."""
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps >= lengths.unsqueeze(1)


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean-pool along time while ignoring padded positions."""
    if mask is None:
        return x.mean(dim=1)

    valid = (~mask).unsqueeze(-1).float()
    total = valid.sum(dim=1).clamp_min(1.0)
    return (x * valid).sum(dim=1) / total


class TemporalAttention(nn.Module):
    """Attention pooling over the temporal axis."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        weights = F.softmax(scores, dim=1)

        if mask is not None:
            valid = (~mask).float()
            weights = weights * valid
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class TemporalConvBlock(nn.Module):
    """Residual temporal convolution block for short-range motion patterns."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + x)


class DilatedTemporalBlock(nn.Module):
    """Residual dilated temporal convolution block."""

    def __init__(self, hidden_dim: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(hidden_dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer sequence models."""

    def __init__(self, hidden_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SequenceModelBase(nn.Module):
    """Shared helpers for sequence models."""

    def _resolve_mask(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if mask is None and lengths is not None:
            mask = lengths_to_padding_mask(lengths, x.size(1))
        return mask

    def _run_gru(
        self,
        gru: nn.GRU,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if lengths is None:
            out, _ = gru(x)
            return out

        clipped_lengths = lengths.clamp(min=1, max=x.size(1)).to(torch.int64).cpu()
        packed = pack_padded_sequence(
            x,
            clipped_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = gru(packed)
        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.size(1),
        )
        return out


class ISLModel(SequenceModelBase):
    """Baseline bidirectional GRU model with attention pooling."""

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 71,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = self._resolve_mask(x, mask, lengths)
        x = self.input_norm(x)
        x = self.input_proj(x)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        x = self._run_gru(self.gru, x, lengths)
        pooled = self.attention(x, mask)
        return self.classifier(pooled)


class ISLModelHybrid(SequenceModelBase):
    """
    Stronger hand-aware model for robust sign recognition.

    The model uses separate right/left hand and pose streams, augments them
    with temporal velocity features, applies temporal convolutions for local
    motion, then fuses everything with a BiGRU and attention pooling.
    """

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 71,
        dropout: float = 0.35,
    ):
        super().__init__()

        if input_dim != TOTAL_INPUT_DIM:
            raise ValueError(f"Expected input_dim={TOTAL_INPUT_DIM}, got {input_dim}")

        hand_stream_dim = max(hidden_dim // 4, 48)
        pose_stream_dim = max(hidden_dim // 5, 40)
        flag_stream_dim = max(hidden_dim // 12, 16)
        fused_dim = hand_stream_dim * 2 + pose_stream_dim + flag_stream_dim

        self.input_norm = nn.LayerNorm(input_dim)
        self.right_proj = nn.Sequential(
            nn.Linear(RIGHT_HAND_DIM * 2, hand_stream_dim),
            nn.LayerNorm(hand_stream_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.35),
        )
        self.left_proj = nn.Sequential(
            nn.Linear(LEFT_HAND_DIM * 2, hand_stream_dim),
            nn.LayerNorm(hand_stream_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.35),
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(POSE_DIM * 2, pose_stream_dim),
            nn.LayerNorm(pose_stream_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
        )
        self.flag_proj = nn.Sequential(
            nn.Linear(FLAG_DIM * 2, flag_stream_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.temporal_blocks = nn.Sequential(
            TemporalConvBlock(hidden_dim, dropout),
            TemporalConvBlock(hidden_dim, dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim, num_classes),
        )

    def _velocity(self, x: torch.Tensor) -> torch.Tensor:
        delta = x[:, 1:] - x[:, :-1]
        pad = torch.zeros_like(x[:, :1])
        return torch.cat([pad, delta], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = self._resolve_mask(x, mask, lengths)
        x = self.input_norm(x)
        velocity = self._velocity(x)

        right = torch.cat(
            [x[..., :RIGHT_HAND_DIM], velocity[..., :RIGHT_HAND_DIM]],
            dim=-1,
        )
        left_start = RIGHT_HAND_DIM
        left_end = left_start + LEFT_HAND_DIM
        left = torch.cat(
            [x[..., left_start:left_end], velocity[..., left_start:left_end]],
            dim=-1,
        )
        pose_start = left_end
        pose_end = pose_start + POSE_DIM
        pose = torch.cat(
            [x[..., pose_start:pose_end], velocity[..., pose_start:pose_end]],
            dim=-1,
        )
        flags = torch.cat([x[..., pose_end:], velocity[..., pose_end:]], dim=-1)

        x = torch.cat(
            [
                self.right_proj(right),
                self.left_proj(left),
                self.pose_proj(pose),
                self.flag_proj(flags),
            ],
            dim=-1,
        )
        x = self.fuse(x)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        x = self.temporal_blocks(x.transpose(1, 2)).transpose(1, 2)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        x = self._run_gru(self.gru, x, lengths)
        pooled_attn = self.attention(x, mask)
        pooled_mean = masked_mean(x, mask)
        pooled = torch.cat([pooled_attn, pooled_mean], dim=1)
        return self.classifier(pooled)


class ISLModelTransformer(SequenceModelBase):
    """
    Transformer encoder model for landmark sequence classification.

    This variant is useful when signs need longer-range temporal context than a
    recurrent model captures cleanly. It uses raw landmarks plus velocity cues,
    masked self-attention, and combined attention/mean pooling.
    """

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 71,
        dropout: float = 0.25,
    ):
        super().__init__()

        num_heads = 8 if hidden_dim % 8 == 0 else 4 if hidden_dim % 4 == 0 else 2
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.position = PositionalEncoding(hidden_dim, dropout=dropout * 0.5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention = TemporalAttention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _velocity(self, x: torch.Tensor) -> torch.Tensor:
        delta = x[:, 1:] - x[:, :-1]
        pad = torch.zeros_like(x[:, :1])
        return torch.cat([pad, delta], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = self._resolve_mask(x, mask, lengths)
        x = self.input_norm(x)
        x = torch.cat([x, self._velocity(x)], dim=-1)
        x = self.position(self.input_proj(x))

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        x = self.encoder(x, src_key_padding_mask=mask)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        pooled = torch.cat([self.attention(x, mask), masked_mean(x, mask)], dim=1)
        return self.classifier(pooled)


class ISLModelTCN(SequenceModelBase):
    """
    Dilated temporal-convolution model for fast training experiments.

    TCNs are often strong on small landmark datasets because they learn local
    motion patterns without the optimization cost of a large recurrent model.
    """

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 71,
        dropout: float = 0.3,
    ):
        super().__init__()

        dilations = [2**i for i in range(max(1, num_layers))]
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.35),
        )
        self.temporal = nn.Sequential(
            *[
                DilatedTemporalBlock(hidden_dim, dilation=d, dropout=dropout * 0.55)
                for d in dilations
            ]
        )
        self.attention = TemporalAttention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _velocity(self, x: torch.Tensor) -> torch.Tensor:
        delta = x[:, 1:] - x[:, :-1]
        pad = torch.zeros_like(x[:, :1])
        return torch.cat([pad, delta], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = self._resolve_mask(x, mask, lengths)
        x = self.input_norm(x)
        x = torch.cat([x, self._velocity(x)], dim=-1)
        x = self.input_proj(x)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        pooled = torch.cat([self.attention(x, mask), masked_mean(x, mask)], dim=1)
        return self.classifier(pooled)


class ISLModelLite(SequenceModelBase):
    """Smaller CNN + BiGRU model for low-latency inference."""

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        hidden_dim: int = 128,
        num_classes: int = 71,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = self._resolve_mask(x, mask, lengths)
        x = self.input_norm(x)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        x = self.conv(x.transpose(1, 2)).transpose(1, 2)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        x = self._run_gru(self.gru, x, lengths)
        pooled = self.attention(x, mask)
        return self.classifier(pooled)
