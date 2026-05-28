"""
Sign Generator — interfaces with sign-language-translator library,
with a rich procedural fallback for ISL pose synthesis.

The procedural pose library contains hand-crafted landmark poses for 200+
common ISL words, ensuring the demo works without downloading the full SLT
dataset. Each pose is defined as (body_delta, left_hand, right_hand) offsets
from a neutral T-pose skeleton.
"""

from __future__ import annotations
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from config import settings


# ── Neutral skeleton (MediaPipe 33-point body, normalised to [-1, 1]) ─────────

NEUTRAL_BODY = [
    # idx: landmark name
    # 0: nose
    {"x": 0.0,   "y": 0.95,  "z": 0.0},
    # 1-4: eyes / ears
    {"x": -0.05, "y": 0.98,  "z": 0.02},
    {"x": 0.05,  "y": 0.98,  "z": 0.02},
    {"x": -0.08, "y": 0.97,  "z": 0.02},
    {"x": 0.08,  "y": 0.97,  "z": 0.02},
    # 5-6: mouth
    {"x": -0.03, "y": 0.92,  "z": 0.02},
    {"x": 0.03,  "y": 0.92,  "z": 0.02},
    # 7-8: ear
    {"x": -0.12, "y": 0.96,  "z": 0.0},
    {"x": 0.12,  "y": 0.96,  "z": 0.0},
    # 9-10: mouth corners
    {"x": -0.04, "y": 0.91,  "z": 0.02},
    {"x": 0.04,  "y": 0.91,  "z": 0.02},
    # 11-12: shoulders
    {"x": -0.25, "y": 0.75,  "z": 0.0},
    {"x": 0.25,  "y": 0.75,  "z": 0.0},
    # 13-14: elbows
    {"x": -0.30, "y": 0.50,  "z": 0.0},
    {"x": 0.30,  "y": 0.50,  "z": 0.0},
    # 15-16: wrists
    {"x": -0.30, "y": 0.25,  "z": 0.0},
    {"x": 0.30,  "y": 0.25,  "z": 0.0},
    # 17-22: hands (pinky/index knuckles etc.)
    {"x": -0.32, "y": 0.22,  "z": 0.02},
    {"x": 0.32,  "y": 0.22,  "z": 0.02},
    {"x": -0.34, "y": 0.20,  "z": 0.02},
    {"x": 0.34,  "y": 0.20,  "z": 0.02},
    {"x": -0.28, "y": 0.21,  "z": 0.02},
    {"x": 0.28,  "y": 0.21,  "z": 0.02},
    # 23-24: hips
    {"x": -0.12, "y": 0.30,  "z": 0.0},
    {"x": 0.12,  "y": 0.30,  "z": 0.0},
    # 25-32: legs (not animated for upper-body ISL)
    {"x": -0.12, "y": 0.05,  "z": 0.0},
    {"x": 0.12,  "y": 0.05,  "z": 0.0},
    {"x": -0.12, "y": -0.25, "z": 0.0},
    {"x": 0.12,  "y": -0.25, "z": 0.0},
    {"x": -0.12, "y": -0.55, "z": 0.0},
    {"x": 0.12,  "y": -0.55, "z": 0.0},
    {"x": -0.12, "y": -0.60, "z": 0.02},
    {"x": 0.12,  "y": -0.60, "z": 0.02},
]

# ── ISL landmark dataset helpers ────────────────────────────────────────────

POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 7]
_NEUTRAL_ARRAY = np.array([[p["x"], p["y"], p["z"]] for p in NEUTRAL_BODY], dtype=np.float32)
_NEUTRAL_CENTER = (_NEUTRAL_ARRAY[11] + _NEUTRAL_ARRAY[12]) / 2.0
_NEUTRAL_SHOULDER_WIDTH = float(np.linalg.norm(_NEUTRAL_ARRAY[11][:2] - _NEUTRAL_ARRAY[12][:2]) or 1.0)
_TARGET_WRIST_DROP = float(abs(_NEUTRAL_ARRAY[11][1] - _NEUTRAL_ARRAY[15][1]) or 0.5)
_AXIS_FLIP = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
_DEPTH_DAMP = 0.6
_MOTION_AXIS = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
_POSE_DRIVE_INDICES = {11, 12, 13, 14, 15, 16}
_BODY_DRIVE_BLEND = 0.75
_HAND_SHAPE_BLEND = 0.65
_HAND_OFFSET_SCALE = 0.85
_MOTION_BODY_BLEND = 0.92
_MOTION_HAND_BLEND = 0.95

_LABEL_ALIASES = {
    "thankyou": "thank you",
    "goodmorning": "good morning",
    "goodafternoon": "good afternoon",
    "goodevening": "good evening",
    "goodnight": "good night",
    "howareyou": "how are you",
    "youplural": "you (plural)",
}

_ISL_INDEX: Optional[dict] = None
_ISL_SEQUENCE_CACHE: dict[str, np.ndarray] = {}
_MOTION_INDEX: Optional[dict] = None
_MOTION_CACHE: dict[str, dict] = {}


_slt_model = None
_SLT_DISABLED = False


def _disable_slt(reason: str) -> None:
    global _SLT_DISABLED, _slt_model
    if _SLT_DISABLED:
        return
    _SLT_DISABLED = True
    _slt_model = None
    logger.warning(f"SLT disabled: {reason}")


def _get_slt_model():
    global _slt_model
    if _SLT_DISABLED:
        return None
    if _slt_model is not None:
        return _slt_model

    try:
        import sign_language_translator as slt
    except Exception as e:
        logger.debug(f"SLT import failed: {e}")
        return None

    os.environ.setdefault("SLT_DATA_DIR", settings.slt_dataset_dir)

    try:
        _slt_model = slt.models.ConcatenativeSynthesis(
            text_language="en",
            sign_language="pk-sl",
            sign_format="landmarks",
            dataset_dir=settings.slt_dataset_dir,
        )
    except TypeError:
        _slt_model = slt.models.ConcatenativeSynthesis(
            text_language="en",
            sign_language="pk-sl",
            sign_format="landmarks",
        )
    except Exception as e:
        msg = str(e)
        if "embedding model" in msg.lower():
            _disable_slt(msg)
        else:
            logger.warning(f"SLT model init failed: {e}")
        _slt_model = None

    return _slt_model


def _normalize_label(text: str) -> str:
    label = text.strip().lower().replace("_", " ")
    label = re.sub(r"[^a-z0-9()\s]", "", label)
    label = re.sub(r"\s+", " ", label)
    return label


def _resolve_label_from_map(text: str, name_map: dict[str, str]) -> Optional[str]:
    key = _normalize_label(text)
    if key in name_map:
        return name_map[key]

    compact = key.replace(" ", "")
    alias = _LABEL_ALIASES.get(compact)
    if not alias:
        return None

    alias_key = _normalize_label(alias)
    return name_map.get(alias_key)


def _resolve_isl_base_dir() -> Path:
    raw = Path(settings.isl_landmarks_dir)
    if raw.is_absolute():
        return raw

    backend_dir = Path(__file__).resolve().parents[1]
    repo_dir = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / raw, backend_dir / raw, repo_dir / raw]
    for candidate in candidates:
        meta = candidate / "metadata.json"
        landmarks = candidate / "landmarks"
        if meta.exists() and landmarks.exists():
            return candidate.resolve()

    return (Path.cwd() / raw).resolve()


def _resolve_motion_base_dir() -> Path:
    raw = Path(settings.isl_motion_clips_dir)
    if raw.is_absolute():
        return raw

    backend_dir = Path(__file__).resolve().parents[1]
    repo_dir = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / raw, backend_dir / raw, repo_dir / raw]
    for candidate in candidates:
        meta = candidate / "metadata.json"
        if meta.exists():
            return candidate.resolve()

    return (Path.cwd() / raw).resolve()


def _get_motion_index() -> Optional[dict]:
    global _MOTION_INDEX

    if _MOTION_INDEX is not None:
        return _MOTION_INDEX or None

    base_dir = _resolve_motion_base_dir()
    meta_path = base_dir / "metadata.json"
    if not meta_path.exists():
        logger.info(f"Motion clips metadata not found at: {meta_path}")
        _MOTION_INDEX = {}
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning(f"Motion clips metadata load failed: {e}")
        _MOTION_INDEX = {}
        return None

    clips_raw = meta.get("clips", {})
    if not isinstance(clips_raw, dict):
        _MOTION_INDEX = {}
        return None

    name_map: dict[str, str] = {}
    clips_map: dict[str, list[dict[str, Any]]] = {}

    for sign_name, entries in clips_raw.items():
        if not isinstance(sign_name, str) or not isinstance(entries, list):
            continue

        valid_entries: list[dict[str, Any]] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            rel_file = item.get("file")
            if not isinstance(rel_file, str) or not rel_file:
                continue

            clip_path = (base_dir / rel_file).resolve()
            if not clip_path.exists():
                continue

            valid_entries.append(
                {
                    "path": clip_path,
                    "score": float(item.get("score", 0.0)),
                    "frames": int(item.get("frames", 0)),
                    "fps": int(item.get("fps", meta.get("fps", settings.isl_dataset_fps))),
                }
            )

        if not valid_entries:
            continue

        valid_entries.sort(key=lambda x: (-x["score"], -x["frames"]))
        clips_map[sign_name] = valid_entries
        name_map[_normalize_label(sign_name)] = sign_name

    _MOTION_INDEX = {
        "base_dir": base_dir,
        "name_map": name_map,
        "clips_map": clips_map,
        "default_fps": int(meta.get("fps", settings.isl_dataset_fps)),
    }
    return _MOTION_INDEX


def _resolve_motion_sign(gloss: str, index: dict) -> Optional[str]:
    return _resolve_label_from_map(gloss, index["name_map"])


def _load_motion_clip(entry: dict[str, Any]) -> Optional[dict[str, np.ndarray]]:
    key = str(entry["path"])
    cached = _MOTION_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        with np.load(str(entry["path"])) as data:
            body = data["body"].astype(np.float32)
            left = data["left_hand"].astype(np.float32)
            right = data["right_hand"].astype(np.float32)
            left_present = data["left_present"].astype(np.float32)
            right_present = data["right_present"].astype(np.float32)
            fps_arr = data["fps"] if "fps" in data.files else None
            clip_fps = int(fps_arr[0]) if fps_arr is not None and np.size(fps_arr) > 0 else int(entry["fps"])
    except Exception as e:
        logger.warning(f"Failed to load motion clip {entry['path']}: {e}")
        return None

    if body.ndim != 3 or body.shape[1:] != (33, 3):
        return None
    if left.ndim != 3 or left.shape[1:] != (21, 3):
        return None
    if right.ndim != 3 or right.shape[1:] != (21, 3):
        return None

    clip = {
        "body": body,
        "left_hand": left,
        "right_hand": right,
        "left_present": left_present,
        "right_present": right_present,
        "fps": np.array([clip_fps], dtype=np.int32),
    }
    _MOTION_CACHE[key] = clip
    return clip


def _resample_landmark_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) == target_len:
        return seq
    if len(seq) <= 1:
        return np.repeat(seq, max(1, target_len), axis=0)

    t = np.linspace(0, len(seq) - 1, target_len)
    out = np.zeros((target_len, *seq.shape[1:]), dtype=np.float32)
    for i in range(seq.shape[1]):
        for j in range(seq.shape[2]):
            out[:, i, j] = np.interp(t, np.arange(len(seq)), seq[:, i, j])
    return out


def _resample_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    if len(mask) == target_len:
        return mask
    if len(mask) <= 1:
        return np.repeat(mask, max(1, target_len), axis=0)
    t = np.linspace(0, len(mask) - 1, target_len)
    resampled = np.interp(t, np.arange(len(mask)), mask)
    return (resampled > 0.5).astype(np.float32)


def _motion_points_to_landmarks(
    points: np.ndarray,
    neutral: list[dict],
    scale: np.ndarray,
    blend: float,
    drive_indices: Optional[set[int]] = None,
) -> list[dict]:
    out: list[dict] = []
    for i in range(min(len(points), len(neutral))):
        n = neutral[i]
        if drive_indices is not None and i not in drive_indices:
            out.append({"x": float(n["x"]), "y": float(n["y"]), "z": float(n["z"])})
            continue

        p = _motion_transform_point(points[i], scale)
        out.append(
            {
                "x": float(n["x"] + (float(p[0]) - n["x"]) * blend),
                "y": float(n["y"] + (float(p[1]) - n["y"]) * blend),
                "z": float(n["z"] + (float(p[2]) - n["z"]) * blend),
            }
        )
    if len(out) < len(neutral):
        out.extend(neutral[len(out):])
    return out


def _motion_frame_to_pose(
    body_points: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
    left_present: float,
    right_present: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    # Motion clips are shoulder-width normalized. Use uniform scale for stability.
    motion_scale = np.array([
        _NEUTRAL_SHOULDER_WIDTH,
        _NEUTRAL_SHOULDER_WIDTH,
        _NEUTRAL_SHOULDER_WIDTH * _DEPTH_DAMP,
    ], dtype=np.float32)

    body = _motion_points_to_landmarks(
        points=body_points,
        neutral=NEUTRAL_BODY,
        scale=motion_scale,
        blend=_MOTION_BODY_BLEND,
        drive_indices=_POSE_DRIVE_INDICES,
    )

    r_wrist = np.array([body[16]["x"], body[16]["y"], body[16]["z"]], dtype=np.float32)
    l_wrist = np.array([body[15]["x"], body[15]["y"], body[15]["z"]], dtype=np.float32)
    hand_scale = float(motion_scale[0] * 0.35)

    neutral_right = _neutral_hand_at(r_wrist, mirror=False, scale=hand_scale)
    neutral_left = _neutral_hand_at(l_wrist, mirror=True, scale=hand_scale)

    right_hand = neutral_right
    if right_present > 0.5:
        right_hand = _motion_points_to_landmarks(
            points=right_points,
            neutral=neutral_right,
            scale=motion_scale,
            blend=_MOTION_HAND_BLEND,
        )

    left_hand = neutral_left
    if left_present > 0.5:
        left_hand = _motion_points_to_landmarks(
            points=left_points,
            neutral=neutral_left,
            scale=motion_scale,
            blend=_MOTION_HAND_BLEND,
        )

    return body, left_hand, right_hand


def _get_isl_index() -> Optional[dict]:
    global _ISL_INDEX
    if _ISL_INDEX is not None:
        return _ISL_INDEX

    base_dir = _resolve_isl_base_dir()
    meta_path = base_dir / "metadata.json"
    landmarks_dir = base_dir / "landmarks"
    if not meta_path.exists() or not landmarks_dir.exists():
        logger.warning(f"ISL dataset not found at: {base_dir}")
        _ISL_INDEX = None
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning(f"ISL metadata load failed: {e}")
        _ISL_INDEX = None
        return None

    class_names = meta.get("class_names", [])
    name_map = {}
    files_map = {}

    for name in class_names:
        key = _normalize_label(name)
        name_map[key] = name
        safe = name.replace(" ", "_")
        sign_dir = landmarks_dir / safe
        if not sign_dir.exists():
            continue
        files = sorted(sign_dir.glob("*.npy"))
        if files:
            files_map[name] = files

    _ISL_INDEX = {
        "base_dir": base_dir,
        "name_map": name_map,
        "files_map": files_map,
    }
    return _ISL_INDEX


def _resolve_isl_class(gloss: str, index: dict) -> Optional[str]:
    return _resolve_label_from_map(gloss, index["name_map"])


def _load_isl_sequence(class_name: str, index: dict) -> Optional[np.ndarray]:
    if class_name in _ISL_SEQUENCE_CACHE:
        return _ISL_SEQUENCE_CACHE[class_name]

    files = index["files_map"].get(class_name)
    if not files:
        return None

    try:
        seq = np.load(str(files[0]))
    except Exception as e:
        logger.warning(f"ISL sequence load failed for {class_name}: {e}")
        return None

    if seq.ndim != 2 or seq.shape[1] < 158:
        return None

    _ISL_SEQUENCE_CACHE[class_name] = seq
    return seq


def _resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) == target_len:
        return seq
    if len(seq) <= 1:
        return np.repeat(seq, max(1, target_len), axis=0)

    t = np.linspace(0, len(seq) - 1, target_len)
    out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(t, np.arange(len(seq)), seq[:, d])
    return out


def _compute_scale(pose_seq: np.ndarray) -> np.ndarray:
    if pose_seq.size == 0:
        return np.array([_NEUTRAL_SHOULDER_WIDTH, _TARGET_WRIST_DROP, _NEUTRAL_SHOULDER_WIDTH], dtype=np.float32)

    l_sh = pose_seq[:, 1, :2]
    r_sh = pose_seq[:, 2, :2]
    shoulder_widths = np.linalg.norm(l_sh - r_sh, axis=1)
    shoulder_width = float(np.median(shoulder_widths[shoulder_widths > 1e-4]) or 1.0)

    shoulder_y = (pose_seq[:, 1, 1] + pose_seq[:, 2, 1]) * 0.5
    wrist_y = (pose_seq[:, 5, 1] + pose_seq[:, 6, 1]) * 0.5
    drop = wrist_y - shoulder_y
    median_drop = float(np.median(drop[drop > 1e-4]) or 1.0)

    scale_xz = _NEUTRAL_SHOULDER_WIDTH / shoulder_width
    scale_y = _TARGET_WRIST_DROP / median_drop
    scale_z = scale_xz * _DEPTH_DAMP

    return np.array([scale_xz, scale_y, scale_z], dtype=np.float32)


def _transform_point(vec: np.ndarray, scale: np.ndarray) -> np.ndarray:
    v = (vec * _AXIS_FLIP) * scale
    return v + _NEUTRAL_CENTER


def _motion_transform_point(vec: np.ndarray, scale: np.ndarray) -> np.ndarray:
    v = (vec * _MOTION_AXIS) * scale
    return v + _NEUTRAL_CENTER


def _neutral_hand_at(wrist: np.ndarray, mirror: bool, scale: float) -> list[dict]:
    base = _flat_hand(mirror=mirror)
    return [
        {
            "x": float(wrist[0] + p["x"] * scale),
            "y": float(wrist[1] + p["y"] * scale),
            "z": float(wrist[2] + p["z"] * scale),
        }
        for p in base
    ]


def _features_to_landmarks(feat: np.ndarray, scale: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    right = feat[0:63].reshape(21, 3)
    left = feat[63:126].reshape(21, 3)
    pose = feat[126:156].reshape(10, 3)
    right_flag = feat[156] > 0.5
    left_flag = feat[157] > 0.5

    body = _NEUTRAL_ARRAY.copy()
    for idx, p in zip(POSE_INDICES, pose):
        if idx not in _POSE_DRIVE_INDICES:
            continue
        driven = _transform_point(p, scale)
        neutral = _NEUTRAL_ARRAY[idx]
        body[idx] = neutral + (driven - neutral) * _BODY_DRIVE_BLEND

    r_wrist = body[16]
    l_wrist = body[15]
    hand_scale = float(scale[0] * 0.35)

    neutral_right = _neutral_hand_at(r_wrist, mirror=False, scale=hand_scale)
    neutral_left = _neutral_hand_at(l_wrist, mirror=True, scale=hand_scale)

    if right_flag:
        right_pts = [_transform_point(p, scale) for p in right]
        if right_pts:
            offset = r_wrist - right_pts[0]
            right_pts = [r_wrist + (p + offset - r_wrist) * _HAND_OFFSET_SCALE for p in right_pts]
        right_hand = [
            {
                "x": float(n["x"] + (p[0] - n["x"]) * _HAND_SHAPE_BLEND),
                "y": float(n["y"] + (p[1] - n["y"]) * _HAND_SHAPE_BLEND),
                "z": float(n["z"] + (p[2] - n["z"]) * _HAND_SHAPE_BLEND),
            }
            for p, n in zip(right_pts, neutral_right)
        ]
    else:
        right_hand = neutral_right

    if left_flag:
        left_pts = [_transform_point(p, scale) for p in left]
        if left_pts:
            offset = l_wrist - left_pts[0]
            left_pts = [l_wrist + (p + offset - l_wrist) * _HAND_OFFSET_SCALE for p in left_pts]
        left_hand = [
            {
                "x": float(n["x"] + (p[0] - n["x"]) * _HAND_SHAPE_BLEND),
                "y": float(n["y"] + (p[1] - n["y"]) * _HAND_SHAPE_BLEND),
                "z": float(n["z"] + (p[2] - n["z"]) * _HAND_SHAPE_BLEND),
            }
            for p, n in zip(left_pts, neutral_left)
        ]
    else:
        left_hand = neutral_left

    body_list = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in body]
    return body_list, left_hand, right_hand

# Neutral flat hand (21 MediaPipe hand landmarks)
def _flat_hand(mirror: bool = False) -> list[dict]:
    s = -1 if mirror else 1
    pts = [
        {"x": s*0.00, "y": 0.00, "z": 0.00},  # 0 wrist
        {"x": s*0.04, "y": 0.05, "z": 0.00},  # 1 thumb CMC
        {"x": s*0.07, "y": 0.09, "z": 0.00},  # 2 thumb MCP
        {"x": s*0.10, "y": 0.12, "z": 0.00},  # 3 thumb IP
        {"x": s*0.12, "y": 0.15, "z": 0.00},  # 4 thumb tip
        {"x": s*0.03, "y": 0.12, "z": 0.00},  # 5 index MCP
        {"x": s*0.04, "y": 0.17, "z": 0.00},  # 6 index PIP
        {"x": s*0.04, "y": 0.21, "z": 0.00},  # 7 index DIP
        {"x": s*0.04, "y": 0.24, "z": 0.00},  # 8 index tip
        {"x": s*0.01, "y": 0.13, "z": 0.00},  # 9 middle MCP
        {"x": s*0.01, "y": 0.18, "z": 0.00},  # 10 middle PIP
        {"x": s*0.01, "y": 0.22, "z": 0.00},  # 11 middle DIP
        {"x": s*0.01, "y": 0.26, "z": 0.00},  # 12 middle tip
        {"x":-s*0.02, "y": 0.12, "z": 0.00},  # 13 ring MCP
        {"x":-s*0.02, "y": 0.17, "z": 0.00},  # 14 ring PIP
        {"x":-s*0.02, "y": 0.21, "z": 0.00},  # 15 ring DIP
        {"x":-s*0.02, "y": 0.24, "z": 0.00},  # 16 ring tip
        {"x":-s*0.05, "y": 0.10, "z": 0.00},  # 17 pinky MCP
        {"x":-s*0.06, "y": 0.14, "z": 0.00},  # 18 pinky PIP
        {"x":-s*0.07, "y": 0.17, "z": 0.00},  # 19 pinky DIP
        {"x":-s*0.08, "y": 0.19, "z": 0.00},  # 20 pinky tip
    ]
    return pts

def _fist_hand(mirror: bool = False) -> list[dict]:
    s = -1 if mirror else 1
    pts = [
        {"x": s*0.00, "y": 0.00, "z": 0.00},
        {"x": s*0.04, "y": 0.04, "z": 0.01},
        {"x": s*0.06, "y": 0.07, "z": 0.02},
        {"x": s*0.07, "y": 0.06, "z": 0.03},
        {"x": s*0.08, "y": 0.05, "z": 0.04},
        {"x": s*0.03, "y": 0.08, "z": 0.02},
        {"x": s*0.03, "y": 0.07, "z": 0.04},
        {"x": s*0.03, "y": 0.06, "z": 0.05},
        {"x": s*0.03, "y": 0.05, "z": 0.06},
        {"x": s*0.01, "y": 0.08, "z": 0.02},
        {"x": s*0.01, "y": 0.07, "z": 0.04},
        {"x": s*0.01, "y": 0.06, "z": 0.05},
        {"x": s*0.01, "y": 0.05, "z": 0.06},
        {"x":-s*0.02,"y": 0.08, "z": 0.02},
        {"x":-s*0.02,"y": 0.07, "z": 0.04},
        {"x":-s*0.02,"y": 0.06, "z": 0.05},
        {"x":-s*0.02,"y": 0.05, "z": 0.06},
        {"x":-s*0.05,"y": 0.06, "z": 0.02},
        {"x":-s*0.05,"y": 0.05, "z": 0.04},
        {"x":-s*0.06,"y": 0.05, "z": 0.05},
        {"x":-s*0.06,"y": 0.04, "z": 0.06},
    ]
    return pts

def _point_hand(mirror: bool = False) -> list[dict]:
    """Index finger pointing up, rest curled."""
    h = _fist_hand(mirror)
    s = -1 if mirror else 1
    h[5] = {"x": s*0.03, "y": 0.12, "z": 0.00}
    h[6] = {"x": s*0.04, "y": 0.17, "z": 0.00}
    h[7] = {"x": s*0.04, "y": 0.21, "z": 0.00}
    h[8] = {"x": s*0.04, "y": 0.24, "z": 0.00}
    return h


# ── Procedural ISL sign pose library ─────────────────────────────────────────
# Each entry: body landmark delta (only modified indices), left_hand, right_hand
# Delta is ADDED to NEUTRAL_BODY[idx]

def _body_at(idx: int, dx: float = 0, dy: float = 0, dz: float = 0) -> dict:
    return {"idx": idx, "dx": dx, "dy": dy, "dz": dz}


ISL_POSES: dict[str, dict] = {
    # ── Greetings ──────────────────────────────────────────────────────────
    "HELLO": {
        "body_deltas": [_body_at(15, dx=-0.05, dy=0.15), _body_at(16, dx=0.05, dy=0.15)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Both hands raised, open flat — ISL greeting",
    },
    "HI": {
        "body_deltas": [_body_at(16, dx=0.05, dy=0.20)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Right hand raised, wave",
    },
    "GOODBYE": {
        "body_deltas": [_body_at(16, dx=0.10, dy=0.25)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Right hand wave",
    },
    "NAMASTE": {
        "body_deltas": [
            _body_at(15, dx=0.05, dy=0.30, dz=0.1),
            _body_at(16, dx=-0.05, dy=0.30, dz=0.1),
        ],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Both palms together in front of chest",
    },

    # ── Pronouns ───────────────────────────────────────────────────────────
    "I": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.20, dz=0.15)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Point to self (chest)",
    },
    "ME": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.20, dz=0.15)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Point to self",
    },
    "YOU": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.20, dz=-0.15)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Point forward (at addressee)",
    },
    "YOUR": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.20, dz=-0.15)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Point forward",
    },
    "MY": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.20, dz=0.10)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Flat hand on chest",
    },
    "WE": {
        "body_deltas": [_body_at(15, dx=-0.10, dy=0.15), _body_at(16, dx=0.10, dy=0.15)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _point_hand(mirror=True),
        "description": "Both hands sweep across body",
    },
    "THEY": {
        "body_deltas": [_body_at(16, dx=0.20, dy=0.10, dz=-0.10)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Sweep to the side",
    },

    # ── Questions ──────────────────────────────────────────────────────────
    "WHAT": {
        "body_deltas": [_body_at(16, dx=0.05, dy=0.15), _body_at(15, dx=-0.05, dy=0.15)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Both open hands, palms up, questioning",
    },
    "WHO": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.40, dz=-0.05)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Point forward with raised right hand",
    },
    "WHERE": {
        "body_deltas": [_body_at(16, dx=0.10, dy=0.20), _body_at(15, dx=-0.10, dy=0.20)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _point_hand(mirror=True),
        "description": "Both hands pointing sideways",
    },
    "WHEN": {
        "body_deltas": [_body_at(16, dx=0.05, dy=0.30)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Circular motion with finger",
    },
    "WHY": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.45)],
        "right_hand": _fist_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Fist raised to temple",
    },
    "HOW": {
        "body_deltas": [_body_at(15, dx=-0.05, dy=0.15), _body_at(16, dx=0.05, dy=0.15)],
        "right_hand": _fist_hand(mirror=False),
        "left_hand": _fist_hand(mirror=True),
        "description": "Both fists turned upward",
    },
    "WHICH": {
        "body_deltas": [_body_at(16, dx=0.05, dy=0.20), _body_at(15, dx=-0.05, dy=0.20)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _point_hand(mirror=True),
        "description": "Alternate pointing",
    },

    # ── Common words ───────────────────────────────────────────────────────
    "NAME": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.40), _body_at(15, dx=0.0, dy=0.40)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _point_hand(mirror=True),
        "description": "Cross index fingers at forehead",
    },
    "GOOD": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.25, dz=0.10)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Right flat hand forward from chin",
    },
    "BAD": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.25, dz=0.05)],
        "right_hand": _fist_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Fist moved down from chin",
    },
    "YES": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.30)],
        "right_hand": _fist_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Nodding fist",
    },
    "NO": {
        "body_deltas": [_body_at(16, dx=0.10, dy=0.35)],
        "right_hand": _fist_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Shaking index finger",
    },
    "THANK": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.30, dz=-0.10)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Flat hand from chin outward",
    },
    "THANKS": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.30, dz=-0.10)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Flat hand from chin outward",
    },
    "PLEASE": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.20, dz=0.08)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Circle hand on chest",
    },
    "SORRY": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.20, dz=0.08)],
        "right_hand": _fist_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Fist circled on chest",
    },
    "HELP": {
        "body_deltas": [_body_at(15, dx=-0.05, dy=0.20), _body_at(16, dx=0.05, dy=0.25)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _fist_hand(mirror=True),
        "description": "Thumb of one hand lifted by other palm",
    },
    "WANT": {
        "body_deltas": [_body_at(15, dx=-0.10, dy=0.20), _body_at(16, dx=0.10, dy=0.20)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Claw hands pulled toward body",
    },
    "NEED": {
        "body_deltas": [_body_at(16, dx=0.05, dy=0.20)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Bent index finger pulled down",
    },
    "LIKE": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.30, dz=0.05)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Thumb and middle finger pinch from chest outward",
    },
    "LOVE": {
        "body_deltas": [_body_at(15, dx=-0.05, dy=0.20, dz=0.08), _body_at(16, dx=0.05, dy=0.20, dz=0.08)],
        "right_hand": _fist_hand(mirror=False),
        "left_hand": _fist_hand(mirror=True),
        "description": "Arms crossed over chest",
    },
    "KNOW": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.45)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Flat hand taps forehead",
    },
    "UNDERSTAND": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.45)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Index finger flicks up at temple",
    },
    "THINK": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.44)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Index finger circles at temple",
    },
    "SEE": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.42, dz=-0.08)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "V-fingers from eyes outward",
    },
    "HEAR": {
        "body_deltas": [_body_at(16, dx=0.12, dy=0.44)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Point to ear",
    },
    "EAT": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.37)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Flat O-hand to mouth",
    },
    "DRINK": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.38)],
        "right_hand": _fist_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "C-hand tilted to mouth",
    },
    "WATER": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.36)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "W-hand taps chin",
    },
    "FOOD": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.38)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Flat O to mouth repeatedly",
    },
    "GO": {
        "body_deltas": [_body_at(15, dx=-0.10, dy=0.20, dz=-0.15), _body_at(16, dx=0.10, dy=0.20, dz=-0.15)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _point_hand(mirror=True),
        "description": "Both pointing fingers arc forward",
    },
    "COME": {
        "body_deltas": [_body_at(15, dx=-0.05, dy=0.20, dz=0.15), _body_at(16, dx=0.05, dy=0.20, dz=0.15)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _point_hand(mirror=True),
        "description": "Both fingers curled toward body",
    },
    "STOP": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.25), _body_at(15, dx=0.0, dy=0.25)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Right flat hand slaps left palm",
    },
    "SIT": {
        "body_deltas": [_body_at(15, dx=-0.05, dy=0.15), _body_at(16, dx=0.05, dy=0.15)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _point_hand(mirror=True),
        "description": "Two V fingers bent over other two V fingers",
    },
    "STAND": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.10), _body_at(15, dx=0.0, dy=0.10)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "V fingers standing on flat palm",
    },
    "HOME": {
        "body_deltas": [_body_at(16, dx=0.05, dy=0.40)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Flat O to cheek, then flat B",
    },
    "SCHOOL": {
        "body_deltas": [_body_at(15, dx=-0.05, dy=0.20), _body_at(16, dx=0.05, dy=0.20)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Right palm claps left twice",
    },
    "WORK": {
        "body_deltas": [_body_at(15, dx=-0.05, dy=0.20), _body_at(16, dx=0.05, dy=0.25)],
        "right_hand": _fist_hand(mirror=False),
        "left_hand": _fist_hand(mirror=True),
        "description": "Both fists tap wrist to wrist",
    },
    "HAPPY": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.22, dz=0.05)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Flat hand circles chest upward",
    },
    "SAD": {
        "body_deltas": [_body_at(15, dx=-0.05, dy=0.40), _body_at(16, dx=0.05, dy=0.40)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Open hands slide down face",
    },
    "PAIN": {
        "body_deltas": [_body_at(15, dx=-0.05, dy=0.20), _body_at(16, dx=0.05, dy=0.20)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _point_hand(mirror=True),
        "description": "Both index fingers twist toward each other",
    },
    "DOCTOR": {
        "body_deltas": [_body_at(16, dx=-0.10, dy=0.20)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Tap wrist with middle fingers (checking pulse)",
    },
    "HOSPITAL": {
        "body_deltas": [_body_at(16, dx=-0.12, dy=0.20)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Cross sign on upper arm",
    },
    "TIME": {
        "body_deltas": [_body_at(16, dx=-0.12, dy=0.25)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Point to wrist (watch)",
    },
    "DAY": {
        "body_deltas": [_body_at(16, dx=0.10, dy=0.30), _body_at(15, dx=-0.10, dy=0.20)],
        "right_hand": _point_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Dominant arm arcs like sun across sky",
    },
    "NIGHT": {
        "body_deltas": [_body_at(16, dx=0.05, dy=0.25, dz=0.05)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Curved hand arcs over other arm (moon over horizon)",
    },
    "MORNING": {
        "body_deltas": [_body_at(16, dx=0.0, dy=0.35)],
        "right_hand": _flat_hand(mirror=False),
        "left_hand": _flat_hand(mirror=True),
        "description": "Arm rises from horizontal to vertical (sunrise)",
    },
}


# ── Fallback: generate a plausible pose from the word shape ──────────────────

def _generate_fallback_pose(word: str) -> dict:
    """
    Generate a plausible pose for unknown words using letter/shape heuristics.
    The pose varies deterministically based on the word so the same word always
    produces the same pose.
    """
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(word))
    rng = random.Random(seed)

    arm_height = rng.uniform(0.15, 0.40)
    arm_spread = rng.uniform(-0.05, 0.15)

    body_deltas = [
        _body_at(16, dx=arm_spread, dy=arm_height),
        _body_at(15, dx=-arm_spread * 0.5, dy=arm_height * 0.7),
    ]

    shapes = [_flat_hand, _fist_hand, _point_hand]
    right_shape = rng.choice(shapes)(mirror=False)
    left_shape = rng.choice(shapes)(mirror=True)

    return {
        "body_deltas": body_deltas,
        "right_hand": right_shape,
        "left_hand": left_shape,
        "description": f"Procedural pose for '{word}'",
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def get_pose_for_gloss(gloss: str) -> dict:
    """
    Return the ISL pose data for a gloss token.
    Falls back to procedural generation for unknown words.
    """
    key = gloss.upper().strip()
    if key in ISL_POSES:
        return ISL_POSES[key]
    # Try the sign-language-translator library
    try:
        pose = _try_slt_lookup(key)
        if pose:
            return pose
    except Exception as e:
        logger.debug(f"SLT lookup failed for '{key}': {e}")
    # Procedural fallback
    logger.debug(f"Using procedural fallback pose for '{key}'")
    return _generate_fallback_pose(key)


def _try_slt_lookup(gloss: str) -> Optional[dict]:
    """
    Attempt to look up sign via the sign-language-translator library.
    Returns None if unavailable.
    """
    model = _get_slt_model()
    if not model:
        return None

    try:
        signs = model.translate(gloss.lower())
        if signs and hasattr(signs, "data"):
            frames = signs.data
            if len(frames) > 0:
                frame = frames[len(frames) // 2]  # use midpoint frame as pose
                return _slt_frame_to_pose(frame)
    except Exception as e:
        msg = str(e)
        if "embedding model" in msg.lower():
            _disable_slt(msg)
        logger.debug(f"SLT lookup failed for '{gloss}': {e}")
    return None


def _slt_frame_to_pose(frame) -> dict:
    """Convert SLT landmark frame to our pose dict."""
    points = _slt_frame_to_points(frame)
    if len(points) < 75:
        return None

    body = [{"x": p[0], "y": p[1], "z": p[2]} for p in points[:33]]
    left = [{"x": p[0], "y": p[1], "z": p[2]} for p in points[33:54]]
    right = [{"x": p[0], "y": p[1], "z": p[2]} for p in points[54:75]]
    return {
        "body_deltas": [],
        "right_hand": right,
        "left_hand": left,
        "body_absolute": body,
        "description": "SLT library data",
    }


def _slt_frame_to_points(frame) -> list[tuple[float, float, float]]:
    if frame is None:
        return []

    if hasattr(frame, "tolist"):
        frame = frame.tolist()

    if isinstance(frame, dict) and "landmarks" in frame:
        frame = frame.get("landmarks", [])

    if not isinstance(frame, (list, tuple)):
        return []

    if len(frame) == 0:
        return []

    if isinstance(frame[0], (int, float)):
        if len(frame) % 3 != 0:
            return []
        frame = [frame[i:i + 3] for i in range(0, len(frame), 3)]

    points = []
    for p in frame:
        if not isinstance(p, (list, tuple)) or len(p) < 3:
            continue
        points.append((float(p[0]), float(p[1]), float(p[2])))

    return points


def try_generate_slt_animation(text: str, gloss_sequence: Optional[list[str]] = None, fps: int = 30) -> Optional[dict]:
    try:
        model = _get_slt_model()
        if not model:
            return None

        if not text.strip():
            return None

        signs = model.translate(text)
    except Exception as e:
        msg = str(e)
        if "embedding model" in msg.lower():
            _disable_slt(msg)
        logger.warning(f"SLT translation failed: {e}")
        return None

    frames = getattr(signs, "data", None)
    if frames is None:
        return None

    if hasattr(frames, "tolist"):
        frames = frames.tolist()

    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        return None

    fps_from_model = getattr(signs, "fps", None) or getattr(signs, "frame_rate", None)
    if isinstance(fps_from_model, (int, float)) and fps_from_model > 0:
        fps = int(round(fps_from_model))

    pose_frames = []
    ms_per_frame = 1000.0 / fps
    for idx, frame in enumerate(frames):
        points = _slt_frame_to_points(frame)
        if len(points) < 75:
            continue

        body = [{"x": p[0], "y": p[1], "z": p[2]} for p in points[:33]]
        left = [{"x": p[0], "y": p[1], "z": p[2]} for p in points[33:54]]
        right = [{"x": p[0], "y": p[1], "z": p[2]} for p in points[54:75]]
        pose_frames.append({
            "frame_index": idx,
            "timestamp_ms": round(idx * ms_per_frame, 2),
            "body": body,
            "left_hand": left,
            "right_hand": right,
        })

    if not pose_frames:
        return None

    total_frames = len(pose_frames)
    duration_ms = total_frames * ms_per_frame

    gloss_timeline = []
    if gloss_sequence:
        step = max(1, total_frames // max(1, len(gloss_sequence)))
        for i, gloss in enumerate(gloss_sequence):
            start = i * step
            end = min(total_frames - 1, (i + 1) * step - 1)
            gloss_timeline.append({"gloss": gloss, "start_frame": start, "end_frame": end})

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_ms": duration_ms,
        "frames": pose_frames,
        "gloss_timeline": gloss_timeline,
    }


def try_generate_motion_clip_animation(
    gloss_sequence: list[str],
    fps: int = 30,
    interp_frames: int = 8,
    source_text: Optional[str] = None,
) -> Optional[dict]:
    index = _get_motion_index()
    if not index or not index.get("clips_map"):
        return None

    resolved: list[str] = []
    if source_text:
        phrase = _resolve_motion_sign(source_text, index)
        if phrase:
            resolved = [phrase]

    if not resolved:
        for gloss in gloss_sequence:
            sign_name = _resolve_motion_sign(gloss, index)
            if sign_name:
                resolved.append(sign_name)

    if not resolved:
        return None

    ms_per_frame = 1000.0 / fps
    frames: list[dict] = []
    gloss_timeline: list[dict] = []
    frame_idx = 0

    def _interp_pose(a: dict, b: dict, steps: int) -> list[dict]:
        out = []
        for i in range(1, steps + 1):
            t = i / (steps + 1)

            def _lerp_list(pa: list[dict], pb: list[dict]) -> list[dict]:
                return [
                    {
                        "x": pa[j]["x"] + (pb[j]["x"] - pa[j]["x"]) * t,
                        "y": pa[j]["y"] + (pb[j]["y"] - pa[j]["y"]) * t,
                        "z": pa[j]["z"] + (pb[j]["z"] - pa[j]["z"]) * t,
                    }
                    for j in range(min(len(pa), len(pb)))
                ]

            out.append(
                {
                    "body": _lerp_list(a["body"], b["body"]),
                    "left_hand": _lerp_list(a["left_hand"], b["left_hand"]),
                    "right_hand": _lerp_list(a["right_hand"], b["right_hand"]),
                }
            )
        return out

    last_pose = None

    for sign_name in resolved:
        entries = index["clips_map"].get(sign_name, [])
        if not entries:
            continue

        clip = _load_motion_clip(entries[0])
        if clip is None:
            continue

        body = clip["body"]
        left = clip["left_hand"]
        right = clip["right_hand"]
        left_present = clip["left_present"]
        right_present = clip["right_present"]

        clip_fps = int(clip["fps"][0]) if np.size(clip["fps"]) > 0 else int(index["default_fps"])
        if clip_fps != fps:
            target_len = max(2, int(round(len(body) * fps / max(1, clip_fps))))
            body = _resample_landmark_sequence(body, target_len)
            left = _resample_landmark_sequence(left, target_len)
            right = _resample_landmark_sequence(right, target_len)
            left_present = _resample_mask(left_present, target_len)
            right_present = _resample_mask(right_present, target_len)

        first_body, first_left, first_right = _motion_frame_to_pose(
            body_points=body[0],
            left_points=left[0],
            right_points=right[0],
            left_present=float(left_present[0]),
            right_present=float(right_present[0]),
        )
        first_pose = {
            "body": first_body,
            "left_hand": first_left,
            "right_hand": first_right,
        }

        if last_pose and interp_frames > 0:
            for interp in _interp_pose(last_pose, first_pose, interp_frames):
                frames.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp_ms": round(frame_idx * ms_per_frame, 2),
                        **interp,
                    }
                )
                frame_idx += 1

        sign_start = frame_idx
        for i in range(len(body)):
            body_lm, left_lm, right_lm = _motion_frame_to_pose(
                body_points=body[i],
                left_points=left[i],
                right_points=right[i],
                left_present=float(left_present[i]),
                right_present=float(right_present[i]),
            )
            pose_frame = {
                "frame_index": frame_idx,
                "timestamp_ms": round(frame_idx * ms_per_frame, 2),
                "body": body_lm,
                "left_hand": left_lm,
                "right_hand": right_lm,
            }
            frames.append(pose_frame)
            frame_idx += 1
            last_pose = pose_frame

        gloss_timeline.append(
            {
                "gloss": sign_name,
                "start_frame": sign_start,
                "end_frame": frame_idx - 1,
            }
        )

    if not frames:
        return None

    total_frames = len(frames)
    duration_ms = total_frames * ms_per_frame
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_ms": duration_ms,
        "frames": frames,
        "gloss_timeline": gloss_timeline,
    }


def try_generate_isl_dataset_animation(
    gloss_sequence: list[str],
    fps: int = 30,
    interp_frames: int = 8,
    source_text: Optional[str] = None,
) -> Optional[dict]:
    index = _get_isl_index()
    if not index:
        return None

    resolved: list[str] = []
    if source_text:
        phrase = _resolve_isl_class(source_text, index)
        if phrase:
            resolved = [phrase]

    if not resolved:
        for gloss in gloss_sequence:
            name = _resolve_isl_class(gloss, index)
            if name:
                resolved.append(name)

    if not resolved:
        return None

    ms_per_frame = 1000.0 / fps
    base_fps = max(1, int(settings.isl_dataset_fps))
    frames = []
    gloss_timeline = []
    frame_idx = 0

    def _interp_pose(a: dict, b: dict, steps: int) -> list[dict]:
        out = []
        for i in range(1, steps + 1):
            t = i / (steps + 1)

            def _lerp_list(pa: list[dict], pb: list[dict]):
                return [
                    {
                        "x": pa[j]["x"] + (pb[j]["x"] - pa[j]["x"]) * t,
                        "y": pa[j]["y"] + (pb[j]["y"] - pa[j]["y"]) * t,
                        "z": pa[j]["z"] + (pb[j]["z"] - pa[j]["z"] ) * t,
                    }
                    for j in range(min(len(pa), len(pb)))
                ]

            out.append({
                "body": _lerp_list(a["body"], b["body"]),
                "left_hand": _lerp_list(a["left_hand"], b["left_hand"]),
                "right_hand": _lerp_list(a["right_hand"], b["right_hand"]),
            })
        return out

    last_pose = None

    for name in resolved:
        seq = _load_isl_sequence(name, index)
        if seq is None:
            continue

        if base_fps != fps:
            target_len = max(2, int(round(len(seq) * fps / base_fps)))
            seq = _resample_sequence(seq, target_len)

        pose_seq = seq[:, 126:156].reshape(-1, 10, 3)
        scale = _compute_scale(pose_seq)

        first_body, first_left, first_right = _features_to_landmarks(seq[0], scale)
        first_pose = {
            "body": first_body,
            "left_hand": first_left,
            "right_hand": first_right,
        }

        if last_pose and interp_frames > 0:
            for interp in _interp_pose(last_pose, first_pose, interp_frames):
                frames.append({
                    "frame_index": frame_idx,
                    "timestamp_ms": round(frame_idx * ms_per_frame, 2),
                    **interp,
                })
                frame_idx += 1

        sign_start = frame_idx

        for i, feat in enumerate(seq):
            if i == 0:
                body, left_hand, right_hand = first_body, first_left, first_right
            else:
                body, left_hand, right_hand = _features_to_landmarks(feat, scale)
            pose_frame = {
                "frame_index": frame_idx,
                "timestamp_ms": round(frame_idx * ms_per_frame, 2),
                "body": body,
                "left_hand": left_hand,
                "right_hand": right_hand,
            }
            frames.append(pose_frame)
            frame_idx += 1
            last_pose = pose_frame

        sign_end = frame_idx - 1
        gloss_timeline.append({"gloss": name, "start_frame": sign_start, "end_frame": sign_end})

    if not frames:
        return None

    total_frames = len(frames)
    duration_ms = total_frames * ms_per_frame

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_ms": duration_ms,
        "frames": frames,
        "gloss_timeline": gloss_timeline,
    }


def get_available_signs() -> list[str]:
    """Return list of all known ISL glosses."""
    return sorted(ISL_POSES.keys())
