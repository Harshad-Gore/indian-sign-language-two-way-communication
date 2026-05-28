"""
Animation Engine — converts gloss sequences into smooth pose animation frames.

Pipeline:
  1. Look up each gloss pose from sign_generator
  2. Apply body deltas to neutral skeleton
  3. Generate N hold frames per sign
  4. Insert M interpolation frames (cubic Hermite spline) between signs
  5. Add idle breathing oscillation
  6. Return a complete AnimationData payload
"""

from __future__ import annotations
import copy
import math
from typing import Optional

from services.sign_generator import (
    get_pose_for_gloss,
    try_generate_slt_animation,
    try_generate_motion_clip_animation,
    try_generate_isl_dataset_animation,
    NEUTRAL_BODY,
    _flat_hand,
)
from loguru import logger


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_hand(mirror: bool = False) -> list[dict]:
    return _flat_hand(mirror)


def _apply_deltas(body: list[dict], deltas: list[dict]) -> list[dict]:
    result = [dict(p) for p in body]
    for d in deltas:
        idx = d["idx"]
        result[idx]["x"] += d.get("dx", 0)
        result[idx]["y"] += d.get("dy", 0)
        result[idx]["z"] += d.get("dz", 0)
    return result


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(t: float) -> float:
    """Smoothstep easing — feels much nicer than linear."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _cubic_hermite(p0: float, p1: float, t: float, tension: float = 0.5) -> float:
    """Cubic Hermite interpolation between p0 and p1."""
    t2 = t * t
    t3 = t2 * t
    h1 = 2 * t3 - 3 * t2 + 1
    h2 = -2 * t3 + 3 * t2
    m0 = tension * (p1 - p0)
    m1 = tension * (p1 - p0)
    return h1 * p0 + h2 * p1 + (t3 - 2 * t2 + t) * m0 + (t3 - t2) * m1


def _interp_landmarks(
    lm_a: list[dict],
    lm_b: list[dict],
    t: float,
    ease: bool = True,
) -> list[dict]:
    """Interpolate between two landmark arrays."""
    et = _smoothstep(t) if ease else t
    result = []
    for a, b in zip(lm_a, lm_b):
        result.append({
            "x": _lerp(a["x"], b["x"], et),
            "y": _lerp(a["y"], b["y"], et),
            "z": _lerp(a["z"], b["z"], et),
            "visibility": _lerp(a.get("visibility", 1.0), b.get("visibility", 1.0), et),
        })
    return result


def _add_breathing(body: list[dict], frame_idx: int, fps: int = 30) -> list[dict]:
    """Add subtle idle breathing motion to the shoulders and torso."""
    t = frame_idx / fps
    breath = math.sin(t * 2 * math.pi * 0.25) * 0.008   # 0.25 Hz breathing
    result = [dict(p) for p in body]
    # Shoulders rise and fall
    for idx in [11, 12]:
        result[idx]["y"] += breath
    # Chest expands slightly
    for idx in [11, 12]:
        result[idx]["z"] += abs(breath) * 0.5
    return result


def _to_landmark_points(lm_list: list[dict]) -> list[dict]:
    return [
        {
            "x": float(p.get("x", 0)),
            "y": float(p.get("y", 0)),
            "z": float(p.get("z", 0)),
            "visibility": float(p.get("visibility", 1.0)),
        }
        for p in lm_list
    ]


# ── Main animation builder ────────────────────────────────────────────────────

def generate_animation(
    gloss_sequence: list[str],
    fps: int = 30,
    hold_frames: int = 20,
    interp_frames: int = 10,
    idle_animation: bool = True,
    source_text: Optional[str] = None,
) -> dict:
    """
    Generate a complete animation from a gloss sequence.

    Returns a dict matching the AnimationData schema.
    """
    if source_text:
        try:
            slt_anim = try_generate_slt_animation(source_text, gloss_sequence=gloss_sequence, fps=fps)
            if slt_anim:
                logger.info(f"Using SLT animation: {slt_anim['total_frames']} frames @ {slt_anim['fps']}fps")
                return slt_anim
        except Exception as e:
            logger.warning(f"SLT animation error: {e}")

    motion_anim = try_generate_motion_clip_animation(
        gloss_sequence=gloss_sequence,
        fps=fps,
        interp_frames=interp_frames,
        source_text=source_text,
    )
    if motion_anim:
        logger.info(f"Using motion-clip animation: {motion_anim['total_frames']} frames @ {motion_anim['fps']}fps")
        return motion_anim

    dataset_anim = try_generate_isl_dataset_animation(
        gloss_sequence=gloss_sequence,
        fps=fps,
        interp_frames=interp_frames,
        source_text=source_text,
    )
    if dataset_anim:
        logger.info(f"Using ISL dataset animation: {dataset_anim['total_frames']} frames @ {dataset_anim['fps']}fps")
        return dataset_anim

    if not gloss_sequence:
        return _idle_animation(fps=fps, duration_s=2.0)

    logger.info(f"Generating animation for: {gloss_sequence}")

    # ── Build pose list ────────────────────────────────────────────────────
    poses = []
    for gloss in gloss_sequence:
        pose_data = get_pose_for_gloss(gloss)
        body = pose_data.get("body_absolute") or _apply_deltas(NEUTRAL_BODY, pose_data.get("body_deltas", []))
        right_hand = pose_data.get("right_hand", _default_hand(mirror=False))
        left_hand = pose_data.get("left_hand", _default_hand(mirror=True))
        poses.append({
            "gloss": gloss,
            "body": body,
            "right_hand": right_hand,
            "left_hand": left_hand,
        })

    # ── Neutral pose for transitions ──────────────────────────────────────
    neutral = {
        "gloss": "__neutral__",
        "body": list(NEUTRAL_BODY),
        "right_hand": _default_hand(mirror=False),
        "left_hand": _default_hand(mirror=True),
    }

    # ── Build frame array ─────────────────────────────────────────────────
    frames = []
    gloss_timeline = []
    frame_idx = 0
    ms_per_frame = 1000.0 / fps

    prev_pose = neutral

    for pose in poses:
        sign_start = frame_idx

        # Interpolation from previous → current sign
        for i in range(interp_frames):
            t = i / interp_frames
            body = _interp_landmarks(prev_pose["body"], pose["body"], t)
            rh = _interp_landmarks(prev_pose["right_hand"], pose["right_hand"], t)
            lh = _interp_landmarks(prev_pose["left_hand"], pose["left_hand"], t)

            if idle_animation:
                body = _add_breathing(body, frame_idx, fps)

            frames.append({
                "frame_index": frame_idx,
                "timestamp_ms": round(frame_idx * ms_per_frame, 2),
                "body": _to_landmark_points(body),
                "right_hand": _to_landmark_points(rh),
                "left_hand": _to_landmark_points(lh),
            })
            frame_idx += 1

        # Hold frames
        for i in range(hold_frames):
            body = list(pose["body"])
            if idle_animation:
                body = _add_breathing(body, frame_idx, fps)
            frames.append({
                "frame_index": frame_idx,
                "timestamp_ms": round(frame_idx * ms_per_frame, 2),
                "body": _to_landmark_points(body),
                "right_hand": _to_landmark_points(pose["right_hand"]),
                "left_hand": _to_landmark_points(pose["left_hand"]),
            })
            frame_idx += 1

        sign_end = frame_idx - 1
        gloss_timeline.append({
            "gloss": pose["gloss"],
            "start_frame": sign_start,
            "end_frame": sign_end,
        })

        prev_pose = pose

    # ── Return to neutral ──────────────────────────────────────────────────
    for i in range(interp_frames):
        t = i / interp_frames
        body = _interp_landmarks(prev_pose["body"], neutral["body"], t)
        rh = _interp_landmarks(prev_pose["right_hand"], neutral["right_hand"], t)
        lh = _interp_landmarks(prev_pose["left_hand"], neutral["left_hand"], t)
        if idle_animation:
            body = _add_breathing(body, frame_idx, fps)
        frames.append({
            "frame_index": frame_idx,
            "timestamp_ms": round(frame_idx * ms_per_frame, 2),
            "body": _to_landmark_points(body),
            "right_hand": _to_landmark_points(rh),
            "left_hand": _to_landmark_points(lh),
        })
        frame_idx += 1

    total_frames = len(frames)
    duration_ms = total_frames * ms_per_frame

    logger.info(f"Animation: {total_frames} frames, {duration_ms:.0f}ms, {len(gloss_sequence)} signs")

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_ms": duration_ms,
        "frames": frames,
        "gloss_timeline": gloss_timeline,
    }


def _idle_animation(fps: int = 30, duration_s: float = 3.0) -> dict:
    """Return a breathing idle animation when no gloss is given."""
    ms_per_frame = 1000.0 / fps
    total_frames = int(fps * duration_s)
    frames = []
    for i in range(total_frames):
        body = _add_breathing(list(NEUTRAL_BODY), i, fps)
        frames.append({
            "frame_index": i,
            "timestamp_ms": round(i * ms_per_frame, 2),
            "body": _to_landmark_points(body),
            "right_hand": _to_landmark_points(_default_hand(mirror=False)),
            "left_hand": _to_landmark_points(_default_hand(mirror=True)),
        })
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_ms": total_frames * ms_per_frame,
        "frames": frames,
        "gloss_timeline": [],
    }
