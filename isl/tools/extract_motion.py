"""
Extract per-sign motion clips from ISL videos with MediaPipe Holistic.

Output format (per clip, .npz):
  - body: (T, 33, 3) float32
  - left_hand: (T, 21, 3) float32
  - right_hand: (T, 21, 3) float32
  - left_present: (T,) float32
  - right_present: (T,) float32
  - fps: (1,) int32

All landmarks are normalized per frame:
  - translation: center at mid-shoulder
  - scale: divide by shoulder width
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import cv2
import mediapipe as mp
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class ClipResult:
    body: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    left_present: np.ndarray
    right_present: np.ndarray
    fps: int
    score: float
    frames: int
    motion_score: float


def _resolve_holistic_cls():
    """
    Resolve MediaPipe Holistic class across API variants.

    Some environments expose Holistic at `mp.solutions`, while others expose it
    through `mediapipe.python.solutions`.
    """
    try:
        return mp.solutions.holistic.Holistic
    except Exception:
        pass

    try:
        from mediapipe import solutions as mp_solutions  # type: ignore
        return mp_solutions.holistic.Holistic
    except Exception:
        pass

    try:
        from mediapipe.python.solutions.holistic import Holistic  # type: ignore
        return Holistic
    except Exception as e:
        raise RuntimeError(
            "MediaPipe Holistic API not found. Install a full mediapipe build "
            "(example: pip install mediapipe==0.10.14)."
        ) from e


def parse_sign_name(folder_name: str) -> str:
    if ". " in folder_name:
        return folder_name.split(". ", 1)[1].strip()
    return folder_name.strip()


def safe_sign_name(name: str) -> str:
    # Keep stable filesystem names and collapse whitespace.
    cleaned = re.sub(r"\s+", "_", name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_()\-]", "", cleaned)
    return cleaned or "unknown_sign"


def iter_sign_videos(data_dir: Path) -> Iterator[tuple[str, Path, list[Path]]]:
    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for sign_dir in sorted(category_dir.iterdir()):
            if not sign_dir.is_dir():
                continue
            videos = [p for p in sorted(sign_dir.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
            if not videos:
                continue
            yield parse_sign_name(sign_dir.name), sign_dir, videos


def _normalize_frame(
    body: Optional[np.ndarray],
    left_hand: Optional[np.ndarray],
    right_hand: Optional[np.ndarray],
    last_center: Optional[np.ndarray],
    last_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if body is not None and body.shape[0] >= 33:
        left_shoulder = body[11, :3]
        right_shoulder = body[12, :3]
        center = (left_shoulder + right_shoulder) / 2.0
        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        scale = float(shoulder_width) if shoulder_width > 1e-4 else last_scale
    else:
        center = last_center if last_center is not None else np.array([0.5, 0.5, 0.0], dtype=np.float32)
        scale = last_scale if last_scale > 1e-4 else 1.0

    def _norm(arr: Optional[np.ndarray], expected_len: int) -> np.ndarray:
        if arr is None or arr.shape[0] < expected_len:
            return np.zeros((expected_len, 3), dtype=np.float32)
        return ((arr[:expected_len, :3] - center) / scale).astype(np.float32)

    body_n = _norm(body, 33)
    left_n = _norm(left_hand, 21)
    right_n = _norm(right_hand, 21)
    return body_n, left_n, right_n, center, scale


def _motion_profile(
    body: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_present: np.ndarray,
    right_present: np.ndarray,
) -> np.ndarray:
    if len(body) <= 1:
        return np.zeros(len(body), dtype=np.float32)

    profile = np.zeros(len(body), dtype=np.float32)
    for i in range(1, len(body)):
        # Track arm and hand endpoints to estimate sign activity.
        prev = [body[i - 1, 15], body[i - 1, 16]]
        curr = [body[i, 15], body[i, 16]]

        if left_present[i - 1] > 0 and left_present[i] > 0:
            prev.append(left[i - 1, 8])   # left index tip
            curr.append(left[i, 8])
        if right_present[i - 1] > 0 and right_present[i] > 0:
            prev.append(right[i - 1, 8])  # right index tip
            curr.append(right[i, 8])

        prev_arr = np.asarray(prev, dtype=np.float32)
        curr_arr = np.asarray(curr, dtype=np.float32)
        profile[i] = float(np.mean(np.linalg.norm(curr_arr - prev_arr, axis=1)))

    if len(profile) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        profile = np.convolve(profile, kernel, mode="same")
    return profile


def _trim_static(
    body: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_present: np.ndarray,
    right_present: np.ndarray,
    pad_frames: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if len(body) < 10:
        return body, left, right, left_present, right_present, 0.0

    profile = _motion_profile(body, left, right, left_present, right_present)
    max_motion = float(np.max(profile))
    if max_motion <= 1e-6:
        return body, left, right, left_present, right_present, 0.0

    threshold = max(0.0035, max_motion * 0.10)
    active = np.where(profile >= threshold)[0]
    if active.size == 0:
        return body, left, right, left_present, right_present, float(np.mean(profile))

    start = max(0, int(active[0]) - pad_frames)
    end = min(len(body), int(active[-1]) + pad_frames + 1)

    return (
        body[start:end],
        left[start:end],
        right[start:end],
        left_present[start:end],
        right_present[start:end],
        float(np.mean(profile[start:end])),
    )


def extract_clip(video_path: Path, target_fps: int, max_frames: int, trim_static: bool) -> Optional[ClipResult]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_interval = max(1, int(round(source_fps / max(1, target_fps))))

    holistic_cls = _resolve_holistic_cls()
    holistic = holistic_cls(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    body_seq: list[np.ndarray] = []
    left_seq: list[np.ndarray] = []
    right_seq: list[np.ndarray] = []
    left_mask: list[float] = []
    right_mask: list[float] = []

    frame_idx = 0
    last_center = None
    last_scale = 1.0

    while len(body_seq) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        body = None
        if results.pose_landmarks:
            body = np.array([[p.x, p.y, p.z] for p in results.pose_landmarks.landmark], dtype=np.float32)

        left_hand = None
        if results.left_hand_landmarks:
            left_hand = np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark], dtype=np.float32)

        right_hand = None
        if results.right_hand_landmarks:
            right_hand = np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark], dtype=np.float32)

        body_n, left_n, right_n, last_center, last_scale = _normalize_frame(
            body=body,
            left_hand=left_hand,
            right_hand=right_hand,
            last_center=last_center,
            last_scale=last_scale,
        )

        body_seq.append(body_n)
        left_seq.append(left_n)
        right_seq.append(right_n)
        left_mask.append(1.0 if left_hand is not None else 0.0)
        right_mask.append(1.0 if right_hand is not None else 0.0)

    cap.release()
    holistic.close()

    if not body_seq:
        return None

    body = np.stack(body_seq).astype(np.float32)
    left = np.stack(left_seq).astype(np.float32)
    right = np.stack(right_seq).astype(np.float32)
    left_present = np.asarray(left_mask, dtype=np.float32)
    right_present = np.asarray(right_mask, dtype=np.float32)

    motion_score = 0.0
    if trim_static:
        body, left, right, left_present, right_present, motion_score = _trim_static(
            body, left, right, left_present, right_present
        )
    else:
        motion_score = float(np.mean(_motion_profile(body, left, right, left_present, right_present)))

    if len(body) < 2:
        return None

    pose_valid = float(np.mean(np.linalg.norm(body[:, 11, :2] - body[:, 12, :2], axis=1) > 1e-5))
    left_ratio = float(np.mean(left_present))
    right_ratio = float(np.mean(right_present))
    hand_coverage = 0.5 * (left_ratio + right_ratio)
    score = float(0.55 * hand_coverage + 0.25 * pose_valid + 0.20 * min(1.0, motion_score / 0.06))

    return ClipResult(
        body=body,
        left_hand=left,
        right_hand=right,
        left_present=left_present,
        right_present=right_present,
        fps=int(target_fps),
        score=score,
        frames=int(body.shape[0]),
        motion_score=motion_score,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ISL motion clips from sign videos")
    parser.add_argument("--data-dir", default="data", help="Root directory of sign videos")
    parser.add_argument(
        "--output-dir",
        default="motion_clips",
        help="Output directory for motion clips",
    )
    parser.add_argument("--fps", type=int, default=24, help="Target extraction FPS")
    parser.add_argument("--max-frames", type=int, default=220, help="Max frames per clip")
    parser.add_argument(
        "--trim-static",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim low-motion lead-in and tail frames",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_root = Path(args.output_dir)
    landmarks_dir = output_root / "landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fps": int(args.fps),
        "landmark_format": {
            "body": [33, 3],
            "left_hand": [21, 3],
            "right_hand": [21, 3],
        },
        "clips": {},
    }

    total_saved = 0
    for sign_name, _, videos in iter_sign_videos(data_dir):
        safe_name = safe_sign_name(sign_name)
        sign_out = landmarks_dir / safe_name
        sign_out.mkdir(parents=True, exist_ok=True)

        entries: list[dict] = []
        for idx, video in enumerate(videos):
            clip = extract_clip(
                video_path=video,
                target_fps=int(args.fps),
                max_frames=int(args.max_frames),
                trim_static=bool(args.trim_static),
            )
            if clip is None:
                continue

            clip_id = f"{safe_name}__{idx:03d}"
            clip_path = sign_out / f"{clip_id}.npz"
            np.savez_compressed(
                clip_path,
                body=clip.body,
                left_hand=clip.left_hand,
                right_hand=clip.right_hand,
                left_present=clip.left_present,
                right_present=clip.right_present,
                fps=np.array([clip.fps], dtype=np.int32),
            )
            total_saved += 1

            entries.append(
                {
                    "id": clip_id,
                    "file": str(clip_path.relative_to(output_root).as_posix()),
                    "frames": clip.frames,
                    "fps": clip.fps,
                    "score": round(clip.score, 5),
                    "motion_score": round(float(clip.motion_score), 6),
                    "left_coverage": round(float(np.mean(clip.left_present)), 5),
                    "right_coverage": round(float(np.mean(clip.right_present)), 5),
                    "source": str(video),
                }
            )

        if entries:
            entries.sort(key=lambda x: (-x["score"], -x["frames"]))
            metadata["clips"][sign_name] = entries

    meta_path = output_root / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved {total_saved} clip(s) to: {output_root}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
