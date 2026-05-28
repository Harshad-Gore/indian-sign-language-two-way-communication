"""
Landmark extraction from ISL video dataset.

Processes all videos through MediaPipe Hands + Pose and saves
per-frame landmark sequences as compressed NumPy arrays.

Each video produces a feature array of shape (num_frames, FEATURE_DIM):
  - Right hand: 21 landmarks × 3 coords = 63
  - Left hand:  21 landmarks × 3 coords = 63
  - Pose (upper body key points): 10 landmarks × 3 coords = 30
  - Hand presence flags: 2
  Total: 158 features per frame

Landmarks are normalized relative to the body center (mid-shoulder)
for translation invariance, and scaled by shoulder width for
size invariance.
"""

import os
import sys
import json
import time
import logging
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple, Dict, List

logger = logging.getLogger(__name__)

# Upper-body pose indices we care about
POSE_INDICES = [
    0,   # Nose
    11,  # Left shoulder
    12,  # Right shoulder
    13,  # Left elbow
    14,  # Right elbow
    15,  # Left wrist
    16,  # Right wrist
    23,  # Left hip
    24,  # Right hip
    7,   # Left ear (head reference)
]
NUM_POSE = len(POSE_INDICES)  # 10
HAND_LANDMARKS = 21
# Feature dimension: right_hand(63) + left_hand(63) + pose(30) + flags(2)
FEATURE_DIM = HAND_LANDMARKS * 3 * 2 + NUM_POSE * 3 + 2  # 158


def _normalize_landmarks(
    right_hand: Optional[np.ndarray],
    left_hand: Optional[np.ndarray],
    pose: Optional[np.ndarray],
) -> np.ndarray:
    """
    Normalize landmarks for translation/scale invariance.
    
    Returns a flat feature vector of size FEATURE_DIM.
    Missing hands are filled with zeros; presence flags indicate availability.
    """
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    
    # Determine body center and scale from pose
    center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    scale = 1.0
    
    if pose is not None and len(pose) >= 25:
        left_shoulder = pose[11, :3]
        right_shoulder = pose[12, :3]
        # Body center = mid-shoulder point
        center = (left_shoulder + right_shoulder) / 2.0
        # Scale = shoulder width (for size invariance)
        shoulder_dist = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        if shoulder_dist > 0.01:
            scale = shoulder_dist
    
    offset = 0
    
    # Right hand (63 features)
    if right_hand is not None:
        rh = (right_hand - center) / scale
        features[offset:offset + 63] = rh.flatten()
        features[-2] = 1.0  # right hand present flag
    offset += 63
    
    # Left hand (63 features)
    if left_hand is not None:
        lh = (left_hand - center) / scale
        features[offset:offset + 63] = lh.flatten()
        features[-1] = 1.0  # left hand present flag
    offset += 63
    
    # Pose (30 features)
    if pose is not None and len(pose) >= 25:
        for i, idx in enumerate(POSE_INDICES):
            p = (pose[idx, :3] - center) / scale
            features[offset + i * 3:offset + i * 3 + 3] = p
    offset += NUM_POSE * 3
    
    return features


def extract_video_landmarks(
    video_path: str,
    max_frames: int = 300,
    sample_fps: float = 15.0,
) -> Optional[np.ndarray]:
    """
    Extract normalized landmarks from a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to extract
        sample_fps: Target FPS for temporal downsampling
        
    Returns:
        Array of shape (num_frames, FEATURE_DIM) or None if video can't be read
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open: {video_path}")
        return None
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0
    
    # Frame sampling interval
    sample_interval = max(1, int(round(video_fps / sample_fps)))
    
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
        model_complexity=1,
    )
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # Full model for better quality in training data
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
    )
    
    frames_features = []
    frame_idx = 0
    
    while len(frames_features) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Temporal downsampling
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue
        frame_idx += 1
        
        # Process frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        hand_results = hands.process(rgb)
        pose_results = pose.process(rgb)
        
        # Extract hand landmarks
        right_hand = None
        left_hand = None
        
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_lm, handedness in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness,
            ):
                lm = np.array([[l.x, l.y, l.z] for l in hand_lm.landmark], dtype=np.float32)
                label = handedness.classification[0].label
                # Un-mirror: MediaPipe's "Right" in image = user's left hand
                if label == "Right":
                    left_hand = lm
                else:
                    right_hand = lm
        
        # Extract pose
        pose_data = None
        if pose_results.pose_landmarks:
            pose_data = np.array(
                [[l.x, l.y, l.z, l.visibility] for l in pose_results.pose_landmarks.landmark],
                dtype=np.float32,
            )
        
        # Normalize and store
        feat = _normalize_landmarks(right_hand, left_hand, pose_data)
        frames_features.append(feat)
    
    cap.release()
    hands.close()
    pose.close()
    
    if len(frames_features) == 0:
        logger.warning(f"No frames extracted: {video_path}")
        return None
    
    return np.array(frames_features, dtype=np.float32)


def extract_dataset(
    data_dir: str,
    output_dir: str,
    sample_fps: float = 15.0,
) -> Dict:
    """
    Extract landmarks from all videos in the ISL dataset.
    
    Expected structure:
        data_dir/
            Category/
                XX. SignName/
                    video1.MOV
                    video2.MP4
                    ...
    
    Saves:
        output_dir/
            landmarks/
                Category__SignName/
                    0.npy, 1.npy, ...  (one per video)
            metadata.json   (class names, counts, feature dim)
    
    Returns:
        metadata dict
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    landmarks_dir = out_path / "landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover all sign classes
    class_map = {}  # class_name -> class_id
    class_names = []
    all_videos = []  # (video_path, class_name)
    
    for category in sorted(data_path.iterdir()):
        if not category.is_dir():
            continue
        for sign_dir in sorted(category.iterdir()):
            if not sign_dir.is_dir():
                continue
            # Parse sign name: "48. Hello" -> "Hello"
            name_parts = sign_dir.name.split(". ", 1)
            sign_name = name_parts[1] if len(name_parts) > 1 else sign_dir.name
            sign_name = sign_name.strip()
            
            if sign_name not in class_map:
                class_map[sign_name] = len(class_names)
                class_names.append(sign_name)
            
            # Find all video files
            videos = sorted([
                f for f in sign_dir.iterdir()
                if f.suffix.upper() in ('.MOV', '.MP4', '.AVI', '.MKV')
            ])
            for v in videos:
                all_videos.append((str(v), sign_name))
    
    print(f"\nDataset: {len(class_names)} classes, {len(all_videos)} videos")
    print(f"Classes: {class_names}\n")
    
    # Process videos
    success = 0
    fail = 0
    class_counts = {name: 0 for name in class_names}
    t0 = time.time()
    
    for i, (video_path, sign_name) in enumerate(all_videos):
        # Progress
        elapsed = time.time() - t0
        if i > 0:
            eta = elapsed / i * (len(all_videos) - i)
            eta_str = f"ETA: {eta / 60:.1f}min"
        else:
            eta_str = ""
        print(f"\r  [{i + 1}/{len(all_videos)}] {sign_name:25s} {eta_str}    ", end="", flush=True)
        
        # Check if already extracted
        safe_name = sign_name.replace(" ", "_")
        sample_dir = landmarks_dir / safe_name
        sample_dir.mkdir(exist_ok=True)
        sample_idx = class_counts[sign_name]
        out_file = sample_dir / f"{sample_idx}.npy"
        
        if out_file.exists():
            # Skip already extracted
            class_counts[sign_name] += 1
            success += 1
            continue
        
        # Extract
        features = extract_video_landmarks(video_path, sample_fps=sample_fps)
        
        if features is not None and len(features) >= 3:
            np.save(str(out_file), features)
            class_counts[sign_name] += 1
            success += 1
        else:
            fail += 1
            logger.warning(f"Failed: {video_path}")
    
    print(f"\n\nExtraction complete: {success} success, {fail} failed")
    print(f"Time: {(time.time() - t0) / 60:.1f} minutes")
    
    # Save metadata
    metadata = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_map": class_map,
        "class_counts": class_counts,
        "feature_dim": FEATURE_DIM,
        "sample_fps": sample_fps,
        "total_videos": len(all_videos),
        "successful": success,
        "failed": fail,
    }
    
    meta_path = out_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {meta_path}")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract landmarks from ISL video dataset")
    parser.add_argument("--data-dir", default="data", help="Path to dataset root")
    parser.add_argument("--output-dir", default="extracted_data", help="Output directory")
    parser.add_argument("--fps", type=float, default=15.0, help="Sampling FPS")
    args = parser.parse_args()
    
    extract_dataset(args.data_dir, args.output_dir, sample_fps=args.fps)
