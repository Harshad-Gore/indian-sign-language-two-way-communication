"""
High-performance hand + pose tracker using MediaPipe.
Optimized for minimal latency with maximum landmark quality.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Result from a single frame of tracking."""
    # Hand landmarks: (21, 3) arrays of (x, y, z) normalized [0,1]
    left_hand: Optional[np.ndarray] = None
    right_hand: Optional[np.ndarray] = None
    # Handedness confidence
    left_hand_score: float = 0.0
    right_hand_score: float = 0.0
    # Pose landmarks: (33, 4) array of (x, y, z, visibility)
    pose: Optional[np.ndarray] = None
    # Face landmarks: (478, 3) if available
    face: Optional[np.ndarray] = None
    # Timing
    tracking_time_ms: float = 0.0
    # Frame dimensions
    frame_width: int = 0
    frame_height: int = 0

    @property
    def has_hands(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None

    @property
    def has_right_hand(self) -> bool:
        return self.right_hand is not None

    @property
    def has_left_hand(self) -> bool:
        return self.left_hand is not None

    @property
    def num_hands(self) -> int:
        count = 0
        if self.left_hand is not None:
            count += 1
        if self.right_hand is not None:
            count += 1
        return count


class HandTracker:
    """
    Fast hand + pose tracker using separate MediaPipe solutions
    for optimal speed and accuracy.
    
    Uses MediaPipe Hands (faster, more accurate for hands) + 
    MediaPipe Pose (for body reference frame).
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.4,
        model_complexity: int = 1,
        pose_model_complexity: int = 1,
        use_pose: bool = True,
        use_face: bool = False,
    ):
        self.use_pose = use_pose
        self.use_face = use_face

        # MediaPipe Hands — fast and accurate hand landmark detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

        # MediaPipe Pose — for body reference (shoulder, hip positions)
        self.mp_pose = mp.solutions.pose
        self.pose = None
        if use_pose:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=pose_model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.4,
            )

        # MediaPipe Face Mesh — optional, for facial expression analysis
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = None
        if use_face:
            self.face_mesh = self.mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        logger.info(
            f"HandTracker initialized: hands(complexity={model_complexity}), "
            f"pose={'on' if use_pose else 'off'}, face={'on' if use_face else 'off'}"
        )

    def process(self, frame: np.ndarray) -> TrackingResult:
        """
        Process a BGR frame and return all tracking results.
        Designed for maximum speed — each model runs independently.
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        result = TrackingResult(frame_width=w, frame_height=h)

        # Convert once
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # Performance hint for MediaPipe

        # 1. Hand detection (primary)
        hand_results = self.hands.process(rgb)
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness,
            ):
                landmarks = self._extract_hand_landmarks(hand_landmarks)
                label = handedness.classification[0].label  # "Left" or "Right"
                score = handedness.classification[0].score

                # MediaPipe mirrors: "Left" in image = user's right hand
                # We un-mirror to get the actual hand
                if label == "Right":
                    result.left_hand = landmarks
                    result.left_hand_score = score
                else:
                    result.right_hand = landmarks
                    result.right_hand_score = score

        # 2. Pose detection (for body reference)
        if self.pose is not None:
            pose_results = self.pose.process(rgb)
            if pose_results.pose_landmarks:
                result.pose = self._extract_pose_landmarks(pose_results.pose_landmarks)

        # 3. Face detection (optional)
        if self.face_mesh is not None:
            face_results = self.face_mesh.process(rgb)
            if face_results.multi_face_landmarks:
                result.face = self._extract_face_landmarks(
                    face_results.multi_face_landmarks[0]
                )

        result.tracking_time_ms = (time.perf_counter() - t0) * 1000
        return result

    def _extract_hand_landmarks(self, hand_landmarks) -> np.ndarray:
        """Extract 21 hand landmarks as (21, 3) array."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(hand_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]
        return landmarks

    def _extract_pose_landmarks(self, pose_landmarks) -> np.ndarray:
        """Extract 33 pose landmarks as (33, 4) array."""
        landmarks = np.zeros((33, 4), dtype=np.float32)
        for i, lm in enumerate(pose_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]
        return landmarks

    def _extract_face_landmarks(self, face_landmarks) -> np.ndarray:
        """Extract face mesh landmarks."""
        count = len(face_landmarks.landmark)
        landmarks = np.zeros((count, 3), dtype=np.float32)
        for i, lm in enumerate(face_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]
        return landmarks

    def release(self):
        """Release all MediaPipe resources."""
        self.hands.close()
        if self.pose is not None:
            self.pose.close()
        if self.face_mesh is not None:
            self.face_mesh.close()
