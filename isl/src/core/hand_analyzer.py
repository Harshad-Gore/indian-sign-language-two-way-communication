"""
Deep geometric analysis of hand landmarks.
Computes finger states, angles, orientations, and relative positions
with high precision for sign language recognition.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class FingerState(Enum):
    """State of a single finger."""
    EXTENDED = auto()     # Finger is straight / open
    HALF_BENT = auto()    # Finger is partially bent
    CLOSED = auto()       # Finger is fully bent / closed

    def __repr__(self):
        return self.name


class ThumbState(Enum):
    """Thumb-specific states."""
    EXTENDED = auto()     # Thumb sticks out
    ACROSS = auto()       # Thumb crosses over palm
    CLOSED = auto()       # Thumb tucked in
    TOUCHING_INDEX = auto()  # Thumb tip touches index finger

    def __repr__(self):
        return self.name


class PalmFacing(Enum):
    """Which direction the palm faces."""
    FORWARD = auto()      # Palm faces camera
    BACKWARD = auto()     # Back of hand faces camera (palm away)
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()

    def __repr__(self):
        return self.name


class HandRegion(Enum):
    """Where the hand is relative to the body."""
    ABOVE_HEAD = auto()
    HEAD_LEVEL = auto()
    FACE_LEVEL = auto()
    CHIN_LEVEL = auto()
    CHEST_LEVEL = auto()
    WAIST_LEVEL = auto()
    BELOW_WAIST = auto()
    NEUTRAL = auto()  # No pose reference

    def __repr__(self):
        return self.name


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class FingerAnalysis:
    """Analysis of all five fingers."""
    thumb: FingerState = FingerState.CLOSED
    index: FingerState = FingerState.CLOSED
    middle: FingerState = FingerState.CLOSED
    ring: FingerState = FingerState.CLOSED
    pinky: FingerState = FingerState.CLOSED
    thumb_detail: ThumbState = ThumbState.CLOSED
    # Curl angles (0 = straight, 180 = fully bent)
    thumb_curl: float = 0.0
    index_curl: float = 0.0
    middle_curl: float = 0.0
    ring_curl: float = 0.0
    pinky_curl: float = 0.0

    def extended_count(self) -> int:
        count = 0
        for f in [self.index, self.middle, self.ring, self.pinky]:
            if f == FingerState.EXTENDED:
                count += 1
        if self.thumb == FingerState.EXTENDED:
            count += 1
        return count

    def as_tuple(self) -> Tuple[FingerState, ...]:
        """Return (thumb, index, middle, ring, pinky)."""
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)

    def extended_pattern(self) -> Tuple[bool, ...]:
        """Return boolean tuple of which fingers are extended."""
        return tuple(f == FingerState.EXTENDED for f in self.as_tuple())

    def closed_pattern(self) -> Tuple[bool, ...]:
        """Return boolean tuple of which fingers are closed."""
        return tuple(f == FingerState.CLOSED for f in self.as_tuple())


@dataclass
class HandAnalysis:
    """Complete analysis of a single hand."""
    fingers: FingerAnalysis = field(default_factory=FingerAnalysis)
    palm_facing: PalmFacing = PalmFacing.FORWARD
    hand_region: HandRegion = HandRegion.NEUTRAL
    # Hand center position (normalized 0-1)
    center_x: float = 0.5
    center_y: float = 0.5
    center_z: float = 0.0
    # Wrist position
    wrist_x: float = 0.5
    wrist_y: float = 0.5
    # Key distances (normalized)
    thumb_index_distance: float = 0.0
    index_middle_distance: float = 0.0
    # Hand span (max spread between fingertips)
    hand_span: float = 0.0
    # Orientation angle (degrees, wrist-to-middle-finger direction)
    orientation_angle: float = 0.0
    # Whether hand is making a fist
    is_fist: bool = False
    # Whether hand is fully open (all extended)
    is_open: bool = False
    # Raw landmarks for advanced checks
    landmarks: Optional[np.ndarray] = None


# ── Landmark indices ─────────────────────────────────────────────────────────

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Pose landmark indices
POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24
POSE_LEFT_EAR = 7
POSE_RIGHT_EAR = 8


# ── Analysis functions ───────────────────────────────────────────────────────

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def analyze_finger_curl(mcp: np.ndarray, pip: np.ndarray, dip: np.ndarray, tip: np.ndarray) -> float:
    """
    Compute curl angle of a finger using joint angles.
    Returns 0 for straight, higher values for more bent.
    """
    v1 = pip - mcp
    v2 = dip - pip
    v3 = tip - dip
    angle1 = _angle_between(v1, v2)
    angle2 = _angle_between(v2, v3)
    # Combined curl metric
    return (angle1 + angle2) / 2.0


def analyze_finger_state(
    mcp: np.ndarray,
    pip: np.ndarray,
    dip: np.ndarray,
    tip: np.ndarray,
    wrist: np.ndarray,
) -> Tuple[FingerState, float]:
    """
    Determine if a finger is extended, half-bent, or closed.
    Uses both curl angle and tip-to-MCP distance ratio.
    """
    curl = analyze_finger_curl(mcp, pip, dip, tip)

    # Distance-based check: is tip further from wrist than MCP?
    tip_dist = _distance(tip, wrist)
    mcp_dist = _distance(mcp, wrist)
    pip_dist = _distance(pip, wrist)

    # Tip should be further from wrist than PIP for extended finger
    tip_beyond_pip = tip_dist > pip_dist * 0.85

    if curl < 40 and tip_beyond_pip:
        return FingerState.EXTENDED, curl
    elif curl < 80:
        return FingerState.HALF_BENT, curl
    else:
        return FingerState.CLOSED, curl


def analyze_thumb_state(
    landmarks: np.ndarray,
) -> Tuple[FingerState, ThumbState, float]:
    """
    Special analysis for thumb — it moves differently than other fingers.
    """
    wrist = landmarks[WRIST]
    thumb_cmc = landmarks[THUMB_CMC]
    thumb_mcp = landmarks[THUMB_MCP]
    thumb_ip = landmarks[THUMB_IP]
    thumb_tip = landmarks[THUMB_TIP]
    index_mcp = landmarks[INDEX_MCP]
    index_tip = landmarks[INDEX_TIP]
    pinky_mcp = landmarks[PINKY_MCP]

    # Curl angle
    curl = analyze_finger_curl(thumb_cmc, thumb_mcp, thumb_ip, thumb_tip)

    # Distance from thumb tip to index tip (for circle/pinch detection)
    thumb_index_dist = _distance(thumb_tip, index_tip)

    # Distance from thumb tip to index MCP (for "across" detection)
    thumb_to_index_mcp = _distance(thumb_tip, index_mcp)

    # Hand width reference
    hand_width = _distance(index_mcp, pinky_mcp)

    # Check if thumb tip is touching index finger tip
    if thumb_index_dist < hand_width * 0.35:
        detail = ThumbState.TOUCHING_INDEX
    # Check if thumb is extended outward
    elif curl < 40:
        detail = ThumbState.EXTENDED
    # Check if thumb tip is near index MCP (across palm)
    elif thumb_to_index_mcp < hand_width * 0.6:
        detail = ThumbState.ACROSS
    else:
        detail = ThumbState.CLOSED

    # General state
    if detail in (ThumbState.EXTENDED, ThumbState.TOUCHING_INDEX):
        state = FingerState.EXTENDED
    elif curl < 60:
        state = FingerState.HALF_BENT
    else:
        state = FingerState.CLOSED

    return state, detail, curl


def analyze_palm_facing(landmarks: np.ndarray) -> PalmFacing:
    """
    Determine palm orientation using the cross product of
    wrist→index_mcp and wrist→pinky_mcp vectors.
    """
    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]

    # Vectors along the hand
    v_index = index_mcp - wrist
    v_pinky = pinky_mcp - wrist

    # Normal to the palm plane
    normal = np.cross(v_index, v_pinky)

    if np.linalg.norm(normal) < 1e-8:
        return PalmFacing.FORWARD

    normal = normal / np.linalg.norm(normal)

    # In MediaPipe coordinates:
    # x: right, y: down, z: toward camera (negative = toward camera)
    # Palm normal z-component tells us facing direction
    nz = normal[2]
    nx = normal[0]
    ny = normal[1]

    # Dominant axis
    abs_components = [abs(nx), abs(ny), abs(nz)]
    dominant = np.argmax(abs_components)

    if dominant == 2:  # z-axis dominant
        return PalmFacing.FORWARD if nz < 0 else PalmFacing.BACKWARD
    elif dominant == 1:  # y-axis dominant
        return PalmFacing.DOWN if ny > 0 else PalmFacing.UP
    else:  # x-axis dominant
        return PalmFacing.RIGHT if nx > 0 else PalmFacing.LEFT


def analyze_hand_region(
    hand_landmarks: np.ndarray,
    pose_landmarks: Optional[np.ndarray],
) -> HandRegion:
    """
    Determine where the hand is relative to the body.
    Uses pose landmarks for reference if available.
    """
    if pose_landmarks is None:
        return HandRegion.NEUTRAL

    # Hand center Y position
    hand_center_y = np.mean(hand_landmarks[:, 1])

    # Body reference points
    nose_y = pose_landmarks[POSE_NOSE, 1]
    shoulder_y = (
        pose_landmarks[POSE_LEFT_SHOULDER, 1]
        + pose_landmarks[POSE_RIGHT_SHOULDER, 1]
    ) / 2
    hip_y = (
        pose_landmarks[POSE_LEFT_HIP, 1] + pose_landmarks[POSE_RIGHT_HIP, 1]
    ) / 2
    ear_y = (
        pose_landmarks[POSE_LEFT_EAR, 1] + pose_landmarks[POSE_RIGHT_EAR, 1]
    ) / 2

    # In MediaPipe, y increases downward (0=top, 1=bottom)
    if hand_center_y < ear_y - 0.05:
        return HandRegion.ABOVE_HEAD
    elif hand_center_y < nose_y:
        return HandRegion.HEAD_LEVEL
    elif hand_center_y < (nose_y + shoulder_y) / 2:
        return HandRegion.FACE_LEVEL
    elif hand_center_y < shoulder_y + 0.03:
        return HandRegion.CHIN_LEVEL
    elif hand_center_y < (shoulder_y + hip_y) / 2:
        return HandRegion.CHEST_LEVEL
    elif hand_center_y < hip_y:
        return HandRegion.WAIST_LEVEL
    else:
        return HandRegion.BELOW_WAIST


def analyze_hand(
    hand_landmarks: np.ndarray,
    pose_landmarks: Optional[np.ndarray] = None,
) -> HandAnalysis:
    """
    Complete geometric analysis of a hand from its 21 landmarks.
    This is the main entry point for hand analysis.
    """
    lm = hand_landmarks
    wrist = lm[WRIST]

    # ── Finger analysis ──────────────────────────────────────────────────
    fingers = FingerAnalysis()

    # Thumb (special)
    thumb_state, thumb_detail, thumb_curl = analyze_thumb_state(lm)
    fingers.thumb = thumb_state
    fingers.thumb_detail = thumb_detail
    fingers.thumb_curl = thumb_curl

    # Index
    state, curl = analyze_finger_state(
        lm[INDEX_MCP], lm[INDEX_PIP], lm[INDEX_DIP], lm[INDEX_TIP], wrist
    )
    fingers.index = state
    fingers.index_curl = curl

    # Middle
    state, curl = analyze_finger_state(
        lm[MIDDLE_MCP], lm[MIDDLE_PIP], lm[MIDDLE_DIP], lm[MIDDLE_TIP], wrist
    )
    fingers.middle = state
    fingers.middle_curl = curl

    # Ring
    state, curl = analyze_finger_state(
        lm[RING_MCP], lm[RING_PIP], lm[RING_DIP], lm[RING_TIP], wrist
    )
    fingers.ring = state
    fingers.ring_curl = curl

    # Pinky
    state, curl = analyze_finger_state(
        lm[PINKY_MCP], lm[PINKY_PIP], lm[PINKY_DIP], lm[PINKY_TIP], wrist
    )
    fingers.pinky = state
    fingers.pinky_curl = curl

    # ── Palm facing ──────────────────────────────────────────────────────
    palm = analyze_palm_facing(lm)

    # ── Hand region ──────────────────────────────────────────────────────
    region = analyze_hand_region(lm, pose_landmarks)

    # ── Key distances ────────────────────────────────────────────────────
    hand_width = _distance(lm[INDEX_MCP], lm[PINKY_MCP])
    thumb_index_dist = _distance(lm[THUMB_TIP], lm[INDEX_TIP]) / (hand_width + 1e-8)
    index_middle_dist = _distance(lm[INDEX_TIP], lm[MIDDLE_TIP]) / (hand_width + 1e-8)

    # Hand span (max distance between any two fingertips)
    tips = [lm[THUMB_TIP], lm[INDEX_TIP], lm[MIDDLE_TIP], lm[RING_TIP], lm[PINKY_TIP]]
    max_span = 0
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            d = _distance(tips[i], tips[j])
            max_span = max(max_span, d)
    hand_span = max_span / (hand_width + 1e-8)

    # Center position
    center = np.mean(lm, axis=0)

    # Orientation angle (wrist to middle MCP)
    direction = lm[MIDDLE_MCP] - wrist
    orientation = np.degrees(np.arctan2(direction[1], direction[0]))

    # Fist check
    ext_count = fingers.extended_count()
    is_fist = ext_count == 0
    is_open = ext_count >= 4  # At least 4 fingers extended

    return HandAnalysis(
        fingers=fingers,
        palm_facing=palm,
        hand_region=region,
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        wrist_x=float(wrist[0]),
        wrist_y=float(wrist[1]),
        thumb_index_distance=thumb_index_dist,
        index_middle_distance=index_middle_dist,
        hand_span=hand_span,
        orientation_angle=orientation,
        is_fist=is_fist,
        is_open=is_open,
        landmarks=lm,
    )
