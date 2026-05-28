"""
Dynamic gesture / word recognition.
Recognizes common signs (HELLO, THANK YOU, YES, NO, etc.)
by analyzing hand motion trajectories, positions, and finger states over time.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum, auto
import time
import logging

from src.core.hand_analyzer import (
    HandAnalysis,
    FingerState,
    ThumbState,
    PalmFacing,
    HandRegion,
    _distance,
)

logger = logging.getLogger(__name__)


@dataclass
class MotionFrame:
    """Snapshot of hand state at a single time-step."""
    timestamp: float
    right_hand: Optional[HandAnalysis] = None
    left_hand: Optional[HandAnalysis] = None


@dataclass
class GestureResult:
    """Result of gesture recognition."""
    name: str
    confidence: float
    gesture_type: str = "dynamic"  # "static" or "dynamic"


class MotionDirection(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    FORWARD = auto()
    BACKWARD = auto()
    CIRCULAR = auto()
    SIDE_TO_SIDE = auto()
    STATIONARY = auto()


def _compute_motion_direction(positions: List[np.ndarray], threshold: float = 0.02) -> MotionDirection:
    """Classify the dominant motion direction from a sequence of positions."""
    if len(positions) < 3:
        return MotionDirection.STATIONARY

    start = positions[0]
    end = positions[-1]
    delta = end - start

    # Total path length — if negligible, it's stationary regardless of shape
    total_path = sum(np.linalg.norm(positions[i] - positions[i - 1]) for i in range(1, len(positions)))
    if total_path < threshold:
        return MotionDirection.STATIONARY

    # Check for circular motion
    center = np.mean(positions, axis=0)
    distances = [np.linalg.norm(p - center) for p in positions]
    if np.std(distances) < 0.02 and len(positions) > 8:
        # Check if the path loops back near start
        if np.linalg.norm(end - start) < 0.05:
            return MotionDirection.CIRCULAR

    # Check for side-to-side oscillation
    x_values = [p[0] for p in positions]
    if len(x_values) > 6:
        # Count direction changes in X
        diffs = np.diff(x_values)
        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
        if sign_changes >= 3:
            return MotionDirection.SIDE_TO_SIDE

    total_movement = np.linalg.norm(delta[:2])
    if total_movement < threshold:
        return MotionDirection.STATIONARY

    # Dominant direction
    dx, dy = delta[0], delta[1]
    if abs(dx) > abs(dy) * 1.5:
        return MotionDirection.RIGHT if dx > 0 else MotionDirection.LEFT
    elif abs(dy) > abs(dx) * 1.5:
        return MotionDirection.DOWN if dy > 0 else MotionDirection.UP
    else:
        # Check Z for forward/backward
        if len(delta) > 2 and abs(delta[2]) > 0.02:
            return MotionDirection.FORWARD if delta[2] < 0 else MotionDirection.BACKWARD
        return MotionDirection.DOWN if dy > 0 else MotionDirection.UP


def _movement_magnitude(positions: List[np.ndarray]) -> float:
    """Total distance traveled by the hand."""
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(positions)):
        total += np.linalg.norm(positions[i] - positions[i - 1])
    return total


class GestureRecognizer:
    """
    Recognize common sign language gestures from temporal hand data.
    Uses a sliding window of hand analysis frames.
    """

    def __init__(self, window_size: int = 20, min_frames: int = 5):
        """
        Args:
            window_size: Number of frames to keep in the analysis window.
            min_frames: Minimum frames needed before attempting gesture recognition.
        """
        self.window_size = window_size
        self.min_frames = min_frames
        self.buffer: deque[MotionFrame] = deque(maxlen=window_size)
        self.last_gesture: Optional[str] = None
        self.last_gesture_time: float = 0.0
        self.cooldown: float = 0.8  # seconds between same gesture detections

    def add_frame(self, right_hand: Optional[HandAnalysis], left_hand: Optional[HandAnalysis]):
        """Add a new frame to the analysis buffer."""
        self.buffer.append(MotionFrame(
            timestamp=time.time(),
            right_hand=right_hand,
            left_hand=left_hand,
        ))

    def recognize(self) -> Optional[GestureResult]:
        """
        Attempt to recognize a gesture from the current buffer.
        Returns the best matching gesture or None.
        """
        if len(self.buffer) < self.min_frames:
            return None

        candidates: List[GestureResult] = []

        # ── Analyze right hand motion ────────────────────────────────────
        right_positions = []
        right_analyses = []
        for frame in self.buffer:
            if frame.right_hand is not None:
                right_positions.append(
                    np.array([frame.right_hand.center_x, frame.right_hand.center_y, frame.right_hand.center_z])
                )
                right_analyses.append(frame.right_hand)

        # ── Analyze left hand motion ─────────────────────────────────────
        left_positions = []
        left_analyses = []
        for frame in self.buffer:
            if frame.left_hand is not None:
                left_positions.append(
                    np.array([frame.left_hand.center_x, frame.left_hand.center_y, frame.left_hand.center_z])
                )
                left_analyses.append(frame.left_hand)

        both_hands = len(right_positions) > self.min_frames and len(left_positions) > self.min_frames

        # ── HELLO: Open hand wave (side-to-side near head) ──────────────
        if len(right_positions) >= self.min_frames:
            direction = _compute_motion_direction(right_positions)
            recent = right_analyses[-1] if right_analyses else None
            if recent and recent.is_open and direction == MotionDirection.SIDE_TO_SIDE:
                if recent.hand_region in (HandRegion.HEAD_LEVEL, HandRegion.FACE_LEVEL, HandRegion.ABOVE_HEAD):
                    candidates.append(GestureResult("HELLO", 0.9, "dynamic"))

        # ── THANK YOU: Flat hand from chin forward ──────────────────────
        if len(right_positions) >= self.min_frames:
            direction = _compute_motion_direction(right_positions)
            first_rh = right_analyses[0] if right_analyses else None
            last_rh = right_analyses[-1] if right_analyses else None
            if first_rh and last_rh and last_rh.is_open:
                if first_rh.hand_region in (HandRegion.CHIN_LEVEL, HandRegion.FACE_LEVEL):
                    if direction in (MotionDirection.FORWARD, MotionDirection.DOWN):
                        magnitude = _movement_magnitude(right_positions)
                        if magnitude > 0.04:
                            candidates.append(GestureResult("THANK_YOU", 0.85, "dynamic"))

        # ── YES: Fist moving up and down (nodding fist) ────────────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            if recent and recent.is_fist:
                # Check for up-down oscillation
                y_values = [p[1] for p in right_positions]
                if len(y_values) > 4:
                    diffs = np.diff(y_values)
                    sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
                    if sign_changes >= 2:
                        candidates.append(GestureResult("YES", 0.8, "dynamic"))

        # ── NO: Index+middle finger snap/tap against thumb ─────────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            if recent:
                f = recent.fingers
                # Classic "no" in many sign languages: side-to-side with index+middle
                if f.index == FingerState.EXTENDED and f.middle == FingerState.EXTENDED:
                    if f.ring == FingerState.CLOSED and f.pinky == FingerState.CLOSED:
                        direction = _compute_motion_direction(right_positions)
                        if direction == MotionDirection.SIDE_TO_SIDE:
                            candidates.append(GestureResult("NO", 0.8, "dynamic"))

        # ── STOP: Open palm pushes forward ─────────────────────────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            direction = _compute_motion_direction(right_positions)
            if recent and recent.is_open:
                if recent.palm_facing == PalmFacing.FORWARD:
                    if direction in (MotionDirection.FORWARD, MotionDirection.STATIONARY):
                        if recent.hand_region in (HandRegion.CHEST_LEVEL, HandRegion.CHIN_LEVEL):
                            candidates.append(GestureResult("STOP", 0.8, "dynamic"))

        # ── PLEASE: Circular motion on chest ───────────────────────────
        if len(right_positions) >= 8:
            direction = _compute_motion_direction(right_positions)
            recent = right_analyses[-1] if right_analyses else None
            if recent and recent.is_open:
                if direction == MotionDirection.CIRCULAR:
                    if recent.hand_region in (HandRegion.CHEST_LEVEL, HandRegion.CHIN_LEVEL):
                        candidates.append(GestureResult("PLEASE", 0.8, "dynamic"))

        # ── HELP: Fist on open palm, both move up ──────────────────────
        if both_hands:
            recent_r = right_analyses[-1]
            recent_l = left_analyses[-1]
            if recent_r.is_fist and recent_l.is_open:
                r_dir = _compute_motion_direction(right_positions)
                if r_dir == MotionDirection.UP:
                    candidates.append(GestureResult("HELP", 0.8, "dynamic"))

        # ── EAT: Hand to mouth ─────────────────────────────────────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            if recent and recent.hand_region in (HandRegion.FACE_LEVEL, HandRegion.CHIN_LEVEL):
                direction = _compute_motion_direction(right_positions)
                if direction == MotionDirection.UP:
                    # Fingers somewhat closed (holding food)
                    if recent.fingers.extended_count() <= 2:
                        candidates.append(GestureResult("EAT", 0.75, "dynamic"))

        # ── DRINK: Thumb extended, tilting to mouth ────────────────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            if recent:
                if (recent.fingers.thumb_detail == ThumbState.EXTENDED and
                    recent.fingers.index == FingerState.CLOSED and
                    recent.hand_region in (HandRegion.FACE_LEVEL, HandRegion.CHIN_LEVEL)):
                    candidates.append(GestureResult("DRINK", 0.75, "dynamic"))

        # ── LOVE: Both arms crossed on chest ───────────────────────────
        if both_hands:
            recent_r = right_analyses[-1]
            recent_l = left_analyses[-1]
            if (recent_r.is_fist and recent_l.is_fist and
                recent_r.hand_region == HandRegion.CHEST_LEVEL and
                recent_l.hand_region == HandRegion.CHEST_LEVEL):
                # Arms should be crossing (right hand on left side and vice versa)
                if recent_r.center_x < recent_l.center_x:
                    candidates.append(GestureResult("LOVE", 0.8, "dynamic"))

        # ── GOOD: Thumbs up (static check, already handled in fingerspelling,
        #    but sometimes used as a gesture too) ───────────────────────
        if len(right_analyses) > 0:
            recent = right_analyses[-1]
            if recent.is_fist and recent.fingers.thumb_detail == ThumbState.EXTENDED:
                if recent.fingers.thumb == FingerState.EXTENDED:
                    if recent.landmarks is not None:
                        # Thumb pointing up
                        from src.core.hand_analyzer import THUMB_TIP, THUMB_MCP as THUMB_MCP_IDX
                        if recent.landmarks[THUMB_TIP][1] < recent.landmarks[THUMB_MCP_IDX][1]:
                            candidates.append(GestureResult("GOOD", 0.8, "static"))

        # ── BAD: Thumbs down ──────────────────────────────────────────
        if len(right_analyses) > 0:
            recent = right_analyses[-1]
            if recent.is_fist and recent.fingers.thumb_detail == ThumbState.EXTENDED:
                if recent.landmarks is not None:
                    from src.core.hand_analyzer import THUMB_TIP, THUMB_MCP as THUMB_MCP_IDX
                    if recent.landmarks[THUMB_TIP][1] > recent.landmarks[THUMB_MCP_IDX][1]:
                        candidates.append(GestureResult("BAD", 0.8, "static"))

        # ── ILY (I Love You): Thumb + index + pinky extended ──────────
        if len(right_analyses) > 0:
            recent = right_analyses[-1]
            f = recent.fingers
            if (f.thumb == FingerState.EXTENDED and f.index == FingerState.EXTENDED and
                f.middle == FingerState.CLOSED and f.ring == FingerState.CLOSED and
                f.pinky == FingerState.EXTENDED):
                candidates.append(GestureResult("I_LOVE_YOU", 0.9, "static"))

        # ── SORRY: Fist circling on chest ─────────────────────────────
        if len(right_positions) >= 8:
            recent = right_analyses[-1] if right_analyses else None
            direction = _compute_motion_direction(right_positions)
            if recent and recent.is_fist:
                if direction == MotionDirection.CIRCULAR:
                    if recent.hand_region in (HandRegion.CHEST_LEVEL, HandRegion.CHIN_LEVEL):
                        candidates.append(GestureResult("SORRY", 0.82, "dynamic"))

        # ── WANT: Open hand pulling toward body ──────────────────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            first_rh = right_analyses[0] if right_analyses else None
            if recent and first_rh:
                f = recent.fingers
                # Curved / claw hand moving backward (toward body)
                if f.index in (FingerState.HALF_BENT, FingerState.EXTENDED):
                    direction = _compute_motion_direction(right_positions)
                    if direction == MotionDirection.BACKWARD:
                        if recent.hand_region in (HandRegion.CHEST_LEVEL, HandRegion.CHIN_LEVEL):
                            candidates.append(GestureResult("WANT", 0.78, "dynamic"))

        # ── WHERE: Index finger wagging side-to-side (palm forward) ──
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            if recent:
                f = recent.fingers
                if (f.index == FingerState.EXTENDED and f.middle == FingerState.CLOSED and
                    f.ring == FingerState.CLOSED and f.pinky == FingerState.CLOSED):
                    direction = _compute_motion_direction(right_positions)
                    if direction == MotionDirection.SIDE_TO_SIDE:
                        if recent.palm_facing in (PalmFacing.FORWARD, PalmFacing.UP):
                            candidates.append(GestureResult("WHERE", 0.78, "dynamic"))

        # ── WHAT: Open hand, palms up, side-to-side ─────────────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            if recent and recent.is_open:
                direction = _compute_motion_direction(right_positions)
                if direction == MotionDirection.SIDE_TO_SIDE:
                    if recent.palm_facing == PalmFacing.UP:
                        candidates.append(GestureResult("WHAT", 0.78, "dynamic"))

        # ── WHO: Index circling near mouth/chin ─────────────────────
        if len(right_positions) >= 8:
            recent = right_analyses[-1] if right_analyses else None
            if recent:
                f = recent.fingers
                if (f.index == FingerState.EXTENDED and f.middle == FingerState.CLOSED and
                    f.ring == FingerState.CLOSED and f.pinky == FingerState.CLOSED):
                    direction = _compute_motion_direction(right_positions)
                    if direction == MotionDirection.CIRCULAR:
                        if recent.hand_region in (HandRegion.CHIN_LEVEL, HandRegion.FACE_LEVEL):
                            candidates.append(GestureResult("WHO", 0.76, "dynamic"))

        # ── COME: Open hand beckoning toward body ───────────────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            first_rh = right_analyses[0] if right_analyses else None
            if recent and first_rh and recent.is_open:
                direction = _compute_motion_direction(right_positions)
                if direction == MotionDirection.BACKWARD:
                    if recent.palm_facing == PalmFacing.UP:
                        candidates.append(GestureResult("COME", 0.78, "dynamic"))

        # ── GO: Both index fingers pointing forward, moving forward ─
        if both_hands:
            recent_r = right_analyses[-1]
            recent_l = left_analyses[-1]
            fr = recent_r.fingers
            fl = recent_l.fingers
            if (fr.index == FingerState.EXTENDED and fl.index == FingerState.EXTENDED):
                r_dir = _compute_motion_direction(right_positions)
                if r_dir == MotionDirection.FORWARD:
                    candidates.append(GestureResult("GO", 0.78, "dynamic"))

        # ── HOME: Flat hand from chin to cheek (touching chin → ear) ─
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            first_rh = right_analyses[0] if right_analyses else None
            if recent and first_rh:
                if first_rh.hand_region in (HandRegion.CHIN_LEVEL, HandRegion.FACE_LEVEL):
                    if recent.hand_region in (HandRegion.FACE_LEVEL, HandRegion.CHIN_LEVEL):
                        f = recent.fingers
                        if f.extended_count() <= 2:
                            direction = _compute_motion_direction(right_positions)
                            if direction in (MotionDirection.RIGHT, MotionDirection.LEFT):
                                mag = _movement_magnitude(right_positions)
                                if mag > 0.02 and mag < 0.12:
                                    candidates.append(GestureResult("HOME", 0.76, "dynamic"))

        # ── WATER: W-hand (index+middle+ring) tapping chin ──────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            if recent:
                f = recent.fingers
                if (f.index == FingerState.EXTENDED and f.middle == FingerState.EXTENDED and
                    f.ring == FingerState.EXTENDED and f.pinky == FingerState.CLOSED):
                    if recent.hand_region in (HandRegion.CHIN_LEVEL, HandRegion.FACE_LEVEL):
                        # Check for small repeated motion (tapping)
                        y_values = [p[1] for p in right_positions]
                        if len(y_values) > 4:
                            diffs = np.diff(y_values)
                            sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
                            if sign_changes >= 2:
                                candidates.append(GestureResult("WATER", 0.78, "dynamic"))

        # ── MORE: Both flat-O hands tapping together ────────────────
        if both_hands:
            recent_r = right_analyses[-1]
            recent_l = left_analyses[-1]
            fr = recent_r.fingers
            fl = recent_l.fingers
            if (fr.index in (FingerState.HALF_BENT, FingerState.CLOSED) and
                fl.index in (FingerState.HALF_BENT, FingerState.CLOSED)):
                # Hands should be close together
                r_pos = right_positions[-1]
                l_pos = left_positions[-1]
                dist = np.linalg.norm(r_pos - l_pos)
                if dist < 0.1:
                    y_r = [p[1] for p in right_positions]
                    if len(y_r) > 4:
                        diffs = np.diff(y_r)
                        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
                        if sign_changes >= 2:
                            candidates.append(GestureResult("MORE", 0.76, "dynamic"))

        # ── AGAIN: Curved hand arching upward ───────────────────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            if recent:
                f = recent.fingers
                if f.index in (FingerState.HALF_BENT, FingerState.EXTENDED):
                    direction = _compute_motion_direction(right_positions)
                    if direction == MotionDirection.UP:
                        if recent.hand_region in (HandRegion.CHEST_LEVEL, HandRegion.CHIN_LEVEL):
                            if recent.palm_facing == PalmFacing.UP:
                                candidates.append(GestureResult("AGAIN", 0.74, "dynamic"))

        # ── WAIT: Both open hands palms up, slight bounce ───────────
        if both_hands:
            recent_r = right_analyses[-1]
            recent_l = left_analyses[-1]
            if recent_r.is_open and recent_l.is_open:
                if (recent_r.palm_facing == PalmFacing.UP and
                    recent_l.palm_facing == PalmFacing.UP):
                    r_dir = _compute_motion_direction(right_positions)
                    if r_dir in (MotionDirection.STATIONARY, MotionDirection.DOWN):
                        candidates.append(GestureResult("WAIT", 0.74, "dynamic"))

        # ── THINK: Index finger tapping forehead/temple ─────────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            if recent:
                f = recent.fingers
                if (f.index == FingerState.EXTENDED and f.middle == FingerState.CLOSED and
                    f.ring == FingerState.CLOSED and f.pinky == FingerState.CLOSED):
                    if recent.hand_region in (HandRegion.HEAD_LEVEL, HandRegion.FACE_LEVEL):
                        y_vals = [p[1] for p in right_positions]
                        if len(y_vals) > 4:
                            diffs = np.diff(y_vals)
                            sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
                            if sign_changes >= 2:
                                candidates.append(GestureResult("THINK", 0.78, "dynamic"))

        # ── KNOW: Flat hand tapping forehead (open hand forehead) ───
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            if recent and recent.is_open:
                if recent.hand_region in (HandRegion.HEAD_LEVEL, HandRegion.FACE_LEVEL):
                    direction = _compute_motion_direction(right_positions)
                    if direction in (MotionDirection.UP, MotionDirection.FORWARD):
                        magnitude = _movement_magnitude(right_positions)
                        if 0.01 < magnitude < 0.08:
                            candidates.append(GestureResult("KNOW", 0.76, "dynamic"))

        # ── DON'T_KNOW: Open palms flip outward at shoulders ───────
        if both_hands:
            recent_r = right_analyses[-1]
            recent_l = left_analyses[-1]
            if recent_r.is_open and recent_l.is_open:
                if (recent_r.palm_facing == PalmFacing.UP and
                    recent_l.palm_facing == PalmFacing.UP):
                    if (recent_r.hand_region in (HandRegion.CHEST_LEVEL, HandRegion.CHIN_LEVEL) and
                        recent_l.hand_region in (HandRegion.CHEST_LEVEL, HandRegion.CHIN_LEVEL)):
                        r_dir = _compute_motion_direction(right_positions)
                        l_dir = _compute_motion_direction(left_positions)
                        if r_dir in (MotionDirection.UP, MotionDirection.SIDE_TO_SIDE):
                            candidates.append(GestureResult("DON'T_KNOW", 0.76, "dynamic"))

        # ── LIKE: Open hand pulling away from chest (pinch out) ─────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            first_rh = right_analyses[0] if right_analyses else None
            if recent and first_rh:
                if first_rh.hand_region == HandRegion.CHEST_LEVEL:
                    direction = _compute_motion_direction(right_positions)
                    if direction == MotionDirection.FORWARD:
                        f = recent.fingers
                        if f.thumb == FingerState.EXTENDED and f.middle == FingerState.EXTENDED:
                            candidates.append(GestureResult("LIKE", 0.74, "dynamic"))

        # ── FRIEND: Both index fingers hooking together ─────────────
        if both_hands:
            recent_r = right_analyses[-1]
            recent_l = left_analyses[-1]
            fr = recent_r.fingers
            fl = recent_l.fingers
            if (fr.index in (FingerState.HALF_BENT, FingerState.EXTENDED) and
                fl.index in (FingerState.HALF_BENT, FingerState.EXTENDED)):
                if (fr.middle == FingerState.CLOSED and fl.middle == FingerState.CLOSED):
                    r_pos = right_positions[-1]
                    l_pos = left_positions[-1]
                    if np.linalg.norm(r_pos - l_pos) < 0.08:
                        candidates.append(GestureResult("FRIEND", 0.76, "dynamic"))

        # ── FAMILY: Both F-hands circling forward ───────────────────
        if both_hands:
            recent_r = right_analyses[-1]
            recent_l = left_analyses[-1]
            r_dir = _compute_motion_direction(right_positions)
            l_dir = _compute_motion_direction(left_positions)
            if r_dir == MotionDirection.CIRCULAR and l_dir == MotionDirection.CIRCULAR:
                candidates.append(GestureResult("FAMILY", 0.74, "dynamic"))

        # ── SCHOOL: Clap motion (hands clapping) ───────────────────
        if both_hands:
            recent_r = right_analyses[-1]
            recent_l = left_analyses[-1]
            if recent_r.is_open and recent_l.is_open:
                r_pos = right_positions[-1]
                l_pos = left_positions[-1]
                dist = np.linalg.norm(r_pos - l_pos)
                if dist < 0.06:
                    y_r = [p[1] for p in right_positions]
                    if len(y_r) > 4:
                        diffs = np.diff(y_r)
                        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
                        if sign_changes >= 2:
                            candidates.append(GestureResult("SCHOOL", 0.74, "dynamic"))

        # ── WORK: Both fists, dominant fist tapping non-dominant ────
        if both_hands:
            recent_r = right_analyses[-1]
            recent_l = left_analyses[-1]
            if recent_r.is_fist and recent_l.is_fist:
                r_pos = right_positions[-1]
                l_pos = left_positions[-1]
                dist = np.linalg.norm(r_pos - l_pos)
                if dist < 0.08:
                    y_r = [p[1] for p in right_positions]
                    if len(y_r) > 4:
                        diffs = np.diff(y_r)
                        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
                        if sign_changes >= 2:
                            candidates.append(GestureResult("WORK", 0.76, "dynamic"))

        # ── NEED: X-hand (hooked index) bouncing down ──────────────
        if len(right_positions) >= self.min_frames:
            recent = right_analyses[-1] if right_analyses else None
            if recent:
                f = recent.fingers
                if (f.index == FingerState.HALF_BENT and f.middle == FingerState.CLOSED and
                    f.ring == FingerState.CLOSED and f.pinky == FingerState.CLOSED):
                    direction = _compute_motion_direction(right_positions)
                    if direction == MotionDirection.DOWN:
                        candidates.append(GestureResult("NEED", 0.76, "dynamic"))

        # ── Apply cooldown and select best ────────────────────────────
        now = time.time()
        filtered = []
        for c in candidates:
            if c.name == self.last_gesture and (now - self.last_gesture_time) < self.cooldown:
                c.confidence *= 0.5  # Reduce repeated gesture confidence
            filtered.append(c)

        if not filtered:
            return None

        filtered.sort(key=lambda x: x.confidence, reverse=True)
        best = filtered[0]

        if best.confidence >= 0.65:
            self.last_gesture = best.name
            self.last_gesture_time = now
            return best

        return None

    def clear(self):
        """Clear the motion buffer."""
        self.buffer.clear()
        self.last_gesture = None
