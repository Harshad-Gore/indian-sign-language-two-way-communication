"""
Real-time facial expression recognition using MediaPipe FaceMesh landmarks.

Analyzes 478 face landmarks to detect emotions with confidence scores.
Pure geometric analysis — no ML model needed, zero extra latency.

Detects: Happy, Sad, Surprised, Angry, Fearful, Disgusted, Neutral
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import deque
import time
import math

# ── Key FaceMesh landmark indices ────────────────────────────────────────
# Eye landmarks (for Eye Aspect Ratio)
_LEFT_EYE = {
    "outer": 33, "inner": 133,
    "upper1": 160, "upper2": 158,
    "lower1": 144, "lower2": 153,
}
_RIGHT_EYE = {
    "outer": 263, "inner": 362,
    "upper1": 385, "upper2": 387,
    "lower1": 380, "lower2": 373,
}

# Eyebrow landmarks
_LEFT_BROW  = [70, 63, 105, 66, 107]   # inner → outer
_RIGHT_BROW = [300, 293, 334, 296, 336]

# Mouth landmarks
_MOUTH = {
    "top": 13,       # upper lip center top
    "bottom": 14,    # lower lip center bottom
    "left": 61,      # left corner
    "right": 291,    # right corner
    "upper_inner": 82,   # inner upper lip
    "lower_inner": 87,   # inner lower lip (for lip press detection)
}

# Reference points
_NOSE_TIP = 1
_FOREHEAD = 10    # top of face
_CHIN     = 152   # bottom of face

# Iris landmarks (refined) — for gaze/attention
_LEFT_IRIS  = 468
_RIGHT_IRIS = 473


@dataclass
class ExpressionResult:
    """Result from facial expression analysis."""
    # Primary detected emotion
    emotion: str = "Neutral"
    confidence: float = 0.0
    # All emotion scores (for display)
    scores: dict = field(default_factory=dict)
    # Emoji representation
    emoji: str = "😐"
    # Raw metrics (for debugging / advanced UI)
    mouth_open: float = 0.0     # 0 = closed, 1 = wide open
    smile: float = 0.0          # -1 = frown, 0 = neutral, 1 = full smile
    eye_openness: float = 0.0   # 0 = closed, 1 = wide open
    brow_raise: float = 0.0     # 0 = lowered, 0.5 = neutral, 1 = raised
    # Valid result?
    valid: bool = False


# Emoji map for each emotion
_EMOJI_MAP = {
    "Happy":     "😊",
    "Sad":       "😢",
    "Surprised": "😮",
    "Angry":     "😠",
    "Fearful":   "😨",
    "Disgusted": "🤢",
    "Neutral":   "😐",
}

# Color map for each emotion (BGR for OpenCV)
EMOTION_COLORS = {
    "Happy":     (80, 230, 255),   # Warm yellow
    "Sad":       (210, 150, 80),   # Blue-ish
    "Surprised": (100, 220, 255),  # Orange-ish
    "Angry":     (70, 70, 240),    # Red
    "Fearful":   (200, 150, 200),  # Purple-ish
    "Disgusted": (80, 200, 130),   # Green
    "Neutral":   (180, 180, 180),  # Gray
}


class FacialExpressionAnalyzer:
    """
    Analyzes MediaPipe FaceMesh landmarks (478 points) to detect
    facial expressions in real-time.

    Uses geometric ratios:
    - Eye Aspect Ratio (EAR) → eye openness
    - Mouth Aspect Ratio (MAR) → mouth openness
    - Smile ratio → lip corner elevation
    - Brow-eye distance → eyebrow position
    - Lip press ratio → anger indicator

    Temporal smoothing with exponential moving average for stability.
    """

    def __init__(self, smoothing: float = 0.35, history_size: int = 10):
        """
        Args:
            smoothing: EMA factor (0 = no smoothing, 1 = maximum smoothing).
            history_size: Number of frames for emotion voting.
        """
        self._smoothing = smoothing
        self._history: deque = deque(maxlen=history_size)

        # Smoothed metrics
        self._smooth_ear = 0.25        # baseline EAR ~0.25
        self._smooth_mar = 0.0
        self._smooth_smile = 0.0
        self._smooth_brow = 0.5
        self._smooth_lip_press = 0.0

        # Calibration baselines (auto-calibrate over first N frames)
        self._calibrating = True
        self._calib_frames = 0
        self._calib_ear_sum = 0.0
        self._calib_mar_sum = 0.0
        self._calib_smile_sum = 0.0
        self._calib_brow_sum = 0.0
        self._calib_count = 30          # calibrate over 30 frames (~1 second)

        # Baselines (will be set after calibration)
        self._base_ear = 0.25
        self._base_mar = 0.02
        self._base_smile = 0.0
        self._base_brow = 0.5

        # Last result for sticky display
        self._last_result = ExpressionResult()
        self._last_valid_time = 0.0

    def analyze(self, face_landmarks: Optional[np.ndarray]) -> ExpressionResult:
        """
        Analyze face landmarks and return expression result.

        Args:
            face_landmarks: (478, 3) array of (x, y, z) normalized landmarks,
                            or None if no face detected.

        Returns:
            ExpressionResult with detected emotion and confidence.
        """
        if face_landmarks is None or len(face_landmarks) < 468:
            # No face — decay to neutral over time
            if time.time() - self._last_valid_time > 1.0:
                return ExpressionResult()
            return self._last_result

        result = ExpressionResult(valid=True)
        self._last_valid_time = time.time()

        # ── 1. Compute raw metrics ───────────────────────────────────────
        ear = self._compute_ear(face_landmarks)
        mar = self._compute_mar(face_landmarks)
        smile = self._compute_smile(face_landmarks)
        brow = self._compute_brow_raise(face_landmarks)
        lip_press = self._compute_lip_press(face_landmarks)

        # ── 2. Calibration (first N frames) ──────────────────────────────
        if self._calibrating:
            self._calib_ear_sum += ear
            self._calib_mar_sum += mar
            self._calib_smile_sum += smile
            self._calib_brow_sum += brow
            self._calib_frames += 1

            if self._calib_frames >= self._calib_count:
                n = self._calib_count
                self._base_ear = self._calib_ear_sum / n
                self._base_mar = self._calib_mar_sum / n
                self._base_smile = self._calib_smile_sum / n
                self._base_brow = self._calib_brow_sum / n
                self._calibrating = False

            # During calibration, just return neutral
            result.emotion = "Neutral"
            result.confidence = 0.5
            result.emoji = "😐"
            result.scores = {"Neutral": 0.5}
            self._last_result = result
            return result

        # ── 3. Smooth metrics (EMA) ──────────────────────────────────────
        a = self._smoothing
        self._smooth_ear = a * self._smooth_ear + (1 - a) * ear
        self._smooth_mar = a * self._smooth_mar + (1 - a) * mar
        self._smooth_smile = a * self._smooth_smile + (1 - a) * smile
        self._smooth_brow = a * self._smooth_brow + (1 - a) * brow
        self._smooth_lip_press = a * self._smooth_lip_press + (1 - a) * lip_press

        # ── 4. Compute relative metrics (vs baseline) ────────────────────
        ear_delta = self._smooth_ear - self._base_ear
        mar_delta = self._smooth_mar - self._base_mar
        smile_delta = self._smooth_smile - self._base_smile
        brow_delta = self._smooth_brow - self._base_brow

        # Store raw for UI
        result.mouth_open = max(0.0, min(1.0, mar_delta * 8.0 + 0.1))
        result.smile = max(-1.0, min(1.0, smile_delta * 5.0))
        result.eye_openness = max(0.0, min(1.0, (ear_delta + 0.05) * 6.0 + 0.3))
        result.brow_raise = max(0.0, min(1.0, brow_delta * 4.0 + 0.5))

        # ── 5. Score each emotion ────────────────────────────────────────
        scores = self._score_emotions(
            ear_delta, mar_delta, smile_delta,
            brow_delta, self._smooth_lip_press
        )

        # ── 6. Temporal voting ───────────────────────────────────────────
        best_emotion = max(scores, key=scores.get)
        self._history.append(best_emotion)

        # Count votes
        vote_counts = {}
        for e in self._history:
            vote_counts[e] = vote_counts.get(e, 0) + 1

        voted_emotion = max(vote_counts, key=vote_counts.get)
        vote_confidence = vote_counts[voted_emotion] / len(self._history)

        # Blend raw score with vote confidence
        raw_conf = scores.get(voted_emotion, 0.0)
        final_conf = 0.6 * raw_conf + 0.4 * vote_confidence

        result.emotion = voted_emotion
        result.confidence = max(0.0, min(1.0, final_conf))
        result.emoji = _EMOJI_MAP.get(voted_emotion, "😐")
        result.scores = scores

        self._last_result = result
        return result

    # ── Geometric computations ───────────────────────────────────────────

    def _dist(self, lm: np.ndarray, i: int, j: int) -> float:
        """Euclidean distance between two landmarks (2D, ignoring z)."""
        dx = lm[i][0] - lm[j][0]
        dy = lm[i][1] - lm[j][1]
        return math.sqrt(dx * dx + dy * dy)

    def _face_height(self, lm: np.ndarray) -> float:
        """Vertical face height (forehead to chin) for normalization."""
        return max(self._dist(lm, _FOREHEAD, _CHIN), 0.001)

    def _compute_ear(self, lm: np.ndarray) -> float:
        """
        Eye Aspect Ratio (EAR) — averaged over both eyes.
        Higher = eyes more open.
        EAR = (|upper1-lower1| + |upper2-lower2|) / (2 * |outer-inner|)
        """
        def _eye_ear(eye: dict) -> float:
            v1 = self._dist(lm, eye["upper1"], eye["lower1"])
            v2 = self._dist(lm, eye["upper2"], eye["lower2"])
            h  = self._dist(lm, eye["outer"],  eye["inner"])
            return (v1 + v2) / (2.0 * max(h, 0.001))

        left_ear = _eye_ear(_LEFT_EYE)
        right_ear = _eye_ear(_RIGHT_EYE)
        return (left_ear + right_ear) / 2.0

    def _compute_mar(self, lm: np.ndarray) -> float:
        """
        Mouth Aspect Ratio (MAR) — how open the mouth is.
        MAR = vertical_opening / horizontal_width, normalized by face height.
        """
        v = self._dist(lm, _MOUTH["top"], _MOUTH["bottom"])
        h = self._dist(lm, _MOUTH["left"], _MOUTH["right"])
        fh = self._face_height(lm)
        return (v / max(h, 0.001)) * (h / fh)

    def _compute_smile(self, lm: np.ndarray) -> float:
        """
        Smile metric — measures lip corner elevation relative to mouth center.
        Positive = smile (corners up), Negative = frown (corners down).
        Normalized by face height.
        """
        fh = self._face_height(lm)

        # Mouth center y
        mouth_center_y = (lm[_MOUTH["top"]][1] + lm[_MOUTH["bottom"]][1]) / 2.0

        # Average lip corner y (lower = higher on screen = more smile)
        corner_y = (lm[_MOUTH["left"]][1] + lm[_MOUTH["right"]][1]) / 2.0

        # Smile = corners above center (negative delta in normalized coords)
        # Normalized by face height
        smile = (mouth_center_y - corner_y) / fh

        return smile

    def _compute_brow_raise(self, lm: np.ndarray) -> float:
        """
        Eyebrow raise metric — distance from brow to eye, normalized by face height.
        Higher value = more raised brows.
        """
        fh = self._face_height(lm)

        # Left: average brow y vs left eye upper
        left_brow_y = np.mean([lm[i][1] for i in _LEFT_BROW])
        left_eye_y = lm[_LEFT_EYE["upper1"]][1]
        left_dist = (left_eye_y - left_brow_y) / fh

        # Right
        right_brow_y = np.mean([lm[i][1] for i in _RIGHT_BROW])
        right_eye_y = lm[_RIGHT_EYE["upper1"]][1]
        right_dist = (right_eye_y - right_brow_y) / fh

        return (left_dist + right_dist) / 2.0

    def _compute_lip_press(self, lm: np.ndarray) -> float:
        """
        Lip press metric — how tightly lips are pressed together.
        Small gap = pressed (anger indicator).
        """
        fh = self._face_height(lm)
        inner_gap = self._dist(lm, _MOUTH["upper_inner"], _MOUTH["lower_inner"])
        return 1.0 - min(inner_gap / (fh * 0.05), 1.0)

    # ── Emotion scoring ──────────────────────────────────────────────────

    def _score_emotions(
        self,
        ear_delta: float,
        mar_delta: float,
        smile_delta: float,
        brow_delta: float,
        lip_press: float,
    ) -> dict:
        """
        Score each emotion based on facial metric deltas.
        Returns dict of emotion → confidence [0, 1].
        """
        scores = {}

        # ── Happy: smile up, may have open mouth (laughing) ──────────────
        smile_score = max(0, smile_delta * 12.0)
        mouth_bonus = max(0, mar_delta * 3.0) * 0.3
        scores["Happy"] = min(1.0, smile_score + mouth_bonus)

        # ── Sad: frown (negative smile), lowered brows ───────────────────
        frown_score = max(0, -smile_delta * 10.0)
        low_brow = max(0, -brow_delta * 6.0) * 0.3
        scores["Sad"] = min(1.0, frown_score + low_brow)

        # ── Surprised: wide eyes, open mouth, raised brows ───────────────
        wide_eyes = max(0, ear_delta * 15.0)
        open_mouth = max(0, mar_delta * 6.0)
        raised_brows = max(0, brow_delta * 8.0)
        scores["Surprised"] = min(1.0, (wide_eyes + open_mouth + raised_brows) / 2.5)

        # ── Angry: furrowed brows, pressed lips, narrowed eyes ───────────
        furrow = max(0, -brow_delta * 8.0)
        pressed = lip_press * 0.5
        narrow = max(0, -ear_delta * 10.0) * 0.3
        scores["Angry"] = min(1.0, (furrow + pressed + narrow) / 1.5)

        # ── Fearful: wide eyes, slightly open mouth, raised tense brows ──
        fear_eyes = max(0, ear_delta * 10.0)
        fear_mouth = max(0, mar_delta * 3.0)
        fear_brow = max(0, brow_delta * 5.0)
        # Distinguish from surprise: fear has less mouth opening
        fear_penalty = max(0, mar_delta * 4.0)  # penalize large mouth opening
        scores["Fearful"] = min(1.0, max(0, (fear_eyes + fear_brow) / 2.0 - fear_penalty * 0.5))

        # ── Disgusted: one-sided sneer, slight mouth open ────────────────
        # Hard to detect reliably from landmarks alone — keep it subtle
        sneer = max(0, -smile_delta * 4.0)
        slight_open = max(0, mar_delta * 2.0 - 0.05)
        scores["Disgusted"] = min(1.0, (sneer + slight_open) / 2.5)

        # ── Neutral: absence of strong signals ───────────────────────────
        max_non_neutral = max(
            scores.get("Happy", 0), scores.get("Sad", 0),
            scores.get("Surprised", 0), scores.get("Angry", 0),
            scores.get("Fearful", 0), scores.get("Disgusted", 0),
        )
        scores["Neutral"] = max(0.0, 1.0 - max_non_neutral * 1.5)

        # Normalize so they sum to 1.0 (softmax-like)
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total

        return scores

    def reset_calibration(self):
        """Force a new calibration cycle (useful when user changes position)."""
        self._calibrating = True
        self._calib_frames = 0
        self._calib_ear_sum = 0.0
        self._calib_mar_sum = 0.0
        self._calib_smile_sum = 0.0
        self._calib_brow_sum = 0.0
