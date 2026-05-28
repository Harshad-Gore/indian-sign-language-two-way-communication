"""
Main sign language recognition engine.
Combines fingerspelling + gesture recognition with temporal smoothing,
confidence filtering, and intelligent debouncing.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np
import logging

from src.core.hand_analyzer import HandAnalysis, analyze_hand
from src.core.hand_tracker import TrackingResult
from src.core.speaker import Speaker
from src.recognition.finger_spelling import recognize_letter, recognize_isl_letter
from src.recognition.gesture_recognizer import GestureRecognizer, GestureResult

logger = logging.getLogger(__name__)


@dataclass
class SignResult:
    """Complete recognition result for a frame."""
    # The recognized sign/letter (None if nothing recognized)
    sign: Optional[str] = None
    # Confidence 0-1
    confidence: float = 0.0
    # Whether this is a letter (fingerspelling) or gesture/word
    sign_type: str = "unknown"  # "letter", "gesture", "word"
    # Is this a new detection or continuation of previous
    is_new: bool = False
    # How long this sign has been held (seconds)
    hold_duration: float = 0.0
    # Stable sign (passed temporal smoothing)
    is_stable: bool = False
    # Raw hand analysis for the frame
    right_hand: Optional[HandAnalysis] = None
    left_hand: Optional[HandAnalysis] = None


class SignEngine:
    """
    Production-grade sign language recognition engine.
    
    Features:
    - Instant static sign recognition (fingerspelling)
    - Dynamic gesture recognition with motion analysis
    - Temporal smoothing to prevent flickering
    - Smart debouncing to avoid repeated detections
    - Confidence-weighted voting across frames
    - Automatic word/sentence building
    """

    def __init__(
        self,
        # Temporal smoothing
        stability_frames: int = 6,        # Frames a sign must be consistent to be "stable"
        # Confidence thresholds
        min_confidence: float = 0.75,      # Minimum confidence to consider
        high_confidence: float = 0.92,     # Above this, accept immediately
        # Debouncing
        letter_hold_time: float = 1.5,     # How long a letter must be held to register
        gesture_cooldown: float = 2.0,     # Minimum time between same gesture detections
        # Gesture recognition
        gesture_window: int = 20,          # Frames for gesture analysis
        gesture_min_frames: int = 8,       # Minimum frames for gesture detection
        # TTS
        enable_voice: bool = True,         # Speak committed signs aloud
    ):
        self.stability_frames = stability_frames
        self.min_confidence = min_confidence
        self.high_confidence = high_confidence
        self.letter_hold_time = letter_hold_time
        self.gesture_cooldown = gesture_cooldown

        # Gesture recognizer
        self.gesture_recognizer = GestureRecognizer(
            window_size=gesture_window,
            min_frames=gesture_min_frames,
        )

        # Temporal state
        self.prediction_history: deque[Tuple[Optional[str], float]] = deque(maxlen=12)
        self.current_sign: Optional[str] = None
        self.current_sign_start: float = 0.0
        self.current_sign_type: str = "unknown"
        self.last_committed_sign: Optional[str] = None
        self.last_committed_time: float = 0.0

        # Letter commit lock — prevents same letter repeating without a reset
        self.letter_committed_lock: Optional[str] = None

        # Voice output
        self.speaker: Optional[Speaker] = None
        if enable_voice:
            try:
                self.speaker = Speaker()
            except Exception as e:
                logger.warning(f"Voice output disabled: {e}")

        # Sentence building
        self.committed_signs: List[str] = []
        self.sentence: str = ""

    def process_frame(self, tracking: TrackingResult) -> SignResult:
        """
        Process a single tracking frame and return recognition result.
        This is the main entry point — call once per frame.
        """
        result = SignResult()

        # ── Analyze hands ────────────────────────────────────────────────
        right_analysis = None
        left_analysis = None

        if tracking.right_hand is not None:
            right_analysis = analyze_hand(tracking.right_hand, tracking.pose)
            result.right_hand = right_analysis

        if tracking.left_hand is not None:
            left_analysis = analyze_hand(tracking.left_hand, tracking.pose)
            result.left_hand = left_analysis

        # Feed gesture recognizer (always, for motion accumulation)
        self.gesture_recognizer.add_frame(right_analysis, left_analysis)

        # ── No hands visible ────────────────────────────────────────────
        if right_analysis is None and left_analysis is None:
            self.prediction_history.append((None, 0.0))
            self._check_sign_end()
            return result

        # ── Try fingerspelling (static sign, instant) ───────────────────
        letter = None
        letter_conf = 0.0
        if right_analysis is not None:
            letter, letter_conf = recognize_letter(right_analysis)

        # ── Try gesture recognition (dynamic sign, temporal) ────────────
        gesture: Optional[GestureResult] = self.gesture_recognizer.recognize()

        # ── Select best candidate ───────────────────────────────────────
        best_sign = None
        best_conf = 0.0
        best_type = "unknown"

        if letter and letter_conf >= self.min_confidence:
            best_sign = letter
            best_conf = letter_conf
            best_type = "letter"

        if gesture and gesture.confidence > best_conf:
            best_sign = gesture.name
            best_conf = gesture.confidence
            best_type = "gesture"

        # ── Temporal smoothing ──────────────────────────────────────────
        self.prediction_history.append((best_sign, best_conf))
        smoothed_sign, smoothed_conf = self._get_smoothed_prediction()

        # ── Update current sign state ───────────────────────────────────
        now = time.time()

        if smoothed_sign is not None and smoothed_conf >= self.min_confidence:
            if smoothed_sign != self.current_sign:
                # New sign detected
                self.current_sign = smoothed_sign
                self.current_sign_start = now
                self.current_sign_type = best_type
                result.is_new = True
                # Different sign unlocks the letter commit lock
                if smoothed_sign != self.letter_committed_lock:
                    self.letter_committed_lock = None
            
            hold = now - self.current_sign_start
            result.sign = smoothed_sign
            result.confidence = smoothed_conf
            result.sign_type = self.current_sign_type
            result.hold_duration = hold

            # Check stability
            if self._is_stable(smoothed_sign) or smoothed_conf >= self.high_confidence:
                result.is_stable = True

                # Auto-commit for gestures or held letters
                if self.current_sign_type == "gesture" and hold > 0.5:
                    self._commit_sign(smoothed_sign)
                elif self.current_sign_type == "letter" and hold >= self.letter_hold_time:
                    # Lock-based: same letter can't repeat until a different sign appears
                    if self.letter_committed_lock != smoothed_sign:
                        self._commit_sign(smoothed_sign)
                        self.letter_committed_lock = smoothed_sign
        else:
            self._check_sign_end()

        return result

    def _get_smoothed_prediction(self) -> Tuple[Optional[str], float]:
        """
        Get the most consistent prediction from recent history.
        Uses weighted voting — recent frames count more.
        """
        if not self.prediction_history:
            return None, 0.0

        # Count votes with exponential recency weighting
        votes: Dict[str, float] = {}
        n = len(self.prediction_history)
        for i, (sign, conf) in enumerate(self.prediction_history):
            if sign is not None and conf >= self.min_confidence:
                weight = conf * (0.5 + 0.5 * (i / n))  # More recent = higher weight
                votes[sign] = votes.get(sign, 0) + weight

        if not votes:
            return None, 0.0

        # Best vote
        best = max(votes.items(), key=lambda x: x[1])
        # Normalize confidence
        total = sum(votes.values())
        normalized_conf = best[1] / total if total > 0 else 0

        return best[0], min(normalized_conf, 1.0)

    def _is_stable(self, sign: str) -> bool:
        """Check if the sign has been consistent for enough frames."""
        if len(self.prediction_history) < self.stability_frames:
            return False

        recent = list(self.prediction_history)[-self.stability_frames:]
        consistent = sum(1 for s, c in recent if s == sign and c >= self.min_confidence)
        return consistent >= self.stability_frames - 1  # Allow 1 frame of noise

    def _check_sign_end(self):
        """Check if the current sign has ended (hand disappeared or changed)."""
        if self.current_sign is not None:
            # Check if last few frames had no prediction
            recent = list(self.prediction_history)[-3:]
            no_sign_count = sum(1 for s, c in recent if s is None or c < self.min_confidence)
            if no_sign_count >= 2:
                self.current_sign = None
                self.current_sign_type = "unknown"
                self.letter_committed_lock = None  # Reset lock when sign disappears

    def _commit_sign(self, sign: str):
        """Commit a sign as finalized (add to sentence)."""
        now = time.time()

        # Avoid duplicate commits
        if (sign == self.last_committed_sign and
            now - self.last_committed_time < self.gesture_cooldown):
            return

        self.committed_signs.append(sign)
        self.last_committed_sign = sign
        self.last_committed_time = now
        self.sentence = " ".join(self.committed_signs)

        logger.info(f"Committed: {sign} | Sentence: {self.sentence}")

        # Speak the sign aloud
        if self.speaker is not None:
            self.speaker.say(sign)

    def get_sentence(self) -> str:
        """Get the current built sentence."""
        return self.sentence

    def get_committed_signs(self) -> List[str]:
        """Get all committed signs."""
        return list(self.committed_signs)

    def clear_sentence(self):
        """Clear the sentence buffer."""
        self.committed_signs.clear()
        self.sentence = ""

    def set_sentence(self, text: str):
        """Replace the entire sentence with the given text."""
        self.committed_signs.clear()
        for word in text.split():
            self.committed_signs.append(word)
        self.sentence = " ".join(self.committed_signs)

    def speak_sentence(self):
        """Speak the full current sentence aloud via TTS."""
        if self.speaker is not None and self.sentence.strip():
            self.speaker.say_raw(self.sentence)

    def speak_text(self, text: str):
        """Speak arbitrary text aloud via TTS."""
        if self.speaker is not None and text and text.strip():
            self.speaker.say_raw(text)

    def backspace(self):
        """Remove the last committed sign."""
        if self.committed_signs:
            self.committed_signs.pop()
            self.sentence = " ".join(self.committed_signs)

    def add_space(self):
        """Add a space marker to the sentence."""
        self.committed_signs.append(" ")
        self.sentence = " ".join(s for s in self.committed_signs if s.strip())

    def reset(self):
        """Full reset of all state."""
        self.prediction_history.clear()
        self.current_sign = None
        self.current_sign_start = 0.0
        self.current_sign_type = "unknown"
        self.last_committed_sign = None
        self.last_committed_time = 0.0
        self.letter_committed_lock = None
        self.committed_signs.clear()
        self.sentence = ""
        self.gesture_recognizer.clear()

    def shutdown(self):
        """Release resources (call when done)."""
        if self.speaker is not None:
            self.speaker.shutdown()
            self.speaker = None
