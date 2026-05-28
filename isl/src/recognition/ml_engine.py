"""
ML-powered sign language recognition engine.

Replaces the rule-based sign_engine with one that uses a trained LSTM model
for recognizing ISL signs from landmark sequences. Falls back to the
rule-based system if no trained model is found.

Integrates:
  - ML-based sign recognition (71 ISL signs)
  - Temporal smoothing & confidence weighting
  - Smart debouncing to prevent flickering
  - Automatic word/sentence building
  - Text-to-speech output
"""

import os
import time
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

from src.core.hand_tracker import TrackingResult
from src.core.hand_analyzer import HandAnalysis, analyze_hand
from src.core.speaker import Speaker

logger = logging.getLogger(__name__)


@dataclass
class SignResult:
    """Complete recognition result for a frame."""
    sign: Optional[str] = None
    confidence: float = 0.0
    sign_type: str = "unknown"  # "ml", "letter", "gesture"
    is_new: bool = False
    hold_duration: float = 0.0
    is_stable: bool = False
    right_hand: Optional[HandAnalysis] = None
    left_hand: Optional[HandAnalysis] = None
    top_predictions: Optional[List[Tuple[str, float]]] = None


class MLSignEngine:
    """
    Production-grade sign language recognition engine using trained ML model.
    
    Features:
    - ML-based sign recognition from landmark sequences
    - Temporal smoothing with weighted voting
    - Smart debouncing & commitment logic
    - Sentence building
    - TTS output
    """

    def __init__(
        self,
        model_path: str = "models/best_model.pt",
        # Temporal smoothing
        stability_frames: int = 4,
        # Confidence thresholds
        min_confidence: float = 0.55,
        high_confidence: float = 0.82,
        agreement_threshold: float = 0.58,
        # Debouncing
        sign_hold_time: float = 0.85,
        sign_cooldown: float = 1.2,
        # ML recognizer settings
        predict_interval: int = 1,
        # TTS
        enable_voice: bool = True,
    ):
        self.stability_frames = stability_frames
        self.min_confidence = min_confidence
        self.high_confidence = high_confidence
        self.agreement_threshold = agreement_threshold
        self.sign_hold_time = sign_hold_time
        self.sign_cooldown = sign_cooldown
        
        # ML recognizer
        self.recognizer = None
        self._ml_available = False
        try:
            from src.ml.recognizer import MLRecognizer
            self.recognizer = MLRecognizer(
                model_path=model_path,
                predict_interval=predict_interval,
            )
            self._ml_available = True
            logger.info(f"ML model loaded: {len(self.recognizer.class_names)} signs")
        except FileNotFoundError as e:
            logger.warning(f"ML model not found: {e}")
            logger.info("Falling back to rule-based recognition")
        except Exception as e:
            logger.error(f"ML model error: {e}")
            logger.info("Falling back to rule-based recognition")
        
        # Fall back to rule-based if ML not available
        if not self._ml_available:
            from src.recognition.finger_spelling import recognize_letter
            from src.recognition.gesture_recognizer import GestureRecognizer
            self._recognize_letter = recognize_letter
            self._gesture_recognizer = GestureRecognizer(window_size=20, min_frames=8)
        
        # Temporal state
        self.prediction_history: deque[Tuple[Optional[str], float]] = deque(maxlen=15)
        self.current_sign: Optional[str] = None
        self.current_sign_start: float = 0.0
        self.current_sign_type: str = "unknown"
        self.last_committed_sign: Optional[str] = None
        self.last_committed_time: float = 0.0
        self.sign_committed_lock: Optional[str] = None
        
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

    def _undo_flip(self, landmarks: np.ndarray) -> np.ndarray:
        """Mirror x-coordinates to undo cv2.flip applied before tracking.
        
        The pipeline flips the frame horizontally for natural mirror display
        BEFORE MediaPipe tracking. This means all landmark x-coordinates are
        mirrored compared to the training data (raw non-mirrored videos).
        We undo this by: x_corrected = 1.0 - x_flipped.
        """
        corrected = landmarks.copy()
        corrected[:, 0] = 1.0 - corrected[:, 0]
        return corrected

    def process_frame(self, tracking: TrackingResult) -> SignResult:
        """Process a single tracking frame and return recognition result."""
        result = SignResult()
        
        # ── Analyze hands (for visualization) ────────────────────────
        # Visualization uses the flipped landmarks (matches the displayed frame)
        right_analysis = None
        left_analysis = None
        
        if tracking.right_hand is not None:
            right_analysis = analyze_hand(tracking.right_hand, tracking.pose)
            result.right_hand = right_analysis
        if tracking.left_hand is not None:
            left_analysis = analyze_hand(tracking.left_hand, tracking.pose)
            result.left_hand = left_analysis
        
        # ── Recognition ──────────────────────────────────────────────
        best_sign = None
        best_conf = 0.0
        best_type = "unknown"
        
        if self._ml_available:
            # ── Correct for pipeline's cv2.flip ──────────────────────
            # The pipeline applies cv2.flip(frame, 1) BEFORE tracking for
            # mirror display. This causes TWO mismatches vs training data:
            #   1. Hand assignment is swapped (MediaPipe labels flip, but
            #      the tracker's un-mirror code assumes non-flipped input)
            #   2. x-coordinates are mirrored (x_live ≈ 1.0 - x_training)
            #
            # Fix: swap right_hand ↔ left_hand AND mirror x-coordinates
            # so ML features match the training convention (raw video).
            ml_right = None
            ml_left = None
            ml_pose = None
            
            # Swap hands: tracker's left_hand is actually the user's right
            # hand (due to double-swap from flip + un-mirror code)
            if tracking.left_hand is not None:
                ml_right = self._undo_flip(tracking.left_hand)
            if tracking.right_hand is not None:
                ml_left = self._undo_flip(tracking.right_hand)
            if tracking.pose is not None:
                ml_pose = self._undo_flip(tracking.pose)
            
            # Feed corrected landmarks to ML recognizer
            self.recognizer.add_frame(
                ml_right,
                ml_left,
                ml_pose,
            )
            
            # Get prediction
            sign, conf = self.recognizer.predict()
            if sign is not None and conf >= self.min_confidence:
                best_sign = sign
                best_conf = conf
                best_type = "ml"
                
                # Get top predictions for display
                result.top_predictions = self.recognizer.get_top_k(3)
        else:
            # Fallback: rule-based recognition
            if not self._ml_available:
                self._gesture_recognizer.add_frame(right_analysis, left_analysis)
            
            if right_analysis is not None:
                letter, letter_conf = self._recognize_letter(right_analysis)
                if letter and letter_conf >= self.min_confidence:
                    best_sign = letter
                    best_conf = letter_conf
                    best_type = "letter"
            
            gesture = self._gesture_recognizer.recognize()
            if gesture and gesture.confidence > best_conf:
                best_sign = gesture.name
                best_conf = gesture.confidence
                best_type = "gesture"
        
        # ── No hands → decay ────────────────────────────────────────
        if tracking.right_hand is None and tracking.left_hand is None:
            self.prediction_history.append((None, 0.0))
            self._check_sign_end()
            return result
        
        # ── Temporal smoothing ───────────────────────────────────────
        self.prediction_history.append((best_sign, best_conf))
        smoothed_sign, smoothed_conf, agreement, raw_conf = self._get_smoothed_prediction()
        
        # ── Update current sign state ────────────────────────────────
        now = time.time()
        
        if (
            smoothed_sign is not None
            and smoothed_conf >= self.min_confidence
            and agreement >= self.agreement_threshold
        ):
            if smoothed_sign != self.current_sign:
                self.current_sign = smoothed_sign
                self.current_sign_start = now
                self.current_sign_type = best_type
                result.is_new = True
                if smoothed_sign != self.sign_committed_lock:
                    self.sign_committed_lock = None
            
            hold = now - self.current_sign_start
            result.sign = smoothed_sign
            result.confidence = smoothed_conf
            result.sign_type = self.current_sign_type
            result.hold_duration = hold
            
            # Check stability
            if (
                self._is_stable(smoothed_sign)
                or (raw_conf >= self.high_confidence and agreement >= self.agreement_threshold * 0.85)
            ):
                result.is_stable = True
                
                if hold >= self.sign_hold_time:
                    if self.sign_committed_lock != smoothed_sign:
                        self._commit_sign(smoothed_sign)
                        self.sign_committed_lock = smoothed_sign
        else:
            self._check_sign_end()
        
        return result

    def _get_smoothed_prediction(self) -> Tuple[Optional[str], float, float, float]:
        """Return the most consistent prediction plus agreement and raw confidence."""
        if not self.prediction_history:
            return None, 0.0, 0.0, 0.0
        
        votes: Dict[str, float] = {}
        weighted_confidence: Dict[str, float] = {}
        weighted_counts: Dict[str, float] = {}
        n = len(self.prediction_history)
        for i, (sign, conf) in enumerate(self.prediction_history):
            if sign is not None and conf >= self.min_confidence * 0.8:
                weight = conf * (0.6 + 0.4 * ((i + 1) / n))
                votes[sign] = votes.get(sign, 0) + weight
                weighted_confidence[sign] = weighted_confidence.get(sign, 0.0) + (conf * weight)
                weighted_counts[sign] = weighted_counts.get(sign, 0.0) + weight
        
        if not votes:
            return None, 0.0, 0.0, 0.0
        
        best = max(votes.items(), key=lambda x: x[1])
        total = sum(votes.values())
        agreement = best[1] / total if total > 0 else 0.0
        avg_conf = weighted_confidence[best[0]] / max(weighted_counts[best[0]], 1e-6)
        combined_conf = avg_conf * (0.65 + 0.35 * agreement)

        return best[0], min(combined_conf, 1.0), min(agreement, 1.0), min(avg_conf, 1.0)

    def _is_stable(self, sign: str) -> bool:
        """Check if sign has been consistent for enough frames."""
        if len(self.prediction_history) < self.stability_frames:
            return False
        recent = list(self.prediction_history)[-self.stability_frames:]
        consistent = sum(1 for s, c in recent if s == sign and c >= self.min_confidence * 0.8)
        return consistent >= self.stability_frames - 1

    def _check_sign_end(self):
        """Check if the current sign has ended."""
        if self.current_sign is not None:
            recent = list(self.prediction_history)[-4:]
            no_sign = sum(1 for s, c in recent if s is None or c < self.min_confidence * 0.75)
            if no_sign >= 3:
                self.current_sign = None
                self.current_sign_type = "unknown"
                self.sign_committed_lock = None

    def _commit_sign(self, sign: str):
        """Commit a sign as finalized."""
        now = time.time()
        if (sign == self.last_committed_sign and
            now - self.last_committed_time < self.sign_cooldown):
            return
        
        self.committed_signs.append(sign)
        self.last_committed_sign = sign
        self.last_committed_time = now
        self.sentence = " ".join(self.committed_signs)
        
        logger.info(f"Committed: {sign} | Sentence: {self.sentence}")
        
        if self.speaker is not None:
            self.speaker.say(sign)

    # ── Public API ───────────────────────────────────────────────────

    def get_sentence(self) -> str:
        return self.sentence

    def get_committed_signs(self) -> List[str]:
        return list(self.committed_signs)

    def clear_sentence(self):
        self.committed_signs.clear()
        self.sentence = ""

    def set_sentence(self, text: str):
        self.committed_signs.clear()
        for word in text.split():
            self.committed_signs.append(word)
        self.sentence = " ".join(self.committed_signs)

    def speak_sentence(self):
        if self.speaker is not None and self.sentence.strip():
            self.speaker.say_raw(self.sentence)

    def speak_text(self, text: str):
        if self.speaker is not None and text and text.strip():
            self.speaker.say_raw(text)

    def backspace(self):
        if self.committed_signs:
            self.committed_signs.pop()
            self.sentence = " ".join(self.committed_signs)

    def add_space(self):
        self.committed_signs.append(" ")
        self.sentence = " ".join(s for s in self.committed_signs if s.strip())

    def reset(self):
        self.prediction_history.clear()
        self.current_sign = None
        self.current_sign_start = 0.0
        self.current_sign_type = "unknown"
        self.last_committed_sign = None
        self.last_committed_time = 0.0
        self.sign_committed_lock = None
        self.committed_signs.clear()
        self.sentence = ""
        if self._ml_available:
            self.recognizer.reset()
        else:
            self._gesture_recognizer.clear()

    def shutdown(self):
        if self.speaker is not None:
            self.speaker.shutdown()
            self.speaker = None
