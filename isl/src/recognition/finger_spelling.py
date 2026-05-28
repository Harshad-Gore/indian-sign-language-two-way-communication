"""
ASL / ISL Fingerspelling (Alphabet) recognition.
Uses precise geometric rules based on finger states, angles, and positions.
Each letter is defined by a unique combination of finger configurations.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from src.core.hand_analyzer import (
    HandAnalysis,
    FingerState,
    ThumbState,
    PalmFacing,
    _distance,
    THUMB_TIP,
    INDEX_TIP,
    INDEX_MCP,
    INDEX_DIP,
    INDEX_PIP,
    MIDDLE_TIP,
    MIDDLE_MCP,
    MIDDLE_DIP,
    MIDDLE_PIP,
    RING_TIP,
    RING_MCP,
    RING_DIP,
    RING_PIP,
    PINKY_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_DIP,
    THUMB_IP,
    THUMB_MCP as THUMB_MCP_IDX,
    WRIST,
)
import logging

logger = logging.getLogger(__name__)

E = FingerState.EXTENDED
H = FingerState.HALF_BENT
C = FingerState.CLOSED


def _is(state: FingerState, target: FingerState) -> bool:
    return state == target


def _extended(state: FingerState) -> bool:
    return state == FingerState.EXTENDED


def _closed(state: FingerState) -> bool:
    return state == FingerState.CLOSED


def _not_extended(state: FingerState) -> bool:
    return state != FingerState.EXTENDED


def _tips_touching(lm: np.ndarray, i1: int, i2: int, threshold: float = 0.04) -> bool:
    """Check if two landmark points are close together."""
    return _distance(lm[i1], lm[i2]) < threshold


def _tip_below_pip(lm: np.ndarray, tip: int, pip: int) -> bool:
    """Check if fingertip is below (higher y = lower in screen) its PIP joint."""
    return lm[tip][1] > lm[pip][1]


def _is_neutral_hand(hand: HandAnalysis) -> bool:
    """
    Reject resting/idle hands that aren't deliberately forming a sign.
    A resting hand typically has most fingers in the ambiguous half-bent zone
    with the thumb relaxed (not in a distinctive position).
    """
    f = hand.fingers
    half_count = sum(1 for s in [f.index, f.middle, f.ring, f.pinky]
                     if s == FingerState.HALF_BENT)
    # 3+ fingers half-bent AND thumb NOT in a distinctive state → resting hand
    if half_count >= 3 and f.thumb_detail not in (ThumbState.TOUCHING_INDEX, ThumbState.EXTENDED):
        return True
    return False


def recognize_letter(hand: HandAnalysis) -> Tuple[Optional[str], float]:
    """
    Recognize an ASL alphabet letter from hand analysis.
    
    Returns:
        (letter, confidence) or (None, 0.0) if no match.
    """
    f = hand.fingers
    lm = hand.landmarks
    if lm is None:
        return None, 0.0

    # Reject neutral/resting hands before attempting any matching
    if _is_neutral_hand(hand):
        return None, 0.0

    palm = hand.palm_facing
    td = f.thumb_detail
    hand_width = _distance(lm[INDEX_MCP], lm[PINKY_MCP])

    # We'll collect candidates with confidence scores
    candidates: list[Tuple[str, float]] = []

    # ═══════════════════════════════════════════════════════════════════════
    # A: Fist with thumb alongside (thumb extends to the side, not tucked)
    # ═══════════════════════════════════════════════════════════════════════
    if (_closed(f.index) and _closed(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if td in (ThumbState.EXTENDED, ThumbState.ACROSS):
            # Require tight fist — all finger curls must be high (not loosely bent)
            curls_a = [f.index_curl, f.middle_curl, f.ring_curl, f.pinky_curl]
            if all(c > 85 for c in curls_a):
                # Palm should not face down (dangling/resting hand)
                if palm != PalmFacing.DOWN:
                    score = 0.85
                    if f.thumb_curl < 60:
                        score += 0.05
                    candidates.append(("A", score))

    # ═══════════════════════════════════════════════════════════════════════
    # B: Four fingers extended & together, thumb across palm, palm forward
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _extended(f.middle) and _extended(f.ring) and _extended(f.pinky)):
        if td in (ThumbState.ACROSS, ThumbState.CLOSED):
            # Require intentional palm-forward or palm-left/right pose
            if palm in (PalmFacing.FORWARD, PalmFacing.LEFT, PalmFacing.RIGHT):
                score = 0.85
                # Fingers should be close together
                idx_mid = _distance(lm[INDEX_TIP], lm[MIDDLE_TIP])
                if idx_mid < hand_width * 0.6:
                    score += 0.1
                candidates.append(("B", score))

    # ═══════════════════════════════════════════════════════════════════════
    # C: All fingers uniformly curved into a C-shape, palm faces sideways
    # ═══════════════════════════════════════════════════════════════════════
    if (f.index == H and f.middle == H and f.ring == H and f.pinky == H):
        # ALL four fingers must be half-bent (the C-curve), not extended
        if td == ThumbState.EXTENDED and f.thumb in (E, H):
            curls_c = [f.index_curl, f.middle_curl, f.ring_curl, f.pinky_curl]
            avg_curl = np.mean(curls_c)
            curl_std = np.std(curls_c)
            # Require uniform curvature in the C-shape range
            if 35 < avg_curl < 75 and curl_std < 15:
                # C is shown from the side — palm must face LEFT or RIGHT
                if palm in (PalmFacing.LEFT, PalmFacing.RIGHT):
                    # Visible C gap between thumb and index finger
                    if hand.thumb_index_distance > 0.8:
                        candidates.append(("C", 0.88))

    # ═══════════════════════════════════════════════════════════════════════
    # D: Index extended, middle/ring/pinky closed, thumb touches middle
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _closed(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if td == ThumbState.TOUCHING_INDEX or _tips_touching(lm, THUMB_TIP, MIDDLE_TIP, 0.06):
            candidates.append(("D", 0.85))

    # ═══════════════════════════════════════════════════════════════════════
    # E: All fingers bent down, thumb across front
    # ═══════════════════════════════════════════════════════════════════════
    if (_not_extended(f.index) and _not_extended(f.middle) and _not_extended(f.ring) and _not_extended(f.pinky)):
        # Tips should be curled inward (below PIP joints)
        tips_down = sum(1 for t, p in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), 
                                        (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]
                       if _tip_below_pip(lm, t, p))
        if tips_down >= 3 and td in (ThumbState.ACROSS, ThumbState.CLOSED):
            score = 0.7 + 0.05 * tips_down
            # Differentiate from A: in E, fingertips are visible and curled
            if f.index == H or f.middle == H:
                score += 0.05
            candidates.append(("E", score))

    # ═══════════════════════════════════════════════════════════════════════
    # F: Index+Thumb circle/touching, middle/ring/pinky extended
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.middle) and _extended(f.ring) and _extended(f.pinky)):
        if td == ThumbState.TOUCHING_INDEX or _tips_touching(lm, THUMB_TIP, INDEX_TIP, 0.05):
            score = 0.9
            candidates.append(("F", score))

    # ═══════════════════════════════════════════════════════════════════════
    # G: Index points sideways, thumb extends
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _closed(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if palm in (PalmFacing.LEFT, PalmFacing.RIGHT):
            score = 0.75
            if _extended(f.thumb) or f.thumb == H:
                score += 0.1
            candidates.append(("G", score))

    # ═══════════════════════════════════════════════════════════════════════
    # H: Index + middle extend sideways
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _extended(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if palm in (PalmFacing.LEFT, PalmFacing.RIGHT, PalmFacing.DOWN):
            score = 0.8
            candidates.append(("H", score))

    # ═══════════════════════════════════════════════════════════════════════
    # I: Pinky only extended, others closed
    # ═══════════════════════════════════════════════════════════════════════
    if (_closed(f.index) and _closed(f.middle) and _closed(f.ring) and _extended(f.pinky)):
        if _closed(f.thumb) or td == ThumbState.ACROSS:
            candidates.append(("I", 0.9))

    # ═══════════════════════════════════════════════════════════════════════
    # K: Index extended up, middle extended at angle, thumb between
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _extended(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if palm in (PalmFacing.FORWARD, PalmFacing.BACKWARD):
            # Check thumb is between index and middle
            if _extended(f.thumb) or f.thumb == H:
                score = 0.75
                candidates.append(("K", score))

    # ═══════════════════════════════════════════════════════════════════════
    # L: Index + thumb make L shape (index up, thumb out)
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _closed(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if _extended(f.thumb) or td == ThumbState.EXTENDED:
            # Index and thumb should form roughly 90 degrees
            v_thumb = lm[THUMB_TIP] - lm[WRIST]
            v_index = lm[INDEX_TIP] - lm[WRIST]
            from src.core.hand_analyzer import _angle_between
            angle = _angle_between(v_thumb[:2], v_index[:2])
            if 45 < angle < 135:
                score = 0.85
                candidates.append(("L", score))

    # ═══════════════════════════════════════════════════════════════════════
    # M: Three fingers (index, middle, ring) over thumb
    # ═══════════════════════════════════════════════════════════════════════
    if (_not_extended(f.index) and _not_extended(f.middle) and _not_extended(f.ring)):
        if _closed(f.pinky) or _not_extended(f.pinky):
            # Thumb should be tucked under fingers
            if td in (ThumbState.CLOSED, ThumbState.ACROSS):
                # Tips below MCP joints
                if (lm[INDEX_TIP][1] > lm[INDEX_MCP][1] and
                    lm[MIDDLE_TIP][1] > lm[MIDDLE_MCP][1] and
                    lm[RING_TIP][1] > lm[RING_MCP][1]):
                    candidates.append(("M", 0.7))

    # ═══════════════════════════════════════════════════════════════════════
    # N: Index + middle over thumb (similar to M but only 2 fingers)
    # ═══════════════════════════════════════════════════════════════════════
    if (_not_extended(f.index) and _not_extended(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if td in (ThumbState.CLOSED, ThumbState.ACROSS):
            if (lm[INDEX_TIP][1] > lm[INDEX_MCP][1] and
                lm[MIDDLE_TIP][1] > lm[MIDDLE_MCP][1]):
                candidates.append(("N", 0.7))

    # ═══════════════════════════════════════════════════════════════════════
    # O: All fingers curved into O shape, thumb meets fingers
    # ═══════════════════════════════════════════════════════════════════════
    if (f.index in (H, C) and f.middle in (H, C) and f.ring in (H, C) and f.pinky in (H, C)):
        if td == ThumbState.TOUCHING_INDEX or _tips_touching(lm, THUMB_TIP, INDEX_TIP, 0.06):
            # All fingertips should be close together 
            tip_spread = _distance(lm[INDEX_TIP], lm[PINKY_TIP])
            if tip_spread < hand_width * 0.8:
                candidates.append(("O", 0.8))

    # ═══════════════════════════════════════════════════════════════════════
    # P: Like K but pointing down (index out, middle angled, palm down)
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and f.middle in (E, H) and _closed(f.ring) and _closed(f.pinky)):
        if palm == PalmFacing.DOWN:
            candidates.append(("P", 0.75))

    # ═══════════════════════════════════════════════════════════════════════
    # Q: Like G but pointing down  
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _closed(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if palm == PalmFacing.DOWN and (_extended(f.thumb) or f.thumb == H):
            candidates.append(("Q", 0.75))

    # ═══════════════════════════════════════════════════════════════════════
    # R: Index + middle crossed/together, others closed
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _extended(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        # Index and middle tips should be very close (crossed)
        idx_mid_dist = _distance(lm[INDEX_TIP], lm[MIDDLE_TIP])
        if idx_mid_dist < hand_width * 0.25:
            candidates.append(("R", 0.8))

    # ═══════════════════════════════════════════════════════════════════════
    # S: Fist with thumb across front of fingers
    # ═══════════════════════════════════════════════════════════════════════
    if (_closed(f.index) and _closed(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if td == ThumbState.ACROSS:
            candidates.append(("S", 0.8))

    # ═══════════════════════════════════════════════════════════════════════
    # T: Thumb between index and middle (tucked in fist)
    # ═══════════════════════════════════════════════════════════════════════
    if (_closed(f.index) and _closed(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if td == ThumbState.CLOSED:
            # Thumb tip should be between index and middle fingers
            thumb_between = (lm[THUMB_TIP][0] > min(lm[INDEX_MCP][0], lm[MIDDLE_MCP][0]) - 0.02)
            if thumb_between:
                candidates.append(("T", 0.7))

    # ═══════════════════════════════════════════════════════════════════════
    # U: Index + middle extended together, others closed
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _extended(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        idx_mid_dist = _distance(lm[INDEX_TIP], lm[MIDDLE_TIP])
        if idx_mid_dist < hand_width * 0.5:
            if palm in (PalmFacing.FORWARD, PalmFacing.BACKWARD):
                score = 0.8
                if _closed(f.thumb) or td == ThumbState.ACROSS:
                    score += 0.05
                candidates.append(("U", score))

    # ═══════════════════════════════════════════════════════════════════════
    # V: Index + middle extended apart (peace sign)
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _extended(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        idx_mid_dist = _distance(lm[INDEX_TIP], lm[MIDDLE_TIP])
        if idx_mid_dist > hand_width * 0.4:
            score = 0.85
            if _closed(f.thumb) or td in (ThumbState.ACROSS, ThumbState.CLOSED):
                score += 0.05
            candidates.append(("V", score))

    # ═══════════════════════════════════════════════════════════════════════
    # W: Index + middle + ring extended apart
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _extended(f.middle) and _extended(f.ring) and _closed(f.pinky)):
        if _closed(f.thumb) or td in (ThumbState.ACROSS, ThumbState.CLOSED):
            candidates.append(("W", 0.85))

    # ═══════════════════════════════════════════════════════════════════════
    # X: Index hooked/bent, others closed
    # ═══════════════════════════════════════════════════════════════════════
    if (f.index == H and _closed(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if _closed(f.thumb) or td in (ThumbState.ACROSS, ThumbState.CLOSED):
            candidates.append(("X", 0.8))

    # ═══════════════════════════════════════════════════════════════════════
    # Y: Thumb + pinky extended, others closed (hang loose / shaka)
    # ═══════════════════════════════════════════════════════════════════════
    if (_closed(f.index) and _closed(f.middle) and _closed(f.ring) and _extended(f.pinky)):
        if _extended(f.thumb) or td == ThumbState.EXTENDED:
            candidates.append(("Y", 0.9))

    # ═══════════════════════════════════════════════════════════════════════
    # 1 (One): Index only extended
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _closed(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if _closed(f.thumb) or td in (ThumbState.ACROSS, ThumbState.CLOSED):
            if palm in (PalmFacing.FORWARD, PalmFacing.BACKWARD):
                candidates.append(("1", 0.8))

    # ═══════════════════════════════════════════════════════════════════════
    # 2 (Two): Index + middle extended apart (similar to V, palm forward)
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _extended(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if _closed(f.thumb) or td in (ThumbState.ACROSS, ThumbState.CLOSED):
            idx_mid_dist = _distance(lm[INDEX_TIP], lm[MIDDLE_TIP])
            if idx_mid_dist > hand_width * 0.4:
                if palm == PalmFacing.FORWARD:
                    candidates.append(("2", 0.78))

    # ═══════════════════════════════════════════════════════════════════════
    # 3 (Three): Thumb + index + middle extended
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _extended(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if _extended(f.thumb) or td == ThumbState.EXTENDED:
            if palm in (PalmFacing.FORWARD, PalmFacing.BACKWARD):
                candidates.append(("3", 0.8))

    # ═══════════════════════════════════════════════════════════════════════
    # 4 (Four): All four fingers extended, thumb closed across palm
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _extended(f.middle) and _extended(f.ring) and _extended(f.pinky)):
        if td in (ThumbState.ACROSS, ThumbState.CLOSED):
            if palm == PalmFacing.FORWARD:
                candidates.append(("4", 0.8))

    # ═══════════════════════════════════════════════════════════════════════
    # 5 / Open hand: All fingers maximally spread, palm forward
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.index) and _extended(f.middle) and _extended(f.ring) and _extended(f.pinky)):
        if _extended(f.thumb) or td == ThumbState.EXTENDED:
            if palm == PalmFacing.FORWARD:
                # Require fingers to be very straight (deliberately extended)
                curls_5 = [f.index_curl, f.middle_curl, f.ring_curl, f.pinky_curl]
                if all(c < 25 for c in curls_5):
                    # Require maximum spread (fingers deliberately fanned out)
                    if hand.hand_span > 2.0:
                        candidates.append(("5", 0.88))

    # ═══════════════════════════════════════════════════════════════════════
    # 6 (Six): Thumb + pinky extended, middle 3 closed (like Y but palm fwd)
    # ═══════════════════════════════════════════════════════════════════════
    if (_closed(f.index) and _closed(f.middle) and _closed(f.ring) and _extended(f.pinky)):
        if _extended(f.thumb) or td == ThumbState.EXTENDED:
            if palm == PalmFacing.FORWARD:
                candidates.append(("6", 0.78))

    # ═══════════════════════════════════════════════════════════════════════
    # 7 (Seven): Thumb + ring extended, others closed (pinky side)
    # ═══════════════════════════════════════════════════════════════════════
    if (_closed(f.index) and _closed(f.middle) and _extended(f.ring) and _closed(f.pinky)):
        if _extended(f.thumb) or td == ThumbState.EXTENDED:
            if palm == PalmFacing.FORWARD:
                candidates.append(("7", 0.78))

    # ═══════════════════════════════════════════════════════════════════════
    # 8 (Eight): Thumb + middle extended, index/ring/pinky closed
    # ═══════════════════════════════════════════════════════════════════════
    if (_closed(f.index) and _extended(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if _extended(f.thumb) or td == ThumbState.EXTENDED:
            if palm == PalmFacing.FORWARD:
                candidates.append(("8", 0.78))

    # ═══════════════════════════════════════════════════════════════════════
    # 9 (Nine): Thumb + index circle (like F/OK), palm forward
    # ═══════════════════════════════════════════════════════════════════════
    if (_extended(f.middle) and _extended(f.ring) and _extended(f.pinky)):
        if td == ThumbState.TOUCHING_INDEX or _tips_touching(lm, THUMB_TIP, INDEX_TIP, 0.05):
            if palm == PalmFacing.FORWARD:
                candidates.append(("9", 0.78))

    # ═══════════════════════════════════════════════════════════════════════
    # 0 (Zero): All fingers form O (similar to letter O)
    # ═══════════════════════════════════════════════════════════════════════
    if (f.index in (H, C) and f.middle in (H, C) and f.ring in (H, C) and f.pinky in (H, C)):
        if td == ThumbState.TOUCHING_INDEX or _tips_touching(lm, THUMB_TIP, INDEX_TIP, 0.06):
            tip_spread = _distance(lm[INDEX_TIP], lm[PINKY_TIP])
            if tip_spread < hand_width * 0.8:
                if palm == PalmFacing.FORWARD:
                    candidates.append(("0", 0.76))

    # ═══════════════════════════════════════════════════════════════════════
    # Thumbs up: Thumb extended, all others closed
    # ═══════════════════════════════════════════════════════════════════════
    if (_closed(f.index) and _closed(f.middle) and _closed(f.ring) and _closed(f.pinky)):
        if td == ThumbState.EXTENDED and _extended(f.thumb):
            # Thumb should point upward
            if lm[THUMB_TIP][1] < lm[THUMB_MCP_IDX][1]:
                candidates.append(("THUMBS_UP", 0.85))

    # ═══════════════════════════════════════════════════════════════════════
    # Select best candidate
    # ═══════════════════════════════════════════════════════════════════════
    if not candidates:
        return None, 0.0

    # Sort by confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


# ═══════════════════════════════════════════════════════════════════════════════
# ISL (Indian Sign Language) specific alphabet additions
# ISL uses a two-handed system for some letters — define additional rules here
# ═══════════════════════════════════════════════════════════════════════════════

def recognize_isl_letter(
    right_hand: Optional[HandAnalysis],
    left_hand: Optional[HandAnalysis] = None,
) -> Tuple[Optional[str], float]:
    """
    Recognize ISL alphabet. ISL shares many one-handed signs with ASL
    but has some unique two-handed signs.
    Falls back to ASL recognition for common signs.
    """
    if right_hand is None:
        return None, 0.0

    # Try ISL-specific signs first (two-handed)
    if left_hand is not None:
        # ISL uses both hands for some letters
        pass  # Extend with ISL-specific two-handed rules as needed

    # Fall back to single-hand ASL-compatible recognition
    return recognize_letter(right_hand)
