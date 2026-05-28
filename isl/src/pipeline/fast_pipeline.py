"""
Optimized real-time pipeline for sign language interpretation.
Minimal latency, maximum throughput — designed for live conversation speed.
Modern glassmorphism HUD overlay for a polished user experience.
"""

import cv2
import numpy as np
import time
import math
import os
from collections import deque
from typing import Optional, Callable, Tuple, List, Dict
import logging

from src.core.hand_tracker import HandTracker, TrackingResult
from src.core.hand_analyzer import HandAnalysis, analyze_hand
from src.core.grammar import correct_grammar
from src.core.translator import TranslationService
from src.core.face_expression import FacialExpressionAnalyzer, ExpressionResult, EMOTION_COLORS

# Use ML engine if available, otherwise fall back to rule-based
try:
    from src.recognition.ml_engine import MLSignEngine as SignEngine, SignResult
except ImportError:
    from src.recognition.sign_engine import SignEngine, SignResult

# Optional: Pillow for Unicode text rendering (translations)
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

logger = logging.getLogger(__name__)

# ── Modern Color Palette ────────────────────────────────────────────────
# All colors are BGR for OpenCV
ACCENT_CYAN     = (255, 217, 0)     # #00D9FF  — primary accent
ACCENT_TEAL     = (200, 180, 0)     # #00B4C8
ACCENT_GREEN    = (130, 235, 0)     # #00EB82
ACCENT_ORANGE   = (50, 170, 255)    # #FFAA32
ACCENT_RED      = (70, 70, 240)     # #F04646
ACCENT_PURPLE   = (247, 85, 168)    # #A855F7
TEXT_WHITE       = (245, 245, 245)
TEXT_DIM         = (160, 160, 160)
TEXT_MUTED       = (100, 100, 100)
PANEL_BG         = (30, 30, 30)
CARD_BG          = (40, 40, 42)


def _rounded_rect(img, pt1, pt2, color, radius, thickness=-1, alpha=1.0):
    """Draw a rounded rectangle, optionally semi-transparent."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    overlay = img.copy() if alpha < 1.0 else img

    # Four corner circles + two rectangles to fill
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, thickness, cv2.LINE_AA)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, thickness, cv2.LINE_AA)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, thickness, cv2.LINE_AA)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, thickness, cv2.LINE_AA)

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _gradient_bar(img, x, y, w, h, progress, color_start, color_end):
    """Draw a horizontal gradient progress bar."""
    filled = int(w * max(0.0, min(1.0, progress)))
    if filled <= 0:
        return
    for i in range(filled):
        t = i / max(w - 1, 1)
        b = int(color_start[0] + (color_end[0] - color_start[0]) * t)
        g = int(color_start[1] + (color_end[1] - color_start[1]) * t)
        r = int(color_start[2] + (color_end[2] - color_start[2]) * t)
        cv2.line(img, (x + i, y), (x + i, y + h), (b, g, r), 1)


def _glow_circle(img, center, radius, color, intensity=0.4):
    """Draw a soft glow circle."""
    overlay = img.copy()
    for r in range(radius + 6, radius, -1):
        alpha_layer = overlay.copy()
        cv2.circle(alpha_layer, center, r, color, 2, cv2.LINE_AA)
        a = intensity * (1.0 - (r - radius) / 6.0)
        cv2.addWeighted(alpha_layer, a, overlay, 1.0 - a, 0, overlay)
    cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    np.copyto(img, overlay)


def _pulse_factor(speed: float = 2.0) -> float:
    """Returns a 0-1 sinusoidal pulse factor based on current time."""
    return (math.sin(time.time() * speed * math.pi) + 1.0) / 2.0


class FastPipeline:
    """
    Ultra-fast real-time pipeline that handles:
    1. Camera capture
    2. Hand tracking (MediaPipe)
    3. Hand analysis (geometric)
    4. Sign recognition (fingerspelling + gestures)
    5. Visualization (modern HUD overlay)
    
    Runs in a single thread for minimal overhead.
    Target: 30+ FPS on modern hardware.
    """

    def __init__(
        self,
        camera_index: int = 0,
        camera_width: int = 1280,
        camera_height: int = 720,
        camera_fps: int = 30,
        use_pose: bool = True,
        show_visualization: bool = True,
    ):
        self.camera_index = camera_index
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.show_visualization = show_visualization

        # Initialize hand tracker (with face mesh enabled)
        self.tracker = HandTracker(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
            model_complexity=1,
            pose_model_complexity=1,
            use_pose=use_pose,
            use_face=True,
        )

        # Facial expression analyzer
        self._face_analyzer = FacialExpressionAnalyzer(smoothing=0.35, history_size=10)
        self._expression: ExpressionResult = ExpressionResult()

        # Initialize sign engine (uses tuned defaults)
        self.engine = SignEngine()

        # FPS tracking
        self.fps_history: deque[float] = deque(maxlen=60)
        self.frame_count = 0

        # State
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None

        # UI animation state
        self._detection_alpha = 0.0   # smooth fade for detection card
        self._last_sign = None        # for transition animation

        # Edit mode state
        self._edit_mode = False
        self._edit_text = ""
        self._edit_cursor = 0
        self._edit_blink = 0.0
        self._speak_flash = 0.0

        # Dialog states (voice / language picker)
        self._voice_dialog = False
        self._voice_dialog_sel = 0      # highlighted index
        self._lang_dialog = False
        self._lang_dialog_sel = 0

        # Grammar correction
        self._grammar_enabled = True
        self._corrected_cache: Dict[str, str] = {}

        # Translation
        self._translator = TranslationService()

        # Conversation history: list of {time, raw, corrected}
        self._conversation_history: List[Dict] = []

        # Unicode font cache
        self._unicode_fonts: Dict[int, object] = {}

    def start(self, on_result: Optional[Callable[[SignResult], None]] = None):
        """
        Start the real-time pipeline. Blocks until user quits.
        
        Args:
            on_result: Optional callback called on every frame with the SignResult.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        # Reduce buffering for lower latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        logger.info("Pipeline started")

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break

                t0 = time.perf_counter()

                # Flip for mirror effect (natural for user)
                frame = cv2.flip(frame, 1)

                # 1. Track hands + pose + face
                tracking = self.tracker.process(frame)

                # 2. Analyze facial expression
                self._expression = self._face_analyzer.analyze(tracking.face)

                # 3. Recognize sign
                sign_result = self.engine.process_frame(tracking)

                # 4. Callback
                if on_result:
                    on_result(sign_result)

                # FPS
                frame_time = time.perf_counter() - t0
                self.fps_history.append(frame_time)
                fps = 1.0 / (np.mean(self.fps_history)) if self.fps_history else 0

                # 5. Visualization
                if self.show_visualization:
                    vis = self._draw_visualization(frame, tracking, sign_result, fps)
                    cv2.namedWindow("Sign Language Interpreter", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Sign Language Interpreter", 1280, 720)
                    cv2.imshow("Sign Language Interpreter", vis)

                    key = cv2.waitKey(1) & 0xFF
                    if self._edit_mode:
                        self._handle_edit_key(key)
                    elif self._voice_dialog:
                        self._handle_voice_dialog_key(key)
                    elif self._lang_dialog:
                        self._handle_lang_dialog_key(key)
                    else:
                        if key == ord('q') or key == 27:  # q or ESC
                            break
                        elif key == ord('c'):
                            self._archive_and_clear()
                        elif key == ord('r'):
                            self._conversation_history.clear()
                            self.engine.reset()
                        elif key == 8:  # Backspace
                            self.engine.backspace()
                        elif key == ord('e'):
                            self._enter_edit_mode()
                        elif key == ord('s'):
                            self._speak_sentence()
                        elif key == ord('v'):
                            self._open_voice_dialog()
                        elif key == ord('t'):
                            self._open_lang_dialog()
                        elif key == ord('g'):
                            self._grammar_enabled = not self._grammar_enabled

                self.frame_count += 1

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.stop()

    def _enter_edit_mode(self):
        """Enter sentence edit mode."""
        self._edit_mode = True
        self._edit_text = self.engine.get_sentence()
        self._edit_cursor = len(self._edit_text)
        logger.info("Edit mode entered")

    def _exit_edit_mode(self, save: bool = True):
        """Exit sentence edit mode, optionally saving changes."""
        if save:
            self.engine.set_sentence(self._edit_text.strip())
            logger.info(f"Sentence updated: {self._edit_text.strip()}")
        self._edit_mode = False
        self._edit_text = ""
        self._edit_cursor = 0

    def _handle_edit_key(self, key: int):
        """Handle a keypress while in edit mode."""
        if key == 255:
            return  # No key pressed
        if key == 13:  # Enter — save & exit
            self._exit_edit_mode(save=True)
        elif key == 27:  # ESC — discard & exit
            self._exit_edit_mode(save=False)
        elif key == 8:  # Backspace
            if self._edit_cursor > 0:
                self._edit_text = (self._edit_text[:self._edit_cursor - 1]
                                   + self._edit_text[self._edit_cursor:])
                self._edit_cursor -= 1
        elif key == 0:  # Special key prefix (arrow keys etc.)
            pass
        elif key == 81 or key == 2:  # Left arrow (platform-dependent)
            self._edit_cursor = max(0, self._edit_cursor - 1)
        elif key == 83 or key == 3:  # Right arrow
            self._edit_cursor = min(len(self._edit_text), self._edit_cursor + 1)
        elif 32 <= key <= 126:  # Printable ASCII
            ch = chr(key)
            self._edit_text = (self._edit_text[:self._edit_cursor]
                               + ch
                               + self._edit_text[self._edit_cursor:])
            self._edit_cursor += 1

    def _speak_sentence(self):
        """Speak the full sentence aloud (uses corrected version if grammar on)."""
        sentence = self.engine.get_sentence()
        if sentence.strip():
            text = self._get_corrected(sentence) if self._grammar_enabled else sentence
            # Clean sign-name artifacts (underscores → spaces)
            text = text.replace("_", " ")
            self.engine.speak_text(text)
            self._speak_flash = time.time()
            logger.info("Speaking full sentence")

    def _archive_and_clear(self):
        """Archive current sentence to conversation history, then clear."""
        sentence = self.engine.get_sentence()
        if sentence.strip():
            corrected = self._get_corrected(sentence) if self._grammar_enabled else sentence
            self._conversation_history.append({
                "time": time.time(),
                "raw": sentence,
                "corrected": corrected,
            })
        self.engine.clear_sentence()
        self._corrected_cache.clear()

    def _get_corrected(self, raw: str) -> str:
        """Get grammar-corrected version (cached)."""
        if raw in self._corrected_cache:
            return self._corrected_cache[raw]
        corrected = correct_grammar(raw)
        self._corrected_cache[raw] = corrected
        return corrected

    # ── Voice Dialog ─────────────────────────────────────────────────────

    def _open_voice_dialog(self):
        """Open the voice selection dialog."""
        from src.core.speaker import VOICE_PRESETS
        self._voice_dialog = True
        if self.engine.speaker is not None:
            self._voice_dialog_sel = self.engine.speaker.preset_index
        else:
            self._voice_dialog_sel = 0

    def _handle_voice_dialog_key(self, key: int):
        """Handle keypresses in the voice dialog."""
        from src.core.speaker import VOICE_PRESETS
        if key == 255:
            return
        n = len(VOICE_PRESETS)
        if key == 27:  # ESC — cancel
            self._voice_dialog = False
        elif key == 13:  # Enter — confirm
            if self.engine.speaker is not None:
                self.engine.speaker.set_voice_by_index(self._voice_dialog_sel)
            self._voice_dialog = False
        elif key in (ord('w'), 82, 0):  # w / Up arrow
            self._voice_dialog_sel = (self._voice_dialog_sel - 1) % n
        elif key in (ord('s'), 84, 1):  # s / Down arrow — be careful, 's' is speak
            self._voice_dialog_sel = (self._voice_dialog_sel + 1) % n
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < n:
                self._voice_dialog_sel = idx
                if self.engine.speaker is not None:
                    self.engine.speaker.set_voice_by_index(idx)
                self._voice_dialog = False

    # ── Language Dialog ──────────────────────────────────────────────────

    def _open_lang_dialog(self):
        """Open the language selection dialog."""
        from src.core.translator import LANGUAGES
        self._lang_dialog = True
        self._lang_dialog_sel = self._translator.lang_index

    def _handle_lang_dialog_key(self, key: int):
        """Handle keypresses in the language dialog."""
        from src.core.translator import LANGUAGES
        if key == 255:
            return
        n = len(LANGUAGES)
        if key == 27:  # ESC — cancel
            self._lang_dialog = False
        elif key == 13:  # Enter — confirm
            self._translator.set_language_by_index(self._lang_dialog_sel)
            self._unicode_fonts.clear()
            self._lang_dialog = False
        elif key in (ord('w'), 82, 0):  # Up
            self._lang_dialog_sel = (self._lang_dialog_sel - 1) % n
        elif key in (ord('s'), 84, 1):  # Down
            self._lang_dialog_sel = (self._lang_dialog_sel + 1) % n
        elif ord('0') <= key <= ord('9'):
            # 1–9 maps to indices 0–8, 0 maps to 9
            idx = (key - ord('1')) if key != ord('0') else 9
            if 0 <= idx < n:
                self._lang_dialog_sel = idx
                self._translator.set_language_by_index(idx)
                self._unicode_fonts.clear()
                self._lang_dialog = False

    def _get_unicode_font(self, size: int = 18):
        """Get a cached Unicode-capable font for PIL rendering."""
        if size in self._unicode_fonts:
            return self._unicode_fonts[size]

        if not _PIL_OK:
            return None

        # Try language-specific fonts, then fallback
        candidates = self._translator.get_font_candidates()
        font_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")

        for name in candidates:
            for path in [name, os.path.join(font_dir, name)]:
                try:
                    font = ImageFont.truetype(path, size)
                    self._unicode_fonts[size] = font
                    return font
                except Exception:
                    continue

        # Ultimate fallback
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        self._unicode_fonts[size] = font
        return font

    def _put_unicode_text(self, img, text, pos, font_size=18, color=TEXT_WHITE):
        """Render Unicode text on an OpenCV image using Pillow."""
        if not _PIL_OK or not text:
            # Fallback to cv2 (ASCII only)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
            return

        try:
            font = self._get_unicode_font(font_size)
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            rgb_color = (color[2], color[1], color[0])
            draw.text(pos, text, font=font, fill=rgb_color)
            result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            np.copyto(img, result)
        except Exception:
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)

    def stop(self):
        """Stop the pipeline and release resources."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.tracker.release()
        self._translator.shutdown()
        self.engine.shutdown()
        logger.info(f"Pipeline stopped. Processed {self.frame_count} frames.")

    # ── Visualization ────────────────────────────────────────────────────

    def _draw_visualization(
        self,
        frame: np.ndarray,
        tracking: TrackingResult,
        sign_result: SignResult,
        fps: float,
    ) -> np.ndarray:
        """Create a modern glassmorphism HUD overlay on the camera feed."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Slight vignette for cinematic feel
        self._apply_vignette(vis)

        # Draw hand skeletons with glow
        self._draw_hands(vis, tracking)

        # Draw pose arms
        self._draw_arms(vis, tracking)

        # Smooth detection alpha for card fade-in/out
        if sign_result.sign is not None:
            self._detection_alpha = min(1.0, self._detection_alpha + 0.15)
            self._last_sign = sign_result.sign
        else:
            self._detection_alpha = max(0.0, self._detection_alpha - 0.08)

        # ── Top Status Bar (glass) ───────────────────────────────────────
        self._draw_top_bar(vis, fps, tracking)

        # ── Chat History (left side, scrolling) ──────────────────────────
        if self._conversation_history:
            self._draw_chat_history(vis)

        # ── Detection Card (right side, floating) ────────────────────────
        self._draw_detection_card(vis, sign_result)

        # ── Sentence Bar (bottom, floating glass) ────────────────────────
        # Submit current sentence for translation in background
        sentence = self.engine.get_sentence()
        if sentence.strip() and self._translator.is_active:
            corrected = self._get_corrected(sentence) if self._grammar_enabled else sentence
            self._translator.translate(corrected)
        self._draw_sentence_bar(vis)

        # ── Controls Hint (bottom-left, subtle) ──────────────────────────
        self._draw_controls_hint(vis)

        # ── Hand Status Indicator (bottom-right) ─────────────────────────
        self._draw_hand_status(vis, tracking)

        # ── Facial Expression Widget (top-left, subtle) ──────────────────
        self._draw_emotion_widget(vis)

        # ── Speak Flash (brief green tint when speaking) ─────────────────
        if self._speak_flash > 0:
            elapsed = time.time() - self._speak_flash
            if elapsed < 0.6:
                flash_alpha = 0.15 * (1.0 - elapsed / 0.6)
                flash_layer = np.full_like(vis, ACCENT_GREEN, dtype=np.uint8)
                cv2.addWeighted(flash_layer, flash_alpha, vis, 1.0 - flash_alpha, 0, vis)

        # ── Edit Mode Overlay ────────────────────────────────────────────
        if self._edit_mode:
            self._draw_edit_overlay(vis)

        # ── Voice / Language Dialog Overlays ─────────────────────────────
        if self._voice_dialog:
            self._draw_voice_dialog(vis)
        if self._lang_dialog:
            self._draw_lang_dialog(vis)

        return vis

    def _apply_vignette(self, frame: np.ndarray):
        """Apply a subtle vignette effect for depth."""
        h, w = frame.shape[:2]
        X = cv2.getGaussianKernel(w, w * 0.6)
        Y = cv2.getGaussianKernel(h, h * 0.6)
        M = Y * X.T
        M = M / M.max()
        # Blend — keep 70% original at edges
        M = 0.3 + 0.7 * M
        for i in range(3):
            frame[:, :, i] = (frame[:, :, i] * M).astype(np.uint8)

    def _draw_glass_panel(self, img, x, y, w, h, radius=16, alpha=0.55, border_color=None):
        """Draw a frosted glass panel (semi-transparent dark background with border)."""
        overlay = img.copy()
        _rounded_rect(overlay, (x, y), (x + w, y + h), PANEL_BG, radius, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        # Subtle border
        border = border_color or (80, 80, 80)
        _rounded_rect(img, (x, y), (x + w, y + h), border, radius, 1)

    def _draw_top_bar(self, vis, fps, tracking):
        """Modern top status bar with voice + language indicators."""
        h, w = vis.shape[:2]
        bar_w = min(820, w - 40)
        bar_h = 44
        bx = (w - bar_w) // 2
        by = 14

        self._draw_glass_panel(vis, bx, by, bar_w, bar_h, radius=22, alpha=0.6)

        # Title
        cv2.putText(vis, "SIGN LANGUAGE INTERPRETER", (bx + 20, by + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_WHITE, 1, cv2.LINE_AA)

        # Separator
        sep_x = bx + 295
        cv2.circle(vis, (sep_x, by + 24), 2, TEXT_MUTED, -1, cv2.LINE_AA)

        # FPS indicator
        fps_color = ACCENT_GREEN if fps > 24 else (ACCENT_ORANGE if fps > 15 else ACCENT_RED)
        cv2.putText(vis, f"{fps:.0f}", (sep_x + 10, by + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, fps_color, 1, cv2.LINE_AA)

        # Separator
        sep2_x = sep_x + 50
        cv2.circle(vis, (sep2_x, by + 24), 2, TEXT_MUTED, -1, cv2.LINE_AA)

        # Voice persona badge
        voice_name = "Male"
        if self.engine.speaker is not None:
            voice_name = self.engine.speaker.current_voice_name
        voice_color = ACCENT_PURPLE
        cv2.putText(vis, voice_name, (sep2_x + 10, by + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, voice_color, 1, cv2.LINE_AA)

        # Separator
        sep3_x = sep2_x + 60
        cv2.circle(vis, (sep3_x, by + 24), 2, TEXT_MUTED, -1, cv2.LINE_AA)

        # Language badge
        lang_name = self._translator.current_name
        lang_color = ACCENT_CYAN if self._translator.is_active else TEXT_MUTED
        cv2.putText(vis, lang_name, (sep3_x + 10, by + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, lang_color, 1, cv2.LINE_AA)

        # Grammar indicator
        sep4_x = sep3_x + 75
        cv2.circle(vis, (sep4_x, by + 24), 2, TEXT_MUTED, -1, cv2.LINE_AA)
        g_label = "Grammar ON" if self._grammar_enabled else "Grammar OFF"
        g_color = ACCENT_GREEN if self._grammar_enabled else TEXT_MUTED
        cv2.putText(vis, g_label, (sep4_x + 10, by + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, g_color, 1, cv2.LINE_AA)

        # Mood indicator (from face expression)
        sep5_x = sep4_x + 90
        cv2.circle(vis, (sep5_x, by + 24), 2, TEXT_MUTED, -1, cv2.LINE_AA)
        if self._expression.valid:
            mood_color = EMOTION_COLORS.get(self._expression.emotion, TEXT_DIM)
            mood_label = self._expression.emotion
            cv2.putText(vis, mood_label, (sep5_x + 10, by + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, mood_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, "No Face", (sep5_x + 10, by + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, TEXT_MUTED, 1, cv2.LINE_AA)

        # Live dot (pulsing)
        pulse = _pulse_factor(1.5)
        dot_radius = int(4 + 2 * pulse)
        live_x = bx + bar_w - 50
        cv2.circle(vis, (live_x, by + 22), dot_radius, ACCENT_GREEN, -1, cv2.LINE_AA)
        cv2.putText(vis, "LIVE", (live_x + 10, by + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, ACCENT_GREEN, 1, cv2.LINE_AA)

    def _draw_detection_card(self, vis, sign_result: SignResult):
        """Floating detection card on the right side of the frame."""
        h, w = vis.shape[:2]
        card_w = 280
        card_h = 180
        cx = w - card_w - 20
        cy = 75

        if self._detection_alpha <= 0.01 and sign_result.sign is None:
            # Draw idle card
            self._draw_glass_panel(vis, cx, cy, card_w, card_h, radius=14, alpha=0.35)
            # Empty state icon
            icon_y = cy + card_h // 2 - 10
            cv2.putText(vis, "Show a sign...", (cx + 50, icon_y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MUTED, 1, cv2.LINE_AA)
            # Animated hand icon (simple wave)
            wave_offset = int(4 * math.sin(time.time() * 3))
            hand_cx = cx + 28
            hand_cy = icon_y + 2 + wave_offset
            cv2.putText(vis, "?", (hand_cx, hand_cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_DIM, 2, cv2.LINE_AA)
            return

        # Active detection card
        sign_display = sign_result.sign or self._last_sign or ""
        if not sign_display:
            return

        is_stable = sign_result.is_stable if sign_result.sign else False
        conf = sign_result.confidence if sign_result.sign else 0.0
        stype = (sign_result.sign_type or "").upper()

        # Card border color based on stability
        border_color = ACCENT_GREEN if is_stable else ACCENT_CYAN
        effective_alpha = 0.6 * self._detection_alpha

        self._draw_glass_panel(vis, cx, cy, card_w, card_h, radius=14,
                               alpha=effective_alpha, border_color=border_color)

        # Accent glow line at top of card
        glow_overlay = vis.copy()
        cv2.line(glow_overlay, (cx + 14, cy + 2), (cx + card_w - 14, cy + 2),
                 border_color, 2, cv2.LINE_AA)
        cv2.addWeighted(glow_overlay, 0.7 * self._detection_alpha, vis,
                        1 - 0.7 * self._detection_alpha, 0, vis)

        # "DETECTED" label
        label_y = cy + 30
        cv2.putText(vis, "DETECTED", (cx + 16, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)

        # Type badge (pill shape)
        if stype:
            badge_w = len(stype) * 10 + 16
            badge_x = cx + card_w - badge_w - 14
            _rounded_rect(vis, (badge_x, label_y - 12), (badge_x + badge_w, label_y + 4),
                          (60, 60, 65), 8, -1)
            cv2.putText(vis, stype, (badge_x + 8, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, TEXT_DIM, 1, cv2.LINE_AA)

        # Sign name (big)
        sign_y = cy + 72
        font_scale = 1.1 if len(sign_display) <= 5 else 0.75
        text_color = tuple(int(c * (0.6 + 0.4 * self._detection_alpha)) for c in border_color)
        cv2.putText(vis, sign_display, (cx + 18, sign_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

        # Confidence bar
        bar_y = cy + 95
        bar_x = cx + 16
        bar_w = card_w - 32
        bar_h_px = 8
        # Background track
        _rounded_rect(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px),
                      (55, 55, 60), 4, -1)
        # Filled portion
        _gradient_bar(vis, bar_x, bar_y, bar_w, bar_h_px, conf,
                      ACCENT_TEAL, ACCENT_CYAN)
        # Confidence text
        cv2.putText(vis, f"{conf:.0%}", (bar_x + bar_w + 6, bar_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DIM, 1, cv2.LINE_AA)

        # Hold timer
        if sign_result.sign and sign_result.hold_duration > 0:
            hold = sign_result.hold_duration
            hold_needed = getattr(self.engine, 'sign_hold_time', getattr(self.engine, 'letter_hold_time', 1.5))
            progress = min(hold / hold_needed, 1.0)

            timer_y = cy + 120
            cv2.putText(vis, "HOLD", (bar_x, timer_y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)
            # Timer bar
            timer_bar_x = bar_x + 42
            timer_bar_w = bar_w - 42
            _rounded_rect(vis, (timer_bar_x, timer_y - 5),
                          (timer_bar_x + timer_bar_w, timer_y + 5), (50, 50, 55), 5, -1)

            if progress >= 1.0:
                pulse = _pulse_factor(3.0)
                fill_color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in ACCENT_GREEN)
            else:
                fill_color = ACCENT_ORANGE

            filled_w = int(timer_bar_w * progress)
            if filled_w > 0:
                _rounded_rect(vis, (timer_bar_x, timer_y - 5),
                              (timer_bar_x + filled_w, timer_y + 5), fill_color, 5, -1)

            if progress >= 1.0:
                cv2.putText(vis, "COMMITTED!", (timer_bar_x + timer_bar_w + 6, timer_y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, ACCENT_GREEN, 1, cv2.LINE_AA)

        # Stability indicator
        status_y = cy + card_h - 20
        if is_stable:
            cv2.circle(vis, (cx + 22, status_y), 5, ACCENT_GREEN, -1, cv2.LINE_AA)
            cv2.putText(vis, "Stable", (cx + 34, status_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, ACCENT_GREEN, 1, cv2.LINE_AA)
        else:
            pulse = _pulse_factor(2.0)
            dot_col = tuple(int(c * (0.5 + 0.5 * pulse)) for c in ACCENT_ORANGE)
            cv2.circle(vis, (cx + 22, status_y), 5, dot_col, -1, cv2.LINE_AA)
            cv2.putText(vis, "Tracking...", (cx + 34, status_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, ACCENT_ORANGE, 1, cv2.LINE_AA)

    def _draw_sentence_bar(self, vis):
        """Floating glass sentence bar with grammar correction + translation."""
        h, w = vis.shape[:2]
        sentence = self.engine.get_sentence()

        # Dynamic height based on active features
        has_grammar = self._grammar_enabled and sentence.strip()
        has_translation = self._translator.is_active and sentence.strip()
        bar_h = 42  # base
        if has_grammar:
            bar_h += 22
        if has_translation:
            bar_h += 24
        if sentence.strip():
            bar_h += 18  # for committed signs trail

        margin = 20
        bar_w = w - 2 * margin
        bx = margin
        by = h - bar_h - margin

        self._draw_glass_panel(vis, bx, by, bar_w, bar_h, radius=14, alpha=0.6)

        # Raw label
        cv2.putText(vis, "SENTENCE", (bx + 16, by + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.line(vis, (bx + 95, by + 10), (bx + 95, by + 22), (60, 60, 65), 1, cv2.LINE_AA)

        y_cursor = by + 18

        if sentence:
            max_chars = (bar_w - 120) // 10
            display = sentence if len(sentence) <= max_chars else "..." + sentence[-max_chars:]
            cv2.putText(vis, display, (bx + 105, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM if has_grammar else TEXT_WHITE,
                        1, cv2.LINE_AA)
            y_cursor += 20

            # Grammar-corrected version
            if has_grammar:
                corrected = self._get_corrected(sentence)
                if corrected != sentence:
                    max_gc = (bar_w - 120) // 10
                    gc_display = corrected if len(corrected) <= max_gc else "..." + corrected[-max_gc:]
                    cv2.putText(vis, gc_display, (bx + 105, y_cursor),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_GREEN, 1, cv2.LINE_AA)
                    # Small label
                    cv2.putText(vis, "CORRECTED", (bx + 16, y_cursor),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, ACCENT_GREEN, 1, cv2.LINE_AA)
                y_cursor += 20

            # Translation
            if has_translation:
                translated = self._translator.get_result()
                if translated:
                    lang_flag = self._translator.current_flag
                    cv2.putText(vis, lang_flag, (bx + 16, y_cursor),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, ACCENT_CYAN, 1, cv2.LINE_AA)
                    # Use PIL for non-ASCII, cv2 for ASCII
                    if all(ord(c) < 128 for c in translated):
                        max_tc = (bar_w - 120) // 10
                        t_display = translated if len(translated) <= max_tc else "..." + translated[-max_tc:]
                        cv2.putText(vis, t_display, (bx + 55, y_cursor),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_CYAN, 1, cv2.LINE_AA)
                    else:
                        self._put_unicode_text(vis, translated, (bx + 55, y_cursor - 14),
                                               font_size=16, color=ACCENT_CYAN)
                else:
                    cv2.putText(vis, "Translating...", (bx + 55, y_cursor),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_MUTED, 1, cv2.LINE_AA)
                y_cursor += 22

            # Committed signs trail
            signs = self.engine.get_committed_signs()
            if signs:
                signs_text = " > ".join(signs[-8:])
                max_sc = (bar_w - 40) // 8
                if len(signs_text) > max_sc:
                    signs_text = "..." + signs_text[-max_sc:]
                cv2.putText(vis, signs_text, (bx + 16, y_cursor),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_MUTED, 1, cv2.LINE_AA)

            # Speaking indicator
            if self._speak_flash > 0 and (time.time() - self._speak_flash) < 1.5:
                pulse = _pulse_factor(4.0)
                spk_col = tuple(int(c * (0.5 + 0.5 * pulse)) for c in ACCENT_GREEN)
                cv2.putText(vis, "Speaking...", (bx + bar_w - 100, by + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, spk_col, 1, cv2.LINE_AA)

            # Word count
            wc = len(sentence.split())
            cv2.putText(vis, f"{wc}w", (bx + bar_w - 30, by + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_MUTED, 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, "Waiting for signs...", (bx + 105, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUTED, 1, cv2.LINE_AA)

    def _draw_controls_hint(self, vis):
        """Subtle controls hint above the sentence bar."""
        h, w = vis.shape[:2]
        # Two rows of hints
        row1 = [
            ("Q", "Quit"), ("C", "Clear"), ("R", "Reset"),
            ("<-", "Undo"), ("E", "Edit"), ("S", "Speak"),
        ]
        row2 = [
            ("V", "Voice"), ("T", "Lang"), ("G", "Grammar"),
        ]
        for row_idx, hints in enumerate([row1, row2]):
            hx = 25
            hy = h - 130 + row_idx * 22
            for key, action in hints:
                kw = len(key) * 9 + 12
                _rounded_rect(vis, (hx, hy - 9), (hx + kw, hy + 7), (55, 55, 60), 5, -1)
                cv2.putText(vis, key, (hx + 6, hy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_DIM, 1, cv2.LINE_AA)
                cv2.putText(vis, action, (hx + kw + 4, hy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_MUTED, 1, cv2.LINE_AA)
                hx += kw + len(action) * 6 + 18

    def _draw_hand_status(self, vis, tracking):
        """Hand connection status indicator, bottom-right."""
        h, w = vis.shape[:2]
        sx = w - 170
        sy = h - 100

        if tracking.has_right_hand and tracking.has_left_hand:
            status = "Both Hands"
            color = ACCENT_GREEN
        elif tracking.has_right_hand:
            status = "Right Hand"
            color = ACCENT_CYAN
        elif tracking.has_left_hand:
            status = "Left Hand"
            color = ACCENT_CYAN
        else:
            status = "No Hands"
            color = TEXT_MUTED

        # Status pill
        pill_w = 140
        pill_h = 24
        _rounded_rect(vis, (sx, sy - 6), (sx + pill_w, sy + pill_h - 6),
                      (45, 45, 50), 12, -1, alpha=0.5)

        # Dot
        dot_color = color
        if not tracking.has_hands:
            pulse = _pulse_factor(1.0)
            dot_color = tuple(int(c * (0.4 + 0.6 * pulse)) for c in TEXT_MUTED)

        cv2.circle(vis, (sx + 14, sy + 6), 5, dot_color, -1, cv2.LINE_AA)
        cv2.putText(vis, status, (sx + 26, sy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    def _draw_emotion_widget(self, vis):
        """
        Subtle floating emotion indicator — top-left corner.
        Shows detected emotion with emoji, label, confidence bar,
        and a mini breakdown of top emotions.
        """
        expr = self._expression
        if not expr.valid:
            return

        h, w = vis.shape[:2]
        # Position: below conversation history panel (or top-left if no history)
        if self._conversation_history:
            px = 14
            # Place below history panel
            max_msgs = min(6, len(self._conversation_history))
            py = 70 + max_msgs * 38 + 32 + 16
        else:
            px = 14
            py = 70

        widget_w = 200
        widget_h = 110

        # Ensure we don't go off-screen
        if py + widget_h > h - 180:
            py = h - 180 - widget_h

        # ── Glass panel ──────────────────────────────────────────────────
        emotion_color = EMOTION_COLORS.get(expr.emotion, TEXT_DIM)
        self._draw_glass_panel(vis, px, py, widget_w, widget_h,
                               radius=12, alpha=0.45, border_color=emotion_color)

        # ── Top accent line ──────────────────────────────────────────────
        glow_overlay = vis.copy()
        cv2.line(glow_overlay, (px + 12, py + 2), (px + widget_w - 12, py + 2),
                 emotion_color, 2, cv2.LINE_AA)
        cv2.addWeighted(glow_overlay, 0.5, vis, 0.5, 0, vis)

        # ── Header: "MOOD" label ─────────────────────────────────────────
        cv2.putText(vis, "MOOD", (px + 12, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)

        # ── Emoji + Emotion name ─────────────────────────────────────────
        # Emoji via PIL (Unicode), fallback to text
        emoji_x = px + 14
        emoji_y = py + 28
        if _PIL_OK:
            self._put_unicode_text(vis, expr.emoji, (emoji_x, emoji_y),
                                   font_size=22, color=emotion_color)
        else:
            cv2.putText(vis, expr.emotion[0], (emoji_x, emoji_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2, cv2.LINE_AA)

        # Emotion label
        label_x = emoji_x + 32
        cv2.putText(vis, expr.emotion, (label_x, py + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, emotion_color, 1, cv2.LINE_AA)

        # Confidence percentage
        conf_text = f"{expr.confidence:.0%}"
        cv2.putText(vis, conf_text, (px + widget_w - 46, py + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

        # ── Main confidence bar ──────────────────────────────────────────
        bar_x = px + 14
        bar_y = py + 56
        bar_w = widget_w - 28
        bar_h = 6

        # Track background
        _rounded_rect(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 55), 3, -1)
        # Filled
        _gradient_bar(vis, bar_x, bar_y, bar_w, bar_h, expr.confidence,
                      emotion_color, tuple(min(255, c + 60) for c in emotion_color))

        # ── Mini emotion breakdown (top 3) ───────────────────────────────
        if expr.scores:
            sorted_emotions = sorted(expr.scores.items(), key=lambda x: x[1], reverse=True)
            top3 = sorted_emotions[:3]
            mini_y = py + 72
            for i, (emo, score) in enumerate(top3):
                emo_col = EMOTION_COLORS.get(emo, TEXT_MUTED)
                is_primary = (emo == expr.emotion)

                # Mini label
                label = emo[:3].upper()  # truncate to 3 chars
                cv2.putText(vis, label, (bar_x, mini_y + i * 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22,
                            emo_col if is_primary else TEXT_MUTED, 1, cv2.LINE_AA)

                # Mini bar
                mini_bar_x = bar_x + 30
                mini_bar_w = bar_w - 46
                mini_bar_h = 4
                my = mini_y + i * 13 - 4

                _rounded_rect(vis, (mini_bar_x, my), (mini_bar_x + mini_bar_w, my + mini_bar_h),
                              (40, 40, 44), 2, -1)
                filled = int(mini_bar_w * score)
                if filled > 0:
                    bar_col = emo_col if is_primary else tuple(c // 2 for c in emo_col)
                    _rounded_rect(vis, (mini_bar_x, my), (mini_bar_x + filled, my + mini_bar_h),
                                  bar_col, 2, -1)

                # Score
                cv2.putText(vis, f"{score:.0%}", (mini_bar_x + mini_bar_w + 4, mini_y + i * 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.2, TEXT_MUTED, 1, cv2.LINE_AA)

    def _draw_edit_overlay(self, vis):
        """Draw a polished full-screen edit overlay for sentence editing."""
        h, w = vis.shape[:2]

        # Dim background with blur-like tint
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 14), -1)
        cv2.addWeighted(overlay, 0.60, vis, 0.40, 0, vis)

        # Central card — generous sizing
        card_w = min(760, w - 60)
        card_h = 280
        cx = (w - card_w) // 2
        cy = (h - card_h) // 2

        # Card background
        self._draw_glass_panel(vis, cx, cy, card_w, card_h, radius=20,
                               alpha=0.90, border_color=ACCENT_CYAN)

        # Top accent gradient bar
        for i in range(3):
            progress = i / 3.0
            color = tuple(int(ACCENT_CYAN[c] * (1.0 - progress) + ACCENT_PURPLE[c] * progress)
                          for c in range(3))
            x1 = cx + 20 + int((card_w - 40) * progress / 1)
            x2 = cx + 20 + int((card_w - 40) * (progress + 0.34))
            cv2.line(vis, (x1, cy + 2), (min(x2, cx + card_w - 20), cy + 2),
                     color, 3, cv2.LINE_AA)

        # ── Header row ──────────────────────────────────────
        # Icon (pencil-like indicator)
        cv2.rectangle(vis, (cx + 22, cy + 18), (cx + 28, cy + 32),
                      ACCENT_CYAN, -1, cv2.LINE_AA)
        cv2.rectangle(vis, (cx + 24, cy + 14), (cx + 26, cy + 18),
                      ACCENT_CYAN, -1, cv2.LINE_AA)

        cv2.putText(vis, "EDIT MODE", (cx + 38, cy + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_WHITE, 1, cv2.LINE_AA)

        # Word & char counts (right side of header)
        words = len(self._edit_text.split()) if self._edit_text.strip() else 0
        chars = len(self._edit_text)
        stats = f"{words} word{'s' if words != 1 else ''}  |  {chars} char{'s' if chars != 1 else ''}"
        sw = len(stats) * 7
        cv2.putText(vis, stats, (cx + card_w - sw - 24, cy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, TEXT_MUTED, 1, cv2.LINE_AA)

        # Separator
        cv2.line(vis, (cx + 16, cy + 44), (cx + card_w - 16, cy + 44),
                 (55, 55, 60), 1, cv2.LINE_AA)

        # ── Label ────────────────────────────────────────────
        cv2.putText(vis, "Type your sentence below:", (cx + 22, cy + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, TEXT_MUTED, 1, cv2.LINE_AA)

        # ── Input field ──────────────────────────────────────
        field_x = cx + 20
        field_y = cy + 74
        field_w = card_w - 40
        field_h = 54

        # Field background with subtle inner shadow
        _rounded_rect(vis, (field_x, field_y), (field_x + field_w, field_y + field_h),
                      (18, 18, 22), 12, -1)
        # Top inner shadow
        cv2.line(vis, (field_x + 12, field_y + 1), (field_x + field_w - 12, field_y + 1),
                 (10, 10, 14), 1, cv2.LINE_AA)
        # Border (brighter on focus)
        _rounded_rect(vis, (field_x, field_y), (field_x + field_w, field_y + field_h),
                      ACCENT_CYAN, 12, 1)

        # Text with cursor
        display_text = self._edit_text
        cursor_pos = self._edit_cursor

        char_px = 11  # approximate pixel width per character
        max_visible = (field_w - 30) // char_px
        if len(display_text) > max_visible:
            start = max(0, cursor_pos - max_visible + 5)
            end = start + max_visible
            if end > len(display_text):
                end = len(display_text)
                start = max(0, end - max_visible)
            visible = display_text[start:end]
            cursor_in_visible = cursor_pos - start
        else:
            visible = display_text
            cursor_in_visible = cursor_pos

        text_x = field_x + 14
        text_y = field_y + 36
        cv2.putText(vis, visible, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_WHITE, 1, cv2.LINE_AA)

        # Placeholder when empty
        if not display_text:
            cv2.putText(vis, "Start typing...", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 90), 1, cv2.LINE_AA)

        # Blinking cursor with glow
        if int(time.time() * 2.5) % 2 == 0:
            cursor_x = text_x + cursor_in_visible * char_px
            # Cursor glow
            glow_overlay = vis.copy()
            cv2.line(glow_overlay, (cursor_x, field_y + 10),
                     (cursor_x, field_y + field_h - 10), ACCENT_CYAN, 4, cv2.LINE_AA)
            cv2.addWeighted(glow_overlay, 0.25, vis, 0.75, 0, vis)
            # Solid cursor
            cv2.line(vis, (cursor_x, field_y + 10),
                     (cursor_x, field_y + field_h - 10), ACCENT_CYAN, 2, cv2.LINE_AA)

        # ── Preview (grammar-corrected) ──────────────────────
        if self._grammar_enabled and self._edit_text.strip():
            preview = self._get_corrected(self._edit_text)
            if preview != self._edit_text:
                pvy = field_y + field_h + 16
                cv2.putText(vis, "Grammar Preview:", (cx + 22, pvy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, ACCENT_GREEN, 1, cv2.LINE_AA)
                max_prev = (card_w - 60) // 8
                ptext = preview if len(preview) <= max_prev else preview[:max_prev - 2] + ".."
                cv2.putText(vis, ptext, (cx + 130, pvy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 230, 180), 1, cv2.LINE_AA)

        # ── Action buttons ───────────────────────────────────
        btn_y = cy + card_h - 52
        buttons = [
            ("ENTER", "Save & Close", ACCENT_GREEN, True),
            ("ESC", "Discard", ACCENT_ORANGE, False),
            ("BKSP", "Delete Char", TEXT_DIM, False),
            ("<- / ->", "Move Cursor", TEXT_DIM, False),
        ]
        bx = cx + 20
        for key, label, color, primary in buttons:
            kw = len(key) * 8 + 16
            # Button background
            bg = (40, 70, 50) if primary else (50, 50, 55)
            _rounded_rect(vis, (bx, btn_y - 12), (bx + kw, btn_y + 12), bg, 8, -1)
            _rounded_rect(vis, (bx, btn_y - 12), (bx + kw, btn_y + 12),
                          color if primary else (75, 75, 80), 8, 1)
            cv2.putText(vis, key, (bx + 8, btn_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)
            cv2.putText(vis, label, (bx + kw + 8, btn_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, TEXT_MUTED, 1, cv2.LINE_AA)
            bx += kw + len(label) * 6 + 28

    # ── Dialog overlays ──────────────────────────────────────────────────

    def _draw_picker_dialog(self, vis, title: str, items, selected: int,
                            accent=ACCENT_PURPLE):
        """Generic picker dialog — used for voice and language selection."""
        h, w = vis.shape[:2]

        # Dim background
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.50, vis, 0.50, 0, vis)

        n = len(items)
        row_h = 42
        card_w = min(420, w - 80)
        card_h = 64 + n * row_h + 20
        cx = (w - card_w) // 2
        cy = (h - card_h) // 2

        # Card
        self._draw_glass_panel(vis, cx, cy, card_w, card_h, radius=18,
                               alpha=0.88, border_color=accent)

        # Accent glow at top
        glow = vis.copy()
        cv2.line(glow, (cx + 18, cy + 2), (cx + card_w - 18, cy + 2),
                 accent, 3, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.5, vis, 0.5, 0, vis)

        # Title
        cv2.putText(vis, title, (cx + 24, cy + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent, 1, cv2.LINE_AA)

        # Thin separator
        cv2.line(vis, (cx + 16, cy + 44), (cx + card_w - 16, cy + 44),
                 (60, 60, 65), 1, cv2.LINE_AA)

        # Items
        iy = cy + 60
        for idx, (label, desc) in enumerate(items):
            is_sel = idx == selected
            # Highlight bar
            if is_sel:
                _rounded_rect(vis, (cx + 10, iy - 4), (cx + card_w - 10, iy + row_h - 10),
                              accent, 8, -1, alpha=0.18)
                # Selection indicator
                cv2.circle(vis, (cx + 26, iy + 14), 6, accent, -1, cv2.LINE_AA)
                cv2.circle(vis, (cx + 26, iy + 14), 3, (25, 25, 28), -1, cv2.LINE_AA)
            else:
                cv2.circle(vis, (cx + 26, iy + 14), 6, (85, 85, 90), 1, cv2.LINE_AA)

            # Shortcut number
            num = str(idx + 1) if idx < 9 else "0"
            if idx < 10:
                cv2.putText(vis, num, (cx + 46, iy + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            accent if is_sel else TEXT_MUTED, 1, cv2.LINE_AA)

            # Label
            cv2.putText(vis, label, (cx + 64, iy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        TEXT_WHITE if is_sel else TEXT_DIM, 1, cv2.LINE_AA)

            # Description (right-aligned)
            if desc:
                dw = len(desc) * 8
                cv2.putText(vis, desc, (cx + card_w - dw - 20, iy + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            TEXT_MUTED, 1, cv2.LINE_AA)

            iy += row_h

        # Bottom hint bar
        hy = cy + card_h - 18
        hints = [("ENTER", "Select", ACCENT_GREEN), ("ESC", "Cancel", ACCENT_ORANGE),
                 ("W/S", "Navigate", TEXT_DIM)]
        hx = cx + 20
        for key, lbl, c in hints:
            kw = len(key) * 8 + 12
            _rounded_rect(vis, (hx, hy - 10), (hx + kw, hy + 8), (55, 55, 60), 5, -1)
            cv2.putText(vis, key, (hx + 6, hy + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, c, 1, cv2.LINE_AA)
            cv2.putText(vis, lbl, (hx + kw + 4, hy + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_MUTED, 1, cv2.LINE_AA)
            hx += kw + len(lbl) * 6 + 18

    def _draw_voice_dialog(self, vis):
        """Draw the voice selection dialog."""
        from src.core.speaker import VOICE_PRESETS
        items = []
        for p in VOICE_PRESETS:
            rate_desc = f"Rate: {p['rate']}"
            items.append((p["name"], rate_desc))
        self._draw_picker_dialog(vis, "SELECT VOICE", items,
                                 self._voice_dialog_sel, accent=ACCENT_PURPLE)

    def _draw_lang_dialog(self, vis):
        """Draw the language selection dialog."""
        from src.core.translator import LANGUAGES
        items = []
        for code, name, flag in LANGUAGES:
            tag = flag if flag else "--"
            items.append((name, tag))
        self._draw_picker_dialog(vis, "SELECT LANGUAGE", items,
                                 self._lang_dialog_sel, accent=ACCENT_CYAN)

    def _draw_chat_history(self, vis):
        """Draw scrolling conversation history on the left side."""
        h, w = vis.shape[:2]
        panel_w = 260
        panel_margin = 14
        px = panel_margin
        py = 70

        # Show last N messages that fit
        max_messages = 6
        messages = self._conversation_history[-max_messages:]
        if not messages:
            return

        # Calculate panel height
        msg_h = 38
        panel_h = len(messages) * msg_h + 32

        # Glass panel
        self._draw_glass_panel(vis, px, py, panel_w, panel_h, radius=12, alpha=0.45)

        # Header
        cv2.putText(vis, "HISTORY", (px + 12, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(vis, f"({len(self._conversation_history)})",
                    (px + 75, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.line(vis, (px + 10, py + 24), (px + panel_w - 10, py + 24),
                 (60, 60, 65), 1, cv2.LINE_AA)

        # Messages
        my = py + 38
        for msg in messages:
            # Timestamp
            t = time.localtime(msg["time"])
            ts = f"{t.tm_hour:02d}:{t.tm_min:02d}"
            cv2.putText(vis, ts, (px + 10, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, TEXT_MUTED, 1, cv2.LINE_AA)

            # Use corrected text if available, else raw
            text = msg.get("corrected", msg.get("raw", ""))
            max_c = (panel_w - 60) // 7
            if len(text) > max_c:
                text = text[:max_c - 2] + ".."

            cv2.putText(vis, text, (px + 50, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_WHITE, 1, cv2.LINE_AA)

            # Thin separator
            cv2.line(vis, (px + 10, my + 10), (px + panel_w - 10, my + 10),
                     (50, 50, 55), 1, cv2.LINE_AA)
            my += msg_h

    def _draw_hands(self, frame: np.ndarray, tracking: TrackingResult):
        """Draw hand landmarks with glow-enhanced finger-colored skeleton."""
        h, w = frame.shape[:2]

        FINGER_COLORS = {
            "thumb": (100, 100, 255),    # Warm red
            "index": (50, 180, 255),     # Orange
            "middle": (50, 235, 255),    # Yellow
            "ring": (80, 230, 130),      # Green
            "pinky": (230, 120, 230),    # Magenta
            "palm": (170, 170, 180),     # Light gray
        }

        GLOW_COLORS = {
            "thumb": (80, 80, 200),
            "index": (30, 140, 200),
            "middle": (30, 190, 200),
            "ring": (50, 180, 100),
            "pinky": (180, 90, 180),
            "palm": (120, 120, 130),
        }

        CONNECTIONS = [
            (0, 1, "thumb"), (1, 2, "thumb"), (2, 3, "thumb"), (3, 4, "thumb"),
            (0, 5, "index"), (5, 6, "index"), (6, 7, "index"), (7, 8, "index"),
            (0, 9, "middle"), (9, 10, "middle"), (10, 11, "middle"), (11, 12, "middle"),
            (0, 13, "ring"), (13, 14, "ring"), (14, 15, "ring"), (15, 16, "ring"),
            (0, 17, "pinky"), (17, 18, "pinky"), (18, 19, "pinky"), (19, 20, "pinky"),
            (5, 9, "palm"), (9, 13, "palm"), (13, 17, "palm"),
        ]

        TIPS = {4, 8, 12, 16, 20}

        for hand_lm in [tracking.right_hand, tracking.left_hand]:
            if hand_lm is None:
                continue

            # Glow layer for connections
            glow = frame.copy()
            for s, e, group in CONNECTIONS:
                pt1 = (int(hand_lm[s][0] * w), int(hand_lm[s][1] * h))
                pt2 = (int(hand_lm[e][0] * w), int(hand_lm[e][1] * h))
                # Outer glow
                cv2.line(glow, pt1, pt2, GLOW_COLORS[group], 6, cv2.LINE_AA)
            cv2.addWeighted(glow, 0.3, frame, 0.7, 0, frame)

            # Sharp connections
            for s, e, group in CONNECTIONS:
                pt1 = (int(hand_lm[s][0] * w), int(hand_lm[s][1] * h))
                pt2 = (int(hand_lm[e][0] * w), int(hand_lm[e][1] * h))
                cv2.line(frame, pt1, pt2, FINGER_COLORS[group], 2, cv2.LINE_AA)

            # Landmarks with glow
            for i, lm in enumerate(hand_lm):
                x, y_pos = int(lm[0] * w), int(lm[1] * h)
                if i == 0:  # Wrist — bright dot with ring
                    cv2.circle(frame, (x, y_pos), 7, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y_pos), 9, ACCENT_CYAN, 1, cv2.LINE_AA)
                elif i in TIPS:  # Fingertips — glowing dots
                    cv2.circle(frame, (x, y_pos), 6, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y_pos), 8, ACCENT_CYAN, 1, cv2.LINE_AA)
                else:  # Joints
                    cv2.circle(frame, (x, y_pos), 3, (220, 220, 225), -1, cv2.LINE_AA)

    def _draw_arms(self, frame: np.ndarray, tracking: TrackingResult):
        """Draw minimal arm skeleton from pose landmarks with subtle glow."""
        if tracking.pose is None:
            return

        h, w = frame.shape[:2]
        pose = tracking.pose

        ARM_CONNECTIONS = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 12),            # Shoulders
        ]

        for s, e in ARM_CONNECTIONS:
            if s < len(pose) and e < len(pose):
                vis_s = pose[s][3] if len(pose[s]) > 3 else 1.0
                vis_e = pose[e][3] if len(pose[e]) > 3 else 1.0
                if vis_s > 0.5 and vis_e > 0.5:
                    pt1 = (int(pose[s][0] * w), int(pose[s][1] * h))
                    pt2 = (int(pose[e][0] * w), int(pose[e][1] * h))
                    # Glow
                    glow = frame.copy()
                    cv2.line(glow, pt1, pt2, (80, 160, 80), 6, cv2.LINE_AA)
                    cv2.addWeighted(glow, 0.25, frame, 0.75, 0, glow)
                    np.copyto(frame, glow)
                    # Sharp line
                    cv2.line(frame, pt1, pt2, (120, 210, 120), 2, cv2.LINE_AA)
