"""
Non-blocking Translation Service for Sign Language Interpreter
==============================================================
Translates English sentences into multiple languages using
Google Translate (free, no API key) via deep-translator.

Features:
  - Background thread for non-blocking translation
  - Cache to avoid re-translating the same text
  - Graceful fallback if library not installed or offline
  - Language cycling with a single keypress
"""

import threading
import queue
import time
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# ── Check availability ───────────────────────────────────────────────────
_TRANSLATOR_OK = False
try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_OK = True
    logger.info("Translation service: deep-translator loaded")
except ImportError:
    logger.warning(
        "deep-translator not installed — translation disabled. "
        "Install with: pip install deep-translator"
    )

# ── Supported languages ─────────────────────────────────────────────────
# (code, display_name, flag_emoji)
LANGUAGES = [
    ("off",   "Off",        ""),
    ("hi",    "Hindi",      "HI"),
    ("es",    "Spanish",    "ES"),
    ("fr",    "French",     "FR"),
    ("ar",    "Arabic",     "AR"),
    ("ja",    "Japanese",   "JA"),
    ("zh-CN", "Chinese",    "ZH"),
    ("de",    "German",     "DE"),
    ("pt",    "Portuguese", "PT"),
    ("ko",    "Korean",     "KO"),
    ("ru",    "Russian",    "RU"),
]

# Font candidates for non-Latin scripts (Windows system fonts)
LANG_FONTS = {
    "hi":    ["NirmalaS.ttf", "Nirmala.ttf", "mangal.ttf", "arial.ttf"],
    "ar":    ["segoeui.ttf", "arial.ttf"],
    "ja":    ["YuGothR.ttc", "msgothic.ttc", "msyh.ttc"],
    "zh-CN": ["msyh.ttc", "simsun.ttc", "YuGothR.ttc"],
    "ko":    ["malgun.ttf", "gulim.ttc"],
    "ru":    ["segoeui.ttf", "arial.ttf"],
    "default": ["segoeui.ttf", "arial.ttf"],
}


class TranslationService:
    """
    Background translation service.
    Non-blocking — submit text and poll for results.
    """

    def __init__(self):
        self._lang_index = 0  # index into LANGUAGES
        self._cache: Dict[Tuple[str, str], str] = {}
        self._latest_result: Optional[str] = None
        self._latest_source: str = ""
        self._lock = threading.Lock()
        self._alive = True
        self._queue: queue.Queue = queue.Queue()

        if _TRANSLATOR_OK:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        else:
            self._thread = None

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return _TRANSLATOR_OK

    @property
    def current_code(self) -> str:
        return LANGUAGES[self._lang_index][0]

    @property
    def current_name(self) -> str:
        return LANGUAGES[self._lang_index][1]

    @property
    def current_flag(self) -> str:
        return LANGUAGES[self._lang_index][2]

    @property
    def is_active(self) -> bool:
        return self.current_code != "off" and _TRANSLATOR_OK

    # ── Control ──────────────────────────────────────────────────────────

    def cycle_language(self) -> str:
        """Cycle to the next language. Returns the new display name."""
        self._lang_index = (self._lang_index + 1) % len(LANGUAGES)
        with self._lock:
            self._latest_result = None
            self._latest_source = ""
        return self.current_name

    def set_language_by_index(self, index: int) -> str:
        """Set language by index. Returns new display name."""
        self._lang_index = index % len(LANGUAGES)
        with self._lock:
            self._latest_result = None
            self._latest_source = ""
        return self.current_name

    @property
    def lang_index(self) -> int:
        return self._lang_index

    @property
    def lang_count(self) -> int:
        return len(LANGUAGES)

    def translate(self, text: str) -> None:
        """
        Submit text for background translation.
        Non-blocking — call get_result() to retrieve the output.
        """
        if not _TRANSLATOR_OK or self.current_code == "off":
            return
        if not text or not text.strip():
            with self._lock:
                self._latest_result = None
            return

        text = text.strip()
        lang = self.current_code

        # Check cache
        cache_key = (text, lang)
        if cache_key in self._cache:
            with self._lock:
                self._latest_result = self._cache[cache_key]
                self._latest_source = text
            return

        # Submit to worker
        self._queue.put((text, lang))

    def get_result(self) -> Optional[str]:
        """Get the latest translated text (None if pending or inactive)."""
        if not self.is_active:
            return None
        with self._lock:
            return self._latest_result

    def get_font_candidates(self) -> list:
        """Get font file candidates for the current language."""
        return LANG_FONTS.get(self.current_code, LANG_FONTS["default"])

    def shutdown(self):
        """Stop the worker thread."""
        self._alive = False
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=3)
        logger.info("Translation service shut down")

    # ── Background worker ────────────────────────────────────────────────

    def _worker(self):
        """Process translation requests in the background."""
        while self._alive:
            try:
                item = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break

            text, lang = item

            try:
                translator = GoogleTranslator(source="en", target=lang)
                result = translator.translate(text)

                if result:
                    cache_key = (text, lang)
                    self._cache[cache_key] = result

                    with self._lock:
                        if self.current_code == lang:
                            self._latest_result = result
                            self._latest_source = text
                            logger.debug(f"Translated [{lang}]: {text} → {result}")

            except Exception as e:
                logger.debug(f"Translation error ({lang}): {e}")
                with self._lock:
                    self._latest_result = f"[translation error]"


def is_available() -> bool:
    """Whether translation is available."""
    return _TRANSLATOR_OK
