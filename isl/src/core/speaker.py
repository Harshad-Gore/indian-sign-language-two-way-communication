"""
Non-blocking text-to-speech for real-time sign language output.
Each utterance is spoken in a fresh subprocess to completely
isolate SAPI COM calls from the main process (MediaPipe/TF).

Supports voice personas (Male/Female/Child) with pitch and rate control.
"""

import subprocess
import sys
import threading
import queue
import os
import logging
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Friendly spoken forms for gesture names
_SPOKEN_FORMS: Dict[str, str] = {
    "HELLO": "Hello",
    "THANK_YOU": "Thank you",
    "YES": "Yes",
    "NO": "No",
    "STOP": "Stop",
    "PLEASE": "Please",
    "HELP": "Help",
    "EAT": "Eat",
    "DRINK": "Drink",
    "LOVE": "Love",
    "GOOD": "Good",
    "BAD": "Bad",
    "I_LOVE_YOU": "I love you",
    "THUMBS_UP": "Thumbs up",
    "SORRY": "Sorry",
    "WANT": "Want",
    "WHERE": "Where",
    "WHAT": "What",
    "WHO": "Who",
    "COME": "Come",
    "GO": "Go",
    "HOME": "Home",
    "WATER": "Water",
    "MORE": "More",
    "AGAIN": "Again",
    "WAIT": "Wait",
    "THINK": "Think",
    "KNOW": "Know",
    "DON'T_KNOW": "Don't know",
    "LIKE": "Like",
    "FRIEND": "Friend",
    "FAMILY": "Family",
    "SCHOOL": "School",
    "WORK": "Work",
    "NEED": "Need",
}


def _spoken_text(sign: str) -> str:
    """Convert a sign name to natural spoken text."""
    if sign in _SPOKEN_FORMS:
        return _SPOKEN_FORMS[sign]
    if len(sign) == 1:
        return sign
    return sign.replace("_", " ").title()


# ── Helper scripts (written at import time) ─────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SPEAK_SCRIPT = os.path.join(_SCRIPT_DIR, "_speak.py")

# Updated speak script: accepts voice_index and rate as optional args
_SPEAK_CODE = '''\
import sys
try:
    import pythoncom
    pythoncom.CoInitialize()
    import win32com.client
    voice = win32com.client.Dispatch("SAPI.SpVoice")

    # Mode: list voices or speak
    if len(sys.argv) > 1 and sys.argv[1] == "__list__":
        voices = voice.GetVoices()
        for i in range(voices.Count):
            print(f"{i}|{voices.Item(i).GetDescription()}")
    else:
        text = sys.argv[1] if len(sys.argv) > 1 else ""
        voice_idx = int(sys.argv[2]) if len(sys.argv) > 2 else -1
        rate = int(sys.argv[3]) if len(sys.argv) > 3 else 2

        if voice_idx >= 0:
            voices = voice.GetVoices()
            if voice_idx < voices.Count:
                voice.Voice = voices.Item(voice_idx)

        voice.Rate = rate
        voice.Volume = 100
        voice.Speak(text)

    pythoncom.CoUninitialize()
except Exception as e:
    sys.stderr.write(str(e))
'''

try:
    with open(_SPEAK_SCRIPT, "w", encoding="utf-8") as f:
        f.write(_SPEAK_CODE)
except Exception:
    pass


# ── Voice presets ────────────────────────────────────────────────────────
VOICE_PRESETS = [
    {
        "name": "Male",
        "keywords": ["david", "mark", "james", "richard", "male"],
        "rate": 2,
    },
    {
        "name": "Female",
        "keywords": ["zira", "hazel", "eva", "susan", "jenny", "female"],
        "rate": 2,
    },
    {
        "name": "Child",
        "keywords": ["zira", "hazel", "eva", "female"],  # female voice sped up
        "rate": 5,
    },
]


class Speaker:
    """
    Background TTS with voice persona support.

    Features:
      - Male / Female / Child presets (auto-detects SAPI voices)
      - Non-blocking queue-based speaking
      - Subprocess isolation for COM safety

    Usage:
        speaker = Speaker()
        speaker.say("Hello")       # speak with current voice
        speaker.cycle_voice()      # switch Male → Female → Child
        speaker.say_raw("Full sentence here")  # speak exact text
        speaker.shutdown()
    """

    def __init__(self):
        self._queue: queue.Queue[Optional[Tuple[str, int, int]]] = queue.Queue()
        self._alive = True
        self._python = sys.executable
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        # Voice persona state
        self._preset_index = 0
        self._voice_index = -1  # SAPI voice index (-1 = default)
        self._rate = 2
        self._available_voices: List[Tuple[int, str]] = []

        # Enumerate system voices in background
        self._enumerate_voices()
        self._apply_preset()

        logger.info(f"Speaker initialized — voice: {self.current_voice_name}")

    # ── Voice persona control ────────────────────────────────────────────

    @property
    def current_voice_name(self) -> str:
        """Current voice preset name."""
        return VOICE_PRESETS[self._preset_index]["name"]

    @property
    def voice_count(self) -> int:
        return len(VOICE_PRESETS)

    def cycle_voice(self) -> str:
        """Cycle to the next voice preset. Returns new preset name."""
        self._preset_index = (self._preset_index + 1) % len(VOICE_PRESETS)
        self._apply_preset()
        logger.info(f"Voice switched to: {self.current_voice_name}")
        return self.current_voice_name

    def set_voice_by_index(self, index: int) -> str:
        """Set voice preset by index. Returns new preset name."""
        self._preset_index = index % len(VOICE_PRESETS)
        self._apply_preset()
        logger.info(f"Voice set to: {self.current_voice_name}")
        return self.current_voice_name

    @property
    def preset_index(self) -> int:
        return self._preset_index

    def _enumerate_voices(self):
        """Enumerate available SAPI voices via subprocess."""
        try:
            proc = subprocess.run(
                [self._python, _SPEAK_SCRIPT, "__list__"],
                timeout=5,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                for line in proc.stdout.strip().split("\n"):
                    parts = line.strip().split("|", 1)
                    if len(parts) == 2:
                        idx = int(parts[0])
                        name = parts[1]
                        self._available_voices.append((idx, name))
                logger.info(
                    f"Found {len(self._available_voices)} SAPI voices: "
                    + ", ".join(n for _, n in self._available_voices)
                )
        except Exception as e:
            logger.debug(f"Voice enumeration failed: {e}")

    def _apply_preset(self):
        """Apply the current voice preset by finding a matching SAPI voice."""
        preset = VOICE_PRESETS[self._preset_index]
        self._rate = preset["rate"]
        keywords = preset["keywords"]

        # Find best matching voice
        for idx, name in self._available_voices:
            name_lower = name.lower()
            for kw in keywords:
                if kw in name_lower:
                    self._voice_index = idx
                    return

        # No match — use default
        self._voice_index = -1

    # ── Speaking ─────────────────────────────────────────────────────────

    def say(self, sign: str) -> None:
        """Queue a sign name for speech. Non-blocking."""
        if not self._alive:
            return
        text = _spoken_text(sign)
        self._queue.put((text, self._voice_index, self._rate))

    def say_raw(self, text: str) -> None:
        """Queue arbitrary text for speech. Non-blocking."""
        if not self._alive or not text or not text.strip():
            return
        # Clean sign-name artifacts (underscores → spaces)
        cleaned = text.strip().replace("_", " ")
        self._queue.put((cleaned, self._voice_index, self._rate))

    def shutdown(self) -> None:
        """Stop the worker thread."""
        self._alive = False
        self._queue.put(None)
        self._thread.join(timeout=5)
        logger.info("Speaker shut down")

    def _worker(self) -> None:
        """Pull text from queue and speak each via a subprocess."""
        while True:
            item = self._queue.get()
            if item is None:
                break

            text, voice_idx, rate = item
            try:
                proc = subprocess.run(
                    [
                        self._python, _SPEAK_SCRIPT,
                        text,
                        str(voice_idx),
                        str(rate),
                    ],
                    timeout=15,
                    capture_output=True,
                )
                if proc.returncode != 0:
                    err = proc.stderr.decode(errors="replace").strip()
                    if err:
                        logger.debug(f"TTS subprocess error: {err}")
            except subprocess.TimeoutExpired:
                logger.debug("TTS subprocess timed out")
            except Exception as e:
                logger.debug(f"TTS error: {e}")

