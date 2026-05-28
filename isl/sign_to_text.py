"""
Sign Language to Text — Real-time Interpreter
==============================================
High-performance, accurate sign language recognition using:
- MediaPipe Hands for precise 21-landmark hand tracking
- Deep geometric analysis (finger states, angles, palm orientation)
- ASL/ISL alphabet fingerspelling recognition
- Dynamic gesture recognition for common words
- Temporal smoothing for flicker-free results
- Intelligent auto-commit with hold-time for letters

Usage:
    python sign_to_text.py [--camera 0] [--no-pose] [--fast]
"""

import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.fast_pipeline import FastPipeline
from src.recognition.sign_engine import SignResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


BANNER = r"""
 ╔═══════════════════════════════════════════════════════════════╗
 ║          SIGN LANGUAGE INTERPRETER — v3.0                    ║
 ║          Real-time • Accurate • Fast                         ║
 ╠═══════════════════════════════════════════════════════════════╣
 ║  Controls:                                                   ║
 ║    Q / ESC    — Quit                                         ║
 ║    C          — Clear sentence (archives to history)         ║
 ║    R          — Full reset                                   ║
 ║    Backspace  — Undo last sign                               ║
 ║    E          — Edit sentence (type to modify, Enter saves)  ║
 ║    S          — Speak entire sentence aloud                  ║
 ║    V          — Cycle voice persona (Male/Female/Child)      ║
 ║    T          — Cycle translation language                   ║
 ║    G          — Toggle grammar correction                    ║
 ╚═══════════════════════════════════════════════════════════════╝
"""


def on_sign_result(result: SignResult):
    """Console callback for sign results (optional live logging)."""
    if result.sign and result.is_new:
        stable_mark = " [STABLE]" if result.is_stable else ""
        print(f"  >> {result.sign} ({result.confidence:.0%} {result.sign_type}){stable_mark}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Sign Language to Text Interpreter",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose tracking (faster)")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization window")
    parser.add_argument("--fast", action="store_true", help="Use lite model for max speed")
    parser.add_argument("--verbose", action="store_true", help="Print each detection to console")

    args = parser.parse_args()

    print(BANNER)

    pipeline = FastPipeline(
        camera_index=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=30,
        use_pose=not args.no_pose,
        show_visualization=not args.no_viz,
    )

    callback = on_sign_result if args.verbose else None

    try:
        pipeline.start(on_result=callback)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        # Print final sentence
        sentence = pipeline.engine.get_sentence()
        signs = pipeline.engine.get_committed_signs()
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        if sentence:
            print(f"  Sentence: {sentence}")
            print(f"  Signs:    {' | '.join(signs)}")
        else:
            print("  (No signs were committed)")
        print(f"  Frames:   {pipeline.frame_count}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
