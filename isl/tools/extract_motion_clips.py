"""
Backward-compatible entrypoint for motion clip extraction.

Use:
  python isl/tools/extract_motion_clips.py ...
or
  python isl/tools/extract_motion.py ...
"""

from extract_motion import main


if __name__ == "__main__":
    main()
