"""Classify audio files as human or AI-generated.

    python -m voice_detector.predict recording.wav call.mp3
"""

import argparse
from pathlib import Path

from .model import Detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/detector.pt"))
    parser.add_argument("--threshold", type=float, default=None,
                        help="P(AI) above which a clip is called AI-generated (default: from the checkpoint)")
    args = parser.parse_args()

    detector = Detector(str(args.checkpoint))
    threshold = detector.threshold if args.threshold is None else args.threshold
    for path in args.files:
        score, per_window = detector.score_file(str(path))
        verdict = "AI-generated" if score >= threshold else "human"
        print(f"{verdict:>12}  P(AI)={score:.3f}  windows={len(per_window)}  {path}")


if __name__ == "__main__":
    main()
