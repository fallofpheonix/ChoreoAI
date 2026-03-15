#!/usr/bin/env python3
"""
scripts/extract_poses.py — Batch-extract skeleton poses from a folder of videos.

Usage::

    python scripts/extract_poses.py \\
        --video-dir data/videos \\
        --output-dir data/raw \\
        --backend stub
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch pose extraction.")
    parser.add_argument("--video-dir", required=True, help="Directory of input videos.")
    parser.add_argument("--output-dir", required=True, help="Output root directory.")
    parser.add_argument("--backend", default="mediapipe", choices=["mediapipe", "stub"])
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--ext",
        nargs="+",
        default=["mp4", "avi", "mov"],
        help="Video file extensions to process.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from choreoai.pose_extractor import extract_poses, save_poses
    from choreoai.preprocess import normalize_poses

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    videos = []
    for ext in args.ext:
        videos.extend(video_dir.glob(f"**/*.{ext}"))

    if not videos:
        logger.warning("No videos found in %s with extensions %s", video_dir, args.ext)
        return

    for video in sorted(videos):
        rel = video.relative_to(video_dir).with_suffix("")
        out_dir = output_dir / rel
        out_path = out_dir / "poses.npy"

        if out_path.exists():
            logger.info("Skipping (already processed): %s", video.name)
            continue

        try:
            poses = extract_poses(video, backend=args.backend)
            if args.normalize:
                poses = normalize_poses(poses)
            save_poses(poses, out_path)
            logger.info("Processed: %s → %s %s", video.name, out_path, tuple(poses.shape))
        except Exception as exc:
            logger.error("Failed to process %s: %s", video.name, exc)


if __name__ == "__main__":
    main()
