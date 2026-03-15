#!/usr/bin/env python3
"""
scripts/build_manifest.py — Scan a data directory and emit a manifest JSON.

Usage::

    python scripts/build_manifest.py --data-dir data/raw --output data/manifest.json
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
    parser = argparse.ArgumentParser(description="Build a ChoreoAI data manifest.")
    parser.add_argument("--data-dir", required=True, help="Root data directory.")
    parser.add_argument("--output", default="data/manifest.json")
    parser.add_argument("--require-text", action="store_true")
    parser.add_argument("--require-image", action="store_true")
    parser.add_argument("--require-audio", action="store_true")
    args = parser.parse_args()

    # Delayed import to avoid issues when running outside the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from choreoai.dataset import save_manifest, scan_data_directory

    entries = scan_data_directory(
        args.data_dir,
        require_text=args.require_text,
        require_image=args.require_image,
        require_audio=args.require_audio,
    )
    save_manifest(entries, args.output)
    logger.info("Saved manifest with %d entries → %s", len(entries), args.output)


if __name__ == "__main__":
    main()
