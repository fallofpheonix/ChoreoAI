"""
cli.py — Command-line interface for ChoreoAI.

Entry points:

  choreoai generate  — generate motion from a text prompt
  choreoai extract   — extract poses from a video file
  choreoai evaluate  — compute evaluation metrics
  choreoai scan      — scan a data directory and build a manifest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("choreoai")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate motion from a text prompt and optionally save visualisation."""
    from choreoai.inference import generate_motion
    from choreoai.visualize import animate_skeleton, export_threejs_json

    logger.info("Generating motion for: %r", args.text)
    motion = generate_motion(
        text=args.text,
        text_encoder_ckpt=args.text_encoder_ckpt,
        generator_ckpt=args.generator_ckpt,
        num_joints=args.num_joints,
        seq_len=args.seq_len,
        latent_dim=args.latent_dim,
        num_timesteps=args.num_timesteps,
        device=args.device,
    )  # (T, K, 3)

    logger.info("Generated motion shape: %s", tuple(motion.shape))

    if args.output_mp4:
        anim = animate_skeleton(motion, fps=args.fps, output_path=args.output_mp4)
        logger.info("Animation saved → %s", args.output_mp4)

    if args.output_json:
        payload = export_threejs_json(motion, fps=args.fps, name="generated")
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(payload, fh)
        logger.info("Three.js JSON saved → %s", args.output_json)

    if not args.output_mp4 and not args.output_json:
        logger.info("No output path specified. Use --output-mp4 or --output-json.")


def _cmd_extract(args: argparse.Namespace) -> None:
    """Extract skeleton poses from a video file."""
    from choreoai.pose_extractor import extract_poses, save_poses
    from choreoai.preprocess import normalize_poses

    video = Path(args.video)
    if not video.exists():
        logger.error("Video file not found: %s", video)
        sys.exit(1)

    logger.info("Extracting poses from %s …", video)
    poses = extract_poses(video, backend=args.backend)
    if args.normalize:
        poses = normalize_poses(poses)

    out_path = Path(args.output) if args.output else video.with_suffix(".npy")
    save_poses(poses, out_path)
    logger.info("Poses saved → %s  shape=%s", out_path, tuple(poses.shape))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Compute FMD and/or retrieval metrics for generated motion."""
    import torch
    from choreoai.encoders.motion_encoder import MotionEncoder
    from choreoai.evaluate import compute_fmd, extract_motion_features

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    encoder = MotionEncoder(latent_dim=args.latent_dim).to(device)

    import numpy as np

    def _load(path: str):  # type: ignore[return]
        arr = np.load(path)
        import torch as _torch
        return _torch.from_numpy(arr)

    real_poses = [_load(p) for p in Path(args.real_dir).glob("**/*.npy")]
    gen_poses = [_load(p) for p in Path(args.gen_dir).glob("**/*.npy")]

    logger.info("Loaded %d real and %d generated samples.", len(real_poses), len(gen_poses))

    real_feats = extract_motion_features(real_poses, encoder, device=device)
    gen_feats = extract_motion_features(gen_poses, encoder, device=device)

    fmd = compute_fmd(real_feats, gen_feats)
    print(f"FMD: {fmd:.4f}")


def _cmd_scan(args: argparse.Namespace) -> None:
    """Scan a data directory and emit a JSON manifest."""
    from choreoai.dataset import save_manifest, scan_data_directory

    entries = scan_data_directory(
        args.data_dir,
        require_text=args.require_text,
        require_image=args.require_image,
        require_audio=args.require_audio,
    )
    out_path = Path(args.output)
    save_manifest(entries, out_path)
    logger.info("Manifest with %d entries saved → %s", len(entries), out_path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="choreoai",
        description="ChoreoAI — Multimodal-to-motion translation CLI",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- generate --
    gen_p = sub.add_parser("generate", help="Generate motion from a text prompt.")
    gen_p.add_argument("text", type=str, help="Text prompt.")
    gen_p.add_argument("--text-encoder-ckpt", default=None)
    gen_p.add_argument("--generator-ckpt", default=None)
    gen_p.add_argument("--num-joints", type=int, default=17)
    gen_p.add_argument("--seq-len", type=int, default=120)
    gen_p.add_argument("--latent-dim", type=int, default=256)
    gen_p.add_argument("--num-timesteps", type=int, default=1000)
    gen_p.add_argument("--fps", type=int, default=30)
    gen_p.add_argument("--device", default=None)
    gen_p.add_argument("--output-mp4", default=None, help="Save animation to MP4.")
    gen_p.add_argument("--output-json", default=None, help="Save Three.js JSON.")

    # -- extract --
    ext_p = sub.add_parser("extract", help="Extract poses from a video.")
    ext_p.add_argument("video", type=str, help="Path to input video.")
    ext_p.add_argument("--backend", default="mediapipe", choices=["mediapipe", "stub"])
    ext_p.add_argument("--normalize", action="store_true")
    ext_p.add_argument("--output", default=None, help="Output .npy path.")

    # -- evaluate --
    eval_p = sub.add_parser("evaluate", help="Compute motion evaluation metrics.")
    eval_p.add_argument("real_dir", help="Directory of real motion .npy files.")
    eval_p.add_argument("gen_dir", help="Directory of generated motion .npy files.")
    eval_p.add_argument("--latent-dim", type=int, default=256)
    eval_p.add_argument("--device", default=None)

    # -- scan --
    scan_p = sub.add_parser("scan", help="Scan a data directory and build a manifest.")
    scan_p.add_argument("data_dir", help="Root data directory.")
    scan_p.add_argument("--output", default="manifest.json")
    scan_p.add_argument("--require-text", action="store_true")
    scan_p.add_argument("--require-image", action="store_true")
    scan_p.add_argument("--require-audio", action="store_true")

    return parser


def main() -> None:
    """ChoreoAI CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    )

    dispatch = {
        "generate": _cmd_generate,
        "extract": _cmd_extract,
        "evaluate": _cmd_evaluate,
        "scan": _cmd_scan,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
