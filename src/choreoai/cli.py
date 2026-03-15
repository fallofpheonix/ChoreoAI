"""
cli.py — Command-line interface for ChoreoAI.

Entry points:

  choreoai generate     — generate motion from a text prompt
  choreoai extract      — extract poses from a video file
  choreoai evaluate     — compute evaluation metrics
  choreoai scan         — scan a data directory and build a manifest
  choreoai validate     — validate dataset layout and poses
  choreoai summarize    — print sequence-level dataset statistics
  choreoai stage        — copy a raw poses.npy into dataset layout
  choreoai bootstrap    — stage every .npy file from a raw directory
  choreoai preprocess   — repair, smooth, and normalize pose sequences
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("choreoai")


# ---------------------------------------------------------------------------
# Generation / Inference Commands (from main)
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
        animate_skeleton(motion, fps=args.fps, output_path=args.output_mp4)
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
    import numpy as np
    import torch

    from choreoai.encoders.motion_encoder import MotionEncoder
    from choreoai.evaluate import compute_fmd, extract_motion_features

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    encoder = MotionEncoder(latent_dim=args.latent_dim).to(device)

    def _load(path: str) -> torch.Tensor:
        arr = np.load(path)
        return torch.from_numpy(arr)

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
# Dataset Management Commands (from copilot/sub-pr-2)
# ---------------------------------------------------------------------------


def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    """Validate dataset layout and poses.npy files."""
    from choreoai.dataset_index import validate_dataset

    root = Path(args.root)
    errors = validate_dataset(root)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 2
    print("OK")
    return 0


def _cmd_summarize_dataset(args: argparse.Namespace) -> int:
    """Print sequence-level dataset statistics."""
    from choreoai.dataset_index import summarize_dataset

    root = Path(args.root)
    summaries = summarize_dataset(root)
    if not summaries:
        print("sequences=0")
        return 0

    print(f"sequences={len(summaries)}")
    for item in summaries:
        print(
            f"{item.seq_id} frames={item.frames} joints={item.joints} dims={item.dims} "
            f"dtype={item.dtype} text={int(item.has_text)} image={int(item.has_image)} "
            f"audio={int(item.has_audio)}"
        )
    return 0


def _cmd_stage_sequence(args: argparse.Namespace) -> int:
    """Copy a raw poses.npy into dataset layout."""
    from choreoai.dataset_index import stage_pose_sequence

    seq_dir = stage_pose_sequence(
        source_path=Path(args.source),
        dataset_root=Path(args.root),
        seq_id=args.seq_id,
        text=args.text,
        force=args.force,
    )
    print(seq_dir)
    return 0


def _cmd_bootstrap_dataset(args: argparse.Namespace) -> int:
    """Stage every .npy file from a raw directory."""
    from choreoai.dataset_index import bootstrap_dataset_from_raw

    created = bootstrap_dataset_from_raw(
        raw_root=Path(args.raw_root),
        dataset_root=Path(args.root),
        force=args.force,
    )
    print(f"created={len(created)}")
    for seq_dir in created:
        print(seq_dir)
    return 0


def _cmd_preprocess_dataset(args: argparse.Namespace) -> int:
    """Repair, smooth, and normalize pose sequences."""
    from choreoai.preprocess_np import PreprocessConfig, preprocess_dataset

    config = PreprocessConfig(
        smooth_window=args.smooth_window,
        center_joint=args.center_joint,
    )
    created = preprocess_dataset(
        input_root=Path(args.root),
        output_root=Path(args.output_root),
        config=config,
        force=args.force,
    )
    print(f"processed={len(created)}")
    for seq_dir in created:
        print(seq_dir)
    return 0


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

    # -- validate (dataset management) --
    v = sub.add_parser("validate", help="Validate dataset layout and poses.npy.")
    v.add_argument("--root", required=True, help="Dataset root directory.")

    # -- summarize (dataset management) --
    s = sub.add_parser("summarize", help="Print sequence-level dataset statistics.")
    s.add_argument("--root", required=True, help="Dataset root directory.")

    # -- stage (dataset management) --
    p = sub.add_parser("stage", help="Copy a raw poses.npy into dataset layout.")
    p.add_argument("--source", required=True, help="Source .npy path.")
    p.add_argument("--root", required=True, help="Dataset root directory.")
    p.add_argument("--seq-id", required=True, help="Sequence identifier.")
    p.add_argument("--text", help="Optional text prompt to store with the sequence.")
    p.add_argument("--force", action="store_true", help="Overwrite existing sequence directory.")

    # -- bootstrap (dataset management) --
    b = sub.add_parser("bootstrap", help="Stage every .npy file from a raw directory.")
    b.add_argument("--raw-root", required=True, help="Directory containing raw .npy pose files.")
    b.add_argument("--root", required=True, help="Dataset root directory.")
    b.add_argument("--force", action="store_true", help="Overwrite existing sequence directories.")

    # -- preprocess (dataset management) --
    pp = sub.add_parser("preprocess", help="Repair, smooth, and normalize pose sequences.")
    pp.add_argument("--root", required=True, help="Input dataset root directory.")
    pp.add_argument("--output-root", required=True, help="Output dataset root directory.")
    pp.add_argument("--smooth-window", type=int, default=5, help="Odd moving-average window size.")
    pp.add_argument("--center-joint", type=int, default=0, help="Joint index used as root.")
    pp.add_argument("--force", action="store_true", help="Overwrite existing sequence directories.")

    return parser


def main() -> int:
    """ChoreoAI CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    )

    dispatch_void = {
        "generate": _cmd_generate,
        "extract": _cmd_extract,
        "evaluate": _cmd_evaluate,
        "scan": _cmd_scan,
    }

    dispatch_int = {
        "validate": _cmd_validate_dataset,
        "summarize": _cmd_summarize_dataset,
        "stage": _cmd_stage_sequence,
        "bootstrap": _cmd_bootstrap_dataset,
        "preprocess": _cmd_preprocess_dataset,
    }

    if args.command in dispatch_void:
        dispatch_void[args.command](args)
        return 0
    elif args.command in dispatch_int:
        return dispatch_int[args.command](args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
