from __future__ import annotations

import argparse
import sys
from pathlib import Path

from choreoai.dataset import (
    bootstrap_dataset_from_raw,
    stage_pose_sequence,
    summarize_dataset,
    validate_dataset,
)
from choreoai.preprocess import PreprocessConfig, preprocess_dataset


def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    root = Path(args.root)
    errors = validate_dataset(root)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 2
    print("OK")
    return 0


def _cmd_summarize_dataset(args: argparse.Namespace) -> int:
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


def main() -> int:
    parser = argparse.ArgumentParser(prog="choreoai")
    sub = parser.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate-dataset", help="validate dataset layout and poses.npy")
    v.add_argument("--root", required=True, help="dataset root directory")
    v.set_defaults(func=_cmd_validate_dataset)

    s = sub.add_parser("summarize-dataset", help="print sequence-level dataset statistics")
    s.add_argument("--root", required=True, help="dataset root directory")
    s.set_defaults(func=_cmd_summarize_dataset)

    p = sub.add_parser("stage-sequence", help="copy a raw poses.npy into dataset layout")
    p.add_argument("--source", required=True, help="source .npy path")
    p.add_argument("--root", required=True, help="dataset root directory")
    p.add_argument("--seq-id", required=True, help="sequence identifier")
    p.add_argument("--text", help="optional text prompt to store with the sequence")
    p.add_argument("--force", action="store_true", help="overwrite existing sequence directory")
    p.set_defaults(func=_cmd_stage_sequence)

    b = sub.add_parser("bootstrap-dataset", help="stage every .npy file from a raw directory")
    b.add_argument("--raw-root", required=True, help="directory containing raw .npy pose files")
    b.add_argument("--root", required=True, help="dataset root directory")
    b.add_argument("--force", action="store_true", help="overwrite existing sequence directories")
    b.set_defaults(func=_cmd_bootstrap_dataset)

    pp = sub.add_parser("preprocess-dataset", help="repair, smooth, and normalize pose sequences")
    pp.add_argument("--root", required=True, help="input dataset root directory")
    pp.add_argument("--output-root", required=True, help="output dataset root directory")
    pp.add_argument("--smooth-window", type=int, default=5, help="odd moving-average window size")
    pp.add_argument("--center-joint", type=int, default=0, help="joint index used as root for centering")
    pp.add_argument("--force", action="store_true", help="overwrite existing sequence directories")
    pp.set_defaults(func=_cmd_preprocess_dataset)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
