#!/usr/bin/env python3
"""Search/extract short video clips and prepare optional LTX LoRA training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
BACHELORARBEIT_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = BACHELORARBEIT_ROOT
CODE_ROOT = BACHELORARBEIT_ROOT / "code"
for path in [str(CODE_ROOT), str(BACHELORARBEIT_ROOT), str(WORKSPACE_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.Video.reference_video_training_dataset import (
    DEFAULT_REFERENCE_LTX_DATASET_DIR,
    DEFAULT_VIDEO_TRAINING_DATASET_DIR,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_RESOLUTION_BUCKET,
    DEFAULT_TRIGGER_TOKEN,
    ReferenceVideoTrainingError,
    automate_reference_video_training,
)
from src.Video.video_dataset_service import (
    DEFAULT_VIDEO_REFERENCE_DATASET_DIR,
    MAX_REFERENCE_CLIP_SECONDS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automate the reference-video pipeline: search stored candidates, extract max-20s clips, "
            "write MusicGen-style CSV/JSONL manifests, prepare LTX data, and optionally run preprocessing/training."
        )
    )
    parser.add_argument("--query", "-q", default="lofi rainy night window study", help="Search query for stored candidates.")
    parser.add_argument("--max-results", type=int, default=10, help="How many candidate results to rank.")
    parser.add_argument("--clip-count", type=int, default=4, help="How many top results to extract.")
    parser.add_argument("--clip-duration", type=float, default=20.0, help="Clip duration, capped at 20 seconds.")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--gif", action="store_true", help="Also create GIF previews for extracted clips.")
    parser.add_argument("--skip-extract", action="store_true", help="Use already extracted reference clips only.")
    parser.add_argument("--reference-dataset-dir", default=str(DEFAULT_VIDEO_REFERENCE_DATASET_DIR))
    parser.add_argument("--manifest-dir", default=str(DEFAULT_VIDEO_TRAINING_DATASET_DIR))
    parser.add_argument("--ltx-dataset-dir", default=str(DEFAULT_REFERENCE_LTX_DATASET_DIR))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--prepare-ltx", action="store_true", help="Normalize clips and write LTX dataset/config.")
    parser.add_argument("--preprocess", action="store_true", help="Run LTX preprocessing after preparing the LTX dataset.")
    parser.add_argument("--train", action="store_true", help="Run LTX LoRA training after preprocessing is ready.")
    parser.add_argument("--max-ltx-items", type=int, default=None)
    parser.add_argument("--resolution-bucket", default=DEFAULT_RESOLUTION_BUCKET)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--trigger-token", default=DEFAULT_TRIGGER_TOKEN)
    parser.add_argument("--preprocess-timeout", type=int, default=3600)
    parser.add_argument("--train-timeout", type=int, default=7200)
    parser.add_argument(
        "--require-verified-license",
        action="store_true",
        help="Set training_allowed from metadata only. Default keeps analysis/training automation unblocked.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _print_summary(result: dict) -> None:
    extraction = result.get("extraction") or {}
    manifest = result.get("manifest") or {}
    ltx = result.get("ltx") or {}
    print("Video clip automation finished.")
    if extraction:
        print(
            f"extracted: {extraction.get('extracted_count', 0)}/"
            f"{extraction.get('requested_clip_count', 0)} clips"
        )
        if extraction.get("failed_count"):
            print(f"failed:    {extraction.get('failed_count')}")
    counts = manifest.get("counts") or {}
    paths = manifest.get("paths") or {}
    print(f"manifest rows: {counts.get('rows', 0)}")
    print(f"csv:           {paths.get('csv')}")
    print(f"data.jsonl:    {paths.get('data_jsonl')}")
    if ltx:
        ltx_counts = ltx.get("counts") or {}
        ltx_paths = ltx.get("paths") or {}
        commands = ltx.get("commands") or {}
        readiness = ltx.get("readiness") or {}
        print(f"ltx rows:      {ltx_counts.get('dataset_rows', 0)}")
        print(f"ltx dataset:   {ltx_paths.get('dataset_jsonl')}")
        print(f"ltx config:    {ltx_paths.get('config')}")
        print(f"preprocess:    {commands.get('preprocess_text')}")
        print(f"train:         {commands.get('train_text')}")
        print(f"ready preprocess: {readiness.get('ready_for_preprocess')}")
        print(f"ready train:      {readiness.get('ready_for_train')}")
    if result.get("preprocess_result"):
        ready = (result["preprocess_result"].get("readiness") or {}).get("ready_for_train")
        print(f"preprocess done; ready_for_train={ready}")
    if result.get("train_result"):
        print(f"train return code: {result['train_result'].get('returncode')}")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    duration = min(max(float(args.clip_duration), 0.1), MAX_REFERENCE_CLIP_SECONDS)
    try:
        result = automate_reference_video_training(
            query=args.query,
            max_results=args.max_results,
            clip_count=args.clip_count,
            clip_duration_seconds=duration,
            start_seconds=args.start_seconds,
            create_gif=args.gif,
            reference_dataset_dir=Path(args.reference_dataset_dir),
            manifest_dir=Path(args.manifest_dir),
            ltx_dataset_dir=Path(args.ltx_dataset_dir),
            run_dir=Path(args.run_dir) if args.run_dir else None,
            skip_extract=args.skip_extract,
            prepare_ltx=bool(args.prepare_ltx or args.preprocess or args.train),
            run_preprocess=bool(args.preprocess or args.train),
            run_train=args.train,
            max_ltx_items=args.max_ltx_items,
            resolution_bucket=args.resolution_bucket,
            fps=args.fps,
            model_source=args.model_source,
            trigger_token=args.trigger_token,
            allow_unverified_training=not args.require_verified_license,
            preprocess_timeout=args.preprocess_timeout,
            train_timeout=args.train_timeout,
        )
    except (ReferenceVideoTrainingError, Exception) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
