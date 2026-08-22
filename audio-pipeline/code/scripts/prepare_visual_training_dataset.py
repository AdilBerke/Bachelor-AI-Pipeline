#!/usr/bin/env python3
"""Prepare generated GIF/video visuals for local LTX LoRA training."""

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

from src.Video.visual_training_dataset import (  # noqa: E402
    DEFAULT_LTX_DATASET_DIR,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_RESOLUTION_BUCKET,
    DEFAULT_TRIGGER_TOKEN,
    VisualTrainingDatasetError,
    ensure_smoke_training_visuals,
    export_ltx_training_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export registered generated visuals as an LTX-compatible training dataset."
    )
    parser.add_argument("--generated-visuals-dir", default=None)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_LTX_DATASET_DIR))
    parser.add_argument("--ensure-smoke-data", action="store_true")
    parser.add_argument("--smoke-count", type=int, default=6)
    parser.add_argument("--smoke-duration", type=float, default=4.0)
    parser.add_argument("--resolution-bucket", default=DEFAULT_RESOLUTION_BUCKET)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-duration", type=float, default=4.0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--trigger-token", default=DEFAULT_TRIGGER_TOKEN)
    parser.add_argument("--no-include-gifs", action="store_true")
    parser.add_argument("--no-convert-gifs", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        smoke = None
        if args.ensure_smoke_data:
            smoke = ensure_smoke_training_visuals(
                root_dir=Path(args.generated_visuals_dir) if args.generated_visuals_dir else None,
                count=args.smoke_count,
                duration_seconds=args.smoke_duration,
            )
        report = export_ltx_training_dataset(
            generated_visuals_dir=Path(args.generated_visuals_dir) if args.generated_visuals_dir else None,
            dataset_dir=Path(args.dataset_dir),
            resolution_bucket=args.resolution_bucket,
            fps=args.fps,
            max_duration_seconds=args.max_duration,
            max_items=args.max_items,
            include_gifs=not args.no_include_gifs,
            convert_gifs=not args.no_convert_gifs,
            validation_ratio=args.validation_ratio,
            model_source=args.model_source,
            trigger_token=args.trigger_token,
        )
    except VisualTrainingDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = {"smoke_data": smoke, "report": report}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    counts = report["counts"]
    readiness = report["readiness"]
    print("Visual training dataset prepared.")
    if smoke:
        print(f"smoke visuals created: {smoke['created_count']}")
    print(f"dataset:    {report['paths']['dataset_jsonl']}")
    print(f"assets:     {report['paths']['assets_dir']}")
    print(f"config:     {report['paths']['config']}")
    print(f"rows:       {counts['dataset_rows']} total, {counts['train_rows']} train, {counts['validation_rows']} validation")
    print(f"skipped:    {counts['skipped']}")
    print(f"preprocess: {report['commands']['preprocess_text']}")
    print(f"train:      {report['commands']['train_text']}")
    print(f"ready for preprocess: {readiness['ready_for_preprocess']}")
    print(f"ready for train:      {readiness['ready_for_train']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
