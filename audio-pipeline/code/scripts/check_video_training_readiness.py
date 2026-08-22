#!/usr/bin/env python3
"""Check whether local video training can be started safely."""

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
    check_ltx_training_readiness,
    collect_video_training_environment,
    validate_ltx_training_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the generated-visuals LTX training dataset and environment.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_LTX_DATASET_DIR))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    payload = {
        "dataset_dir": str(dataset_dir),
        "dataset_validation": validate_ltx_training_dataset(dataset_dir=dataset_dir),
        "readiness": check_ltx_training_readiness(dataset_dir=dataset_dir),
        "environment": collect_video_training_environment(dataset_dir=dataset_dir),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["readiness"]["ready_for_train"] else 2

    validation = payload["dataset_validation"]
    readiness = payload["readiness"]
    env = payload["environment"]
    torch_info = (env.get("python_packages") or {}).get("torch") or {}
    packages = (env.get("python_packages") or {}).get("packages") or {}
    gpu = env.get("gpu") or {}
    gpus = gpu.get("gpus") or []

    print("Video training readiness")
    print(f"dataset:          {dataset_dir}")
    print(f"dataset valid:    {validation.get('valid')} ({validation.get('row_count')} rows)")
    print(f"ready preprocess: {readiness.get('ready_for_preprocess')}")
    print(f"ready train:      {readiness.get('ready_for_train')}")
    print(f"torch:            {packages.get('torch')} cuda={torch_info.get('cuda_version')}")
    print(f"cuda available:   {torch_info.get('cuda_available')}")
    if gpus:
        best = max(gpus, key=lambda item: int(item.get("memory_free_mb") or 0))
        print(f"gpu:              {best.get('name')} free={best.get('memory_free_mb')}MB")
    print(f"ffmpeg:           {(env.get('ffmpeg') or {}).get('path')}")
    print(f"ltx trainer:      {(env.get('ltx_trainer') or {}).get('root')}")
    if not readiness.get("ready_for_train"):
        print("blocking checks:")
        for check in readiness.get("checks") or []:
            if not check.get("ok"):
                print(f"  - {check.get('name')}: {check.get('detail')}")
    return 0 if readiness.get("ready_for_train") else 2


if __name__ == "__main__":
    raise SystemExit(main())
