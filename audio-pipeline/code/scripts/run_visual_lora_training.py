#!/usr/bin/env python3
"""Run or print the local LTX LoRA preprocessing/training commands."""

from __future__ import annotations

import argparse
import json
import shlex
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

from src.Video.visual_training_dataset import (
    DEFAULT_LTX_DATASET_DIR,
    VisualTrainingDatasetError,
    build_ltx_preprocess_command,
    build_ltx_train_command,
    check_ltx_training_readiness,
    run_ltx_preprocess,
    run_ltx_train,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print or execute the LTX preprocessing/training steps for the generated visuals dataset. "
            "By default this only prints commands; add --preprocess and/or --train to run them."
        )
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_LTX_DATASET_DIR))
    parser.add_argument("--preprocess", action="store_true", help="Run LTX preprocessing to create latents.")
    parser.add_argument("--train", action="store_true", help="Run the LoRA training command.")
    parser.add_argument("--preprocess-timeout", type=int, default=3600)
    parser.add_argument("--train-timeout", type=int, default=7200)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    report_path = dataset_dir / "preflight_report.json"
    config_path = dataset_dir / "ltx_lora_low_vram.yaml"
    dataset_path = dataset_dir / "dataset.jsonl"
    precomputed_dir = dataset_dir / "precomputed"

    if report_path.exists():
        report = _read_json(report_path)
        preprocess_command = [str(part) for part in (report.get("commands") or {}).get("preprocess") or []]
        train_command = [str(part) for part in (report.get("commands") or {}).get("train") or []]
    else:
        preprocess_command = build_ltx_preprocess_command(dataset_path=dataset_path, output_dir=precomputed_dir)
        train_command = build_ltx_train_command(config_path=config_path)
        report = {"commands": {"preprocess": preprocess_command, "train": train_command}}

    readiness = check_ltx_training_readiness(dataset_dir=dataset_dir)
    results = {
        "dataset_dir": str(dataset_dir),
        "readiness_before": readiness,
        "commands": {
            "preprocess": preprocess_command,
            "preprocess_text": _command_text(preprocess_command),
            "train": train_command,
            "train_text": _command_text(train_command),
        },
        "preprocess_result": None,
        "train_result": None,
    }

    try:
        if args.preprocess:
            results["preprocess_result"] = run_ltx_preprocess(
                dataset_dir=dataset_dir,
                timeout=args.preprocess_timeout,
            )
        if args.train:
            results["train_result"] = run_ltx_train(
                dataset_dir=dataset_dir,
                timeout=args.train_timeout,
            )
    except VisualTrainingDatasetError as exc:
        results["error"] = str(exc)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results["readiness_after"] = check_ltx_training_readiness(dataset_dir=dataset_dir)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print("Visual LoRA training commands:")
    print(f"dataset:    {dataset_dir}")
    print(f"preprocess: {results['commands']['preprocess_text']}")
    print(f"train:      {results['commands']['train_text']}")
    print(f"ready for preprocess: {results['readiness_after']['ready_for_preprocess']}")
    print(f"ready for train:      {results['readiness_after']['ready_for_train']}")
    if not args.preprocess and not args.train:
        print("No command executed. Add --preprocess first, then --train.")
    if results["train_result"] is not None:
        print(f"train return code: {results['train_result']['returncode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
