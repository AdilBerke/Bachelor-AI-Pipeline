#!/usr/bin/env python3
"""Register an existing GIF/video file in ausgaben/generated_visuals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
BACHELORARBEIT_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = BACHELORARBEIT_ROOT
CODE_ROOT = BACHELORARBEIT_ROOT / "code"
for path in [str(CODE_ROOT), str(BACHELORARBEIT_ROOT), str(WORKSPACE_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.Video.generated_visuals import GeneratedVisualError, register_generated_visual
from src.Video.visual_prompt_builder import build_visual_prompt, build_visual_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register an existing generated visual file.")
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--type", choices=["gif", "video", "mp4", "webm"], default=None)
    parser.add_argument("--track-title", default="")
    parser.add_argument("--genre", default="lofi")
    parser.add_argument("--mood", default="calm")
    parser.add_argument("--room", default="")
    parser.add_argument("--scene", default="")
    parser.add_argument("--colors", default="")
    parser.add_argument("--lighting", default="")
    parser.add_argument("--style", default="anime-inspired lo-fi background")
    parser.add_argument("--camera-motion", default="")
    parser.add_argument("--atmosphere", default="")
    parser.add_argument("--motifs", default="")
    parser.add_argument("--model", default="manual-register")
    parser.add_argument("--workflow", default="register_generated_visual")
    parser.add_argument("--notes", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    prompt = build_visual_prompt(
        track_title=args.track_title,
        genre=args.genre,
        mood=args.mood,
        room=args.room,
        scene=args.scene,
        colors=args.colors,
        lighting=args.lighting,
        style=args.style,
        camera_motion=args.camera_motion,
        atmosphere=args.atmosphere,
        motifs=args.motifs,
        output_format=args.type or args.file.suffix.lstrip("."),
    )
    metadata: Dict[str, Any] = {
        "source": {"generated": True, "model": args.model, "workflow": args.workflow, "seed": None},
        "track_context": {
            "track_id": "",
            "track_title": args.track_title,
            "genre": args.genre,
            "mood": args.mood,
        },
        "visual_settings": build_visual_settings(
            room=args.room,
            scene=args.scene,
            colors=args.colors,
            lighting=args.lighting,
            style=args.style,
            camera_motion=args.camera_motion,
            atmosphere=args.atmosphere,
            motifs=args.motifs,
        ),
        "prompt": {
            "positive": prompt["positive_prompt"],
            "negative": prompt["negative_prompt"],
            "raw": prompt["raw"],
        },
        "training_tags": prompt["training_tags"],
        "caption": prompt["caption"],
        "notes": args.notes,
    }
    try:
        result = register_generated_visual(file_path=args.file, visual_type=args.type, metadata=metadata)
    except GeneratedVisualError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Registered {result['id']}: {result['file_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
