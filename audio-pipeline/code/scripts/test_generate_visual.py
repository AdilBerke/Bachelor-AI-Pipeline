#!/usr/bin/env python3
"""Generate a short local GIF/video test visual and register it."""

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

from src.Video.generated_visuals import GeneratedVisualError, generate_test_visual  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a registered generated visual test asset.")
    parser.add_argument("--track-title", default="Rainy Night Study")
    parser.add_argument("--genre", default="lofi")
    parser.add_argument("--mood", default="calm")
    parser.add_argument("--room", default="bedroom")
    parser.add_argument("--scene", default="rainy window")
    parser.add_argument("--colors", default="blue,purple,warm light")
    parser.add_argument("--lighting", default="warm lamp")
    parser.add_argument("--style", default="anime-inspired lo-fi background")
    parser.add_argument("--camera-motion", default="subtle loop camera drift")
    parser.add_argument("--atmosphere", default="rain")
    parser.add_argument("--motifs", default="study_person,rainy_window,cozy_room,small_loop_motion")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--format", choices=["gif", "mp4", "webm", "video"], default="gif")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate_test_visual(
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
            duration_seconds=args.duration,
            output_format=args.format,
            seed=args.seed,
            notes=args.notes,
        )
    except GeneratedVisualError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        visual = result["visual"]
        print("Generated visual created.")
        print(f"id:        {visual['id']}")
        print(f"type:      {visual['type']}")
        print(f"file:      {visual['file_path']}")
        print(f"thumbnail: {visual.get('thumbnail_path') or '-'}")
        print(f"metadata:  {visual['metadata_path']}")
        print(f"duration:  {visual['duration_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
