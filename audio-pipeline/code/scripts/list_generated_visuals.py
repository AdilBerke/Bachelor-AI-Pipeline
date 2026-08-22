#!/usr/bin/env python3
"""List generated visual assets from the local registry."""

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

from src.Video.generated_visuals import export_training_dataset, list_generated_visuals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List registered generated GIFs/videos.")
    parser.add_argument("--type", choices=["gif", "video"], default=None)
    parser.add_argument("--mood", default=None)
    parser.add_argument("--genre", default=None)
    parser.add_argument("--motif", default=None)
    parser.add_argument("--color", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--export-dataset", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.export_dataset:
        result = export_training_dataset()
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
        return 0

    visuals = list_generated_visuals(
        visual_type=args.type,
        mood=args.mood,
        genre=args.genre,
        motif=args.motif,
        color=args.color,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps({"count": len(visuals), "visuals": visuals}, ensure_ascii=False, indent=2))
        return 0

    if not visuals:
        print("No generated visuals found.")
        return 0
    for visual in visuals:
        track = visual.get("track_context") or {}
        print(f"{visual.get('id')} [{visual.get('type')}] {visual.get('duration_seconds')}s")
        print(f"  file:  {visual.get('file_path')}")
        print(f"  title: {track.get('track_title') or '-'}")
        print(f"  mood:  {track.get('mood') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
