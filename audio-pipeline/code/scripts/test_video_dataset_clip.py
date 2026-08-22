#!/usr/bin/env python3
"""Manual search/select/extract test for short video reference clips."""

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

from src.Video.video_dataset_service import (  # noqa: E402
    ClipExtractionConfig,
    VideoDatasetError,
    extract_reference_clip,
    find_video_candidate,
    list_reference_clips,
    search_video_candidates,
)


def _json_mapping(value: Optional[str], label: str) -> Dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8") if path.exists() else value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError(f"{label} JSON must be an object.")
    return payload


def _print_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No stored video candidates matched the query.")
        return
    for index, item in enumerate(results):
        tags = ", ".join(str(tag) for tag in (item.get("tags") or [])[:6])
        duration = item.get("duration_seconds") or 0
        print(f"[{index}] {item.get('title')}")
        print(f"    video_id: {item.get('video_id')}")
        print(f"    url:      {item.get('url')}")
        print(f"    channel:  {item.get('channel_title') or '-'}")
        print(f"    duration: {duration}s")
        print(f"    license:  {item.get('license_status')} (training_ready={item.get('training_ready')})")
        print(f"    thumb:    {item.get('thumbnail_url') or '-'}")
        print(f"    tags:     {tags or '-'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search stored video candidates and optionally extract a max-20s "
            "local reference clip with metadata."
        )
    )
    parser.add_argument("--query", "-q", default="", help="Search text, e.g. 'lofi rainy window night'.")
    parser.add_argument("--max-results", type=int, default=5, help="Number of search results to show.")
    parser.add_argument("--select-index", type=int, default=None, help="Index from the displayed search results.")
    parser.add_argument("--video-id", default="", help="Direct YouTube video ID.")
    parser.add_argument("--url", default="", help="Direct video URL.")
    parser.add_argument("--clip-duration", type=float, default=20.0, help="Requested duration; capped at 20s.")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--output-format", choices=["mp4", "webm"], default="mp4")
    parser.add_argument("--gif", action="store_true", help="Also render a GIF preview.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary raw video download files.")
    parser.add_argument(
        "--require-known-license",
        action="store_true",
        help="Abort extraction unless metadata contains a training-compatible license.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Override output dataset directory.")
    parser.add_argument("--visual-settings-json", default="", help="Inline JSON object or path to JSON file.")
    parser.add_argument("--track-json", default="", help="Inline JSON object or path to JSON file.")
    parser.add_argument("--list-clips", action="store_true", help="List existing extracted reference clips.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.list_clips:
            clips = list_reference_clips(dataset_dir=args.dataset_dir, limit=args.max_results)
            if args.json:
                print(json.dumps({"clips": clips, "count": len(clips)}, ensure_ascii=False, indent=2))
            else:
                for index, clip in enumerate(clips):
                    source = clip.get("source_video") or {}
                    info = clip.get("clip") or {}
                    print(f"[{index}] {source.get('title')} -> {info.get('path')}")
            return 0

        source: Dict[str, Any] | str | None = None
        selected_index: Optional[int] = None
        results = search_video_candidates(args.query, max_results=args.max_results)

        if args.video_id or args.url:
            source = find_video_candidate(args.video_id or args.url)
        elif args.select_index is not None:
            if args.select_index < 0 or args.select_index >= len(results):
                raise VideoDatasetError(
                    f"--select-index {args.select_index} is outside the result range 0..{len(results) - 1}."
                )
            selected_index = args.select_index
            source = results[selected_index]
        else:
            if args.json:
                print(json.dumps({"query": args.query, "results": results, "count": len(results)}, ensure_ascii=False, indent=2))
            else:
                _print_results(results)
                print("")
                print("Run again with --select-index N to extract a max-20s clip from one result.")
            return 0

        visual_settings = _json_mapping(args.visual_settings_json, "visual settings")
        track_metadata = _json_mapping(args.track_json, "track")
        config = ClipExtractionConfig(
            start_seconds=args.start_seconds,
            clip_duration_seconds=args.clip_duration,
            output_format=args.output_format,
            create_gif=args.gif,
            keep_temp=args.keep_temp,
            require_training_license=args.require_known_license,
        )
        result = extract_reference_clip(
            source,
            dataset_dir=args.dataset_dir,
            config=config,
            query=args.query or None,
            selected_result_index=selected_index,
            visual_settings=visual_settings,
            track_metadata=track_metadata,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("Clip created.")
            print(f"clip_path:     {result.get('clip_path')}")
            if result.get("gif_path"):
                print(f"gif_path:      {result.get('gif_path')}")
            print(f"metadata_path: {result.get('metadata_path')}")
            print(f"manifest_path: {result.get('manifest_path')}")
            training = (result.get("metadata") or {}).get("training") or {}
            print(f"training_ready: {training.get('training_ready')} ({training.get('reason') or 'ok'})")
        return 0
    except VideoDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
