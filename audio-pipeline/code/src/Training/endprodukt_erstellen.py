#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJEKTWURZEL = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio + Video zu einem Endprodukt zusammenfuehren.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--ausgabe-ordner", required=True)
    parser.add_argument("--audio-titel", default="")
    parser.add_argument("--video-label", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    ausgabe_ordner = Path(args.ausgabe_ordner).expanduser().resolve()

    if not audio_path.is_file():
        print(f"Audio nicht gefunden: {audio_path}", file=sys.stderr)
        return 2
    if not video_path.is_file():
        print(f"Video nicht gefunden: {video_path}", file=sys.stderr)
        return 2
    if video_path.suffix.lower() != ".mp4":
        print(
            f"Nur MP4-Videos koennen mit Audio kombiniert werden (kein Tonspur-Format bei GIFs): {video_path}",
            file=sys.stderr,
        )
        return 2

    ausgabe_ordner.mkdir(parents=True, exist_ok=True)
    ziel = ausgabe_ordner / "final.mp4"

    print("Endprodukt", flush=True)
    print("==========", flush=True)
    print(f"Audio: {audio_path}", flush=True)
    print(f"Video: {video_path}", flush=True)
    print(f"Ziel:  {ziel}", flush=True)
    print("", flush=True)

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(ziel),
        ],
        cwd=PROJEKTWURZEL,
    )
    if result.returncode != 0 or not ziel.exists():
        print("Status: fehlgeschlagen (ffmpeg)", file=sys.stderr)
        return result.returncode or 1

    info = {
        "audioPath": str(audio_path.relative_to(PROJEKTWURZEL)),
        "videoPath": str(video_path.relative_to(PROJEKTWURZEL)),
        "audioTitel": args.audio_titel,
        "videoLabel": args.video_label,
        "erstelltAm": datetime.now().isoformat(timespec="seconds"),
    }
    (ausgabe_ordner / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Status: fertig", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
