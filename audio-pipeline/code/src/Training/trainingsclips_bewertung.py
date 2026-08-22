#!/usr/bin/env python3
"""Erstellt ein kleines Bewertungspaket aus echten LoRA-Trainingsclips.

Die Datei generiert keine neue Musik und startet kein Training. Sie nimmt
vorhandene 30s-WAV-Clips aus dem fertigen LoRA Dataset, waehlt pro Genre eine
kleine, quellenverteilte Stichprobe und schreibt daraus MP3-Dateien plus eine
Bewertungs-CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJEKTWURZEL = Path(__file__).resolve().parents[3]
TRAINING_DIR = Path(__file__).resolve().parent
STANDARD_DATASET = PROJEKTWURZEL / "daten" / "processed" / "lora_training"
STANDARD_OUTPUT = PROJEKTWURZEL / "training" / "bewertungen" / "musicgen"
SPLITS = ("train", "valid", "test")
GENRES = ("jazz_lofi", "chillhop_lofi", "dreamy_lofi", "study_lofi", "guitar_lofi")


def parse_args() -> argparse.Namespace:
    """Liest die Optionen fuer das Trainingsclip-Review."""

    parser = argparse.ArgumentParser(description="Erstellt 30s-Bewertungsaudios aus echten Trainingsclips.")
    parser.add_argument("--dataset-root", default=str(STANDARD_DATASET))
    parser.add_argument("--output-root", default=str(STANDARD_OUTPUT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--clips-pro-genre", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--split", choices=("train", "valid", "test", "alle"), default="train")
    parser.add_argument("--mp3-bitrate", default="192k")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--nur-plan", action="store_true")
    parser.add_argument(
        "--github-push",
        dest="github_push",
        action="store_true",
        default=True,
        help="Pusht den audio-Ordner nach dem Lauf isoliert nach GitHub/main.",
    )
    parser.add_argument(
        "--kein-github-push",
        dest="github_push",
        action="store_false",
        help="Laesst die Testaudios nur lokal.",
    )
    parser.add_argument("--github-branch", default="main")
    return parser.parse_args()


def jetzt_utc() -> str:
    """UTC-Zeitstempel fuer Reports."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    """Gibt Pfade relativ zur Projektwurzel aus."""

    try:
        return str(path.resolve().relative_to(PROJEKTWURZEL))
    except Exception:
        return str(path)


def slug(text: str) -> str:
    """Erzeugt kurze Dateinamenbestandteile."""

    value = str(text or "").lower().strip()
    value = value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "clip"


def freier_run_name(output_root: Path, wanted: str) -> str:
    """Findet einen freien Ordnernamen."""

    base = wanted or "lora_review_001"
    if not (output_root / base).exists():
        return base
    for index in range(2, 1000):
        candidate = f"{base}_{index:03d}"
        if not (output_root / candidate).exists():
            return candidate
    return f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def lese_jsonl(path: Path) -> list[dict[str, Any]]:
    """Liest eine Manifestdatei."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def lade_rows(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    """Laedt die gewuenschten Dataset-Zeilen."""

    splits = SPLITS if split == "alle" else (split,)
    rows: list[dict[str, Any]] = []
    for split_name in splits:
        manifest = dataset_root / split_name / "data.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest fehlt: {manifest}")
        for row in lese_jsonl(manifest):
            clean = dict(row)
            clean["_split"] = split_name
            rows.append(clean)
    return rows


def genre_key(row: dict[str, Any]) -> str:
    """Liest das Genre aus einer Manifestzeile."""

    return str(row.get("primary_genre") or row.get("lora_genre") or row.get("genre") or "unknown")


def source_key(row: dict[str, Any]) -> str:
    """Liest eine Quellenkennung, damit die Auswahl nicht aus einer Quelle kommt."""

    return str(
        row.get("source_video_id")
        or row.get("source_id")
        or row.get("source_audio_path")
        or row.get("source_file")
        or row.get("source_url")
        or row.get("path")
        or "unknown"
    )


def waehle_clips(rows: list[dict[str, Any]], clips_pro_genre: int, seed: int) -> list[dict[str, Any]]:
    """Waehlt pro Genre Clips aus moeglichst unterschiedlichen Quellen."""

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for genre in GENRES:
        per_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if genre_key(row) != genre:
                continue
            path = Path(str(row.get("path") or "")).expanduser()
            if path.exists() and path.suffix.lower() == ".wav":
                per_source[source_key(row)].append(row)
        for items in per_source.values():
            rng.shuffle(items)
        sources = sorted(per_source)
        rng.shuffle(sources)
        genre_rows: list[dict[str, Any]] = []
        while len(genre_rows) < clips_pro_genre:
            progress = False
            for source in sources:
                if len(genre_rows) >= clips_pro_genre:
                    break
                if not per_source[source]:
                    continue
                genre_rows.append(per_source[source].pop())
                progress = True
            if not progress:
                break
        selected.extend(genre_rows)
    return selected


def schreibe_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Schreibt eine CSV-Datei."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def encode_mp3(source: Path, target: Path, bitrate: str) -> None:
    """Wandelt einen WAV-Clip fuer die Bewertung in MP3 um."""

    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "32000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(target),
    ]
    subprocess.run(command, check=True, cwd=PROJEKTWURZEL)


def push_audio_ordner(audio_dir: Path, run_name: str, branch: str) -> dict[str, Any]:
    """Pusht nur die MP3-Testaudios eines Review-Laufs nach GitHub."""

    if not audio_dir.exists():
        return {"status": "skipped", "reason": f"audio-Ordner fehlt: {rel(audio_dir)}"}
    try:
        if str(TRAINING_DIR) not in sys.path:
            sys.path.insert(0, str(TRAINING_DIR))
        from audio_veroeffentlichen import audio_dateien_aus_eingabe, veroeffentliche_audios

        audio_paths = [
            path
            for path in audio_dateien_aus_eingabe(audio_dir)
            if Path(path).suffix.lower() == ".mp3"
        ]
        if not audio_paths:
            return {"status": "skipped", "reason": "keine MP3-Testaudios gefunden"}
        return veroeffentliche_audios(
            audio_paths,
            branch=branch,
            commit_text=f"Testaudios: {run_name}",
        )
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    """Erzeugt das Bewertungspaket."""

    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    run_name = freier_run_name(output_root, args.run_name)
    run_dir = output_root / run_name
    audio_dir = run_dir / "audio"
    wav_dir = run_dir / "wav"

    if run_dir.exists() and args.overwrite and not args.nur_plan:
        shutil.rmtree(run_dir)
    elif run_dir.exists() and not args.nur_plan:
        raise FileExistsError(f"Zielordner existiert bereits: {run_dir}")

    rows = lade_rows(dataset_root, args.split)
    selected = waehle_clips(rows, args.clips_pro_genre, args.seed)
    expected = args.clips_pro_genre * len(GENRES)
    if len(selected) < expected:
        print(f"Warnung: Nur {len(selected)}/{expected} Clips gefunden.", flush=True)

    bewertung_rows: list[dict[str, str]] = []
    source_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for index, row in enumerate(selected, start=1):
        genre = genre_key(row)
        source = Path(str(row["path"])).expanduser().resolve()
        name = f"sample_{index:03d}__{slug(genre)}.mp3"
        wav_name = f"sample_{index:03d}__{slug(genre)}.wav"
        mp3_target = audio_dir / name
        wav_target = wav_dir / wav_name
        if not args.nur_plan:
            encode_mp3(source, mp3_target, args.mp3_bitrate)
            wav_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, wav_target)
        bewertung_rows.append(
            {
                "Sample": f"sample_{index:03d}",
                "Datei": f"audio/{name}",
                "Genre": genre.replace("_", " ").title(),
                "Status": "",
                "Bewertung / Notiz": "",
            }
        )
        source_info = {
            "sample": f"sample_{index:03d}",
            "genre": genre,
            "split": row.get("_split", ""),
            "datei": f"audio/{name}",
            "wav_kopie": f"wav/{wav_name}",
            "quelle": rel(source),
            "source_key": source_key(row),
            "start_time_sec": row.get("start_time_sec", row.get("clip_start_sec", "")),
            "end_time_sec": row.get("end_time_sec", row.get("clip_end_sec", "")),
            "caption": row.get("caption") or row.get("text") or row.get("description") or "",
        }
        source_rows.append(source_info)
        manifest_rows.append(source_info)

    github_result: dict[str, Any] = {"status": "disabled"}

    if not args.nur_plan:
        run_dir.mkdir(parents=True, exist_ok=True)
        schreibe_csv(
            run_dir / "bewertung.csv",
            bewertung_rows,
            ["Sample", "Datei", "Genre", "Status", "Bewertung / Notiz"],
        )
        schreibe_csv(
            run_dir / "quellen.csv",
            source_rows,
            [
                "sample",
                "genre",
                "split",
                "datei",
                "wav_kopie",
                "quelle",
                "source_key",
                "start_time_sec",
                "end_time_sec",
                "caption",
            ],
        )
        report = {
            "created_at": jetzt_utc(),
            "run_name": run_name,
            "dataset_root": rel(dataset_root),
            "output_dir": rel(run_dir),
            "split": args.split,
            "clips_pro_genre": args.clips_pro_genre,
            "selected_count": len(selected),
            "genres": GENRES,
            "mp3_bitrate": args.mp3_bitrate,
            "manifest": manifest_rows,
        }
        if args.github_push:
            github_result = push_audio_ordner(audio_dir, run_name, args.github_branch)
            report["github_push"] = github_result
        (run_dir / "bericht.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Trainingsclip-Bewertung")
    print("=======================")
    print(f"Dataset: {rel(dataset_root)}")
    print(f"Ziel:    {rel(run_dir)}")
    print(f"Clips:   {len(selected)}")
    print(f"CSV:     {rel(run_dir / 'bewertung.csv')}")
    if not args.nur_plan and args.github_push:
        if github_result.get("status") == "pushed":
            print(f"GitHub: {github_result.get('url')}")
        else:
            detail = github_result.get("error") or github_result.get("reason") or github_result.get("status")
            print(f"GitHub: {github_result.get('status')} ({detail})")
    if args.nur_plan:
        print("Planmodus: Es wurden keine Audios geschrieben.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
