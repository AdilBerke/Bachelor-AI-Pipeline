#!/usr/bin/env python3
"""Extrahiert technische Merkmale aus MusicGen-Datasets.

Diese Datei ist der eigene Merkmal-Bereich des Projekts. Sie startet kein
Training und erzeugt keine neue Musik. Stattdessen liest sie die
`train/valid/test`-Manifeste eines Datasets, analysiert die referenzierten WAVs
und speichert messbare Audio-Merkmale fuer Reports und Bachelorarbeit.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import warnings
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


warnings.filterwarnings(
    "ignore",
    message="'audioop' is deprecated.*",
    category=DeprecationWarning,
)
import audioop

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "daten" / "processed" / "lora_training"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "training" / "musicgen" / "merkmale"
SPLITS = ("train", "valid", "test")


def now() -> str:
    """Erzeugt einen lokalen Zeitstempel fuer Reports."""
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    """Gibt Pfade moeglichst relativ zur Projektwurzel aus."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Schreibt JSON-Dateien einheitlich mit UTF-8 und Einrueckung."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Liest ein JSONL-Manifest zeilenweise."""
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                yield json.loads(line)


def summarize_numbers(values: List[float]) -> Dict[str, Optional[float]]:
    """Fasst numerische Messwerte fuer einen kompakten Report zusammen."""
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(min(values), 4),
        "mean": round(statistics.fmean(values), 4),
        "max": round(max(values), 4),
    }


def wav_features(path: Path) -> Dict[str, Any]:
    """Berechnet einfache technische WAV-Merkmale ohne schwere ML-Modelle."""
    with wave.open(str(path), "rb") as handle:
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        sample_rate = int(handle.getframerate())
        frames_count = int(handle.getnframes())
        frames = handle.readframes(frames_count)

    max_value = float((2 ** (8 * sample_width - 1)) - 1) if sample_width > 0 else 1.0
    rms_value = float(audioop.rms(frames, sample_width)) if frames else 0.0
    peak_value = float(audioop.max(frames, sample_width)) if frames else 0.0
    crossing_count = int(audioop.cross(frames, sample_width)) if frames else 0
    duration = frames_count / float(sample_rate) if sample_rate else 0.0

    return {
        "duration_sec": round(duration, 6),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "rms_dbfs": round(20.0 * math.log10(max(rms_value / max_value, 1e-12)), 6),
        "peak_dbfs": round(20.0 * math.log10(max(peak_value / max_value, 1e-12)), 6),
        "zero_crossing_rate": round(crossing_count / max(frames_count, 1), 8),
    }


def extract_features(dataset_root: Path, run_dir: Path, limit: int = 0) -> Dict[str, Any]:
    """Extrahiert technische Merkmale aus den Manifest-Audios.

    Pro Clip werden Dauer, Lautheit, Peak und einfache Signalmerkmale
    gespeichert. Diese Daten helfen, Dataset-Qualitaet und Trainingsmaterial
    nachvollziehbar zu dokumentieren.
    """

    out_path = run_dir / "werte" / "audio_merkmale.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0
    failed = 0
    rms_values: List[float] = []
    peak_values: List[float] = []
    split_counts: Dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as out:
        for split in SPLITS:
            manifest = dataset_root / split / "data.jsonl"
            count = 0
            for row in read_jsonl(manifest):
                if limit and total >= limit:
                    break
                total += 1
                count += 1
                audio_path = Path(str(row.get("path") or ""))
                try:
                    features = wav_features(audio_path)
                    ok += 1
                    rms_values.append(float(features["rms_dbfs"]))
                    peak_values.append(float(features["peak_dbfs"]))
                    payload = {
                        "split": split,
                        "path": str(audio_path),
                        "caption": str(row.get("caption") or row.get("text") or row.get("description") or "")[:220],
                        "quality_score": row.get("quality_score"),
                        **features,
                    }
                except Exception as exc:
                    failed += 1
                    payload = {
                        "split": split,
                        "path": str(audio_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            split_counts[split] = count
            if limit and total >= limit:
                break

    summary = {
        "created_at": now(),
        "dataset_root": rel(dataset_root),
        "merkmale_path": rel(out_path),
        "limit": limit,
        "checked": total,
        "ok": ok,
        "failed": failed,
        "split_counts": split_counts,
        "rms_dbfs": summarize_numbers(rms_values),
        "peak_dbfs": summarize_numbers(peak_values),
    }
    write_json(run_dir / "werte" / "merkmale_zusammenfassung.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Definiert die direkte Bedienung der Audio-Merkmale."""
    parser = argparse.ArgumentParser(description="Technische Audio-Merkmale aus einem MusicGen-Dataset extrahieren.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--limit", "--feature-limit", dest="limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    """CLI-Einstiegspunkt fuer die eigenstaendige Merkmal-Extraktion."""
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        DEFAULT_RUN_ROOT / f"merkmale_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    print("Audio-Merkmale", flush=True)
    print("==============", flush=True)
    print(f"Dataset: {rel(dataset_root)}", flush=True)
    print("Training: nein", flush=True)
    print("", flush=True)

    summary = extract_features(dataset_root, run_dir, int(args.limit or 0))

    print(
        f"Clips: {summary['ok']}/{summary['checked']} ok | Fehler: {summary['failed']}",
        flush=True,
    )
    print(f"Report: {summary['merkmale_path']}", flush=True)
    print(f"Zusammenfassung: {rel(run_dir / 'werte' / 'merkmale_zusammenfassung.json')}", flush=True)
    return 0 if int(summary["failed"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
