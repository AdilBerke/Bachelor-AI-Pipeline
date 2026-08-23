#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJEKTWURZEL = Path(__file__).resolve().parents[3]
STANDARD_QUELLE = PROJEKTWURZEL / "daten" / "processed" / "musicgen_30s_aus_60s_clips"
STANDARD_ZIEL = PROJEKTWURZEL / "daten" / "processed" / "musicgen_genres_30s"
SPLITS = ("train", "valid", "test")


GENRE_REGELN: dict[str, dict[str, Any]] = {
    "jazz_lofi": {
        "label": "Jazz Lofi",
        "caption": "jazz lofi instrumental, warm rhodes or piano chords, mellow bass, soft drums, relaxed groove, no vocals",
        "tags": {
            "jazz": 4,
            "jazzy": 4,
            "jazzhop": 4,
            "rhodes": 3,
            "sax": 3,
            "saxophone": 3,
            "trumpet": 3,
            "seventh": 2,
        },
    },
    "chillhop_lofi": {
        "label": "Chillhop Lofi",
        "caption": "chillhop lofi instrumental, smooth drums, warm bass, mellow chords, relaxed head nod groove, no vocals",
        "tags": {
            "chillhop": 5,
            "chill hop": 5,
            "hip hop": 2,
            "hiphop": 2,
            "beats": 2,
            "groove": 1,
        },
    },
    "study_lofi": {
        "label": "Study Lofi",
        "caption": "study lofi instrumental, calm focus mood, soft drums, warm bass, mellow chords, stable groove, no vocals",
        "tags": {
            "study": 5,
            "focus": 4,
            "work": 3,
            "coding": 3,
            "concentration": 3,
            "productivity": 2,
        },
    },
    "dreamy_lofi": {
        "label": "Dreamy Lofi",
        "caption": "dreamy lofi instrumental, soft warm texture, mellow piano or pads, gentle drums, relaxed mood, no vocals",
        "tags": {
            "dreamy": 5,
            "dream": 3,
            "nostalgic": 3,
            "soft": 1,
            "mellow": 1,
            "atmosphere": 2,
        },
    },
    "piano_lofi": {
        "label": "Piano Lofi",
        "caption": "piano lofi instrumental, mellow piano chords, warm bass, soft drums, calm relaxed groove, no vocals",
        "tags": {
            "piano": 5,
            "keys": 2,
            "keyboard": 2,
            "felt piano": 4,
        },
    },
    "guitar_lofi": {
        "label": "Guitar Lofi",
        "caption": "guitar lofi instrumental, mellow guitar accents, warm chords, soft drums, controlled bass, no vocals",
        "tags": {
            "guitar": 5,
            "gitarre": 5,
            "electric guitar": 5,
            "acoustic guitar": 4,
        },
    },
    "ambient_lofi": {
        "label": "Ambient Lofi",
        "caption": "ambient lofi instrumental, airy texture, soft drums, warm bass, calm atmosphere, subtle vinyl texture, no vocals",
        "tags": {
            "ambient": 5,
            "atmospheric": 4,
            "rain": 4,
            "rainy": 4,
            "space": 2,
            "texture": 2,
        },
    },
    "night_lofi": {
        "label": "Night Lofi",
        "caption": "night lofi instrumental, late night mood, warm chords, soft drums, mellow bass, smooth calm groove, no vocals",
        "tags": {
            "night": 5,
            "midnight": 5,
            "evening": 3,
            "city": 2,
            "drive": 3,
            "rooftop": 3,
        },
    },
    "boom_bap_lofi": {
        "label": "Boom Bap Lofi",
        "caption": "boom bap lofi instrumental, soft dusty drums, warm bass, mellow sample texture, relaxed hip hop groove, no vocals",
        "tags": {
            "boom bap": 6,
            "boombap": 6,
            "90s": 3,
            "dusty": 3,
            "drums": 1,
        },
    },
    "japanese_lofi": {
        "label": "Japanese Lofi",
        "caption": "japanese inspired lofi instrumental, calm warm chords, soft drums, mellow bass, nostalgic relaxed mood, no vocals",
        "tags": {
            "japanese": 5,
            "japan": 4,
            "tokyo": 4,
            "anime": 4,
            "samurai": 3,
        },
    },
    "general_lofi": {
        "label": "General Lofi",
        "caption": "calm controlled lofi instrumental, soft drums, warm bass, mellow chords, relaxed mood, stable groove, no vocals",
        "tags": {
            "lofi": 1,
            "lo-fi": 1,
            "relaxed": 1,
            "calm": 1,
            "chill": 1,
        },
    },
}


def jetzt_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJEKTWURZEL))
    except Exception:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unterteilt MusicGen-Clips in Lofi-Subgenres.")
    parser.add_argument("--quelle", default=str(STANDARD_QUELLE), help="Bestehendes 30s-Dataset.")
    parser.add_argument("--ziel", default=str(STANDARD_ZIEL), help="Neues genre-basiertes Manifest-Dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Bestehenden Zielordner ueberschreiben.")
    parser.add_argument("--audio-header-pruefen", action="store_true", help="Prueft WAV-Header fuer jede Zeile.")
    parser.add_argument("--min-confidence", type=float, default=0.18, help="Darunter wird General Lofi genutzt.")
    parser.add_argument("--dry-run", action="store_true", help="Nur planen, nichts schreiben.")
    return parser.parse_args()


def lese_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            row["_manifest"] = str(path)
            rows.append(row)
    return rows


def textfelder(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "caption",
        "text",
        "description",
        "title",
        "source_title",
        "genre",
        "instrument",
        "name",
        "source_file",
    ):
        value = row.get(key)
        if isinstance(value, str):
            parts.append(value)
    for key in ("keywords", "moods", "tags"):
        value = row.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, str):
            parts.append(value)
    return " ".join(parts).lower()


def normalisiere_text(text: str) -> str:
    text = text.lower()
    text = text.replace("/", " ").replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def genre_scores(row: dict[str, Any]) -> dict[str, float]:
    text = normalisiere_text(textfelder(row))
    scores: dict[str, float] = {}
    for genre, regel in GENRE_REGELN.items():
        score = 0.0
        for tag, gewicht in regel["tags"].items():
            tag_norm = normalisiere_text(tag)
            if tag_norm in text:
                score += float(gewicht)
        scores[genre] = score


    if str(row.get("teil") or "").upper() == "B":
        for genre in list(scores):
            if scores[genre] > 0:
                scores[genre] += 0.2

    return scores


def bestes_genre(row: dict[str, Any], min_confidence: float) -> tuple[str, float, dict[str, float]]:
    scores = genre_scores(row)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_genre, top_score = sorted_scores[0]
    total = sum(max(0.0, value) for value in scores.values())
    confidence = (top_score / total) if total > 0 else 0.0
    if top_score <= 0 or confidence < min_confidence:
        return "general_lofi", round(confidence, 4), scores
    return top_genre, round(confidence, 4), scores


def quelle(row: dict[str, Any]) -> str:
    return str(row.get("source_video_id") or row.get("source_file") or row.get("source_title") or "unbekannt")


def pruefe_wav_header(path: Path) -> str:
    if not path.exists():
        return "audio_fehlt"
    if path.suffix.lower() != ".wav":
        return "nicht_wav"
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = int(handle.getframerate())
            channels = int(handle.getnchannels())
            duration = handle.getnframes() / float(sample_rate or 1)
    except Exception as exc:
        return f"wav_lesefehler:{type(exc).__name__}"
    if sample_rate != 32000:
        return "sample_rate_falsch"
    if channels != 1:
        return "kanaele_falsch"
    if abs(duration - 30.0) > 0.08:
        return "dauer_falsch"
    return ""


def caption_mit_genre(row: dict[str, Any], genre: str) -> str:
    regel = GENRE_REGELN[genre]
    basis = regel["caption"]
    text = normalisiere_text(textfelder(row))
    extras: list[str] = []
    if "transition" in text or "uebergang" in text or "smooth" in text or "continues" in text:
        extras.append("smooth musical transition")
    if "piano" in text and "piano" not in basis:
        extras.append("mellow piano")
    if "rhodes" in text and "rhodes" not in basis:
        extras.append("warm rhodes")
    if "guitar" in text and "guitar" not in basis:
        extras.append("mellow guitar accents")
    if "vinyl" in text:
        extras.append("subtle vinyl texture")
    if "consistent bpm" in text or "stable" in text:
        extras.append("consistent bpm")
    extras.extend(["controlled bass", "no harsh shaker", "no dropout"])
    unique_extras = list(dict.fromkeys(extras))
    return ", ".join([basis, *unique_extras])


def bereinige_row(row: dict[str, Any], split: str, min_confidence: float) -> dict[str, Any]:
    genre, confidence, scores = bestes_genre(row, min_confidence)
    clean = {k: v for k, v in row.items() if not k.startswith("_")}
    original_caption = str(clean.get("caption") or clean.get("text") or clean.get("description") or "")
    clean["original_caption"] = original_caption
    clean["primary_genre"] = genre
    clean["genre"] = GENRE_REGELN[genre]["label"]
    clean["genre_confidence"] = confidence
    clean["genre_scores"] = {key: round(value, 4) for key, value in scores.items() if value > 0}
    clean["genre_tags"] = [genre, "lofi", "instrumental"]
    clean["caption"] = caption_mit_genre(clean, genre)
    clean["text"] = clean["caption"]
    clean["description"] = clean["caption"]
    clean["split"] = split
    clean["genre_classification_method"] = "rule_based_metadata_v1"
    return clean


def schreibe_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def schreibe_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    quelle_root = Path(args.quelle).expanduser().resolve()
    ziel_root = Path(args.ziel).expanduser().resolve()
    report_root = ziel_root / "reports"

    if not all((quelle_root / split / "data.jsonl").exists() for split in SPLITS):
        raise FileNotFoundError(f"Quelle ist kein gueltiges Manifest-Dataset: {quelle_root}")
    if ziel_root.exists() and any(ziel_root.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Ziel existiert bereits. Nutze --overwrite: {ziel_root}")

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        all_rows[split] = lese_jsonl(quelle_root / split / "data.jsonl")

    genre_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    header_issues: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}

    total = sum(len(rows) for rows in all_rows.values())
    checked = 0
    for split, rows in all_rows.items():
        for row in rows:
            checked += 1
            clean = bereinige_row(row, split, args.min_confidence)
            genre = str(clean["primary_genre"])
            genre_counts[genre] += 1
            split_counts[split][genre] += 1
            source_counts[quelle(row)][genre] += 1
            if len(examples[genre]) < 5:
                examples[genre].append(
                    {
                        "split": split,
                        "path": clean.get("path"),
                        "source_title": clean.get("source_title") or clean.get("title"),
                        "caption": clean.get("caption"),
                        "confidence": clean.get("genre_confidence"),
                    }
                )
            if args.audio_header_pruefen:
                problem = pruefe_wav_header(Path(str(clean.get("path") or "")))
                if problem:
                    header_issues.append(
                        {
                            "split": split,
                            "path": clean.get("path"),
                            "problem": problem,
                            "source_title": clean.get("source_title") or clean.get("title"),
                        }
                    )
            output_rows[split].append(clean)
            if checked % 2000 == 0 or checked == total:
                print(f"\rGenres: {checked}/{total}", end="" if checked < total else "\n", flush=True)

    summary = {
        "created_at": jetzt_utc(),
        "source_dataset": rel(quelle_root),
        "target_dataset": rel(ziel_root),
        "method": "rule_based_metadata_v1",
        "manifest_only": True,
        "wav_files_copied": False,
        "total": total,
        "splits": {split: len(rows) for split, rows in output_rows.items()},
        "genre_counts": dict(genre_counts.most_common()),
        "audio_header_checked": bool(args.audio_header_pruefen),
        "audio_header_issues": len(header_issues),
        "training_ready": not header_issues,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if ziel_root.exists() and args.overwrite:
        import shutil

        shutil.rmtree(ziel_root)
    for split, rows in output_rows.items():
        schreibe_jsonl(ziel_root / split / "data.jsonl", rows)
    ziel_root.mkdir(parents=True, exist_ok=True)
    (ziel_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    genre_rows = [
        {
            "primary_genre": genre,
            "genre": GENRE_REGELN[genre]["label"],
            "count": count,
            "percent": round((count / total) * 100.0, 3) if total else 0.0,
            "train": split_counts["train"].get(genre, 0),
            "valid": split_counts["valid"].get(genre, 0),
            "test": split_counts["test"].get(genre, 0),
        }
        for genre, count in genre_counts.most_common()
    ]
    source_rows = []
    for source, counts in sorted(source_counts.items()):
        top_genre, top_count = counts.most_common(1)[0]
        source_rows.append(
            {
                "source": source,
                "clips": sum(counts.values()),
                "top_genre": top_genre,
                "top_genre_label": GENRE_REGELN[top_genre]["label"],
                "top_genre_count": top_count,
                "all_genres": json.dumps(dict(counts.most_common()), ensure_ascii=False),
            }
        )
    example_rows = []
    for genre, rows in examples.items():
        for row in rows:
            example_rows.append({"primary_genre": genre, "genre": GENRE_REGELN[genre]["label"], **row})

    schreibe_csv(report_root / "genre_counts.csv", genre_rows)
    schreibe_csv(report_root / "source_genres.csv", source_rows)
    schreibe_csv(report_root / "caption_examples.csv", example_rows)
    schreibe_csv(report_root / "audio_header_issues.csv", header_issues)
    (report_root / "genre_report.json").write_text(
        json.dumps({**summary, "examples": examples}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["training_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
