#!/usr/bin/env python3
"""Erstellt ein kuratiertes MusicGen-Dataset fuer Melody und Uebergaenge.

Das Skript veraendert das bestehende Dataset nicht. Es liest die vorhandenen
30s-Manifeste, waehlt passende Clips aus und schreibt ein neues Manifest-Dataset
mit verbesserten Captions. Die WAV-Dateien werden aus Speichergruenden nicht
kopiert; die neuen Manifest-Zeilen zeigen weiter auf die bestehenden WAVs.

Ziel fuer das LoRA-Training:
- mehr Clips mit klarer Melodie
- mehr Clips mit kontrollierten Uebergaengen
- weniger unklare/generische Trainingsbeispiele
- weniger Dominanz einzelner Quellen
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJEKTWURZEL = Path(__file__).resolve().parents[3]
QUELL_DATASET = PROJEKTWURZEL / "daten" / "processed" / "musicgen_30s_aus_60s_clips"
ZIEL_DATASET = PROJEKTWURZEL / "daten" / "processed" / "musicgen_kuratiert_uebergang_melodie_30s"
SPLITS = ("train", "valid", "test")
ERWARTETE_DAUER = 30.0
ERWARTETE_SAMPLE_RATE = 32000
ERWARTETE_KANAELE = 1

MELODIE_WOERTER = (
    "melody",
    "melodic",
    "piano",
    "rhodes",
    "guitar",
    "jazz",
    "jazzy",
    "harmony",
    "chords",
    "sax",
    "lead",
)
UEBERGANG_WOERTER = (
    "transition",
    "bridge",
    "continues",
    "change",
    "smooth",
    "next musical idea",
    "previous musical idea",
)
PROBLEM_WOERTER = (
    "aggressive",
    "hard drop",
    "harsh",
    "distorted",
    "vocals",
    "vocal",
    "rap",
    "trap",
)


def jetzt_utc() -> str:
    """UTC-Zeitstempel fuer reproduzierbare Reports."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    """Schreibt Pfade in Reports moeglichst relativ zum Projekt."""

    try:
        return str(path.resolve().relative_to(PROJEKTWURZEL))
    except Exception:
        return str(path)


def parse_args() -> argparse.Namespace:
    """Definiert die Bedienung des Kuratierungsschritts."""

    parser = argparse.ArgumentParser(description="Erstellt ein kuratiertes Melody/Uebergang-Dataset.")
    parser.add_argument("--quelle", default=str(QUELL_DATASET), help="Bestehendes 30s-Dataset.")
    parser.add_argument("--ziel", default=str(ZIEL_DATASET), help="Neues Manifest-Dataset.")
    parser.add_argument("--max-clips", type=int, default=12000, help="0 = alle passenden Clips.")
    parser.add_argument("--max-clips-pro-quelle", type=int, default=450)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--melodie-anteil", type=float, default=0.45)
    parser.add_argument("--uebergang-anteil", type=float, default=0.45)
    parser.add_argument("--kontext-anteil", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--audio-header-pruefen", action="store_true", help="Prueft WAV-Header vor Aufnahme.")
    parser.add_argument("--overwrite", action="store_true", help="Bestehenden Zielordner ersetzen.")
    parser.add_argument("--dry-run", action="store_true", help="Nur planen, nichts schreiben.")
    return parser.parse_args()


def lese_jsonl(path: Path) -> list[dict[str, Any]]:
    """Liest ein JSONL-Manifest."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_manifest"] = str(path)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def text_fuer_row(row: dict[str, Any]) -> str:
    """Sammelt alle Textfelder fuer einfache Heuristiken."""

    parts: list[str] = []
    for key in ("caption", "text", "description", "title", "source_title", "genre", "instrument", "name"):
        value = row.get(key)
        if isinstance(value, str):
            parts.append(value)
    for key in ("keywords", "moods"):
        value = row.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, str):
            parts.append(value)
    return " ".join(parts).lower()


def enthaelt(text: str, woerter: tuple[str, ...]) -> bool:
    """Prueft, ob eines der Stichworte im Text vorkommt."""

    return any(wort in text for wort in woerter)


def clip_typ(row: dict[str, Any]) -> str:
    """Ordnet einen Clip einem Trainingsfokus zu."""

    text = text_fuer_row(row)
    teil = str(row.get("teil") or "").strip().upper()
    hat_melodie = enthaelt(text, MELODIE_WOERTER)
    hat_uebergang = enthaelt(text, UEBERGANG_WOERTER)
    if teil == "A" and hat_melodie and hat_uebergang:
        return "melodie_mit_uebergang_ende"
    if teil == "B" and hat_melodie and hat_uebergang:
        return "uebergang_in_melodie"
    if hat_melodie and hat_uebergang:
        return "melodie_und_uebergang"
    if hat_melodie:
        return "melodie"
    if hat_uebergang or teil in {"A", "B"}:
        return "uebergang"
    return "ruhiger_kontext"


def score_row(row: dict[str, Any]) -> tuple[float, list[str]]:
    """Bewertet, wie passend eine Manifest-Zeile fuer das kuratierte Dataset ist."""

    text = text_fuer_row(row)
    gruende: list[str] = []
    score = 0.0
    if "lofi" in text or "lo-fi" in text:
        score += 0.20
        gruende.append("lofi")
    if enthaelt(text, MELODIE_WOERTER):
        score += 0.30
        gruende.append("melodie")
    if enthaelt(text, UEBERGANG_WOERTER):
        score += 0.30
        gruende.append("uebergang")
    if "no vocals" in text:
        score += 0.05
    if "no abrupt" in text or "smooth" in text:
        score += 0.10
    if str(row.get("teil") or "").strip().upper() in {"A", "B"}:
        score += 0.05
    if enthaelt(text, PROBLEM_WOERTER):
        score -= 0.25
        gruende.append("problemwort")
    return round(max(0.0, min(1.0, score)), 4), gruende


def quelle(row: dict[str, Any]) -> str:
    """Ermittelt eine stabile Quellenkennung fuer Balancing."""

    return str(row.get("source_video_id") or row.get("source_file") or row.get("source_title") or "unbekannt")


def pruefe_wav_header(path: Path) -> str:
    """Prueft Dauer, Sample-Rate und Kanaele ohne das ganze Audio zu dekodieren."""

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
    if sample_rate != ERWARTETE_SAMPLE_RATE:
        return "sample_rate_falsch"
    if channels != ERWARTETE_KANAELE:
        return "kanaele_falsch"
    if abs(duration - ERWARTETE_DAUER) > 0.08:
        return "dauer_falsch"
    return ""


def caption_fuer_typ(row: dict[str, Any], typ: str) -> str:
    """Erzeugt eine genauere Caption fuer LoRA-Training."""

    text = text_fuer_row(row)
    instrumente = str(row.get("instrument") or "").strip()
    basis = "calm controlled lofi instrumental, soft drums, warm bass, relaxed mood, no vocals"
    if "jazz" in text or "jazzy" in text:
        basis = "calm controlled jazz lofi instrumental, warm chords, soft drums, mellow bass, no vocals"
    if "guitar" in text:
        instrumente = "mellow guitar accents, warm chords, soft drums, controlled bass"
    elif "piano" in text or "rhodes" in text or not instrumente:
        instrumente = "mellow piano or rhodes melody, warm chords, soft drums, controlled bass"

    if typ == "melodie_mit_uebergang_ende":
        fokus = "clear melodic lofi phrase that leads into a smooth musical transition near the end, consistent bpm, no abrupt drop"
    elif typ == "uebergang_in_melodie":
        fokus = "smooth transition from a previous lofi idea into a clear melodic phrase, stable groove, consistent bpm"
    elif typ == "melodie_und_uebergang":
        fokus = "clear melodic phrase with a smooth musical transition, consistent bpm, no abrupt rhythm change"
    elif typ == "melodie":
        fokus = "clear melodic phrase, harmonic movement, stable 30 second lofi groove, no monotone second half"
    elif typ == "uebergang":
        fokus = "smooth lofi transition between musical ideas, controlled groove change, stable tempo, no hard drop"
    else:
        fokus = "stable lofi groove context, calm rhythm, consistent bpm, warm harmonic background"
    return f"{basis}, {instrumente}, {fokus}"


def ziel_split(index: int, seed: int) -> str:
    """Erzeugt reproduzierbare 90/5/5-Splits fuer das neue Dataset."""

    rng = random.Random(seed + index * 104729)
    value = rng.random()
    if value < 0.90:
        return "train"
    if value < 0.95:
        return "valid"
    return "test"


def ziel_metadaten_pfad(ziel: Path, split: str, row: dict[str, Any], index: int) -> Path:
    """Legt den Ort fuer die neue Sidecar-Metadatei fest."""

    name = str(row.get("name") or Path(str(row.get("path") or f"clip_{index:06d}")).stem)
    kurz = hashlib.sha1(f"{name}:{index}".encode("utf-8")).hexdigest()[:8]
    sauber = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")[:120] or f"clip_{index:06d}"
    return ziel / split / "metadata" / f"{sauber}__{kurz}.json"


def schreibe_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Schreibt Manifest-Zeilen."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def schreibe_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Schreibt CSV-Reports."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [{"status": "keine"}])


def waehle_balanciert(kandidaten: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Waehlt Clips mit Typ- und Quellen-Balancing aus."""

    rng = random.Random(args.seed)
    nach_typ: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kandidaten:
        nach_typ[row["_kuratiert_typ"]].append(row)
    for rows in nach_typ.values():
        rng.shuffle(rows)
        rows.sort(key=lambda item: item["_kuratiert_score"], reverse=True)

    max_clips = int(args.max_clips) if int(args.max_clips) > 0 else len(kandidaten)
    ziele = {
        "melodie_mit_uebergang_ende": int(max_clips * args.melodie_anteil),
        "uebergang_in_melodie": int(max_clips * args.uebergang_anteil),
        "melodie_und_uebergang": int(max_clips * 0.15),
        "melodie": int(max_clips * 0.10),
        "uebergang": int(max_clips * 0.10),
        "ruhiger_kontext": int(max_clips * args.kontext_anteil),
    }

    ausgewaehlt: list[dict[str, Any]] = []
    quelle_count: Counter[str] = Counter()

    def nimm_aus_typ(typ: str, limit: int) -> None:
        for row in nach_typ.get(typ, []):
            if len(ausgewaehlt) >= max_clips:
                return
            if sum(1 for item in ausgewaehlt if item["_kuratiert_typ"] == typ) >= limit:
                return
            key = quelle(row)
            if quelle_count[key] >= args.max_clips_pro_quelle:
                continue
            quelle_count[key] += 1
            ausgewaehlt.append(row)

    for typ, limit in ziele.items():
        nimm_aus_typ(typ, limit)

    rest = [row for row in kandidaten if row not in ausgewaehlt]
    rest.sort(key=lambda item: item["_kuratiert_score"], reverse=True)
    for row in rest:
        if len(ausgewaehlt) >= max_clips:
            break
        key = quelle(row)
        if quelle_count[key] >= args.max_clips_pro_quelle:
            continue
        quelle_count[key] += 1
        ausgewaehlt.append(row)

    rng.shuffle(ausgewaehlt)
    return ausgewaehlt


def main() -> int:
    """Erstellt das kuratierte Manifest-Dataset."""

    args = parse_args()
    quelle_root = Path(args.quelle).expanduser().resolve()
    ziel_root = Path(args.ziel).expanduser().resolve()
    if not quelle_root.exists():
        print(f"Quelle fehlt: {quelle_root}")
        return 2
    if ziel_root.exists() and not args.overwrite and not args.dry_run:
        print(f"Ziel existiert bereits: {ziel_root}")
        print("Nutze --overwrite oder einen anderen --ziel Pfad.")
        return 2

    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        manifest = quelle_root / split / "data.jsonl"
        rows.extend(lese_jsonl(manifest))

    kandidaten: list[dict[str, Any]] = []
    abgelehnt: list[dict[str, Any]] = []
    for row in rows:
        path = Path(str(row.get("path") or ""))
        score, gruende = score_row(row)
        typ = clip_typ(row)
        grund = ""
        if score < args.min_score:
            grund = "score_zu_niedrig"
        elif args.audio_header_pruefen:
            grund = pruefe_wav_header(path)
        if grund:
            abgelehnt.append(
                {
                    "path": str(row.get("path") or ""),
                    "split": row.get("split"),
                    "typ": typ,
                    "score": score,
                    "grund": grund,
                    "score_gruende": "; ".join(gruende),
                }
            )
            continue
        enriched = dict(row)
        enriched["_kuratiert_typ"] = typ
        enriched["_kuratiert_score"] = score
        enriched["_kuratiert_gruende"] = gruende
        kandidaten.append(enriched)

    ausgewaehlt = waehle_balanciert(kandidaten, args)
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    report_rows: list[dict[str, Any]] = []
    for index, row in enumerate(ausgewaehlt):
        split = ziel_split(index, args.seed)
        typ = row["_kuratiert_typ"]
        caption = caption_fuer_typ(row, typ)
        metadata_path = ziel_metadaten_pfad(ziel_root, split, row, index)
        new_row = {key: value for key, value in row.items() if not key.startswith("_")}
        new_row.update(
            {
                "split": split,
                "caption": caption,
                "text": caption,
                "description": caption,
                "curation_type": typ,
                "training_focus": typ,
                "curation_score": row["_kuratiert_score"],
                "curation_reasons": row["_kuratiert_gruende"],
                "curated_from_dataset": rel(quelle_root),
                "metadata_path": str(metadata_path.resolve()),
            }
        )
        rows_by_split[split].append(new_row)
        report_rows.append(
            {
                "path": new_row["path"],
                "split": split,
                "source_video_id": new_row.get("source_video_id"),
                "teil": new_row.get("teil"),
                "curation_type": typ,
                "curation_score": row["_kuratiert_score"],
                "caption": caption,
            }
        )

    summary = {
        "created_at": jetzt_utc(),
        "source_dataset": rel(quelle_root),
        "target_dataset": rel(ziel_root),
        "source_rows": len(rows),
        "candidate_rows": len(kandidaten),
        "selected_rows": len(ausgewaehlt),
        "rejected_rows": len(abgelehnt),
        "splits": {split: len(rows_by_split[split]) for split in SPLITS},
        "types": dict(Counter(row["_kuratiert_typ"] for row in ausgewaehlt)),
        "sources": len(set(quelle(row) for row in ausgewaehlt)),
        "max_clips_per_source": args.max_clips_pro_quelle,
        "min_score": args.min_score,
        "audio_header_checked": bool(args.audio_header_pruefen),
        "manifest_only": True,
        "wav_files_copied": False,
        "training_ready": bool(ausgewaehlt),
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if ziel_root.exists():
        shutil.rmtree(ziel_root)
    for split, split_rows in rows_by_split.items():
        schreibe_jsonl(ziel_root / split / "data.jsonl", split_rows)
        for row in split_rows:
            metadata_path = Path(str(row["metadata_path"]))
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    reports = ziel_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (ziel_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports / "kuratierungsbericht.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    schreibe_csv(reports / "ausgewaehlte_clips.csv", report_rows)
    schreibe_csv(reports / "abgelehnte_clips.csv", abgelehnt)

    print("Kuratiertes Dataset erstellt.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
