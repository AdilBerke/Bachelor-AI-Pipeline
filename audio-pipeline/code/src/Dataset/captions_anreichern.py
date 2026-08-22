#!/usr/bin/env python3
"""Reichert die statischen Genre-Captions im LoRA-Trainingsdatensatz um echte,
pro Clip gemessene Audiomerkmale an.

Bisher bekam jeder Clip eines Genres exakt dieselbe Caption (siehe
GENRE_CAPTIONS in genre_regeln.py). Dieses Skript berechnet fuer jeden Clip
echte technische Merkmale (BPM, Bassanteil, Snareanteil, Hoehenanteil,
Dynamikspanne - dieselben Werte, die auch fuer die Bewertung genutzt werden)
und haengt daraus abgeleitete, kurze Textbausteine an die bestehende
Genre-Caption an. Es wird nichts erfunden: jeder Textbaustein entspricht
einem tatsaechlich gemessenen Schwellenwert.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src" / "Training"))
from audio_bewertung import decode_audio, score_kennwerte  # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "daten" / "processed" / "lora_training"


def merkmal_fragmente(m: Dict[str, float]) -> list[str]:
    """Leitet kurze Textbausteine aus gemessenen Werten ab (keine Erfindung)."""

    frags: list[str] = []
    bpm = m.get("bpm", 0.0)
    if bpm < 65:
        frags.append("slow, spacious tempo")
    elif bpm > 85:
        frags.append("slightly faster tempo")

    bass = m.get("bass_ratio", 0.0)
    if bass > 0.4:
        frags.append("bass-forward low end")
    elif bass < 0.2:
        frags.append("light, restrained low end")

    snare = m.get("snare_ratio", 0.0)
    if snare > 0.18:
        frags.append("more prominent percussion")
    elif snare < 0.06:
        frags.append("soft, minimal percussion")

    high = m.get("high_ratio", 0.0)
    if high > 0.06:
        frags.append("airy high end")

    rms_range = m.get("rms_range_db", 0.0)
    if rms_range > 12:
        frags.append("notable dynamic variation")
    elif rms_range < 5:
        frags.append("steady, consistent volume")

    return frags


def reichere_datei_an(path: Path) -> tuple[int, int]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    verarbeitet = 0
    fehler = 0
    for row in rows:
        clip_path = Path(str(row.get("path", "")))
        try:
            audio = decode_audio(clip_path)
            m = score_kennwerte(audio)
        except Exception:
            fehler += 1
            continue
        basis = str(row.get("caption", "")).rstrip(", ")
        frags = merkmal_fragmente(m)
        if frags:
            neue_caption = basis + ", " + ", ".join(frags)
        else:
            neue_caption = basis
        row["caption"] = neue_caption
        row["text"] = neue_caption
        row["description"] = neue_caption
        row["caption_merkmale"] = {
            "bpm": round(m.get("bpm", 0.0), 1),
            "bass_ratio": round(m.get("bass_ratio", 0.0), 4),
            "snare_ratio": round(m.get("snare_ratio", 0.0), 4),
            "high_ratio": round(m.get("high_ratio", 0.0), 4),
            "rms_range_db": round(m.get("rms_range_db", 0.0), 2),
        }
        verarbeitet += 1

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return verarbeitet, fehler


def main() -> int:
    zusammenfassung: Dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        path = DATASET_ROOT / split / "data.jsonl"
        if not path.exists():
            continue
        verarbeitet, fehler = reichere_datei_an(path)
        zusammenfassung[split] = {"verarbeitet": verarbeitet, "fehler": fehler}
        print(json.dumps({"split": split, "verarbeitet": verarbeitet, "fehler": fehler}), flush=True)
    print(json.dumps({"ergebnis": zusammenfassung}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
