#!/usr/bin/env python3
"""Baut eine echte Referenzliste fuer genre_pruefung.py (CLAP-Genre-Check).

Die CLAP-basierte Genre-Pruefung (genre_pruefung.py) braucht eine
`bewertung.csv` mit als "Gut" markierten Referenzclips pro Genre, um
Genre-Zentren zu berechnen. Der bisherige Standardordner
(training/bewertungen/musicgen/lora_review_001) stammt aus der
abgeschafften manuellen Human-Review-Funktion und wurde nie ausgefuellt
(Status-Spalte durchgehend leer) - die Pruefung war dadurch unbenutzbar
und musste ueberall per --genre-pruefung-deaktivieren abgeschaltet werden.

Dieses Skript baut stattdessen eine echte Referenzliste direkt aus den
bereits genre-sortierten LoRA-Trainingsclips (daten/processed/lora_training),
verteilt ueber moeglichst viele unabhaengige Quellen pro Genre, und schreibt
sie als bewertung.csv (mit absoluten Pfaden, keine Audiokopien noetig).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENZ_ROOT = PROJECT_ROOT / "daten" / "processed" / "lora_training"
DEFAULT_ZIEL_ORDNER = (
    PROJECT_ROOT / "training" / "bewertungen" / "musicgen" / "clap_referenzen_training"
)


def _manifest_zeilen(referenz_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        path = referenz_root / split / "data.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _diverse_auswahl(rows: List[Dict[str, Any]], anzahl: int) -> List[Dict[str, Any]]:
    """Waehlt Clips reihum ueber moeglichst viele Quellen (source_id) aus."""

    nach_quelle: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        quelle = str(row.get("source_id") or row.get("source_file") or "unbekannt")
        nach_quelle[quelle].append(row)
    for quelle in nach_quelle:
        nach_quelle[quelle].sort(key=lambda r: str(r.get("path") or ""))

    ausgewaehlt: List[Dict[str, Any]] = []
    quellen = sorted(nach_quelle.keys())
    index = 0
    while len(ausgewaehlt) < anzahl and any(nach_quelle[q] for q in quellen):
        quelle = quellen[index % len(quellen)]
        if nach_quelle[quelle]:
            ausgewaehlt.append(nach_quelle[quelle].pop(0))
        index += 1
        if index > anzahl * len(quellen) + len(quellen):
            break
    return ausgewaehlt


def baue_referenzen(referenz_root: Path, ziel_ordner: Path, pro_genre: int) -> Dict[str, int]:
    rows = _manifest_zeilen(referenz_root)
    nach_genre: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        genre = row.get("lora_genre") or row.get("primary_genre") or row.get("genre")
        if not genre:
            continue
        pfad = Path(str(row.get("path") or ""))
        if pfad.is_file():
            nach_genre[str(genre)].append(row)

    ziel_ordner.mkdir(parents=True, exist_ok=True)
    csv_pfad = ziel_ordner / "bewertung.csv"
    zusammenfassung: Dict[str, int] = {}

    with csv_pfad.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample", "Datei", "Genre", "Status", "Bewertung / Notiz"])
        sample_index = 0
        for genre in sorted(nach_genre.keys()):
            auswahl = _diverse_auswahl(nach_genre[genre], pro_genre)
            for row in auswahl:
                sample_index += 1
                writer.writerow(
                    [
                        f"training_ref_{sample_index:04d}",
                        str(Path(str(row.get("path"))).resolve()),
                        genre,
                        "Gut",
                        "auto: echter, genre-sortierter LoRA-Trainingsclip",
                    ]
                )
            zusammenfassung[genre] = len(auswahl)

    return zusammenfassung


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baut eine echte bewertung.csv fuer die CLAP-Genre-Pruefung aus den LoRA-Trainingsclips."
    )
    parser.add_argument("--referenz-root", default=str(DEFAULT_REFERENZ_ROOT))
    parser.add_argument("--ziel-ordner", default=str(DEFAULT_ZIEL_ORDNER))
    parser.add_argument("--pro-genre", type=int, default=20)
    args = parser.parse_args()

    zusammenfassung = baue_referenzen(
        Path(args.referenz_root).expanduser().resolve(),
        Path(args.ziel_ordner).expanduser().resolve(),
        max(2, int(args.pro_genre)),
    )
    print(json.dumps({"ziel_ordner": args.ziel_ordner, "clips_pro_genre": zusammenfassung}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
