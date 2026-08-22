#!/usr/bin/env python3
"""Sammelt frisch erzeugte Testaudios in einen gemeinsamen Ordner.

Jeder Genre-Lauf von audio_erstellen.py erzeugt seine eigene
pipeline_<timestamp>_audio/lange_audio.mp3. Fuer den Sammel-GitHub-Push aller
Testaudios auf einmal (statt 5 Einzel-Commits) kopiert dieses Skript die
zuletzt fertiggestellten Audios - eines je uebergebenem Genre, in
Erzeugungsreihenfolge - mit klaren Genre-Dateinamen in einen neuen Ordner.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJEKTWURZEL = Path(__file__).resolve().parents[3]
AUSGABE_ROOT = PROJEKTWURZEL / "training" / "ausgaben" / "musicgen_generiert"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sammelt Testaudios in einen gemeinsamen Ordner.")
    parser.add_argument("--genres", nargs="+", required=True, help="Genre-Slugs in Erzeugungsreihenfolge.")
    parser.add_argument("--sammelordner", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kandidaten = sorted(
        (p for p in AUSGABE_ROOT.glob("pipeline_*_audio") if (p / "lange_audio.mp3").is_file()),
        key=lambda p: (p / "lange_audio.mp3").stat().st_mtime,
    )
    anzahl = len(args.genres)
    if len(kandidaten) < anzahl:
        print(
            f"Fehler: nur {len(kandidaten)} fertige Audios gefunden, {anzahl} erwartet.",
            file=sys.stderr,
        )
        return 2
    neueste = kandidaten[-anzahl:]
    ziel = Path(args.sammelordner)
    ziel.mkdir(parents=True, exist_ok=True)
    for genre_slug, ordner in zip(args.genres, neueste):
        quelle = ordner / "lange_audio.mp3"
        ziel_datei = ziel / f"{genre_slug}.mp3"
        shutil.copy2(quelle, ziel_datei)
        print(f"Kopiert: {quelle} -> {ziel_datei}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
