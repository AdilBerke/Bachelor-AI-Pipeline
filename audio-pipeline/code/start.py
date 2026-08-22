#!/usr/bin/env python3
"""Kurzer Startpunkt fuer die aktuelle MusicGen-Audio-Pipeline.

Diese Datei ist bewusst klein gehalten: Sie ist der zentrale Einstiegspunkt
fuer das Projekt und leitet Argumente an die eigentliche Pipeline-Datei weiter.
Wenn sie ohne Argumente gestartet wird, nutzt sie eine sichere Standardvorgabe:
frische MusicGen-Longform-Generierung mit BPM-Kontrolle. Dadurch startet der
VS-Code-Run-Button nicht versehentlich die alte Loop-/Audit-Pipeline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# Projektwurzel und eigentliche Pipeline-Datei. `code/start.py` liegt eine Ebene
# unter der Projektwurzel, deshalb ist `parents[1]` der Arbeitsordner.
ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "code" / "src" / "Pipeline" / "ablauf.py"
STANDARD_ARGUMENTE = [
    "--stufen",
    "neue_longform",
    "--dauer",
    "3m",
    "--genre",
    "Chill Lofi",
    "--ziel-bpm",
    "78",
    "--bpm-toleranz",
    "4",
    "--stimmung",
    "calm relaxed evening mood",
    "--instrumente",
    "mellow piano, warm bass, soft drums, subtle vinyl texture",
    "--kandidaten-pro-abschnitt",
    "8",
    "--crossfade-sekunden",
    "8",
    "--vram-limit-fraction",
    "0.90",
]


def main() -> int:
    """Prueft die Pipeline-Datei und startet sie mit denselben CLI-Argumenten."""
    if not PIPELINE.exists():
        print(f"Pipeline nicht gefunden: {PIPELINE}", file=sys.stderr)
        return 2
    argumente = sys.argv[1:]
    if not argumente:
        argumente = STANDARD_ARGUMENTE
        print(
            "Keine Argumente angegeben. Starte sichere Vorgabe: "
            "neue_longform, 3m, Chill Lofi, 78 BPM.",
            flush=True,
        )
    # subprocess.call uebergibt die Kontrolle an die Pipeline und liefert deren
    # Exit-Code zurueck. So bleibt `code/start.py` nur ein transparenter Starter.
    return subprocess.call([sys.executable, str(PIPELINE), *argumente], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
