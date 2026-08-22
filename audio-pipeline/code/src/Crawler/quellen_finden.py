#!/usr/bin/env python3
"""Sucht gezielt Quellen fuer das LoRA-Clip-Dataset.

Die Datei startet keine Downloads, kein Clippen und kein Training. Sie liest den
aktuellen Fehlbestand aus dem LoRA-Zieldataset und startet fuer fehlende
Genres automatisch die Top-5-Suche. Dadurch bleibt die Datensammlung
kontrolliert und gleichmaessig ueber mehrere Lofi-Genres verteilt.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJEKTWURZEL = Path(__file__).resolve().parents[3]
PYTHON = PROJEKTWURZEL / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

DATASET_SRC = PROJEKTWURZEL / "code" / "src" / "Dataset"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))
from genre_regeln import GENRE_CAPTIONS, GENRE_LABELS, STANDARD_GENRES

QUELLEN_SUCHE = PROJEKTWURZEL / "code" / "src" / "Crawler" / "quellen_suche.py"
FEHLENDE_CLIPS_CSV = (
    PROJEKTWURZEL
    / "daten"
    / "processed"
    / "lora_training"
    / "reports"
    / "fehlende_clips_pro_genre.csv"
)
REPORT_DIR = PROJEKTWURZEL / "daten" / "metadata" / "crawler"


GENRE_SUCHE = {
    genre_key: {
        "label": GENRE_LABELS.get(genre_key, genre_key.replace("_", " ").title()),
        "stimmung": GENRE_CAPTIONS.get(genre_key, "calm lofi instrumental, no vocals"),
    }
    for genre_key in STANDARD_GENRES
}


def slug(text: str) -> str:
    """Macht kurze Dateinamen aus freien Texten."""
    value = text.strip().lower()
    value = value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    result = []
    for char in value:
        result.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(result).split("_") if part) or "quellen"


def parse_args() -> argparse.Namespace:
    """CLI fuer die gezielte Quellensuche."""
    parser = argparse.ArgumentParser(description="Sucht Quellen fuer fehlende LoRA-Genres.")
    parser.add_argument(
        "--fehlende-csv",
        default="",
        help="Leer = Fehlbestand aus dem LoRA-Dataset.",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--genres", default="", help="Optional: nur bestimmte interne Genres, kommasepariert.")
    parser.add_argument("--lizenz", default="no copyright creative commons")
    parser.add_argument("--max-suchergebnisse", type=int, default=80)
    parser.add_argument("--min-aufrufe", type=int, default=100000)
    parser.add_argument("--min-likes", type=int, default=0)
    parser.add_argument("--dauer-minuten-min", type=float, default=20.0)
    parser.add_argument("--dauer-minuten-max", type=float, default=180.0)
    parser.add_argument("--bekannte-erlauben", action="store_true")
    parser.add_argument("--ohne-detail-metadaten", action="store_true")
    parser.add_argument("--nur-plan", action="store_true", help="Nur anzeigen und Report schreiben, keine Suche starten.")
    return parser.parse_args()


def standard_fehlende_csv() -> Path:
    """Nutzt den Fehlbestand des LoRA-Datasets."""
    return FEHLENDE_CLIPS_CSV


def lese_fehlende(path: Path) -> list[dict[str, Any]]:
    """Liest den Fehlbestand pro Genre aus dem Zieldataset."""
    if not path.exists():
        raise FileNotFoundError(f"Fehlende-Clips-Report fehlt: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            genre = str(row.get("genre") or "").strip()
            if not genre:
                continue
            rows.append(
                {
                    "genre": genre,
                    "target": int(float(row.get("target") or 0)),
                    "available": int(float(row.get("available") or 0)),
                    "selected": int(float(row.get("selected") or 0)),
                    "missing": int(float(row.get("missing") or 0)),
                }
            )
    return rows


def filter_genres(rows: list[dict[str, Any]], wanted: str) -> list[dict[str, Any]]:
    """Filtert optionale Genre-Auswahl und sortiert nach hoechstem Fehlbestand."""
    wanted_set = {item.strip() for item in wanted.split(",") if item.strip()}
    filtered = [row for row in rows if int(row["missing"]) > 0 and row["genre"] in GENRE_SUCHE]
    if wanted_set:
        filtered = [row for row in filtered if row["genre"] in wanted_set]
    return sorted(filtered, key=lambda item: int(item["missing"]), reverse=True)


def top10_command(args: argparse.Namespace, run_name: str, row: dict[str, Any]) -> list[str]:
    """Baut den Top-5-Suchbefehl fuer ein Genre."""
    genre_info = GENRE_SUCHE[row["genre"]]
    output_name = f"{run_name}_{slug(str(row['genre']))}"
    command = [
        str(PYTHON),
        str(QUELLEN_SUCHE.relative_to(PROJEKTWURZEL)),
        "top10",
        "--genre",
        genre_info["label"],
        "--stimmung",
        genre_info["stimmung"],
        "--lizenz",
        args.lizenz,
        "--dauer-minuten-min",
        str(args.dauer_minuten_min),
        "--dauer-minuten-max",
        str(args.dauer_minuten_max),
        "--max-suchergebnisse",
        str(args.max_suchergebnisse),
        "--min-aufrufe",
        str(args.min_aufrufe),
        "--min-likes",
        str(args.min_likes),
        "--output-name",
        output_name,
    ]
    if not args.bekannte_erlauben:
        command.append("--ausschliessen-bekannte")
    if args.ohne_detail_metadaten:
        command.append("--ohne-detail-metadaten")
    return command


def top10_csv_hat_zeilen(command: list[str]) -> bool:
    """Prueft nach einem Suchlauf, ob die erwartete Top-5-CSV Eintraege hat."""

    if "--output-name" not in command:
        return False
    index = command.index("--output-name")
    if index + 1 >= len(command):
        return False
    output_name = command[index + 1]
    csv_path = REPORT_DIR / f"top10_{output_name}.csv"
    if not csv_path.exists():
        return False
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle)) > 0


def schreibe_report(path: Path, payload: dict[str, Any]) -> None:
    """Schreibt den Suchlauf-Report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Startet die genreweise Top-5-Suche fuer fehlende Clips."""
    args = parse_args()
    run_name = slug(args.run_name or f"top10_quellen_5000_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    fehlende_csv = Path(args.fehlende_csv).expanduser().resolve() if args.fehlende_csv else standard_fehlende_csv()
    rows = filter_genres(lese_fehlende(fehlende_csv), args.genres)

    plan: list[dict[str, Any]] = []
    for row in rows:
        command = top10_command(args, run_name, row)
        plan.append(
            {
                "genre": row["genre"],
                "label": GENRE_SUCHE[row["genre"]]["label"],
                "missing": row["missing"],
                "command": command,
                "status": "planned",
            }
        )

    report_path = REPORT_DIR / f"{run_name}.json"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "fehlende_csv": str(fehlende_csv),
        "nur_plan": bool(args.nur_plan),
        "genres": plan,
        "hinweis": "Nur Top-5-Quellensuche. Kein Download, kein Clippen, kein Training.",
    }

    print("Quellensuche fuer 5000 Clips")
    print("============================")
    if not plan:
        print("Keine fehlenden bekannten Genres gefunden.")
        schreibe_report(report_path, payload)
        print(f"Report: {report_path}")
        return 0

    for item in plan:
        print(f"- {item['label']}: fehlen {item['missing']} Clips")

    if args.nur_plan:
        print("\nNur Plan: Es wurde keine Suche gestartet.")
        schreibe_report(report_path, payload)
        print(f"Report: {report_path}")
        return 0

    errors = 0
    for item in plan:
        print(f"\nSuche: {item['label']}")
        result = subprocess.run(item["command"], cwd=PROJEKTWURZEL)
        item["returncode"] = result.returncode
        item["status"] = "ok" if result.returncode == 0 else "fehler"
        if result.returncode != 0 or not top10_csv_hat_zeilen(item["command"]):
            # Bewusst kein Fallback mit gelockerten Kriterien: Wenn die Top-5-Suche
            # leer bleibt oder ein Video nicht automatisiert geladen werden kann, wird
            # keine schwaechere Quelle nachgeladen. Die Luecke bleibt sichtbar und wird
            # manuell (Link oder Datei-Upload) in der Website geschlossen.
            item["status"] = "leer" if result.returncode == 0 else "fehler"
            errors += 1

    schreibe_report(report_path, payload)
    print("\nFertig.")
    print(f"Report: {report_path}")
    print("Naechster Schritt: fehlende Genres pruefen und passende Quellen manuell (Link/Upload) importieren.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
