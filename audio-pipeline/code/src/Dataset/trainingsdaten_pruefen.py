#!/usr/bin/env python3
"""Erstellt gepruefte Trainingsdaten aus menschlicher Bewertung.

Die Datei kopiert keine WAV-Dateien und startet kein Training. Sie liest das
LoRA Manifest-Dataset, uebernimmt deine Review-Entscheidung aus
`lora_review_001` und schreibt ein neues Manifest-Dataset:

- Quellen mit Status `Schlecht` werden komplett ausgeschlossen.
- Quellen mit Status `Pruefen` bleiben enthalten, werden aber markiert.
- Gute Quellen bleiben normal enthalten.

So kann LoRA spaeter mit demselben Clip-Pool arbeiten, ohne bekannte schlechte
Quellen erneut zu trainieren.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genre_regeln import (
    STANDARD_GENRES,
    bereinige_manifestzeile,
    erkenne_genre as erkenne_genre_gemeinsam,
    quellen_schluessel,
    waehle_quellengetrennt,
)


PROJEKTWURZEL = Path(__file__).resolve().parents[3]
STANDARD_BASIS = PROJEKTWURZEL / "daten" / "processed" / "lora_training"
STANDARD_QUELLE = PROJEKTWURZEL / "daten" / "processed" / "musicgen_youtube_import_30s"
STANDARD_ZIEL = PROJEKTWURZEL / "daten" / "processed" / "lora_training_geprueft"
STANDARD_BEWERTUNG = (
    PROJEKTWURZEL / "training" / "bewertungen" / "musicgen" / "lora_review_001" / "bewertung.csv"
)
STANDARD_QUELLEN = (
    PROJEKTWURZEL / "training" / "bewertungen" / "musicgen" / "lora_review_001" / "quellen.csv"
)
STANDARD_POLICY_DIR = PROJEKTWURZEL / "daten" / "metadata" / "trainingsdaten_pruefung"
STANDARD_PLAN = PROJEKTWURZEL / "training" / "musicgen" / "dataset_pruefung"
SPLITS = ("train", "valid", "test")


def jetzt_utc() -> str:
    """UTC-Zeitstempel fuer reproduzierbare Reports."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    """Gibt Projektpfade kurz aus."""
    try:
        return str(path.resolve().relative_to(PROJEKTWURZEL))
    except Exception:
        return str(path)


def parse_args() -> argparse.Namespace:
    """CLI fuer die sichere Dataset-Bereinigung."""
    parser = argparse.ArgumentParser(description="Erstellt gepruefte MusicGen-Trainingsdaten.")
    parser.add_argument("--basis", default=str(STANDARD_BASIS))
    parser.add_argument("--quelle", default=str(STANDARD_QUELLE), help="Alle vorhandenen Import-Datasets zum Nachfuellen.")
    parser.add_argument("--ziel", default=str(STANDARD_ZIEL))
    parser.add_argument("--bewertung", default=str(STANDARD_BEWERTUNG))
    parser.add_argument("--quellen", default=str(STANDARD_QUELLEN))
    parser.add_argument("--policy-dir", default=str(STANDARD_POLICY_DIR))
    parser.add_argument("--ziel-clips", type=int, default=5000)
    parser.add_argument("--genres", default=",".join(STANDARD_GENRES))
    parser.add_argument("--seed", type=int, default=4027)
    parser.add_argument("--max-pro-quelle-pro-genre", type=int, default=250)
    parser.add_argument("--pruefen-ausschliessen", action="store_true")
    parser.add_argument(
        "--nur-basis",
        action="store_true",
        help="Nur das LoRA Basisdataset filtern, nicht aus allen Importclips nachfuellen.",
    )
    parser.add_argument("--nur-plan", action="store_true")
    return parser.parse_args()


def lese_csv(path: Path) -> list[dict[str, str]]:
    """Liest CSV-Dateien defensiv."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def lese_jsonl(path: Path) -> list[dict[str, Any]]:
    """Liest ein Manifest."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} ist kein JSON-Objekt.")
            rows.append(row)
    return rows


def schreibe_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Schreibt ein Manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def schreibe_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Schreibt einen CSV-Report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def manifest_roots(root: Path) -> list[Path]:
    """Findet ein Dataset oder alle Import-Unterdatasets."""
    root = root.expanduser().resolve()
    if all((root / split / "data.jsonl").exists() for split in SPLITS):
        return [root]
    found = sorted({path.parents[1] for path in root.glob("**/train/data.jsonl")})
    return [item for item in found if all((item / split / "data.jsonl").exists() for split in SPLITS)]


def normalisiere_status(value: str) -> str:
    """Vereinheitlicht menschliche Statuswerte."""
    text = str(value or "").strip().lower()
    if text in {"gut", "ok", "sehr gut"}:
        return "Gut"
    if text in {"schlecht", "nicht gut", "ablehnen"}:
        return "Schlecht"
    if text in {"pruefen", "prüfen", "check", "unsicher"}:
        return "Pruefen"
    return "Pruefen"


def lade_review_policy(bewertung_path: Path, quellen_path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Verknuepft Review-Samples mit ihren echten Quell-IDs."""
    bewertungen = {row.get("Sample", "").strip(): row for row in lese_csv(bewertung_path)}
    quellen = lese_csv(quellen_path)
    policy: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for quelle in quellen:
        sample = quelle.get("sample", "").strip()
        review = bewertungen.get(sample, {})
        status = normalisiere_status(review.get("Status", "Pruefen"))
        source_key = (quelle.get("source_key") or "").strip()
        item = {
            "sample": sample,
            "source_key": source_key,
            "genre": quelle.get("genre", ""),
            "status": status,
            "bewertung": review.get("Bewertung / Notiz", ""),
            "quelle": quelle.get("quelle", ""),
        }
        if source_key:
            policy[source_key] = item
        rows.append(item)
    return policy, rows


def source_id(row: dict[str, Any]) -> str:
    """Findet die Quell-ID in Manifest-Zeilen."""
    return str(row.get("source_id") or row.get("source_key") or row.get("source_video_id") or "").strip()


def erkenne_genre(row: dict[str, Any]) -> str:
    """Rueckwaertskompatibler Name fuer die gemeinsame Genre-Regel."""

    return erkenne_genre_gemeinsam(row)


def genre(row: dict[str, Any]) -> str:
    """Findet das LoRA-Genre in Manifest-Zeilen."""
    return erkenne_genre(row)


def clip_key(row: dict[str, Any]) -> str:
    """Erkennt doppelte Clips ueber Quelle und Zeitfenster."""
    sid = quellen_schluessel(row, PROJEKTWURZEL)
    start = row.get("start_time_sec", row.get("clip_start_sec", ""))
    end = row.get("end_time_sec", row.get("clip_end_sec", ""))
    return f"{sid}|{start}|{end}"


def lade_kandidaten(basis: Path, quelle: Path, *, nur_basis: bool) -> list[dict[str, Any]]:
    """Laedt Basisclips oder alle vorhandenen Importclips zum Nachfuellen."""
    roots = [basis] if nur_basis else manifest_roots(quelle)
    if not roots:
        roots = [basis]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for split in SPLITS:
            for row in lese_jsonl(root / split / "data.jsonl"):
                audio_path = Path(str(row.get("path") or "")).expanduser()
                if not audio_path.is_absolute():
                    audio_path = PROJEKTWURZEL / audio_path
                if not audio_path.exists() or audio_path.suffix.lower() != ".wav":
                    continue
                key = clip_key(row)
                path_key = str(row.get("path") or "")
                if key in seen or path_key in seen:
                    continue
                seen.add(key)
                seen.add(path_key)
                rows.append(row)
    return rows


def filtere_rows(
    rows: list[dict[str, Any]],
    policy: dict[str, dict[str, Any]],
    *,
    pruefen_ausschliessen: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filtert Manifest-Zeilen anhand der Review-Policy."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        sid = source_id(row)
        decision = policy.get(sid)
        status = str(decision.get("status") if decision else "Unbewertet")
        should_reject = status == "Schlecht" or (pruefen_ausschliessen and status == "Pruefen")
        if should_reject:
            rejected.append(
                {
                    "split": row.get("split", ""),
                    "genre": genre(row),
                    "source_key": sid,
                    "path": row.get("path", ""),
                    "grund": f"review_status_{status}",
                    "bewertung": decision.get("bewertung", "") if decision else "",
                }
            )
            continue
        clean = dict(row)
        clean["review_status"] = status
        if decision:
            clean["review_sample"] = decision.get("sample", "")
            clean["review_note"] = decision.get("bewertung", "")
            clean["quality_training_weight"] = 0.5 if status == "Pruefen" else 1.0
        else:
            clean["quality_training_weight"] = 1.0
        clean["quality_filtered_dataset"] = True
        accepted.append(clean)
    return accepted, rejected


def main() -> int:
    """Erstellt die neuen geprueften Manifest-Trainingsdaten."""
    args = parse_args()
    basis = Path(args.basis).expanduser().resolve()
    quelle = Path(args.quelle).expanduser().resolve()
    ziel = Path(args.ziel).expanduser().resolve()
    bewertung = Path(args.bewertung).expanduser().resolve()
    quellen = Path(args.quellen).expanduser().resolve()
    policy_dir = Path(args.policy_dir).expanduser().resolve()

    for path in (bewertung, quellen):
        if not path.exists():
            print(f"Fehlt: {rel(path)}")
            return 2
    if not all((basis / split / "data.jsonl").exists() for split in SPLITS):
        print(f"Basis-Dataset unvollstaendig: {rel(basis)}")
        return 2

    genre_keys = [item.strip() for item in str(args.genres).split(",") if item.strip()]
    if not genre_keys:
        print("Keine Genres angegeben.")
        return 2
    target_per_genre = int(math.ceil(int(args.ziel_clips) / len(genre_keys)))

    policy, policy_rows = lade_review_policy(bewertung, quellen)
    kandidaten = lade_kandidaten(basis, quelle, nur_basis=bool(args.nur_basis))
    accepted_all, rejected_all = filtere_rows(kandidaten, policy, pruefen_ausschliessen=bool(args.pruefen_ausschliessen))
    selected_by_genre_rows: dict[str, list[dict[str, Any]]] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    source_split_diagnostics: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for index, genre_key in enumerate(genre_keys):
        candidates = [row for row in accepted_all if genre(row) == genre_key]
        genre_splits, diagnostics = waehle_quellengetrennt(
            accepted_all,
            genre=genre_key,
            target_count=target_per_genre,
            max_per_source=int(args.max_pro_quelle_pro_genre),
            seed=int(args.seed) + index,
            projektwurzel=PROJEKTWURZEL,
        )
        source_split_diagnostics.append(diagnostics)
        selected = [
            row
            for split in SPLITS
            for row in genre_splits[split]
        ]
        selected_by_genre_rows[genre_key] = selected
        for split in SPLITS:
            for row in genre_splits[split]:
                split_rows[split].append(
                    bereinige_manifestzeile(
                        row,
                        genre=genre_key,
                        split=split,
                        projektwurzel=PROJEKTWURZEL,
                        dataset_marker="quality_balanced_dataset",
                    )
                )
        missing_rows.append(
            {
                "genre": genre_key,
                "target": target_per_genre,
                "available": len(candidates),
                "selected": len(selected),
                "missing": max(0, target_per_genre - len(selected)),
            }
        )
    selected_by_genre = Counter()
    review_status = Counter()
    for rows in split_rows.values():
        for row in rows:
            selected_by_genre[genre(row)] += 1
            review_status[str(row.get("review_status") or "Unbewertet")] += 1

    rejected_by_genre = Counter(item["genre"] for item in rejected_all)
    rejected_by_source = Counter(item["source_key"] for item in rejected_all)
    selected_total = sum(len(rows) for rows in split_rows.values())
    missing_total = sum(int(row["missing"]) for row in missing_rows)
    source_disjoint = all(item["source_disjoint"] for item in source_split_diagnostics)
    enough_sources = all(
        item["minimum_independent_sources_met"] for item in source_split_diagnostics
    )
    summary = {
        "created_at": jetzt_utc(),
        "basis_dataset": rel(basis),
        "source_root": rel(quelle),
        "target_dataset": rel(ziel),
        "method": "review_quality_genre_prefix_source_disjoint_v3",
        "review_csv": rel(bewertung),
        "source_review_csv": rel(quellen),
        "bad_sources_excluded": sorted(
            source for source, item in policy.items() if item.get("status") == "Schlecht"
        ),
        "check_sources_kept": sorted(
            source for source, item in policy.items() if item.get("status") == "Pruefen"
        ),
        "pruefen_ausschliessen": bool(args.pruefen_ausschliessen),
        "target_total": int(args.ziel_clips),
        "target_per_genre": target_per_genre,
        "genres": genre_keys,
        "candidate_total": len(kandidaten),
        "accepted_candidate_total": len(accepted_all),
        "selected_total": selected_total,
        "rejected_total": len(rejected_all),
        "missing_total": missing_total,
        "splits": {split: len(rows) for split, rows in split_rows.items()},
        "source_disjoint": source_disjoint,
        "minimum_independent_sources_per_genre": 5,
        "minimum_independent_sources_met": enough_sources,
        "source_split_diagnostics": source_split_diagnostics,
        "selected_by_genre": dict(sorted(selected_by_genre.items())),
        "missing_by_genre": {row["genre"]: row["missing"] for row in missing_rows},
        "available_by_genre_after_filter": {
            genre_key: sum(1 for row in accepted_all if genre(row) == genre_key) for genre_key in genre_keys
        },
        "rejected_by_genre": dict(sorted(rejected_by_genre.items())),
        "rejected_by_source": dict(sorted(rejected_by_source.items())),
        "review_status_in_dataset": dict(sorted(review_status.items())),
        "duration_sec": 30.0,
        "sample_rate": 32000,
        "channels": 1,
        "training_ready": (
            selected_total >= int(args.ziel_clips)
            and missing_total == 0
            and source_disjoint
            and enough_sources
        ),
        "hinweis": (
            "Dieses Dataset nutzt keine schlechten Review-Quellen, ordnet Genres "
            "ueber Import-Prefixe zu und trennt Ursprungs-MP3s zwischen den Splits."
        ),
    }

    policy_dir.mkdir(parents=True, exist_ok=True)
    schreibe_csv(
        policy_dir / "quellen_pruefung.csv",
        policy_rows,
        ["sample", "source_key", "genre", "status", "bewertung", "quelle"],
    )
    (policy_dir / "quellen_pruefung.json").write_text(
        json.dumps({"created_at": jetzt_utc(), "sources": policy_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.nur_plan:
        STANDARD_PLAN.mkdir(parents=True, exist_ok=True)
        (STANDARD_PLAN / "trainingsdaten_plan.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("Gepruefte Trainingsdaten: Plan")
        print(f"Basis:      {rel(basis)}")
        print(f"Quelle:     {rel(quelle)}")
        print(f"Ziel:       {rel(ziel)}")
        print(f"Akzeptiert: {summary['selected_total']}")
        print(f"Verworfen:  {summary['rejected_total']}")
        print(f"Genres:     {summary['selected_by_genre']}")
        print(f"Fehlen:     {summary['missing_by_genre']}")
        print(f"Quellen:    {'getrennt' if source_disjoint else 'FEHLER: Split-Ueberlappung'}")
        print(f"Report:     {rel(STANDARD_PLAN / 'trainingsdaten_plan.json')}")
        return 0 if summary["training_ready"] else 1

    for split, rows in split_rows.items():
        schreibe_jsonl(ziel / split / "data.jsonl", rows)
    (ziel / "reports").mkdir(parents=True, exist_ok=True)
    schreibe_csv(
        ziel / "reports" / "ausgeschlossene_clips.csv",
        rejected_all,
        ["split", "genre", "source_key", "path", "grund", "bewertung"],
    )
    schreibe_csv(
        ziel / "reports" / "quellen_pruefung.csv",
        policy_rows,
        ["sample", "source_key", "genre", "status", "bewertung", "quelle"],
    )
    schreibe_csv(
        ziel / "reports" / "fehlende_clips_pro_genre.csv",
        missing_rows,
        ["genre", "target", "available", "selected", "missing"],
    )
    (ziel / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Gepruefte Trainingsdaten erstellt")
    print(f"Ziel:       {rel(ziel)}")
    print(f"Akzeptiert: {summary['selected_total']}")
    print(f"Verworfen:  {summary['rejected_total']}")
    print(f"Genres:     {summary['selected_by_genre']}")
    if missing_total:
        print(f"Fehlen:     {summary['missing_by_genre']}")
    return 0 if summary["training_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
