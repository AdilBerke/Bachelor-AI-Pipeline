#!/usr/bin/env python3
"""Erstellt ein genrebalanciertes LoRA-Clip-Zieldataset fuer MusicGen-LoRA.

Die Datei baut keine neuen WAVs und startet kein Training. Sie sammelt nur
bereits vorhandene, saubere 30s-Clips aus direkten MP3/Youtube-Imports und
schreibt daraus ein Manifest-Dataset. Alte 60s-Zwischenpipeline-Clips werden
hart blockiert.
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
    erkenne_genre,
    quelle_passt_zum_genre,
    quellen_schluessel,
    waehle_quellengetrennt,
)


PROJEKTWURZEL = Path(__file__).resolve().parents[3]
STANDARD_QUELLE = PROJEKTWURZEL / "daten" / "processed" / "musicgen_youtube_import_30s"
STANDARD_ZIEL = PROJEKTWURZEL / "daten" / "processed" / "lora_training"
STANDARD_PLAN = PROJEKTWURZEL / "training" / "musicgen" / "dataset_pruefung"
QUELLEN_SPERRLISTE_CSV = PROJEKTWURZEL / "daten" / "metadata" / "trainingsdaten_pruefung" / "quellen_sperrliste.csv"
SPLITS = ("train", "valid", "test")


def jetzt_utc() -> str:
    """UTC-Zeitstempel fuer Reports."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    """Schreibt Projektpfade kurz und lesbar."""
    try:
        return str(path.resolve().relative_to(PROJEKTWURZEL))
    except Exception:
        return str(path)


def parse_args() -> argparse.Namespace:
    """Argumente fuer das LoRA-Zieldataset."""
    parser = argparse.ArgumentParser(description="Plant/erstellt ein LoRA-Clip-Dataset mit gleichmaessigen Lofi-Genres.")
    parser.add_argument("--quelle", default=str(STANDARD_QUELLE))
    parser.add_argument("--ziel", default=str(STANDARD_ZIEL))
    parser.add_argument("--ziel-clips", type=int, default=5000)
    parser.add_argument("--genres", default=",".join(STANDARD_GENRES))
    parser.add_argument("--seed", type=int, default=4027)
    parser.add_argument("--max-pro-quelle-pro-genre", type=int, default=250)
    parser.add_argument("--min-quality-score", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--nur-plan", action="store_true", help="Nur Reports schreiben, keine Manifeste erzeugen.")
    return parser.parse_args()


def lese_jsonl(path: Path) -> list[dict[str, Any]]:
    """Liest JSONL defensiv."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} ist kein JSON-Objekt.")
            item["_manifest_path"] = str(path)
            item["_line_number"] = line_number
            rows.append(item)
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
    """Findet ein einzelnes Dataset oder mehrere Import-Unterdatasets."""
    root = root.expanduser().resolve()
    if all((root / split / "data.jsonl").exists() for split in SPLITS):
        return [root]
    found = sorted({path.parents[1] for path in root.glob("**/train/data.jsonl")})
    return [item for item in found if all((item / split / "data.jsonl").exists() for split in SPLITS)]


def genre_key(row: dict[str, Any]) -> str:
    """Kompatibler Kurzname fuer die gemeinsame Genre-Regel."""

    return erkenne_genre(row)


def source_key(row: dict[str, Any]) -> str:
    """Kompatibler Kurzname fuer den gemeinsamen Quellschluessel."""

    return quellen_schluessel(row, PROJEKTWURZEL)


def gesperrte_quellen() -> set[str]:
    """Liest Quellen, die fuer den LoRA-Zieldatensatz gesperrt sind."""

    ids: set[str] = set()
    if not QUELLEN_SPERRLISTE_CSV.exists():
        return ids
    with QUELLEN_SPERRLISTE_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = str(row.get("status") or "").strip().lower()
            if status not in {"schlecht", "gesperrt", "falsches_genre", "kein_lofi_bezug"}:
                continue
            for key in ("source_key", "video_id", "url"):
                value = str(row.get(key) or "").strip().lower()
                if value:
                    ids.add(value)
    return ids


def quelle_gesperrt(row: dict[str, Any], gesperrt: set[str]) -> bool:
    """Prueft Quelle, Video-ID und URL gegen die Sperrliste."""

    if not gesperrt:
        return False
    kandidaten = {
        str(row.get("source_id") or "").strip().lower(),
        str(row.get("video_id") or "").strip().lower(),
        str(row.get("source_url") or "").strip().lower(),
        str(row.get("webpage_url") or "").strip().lower(),
        source_key(row).lower(),
    }
    return bool(kandidaten.intersection(gesperrt))


def sekundenwert(value: Any) -> str:
    """Normalisiert Zeitangaben, damit 30 und 30.0 denselben Clip bezeichnen."""
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def clip_key(row: dict[str, Any]) -> str:
    """Eindeutiger Clip-Schluessel aus Quelle und Zeitfenster."""
    source = source_key(row)
    start = row.get("start_time_sec", row.get("clip_start_sec", ""))
    end = row.get("end_time_sec", row.get("clip_end_sec", ""))
    # Der Dateiname gehoert bewusst nicht zum Schluessel. Beim Nachclippen aus
    # derselben MP3 koennen identische Zeitfenster in einem anderen Ordner
    # landen; diese sollen fuer LoRA nicht doppelt gezaehlt werden.
    return f"{source}|{sekundenwert(start)}|{sekundenwert(end)}"


def ist_60s_abgeleitet(row: dict[str, Any]) -> bool:
    """Blockiert die alte 60s-Zwischenpipeline."""
    path = str(row.get("path") or "")
    return bool(
        row.get("source_60s_path")
        or row.get("created_from_dataset") == "musicgen_per_source_60s"
        or row.get("conversion_rule") == "60s_clip_split_into_two_30s_parts"
        or "musicgen_30s_aus_60s_clips" in path
        or "musicgen_per_source_60s" in path
    )


def repariere_audio_pfad(row: dict[str, Any], dataset_root: Path) -> tuple[dict[str, Any] | None, str]:
    """Repariert alte Manifest-Pfade, wenn die WAV im aktuellen Importordner liegt."""
    raw_path = str(row.get("path") or "").strip()
    if not raw_path:
        return None, "pfad_fehlt"
    path = Path(raw_path).expanduser()
    if path.exists():
        return row, ""
    matches = list(dataset_root.glob(f"**/{path.name}"))
    if len(matches) == 1:
        fixed = dict(row)
        fixed["path"] = str(matches[0].resolve())
        fixed["lora_path_repaired"] = True
        fixed["lora_original_path"] = raw_path
        return fixed, ""
    return None, "audiodatei_fehlt"


def lade_rows(source_roots: list[Path], min_quality_score: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Laedt und filtert alle Kandidaten aus den Import-Datasets."""
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_clips: set[str] = set()
    gesperrt = gesperrte_quellen()

    for root in source_roots:
        for split in SPLITS:
            for row in lese_jsonl(root / split / "data.jsonl"):
                candidate = dict(row)
                candidate["lora_source_dataset"] = str(root)
                candidate["lora_source_split"] = split
                reason = ""
                if ist_60s_abgeleitet(candidate):
                    reason = "alte_60s_herkunft"
                else:
                    repaired, reason = repariere_audio_pfad(candidate, root)
                    candidate = repaired or candidate
                if not reason and quelle_gesperrt(candidate, gesperrt):
                    reason = "quelle_gesperrt"
                if not reason:
                    try:
                        quality = float(candidate.get("quality_score") or 1.0)
                    except (TypeError, ValueError):
                        quality = 1.0
                    if quality < min_quality_score:
                        reason = "quality_score_zu_niedrig"
                if not reason:
                    ok, source_reason = quelle_passt_zum_genre(candidate)
                    if not ok:
                        reason = source_reason
                if not reason:
                    path_key = str(Path(str(candidate.get("path") or "")).expanduser().resolve())
                    key = clip_key(candidate)
                    if path_key in seen_paths or key in seen_clips:
                        reason = "duplikat"
                    else:
                        seen_paths.add(path_key)
                        seen_clips.add(key)
                if reason:
                    rejected.append(
                        {
                            "split": split,
                            "path": candidate.get("path", ""),
                            "source": source_key(candidate),
                            "genre": genre_key(candidate),
                            "reason": reason,
                        }
                    )
                    continue
                rows.append(candidate)
    return rows, rejected


def main() -> int:
    """CLI-Einstieg."""
    args = parse_args()
    source_root = Path(args.quelle).expanduser().resolve()
    target_root = Path(args.ziel).expanduser().resolve()
    genres = [item.strip() for item in str(args.genres).split(",") if item.strip()]
    if not genres:
        raise ValueError("Mindestens ein Genre muss gesetzt sein.")
    target_per_genre = int(math.ceil(args.ziel_clips / len(genres)))

    source_roots = manifest_roots(source_root)
    if not source_roots:
        raise FileNotFoundError(f"Keine Import-Datasets gefunden: {source_root}")
    if target_root.exists() and any(target_root.iterdir()) and not args.overwrite and not args.nur_plan:
        raise FileExistsError(f"Ziel existiert bereits. Nutze --overwrite: {target_root}")

    rows, rejected = lade_rows(source_roots, args.min_quality_score)
    available_by_genre = Counter(genre_key(row) for row in rows)
    selected_by_genre: dict[str, list[dict[str, Any]]] = {}
    split_map: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    source_split_diagnostics: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    source_report_rows: list[dict[str, Any]] = []
    for index, genre in enumerate(genres):
        genre_splits, diagnostics = waehle_quellengetrennt(
            rows,
            genre=genre,
            target_count=target_per_genre,
            max_per_source=args.max_pro_quelle_pro_genre,
            seed=args.seed + index,
            projektwurzel=PROJEKTWURZEL,
        )
        source_split_diagnostics.append(diagnostics)
        selected = [
            row
            for split in SPLITS
            for row in genre_splits[split]
        ]
        selected_by_genre[genre] = selected
        for split in SPLITS:
            for row in genre_splits[split]:
                split_map[split].append(
                    bereinige_manifestzeile(
                        row,
                        genre=genre,
                        split=split,
                        projektwurzel=PROJEKTWURZEL,
                        dataset_marker="lora_balanced_dataset",
                    )
                )
        missing = max(0, target_per_genre - len(selected))
        missing_rows.append(
            {
                "genre": genre,
                "target": target_per_genre,
                "available": available_by_genre.get(genre, 0),
                "selected": len(selected),
                "missing": missing,
            }
        )
        counts = Counter(source_key(row) for row in selected)
        for source, count in counts.most_common():
            source_split = next(
                (
                    split
                    for split in SPLITS
                    if source in diagnostics["source_keys_by_split"][split]
                ),
                "",
            )
            source_report_rows.append(
                {"genre": genre, "split": source_split, "source": source, "count": count}
            )

    selected_total = sum(len(rows_for_genre) for rows_for_genre in selected_by_genre.values())
    source_disjoint = all(item["source_disjoint"] for item in source_split_diagnostics)
    enough_sources = all(
        item["minimum_independent_sources_met"] for item in source_split_diagnostics
    )
    training_ready = (
        selected_total >= args.ziel_clips
        and all(row["missing"] == 0 for row in missing_rows)
        and source_disjoint
        and enough_sources
    )
    report_root = target_root if not args.nur_plan else STANDARD_PLAN
    reports = report_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    if not args.nur_plan:
        if target_root.exists() and args.overwrite:
            import shutil

            shutil.rmtree(target_root)
            report_root = target_root
            reports = report_root / "reports"
            reports.mkdir(parents=True, exist_ok=True)
        for split, split_items in split_map.items():
            schreibe_jsonl(target_root / split / "data.jsonl", split_items)

    genre_report_rows = []
    for split, split_items in split_map.items():
        counts = Counter(row.get("primary_genre") for row in split_items)
        for genre, count in sorted(counts.items()):
            genre_report_rows.append({"split": split, "genre": genre, "count": count})

    summary = {
        "created_at": jetzt_utc(),
        "source_root": rel(source_root),
        "source_datasets_used": [rel(path) for path in source_roots],
        "target_dataset": rel(target_root),
        "method": "mp3_import_strict_lofi_source_disjoint_v4",
        "target_total": args.ziel_clips,
        "target_per_genre": target_per_genre,
        "genres": genres,
        "available_total": len(rows),
        "selected_total": selected_total,
        "missing_total": max(0, args.ziel_clips - selected_total),
        "training_ready": training_ready,
        "source_disjoint": source_disjoint,
        "minimum_independent_sources_per_genre": 5,
        "minimum_independent_sources_met": enough_sources,
        "source_split_diagnostics": source_split_diagnostics,
        "splits": {split: len(split_map[split]) for split in SPLITS},
        "available_by_genre": dict(sorted(available_by_genre.items())),
        "selected_by_genre": {genre: len(selected_by_genre[genre]) for genre in genres},
        "missing_by_genre": {row["genre"]: row["missing"] for row in missing_rows},
        "rejected_count": len(rejected),
        "old_60s_rows_rejected": sum(1 for row in rejected if row["reason"] == "alte_60s_herkunft"),
        "non_lofi_rows_rejected": sum(
            1
            for row in rejected
            if str(row["reason"]).startswith(("kein_lofi_bezug", "falscher_stil", "genrebezug_fehlt"))
        ),
        "wav_files_copied": False,
        "nur_plan": bool(args.nur_plan),
    }

    summary_name = "zieldatensatz_plan.json" if args.nur_plan else "dataset_summary.json"
    (report_root / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    schreibe_csv(reports / "fehlende_clips_pro_genre.csv", missing_rows, ["genre", "target", "available", "selected", "missing"])
    schreibe_csv(reports / "genre_balance.csv", genre_report_rows, ["split", "genre", "count"])
    schreibe_csv(
        reports / "source_balance.csv",
        source_report_rows,
        ["genre", "split", "source", "count"],
    )
    schreibe_csv(reports / "abgelehnte_clips.csv", rejected, ["split", "path", "source", "genre", "reason"])

    print("LoRA-Zieldataset")
    print("==================")
    print(f"Ziel:       {rel(target_root)}")
    print(f"Verfuegbar: {len(rows)} Clips")
    print(f"Ausgewaehlt:{selected_total} / {args.ziel_clips} Clips")
    print(f"Bereit:     {'ja' if training_ready else 'nein'}")
    for row in missing_rows:
        print(
            f"- {row['genre']}: {row['selected']}/{row['target']} "
            f"(fehlen: {row['missing']})"
        )
    print(f"Quellen:    {'getrennt' if source_disjoint else 'FEHLER: Split-Ueberlappung'}")
    print(f"Report:     {rel(report_root / summary_name)}")
    return 0 if training_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
