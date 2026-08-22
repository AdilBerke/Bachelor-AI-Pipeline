#!/usr/bin/env python3
"""Bereitet das LoRA-Clip-Dataset aus Top-5-Quellen vor.

Diese Pipeline verbindet die bereits vorhandenen Bausteine:

1. Top-5-CSV-Dateien je Genre lesen.
2. MP3s ueber den Crawler-Import herunterladen.
3. Aus MP3s saubere 30s-WAV-Clips schneiden.
4. Das genrebalancierte LoRA-Zieldataset neu bauen.
5. Das Ergebnis validieren.

Wichtig: Ein normaler Start fuehrt die Vorbereitung wirklich aus. Fuer eine
reine Vorschau gibt es ``--plan``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJEKTWURZEL = Path(__file__).resolve().parents[3]
PYTHON = PROJEKTWURZEL / ".venv" / "bin" / "python"
TOP10_ORDNER = PROJEKTWURZEL / "daten" / "metadata" / "crawler"
RAW_MP3_ORDNER = PROJEKTWURZEL / "daten" / "raw" / "audio" / "youtube_imports"
IMPORT_DATASET = PROJEKTWURZEL / "daten" / "processed" / "musicgen_youtube_import_30s"
ZIEL_DATASET = PROJEKTWURZEL / "daten" / "processed" / "lora_training"
REPORT_ORDNER = PROJEKTWURZEL / "daten" / "metadata" / "crawler"
DOWNLOAD_FEHLER_MARKER = REPORT_ORDNER / "download_fehler_quellen.jsonl"
QUALITAET_POLICY_JSON = PROJEKTWURZEL / "daten" / "metadata" / "trainingsdaten_pruefung" / "quellen_pruefung.json"
QUALITAET_POLICY_CSV = PROJEKTWURZEL / "daten" / "metadata" / "trainingsdaten_pruefung" / "quellen_pruefung.csv"
QUELLEN_SPERRLISTE_CSV = PROJEKTWURZEL / "daten" / "metadata" / "trainingsdaten_pruefung" / "quellen_sperrliste.csv"
ZIEL_CLIPS = 5000

DATASET_SRC = PROJEKTWURZEL / "code" / "src" / "Dataset"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))
from genre_regeln import (
    GENRE_CAPTIONS,
    GENRE_KEYWORDS,
    GENRE_LABELS,
    GENRE_PREFIXE,
    QUELLEN_REGEL_VERSION,
    STANDARD_GENRES,
    normalisiere,
    quelle_passt_zum_genre,
)


GENRES: dict[str, dict[str, str]] = {
    genre_key: {
        "label": GENRE_LABELS.get(genre_key, genre_key.replace("_", " ").title()),
        "caption": GENRE_CAPTIONS.get(genre_key, "calm lofi instrumental, no vocals"),
    }
    for genre_key in STANDARD_GENRES
}


@dataclass
class Quelle:
    """Eine ausgewaehlte Quelle aus einer Top-5-CSV."""

    genre_key: str
    genre_label: str
    rang: str
    video_id: str
    url: str
    titel: str
    csv_path: Path
    dataset_name: str = ""


def parse_args() -> argparse.Namespace:
    """Liest die Kommandozeilenargumente."""

    parser = argparse.ArgumentParser(
        description="Laedt Top-5-Quellen je Genre und erstellt daraus 30s-Clips fuer das LoRA-Dataset."
    )
    parser.add_argument(
        "--ausfuehren",
        action="store_true",
        help="Kompatibel mit aelteren Befehlen. Ein normaler Start fuehrt bereits aus.",
    )
    parser.add_argument("--plan", action="store_true", help="Nur anzeigen, nichts herunterladen und keine Clips erzeugen.")
    parser.add_argument("--genres", default=",".join(GENRES), help="Kommagetrennte Genre-Keys.")
    parser.add_argument("--ziel-clips", type=int, default=ZIEL_CLIPS, help="Gesamtzahl der Zielclips fuer das LoRA-Dataset.")
    parser.add_argument("--max-links-pro-genre", type=int, default=3, help="Wie viele Top-5-Links pro Genre verarbeitet werden.")
    parser.add_argument("--min-quellen-score", type=float, default=50.0, help="Mindest-Gesamtscore fuer automatisch geladene Top-5-Quellen.")
    parser.add_argument("--min-genre-score", type=float, default=60.0, help="Mindest-Genretreue fuer automatisch geladene Top-5-Quellen.")
    parser.add_argument("--min-audio-score", type=float, default=55.0, help="Mindest-Audioeignung fuer automatisch geladene Top-5-Quellen.")
    parser.add_argument("--min-quellen-dauer-minuten", type=float, default=20.0, help="Mindestlaenge einer automatisch geladenen Trainingsquelle.")
    parser.add_argument("--max-clips-pro-quelle", type=int, default=300, help="Maximale 30s-Clips pro MP3-Quelle.")
    parser.add_argument("--clip-duration-sec", type=float, default=30.0)
    parser.add_argument("--clip-hop-sec", type=float, default=30.0)
    parser.add_argument("--skip-intro-sec", type=float, default=20.0)
    parser.add_argument("--skip-outro-sec", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=4027)
    parser.add_argument("--erneut-laden", action="store_true", help="MP3s erneut laden, auch wenn sie schon vorhanden sind.")
    parser.add_argument("--nur-download", action="store_true", help="Nur MP3s importieren, keine Clips bauen.")
    parser.add_argument("--nur-clippen", action="store_true", help="Downloads ueberspringen und vorhandene MP3s clippen.")
    parser.add_argument("--lokale-mp3s", action="store_true", help="Nur vorhandene lokale MP3s clippen, keine Downloads.")
    parser.add_argument(
        "--mit-downloads",
        action="store_true",
        help="Neue Top-5-Quellen laden. Ohne diese Option werden nur lokale MP3s genutzt.",
    )
    parser.add_argument(
        "--bewertung-filter",
        action="store_true",
        help="Kompatibel mit alten Befehlen. Sperrlisten werden inzwischen immer beruecksichtigt.",
    )
    parser.add_argument(
        "--max-pro-quelle-pro-genre",
        type=int,
        default=800,
        help="Maximale Clips pro MP3-Quelle im LoRA-Zieldataset.",
    )
    parser.add_argument("--cookies", default="", help="Optionaler yt-dlp Cookies-Dateipfad.")
    parser.add_argument("--cookies-browser", default="", help="Optional, z.B. firefox oder chrome.")
    parser.add_argument("--run-name", default="", help="Optionaler Name fuer Reports.")
    parser.add_argument(
        "--max-download-fehler-in-folge",
        type=int,
        default=3,
        help="Stoppt sauber, wenn zu viele Downloads nacheinander blockiert/fehlgeschlagen sind.",
    )
    args = parser.parse_args()
    if args.plan and args.ausfuehren:
        parser.error("Bitte entweder --plan oder --ausfuehren nutzen, nicht beides.")
    args.ausfuehren = not args.plan
    if not args.mit_downloads:
        args.lokale_mp3s = True
    return args


def jetzt_name() -> str:
    """Erzeugt einen kurzen Zeitstempel fuer Reports."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def rel(path: Path) -> str:
    """Gibt Pfade relativ zur Projektwurzel aus, falls moeglich."""

    try:
        return str(path.resolve().relative_to(PROJEKTWURZEL))
    except ValueError:
        return str(path)


def ffprobe_duration(path: Path) -> float:
    """Liest die Audiodauer mit ffprobe, ohne die ganze MP3 in Python zu laden."""

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return float(payload.get("format", {}).get("duration") or 0.0)


def schreibe_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Schreibt Manifestzeilen direkt, damit kein zweiter Prozess noetig ist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_fuer_index(index: int, seed: int) -> str:
    """Verteilt Clips reproduzierbar auf Train, Valid und Test."""

    import random

    value = random.Random(seed + index * 7919).random()
    if value < 0.90:
        return "train"
    if value < 0.95:
        return "valid"
    return "test"


def schreibe_wav_segment(source: Path, target: Path, start_sec: float, duration_sec: float) -> None:
    """Schneidet ein einzelnes WAV-Segment ueber ffmpeg."""

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{duration_sec:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ar",
            "32000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def wav_dauer_ok(path: Path, expected_sec: float, tolerance: float = 0.10) -> bool:
    """Prueft kurz den WAV-Header."""

    import wave

    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = int(handle.getframerate())
            channels = int(handle.getnchannels())
            duration = handle.getnframes() / float(sample_rate or 1)
    except Exception:
        return False
    return sample_rate == 32000 and channels == 1 and abs(duration - expected_sec) <= tolerance


def slug(text: str, fallback: str = "quelle") -> str:
    """Macht aus freiem Text einen kurzen Dateinamenbestandteil."""

    value = str(text or "").lower().strip()
    value = value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return (value or fallback)[:80]


def video_id(value: str) -> str:
    """Extrahiert eine YouTube-ID aus URL oder Roh-ID."""

    value = str(value or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    for pattern in (
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"/shorts/([A-Za-z0-9_-]{11})",
        r"/live/([A-Za-z0-9_-]{11})",
        r"/embed/([A-Za-z0-9_-]{11})",
    ):
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return ""


def neueste_top10_csv(genre_key: str) -> Path | None:
    """Findet die neueste Top-10-CSV fuer ein Genre."""

    files = []
    for pattern in (
        f"top10_top10_quellen_5000_*_{genre_key}.csv",
        f"top10_top10_quellen_lora_*_{genre_key}.csv",
        f"top10_quellen_5000_*_{genre_key}.csv",
        f"top10_quellen_lora_*_{genre_key}.csv",
    ):
        files.extend(TOP10_ORDNER.glob(pattern))
    files = sorted(set(files), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in files:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    url = str(row.get("url") or "").strip()
                    regel_version = str(row.get("quellen_regel_version") or "").strip()
                    if regel_version != QUELLEN_REGEL_VERSION:
                        continue
                    ausschluss = str(row.get("ausschlussgrund") or "").strip()
                    status = str(row.get("status") or "").strip().lower()
                    if url and not ausschluss and status not in {"skip", "blocked", "fehler", "error"}:
                        return path
        except OSError:
            continue
    return None


def score_wert(row: dict[str, str], key: str) -> float | None:
    """Liest einen numerischen Score aus einer Top-10-CSV."""

    raw = str(row.get(key) or "").strip().replace(",", ".")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def dataset_name_fuer_quelle(quelle: Quelle) -> str:
    """Erzeugt den tatsaechlichen Import-Datasetnamen fuer eine Quelle."""

    if quelle.dataset_name:
        return quelle.dataset_name
    return slug(f"{quelle.genre_key}_{quelle.video_id or slug(quelle.titel)}")


def genre_aus_mp3_name(path: Path) -> str | None:
    """Ordnet lokale MP3-Dateien einem der Trainingsgenres zu."""

    text = normalisiere(path.name).replace("lo fi", "lofi")
    if not any(token in text for token in ("lofi", "lo fi", "lowfi", "chillhop")):
        return None
    prefix = text.split("-", 1)[0].strip()
    for genre_key, prefixes in GENRE_PREFIXE.items():
        normalized_prefixes = [normalisiere(item).replace("lo fi", "lofi") for item in prefixes]
        if prefix in normalized_prefixes or any(text.startswith(item + " ") for item in normalized_prefixes):
            return genre_key
    for genre_key, keywords in GENRE_KEYWORDS.items():
        normalized_keywords = [normalisiere(item).replace("lo fi", "lofi") for item in keywords]
        if any(token in text for token in normalized_keywords):
            return genre_key
    return None


def video_id_aus_mp3_name(path: Path) -> str:
    """Extrahiert die Video-ID aus Dateinamen wie Titel [VIDEOID].mp3."""

    match = re.search(r"\[([A-Za-z0-9_-]{11})\]\.mp3$", path.name)
    if match:
        return match.group(1)
    return slug(path.stem)[:20]


def quellenzeile(quelle: Quelle) -> dict[str, Any]:
    """Formt eine Quelle so, dass die gemeinsamen LoFi-Regeln greifen."""

    return {
        "source_title": quelle.titel,
        "source_file": quelle.csv_path.name,
        "source_url": quelle.url,
        "titel": quelle.titel,
        "url": quelle.url,
        "genre": quelle.genre_key,
        "caption": GENRES[quelle.genre_key]["caption"],
    }


def lokale_mp3_quellen(args: argparse.Namespace, summary: dict[str, Any] | None = None) -> tuple[list[Quelle], dict[str, Any]]:
    """Erzeugt Quellen aus lokal vorhandenen MP3s, ohne YouTube zu kontaktieren."""

    summary = summary or {}
    erlaubte_genres = {item.strip() for item in args.genres.split(",") if item.strip()}
    schlechte_ids = schlechte_quellen_ids()
    quellen: list[Quelle] = []
    skip_info: dict[str, Any] = {
        "volle_genres": [],
        "fertige_quellen": 0,
        "blockierte_quellen": 0,
        "schlechte_quellen": 0,
        "zu_schwache_quellen": 0,
        "fehlende_top10_csv": [],
    }
    for mp3_path in sorted(RAW_MP3_ORDNER.glob("*.mp3")):
        genre_key = genre_aus_mp3_name(mp3_path)
        if genre_key not in GENRES or genre_key not in erlaubte_genres:
            continue
        if genre_voll(summary, genre_key):
            skip_info["volle_genres"].append(f"{GENRES[genre_key]['label']} {genre_stand(summary, genre_key)}")
            continue
        vid = video_id_aus_mp3_name(mp3_path)
        hop_label = str(args.clip_hop_sec).replace(".", "_")
        dataset_name = slug(f"lokal_hop{hop_label}_{genre_key}_{vid}")
        quelle = Quelle(
            genre_key=genre_key,
            genre_label=GENRES[genre_key]["label"],
            rang="lokal",
            video_id=vid,
            url=f"local:{mp3_path.name}",
            titel=mp3_path.stem,
            csv_path=mp3_path,
            dataset_name=dataset_name,
        )
        if ist_schlechte_quelle(quelle, schlechte_ids):
            skip_info["schlechte_quellen"] += 1
            continue
        if quelle_fertig(quelle):
            skip_info["fertige_quellen"] += 1
            continue
        quellen.append(quelle)
    return quellen, skip_info


def manifest_zeilen(dataset_dir: Path) -> int:
    """Zaehlt Manifestzeilen eines Import-Datasets."""

    total = 0
    for split in ("train", "valid", "test"):
        path = dataset_dir / split / "data.jsonl"
        if path.exists():
            total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def quelle_fertig(quelle: Quelle) -> bool:
    """Prueft, ob diese Quelle bereits erfolgreich zu Clips verarbeitet wurde."""

    dataset_dir = IMPORT_DATASET / dataset_name_fuer_quelle(quelle)
    summary_path = dataset_dir / "dataset_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    accepted = int(summary.get("accepted_count") or manifest_zeilen(dataset_dir))
    return bool(summary.get("training_ready")) and accepted > 0


def bekannte_fehlquellen() -> set[str]:
    """Sammelt Video-IDs, die in frueheren Laeufen schon blockiert waren."""

    ids: set[str] = set()
    if DOWNLOAD_FEHLER_MARKER.exists():
        for line in DOWNLOAD_FEHLER_MARKER.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = str(row.get("video_id") or "").strip()
            if vid:
                ids.add(vid)
    for path in REPORT_ORDNER.glob("clips_training_vorbereiten_*.csv"):
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("status") or "") != "download_fehler":
                        continue
                    vid = str(row.get("video_id") or "").strip()
                    if vid:
                        ids.add(vid)
        except OSError:
            continue
    downloads_dir = PROJEKTWURZEL / "daten" / "metadata" / "downloads"
    for path in downloads_dir.glob("mp3_download_report_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("ergebnisse", []):
            if str(row.get("status") or "") != "fehler":
                continue
            vid = video_id(str(row.get("url") or ""))
            if vid:
                ids.add(vid)
    return ids


def schlechte_quellen_ids() -> set[str]:
    """Liest Quellen, die durch menschliche Bewertung ausgeschlossen wurden."""

    ids: set[str] = set()
    if QUALITAET_POLICY_JSON.exists():
        try:
            payload = json.loads(QUALITAET_POLICY_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            for row in payload.get("sources", []):
                if not isinstance(row, dict):
                    continue
                if str(row.get("status") or "").strip().lower() != "schlecht":
                    continue
                source = str(row.get("source_key") or "").strip()
                if source:
                    ids.add(source.lower())
    if QUALITAET_POLICY_CSV.exists():
        try:
            with QUALITAET_POLICY_CSV.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("status") or "").strip().lower() != "schlecht":
                        continue
                    source = str(row.get("source_key") or "").strip()
                    if source:
                        ids.add(source.lower())
        except OSError:
            pass
    if QUELLEN_SPERRLISTE_CSV.exists():
        try:
            with QUELLEN_SPERRLISTE_CSV.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    status = str(row.get("status") or "").strip().lower()
                    if status not in {"schlecht", "gesperrt", "falsches_genre", "kein_lofi_bezug"}:
                        continue
                    for key in ("source_key", "video_id", "url"):
                        source = str(row.get(key) or "").strip()
                        if source:
                            ids.add(source.lower())
                            vid = video_id(source)
                            if vid:
                                ids.add(vid.lower())
        except OSError:
            pass
    return ids


def ist_schlechte_quelle(quelle: Quelle, schlechte_ids: set[str]) -> bool:
    """Prueft, ob eine Quelle im menschlichen Review schlecht bewertet wurde."""

    kandidaten = {
        str(quelle.video_id or "").strip().lower(),
        video_id(str(quelle.url or "")).lower(),
    }
    return bool(schlechte_ids.intersection(kandidaten))


def download_fehler_grund(log_path: Path) -> str:
    """Erkennt haeufige Downloadfehler aus dem Log."""

    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "download_fehler"
    lower = text.lower()
    if "sign in to confirm" in lower or "not a bot" in lower:
        return "youtube_bot_blockiert"
    if "video unavailable" in lower or "private video" in lower:
        return "video_nicht_verfuegbar"
    if "copyright" in lower:
        return "rechte_oder_region_blockiert"
    return "download_fehler"


def markiere_fehlquelle(quelle: Quelle, grund: str, log_path: Path) -> None:
    """Merkt sich fehlgeschlagene Quellen, damit sie spaeter uebersprungen werden."""

    DOWNLOAD_FEHLER_MARKER.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "genre": quelle.genre_key,
        "video_id": quelle.video_id,
        "url": quelle.url,
        "titel": quelle.titel,
        "grund": grund,
        "log": rel(log_path),
    }
    with DOWNLOAD_FEHLER_MARKER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def lese_quellen(args: argparse.Namespace, summary: dict[str, Any] | None = None) -> tuple[list[Quelle], dict[str, Any]]:
    """Liest die gewuenschten Top-10-Quellen aus den Crawler-Reports."""

    summary = summary or {}
    fehlquellen = bekannte_fehlquellen()
    schlechte_ids = schlechte_quellen_ids()
    genre_keys = [item.strip() for item in args.genres.split(",") if item.strip()]
    quellen: list[Quelle] = []
    skip_info: dict[str, Any] = {
        "volle_genres": [],
        "fertige_quellen": 0,
        "blockierte_quellen": 0,
        "schlechte_quellen": 0,
        "zu_schwache_quellen": 0,
        "fehlende_top10_csv": [],
    }
    for genre_key in genre_keys:
        if genre_key not in GENRES:
            raise ValueError(f"Unbekanntes Genre: {genre_key}. Erlaubt: {', '.join(GENRES)}")
        if genre_voll(summary, genre_key):
            skip_info["volle_genres"].append(f"{GENRES[genre_key]['label']} {genre_stand(summary, genre_key)}")
            continue
        csv_path = neueste_top10_csv(genre_key)
        if not csv_path:
            skip_info["fehlende_top10_csv"].append(genre_key)
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            count = 0
            for row in reader:
                url = str(row.get("url") or "").strip()
                if not url:
                    continue
                regel_version = str(row.get("quellen_regel_version") or "").strip()
                if regel_version != QUELLEN_REGEL_VERSION:
                    continue
                ausschluss = str(row.get("ausschlussgrund") or "").strip()
                status = str(row.get("status") or "").strip().lower()
                if ausschluss or status in {"skip", "blocked", "fehler", "error"}:
                    continue
                gesamt_score = score_wert(row, "gesamt_score")
                genre_score = score_wert(row, "genre_score")
                audio_score = score_wert(row, "audio_eignung_score")
                dauer_minuten = score_wert(row, "dauer_minuten")
                if dauer_minuten is not None and dauer_minuten < args.min_quellen_dauer_minuten:
                    skip_info["zu_schwache_quellen"] += 1
                    continue
                if gesamt_score is not None and gesamt_score < args.min_quellen_score:
                    skip_info["zu_schwache_quellen"] += 1
                    continue
                if genre_score is not None and genre_score < args.min_genre_score:
                    skip_info["zu_schwache_quellen"] += 1
                    continue
                if audio_score is not None and audio_score < args.min_audio_score:
                    skip_info["zu_schwache_quellen"] += 1
                    continue
                vid = str(row.get("video_id") or video_id(url)).strip()
                quelle = Quelle(
                    genre_key=genre_key,
                    genre_label=GENRES[genre_key]["label"],
                    rang=str(row.get("rang") or count + 1),
                    video_id=vid,
                    url=url,
                    titel=str(row.get("titel") or row.get("title") or vid or url),
                    csv_path=csv_path,
                )
                if quelle.video_id and quelle.video_id in fehlquellen:
                    skip_info["blockierte_quellen"] += 1
                    continue
                if ist_schlechte_quelle(quelle, schlechte_ids):
                    skip_info["schlechte_quellen"] += 1
                    continue
                if quelle_fertig(quelle):
                    skip_info["fertige_quellen"] += 1
                    continue
                quellen.append(quelle)
                count += 1
                if count >= args.max_links_pro_genre:
                    break
    return quellen, skip_info


def finde_mp3(quelle: Quelle) -> Path | None:
    """Findet die lokale MP3 zu einer Quelle, falls sie schon geladen wurde."""

    if not RAW_MP3_ORDNER.exists():
        return None
    if quelle.url.startswith("local:") and quelle.csv_path.exists():
        return quelle.csv_path
    kandidaten: list[Path] = []
    if quelle.video_id:
        kandidaten.extend(RAW_MP3_ORDNER.glob(f"*{quelle.video_id}*.mp3"))
    if not kandidaten:
        titel_slug = slug(quelle.titel)
        kandidaten.extend(RAW_MP3_ORDNER.glob(f"*{titel_slug[:30]}*.mp3"))
    return sorted(kandidaten, key=lambda path: path.stat().st_mtime)[-1] if kandidaten else None


def befehl_anzeigen(command: list[str]) -> str:
    """Formatiert einen Befehl kompakt fuer Reports."""

    parts: list[str] = []
    for item in command:
        if " " in item:
            parts.append(json.dumps(item, ensure_ascii=False))
        else:
            parts.append(item)
    return " ".join(parts)


def kurzer_titel(text: str, max_len: int = 72) -> str:
    """Kuerzt lange Videotitel fuer eine ruhige Terminalanzeige."""

    value = " ".join(str(text or "").split())
    value = value.encode("ascii", "ignore").decode("ascii")
    value = " ".join(value.split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def log_ist_nur_unvollstaendig(log_path: Path) -> bool:
    """Erkennt den Zustand "noch nicht genug Clips" statt eines echten Fehlers."""

    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    marker = (
        "Bereit:     nein",
        "training_ready\": false",
        "Ausgewaehlt:",
        "Fehlen:",
        "Fehlend:",
    )
    lower = text.lower()
    return any(item.lower() in lower for item in marker) and "traceback" not in lower


def status_label(code: int, log_path: Path) -> str:
    """Gibt einen ehrlichen Status fuer die Terminalanzeige zurueck."""

    if code == 0:
        return "ok"
    if log_ist_nur_unvollstaendig(log_path):
        return "offen"
    return "fehler"


def dataset_stand_zeile(summary: dict[str, Any]) -> str:
    """Erzeugt eine kompakte Fortschrittszeile fuer den aktuellen Datasetstand."""

    if not summary:
        return "Stand | noch kein Dataset-Summary"
    selected = int(summary.get("selected_total") or 0)
    target = int(summary.get("target_total") or 5000)
    missing = int(summary.get("missing_total") or max(0, target - selected))
    genres = summary.get("selected_by_genre") or summary.get("available_by_genre") or {}
    genre_text = []
    if isinstance(genres, dict):
        genre_target = int(summary.get("target_per_genre") or max(1, target // max(1, len(GENRES))))
        for genre_key, info in GENRES.items():
            current = int(genres.get(genre_key) or 0)
            genre_text.append(f"{info['label']} {current}/{genre_target}")
    return f"Stand | {selected}/{target} Clips | fehlen {missing} | " + " | ".join(genre_text)


def run_command(command: list[str], log_path: Path, trocken: bool, status_text: str = "") -> int:
    """Fuehrt einen Unterbefehl aus und schreibt dessen Ausgabe in ein Log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if trocken:
        log_path.write_text("PLAN: " + befehl_anzeigen(command) + "\n", encoding="utf-8")
        return 0
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + befehl_anzeigen(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(command, cwd=PROJEKTWURZEL, stdout=log, stderr=subprocess.STDOUT, text=True)
        start = time.monotonic()
        while process.poll() is None:
            if status_text:
                elapsed = int(time.monotonic() - start)
                sys.stdout.write(f"\r{status_text} | {elapsed}s")
                sys.stdout.flush()
            time.sleep(2.0)
        code = int(process.returncode or 0)
    if status_text:
        elapsed = int(time.monotonic() - start)
        status = status_label(code, log_path)
        sys.stdout.write(f"\r{status_text} | {status} | {elapsed}s\n")
        sys.stdout.flush()
    return code


def download_mp3(quelle: Quelle, args: argparse.Namespace, log_dir: Path, trocken: bool) -> tuple[int, Path | None, list[str]]:
    """Laedt eine Quelle als MP3 oder nutzt eine vorhandene Datei."""

    vorhandene_mp3 = finde_mp3(quelle)
    if vorhandene_mp3 and not args.erneut_laden:
        return 0, vorhandene_mp3, ["bereits_vorhanden"]
    command = [
        str(PYTHON),
        "code/src/Crawler/quellen_suche.py",
        "mp3",
        "--url",
        quelle.url,
        "--titel",
        quelle.titel,
        "--genre",
        quelle.genre_label,
    ]
    if args.erneut_laden:
        command.append("--erneut-laden")
    if args.cookies:
        command.extend(["--cookies", args.cookies])
    if args.cookies_browser:
        command.extend(["--cookies-browser", args.cookies_browser])
    log_path = log_dir / f"download_{quelle.genre_key}_{quelle.rang}_{quelle.video_id or slug(quelle.titel)}.log"
    code = run_command(command, log_path, trocken, f"Download {quelle.genre_label}")
    if code != 0 and not trocken:
        grund = download_fehler_grund(log_path)
        markiere_fehlquelle(quelle, grund, log_path)
        return code, None, [rel(log_path), grund]
    mp3_path = vorhandene_mp3 if trocken else finde_mp3(quelle)
    return code, mp3_path, [rel(log_path)]


def clippe_mp3(
    quelle: Quelle,
    mp3_path: Path,
    args: argparse.Namespace,
    log_dir: Path,
    trocken: bool,
    status_text: str = "",
    max_clips: int | None = None,
) -> tuple[int, list[str]]:
    """Schneidet eine MP3 direkt in 30s-WAV-Clips.

    Frueher wurde dafuer ein zweiter Python-Prozess gestartet. Bei einigen
    langen MP3s kam es dabei zu nativen Abbruechen mit Returncode -11. Diese
    direkte Variante nutzt pro Segment nur ffmpeg und schreibt danach die
    Manifeste selbst; dadurch bleibt der lokale Lauf stabiler und lesbarer.
    """

    dataset_name = dataset_name_fuer_quelle(quelle)
    log_path = log_dir / f"clippen_{quelle.genre_key}_{quelle.rang}_{quelle.video_id or slug(mp3_path.stem)}.log"
    clip_limit = int(max_clips if max_clips is not None else args.max_clips_pro_quelle)
    if trocken:
        command = [
            "lokaler_ffmpeg_schnitt",
            "--quelle",
            str(mp3_path),
            "--dataset",
            dataset_name,
            "--max-clips",
            str(clip_limit),
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("PLAN: " + befehl_anzeigen(command) + "\n", encoding="utf-8")
        return 0, [rel(log_path)]

    target_root = IMPORT_DATASET / dataset_name
    report_dir = target_root / "reports"
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    rejected: list[dict[str, Any]] = []
    accepted = 0
    duration = 0.0
    starts: list[float] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Quelle: {rel(mp3_path)}\n")
        log.write(f"Dataset: {rel(target_root)}\n")
        log.write(f"Max-Clips: {clip_limit}\n\n")
        try:
            duration = ffprobe_duration(mp3_path)
            usable_start = max(0.0, float(args.skip_intro_sec))
            usable_end = max(usable_start, duration - max(0.0, float(args.skip_outro_sec)))
            current = usable_start
            while current + args.clip_duration_sec <= usable_end + 1e-6:
                starts.append(round(current, 3))
                current += args.clip_hop_sec
            if clip_limit > 0 and len(starts) > clip_limit:
                if clip_limit == 1:
                    starts = [starts[0]]
                else:
                    step = (len(starts) - 1) / (clip_limit - 1)
                    indices = [round(i * step) for i in range(clip_limit)]
                    starts = [starts[i] for i in indices]
            elif clip_limit > 0:
                starts = starts[:clip_limit]
            for index, start_sec in enumerate(starts, start=1):
                split = split_fuer_index(index, args.seed)
                source_id = quelle.video_id or slug(mp3_path.stem)
                hash_short = hashlib.sha1(f"{mp3_path}:{start_sec}".encode("utf-8")).hexdigest()[:8]
                wav_path = target_root / split / "clips" / f"{source_id}__clip_{index:05d}__{hash_short}__30s.wav"
                try:
                    schreibe_wav_segment(mp3_path, wav_path, start_sec, args.clip_duration_sec)
                    if not wav_dauer_ok(wav_path, args.clip_duration_sec):
                        raise RuntimeError("wav_header_ungueltig")
                except Exception as exc:
                    wav_path.unlink(missing_ok=True)
                    rejected.append(
                        {
                            "index": index,
                            "split": split,
                            "start_time_sec": start_sec,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                row = {
                    "path": str(wav_path.resolve()),
                    "duration": float(args.clip_duration_sec),
                    "sample_rate": 32000,
                    "channels": 1,
                    "caption": GENRES[quelle.genre_key]["caption"],
                    "text": GENRES[quelle.genre_key]["caption"],
                    "description": GENRES[quelle.genre_key]["caption"],
                    "split": split,
                    "primary_genre": quelle.genre_key,
                    "lora_genre": quelle.genre_key,
                    "genre": quelle.genre_key,
                    "source_audio_path": str(mp3_path.resolve()),
                    "source_file": mp3_path.name,
                    "source_id": source_id,
                    "source_url": quelle.url,
                    "source_title": quelle.titel,
                    "start_time_sec": start_sec,
                    "end_time_sec": round(start_sec + args.clip_duration_sec, 3),
                    "created_from": "lokaler_mp3_import",
                    "metadata_path": str(wav_path.with_suffix(".json").resolve()),
                }
                wav_path.with_suffix(".json").write_text(
                    json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                rows_by_split[split].append(row)
                accepted += 1
                if status_text:
                    percent = accepted / max(1, len(starts)) * 100.0
                    sys.stdout.write(f"\r{status_text} | Clips {accepted}/{len(starts)} ({percent:5.1f}%)")
                    sys.stdout.flush()
            for split, rows in rows_by_split.items():
                schreibe_jsonl(target_root / split / "data.jsonl", rows)
            report_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_audio": rel(mp3_path),
                "source_id": quelle.video_id or slug(mp3_path.stem),
                "source_url": quelle.url,
                "source_title": quelle.titel,
                "dataset_root": rel(target_root),
                "duration_sec": round(duration, 3),
                "clip_duration_sec": args.clip_duration_sec,
                "clip_hop_sec": args.clip_hop_sec,
                "candidate_count": len(starts),
                "accepted_count": accepted,
                "rejected_count": len(rejected),
                "splits": {split: len(rows) for split, rows in rows_by_split.items()},
                "training_ready": accepted > 0,
                "tool": "clips_vorbereiten.py",
            }
            (target_root / "dataset_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (report_dir / "build_report.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if rejected:
                with (report_dir / "rejected_clips.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["index", "split", "start_time_sec", "reason"])
                    writer.writeheader()
                    writer.writerows(rejected)
            log.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        except Exception as exc:
            log.write(f"FEHLER: {type(exc).__name__}: {exc}\n")
            if status_text:
                sys.stdout.write(f"\r{status_text} | fehler\n")
                sys.stdout.flush()
            return 1, [rel(log_path), f"{type(exc).__name__}: {exc}"]
    if status_text:
        sys.stdout.write(f"\r{status_text} | ok | Clips {accepted}/{len(starts)}\n")
        sys.stdout.flush()
    return (0 if accepted > 0 else 1), [rel(log_path)]


def baue_lora_dataset(
    args: argparse.Namespace,
    log_dir: Path,
    trocken: bool,
    status_text: str = "Dataset",
) -> tuple[int, list[str]]:
    """Baut das LoRA-Zielmanifest neu."""

    command = [
        str(PYTHON),
        "code/src/Dataset/zieldatensatz.py",
        "--overwrite",
        "--ziel-clips",
        str(args.ziel_clips),
        "--genres",
        str(args.genres),
        "--seed",
        str(args.seed),
        "--max-pro-quelle-pro-genre",
        str(args.max_pro_quelle_pro_genre),
    ]
    log_path = log_dir / "zieldatensatz_lora.log"
    return run_command(command, log_path, trocken, status_text), [rel(log_path)]


def validiere_lora_dataset(log_dir: Path, trocken: bool) -> tuple[int, list[str]]:
    """Prueft die neu geschriebenen Manifeste."""

    command = [
        str(PYTHON),
        "code/src/Dataset/datensatz.py",
        "validate",
        "--dataset-root",
        str(ZIEL_DATASET),
    ]
    log_path = log_dir / "validierung_lora.log"
    return run_command(command, log_path, trocken, "Pruefung"), [rel(log_path)]


def repariere_import_manifeste(trocken: bool) -> dict[str, Any]:
    """Schreibt fehlende Import-Manifeste aus vorhandenen Clip-JSONs neu.

    Wenn ein Clip-Prozess abstuerzt, koennen bereits WAVs und Sidecar-JSONs
    vorhanden sein, aber die abschliessenden train/valid/test-Manifeste fehlen.
    Diese Funktion rettet solche Clips fuer den naechsten Dataset-Bau.
    """

    repariert: list[dict[str, Any]] = []
    if not IMPORT_DATASET.exists():
        return {"reparierte_datasets": 0, "reparierte_rows": 0, "details": repariert}
    for dataset_dir in sorted(path for path in IMPORT_DATASET.iterdir() if path.is_dir()):
        dataset_rows = 0
        dataset_changed = False
        split_counts: dict[str, int] = {}
        for split in ("train", "valid", "test"):
            clips_dir = dataset_dir / split / "clips"
            rows: list[dict[str, Any]] = []
            if clips_dir.exists():
                for sidecar in sorted(clips_dir.glob("*.json")):
                    try:
                        row = json.loads(sidecar.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    wav_path = Path(str(row.get("path") or sidecar.with_suffix(".wav")))
                    if not wav_path.exists():
                        wav_path = sidecar.with_suffix(".wav")
                    if not wav_path.exists():
                        continue
                    row["path"] = str(wav_path.resolve())
                    row["split"] = split
                    rows.append(row)
            manifest = dataset_dir / split / "data.jsonl"
            old_count = 0
            if manifest.exists():
                old_count = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
            if old_count != len(rows) or not manifest.exists():
                dataset_changed = True
                if not trocken:
                    manifest.parent.mkdir(parents=True, exist_ok=True)
                    with manifest.open("w", encoding="utf-8") as handle:
                        for row in rows:
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            split_counts[split] = len(rows)
            dataset_rows += len(rows)
        if dataset_changed:
            summary_path = dataset_dir / "dataset_summary.json"
            summary = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dataset_root": rel(dataset_dir),
                "accepted_count": dataset_rows,
                "splits": split_counts,
                "training_ready": dataset_rows > 0,
                "repaired_from_sidecars": True,
            }
            if not trocken:
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            repariert.append({"dataset": rel(dataset_dir), "rows": dataset_rows, "splits": split_counts})
    return {
        "reparierte_datasets": len(repariert),
        "reparierte_rows": sum(int(item["rows"]) for item in repariert),
        "details": repariert,
    }


def lade_summary() -> dict[str, Any]:
    """Liest den aktuellen Zielstand des LoRA-Datasets."""

    path = ZIEL_DATASET / "dataset_summary.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def genre_stand(summary: dict[str, Any], genre_key: str) -> str:
    """Gibt den aktuellen Stand eines Genres als `aktuell/ziel` aus."""

    target = int(summary.get("target_per_genre") or 1000) if summary else 1000
    selected = summary.get("selected_by_genre") or summary.get("available_by_genre") or {}
    current = int(selected.get(genre_key) or 0) if isinstance(selected, dict) else 0
    return f"{current}/{target}"


def genre_voll(summary: dict[str, Any], genre_key: str) -> bool:
    """Prueft, ob das Genre sein Ziel bereits erreicht hat."""

    target = int(summary.get("target_per_genre") or 1000) if summary else 1000
    selected = summary.get("selected_by_genre") or summary.get("available_by_genre") or {}
    current = int(selected.get(genre_key) or 0) if isinstance(selected, dict) else 0
    return current >= target


def fehlende_clips_im_genre(summary: dict[str, Any], genre_key: str) -> int:
    """Berechnet, wie viele Clips fuer ein Genre noch gebraucht werden."""

    target = int(summary.get("target_per_genre") or 1000) if summary else 1000
    selected = summary.get("selected_by_genre") or summary.get("available_by_genre") or {}
    current = int(selected.get(genre_key) or 0) if isinstance(selected, dict) else 0
    return max(0, target - current)


def clip_limit_fuer_quelle(args: argparse.Namespace, summary: dict[str, Any], genre_key: str) -> int:
    """Schneidet nur so viele Clips, wie fuer den aktuellen Stand sinnvoll sind."""

    missing = fehlende_clips_im_genre(summary, genre_key)
    if missing <= 0:
        return 0
    reserve = 10
    return min(int(args.max_clips_pro_quelle), max(10, missing + reserve))


def drucke_genre_stand(summary: dict[str, Any]) -> None:
    """Zeigt den finalen Stand pro Genre kompakt an."""

    if not summary:
        return
    print("Genres:", flush=True)
    for genre_key, info in GENRES.items():
        print(f"{info['label']}: {genre_stand(summary, genre_key)}", flush=True)


def ist_training_ready(summary: dict[str, Any]) -> bool:
    """Prueft, ob der Qualitaetsdatensatz vollstaendig trainingsbereit ist."""

    return bool(summary.get("training_ready")) and int(summary.get("selected_total") or 0) >= int(
        summary.get("target_total") or 5000
    )


def schreibe_reports(run_name: str, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Schreibt JSON- und CSV-Reports zum Lauf."""

    REPORT_ORDNER.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_ORDNER / f"{run_name}.json"
    csv_path = REPORT_ORDNER / f"{run_name}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "genre",
        "rang",
        "video_id",
        "status",
        "titel",
        "url",
        "mp3_path",
        "download_returncode",
        "clip_returncode",
        "hinweis",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Report: {rel(json_path)}", flush=True)


def schreibe_quellen_plan(run_name: str, quellen: list[Quelle], summary: dict[str, Any]) -> tuple[Path, Path]:
    """Schreibt vor dem Clippen eine pruefbare MP3-Quellenliste."""

    REPORT_ORDNER.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_ORDNER / f"quellen_plan_{run_name}.csv"
    txt_path = REPORT_ORDNER / f"quellen_plan_{run_name}.txt"
    counts = {genre_key: 0 for genre_key in GENRES}
    fieldnames = [
        "nummer",
        "genre",
        "genre_label",
        "genre_stand",
        "video_id",
        "titel",
        "url",
        "mp3_path",
        "dataset_name",
        "status",
    ]
    rows: list[dict[str, Any]] = []
    for index, quelle in enumerate(quellen, start=1):
        counts[quelle.genre_key] = counts.get(quelle.genre_key, 0) + 1
        mp3_path = finde_mp3(quelle)
        rows.append(
            {
                "nummer": index,
                "genre": quelle.genre_key,
                "genre_label": quelle.genre_label,
                "genre_stand": genre_stand(summary, quelle.genre_key),
                "video_id": quelle.video_id,
                "titel": quelle.titel,
                "url": quelle.url,
                "mp3_path": rel(mp3_path) if mp3_path else "",
                "dataset_name": dataset_name_fuer_quelle(quelle),
                "status": "wird_geplant",
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["MP3-Quellen vor dem Training", ""]
    lines.append("Verteilung:")
    for genre_key, info in GENRES.items():
        lines.append(f"- {info['label']}: {counts.get(genre_key, 0)} MP3-Dateien")
    lines.append("")
    lines.append("Dateien:")
    for row in rows:
        lines.append(
            f"{int(row['nummer']):02d}. {row['genre_label']} | {row['video_id']} | "
            f"{kurzer_titel(str(row['titel']), 90)}"
        )
        lines.append(f"    {row['mp3_path']}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, txt_path


def main() -> int:
    """Startet die Vorbereitungspipeline."""

    args = parse_args()
    run_name = args.run_name or f"clips_training_vorbereiten_{jetzt_name()}"
    trocken = not args.ausfuehren
    log_dir = REPORT_ORDNER / "logs" / run_name

    vorab_reparatur = {"reparierte_datasets": 0, "reparierte_rows": 0, "details": []}
    vorab_dataset_code: int | str = ""
    if args.lokale_mp3s and not args.nur_download:
        vorab_reparatur = repariere_import_manifeste(trocken)
        if not trocken:
            vorab_dataset_code, _ = baue_lora_dataset(args, log_dir, trocken, "")

    summary_vorher = lade_summary()
    if args.lokale_mp3s:
        args.nur_clippen = True
        quellen, skip_info = lokale_mp3_quellen(args, summary_vorher)
    else:
        quellen, skip_info = lese_quellen(args, summary_vorher)
    rows: list[dict[str, Any]] = []

    print("LoRA-Clip-Vorbereitung", flush=True)
    quelle_text = "lokale MP3s" if args.lokale_mp3s else "Top-5-Quellen"
    print(f"Modus: {'START' if args.ausfuehren else 'PLAN'} | Quelle: {quelle_text} | Dateien: {len(quellen)}", flush=True)
    if (
        skip_info["volle_genres"]
        or skip_info["fertige_quellen"]
        or skip_info["blockierte_quellen"]
        or skip_info.get("schlechte_quellen")
        or skip_info.get("zu_schwache_quellen")
    ):
        print(
            "Uebersprungen: "
            f"volle Genres {len(skip_info['volle_genres'])}, "
            f"fertige Quellen {skip_info['fertige_quellen']}, "
            f"blockierte Links {skip_info['blockierte_quellen']}, "
            f"schlechte Quellen {skip_info.get('schlechte_quellen', 0)}, "
            f"zu schwache Quellen {skip_info.get('zu_schwache_quellen', 0)}",
            flush=True,
        )
    if skip_info["fehlende_top10_csv"]:
        print(f"Fehlende Top-5-CSV: {', '.join(skip_info['fehlende_top10_csv'])}", flush=True)
    if vorab_reparatur.get("reparierte_datasets"):
        reparatur_text = "Wuerde reparieren" if trocken else "Repariert"
        print(
            f"{reparatur_text}: {vorab_reparatur['reparierte_rows']} vorhandene Clip-Metadaten aus "
            f"{vorab_reparatur['reparierte_datasets']} Quelle(n)",
            flush=True,
        )
    quellen_csv, quellen_txt = schreibe_quellen_plan(run_name, quellen, summary_vorher)
    print(f"Quellen-CSV: {rel(quellen_csv)}", flush=True)
    print(f"Quellen-TXT: {rel(quellen_txt)}", flush=True)
    print("", flush=True)

    download_fehler_in_folge = 0
    abbruch_grund = ""
    if ist_training_ready(summary_vorher):
        abbruch_grund = "ziel_bereits_erreicht"
        quellen = []

    for index, quelle in enumerate(quellen, start=1):
        status = "geplant"
        hinweise: list[str] = []
        mp3_path: Path | None = finde_mp3(quelle)
        download_code: int | str = ""
        clip_code: int | str = ""

        stand = genre_stand(summary_vorher, quelle.genre_key)
        print(f"{index:02d}/{len(quellen):02d} | {quelle.genre_label} {stand} | {kurzer_titel(quelle.titel)}", flush=True)
        if not args.nur_clippen:
            download_code, mp3_path, download_hinweise = download_mp3(quelle, args, log_dir, trocken)
            hinweise.extend(download_hinweise)
            if download_code != 0:
                download_fehler_in_folge += 1
                status = "download_fehler"
                rows.append(
                    {
                        "genre": quelle.genre_key,
                        "rang": quelle.rang,
                        "video_id": quelle.video_id,
                        "status": status,
                        "titel": quelle.titel,
                        "url": quelle.url,
                        "mp3_path": rel(mp3_path) if mp3_path else "",
                        "download_returncode": download_code,
                        "clip_returncode": "",
                        "hinweis": "; ".join(hinweise),
                    }
                )
                if (
                    args.max_download_fehler_in_folge > 0
                    and download_fehler_in_folge >= args.max_download_fehler_in_folge
                ):
                    abbruch_grund = (
                        f"{download_fehler_in_folge} Download-Fehler hintereinander. "
                        "Sehr wahrscheinlich blockiert YouTube gerade die neuen Quellen."
                    )
                    print(f"Stop | {abbruch_grund}", flush=True)
                    break
                continue
            download_fehler_in_folge = 0
        if args.nur_download:
            status = "download_ok" if mp3_path else "download_geplant"
        else:
            if not mp3_path and trocken:
                status = "clip_geplant"
            elif not mp3_path:
                status = "mp3_nicht_gefunden"
            else:
                quell_clip_limit = clip_limit_fuer_quelle(args, summary_vorher, quelle.genre_key)
                if quell_clip_limit <= 0:
                    status = "genre_voll"
                    hinweise.append("Genre-Ziel bereits erreicht")
                    rows.append(
                        {
                            "genre": quelle.genre_key,
                            "rang": quelle.rang,
                            "video_id": quelle.video_id,
                            "status": status,
                            "titel": quelle.titel,
                            "url": quelle.url,
                            "mp3_path": rel(mp3_path),
                            "download_returncode": download_code,
                            "clip_returncode": "",
                            "hinweis": "; ".join(hinweise),
                        }
                    )
                    continue
                prozent = index / max(1, len(quellen)) * 100.0
                status_text = f"Datei {index}/{len(quellen)} ({prozent:5.1f}%) | {quelle.genre_label}"
                clip_code, clip_hinweise = clippe_mp3(
                    quelle,
                    mp3_path,
                    args,
                    log_dir,
                    trocken,
                    status_text,
                    max_clips=quell_clip_limit,
                )
                hinweise.extend(clip_hinweise)
                if not trocken:
                    repariere_import_manifeste(False)
                    dataset_dir = IMPORT_DATASET / dataset_name_fuer_quelle(quelle)
                    gerettete_clips = manifest_zeilen(dataset_dir)
                    if clip_code == 0:
                        status = "clip_ok"
                    elif gerettete_clips > 0:
                        status = "clip_ok_mit_rettung"
                        hinweise.append(f"{gerettete_clips} Clips aus abgebrochenem Prozess gerettet")
                    else:
                        status = "clip_fehler"
                    baue_lora_dataset(args, log_dir, False, "")
                    summary_vorher = lade_summary()
                    print(dataset_stand_zeile(summary_vorher), flush=True)
                    if ist_training_ready(summary_vorher):
                        abbruch_grund = "5000_clips_erreicht"
                else:
                    status = "clip_geplant" if clip_code == 0 else "clip_fehler"
        rows.append(
            {
                "genre": quelle.genre_key,
                "rang": quelle.rang,
                "video_id": quelle.video_id,
                "status": status,
                "titel": quelle.titel,
                "url": quelle.url,
                "mp3_path": rel(mp3_path) if mp3_path else "",
                "download_returncode": download_code,
                "clip_returncode": clip_code,
                "hinweis": "; ".join(hinweise),
            }
        )
        if abbruch_grund == "5000_clips_erreicht":
            print("Stop | 5000 Clips erreicht. Weitere Quellen werden nicht geschnitten.", flush=True)
            break

    dataset_code: int | str = ""
    validate_code: int | str = ""
    manifest_reparatur = {"reparierte_datasets": 0, "reparierte_rows": 0, "details": []}
    if not args.nur_download:
        print("", flush=True)
        print("Repariere | Import-Manifeste", flush=True)
        manifest_reparatur = repariere_import_manifeste(trocken)
        print(("Plan" if trocken else "Baue") + " | LoRA-Dataset", flush=True)
        dataset_code, dataset_logs = baue_lora_dataset(args, log_dir, trocken, "")
        print("Plan | Validierung" if trocken else "Pruefe | LoRA-Dataset", flush=True)
        validate_code, validate_logs = validiere_lora_dataset(log_dir, trocken)
    else:
        dataset_logs = []
        validate_logs = []

    summary = lade_summary()
    target_total = int(summary.get("target_total") or args.ziel_clips) if summary else args.ziel_clips
    selected_total = int(summary.get("selected_total") or 0) if summary else 0
    if summary and summary.get("missing_total") is not None:
        missing_total = int(summary.get("missing_total") or 0)
    else:
        missing_total = max(0, target_total - selected_total)
    training_ready = bool(summary.get("training_ready")) if summary else False
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "modus": "ausfuehren" if args.ausfuehren else "plan",
        "quellen_count": len(quellen),
        "max_links_pro_genre": args.max_links_pro_genre,
        "max_clips_pro_quelle": args.max_clips_pro_quelle,
        "raw_mp3_ordner": rel(RAW_MP3_ORDNER),
        "import_dataset": rel(IMPORT_DATASET),
        "ziel_dataset": rel(ZIEL_DATASET),
        "dataset_returncode": dataset_code,
        "validate_returncode": validate_code,
        "abbruch_grund": abbruch_grund,
        "vorab_reparatur": vorab_reparatur,
        "vorab_dataset_returncode": vorab_dataset_code,
        "max_download_fehler_in_folge": args.max_download_fehler_in_folge,
        "dataset_logs": dataset_logs,
        "validate_logs": validate_logs,
        "manifest_reparatur": manifest_reparatur,
        "dataset_summary": summary,
        "rows": rows,
    }
    schreibe_reports(run_name, report, rows)

    print("", flush=True)
    print("Fertig.", flush=True)
    print(f"Clips: {selected_total}/{target_total} | Fehlend: {missing_total} | Training: {'bereit' if training_ready else 'nicht bereit'}", flush=True)
    drucke_genre_stand(summary)
    if trocken:
        print("", flush=True)
        print("Planmodus. Zum echten Start dieselbe Datei ohne --plan starten.", flush=True)
    if trocken or validate_code in ("", 0) or selected_total > 0:
        return 0
    return int(validate_code)


if __name__ == "__main__":
    raise SystemExit(main())
