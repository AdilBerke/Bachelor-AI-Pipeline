#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON_CMD = str(PYTHON if PYTHON.exists() else sys.executable)
START = ROOT / "code" / "start.py"
CRAWLER = ROOT / "code" / "src" / "Crawler" / "quellen_suche.py"
AUDIO_BEWERTUNG = ROOT / "code" / "src" / "Training" / "audio_bewertung.py"
TESTAUDIOS_SAMMELN = ROOT / "code" / "src" / "Training" / "testaudios_sammeln.py"
AUDIO_VEROEFFENTLICHEN = ROOT / "code" / "src" / "Training" / "audio_veroeffentlichen.py"
ENDPRODUKT_ERSTELLEN = ROOT / "code" / "src" / "Training" / "endprodukt_erstellen.py"
GENRE_CONFIG = ROOT / "code" / "configs" / "lora_genres.json"
DATASET_SRC = ROOT / "code" / "src" / "Dataset"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))
try:
    from genre_regeln import QUELLEN_REGEL_VERSION
except Exception:
    QUELLEN_REGEL_VERSION = "lofi_quellen_v4_guitar_breiter_2026_08_11"

UNTERSTUETZTE_TOP5_REGEL_VERSIONEN = {
    QUELLEN_REGEL_VERSION,
    "lofi_quellen_v3_top5_strikt_2026_08_11",
}
DATASET_SUMMARY = ROOT / "daten" / "processed" / "lora_training" / "dataset_summary.json"
LORA_DIR = ROOT / "training" / "musicgen" / "lora_training"
LORA_ADAPTER = LORA_DIR / "adapter.pt"
LORA_STAND = LORA_DIR / "stand.json"
LORA_TRAINING_SCRIPT = ROOT / "code" / "src" / "Training" / "lora.py"
ZIELDATENSATZ_SCRIPT = ROOT / "code" / "src" / "Dataset" / "zieldatensatz.py"
IMPORT_30S_ROOT = ROOT / "daten" / "processed" / "musicgen_youtube_import_30s"
JOB_DIR = ROOT / "training" / "musicgen" / "web_jobs"


YOUTUBE_COOKIES_FILE = ROOT / "code" / "configs" / "youtube_cookies.txt"
if YOUTUBE_COOKIES_FILE.exists():
    os.environ.setdefault("YTDLP_COOKIE_FILE", str(YOUTUBE_COOKIES_FILE))
OUTPUT_DIRS = (
    ROOT / "training" / "ausgaben" / "musicgen_generiert",
    ROOT / "training" / "ausgaben" / "musicgen_loops",
)
ENDPRODUKT_ROOT = ROOT / "training" / "ausgaben" / "endprodukte"
VIDEO_GALLERY_ROOTS = (
    ROOT / "Bachelorarbeit" / "lofi_pipeline" / "scenarios",
    ROOT / "Bachelorarbeit" / "training" / "video",
)
SOURCE_AUDIO_DIR = ROOT / "daten" / "raw" / "audio" / "youtube_imports"
MANUAL_UPLOADS_META = ROOT / "daten" / "metadata" / "downloads" / "manual_uploads.jsonl"
WEBSITE_MANUAL_IMPORTS_META = ROOT / "daten" / "metadata" / "downloads" / "website_manual_imports.jsonl"
SOURCE_METADATA_FILES = (
    ROOT / "daten" / "metadata" / "downloads" / "downloaded_videos.jsonl",
    ROOT / "daten" / "metadata" / "downloads" / "youtube_manual_mp3_imports.jsonl",
    ROOT / "daten" / "metadata" / "downloads" / "heruntergeladene_videos.jsonl",
    ROOT / "daten" / "metadata" / "downloads" / "youtube_manuelle_mp3_importe.jsonl",
    MANUAL_UPLOADS_META,
)


MANUAL_IMPORTERS = {"manual_upload"}
UPLOAD_FORMATS = {"mp3", "wav", "m4a"}
PROCESSED_CLIP_ROOT = ROOT / "daten" / "processed"
LORA_SPLIT_FILES = (
    ROOT / "daten" / "processed" / "lora_training" / "train" / "data.jsonl",
    ROOT / "daten" / "processed" / "lora_training" / "valid" / "data.jsonl",
    ROOT / "daten" / "processed" / "lora_training" / "test" / "data.jsonl",
)
REVIEWS_FILE = ROOT / "training" / "bewertungen" / "musicgen" / "website_reviews.json"
CORS_ORIGIN = os.environ.get("LOFILAB_CORS_ORIGIN", "*").strip() or "*"


LOFI_PIPELINE = ROOT / "Bachelorarbeit" / "lofi_pipeline"
LOFI_SCENARIOS_DIR = LOFI_PIPELINE / "scenarios"
LOFI_CONFIGS_DIR = LOFI_PIPELINE / "configs"
LOFI_TRAIN_SCRIPT = LOFI_PIPELINE / "scripts" / "train_lora.py"
LOFI_EVALUATE_SCRIPT = ROOT / "Bachelorarbeit" / "pipeline" / "realistic_rabbit" / "evaluate_video.py"
VIDEO_PROJECTS_DIR = LOFI_PIPELINE / "projects"
LOFI_GENERATE_SCRIPT = ROOT / "Bachelorarbeit" / "pipeline" / "v003_manual" / "generate.py"

JOBS: dict[str, dict[str, Any]] = {}
PROCESSES: dict[str, subprocess.Popen[str]] = {}
LOCK = threading.Lock()


def jetzt() -> str:

    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path | str) -> str:

    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def slug(text: object) -> str:

    value = str(text or "").strip().lower()
    value = value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "lofi"


def read_json(path: Path, fallback: Any) -> Any:

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def write_json(path: Path, payload: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _normalize_id(text: object) -> str:

    return re.sub(r"[^a-zA-Z0-9]+", "", str(text or "")).lower()


def _merge_source_entries() -> dict[str, dict[str, Any]]:

    merged: dict[str, dict[str, Any]] = {}
    for meta_path in SOURCE_METADATA_FILES:
        for row in read_jsonl(meta_path):
            video_id = str(row.get("video_id") or "").strip()
            if not video_id:
                continue
            entry = merged.setdefault(video_id, {"video_id": video_id})
            for key in (
                "title",
                "webpage_url",
                "audio_path",
                "file_size_bytes",
                "finished_at",
                "status",
                "importer",
                "genre",
            ):
                value = row.get(key)
                if value not in (None, ""):
                    entry[key] = value
    return merged


def _website_manual_video_ids() -> set[str]:

    return {
        _normalize_id(row.get("video_id"))
        for row in read_jsonl(WEBSITE_MANUAL_IMPORTS_META)
        if str(row.get("video_id") or "").strip()
    }


def sources_payload() -> list[dict[str, Any]]:

    merged = _merge_source_entries()
    website_manual = _website_manual_video_ids()
    rows: list[dict[str, Any]] = []
    for video_id, entry in merged.items():
        audio_path = Path(str(entry.get("audio_path") or ""))
        norm = _normalize_id(video_id)
        clip_dirs = 0
        if PROCESSED_CLIP_ROOT.exists():
            clip_dirs = sum(
                1 for p in PROCESSED_CLIP_ROOT.glob("*/*") if p.is_dir() and norm in _normalize_id(p.name)
            )
        rows.append(
            {
                "videoId": video_id,
                "title": str(entry.get("title") or video_id),
                "url": entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
                "audioExists": audio_path.exists(),
                "audioPath": rel(audio_path) if audio_path.name else "",
                "fileSizeBytes": int(entry.get("file_size_bytes") or 0),
                "downloadedAt": entry.get("finished_at") or "",
                "clipDirs": clip_dirs,
                "manual": str(entry.get("importer") or "") in MANUAL_IMPORTERS or norm in website_manual,
            }
        )
    rows.sort(key=lambda r: str(r.get("downloadedAt") or ""), reverse=True)
    return rows


def _top10_quality(path: Path) -> tuple[int, int, float]:

    with_views = 0
    titled = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not _top10_regel_ok(row):
                    continue
                if str(row.get("titel") or "").strip():
                    titled += 1
                if str(row.get("view_count") or "").strip():
                    with_views += 1
    except Exception:
        pass
    return (with_views, titled, path.stat().st_mtime if path.exists() else 0.0)


def _top10_regel_ok(row: dict[str, Any]) -> bool:

    return str(row.get("quellen_regel_version") or "").strip() in UNTERSTUETZTE_TOP5_REGEL_VERSIONEN


def _top10_regel_aktuell(row: dict[str, Any]) -> bool:

    return str(row.get("quellen_regel_version") or "").strip() == QUELLEN_REGEL_VERSION


def _latest_top10_files() -> dict[str, Path]:

    report_dir = ROOT / "daten" / "metadata" / "crawler"
    if not report_dir.exists():
        return {}


    known_genres = sorted(load_genres().keys(), key=len, reverse=True)
    candidates: dict[str, list[Path]] = {}
    for path in report_dir.glob("top10_*.csv"):
        name_lower = path.name.lower()
        if "_fallback" in name_lower or "test" in name_lower:
            continue
        for genre_key in known_genres:
            if genre_key in path.stem:
                candidates.setdefault(genre_key, []).append(path)
                break

    latest: dict[str, Path] = {}
    for genre_key, paths in candidates.items():

        usable = [p for p in paths if _top10_quality(p)[0] > 0]
        if not usable:
            continue


        latest[genre_key] = max(usable, key=lambda p: p.stat().st_mtime)
    return latest


def top10_video_ids() -> set[str]:

    ids: set[str] = set()
    for path in _latest_top10_files().values():
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if not _top10_regel_ok(row):
                        continue
                    video_id = str(row.get("video_id") or "").strip()
                    if video_id:
                        ids.add(_normalize_id(video_id))
        except Exception:
            continue
    return ids


MIN_TOP10_ENTRIES_FUER_BEREINIGUNG = 4


def _unvollstaendige_top10_genres() -> list[str]:

    known_genres = sorted(load_genres().keys())
    files_by_genre = _latest_top10_files()
    incomplete: list[str] = []
    for genre_key in known_genres:
        path = files_by_genre.get(genre_key)
        if path is None:
            incomplete.append(genre_key)
            continue
        with_views, _, _ = _top10_quality(path)
        if with_views < MIN_TOP10_ENTRIES_FUER_BEREINIGUNG:
            incomplete.append(genre_key)
    return incomplete


def prune_non_top10_sources() -> dict[str, Any]:

    incomplete = _unvollstaendige_top10_genres()
    if incomplete:
        return {
            "removedCount": 0,
            "removedVideoIds": [],
            "keptManual": 0,
            "keptTop10": 0,
            "aborted": True,
            "grund": (
                "Abgebrochen: Top5-Liste fehlt oder ist unvollständig (< "
                f"{MIN_TOP10_ENTRIES_FUER_BEREINIGUNG} Einträge) für: "
                f"{', '.join(incomplete)}. Erst eine vollständige Top5-Suche für "
                "diese Genres abschließen, dann erneut versuchen."
            ),
        }

    top10_ids = top10_video_ids()
    merged = _merge_source_entries()
    website_manual = _website_manual_video_ids()
    removed: list[str] = []
    kept_manual = 0
    kept_top10 = 0
    for video_id, entry in merged.items():
        norm = _normalize_id(video_id)
        if str(entry.get("importer") or "") in MANUAL_IMPORTERS or norm in website_manual:
            kept_manual += 1
            continue
        if norm in top10_ids:
            kept_top10 += 1
            continue
        delete_source(video_id)
        removed.append(video_id)
    return {
        "removedCount": len(removed),
        "removedVideoIds": removed,
        "keptManual": kept_manual,
        "keptTop10": kept_top10,
    }


def _download_failure_reasons() -> dict[str, dict[str, Any]]:

    path = ROOT / "daten" / "metadata" / "crawler" / "download_fehler_quellen.jsonl"
    reasons: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        video_id = str(row.get("video_id") or "").strip()
        if video_id:
            reasons[_normalize_id(video_id)] = row
    return reasons


def top10_status_payload() -> list[dict[str, Any]]:

    source_ids = {_normalize_id(video_id) for video_id in _merge_source_entries().keys()}
    failure_reasons = _download_failure_reasons()
    files_by_genre = _latest_top10_files()

    result: list[dict[str, Any]] = []
    for genre_key in sorted(load_genres().keys()):
        path = files_by_genre.get(genre_key)
        entries: list[dict[str, Any]] = []
        if path is None:
            result.append({"genreKey": genre_key, "label": genre_label(genre_key), "entries": entries})
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    video_id = str(row.get("video_id") or "").strip()
                    if not video_id:
                        continue
                    norm = _normalize_id(video_id)
                    failure = failure_reasons.get(norm)
                    if norm in source_ids:
                        status = "im_datensatz"
                    elif failure:
                        status = "manuell_noetig"
                    else:
                        status = "offen"
                    entries.append(
                        {
                            "rang": _number(row.get("rang")),
                            "videoId": video_id,
                            "titel": row.get("titel") or video_id,
                            "kanal": row.get("kanal") or "",
                            "url": row.get("url") or f"https://www.youtube.com/watch?v={video_id}",
                            "viewCount": _number(row.get("view_count")),
                            "likeCount": _number(row.get("like_count")),
                            "status": status,
                            "grund": failure.get("grund") if failure else None,
                            "regelVersion": row.get("quellen_regel_version") or "",
                            "regelAktuell": _top10_regel_aktuell(row),
                        }
                    )
        except Exception:
            continue
        result.append({"genreKey": genre_key, "label": genre_label(genre_key), "entries": entries})
    return result


def _pending_top10_entries(genre_keys: set[str] | None = None) -> list[dict[str, str]]:

    entries: list[dict[str, str]] = []
    for genre in top10_status_payload():
        if genre_keys and genre["genreKey"] not in genre_keys:
            continue
        for row in genre["entries"]:
            if row.get("status") != "offen":
                continue
            entries.append(
                {
                    "url": str(row.get("url") or ""),
                    "titel": str(row.get("titel") or ""),
                    "genre": str(genre.get("label") or ""),
                }
            )
    return entries


def save_uploaded_source(*, filename: str, genre: str, title: str, data: bytes) -> dict[str, Any]:

    if not data:
        raise ValueError("Leere Datei")
    ext = Path(filename or "").suffix.lower().lstrip(".") or "mp3"
    if ext not in UPLOAD_FORMATS:
        raise ValueError(f"Format nicht unterstuetzt: {ext}")

    stem = Path(filename or "").stem.strip() or "upload"
    display_title = (title or "").strip() or stem
    if genre:
        display_title = f"{genre_label(genre)} - {display_title}"
    video_id = f"LOCAL{hashlib.sha1(f'{filename}-{time.time()}'.encode('utf-8')).hexdigest()[:8].upper()}"

    SOURCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", display_title).strip()[:150] or "upload"
    target = SOURCE_AUDIO_DIR / f"{safe_title} [{video_id}].{ext}"
    target.write_bytes(data)

    entry = {
        "video_id": video_id,
        "title": display_title,
        "webpage_url": "",
        "status": "downloaded",
        "audio_format": ext,
        "audio_path": str(target),
        "file_size_bytes": len(data),
        "finished_at": jetzt(),
        "genre": profile_from_label(genre) if genre else "",
        "importer": "manual_upload",
    }
    rows = read_jsonl(MANUAL_UPLOADS_META)
    rows.append(entry)
    write_jsonl(MANUAL_UPLOADS_META, rows)

    return {
        "videoId": video_id,
        "title": display_title,
        "audioPath": rel(target),
        "fileSizeBytes": len(data),
    }


def delete_source(video_id: str) -> dict[str, Any]:

    video_id = str(video_id or "").strip()
    if not video_id:
        raise ValueError("Video-ID fehlt")
    norm = _normalize_id(video_id)

    removed_audio = 0
    if SOURCE_AUDIO_DIR.exists():
        for path in list(SOURCE_AUDIO_DIR.glob("*")):
            if path.is_file() and norm in _normalize_id(path.stem):
                path.unlink()
                removed_audio += 1

    removed_meta = 0
    for meta_path in SOURCE_METADATA_FILES:
        rows = read_jsonl(meta_path)
        if not rows:
            continue
        kept = [row for row in rows if str(row.get("video_id") or "").strip() != video_id]
        if len(kept) != len(rows):
            removed_meta += len(rows) - len(kept)
            write_jsonl(meta_path, kept)

    removed_clip_dirs = 0
    if PROCESSED_CLIP_ROOT.exists():
        for path in list(PROCESSED_CLIP_ROOT.glob("*/*")):
            if path.is_dir() and norm in _normalize_id(path.name):
                shutil.rmtree(path, ignore_errors=True)
                removed_clip_dirs += 1

    removed_manifest_rows = 0
    for split_path in LORA_SPLIT_FILES:
        rows = read_jsonl(split_path)
        if not rows:
            continue
        kept = [row for row in rows if norm not in _normalize_id(row.get("path") or "")]
        if len(kept) != len(rows):
            removed_manifest_rows += len(rows) - len(kept)
            write_jsonl(split_path, kept)

    return {
        "videoId": video_id,
        "removedAudioFiles": removed_audio,
        "removedMetadataEntries": removed_meta,
        "removedClipDirs": removed_clip_dirs,
        "removedManifestRows": removed_manifest_rows,
    }


def load_genres() -> dict[str, dict[str, Any]]:

    payload = read_json(GENRE_CONFIG, {})
    genres = payload.get("genres") if isinstance(payload, dict) else None
    return genres if isinstance(genres, dict) else {}


def save_genres(genres: dict[str, dict[str, Any]]) -> None:

    write_json(GENRE_CONFIG, {"genres": genres})


def genre_label(key_or_label: str) -> str:

    genres = load_genres()
    if key_or_label in genres:
        return str(genres[key_or_label].get("label") or key_or_label)
    normalized = slug(key_or_label)
    if normalized in genres:
        return str(genres[normalized].get("label") or key_or_label)
    text = str(key_or_label or "Chill Lofi").strip()
    return text if "lofi" in text.lower().replace("-", "") else f"{text} Lofi"


def profile_from_label(label: str) -> str:

    genres = load_genres()
    normalized = slug(label)
    if normalized in genres:
        return normalized
    for key, info in genres.items():
        if str(info.get("label") or "").lower() == str(label).lower():
            return key
    return normalized


def parse_duration_minutes(value: object) -> int:

    try:
        return max(1, int(round(float(value))))
    except Exception:
        return 20


def duration_text_to_minutes(value: object, fallback: int = 20) -> int:

    text = str(value or "").strip().lower().replace(",", ".").replace(" ", "")
    if not text:
        return fallback
    unit = text[-1] if text[-1:] in {"s", "m", "h"} else "m"
    number_text = text[:-1] if unit in {"s", "m", "h"} else text
    try:
        number = float(number_text)
    except ValueError:
        return fallback
    if unit == "h":
        return max(1, int(round(number * 60)))
    if unit == "s":
        return max(1, int(round(number / 60)))
    return max(1, int(round(number)))


def command_python(*args: str) -> list[str]:

    return [PYTHON_CMD, *args]


def start_command(
    *,
    kind: str,
    commands: list[list[str]],
    prompt_profile: str = "chillhop_lofi",
    duration_min: int = 0,
    output_format: str = "wav_mp3",
    desired_name: str = "",
) -> dict[str, Any]:

    job_id = f"{kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOB_DIR / f"{job_id}.log"
    job = {
        "jobId": job_id,
        "status": "queued",
        "promptProfile": prompt_profile,
        "targetDurationMin": duration_min,
        "outputFormat": output_format,
        "startedAt": jetzt(),
        "progress": 0.0,
        "step": "model_loading",
        "logPath": rel(log_path),
        "commands": commands,
        "kind": kind,
        "lastLines": [],
        "desiredName": desired_name.strip()[:120],
    }
    with LOCK:
        JOBS[job_id] = job

    thread = threading.Thread(target=_run_job, args=(job_id, commands, log_path), daemon=True)
    thread.start()
    return job


def _set_job(job_id: str, **updates: Any) -> None:

    with LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(updates)


def _append_line(job_id: str, line: str) -> None:

    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        lines = list(job.get("lastLines") or [])
        lines.append(line.rstrip())
        job["lastLines"] = lines[-30:]


        command_total = max(1, int(job.get("commandTotal") or 1))
        command_index = max(1, int(job.get("commandIndex") or 1))
        command_progress = _progress_from_line(line, float(job.get("commandProgress") or 0.0))
        if command_progress is not None:
            job["commandProgress"] = command_progress
            slice_start = (command_index - 1) / command_total
            overall = slice_start + command_progress / command_total
            job["progress"] = max(float(job.get("progress") or 0.0), overall)
        job["step"] = _step_from_line(line, str(job.get("step") or "model_loading"))


def _progress_from_line(line: str, current: float) -> float | None:


    gesamt = re.search(r"Gesamt:\s*(\d+(?:\.\d+)?)\s*%", line)
    if gesamt:


        return min(0.99, max(current, float(gesamt.group(1)) / 100.0))


    trainingsbeispiele = re.search(r"Trainingsbeispiele:\s*(\d+)\s*/\s*(\d+)", line)
    if trainingsbeispiele:
        done = float(trainingsbeispiele.group(1))
        total = max(1.0, float(trainingsbeispiele.group(2)))
        return min(1.0, max(current, done / total))
    if "VRAM-Limit" in line or "Rest:" in line:
        return None
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
    if percent:
        return min(1.0, max(current, float(percent.group(1)) / 100.0))


    for pattern, phase_start, phase_end in (
        (r"Suche\s+(\d+)\s*/\s*(\d+)", 0.02, 0.75),
        (r"Details\s+(\d+)\s*/\s*(\d+)", 0.75, 0.97),
    ):
        match = re.search(pattern, line)
        if match:
            done = float(match.group(1))
            total = max(1.0, float(match.group(2)))
            anteil = phase_start + (phase_end - phase_start) * (done / total)
            return min(0.98, max(current, anteil))
    for pattern in (
        r"Clips trainiert:\s*(\d+)\s*/\s*(\d+)",
        r"Datei\s+(\d+)\s*/\s*(\d+)",
        r"Clip\s+(\d+)\s*/\s*(\d+)",
        r"LoRA Step\s+(\d+)\s*/\s*(\d+)",
        r"Import\s+(\d+)\s*/\s*(\d+)",
    ):
        match = re.search(pattern, line)
        if match:
            done = float(match.group(1))
            total = max(1.0, float(match.group(2)))
            return min(0.98, max(current, done / total))
    return None


def _step_from_line(line: str, current: str) -> str:

    lower = line.lower()
    if "export" in lower or "mp3" in lower or "wav" in lower:
        return "export"
    if "pruef" in lower or "qualitaet" in lower or "score" in lower:
        return "segment_check"
    if "crossfade" in lower or "uebergang" in lower or "fade" in lower:
        return "crossfade"
    if "generier" in lower or "kandidat" in lower or "clip" in lower:
        return "segment_generation"
    return current


def _run_job(job_id: str, commands: list[list[str]], log_path: Path) -> None:

    _set_job(job_id, status="running", startedAt=jetzt(), progress=0.01)
    returncode = 0
    started = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for index, command in enumerate(commands, start=1):
                if not command:
                    continue
                command_label = None
                if "--genre" in command:
                    try:
                        command_label = command[command.index("--genre") + 1]
                    except IndexError:
                        command_label = None
                _set_job(
                    job_id,
                    commandIndex=index,
                    commandTotal=len(commands),
                    commandLabel=command_label,
                    commandProgress=0.0,
                    progress=max(float(JOBS[job_id].get("progress") or 0.0), (index - 1) / len(commands)),
                )
                log.write(f"\nBefehl {index}/{len(commands)}: {' '.join(command)}\n\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,


                    start_new_session=True,
                )
                with LOCK:
                    PROCESSES[job_id] = process
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    _append_line(job_id, line)
                returncode = process.wait()
                with LOCK:
                    PROCESSES.pop(job_id, None)
                if returncode != 0:
                    break
                _set_job(job_id, progress=max(float(JOBS[job_id].get("progress") or 0.0), index / len(commands) * 0.95))
    except Exception as exc:
        _set_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}", finishedAt=jetzt())
        return

    if returncode == 0:
        audio_id = _latest_audio_id(since=started)
        updates = {
            "status": "completed",
            "finishedAt": jetzt(),
            "progress": 1.0,
            "step": "done",
        }
        if audio_id:
            updates["audioId"] = audio_id
            desired_name = str(JOBS.get(job_id, {}).get("desiredName") or "").strip()
            if desired_name:
                run_dir = _audio_map().get(audio_id)
                if run_dir is not None:
                    _set_audio_display_name(run_dir, desired_name)
        _set_job(job_id, **updates)
    else:
        _set_job(job_id, status="failed", error=f"Exit-Code {returncode}", finishedAt=jetzt())


def cancel_job(job_id: str) -> bool:

    with LOCK:
        process = PROCESSES.get(job_id)
    if process and process.poll() is None:
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        else:
            def _harte_nachkontrolle(proc: subprocess.Popen[str], group_id: int) -> None:
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(group_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            threading.Thread(target=_harte_nachkontrolle, args=(process, pgid), daemon=True).start()
    _set_job(job_id, status="cancelled", finishedAt=jetzt())
    return True


def _dataset_overview_for(summary_path: Path) -> dict[str, Any]:

    summary = read_json(summary_path, {})
    genres: dict[str, Any] = {}
    if isinstance(summary, dict):
        diagnostics = summary.get("source_split_diagnostics")
        if isinstance(diagnostics, list):
            for row in diagnostics:
                if not isinstance(row, dict):
                    continue
                key = str(row.get("genre") or "")
                if not key:
                    continue
                source_keys: list[str] = []
                by_split = row.get("source_keys_by_split")
                if isinstance(by_split, dict):
                    for split_keys in by_split.values():
                        if isinstance(split_keys, list):
                            source_keys.extend(str(k) for k in split_keys)
                genres[key] = {
                    "selected": int(row.get("selected_total") or 0),
                    "target": int(row.get("target") or 0),
                    "missing": max(0, int(row.get("target") or 0) - int(row.get("selected_total") or 0)),
                    "sources": int(row.get("source_count") or 0),
                    "sourceKeys": sorted(set(source_keys)),
                }
        elif isinstance(summary.get("genres"), dict):
            genres = summary["genres"]
    return {
        "path": rel(summary_path.parent),
        "ready": bool(summary.get("training_ready")) if isinstance(summary, dict) else False,
        "selectedTotal": int(summary.get("selected_total") or 0) if isinstance(summary, dict) else 0,
        "targetTotal": int(summary.get("target_total") or 5000) if isinstance(summary, dict) else 5000,
        "missingTotal": int(summary.get("missing_total") or 0) if isinstance(summary, dict) else 0,
        "rejectedTotal": int(summary.get("rejected_total") or 0) if isinstance(summary, dict) else 0,
        "genres": genres,
    }


def dataset_overview() -> dict[str, Any]:

    return _dataset_overview_for(DATASET_SUMMARY)


def genre_dataset_dir(genre_key: str) -> Path:

    return ROOT / "daten" / "processed" / f"lora_training_{genre_key}"


def genre_dataset_overview(genre_key: str) -> dict[str, Any]:

    return _dataset_overview_for(genre_dataset_dir(genre_key) / "dataset_summary.json")


def _list_checkpoints(run_dir: Path) -> list[dict[str, Any]]:

    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    found: list[dict[str, Any]] = []
    for entry in checkpoints_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("step_"):
            continue
        adapter_file = entry / "lora_adapter.pt"
        if not adapter_file.exists():
            continue
        try:
            step = int(entry.name.removeprefix("step_"))
        except ValueError:
            continue
        found.append({"step": step, "path": rel(adapter_file)})
    found.sort(key=lambda item: item["step"], reverse=True)
    return found


def _lora_status_for(run_dir: Path) -> dict[str, Any]:

    stand = read_json(run_dir / "stand.json", {})
    adapter = run_dir / "adapter.pt"
    if isinstance(stand, dict):
        value = stand.get("adapter") or stand.get("best_adapter") or stand.get("checkpoint")
        if value:
            candidate = ROOT / str(value) if not Path(str(value)).is_absolute() else Path(str(value))
            adapter = candidate
    return {
        "adapter": rel(adapter),
        "adapterExists": adapter.exists(),
        "status": stand.get("status", "fehlt") if isinstance(stand, dict) else "fehlt",
        "checkpointStep": stand.get("checkpoint_step") if isinstance(stand, dict) else None,
        "runName": (stand.get("aktiver_run") or stand.get("run_name")) if isinstance(stand, dict) else None,
        "checkpoints": _list_checkpoints(run_dir),
    }


def lora_status() -> dict[str, Any]:

    return _lora_status_for(LORA_DIR)


def genre_run_dir(genre_key: str) -> Path:

    return ROOT / "training" / "musicgen" / f"lora_training_{genre_key}"


def genre_lora_status(genre_key: str) -> dict[str, Any]:

    return _lora_status_for(genre_run_dir(genre_key))


def lora_genres_overview() -> dict[str, Any]:

    result: dict[str, Any] = {}
    for genre_key in load_genres().keys():
        result[genre_key] = {**genre_lora_status(genre_key), "dataset": genre_dataset_overview(genre_key)}
    return result


def resolve_generation_adapter(genre_key: str) -> tuple[Path | None, str]:

    genre_stand = read_json(genre_run_dir(genre_key) / "stand.json", {})
    genre_adapter = genre_run_dir(genre_key) / "adapter.pt"
    if isinstance(genre_stand, dict) and genre_stand.get("status") == "freigegeben" and genre_adapter.exists():
        return genre_adapter, "genre"

    shared_stand = read_json(LORA_STAND, {})
    if isinstance(shared_stand, dict) and shared_stand.get("status") == "freigegeben" and LORA_ADAPTER.exists():
        return LORA_ADAPTER, "shared_fallback"

    return None, "none"


def resolve_review_adapter(genre_key: str) -> tuple[Path | None, str]:

    adapter, source = resolve_generation_adapter(genre_key)
    if adapter is not None:
        return adapter, source

    shared_stand = read_json(LORA_STAND, {})
    if not isinstance(shared_stand, dict) or shared_stand.get("status") != "bewertung_offen":
        return None, "none"
    for key in ("candidate_adapter", "adapter", "checkpoint"):
        value = str(shared_stand.get(key) or "").strip()
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path, "shared_candidate"
    return None, "none"


def _dir_size(path: Path) -> int:

    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _archive_rename(path: Path) -> str:

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = path.parent / f"{path.name}_archiv_{timestamp}"
    path.rename(archive_path)
    return rel(archive_path)


def archive_lora_run() -> str | None:

    has_content = LORA_ADAPTER.exists() or LORA_STAND.exists() or (LORA_DIR / "checkpoints").exists()
    if not LORA_DIR.exists() or not has_content:
        return None
    return _archive_rename(LORA_DIR)


def archive_genre_dir(path: Path) -> str | None:

    if not path.exists() or not any(path.iterdir()):
        return None
    return _archive_rename(path)


def list_archives() -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []
    roots = [
        (ROOT / "daten" / "processed", "dataset"),
        (ROOT / "training" / "musicgen", "training"),
    ]
    prefix = "lora_training_"
    for root, kind in roots:
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or "_archiv_" not in entry.name:
                continue
            name_part, _, ts_part = entry.name.rpartition("_archiv_")
            genre_key = name_part[len(prefix):] if name_part.startswith(prefix) and name_part != "lora_training" else ""
            rows.append(
                {
                    "path": rel(entry),
                    "kind": kind,
                    "genreKey": genre_key,
                    "sizeBytes": _dir_size(entry),
                    "archivedAt": ts_part,
                }
            )
    rows.sort(key=lambda row: str(row["archivedAt"]), reverse=True)
    return rows


def delete_archive(path_str: str) -> dict[str, Any]:

    target = (ROOT / str(path_str)).resolve()
    try:
        target.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Pfad liegt ausserhalb des Projekts") from exc
    if "_archiv_" not in target.name:
        raise ValueError("Nur archivierte Ordner koennen hier geloescht werden")
    if not target.exists() or not target.is_dir():
        raise ValueError("Archiv-Ordner nicht gefunden")
    freed = _dir_size(target)
    shutil.rmtree(target)
    return {"deleted": rel(target), "freedBytes": freed}


def prompt_profiles() -> list[dict[str, str]]:

    rows = []
    for key, info in load_genres().items():
        label = str(info.get("label") or key.replace("_", " ").title())
        prompt = str(info.get("caption") or "calm lofi instrumental, no vocals")
        rows.append(
            {
                "id": key,
                "label": label,
                "description": prompt[:120],
                "basePrompt": prompt,
            }
        )
    return rows


def model_status() -> dict[str, Any]:

    lora = lora_status()
    return {
        "online": True,
        "baseModel": "facebook/musicgen-melody-large",
        "activeAdapter": lora["adapter"],
        "activeCheckpoint": str(lora.get("checkpointStep") or "LoRA"),
        "device": _device_text(),
        "training": {"active": _process_active("musicgen_steuerung.py train")},
    }


def _device_text() -> str:

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        text = result.stdout.strip().splitlines()[0]
        return f"cuda:0 ({text})" if text else "lokal"
    except Exception:
        return "lokal"


def _process_active(marker: str) -> bool:

    try:
        result = subprocess.run(["ps", "-eo", "cmd"], cwd=ROOT, text=True, stdout=subprocess.PIPE, timeout=2)
    except Exception:
        return False
    return any(marker in line and "web_api.py" not in line for line in result.stdout.splitlines())


def models_payload() -> dict[str, Any]:

    checkpoints = []
    for adapter in sorted(LORA_DIR.glob("checkpoints/*/lora_adapter.pt")):
        step_match = re.search(r"step[_-]?(\d+)", adapter.parent.name)
        step = int(step_match.group(1)) if step_match else 0
        checkpoints.append(
            {
                "id": adapter.parent.name,
                "step": step,
                "createdAt": datetime.fromtimestamp(adapter.stat().st_mtime).isoformat(),
                "recommended": False,
                "notes": rel(adapter),
            }
        )
    if LORA_ADAPTER.exists():
        checkpoints.append(
            {
                "id": "lora_training",
                "step": int(lora_status().get("checkpointStep") or 0),
                "createdAt": datetime.fromtimestamp(LORA_ADAPTER.stat().st_mtime).isoformat(),
                "recommended": True,
                "notes": rel(LORA_ADAPTER),
            }
        )
    return {"baseModel": "facebook/musicgen-melody-large", "checkpoints": checkpoints}


def _latest_audio_id(since: float) -> str:

    best_dir: Path | None = None
    best_time = since
    for base in OUTPUT_DIRS:
        if not base.exists():
            continue
        for path in base.glob("*/lange_audio.mp3"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= best_time:
                best_time = mtime
                best_dir = path.parent
    return _audio_id(best_dir) if best_dir else ""


def _audio_id(run_dir: Path | None) -> str:

    if run_dir is None:
        return ""
    return hashlib.sha1(rel(run_dir).encode("utf-8")).hexdigest()[:16]


def _audio_display_name(run_dir: Path) -> str | None:

    data = read_json(run_dir / "anzeigename.json", {})
    name = str(data.get("name") or "").strip() if isinstance(data, dict) else ""
    return name or None


def _set_audio_display_name(run_dir: Path, name: str) -> str:

    cleaned = name.strip()[:120]
    if not cleaned:
        path = run_dir / "anzeigename.json"
        if path.exists():
            path.unlink()
        return ""
    write_json(run_dir / "anzeigename.json", {"name": cleaned})
    return cleaned


def _audio_map() -> dict[str, Path]:

    mapping: dict[str, Path] = {}
    for base in OUTPUT_DIRS:
        if not base.exists():
            continue
        for run_dir in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
            if (run_dir / "lange_audio.mp3").exists() or (run_dir / "lange_audio.wav").exists():
                mapping[_audio_id(run_dir)] = run_dir
    return mapping


def _gallery_id(path: Path) -> str:

    return hashlib.sha1(rel(path).encode("utf-8")).hexdigest()[:16]


def _video_gallery_map() -> dict[str, Path]:

    mapping: dict[str, Path] = {}
    for root in VIDEO_GALLERY_ROOTS:
        if not root.exists():
            continue
        for gif_path in root.rglob("*.gif"):
            if gif_path.is_file():
                mapping[_gallery_id(gif_path)] = gif_path

        for mp4_path in root.rglob("*.mp4"):
            if mp4_path.parent.name == "samples" and mp4_path.is_file():
                mapping[_gallery_id(mp4_path)] = mp4_path
    return mapping


def _gallery_scenario_label(path: Path) -> str:

    parts = rel(path).split("/")
    for marker in ("scenarios", "video"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return "sonstige"


def _gallery_round(path: Path) -> int:
    for part in rel(path).split("/"):
        if part.startswith("round_"):
            try:
                return int(part.split("_", 1)[1])
            except (IndexError, ValueError):
                pass
    return 0


def _gallery_step(filename: str) -> int:
    import re as _re
    m = _re.match(r"step_(\d+)", filename)
    return int(m.group(1)) if m else 0


def _endprodukt_map() -> dict[str, Path]:

    mapping: dict[str, Path] = {}
    if not ENDPRODUKT_ROOT.exists():
        return mapping
    for run_dir in sorted(
        (p for p in ENDPRODUKT_ROOT.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if (run_dir / "final.mp4").exists():
            mapping[_gallery_id(run_dir)] = run_dir
    return mapping


def endprodukt_payload() -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []
    for endprodukt_id, run_dir in _endprodukt_map().items():
        info = read_json(run_dir / "info.json", {})
        info = info if isinstance(info, dict) else {}
        stat = (run_dir / "final.mp4").stat()
        rows.append(
            {
                "id": endprodukt_id,
                "audioTitel": str(info.get("audioTitel") or ""),
                "videoLabel": str(info.get("videoLabel") or ""),
                "createdAt": str(info.get("erstelltAm") or datetime.fromtimestamp(stat.st_mtime).isoformat()),
                "sizeBytes": stat.st_size,
                "url": f"/api/final-products/{endprodukt_id}/file",
            }
        )
    rows.sort(key=lambda row: str(row["createdAt"]), reverse=True)
    return rows


def video_gallery_payload() -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []
    for gallery_id, path in _video_gallery_map().items():
        stat = path.stat()
        rows.append(
            {
                "id": gallery_id,
                "scenario": _gallery_scenario_label(path),
                "filename": path.name,
                "url": f"/api/video-gallery/{gallery_id}/file",
                "type": "mp4" if path.suffix.lower() == ".mp4" else "gif",
                "sizeBytes": stat.st_size,
                "modifiedAt": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "round": _gallery_round(path),
                "step": _gallery_step(path.name),
            }
        )

    rows.sort(key=lambda row: (str(row["scenario"]), -int(row["round"]), -int(row["step"])))
    return rows


def _wav_duration(path: Path) -> float:

    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return 0.0


def _audio_duration(run_dir: Path, report: dict[str, Any]) -> int:

    for key in ("duration_sec", "duration_seconds", "final_duration_sec", "ziel_dauer_sekunden"):
        try:
            value = float(report.get(key) or 0)
        except Exception:
            value = 0.0
        if value > 0:
            return int(round(value))
    wav_path = run_dir / "lange_audio.wav"
    return int(round(_wav_duration(wav_path))) if wav_path.exists() else 0


def audio_payload() -> list[dict[str, Any]]:

    rows = []
    for audio_id, run_dir in _audio_map().items():
        report = read_json(run_dir / "finale_audio_info.json", {})
        if not isinstance(report, dict) or not report:
            report = read_json(run_dir / "generation_report.json", {})
        report = report if isinstance(report, dict) else {}
        genre = str(report.get("genre") or report.get("prompt_profile") or run_dir.name.split("_")[0] or "lofi")
        formats = "wav_mp3" if (run_dir / "lange_audio.mp3").exists() and (run_dir / "lange_audio.wav").exists() else "mp3"
        if (run_dir / "lange_audio.wav").exists() and not (run_dir / "lange_audio.mp3").exists():
            formats = "wav"
        score_rating = read_json(run_dir / "score_bewertung.json", None)
        rows.append(
            {
                "id": audio_id,
                "title": _audio_display_name(run_dir) or run_dir.name,
                "durationSec": _audio_duration(run_dir, report),
                "createdAt": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                "promptProfile": profile_from_label(genre),
                "prompt": str(report.get("prompt") or report.get("stimmung") or genre),
                "checkpoint": str(report.get("adapter") or report.get("checkpoint") or "LoRA"),
                "formats": formats,
                "reportUrl": f"/api/audio/{audio_id}/report",
                "metadata": {
                    "ordner": rel(run_dir),
                    "crossfade": report.get("crossfade_seconds") or report.get("crossfade_sekunden") or 3,
                    "bericht": report.get("scientific_report") or "",
                    "uebergaenge": (
                        (report.get("transition_quality_report") or {}).get("problem_count")
                        if isinstance(report.get("transition_quality_report"), dict)
                        else ""
                    ),
                },
                "scoreRating": score_rating,
            }
        )
    # Auf der Website sollen aktuell nur die 5 finalen Kontroll-Audios (eine
    # pro Genre) sichtbar sein, nicht die ~140 restlichen Test-/Debug-Laeufe.
    # Die anderen Audios bleiben auf der Platte unveraendert erhalten, sie
    # werden hier nur aus der Anzeige gefiltert.
    rows = [row for row in rows if str(row.get("title") or "").startswith("Kontrolle_")]
    return rows


def _rating_from_review_text(text: str) -> int:

    value = text.lower()
    good = sum(value.count(term) for term in ("sehr gut", "gute audio", "top audio", "gefaellt mir"))
    bad = sum(value.count(term) for term in ("schlecht", "kein ton", "monoton", "signalton", "rauschen"))
    score = 3 + min(2, good // 2) - min(2, bad // 2)
    return max(1, min(5, score))


def _csv_review_rows() -> list[dict[str, Any]]:

    root = ROOT / "training" / "bewertungen" / "musicgen"
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(root.glob("*/bewertung.csv"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            text = csv_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        rel_path = rel(csv_path)
        rows.append(
            {
                "id": hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12],
                "round": csv_path.parent.name,
                "checkpoint": "LoRA",
                "prompt": rel_path,
                "csvUrl": "",
                "rating": _rating_from_review_text(text),
                "notes": "Aus vorhandener Bewertung CSV geladen.",
                "issues": [],
                "positives": [],
                "createdAt": datetime.fromtimestamp(csv_path.stat().st_mtime).isoformat(),
            }
        )
    return rows


def reviews_payload() -> list[dict[str, Any]]:

    saved = read_json(REVIEWS_FILE, [])
    saved_rows = saved if isinstance(saved, list) else []
    existing_ids = {str(row.get("id") or "") for row in saved_rows if isinstance(row, dict)}
    csv_rows = [row for row in _csv_review_rows() if row["id"] not in existing_ids]
    rows = [row for row in saved_rows if isinstance(row, dict)] + csv_rows
    rows.sort(key=lambda row: str(row.get("createdAt") or ""), reverse=True)
    return rows


def add_review(body: dict[str, Any]) -> dict[str, Any]:

    review = {
        "id": f"rev_{uuid.uuid4().hex[:10]}",
        "round": str(body.get("round") or "Bewertung"),
        "checkpoint": str(body.get("checkpoint") or "LoRA"),
        "prompt": str(body.get("prompt") or ""),
        "csvUrl": str(body.get("csvUrl") or ""),
        "rating": max(1, min(5, int(body.get("rating") or 3))),
        "notes": str(body.get("notes") or ""),
        "issues": _list_value(body.get("issues")),
        "positives": _list_value(body.get("positives")),
        "createdAt": jetzt(),
    }
    saved = read_json(REVIEWS_FILE, [])
    rows = saved if isinstance(saved, list) else []
    rows.insert(0, review)
    write_json(REVIEWS_FILE, rows)
    return review


def top10_results(limit: int = 40) -> list[dict[str, Any]]:

    report_dir = ROOT / "daten" / "metadata" / "crawler"
    if not report_dir.exists():
        return []
    paths = sorted(
        report_dir.glob("top10_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths[:8]:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if not _top10_regel_ok(row):
                        continue
                    url = str(row.get("url") or "").strip()
                    video_id = str(row.get("video_id") or row.get("videoId") or "").strip()
                    identity = video_id or url
                    if not identity or identity in seen:
                        continue
                    seen.add(identity)
                    rows.append(
                        {
                            "rang": _number(row.get("rang")),
                            "genre": row.get("eingabe_genre") or row.get("genre") or "",
                            "titel": row.get("titel") or row.get("title") or "",
                            "kanal": row.get("kanal") or row.get("channel") or row.get("channel_title") or "",
                            "url": url,
                            "video_id": video_id,
                            "gesamt_score": _number(row.get("gesamt_score")),
                            "view_count": _number(row.get("view_count")),
                            "like_count": _number(row.get("like_count")),
                            "dauer": row.get("dauer") or row.get("duration") or "",
                            "status": row.get("status") or "top10",
                        }
                    )
                    if len(rows) >= limit:
                        return rows
        except Exception:
            continue
    return rows


def _number(value: Any) -> int | float | None:

    text = str(value or "").strip().replace(".", "").replace(",", ".")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def add_genre(body: dict[str, Any]) -> dict[str, Any]:

    label = genre_label(str(body.get("label") or body.get("name") or body.get("genre") or ""))
    key = slug(str(body.get("key") or label))
    if not key.endswith("_lofi"):
        key = f"{key}_lofi"
    caption = str(
        body.get("caption")
        or f"{label.lower()} instrumental, calm controlled lofi mood, soft drums, light low end, no vocals"
    )
    keywords = _list_value(body.get("keywords")) or [label.lower(), label.lower().replace(" lofi", "")]
    required = _list_value(body.get("required_terms") or body.get("required")) or keywords
    prefixes = _list_value(body.get("prefixes")) or [label.lower()]
    genres = load_genres()
    genres[key] = {
        "label": label,
        "caption": caption,
        "prefixes": prefixes,
        "keywords": keywords,
        "required_terms": required,
    }
    save_genres(genres)
    return {"ok": True, "genre": {"id": key, **genres[key]}}


def _list_value(value: Any) -> list[str]:

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


_VIDEO_CORRECTIONS: dict[str, dict[str, str | bool]] = {
    "keine_animation": {
        "prompt_add": "animated loop, visible motion, gently swaying leaves, twinkling stars, rippling water",
        "negative_add": "completely static, frozen image, no movement, still frame",
    },
    "zu_3d": {
        "prompt_add": "flat anime illustration, hand-drawn 2D anime art, cel shading",
        "negative_add": "3D render, CGI, photorealistic, volumetric lighting, realistic fur",
    },
    "zu_hell": {
        "prompt_add": "deep dark indigo night sky, dark nocturnal atmosphere, dimly lit",
        "negative_add": "bright daylight, overexposed, washed out",
    },
    "schlechte_spiegelung": {
        "prompt_add": "softly rippling lake water with shimmering reflections, mirror-like water surface",
        "negative_add": "flat water, no reflections",
    },
    "kein_see": {
        "prompt_add": "calm dark lake clearly visible with reflections",
        "negative_add": "no water, no lake",
    },
    "ohren_fehlen": {
        "prompt_add": "clearly visible long upright ears, prominent tall ears",
        "negative_add": "missing ears, no ears, hidden ears, floppy ears",
    },
    "dritter_hase": {
        "prompt_add": "exactly two figures, only two characters, a pair",
        "negative_add": "three figures, extra character, fourth figure",
    },
    "falsche_tierart": {
        "prompt_add": "clearly a rabbit with round body and long upright ears, rabbit character",
        "negative_add": "bird, chick, beak, duck, cat, dog, wrong animal",
    },
    "gesichter_fehlen": {
        "prompt_add": "clearly visible faces, round visible eyes, facing forward",
        "negative_add": "no faces, empty faces, turned away",
    },
    "laterne_fehlt": {
        "prompt_add": "warm glowing amber lantern as light source",
        "negative_add": "no lantern, no light source",
    },
    "kein_baum": {
        "prompt_add": "large background tree with gently swaying leaves",
        "negative_add": "",
    },
    "mund_offen": {
        "prompt_add": "calm closed peaceful expression",
        "negative_add": "open mouth, surprised, anxious expression",
    },

    "ueberbelichtung": {
        "prompt_add": "natural exposure, soft ambient lighting, gentle warm tones, shadow detail preserved",
        "negative_add": "overexposed, blown out highlights, white sky, washed out scenery, harsh bright light",
        "guidance_delta": -1.0,
    },
    "loop_springt": {
        "prompt_add": "seamless animated loop, smooth continuous looping motion, no hard cuts",
        "negative_add": "abrupt jump, loop artifact, discontinuous motion, frame skip",
        "step_delta": 10,
    },
    "hintergrund_unruhig": {
        "prompt_add": "stable static background, calm environment, no camera movement, fixed perspective",
        "negative_add": "chaotic background, unstable background, flickering environment, camera shake",
        "guidance_delta": 1.0,
    },
    "pflanzen_statisch": {
        "prompt_add": "gently swaying grass and leaves, visible plant movement in breeze, softly rustling foliage",
        "negative_add": "completely static plants, frozen leaves, no plant movement",
    },
}


def _lofi_python() -> str:
    cfg_path = LOFI_CONFIGS_DIR / "model_paths.yaml"
    try:
        import yaml
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        return str(cfg.get("python") or PYTHON_CMD)
    except Exception:
        return PYTHON_CMD


def _build_video_scenario_yaml(body: dict[str, Any]) -> dict[str, Any]:
    name = slug(str(body.get("name") or "neue_szene"))
    prompt = str(body.get("prompt") or "lofi_girl, animated background loop").strip()
    negative_prompt = str(body.get("negativePrompt") or (
        "completely static, frozen image, no movement, still frame, "
        "3D render, CGI, photorealistic, text, watermark, blurry, low quality"
    )).strip()
    guidance_scale = float(body.get("guidanceScale") or 9.0)
    inference_steps = int(body.get("inferenceSteps") or 60)
    lora_config = str(body.get("loraConfig") or "base_lora.yaml")

    return {
        "scenario_id": name,
        "description": str(body.get("name") or name),
        "precomputed_dir": "",
        "resolution": [832, 480],
        "frames": 97,
        "fps": 8,
        "seed": 777,
        "guidance_scale": guidance_scale,
        "generate_inference_steps": inference_steps,
        "trigger_token": "lofi_girl",
        "lora_config": lora_config,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "metric_targets": {
            "overall": 0.80,
            "motion": 0.40,
            "sharpness": 0.20,
            "flicker": 0.06,
        },
        "_meta": {
            "visual_style": str(body.get("visualStyle") or "lofi_anime"),
            "main_subject": str(body.get("mainSubject") or ""),
        },
    }


def video_create_scenario(body: dict[str, Any]) -> dict[str, Any]:
    import yaml

    name = slug(str(body.get("name") or "neue_szene"))
    scenario_dir = LOFI_SCENARIOS_DIR / name
    if scenario_dir.exists():

        name = f"{name}_{datetime.now().strftime('%m%d_%H%M')}"
        scenario_dir = LOFI_SCENARIOS_DIR / name

    scenario_dir.mkdir(parents=True, exist_ok=True)
    cfg = _build_video_scenario_yaml({**body, "name": name})

    with open(scenario_dir / "scenario.yaml", "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, allow_unicode=True, default_flow_style=False, width=120)

    return {"ok": True, "scenario_id": name, "path": rel(scenario_dir)}


def video_youtube_search(body: dict[str, Any]) -> dict[str, Any]:
    query = str(body.get("query") or "").strip()

    want = max(1, min(int(body.get("count") or 8), 15))
    fetch = min(want * 2, 20)
    if not query:
        return {"videos": []}

    cmd = [
        "yt-dlp",
        f"ytsearch{fetch}:{query}",
        "--dump-json",
        "--no-download",
        "--flat-playlist",
        "--no-warnings",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        videos = []
        for line in result.stdout.splitlines():
            if len(videos) >= want:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid_id = str(entry.get("id") or entry.get("webpage_url_basename") or "")
            dur_raw = entry.get("duration")
            if isinstance(dur_raw, (int, float)) and dur_raw > 0:
                minutes, seconds = divmod(int(dur_raw), 60)
                dur_str = f"{minutes}:{seconds:02d}"
            else:
                dur_str = "?"


            thumb_url = f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg" if vid_id else ""
            view_count = entry.get("view_count") or 0
            videos.append({
                "id": vid_id,
                "title": str(entry.get("title") or ""),
                "url": str(entry.get("webpage_url") or entry.get("url") or f"https://www.youtube.com/watch?v={vid_id}"),
                "channel": str(entry.get("channel") or entry.get("uploader") or ""),
                "duration": dur_str,
                "thumbnail": thumb_url or "",
                "views": int(view_count),
            })
        return {"videos": videos}
    except subprocess.TimeoutExpired:
        return {"videos": [], "error": "Suche Timeout (30s)"}
    except Exception as exc:
        return {"videos": [], "error": str(exc)}


def video_save_project(body: dict[str, Any]) -> dict[str, Any]:
    VIDEO_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    project_id = str(body.get("id") or "").strip() or uuid.uuid4().hex[:10]
    data = {
        "id": project_id,
        "name": str(body.get("name") or project_id),
        "created": str(body.get("created") or datetime.now().isoformat()),
        "updated": datetime.now().isoformat(),
        "phase": str(body.get("phase") or "setup"),
        "setup": body.get("setup") or {},
        "sources": body.get("sources") or [],
        "scenarioId": body.get("scenarioId"),
    }
    (VIDEO_PROJECTS_DIR / f"{project_id}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
    )
    return {"ok": True, "id": project_id}


def video_list_projects() -> dict[str, Any]:
    VIDEO_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    projects = []
    for f in sorted(VIDEO_PROJECTS_DIR.glob("*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            projects.append(json.loads(f.read_text()))
        except Exception:
            pass
    return {"projects": projects}


def video_delete_project(body: dict[str, Any]) -> dict[str, Any]:
    project_id = str(body.get("id") or "").strip()
    path = VIDEO_PROJECTS_DIR / f"{project_id}.json"
    if path.exists():
        path.unlink()
    return {"ok": True}


def video_resolve_url(body: dict[str, Any]) -> dict[str, Any]:
    url = str(body.get("url") or "").strip()
    if not url:
        raise ValueError("URL fehlt")
    result = subprocess.run([
        "yt-dlp", url,
        "--dump-json", "--no-playlist",
        "--no-warnings", "--quiet",
    ], capture_output=True, text=True, timeout=30)
    if result.returncode != 0 or not result.stdout.strip():

        from urllib.parse import urlparse, parse_qs as _pqs
        _parsed = urlparse(url)
        vid_id = None
        if "youtube.com" in _parsed.netloc:
            vid_id = _pqs(_parsed.query).get("v", [None])[0]
        elif "youtu.be" in _parsed.netloc:
            vid_id = _parsed.path.lstrip("/").split("?")[0]
        if not vid_id:
            raise ValueError("Video konnte nicht geladen werden")
        return {
            "video": {
                "id": vid_id,
                "title": "YouTube Video",
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "channel": "",
                "duration": "?",
                "thumbnail": f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg",
                "views": 0,
            }
        }
    info = json.loads(result.stdout.strip().splitlines()[0])
    vid_id = str(info.get("id") or "")
    duration_sec = int(info.get("duration") or 0)
    h, m, s = duration_sec // 3600, (duration_sec % 3600) // 60, duration_sec % 60
    dur_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    view_count = int(info.get("view_count") or 0)
    return {
        "video": {
            "id": vid_id,
            "title": str(info.get("title") or ""),
            "url": f"https://www.youtube.com/watch?v={vid_id}",
            "channel": str(info.get("uploader") or info.get("channel") or ""),
            "duration": dur_str,
            "thumbnail": f"https://i.ytimg.com/vi/{vid_id}/mqdefault.jpg",
            "views": view_count,
        }
    }


def video_make_train_job(body: dict[str, Any]) -> dict[str, Any]:
    import yaml as _yaml

    scenario_id = str(body.get("scenario_id") or "").strip()
    if not scenario_id:
        raise ValueError("scenario_id fehlt")
    scenario_dir = LOFI_SCENARIOS_DIR / scenario_id
    if not scenario_dir.exists():
        raise ValueError(f"Szenario nicht gefunden: {scenario_id}")

    steps = int(body.get("steps") or 200)

    selected_videos: list[dict] = list(body.get("selected_videos") or [])


    force_preprocess: bool = bool(body.get("force_preprocess") or False)

    job_id = f"video_train_{scenario_id}_{int(datetime.now().timestamp())}"
    JOBS[job_id] = {
        "jobId": job_id,
        "scenarioId": scenario_id,
        "kind": f"video_train_{scenario_id}",
        "startedAt": datetime.now().isoformat(),
        "done": False, "output": "", "returncode": None,
        "progress": 0,
        "phase": "",
        "trainStep": 0,
        "trainTotal": steps,
        "round": _next_video_round(scenario_id),
    }

    def _set(progress: int, phase: str, output: str = "") -> None:
        JOBS[job_id]["progress"] = progress
        JOBS[job_id]["phase"] = phase
        if output:
            JOBS[job_id]["output"] = output

    def _run() -> None:
        import re as _re

        assets_dir = scenario_dir / "assets"
        clips_dir = assets_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        with open(scenario_dir / "scenario.yaml") as fh:
            scenario_cfg = _yaml.safe_load(fh) or {}
        prompt = str(scenario_cfg.get("prompt") or "lofi_girl, animated background loop")


        total_vids = max(len(selected_videos), 1)
        dataset_entries: list[dict] = []

        for vi, video in enumerate(selected_videos):
            vid_id = str(video.get("id") or "")
            url = str(video.get("url") or f"https://www.youtube.com/watch?v={vid_id}")
            class_label = str(video.get("class_label") or "clip").replace("/", "_")[:20]
            base_pct = int(15 * vi / total_vids)
            _set(base_pct, f"1/4 · Download {vi+1}/{total_vids}", f"Lade {vid_id}…")

            src_path = assets_dir / f"src_{vid_id}.mp4"
            if not src_path.exists():

                subprocess.run([
                    "yt-dlp", url, "-o", str(src_path),
                    "--format", "best[height<=480][ext=mp4]/best[height<=480]/best",
                    "--download-sections", "*120-360",
                    "--no-warnings", "--quiet",
                ], capture_output=True, timeout=300)

            if not src_path.exists():
                continue

            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(src_path),
            ], capture_output=True, text=True, timeout=30)
            try:
                total_sec = float(probe.stdout.strip())
            except Exception:
                total_sec = 120.0

            CLIP_SEC, INTERVAL, MAX_CLIPS = 13, 30, 8
            clip_idx, start = 0, 10
            _set(base_pct, f"1/4 · Clips schneiden {vi+1}/{total_vids}", f"Schneide {vid_id}…")
            while start + CLIP_SEC < total_sec and clip_idx < MAX_CLIPS:
                clip_name = f"{class_label}_{vid_id[:8]}_{clip_idx:02d}"
                clip_path = clips_dir / f"{clip_name}.mp4"
                cut = subprocess.run([
                    "ffmpeg", "-y",
                    "-ss", str(start), "-t", str(CLIP_SEC),
                    "-i", str(src_path),
                    "-vf", "scale=832:480:force_original_aspect_ratio=increase,crop=832:480",
                    "-an", "-c:v", "libx264", "-crf", "18",
                    str(clip_path),
                ], capture_output=True, timeout=120)
                if cut.returncode == 0 and clip_path.exists():
                    dataset_entries.append({
                        "media_path": f"assets/clips/{clip_name}.mp4",
                        "caption": prompt,
                    })
                    clip_idx += 1
                start += INTERVAL


        if not dataset_entries and not selected_videos:
            for clip_file in sorted(clips_dir.glob("*.mp4")):
                dataset_entries.append({
                    "media_path": f"assets/clips/{clip_file.name}",
                    "caption": prompt,
                })

        if not dataset_entries:
            _set(0, "Fehler", "FEHLER: Keine Clips extrahiert — Videos prüfen")
            JOBS[job_id]["done"] = True
            JOBS[job_id]["returncode"] = 1
            return


        _set(20, f"2/4 · Dataset ({len(dataset_entries)} Clips)", "Schreibe dataset.jsonl…")
        dataset_path = scenario_dir / "dataset.jsonl"
        with open(dataset_path, "w") as fh:
            for entry in dataset_entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        scenario_cfg["dataset_jsonl"] = str(dataset_path)
        scenario_cfg["precomputed_dir"] = str(scenario_dir / "precomputed")
        with open(scenario_dir / "scenario.yaml", "w") as fh:
            _yaml.dump(scenario_cfg, fh, default_flow_style=False, allow_unicode=True)


        precomputed_dir = scenario_dir / "precomputed"
        latents_ok = (precomputed_dir / "latents").exists() and any((precomputed_dir / "latents").iterdir())
        conditions_ok = (precomputed_dir / "conditions").exists() and any((precomputed_dir / "conditions").iterdir())
        skip_preprocess = not selected_videos and latents_ok and conditions_ok and not force_preprocess

        python = _lofi_python()
        if skip_preprocess:
            _set(50, "3/4 · Preprocess übersprungen (Clips unverändert)", "Verwende vorhandene Latents…")
        else:

            text_only = force_preprocess and not selected_videos and latents_ok
            preprocess_cmd = [python, str(LOFI_PIPELINE / "scripts" / "preprocess_scenario.py"),
                              "--scenario", scenario_id]
            if text_only:
                preprocess_cmd.append("--text-only")
                _set(25, "3/4 · Preprocess (nur T5-Text)", "Berechne neue Text-Embeddings…")
            else:
                _set(25, "3/4 · Preprocess (VAE + T5)", "Starte Preprocess…")
            pre_proc = subprocess.Popen(
                preprocess_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            pre_lines = 0
            pre_buf: list[str] = []
            for line in (pre_proc.stdout or []):
                pre_lines += 1
                pre_buf.append(line.rstrip())
                if len(pre_buf) > 5:
                    pre_buf.pop(0)

                pct = min(50, 25 + int(25 * pre_lines / 200))
                JOBS[job_id]["progress"] = pct
                JOBS[job_id]["output"] = pre_buf[-1]
            pre_proc.wait()
            if pre_proc.returncode != 0:
                _set(JOBS[job_id]["progress"], "Fehler", "\n".join(pre_buf[-10:]))
                JOBS[job_id]["done"] = True
                JOBS[job_id]["returncode"] = 1
                return


        _set(50, f"4/4 · LoRA-Training (0/{steps})", "Training startet…")
        round_n = _next_video_round(scenario_id)
        lora_config = str(scenario_cfg.get("lora_config") or "base_lora.yaml")


        ref_ckpt_rel = scenario_cfg.get("reference_checkpoint")
        ref_ckpt_arg: list[str]
        if ref_ckpt_rel:
            ref_ckpt_abs = str(LOFI_PIPELINE / "scenarios" / scenario_id / ref_ckpt_rel)
            ref_ckpt_arg = ["--from-checkpoint", ref_ckpt_abs]
        else:
            ref_ckpt_arg = ["--fresh"]


        _step_re = _re.compile(
            r'"step"\s*:\s*(\d+)'
            r'|[Ss]tep[s]?\s*[:/]\s*(\d+)'
            r'|\|\s*(\d+)/' + str(steps)
        )

        train_proc = subprocess.Popen(
            [python, str(LOFI_TRAIN_SCRIPT),
             "--scenario", scenario_id,
             "--round", str(round_n),
             "--steps", str(steps),
             "--config", lora_config,
             *ref_ckpt_arg,
             "--no-generate"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        train_buf: list[str] = []
        for line in (train_proc.stdout or []):
            train_buf.append(line.rstrip())
            if len(train_buf) > 20:
                train_buf.pop(0)
            m = _step_re.search(line)
            if m:
                cur = int(next(g for g in m.groups() if g is not None))
                cur = min(cur, steps)
                JOBS[job_id]["trainStep"] = cur
                pct = 50 + int(49 * cur / steps)
                JOBS[job_id]["progress"] = pct
                JOBS[job_id]["phase"] = f"4/4 · LoRA-Training ({cur}/{steps})"
            JOBS[job_id]["output"] = train_buf[-1]

        train_proc.wait()
        if train_proc.returncode == 0:

            _set(99, "Generiere Sample…", f"Step {steps} → Video…")
            subprocess.run(
                [python, str(LOFI_GENERATE_SCRIPT.parent.parent / "lofi_pipeline" / "scripts" / "generate_samples.py"),
                 "--scenario", scenario_id,
                 "--round", str(round_n),
                 "--checkpoint", str(steps)],
                timeout=600, capture_output=True,
            )
            JOBS[job_id]["done"] = True
            JOBS[job_id]["returncode"] = 0
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["phase"] = "Fertig"
            JOBS[job_id]["output"] = "\n".join(train_buf[-5:])
        else:
            JOBS[job_id]["done"] = True
            JOBS[job_id]["returncode"] = train_proc.returncode
            JOBS[job_id]["phase"] = "Fehler"
            JOBS[job_id]["output"] = "\n".join(train_buf[-10:])

    import threading as _threading
    _threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "job_id": job_id, "round": _next_video_round(scenario_id)}


def video_make_generate_job(body: dict[str, Any]) -> dict[str, Any]:

    scenario_id = str(body.get("scenario_id") or "").strip()
    if not scenario_id:
        raise ValueError("scenario_id fehlt")

    scenario_dir = LOFI_SCENARIOS_DIR / scenario_id


    ckpt = _find_latest_video_checkpoint(scenario_id)
    if not ckpt:
        raise ValueError(f"Kein Checkpoint fuer Szenario {scenario_id} gefunden")


    round_n = 1
    for part in Path(ckpt).parts:
        if part.startswith("round_"):
            try:
                round_n = int(part.split("_")[1])
            except (IndexError, ValueError):
                pass

    job_id = f"video_gen_{scenario_id}_{int(datetime.now().timestamp())}"
    JOBS[job_id] = {
        "jobId": job_id,
        "scenarioId": scenario_id,
        "kind": f"video_gen_{scenario_id}",
        "startedAt": datetime.now().isoformat(),
        "done": False, "output": "", "returncode": None,
        "progress": 0, "phase": "Generierung startet…",
        "videoUrl": None,
        "round": round_n,
    }

    python = _lofi_python()
    generate_samples_script = LOFI_PIPELINE / "scripts" / "generate_samples.py"

    def _run() -> None:
        cmd = [python, str(generate_samples_script),
               "--scenario", scenario_id,
               "--round", str(round_n),
               "--gpu-fraction", "0.8"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        buf: list[str] = []
        for line in (proc.stdout or []):
            buf.append(line.rstrip())
            if len(buf) > 10:
                buf.pop(0)
            last = buf[-1]
            JOBS[job_id]["output"] = last

            import re as _re
            m = _re.search(r'(\d+)\s*/\s*(\d+)', last)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                if tot > 0:
                    JOBS[job_id]["progress"] = min(99, int(100 * cur / tot))
                    JOBS[job_id]["phase"] = f"Step {cur}/{tot}"
        proc.wait()
        JOBS[job_id]["done"] = True
        JOBS[job_id]["returncode"] = proc.returncode
        if proc.returncode == 0:
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["phase"] = "Fertig"

            samples_dir = scenario_dir / "rounds" / f"round_{round_n:02d}" / "samples"
            mp4s = sorted(samples_dir.glob("*.mp4")) if samples_dir.exists() else []
            if mp4s:
                newest = mp4s[-1]
                JOBS[job_id]["videoUrl"] = (
                    f"/api/video/media/{scenario_id}/rounds/round_{round_n:02d}/samples/{newest.name}"
                )
        else:
            JOBS[job_id]["phase"] = "Fehler"
            JOBS[job_id]["output"] = "\n".join(buf[-10:])

    import threading as _t
    _t.Thread(target=_run, daemon=True).start()
    return {"jobId": job_id, "videoUrl": None, "round": round_n}


def video_make_gif(body: dict[str, Any]) -> dict[str, Any]:
    import re as _re

    scenario_id = str(body.get("scenario_id") or "")
    if not scenario_id:
        raise ValueError("scenario_id erforderlich")


    video_path = str(body.get("video_path") or "").strip()
    if video_path:
        mp4_path = LOFI_SCENARIOS_DIR / scenario_id / video_path
    else:
        round_name = str(body.get("round") or "web_outputs")
        filename = str(body.get("filename") or "")
        if not filename:
            raise ValueError("video_path oder filename erforderlich")

        if _re.match(r"^round_\d+$", round_name):
            mp4_path = LOFI_SCENARIOS_DIR / scenario_id / "rounds" / round_name / "samples" / filename
        else:
            mp4_path = LOFI_SCENARIOS_DIR / scenario_id / round_name / filename

    if not mp4_path.exists():
        raise ValueError(f"MP4 nicht gefunden: {mp4_path}")

    rel = mp4_path.relative_to(LOFI_SCENARIOS_DIR / scenario_id)
    gif_path = mp4_path.with_suffix(".gif")
    cmd = [
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-vf",
        "fps=8,scale=624:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=256:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
        "-loop", "0", str(gif_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    rel_gif = str(rel.parent / gif_path.name)
    return {
        "ok": result.returncode == 0,
        "gif_url": f"/api/video/media/{scenario_id}/{rel_gif}" if result.returncode == 0 else None,
    }


_clip_cache: dict = {}


def _compute_clip_score(mp4_path: "Path", prompt: str) -> "float | None":
    try:
        import cv2 as _cv2
        import torch as _torch
        from PIL import Image as _PIL_Image
        from transformers import CLIPModel, CLIPProcessor

        if "model" not in _clip_cache:
            _clip_cache["model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_cache["processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_cache["model"].eval()

        model = _clip_cache["model"]
        processor = _clip_cache["processor"]

        cap = _cv2.VideoCapture(str(mp4_path))
        total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = [int(total * i / 8) for i in range(8)]
        frames_pil = []
        for idx in indices:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames_pil.append(_PIL_Image.fromarray(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)))
        cap.release()

        if not frames_pil:
            return None

        inputs = processor(
            text=[prompt[:77]],
            images=frames_pil,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with _torch.no_grad():
            out = model(**inputs)
            img_emb = out.image_embeds
            txt_emb = out.text_embeds
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            score = float((img_emb @ txt_emb.T).mean())

        return round(score, 4)
    except Exception:
        return None


def _get_last_clip_score(scenario_id: str) -> "float | None":
    import json as _json
    scenario_dir = LOFI_SCENARIOS_DIR / scenario_id
    clip_files = sorted(scenario_dir.rglob("*.clip.json"))
    if not clip_files:
        return None
    try:
        return float(_json.loads(clip_files[-1].read_text(encoding="utf-8")).get("clip_score") or 0)
    except Exception:
        return None


def video_feedback_history(params: dict) -> dict:
    import json as _json
    raw = params.get("scenario_id") or ""

    scenario_id = str((raw[0] if isinstance(raw, list) else raw) or "").strip()
    if not scenario_id:
        return {"ok": True, "history": []}
    history_path = LOFI_SCENARIOS_DIR / scenario_id / "feedback_history.jsonl"
    if not history_path.exists():
        return {"ok": True, "history": []}
    entries = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(_json.loads(line))
            except Exception:
                pass
    return {"ok": True, "history": entries}


def _get_last_objective_score(scenario_id: str) -> "float | None":
    import json as _json
    scenario_dir = LOFI_SCENARIOS_DIR / scenario_id
    metrics_files = sorted(scenario_dir.rglob("*.metrics.json"))
    if not metrics_files:
        return None
    try:
        data = _json.loads(metrics_files[-1].read_text(encoding="utf-8"))
        v = data.get("overall_score")
        return float(v) if v is not None else None
    except Exception:
        return None


def _append_feedback_history(scenario_id: str, entry: dict) -> None:
    import json as _json
    history_path = LOFI_SCENARIOS_DIR / scenario_id / "feedback_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as fh:
        fh.write(_json.dumps(entry, ensure_ascii=False) + "\n")


def video_quantified_feedback(body: dict[str, Any]) -> dict[str, Any]:
    import yaml

    scenario_id = str(body.get("scenario_id") or "")
    feedback = body.get("feedback") or {}
    dimensions = feedback.get("dimensions") or {}
    issues = list(feedback.get("issues") or [])

    _ = str(feedback.get("freeText") or "").strip()

    scenario_dir = LOFI_SCENARIOS_DIR / scenario_id
    scenario_path = scenario_dir / "scenario.yaml"
    if not scenario_path.exists():
        raise ValueError(f"scenario.yaml fuer {scenario_id} nicht gefunden")

    with open(scenario_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    prompt = str(cfg.get("prompt") or "").strip().rstrip(",")
    negative = str(cfg.get("negative_prompt") or "").strip().rstrip(",")
    adjustments: list[str] = []
    needs_training = False


    dim_map = {
        "stil_treue": {
            "low_prompt": "flat anime illustration, hand-drawn 2D anime art, cel shading",
            "low_neg": "3D render, CGI, photorealistic, volumetric lighting",
            "trains": True,
        },
        "bewegung": {
            "low_prompt": "animated loop, visible continuous motion, gently swaying, rippling",
            "low_neg": "completely static, frozen image, no movement",
            "trains": True,
            "metric_key": "motion",
            "metric_boost": 0.15,
        },
        "schaerfe": {
            "low_prompt": "sharp details, crisp lines, high detail quality",
            "low_neg": "blurry, soft focus, out of focus",
            "trains": False,
            "step_boost": 10,
        },
        "stimmung": {
            "low_prompt": "perfect atmospheric mood, ambient glow, expressive atmosphere",
            "low_neg": "wrong atmosphere, wrong mood",
            "trains": False,
        },
        "motiv_genauigkeit": {
            "low_prompt": "",
            "low_neg": "",
            "trains": False,
            "guidance_boost": 1.0,
        },
    }

    guidance = float(cfg.get("guidance_scale") or 9.0)
    infer_steps = int(cfg.get("generate_inference_steps") or 60)
    metric_targets = dict(cfg.get("metric_targets") or {})

    for dim_key, dim_cfg in dim_map.items():
        score = float(dimensions.get(dim_key) or 3)
        if score < 3:
            if dim_cfg.get("low_prompt") and dim_cfg["low_prompt"] not in prompt:
                prompt = str(dim_cfg["low_prompt"]) + ",\n  " + prompt
                adjustments.append(f"{dim_key}: Prompt erweitert")
            if dim_cfg.get("low_neg") and dim_cfg["low_neg"] not in negative:
                negative = str(dim_cfg["low_neg"]) + ",\n  " + negative
            if dim_cfg.get("trains"):
                needs_training = True
            if dim_cfg.get("metric_key"):
                mk = str(dim_cfg["metric_key"])
                metric_targets[mk] = round(min(0.90, float(metric_targets.get(mk) or 0.4) + float(dim_cfg["metric_boost"])), 3)
                adjustments.append(f"motion target → {metric_targets[mk]}")
            if dim_cfg.get("step_boost"):
                new_steps = min(80, infer_steps + int(dim_cfg["step_boost"]))
                if new_steps != infer_steps:
                    infer_steps = new_steps
                    adjustments.append(f"inference steps → {infer_steps}")
                else:
                    infer_steps = new_steps
            if dim_cfg.get("guidance_boost"):
                new_guidance = min(12.0, guidance + float(dim_cfg["guidance_boost"]))
                if new_guidance != guidance:
                    guidance = new_guidance
                    adjustments.append(f"guidance_scale → {guidance}")
                else:
                    guidance = new_guidance


    for issue_id in issues:
        corr = _VIDEO_CORRECTIONS.get(issue_id)
        if not corr:
            continue
        if corr.get("prompt_add") and str(corr["prompt_add"]) not in prompt:
            prompt = str(corr["prompt_add"]) + ",\n  " + prompt
        if corr.get("negative_add") and str(corr["negative_add"]) not in negative:
            negative = str(corr["negative_add"]) + ",\n  " + negative
        if corr.get("guidance_delta"):
            guidance = round(min(12.0, max(5.0, guidance + float(corr["guidance_delta"]))), 1)
            adjustments.append(f"guidance_scale → {guidance} ({issue_id})")
        if corr.get("step_delta"):
            infer_steps = min(80, max(20, infer_steps + int(corr["step_delta"])))
            adjustments.append(f"inference steps → {infer_steps} ({issue_id})")
        else:
            adjustments.append(issue_id)

    cfg["prompt"] = prompt
    cfg["negative_prompt"] = negative
    cfg["guidance_scale"] = guidance
    cfg["generate_inference_steps"] = infer_steps
    cfg["metric_targets"] = metric_targets


    dim_keys = ["stil_treue", "bewegung", "schaerfe", "stimmung", "motiv_genauigkeit"]
    dim_vals = [float(dimensions.get(k) or 3) for k in dim_keys]
    bars_score = (sum(dim_vals) / len(dim_vals) - 1) / 4

    overall_star_raw = float(feedback.get("overall") or 0)
    has_star = overall_star_raw > 0
    overall_star_norm = (overall_star_raw - 1) / 4 if has_star else 0.0

    objective_score = _get_last_objective_score(scenario_id)
    clip_score = _get_last_clip_score(scenario_id)

    w_obj, w_clip, w_bars, w_star = 0.30, 0.20, 0.30, 0.05
    avail_w = (
        (w_obj if objective_score is not None else 0.0)
        + (w_clip if clip_score is not None else 0.0)
        + w_bars
        + (w_star if has_star else 0.0)
    )
    lofi_qi = round(
        (
            (w_obj * objective_score if objective_score is not None else 0.0)
            + (w_clip * clip_score if clip_score is not None else 0.0)
            + w_bars * bars_score
            + (w_star * overall_star_norm if has_star else 0.0)
        ) / avail_w,
        4,
    )

    _append_feedback_history(scenario_id, {
        "timestamp": datetime.now().isoformat(),
        "bars": dict(zip(dim_keys, dim_vals)),
        "bars_score": round(bars_score, 4),
        "overall_star": int(overall_star_raw),
        "objective_score": objective_score,
        "clip_score": clip_score,
        "lofi_qi": lofi_qi,
        "adjustments_applied": adjustments,
    })


    backup = scenario_path.with_suffix(".yaml.bak")
    backup.write_text(scenario_path.read_text(encoding="utf-8"), encoding="utf-8")
    with open(scenario_path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, allow_unicode=True, default_flow_style=False, width=120)


    ckpt = _find_latest_video_checkpoint(scenario_id)
    if ckpt:
        job = video_make_generate_job({"scenario_id": scenario_id, "steps": infer_steps})
        return {
            "ok": True,
            "adjustments": adjustments,
            "needs_training": needs_training,
            "job_id": job["jobId"],
            "lofi_qi": lofi_qi,
        }

    return {
        "ok": True,
        "adjustments": adjustments,
        "needs_training": needs_training,
        "job_id": None,
        "lofi_qi": lofi_qi,
    }


def video_evaluate(body: dict[str, Any]) -> dict[str, Any]:
    import json as _json
    import yaml as _yaml

    scenario_id = str(body.get("scenario_id") or "").strip()
    video_rel = str(body.get("video_path") or "").strip()
    if not scenario_id or not video_rel:
        raise ValueError("scenario_id und video_path erforderlich")

    mp4 = LOFI_SCENARIOS_DIR / scenario_id / video_rel
    if not mp4.exists():
        raise ValueError(f"Video nicht gefunden: {mp4}")

    metrics_json = mp4.with_suffix(".metrics.json")
    clip_json = mp4.with_suffix(".clip.json")


    cached = False
    if metrics_json.exists() and metrics_json.stat().st_mtime >= mp4.stat().st_mtime:
        metrics = _json.loads(metrics_json.read_text(encoding="utf-8"))
        cached = True
    else:
        if not LOFI_EVALUATE_SCRIPT.exists():
            return {"ok": False, "error": f"evaluate_video.py nicht gefunden: {LOFI_EVALUATE_SCRIPT}"}
        subprocess.run(
            [_lofi_python(), str(LOFI_EVALUATE_SCRIPT), str(mp4), "--save-report"],
            capture_output=True, text=True, timeout=180,
            cwd=str(LOFI_EVALUATE_SCRIPT.parent),
        )
        if not metrics_json.exists():
            return {"ok": False, "error": "evaluate_video.py hat keine Ausgabe erzeugt"}
        metrics = _json.loads(metrics_json.read_text(encoding="utf-8"))


    clip_score: "float | None" = None
    if clip_json.exists() and clip_json.stat().st_mtime >= mp4.stat().st_mtime:
        try:
            clip_score = float(_json.loads(clip_json.read_text(encoding="utf-8")).get("clip_score") or 0)
        except Exception:
            pass
    else:

        scenario_yaml = LOFI_SCENARIOS_DIR / scenario_id / "scenario.yaml"
        prompt = ""
        if scenario_yaml.exists():
            try:
                prompt = str((_yaml.safe_load(scenario_yaml.read_text(encoding="utf-8")) or {}).get("prompt") or "")
            except Exception:
                pass
        if prompt:
            clip_score = _compute_clip_score(mp4, prompt)
            if clip_score is not None:
                clip_json.write_text(
                    _json.dumps({"clip_score": clip_score, "timestamp": datetime.now().isoformat()}, ensure_ascii=False),
                    encoding="utf-8",
                )

    def _extract(d, *keys):
        v = d
        for k in keys:
            if not isinstance(v, dict):
                return None
            v = v.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    flat: dict = {
        "temporal_ssim":        _extract(metrics, "temporal_ssim", "mean"),
        "sharpness":            _extract(metrics, "sharpness", "normalized"),
        "flicker":              _extract(metrics, "flicker", "mean"),
        "motion":               _extract(metrics, "motion", "mean"),
        "color_consistency":    _extract(metrics, "color_consistency", "consistency"),
        "brightness_consistency": _extract(metrics, "brightness_consistency", "consistency"),
        "overall_score":        _extract(metrics, "overall_score"),
    }
    flat = {k: v for k, v in flat.items() if v is not None}

    result: dict = {"ok": True, "cached": cached, "metrics": flat}
    if clip_score is not None:
        result["clip_score"] = clip_score
    return result


def _next_video_round(scenario_id: str) -> int:
    rounds_dir = LOFI_SCENARIOS_DIR / scenario_id / "rounds"
    if not rounds_dir.exists():
        return 1
    nums = [int(d.name.split("_")[1]) for d in rounds_dir.iterdir()
            if d.is_dir() and d.name.startswith("round_")]
    return (max(nums) + 1) if nums else 1


def _find_latest_video_checkpoint(scenario_id: str) -> Path | None:
    scenario_dir = LOFI_SCENARIOS_DIR / scenario_id
    all_ckpts = sorted(scenario_dir.rglob("lora_weights_step_*.safetensors"))
    return all_ckpts[-1] if all_ckpts else None


def video_scenarios_payload() -> list[dict[str, Any]]:
    import yaml

    rows: list[dict[str, Any]] = []
    if not LOFI_SCENARIOS_DIR.exists():
        return rows
    for d in sorted(LOFI_SCENARIOS_DIR.iterdir()):
        if not d.is_dir():
            continue
        cfg: dict[str, Any] = {}
        scenario_yaml = d / "scenario.yaml"
        if scenario_yaml.exists():
            try:
                with open(scenario_yaml) as fh:
                    cfg = yaml.safe_load(fh) or {}
            except Exception:
                pass
        rounds = sorted((d / "rounds").glob("round_*")) if (d / "rounds").exists() else []
        samples_count = sum(
            1 for rnd in rounds
            for _ in (rnd / "samples").glob("*.mp4") if (rnd / "samples").exists()
        )
        rows.append({
            "id": d.name,
            "description": str(cfg.get("description") or d.name),
            "rounds": len(rounds),
            "samples": samples_count,
            "hasCheckpoint": _find_latest_video_checkpoint(d.name) is not None,
        })
    return rows


def make_generate_job(body: dict[str, Any]) -> dict[str, Any]:

    duration_text = str(body.get("durationInput") or "").strip()
    duration = duration_text_to_minutes(duration_text, parse_duration_minutes(body.get("targetDurationMin")))
    duration_cli = duration_text or f"{duration}m"
    profile = str(body.get("promptProfile") or "chillhop_lofi")
    label = genre_label(profile)
    crossfade = str(body.get("crossfadeSec") or 3)
    min_block_seconds = max(90, int(float(body.get("segmentDurationSec") or 120)))
    custom_prompt = str(body.get("customPrompt") or "").strip()
    instruments = str(body.get("instruments") or "").strip()
    target_bpm = str(body.get("targetBpm") or 78)
    genre_key = profile_from_label(label)
    adapter_path, adapter_source = resolve_generation_adapter(genre_key)
    if adapter_path is None:
        raise ValueError(
            f"Kein freigegebener LoRA-Adapter fuer '{label}' vorhanden (weder ein eigenes "
            "Genre-Training noch das geteilte Basis-Modell). Erst Training abschliessen und "
            "einen Checkpoint freigeben."
        )
    command = command_python(
        str(START),
        "--stufen",
        "audio_generieren",
        "--dauer",
        duration_cli,
        "--genre",
        label,
        "--abschnitt-sekunden",
        "30",
        "--kandidaten-pro-abschnitt",
        "5",
        "--block-sekunden",
        "0",
        "--rhythmus-anteil-prozent",
        "5",
        "--min-rhythmus-sekunden",
        str(min_block_seconds),
        "--block-looping-aktiv",
        "--kontinuierliche-bloecke-deaktivieren",
        "--crossfade-sekunden",
        crossfade,
        "--ziel-bpm",
        target_bpm,
        "--bpm-toleranz",
        "4",
        "--min-sekunden-rms-db",
        "-50",
        "--vram-limit-fraction",
        "0.80",
        "--seed",
        str(body.get("seed") or 0),
        "--best-adapter",
        str(adapter_path),
        "--freigabe-langer-lauf",
        "--genre-pruefung-aktiv",
        "--genre-melodie-conditioning-deaktivieren",
        "--anschluss-conditioning-deaktivieren",
    )
    if body.get("githubPush", True):
        command.append("--github-push")
    else:


        command.append("--kein-github-push")
    if custom_prompt:
        command.extend(["--stimmung", custom_prompt])
    if instruments:
        command.extend(["--instrumente", instruments])
    job = start_command(
        kind="audio",
        commands=[command],
        prompt_profile=genre_key,
        duration_min=duration,
        output_format=str(body.get("outputFormat") or "wav_mp3"),
        desired_name=str(body.get("name") or "").strip(),
    )
    job["adapterSource"] = adapter_source
    return job


def make_project_job(action: str, body: dict[str, Any]) -> dict[str, Any]:

    commands: list[list[str]]
    profile = profile_from_label(str(body.get("genre") or "chillhop_lofi"))


    display_profile = profile if str(body.get("genre") or "").strip() else ""
    duration = 0
    if action == "top10":
        label = genre_label(str(body.get("genre") or "Jazz Lofi"))
        command = command_python(
            str(START),
            "--top10",
            "--genre",
            label,
            "--stimmung",
            str(body.get("stimmung") or "calm relaxed lofi mood"),
            "--lizenz",
            str(body.get("lizenz") or "no copyright"),
        )
        if body.get("bekannteErlauben"):
            command.append("--bekannte-erlauben")
        commands = [command]
    elif action == "clips":
        command = command_python(str(START), "--clips-5000", "--ziel-clips", str(body.get("zielClips") or 5000))
        genres = str(body.get("genres") or "").strip()
        if genres:
            command.extend(["--genres", genres])
        if body.get("mitDownloads"):
            command.append("--mit-downloads")
        if body.get("nurPlan"):
            command.append("--plan")
        commands = [command]
    elif action == "import_pending":
        genres_arg = str(body.get("genres") or "").strip()
        genre_keys = {g.strip() for g in genres_arg.split(",") if g.strip()} if genres_arg else None
        pending = _pending_top10_entries(genre_keys)
        if not pending:
            raise ValueError("Keine offenen Top10-Videos zum Importieren gefunden.")
        pending_genres = {row["genre"] for row in pending}
        profile = pending_genres.pop() if len(pending_genres) == 1 else "Alle Genres"
        massenimport_dir = ROOT / "daten" / "metadata" / "downloads"
        massenimport_dir.mkdir(parents=True, exist_ok=True)
        csv_path = massenimport_dir / f"massenimport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["url", "titel", "genre"])
            writer.writeheader()
            writer.writerows(pending)
        commands = [command_python(str(CRAWLER), "mp3", "--url-datei", str(csv_path))]
    elif action == "trainingsdaten":
        commands = [command_python(str(START), "--trainingsdaten")]
    elif action == "merkmale":
        commands = [command_python(str(START), "--merkmale")]
    elif action == "referenzvergleich":
        commands = [command_python(str(START), "--referenzvergleich")]
    elif action == "audio_bewertung":
        audio_id = str(body.get("audioId") or "").strip()
        if not audio_id:
            raise ValueError("audioId fehlt")
        run_dir = _audio_map().get(audio_id)
        if not run_dir:
            raise ValueError(f"Audio nicht gefunden: {audio_id}")
        audio_path = run_dir / "lange_audio.mp3"
        if not audio_path.exists():
            audio_path = run_dir / "lange_audio.wav"
        if not audio_path.exists():
            raise ValueError(f"Audiodatei fehlt in: {rel(run_dir)}")
        report = read_json(run_dir / "finale_audio_info.json", {})
        if not isinstance(report, dict) or not report:
            report = read_json(run_dir / "generation_report.json", {})
        report = report if isinstance(report, dict) else {}
        genre = str(report.get("genre") or report.get("prompt_profile") or run_dir.name.split("_")[0] or "lofi")
        commands = [
            command_python(
                str(AUDIO_BEWERTUNG),
                "audio",
                "--audio",
                str(audio_path),
                "--ausgabe-dir",
                str(run_dir),
                "--genre",
                genre,
            )
        ]
    elif action == "lora_training":
        archive_lora_run()
        command = command_python(str(START), "--lora-training")


        if body.get("loraRank"):
            command.extend(["--lora-rank", str(int(body["loraRank"]))])
        if body.get("loraAlpha"):
            command.extend(["--lora-alpha", str(int(body["loraAlpha"]))])
        if body.get("learningRate"):
            command.extend(["--learning-rate", str(body["learningRate"])])
        if body.get("maxSteps"):
            command.extend(["--max-steps", str(int(body["maxSteps"]))])
        commands = [command]
    elif action == "lora_training_weiter":


        command = command_python(str(START), "--lora-weitere-500")
        if body.get("learningRate"):
            command.extend(["--learning-rate", str(body["learningRate"])])
        if body.get("maxSteps"):
            command.extend(["--max-steps", str(int(body["maxSteps"]))])
        commands = [command]
    elif action == "lora_freigeben":
        checkpoint = str(body.get("checkpoint") or "").strip()
        if not checkpoint:
            raise ValueError("Checkpoint fehlt")
        commands = [command_python(str(START), "--lora-freigeben", "--checkpoint", checkpoint)]
    elif action == "lora_dataset_genre":
        ziel_clips = int(body.get("zielClips") or 1000)
        commands = [
            command_python(
                str(ZIELDATENSATZ_SCRIPT),
                "--quelle",
                str(IMPORT_30S_ROOT),
                "--ziel",
                str(genre_dataset_dir(profile)),
                "--genres",
                profile,
                "--ziel-clips",
                str(ziel_clips),
                "--overwrite",
            )
        ]
    elif action == "lora_dataset_reset":
        archived = archive_genre_dir(genre_dataset_dir(profile))
        return start_command(
            kind=action,
            commands=[command_python("-c", "pass")] if archived else [],
            prompt_profile=profile,
        ) | {"archived": archived}
    elif action == "lora_training_genre":
        archive_genre_dir(genre_run_dir(profile))
        genre_summary = read_json(genre_dataset_dir(profile) / "dataset_summary.json", {})
        train_clips = int(
            (genre_summary.get("target_total") or genre_summary.get("selected_total") or 1000)
            if isinstance(genre_summary, dict)
            else 1000
        )
        ziel_steps = max(1, -(-max(1, train_clips) // 8))
        commands = [
            command_python(
                str(LORA_TRAINING_SCRIPT),
                "--dataset-root",
                str(genre_dataset_dir(profile)),
                "--run-root",
                str(LORA_DIR.parent),
                "--run-name",
                f"lora_training_{profile}",
                "--resume-from",
                "auto",
                "--ziel-step",
                str(ziel_steps),
                "--steps-weiter",
                str(ziel_steps),
                "--train-clips",
                str(train_clips),
                "--learning-rate",
                "5e-6",
                "--vram-limit-fraction",
                "0.80",
                "--save-steps",
                "25",
                "--eval-steps",
                "50",
                "--ausgabe",
                "kompakt",
                "--ohne-resume",
            )
        ]
    elif action == "lora_training_reset":
        archived = archive_genre_dir(genre_run_dir(profile))
        return start_command(
            kind=action,
            commands=[command_python("-c", "pass")] if archived else [],
            prompt_profile=profile,
        ) | {"archived": archived}
    elif action == "lora_freigeben_genre":
        checkpoint = str(body.get("checkpoint") or "").strip()
        if not checkpoint:
            raise ValueError("Checkpoint fehlt")
        commands = [
            command_python(
                str(LORA_TRAINING_SCRIPT),
                "--run-root",
                str(LORA_DIR.parent),
                "--run-name",
                f"lora_training_{profile}",
                "--freigeben-checkpoint",
                checkpoint,
            )
        ]
    elif action == "lora_archive_delete":
        path_arg = str(body.get("path") or "").strip()
        if not path_arg:
            raise ValueError("Pfad fehlt")
        if not body.get("confirm"):
            raise ValueError("Bestaetigung fehlt")
        result = delete_archive(path_arg)
        return start_command(kind=action, commands=[], prompt_profile=profile) | result
    elif action == "testaudios":


        commands = []
        eingeplante_genres: list[str] = []
        genres = load_genres()
        requested = body.get("genres")
        if isinstance(requested, str):
            requested_genres = [requested]
        elif isinstance(requested, list):
            requested_genres = [str(item) for item in requested]
        else:
            requested_genres = []
        genre_keys = [key for key in requested_genres if key in genres] or sorted(genres.keys())
        max_generated_candidates = str(body.get("maxGeneratedCandidates") or "").strip()
        reference_score_limit = str(body.get("referenceScoreLimit") or "").strip()
        for genre_key in genre_keys:
            adapter_path, _source = resolve_review_adapter(genre_key)
            if adapter_path is None:
                continue
            command = command_python(
                str(START),
                "--stufen",
                "audio_generieren",
                "--dauer",
                "1m",
                "--genre",
                genre_label(genre_key),
                "--abschnitt-sekunden",
                "30",
                "--kandidaten-pro-abschnitt",
                "3",
                "--block-sekunden",
                "0",
                "--rhythmus-anteil-prozent",
                "5",
                "--min-rhythmus-sekunden",
                "90",
                "--block-looping-aktiv",
                "--kontinuierliche-bloecke-deaktivieren",
                "--crossfade-sekunden",
                "3",
                "--ziel-bpm",
                "78",
                "--bpm-toleranz",
                "4",
                "--min-sekunden-rms-db",
                "-50",
                "--vram-limit-fraction",
                "0.80",
                "--seed",
                "0",
                "--best-adapter",
                str(adapter_path),
                "--review-adapter-erlauben",
                "--genre-pruefung-aktiv",


                "--genre-melodie-conditioning-deaktivieren",


                "--kein-github-push",
            )
            if max_generated_candidates:
                command.extend(["--max-generierte-kandidaten", max_generated_candidates])
            if reference_score_limit:
                command.extend(["--referenz-score-limit", reference_score_limit])
            eingeplante_genres.append(genre_key)
            commands.append(command)
        if not commands:
            raise ValueError("Kein LoRA-Adapter fuer Testaudios vorhanden. Erst LoRA-Training abschliessen.")
        sammelordner = (
            ROOT
            / "training"
            / "ausgaben"
            / "musicgen_generiert"
            / f"testaudios_sammlung_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        commands.append(
            command_python(
                str(TESTAUDIOS_SAMMELN),
                "--genres",
                *eingeplante_genres,
                "--sammelordner",
                str(sammelordner),
            )
        )
        commands.append(
            command_python(
                str(AUDIO_VEROEFFENTLICHEN),
                str(sammelordner),
                "--commit-text",
                f"Testaudios: {len(eingeplante_genres)} Genres",
            )
        )
    elif action == "endprodukt_erstellen":
        audio_id = str(body.get("audioId") or "").strip()
        video_id = str(body.get("videoId") or "").strip()
        if not audio_id or not video_id:
            raise ValueError("audioId und videoId werden benoetigt.")
        run_dir = _audio_map().get(audio_id)
        video_path = _video_gallery_map().get(video_id)
        if run_dir is None:
            raise ValueError("Audio nicht gefunden.")
        if video_path is None:
            raise ValueError("Video nicht gefunden.")
        if video_path.suffix.lower() != ".mp4":
            raise ValueError("Nur MP4-Videos koennen mit Audio kombiniert werden, keine fertigen GIFs (kein Tonspur-Format).")
        audio_path = run_dir / "lange_audio.mp3"
        if not audio_path.exists():
            audio_path = run_dir / "lange_audio.wav"
        audio_titel = _audio_display_name(run_dir) or run_dir.name
        video_label = f"{_gallery_scenario_label(video_path)} · {video_path.name}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ausgabe_ordner = ENDPRODUKT_ROOT / f"{timestamp}_{slug(audio_titel)}"
        commands = [
            command_python(
                str(ENDPRODUKT_ERSTELLEN),
                "--audio",
                str(audio_path),
                "--video",
                str(video_path),
                "--ausgabe-ordner",
                str(ausgabe_ordner),
                "--audio-titel",
                audio_titel,
                "--video-label",
                video_label,
            )
        ]
    elif action == "trainingsclip_test":
        command = command_python(str(START), "--trainingsclip-test")
        run_name = str(body.get("runName") or "").strip()
        if run_name:
            command.extend(["--run-name", run_name])
        commands = [command]
    elif action == "audio_loopen":
        label = genre_label(str(body.get("genre") or "Alle Lofi Genres"))
        dauer = str(body.get("dauer") or "5m")
        command = command_python(
            str(START),
            "--audio-loopen",
            "--dauer",
            dauer,
            "--genre",
            label,
            "--crossfade-sekunden",
            str(body.get("crossfade") or 5),
            "--fade-in-sekunden",
            str(body.get("fadeIn") or 5),
            "--fade-out-sekunden",
            str(body.get("fadeOut") or 5),
        )
        if body.get("githubPush", True):
            command.append("--github-push")
        commands = [command]
    else:
        raise ValueError(f"Unbekannte Aktion: {action}")
    return start_command(kind=action, commands=commands, prompt_profile=display_profile, duration_min=duration)


def _extract_youtube_id(url: str) -> str:

    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else ""


def make_mp3_job(body: dict[str, Any]) -> dict[str, Any]:

    url = str(body.get("url") or "").strip()
    if not url:
        raise ValueError("URL fehlt")
    raw_genre = str(body.get("genre") or body.get("caption") or "").strip()
    genre = genre_label(raw_genre) if raw_genre else ""
    commands = [command_python(str(CRAWLER), "mp3", "--url", url)]
    if genre:
        commands[0].extend(["--genre", genre])
    if body.get("buildClips"):
        clip_command = command_python(str(START), "--clips-5000")
        if genre:
            clip_command.extend(["--genres", profile_from_label(genre)])
        commands.append(clip_command)
    job = start_command(kind="mp3_import", commands=commands, prompt_profile=profile_from_label(genre or "lofi"))
    video_id = _extract_youtube_id(url)
    job["videoId"] = video_id
    if video_id:
        marker_rows = read_jsonl(WEBSITE_MANUAL_IMPORTS_META)
        marker_rows.append({"video_id": video_id, "url": url, "importedAt": jetzt()})
        write_jsonl(WEBSITE_MANUAL_IMPORTS_META, marker_rows)
    return job


class ApiHandler(BaseHTTPRequestHandler):

    server_version = "LofiLabAPI/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:

        return

    def do_OPTIONS(self) -> None:
        self._send_empty(204)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        _public_media = (
            parsed.path.startswith("/api/video/media/")
            or parsed.path.startswith("/api/audio/")
            or parsed.path.startswith("/api/download/")
        )
        if not _public_media and not self._authorized():
            self._json({"error": "Nicht autorisiert"}, 401)
            return
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/api/status":
                self._json(model_status())
            elif path == "/api/models":
                self._json(models_payload())
            elif path == "/api/prompt-profiles":
                self._json(prompt_profiles())
            elif path == "/api/project/overview":
                self._json(
                    {
                        "status": model_status(),
                        "dataset": dataset_overview(),
                        "lora": lora_status(),
                        "genres": prompt_profiles(),
                        "loraGenres": lora_genres_overview(),
                    }
                )
            elif path == "/api/project/archives":
                self._json(list_archives())
            elif path == "/api/project/top10-results":
                self._json(top10_results())
            elif path == "/api/sources":
                self._json(sources_payload())
            elif path == "/api/sources/top10-status":
                self._json(top10_status_payload())
            elif path == "/api/jobs":
                with LOCK:
                    jobs = list(JOBS.values())
                jobs.sort(key=lambda row: str(row.get("startedAt") or ""), reverse=True)
                self._json([_public_job(row) for row in jobs])
            elif path.startswith("/api/jobs/"):
                job_id = path.split("/")[-1]
                with LOCK:
                    job = JOBS.get(job_id)
                if not job:
                    self._json({"error": "Job nicht gefunden"}, 404)
                else:
                    self._json(_public_job(job))
            elif path == "/api/audio":
                self._json(audio_payload())
            elif path == "/api/reviews":
                self._json(reviews_payload())
            elif path.startswith("/api/audio/") and path.endswith("/download"):
                self._download_audio(path, parse_qs(parsed.query))
            elif path.startswith("/api/audio/") and path.endswith("/report"):
                self._download_report(path)
            elif path == "/api/video-gallery":
                self._json(video_gallery_payload())
            elif path.startswith("/api/video-gallery/") and path.endswith("/file"):
                self._serve_gallery_file(path)
            elif path == "/api/final-products":
                self._json(endprodukt_payload())
            elif path.startswith("/api/final-products/") and path.endswith("/file"):
                self._serve_endprodukt_file(path)
            elif path == "/api/video/scenarios":
                self._json(video_scenarios_payload())
            elif path == "/api/video/projects":
                self._json(video_list_projects())
            elif path == "/api/video/feedback-history":
                self._json(video_feedback_history(parse_qs(parsed.query)))
            elif path.startswith("/api/video/media/"):
                self._serve_video_media(path)
            elif path.startswith("/api/musicgen/youtube-mp3/"):
                job_id = path.split("/")[-1]
                with LOCK:
                    job = JOBS.get(job_id)
                if not job:
                    self._json({"error": "Job nicht gefunden"}, 404)
                else:
                    self._json({"job_id": job_id, "job": _youtube_job(job)})
            else:
                self._json({"error": "Unbekannter Endpunkt"}, 404)
        except Exception as exc:
            try:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            except Exception:
                pass

    def do_POST(self) -> None:
        if not self._authorized():
            self._json({"error": "Nicht autorisiert"}, 401)
            return
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/api/sources/upload":
            self._handle_source_upload(parse_qs(parsed.query))
            return
        body = self._read_body()
        try:
            if path == "/api/generate":
                job = make_generate_job(body)
                self._json({"jobId": job["jobId"], "status": job["status"]})
            elif path == "/api/reviews":
                self._json(add_review(body))
            elif path == "/api/project/genres":
                self._json(add_genre(body))
            elif path.startswith("/api/project/run/"):
                action = path.split("/")[-1]
                job = make_project_job(action, body)
                response = {"jobId": job["jobId"], "status": job["status"]}
                for key in ("archived", "deleted", "freedBytes"):
                    if key in job:
                        response[key] = job[key]
                self._json(response)
            elif path == "/api/musicgen/youtube-mp3":
                job = make_mp3_job(body)
                self._json({"job_id": job["jobId"], "job": _youtube_job(job)})
            elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[-2]
                cancel_job(job_id)
                self._json({"ok": True})
            elif path.startswith("/api/audio/") and path.endswith("/rename"):
                audio_id = path.split("/")[-2]
                run_dir = _audio_map().get(audio_id)
                if run_dir is None:
                    self._json({"error": "Audio nicht gefunden"}, 404)
                else:
                    title = _set_audio_display_name(run_dir, str(body.get("name") or ""))
                    self._json({"ok": True, "title": title or run_dir.name})
            elif path.startswith("/api/sources/") and path.endswith("/delete"):
                video_id = path.split("/")[-2]
                self._json(delete_source(video_id))
            elif path == "/api/sources/prune-non-top10":
                self._json(prune_non_top10_sources())
            elif path == "/api/video/create-scenario":
                self._json(video_create_scenario(body))
            elif path == "/api/video/youtube-search":
                self._json(video_youtube_search(body))
            elif path == "/api/video/resolve-url":
                self._json(video_resolve_url(body))
            elif path == "/api/video/save-project":
                self._json(video_save_project(body))
            elif path == "/api/video/delete-project":
                self._json(video_delete_project(body))
            elif path == "/api/video/train":
                job = video_make_train_job(body)
                self._json({"ok": True, "job_id": job["job_id"], "round": job.get("round", 1)})
            elif path == "/api/video/generate":
                job = video_make_generate_job(body)
                self._json({"ok": True, "job_id": job["jobId"], "video_url": job.get("videoUrl")})
            elif path == "/api/video/make-gif":
                self._json(video_make_gif(body))
            elif path == "/api/video/quantified-feedback":
                self._json(video_quantified_feedback(body))
            elif path == "/api/video/evaluate":
                self._json(video_evaluate(body))
            else:
                self._json({"error": "Unbekannter Endpunkt"}, 404)
        except Exception as exc:
            try:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            except Exception:
                pass

    def _handle_source_upload(self, query: dict[str, list[str]]) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._json({"error": "Keine Datei empfangen"}, 400)
                return
            data = self.rfile.read(length)
            filename = (query.get("filename") or ["upload.mp3"])[0]
            genre = (query.get("genre") or [""])[0]
            title = (query.get("title") or [""])[0]
            self._json(save_uploaded_source(filename=filename, genre=genre, title=title, data=data))
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _send_headers(self, status: int, content_type: str, length: int | None = None) -> None:
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", CORS_ORIGIN)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Connection", "close")
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def _send_empty(self, status: int) -> None:
        self.close_connection = True
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", CORS_ORIGIN)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _authorized(self) -> bool:

        return True

    def _json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_headers(status, "application/json; charset=utf-8", length=len(data))
        self.wfile.write(data)

    def _download_audio(self, path: str, query: dict[str, list[str]]) -> None:
        audio_id = path.split("/")[-2]
        fmt = (query.get("format") or ["mp3"])[0]
        run_dir = _audio_map().get(audio_id)
        if not run_dir:
            self._json({"error": "Audio nicht gefunden"}, 404)
            return
        audio_path = run_dir / f"lange_audio.{fmt}"
        if not audio_path.exists():
            self._json({"error": "Format nicht gefunden"}, 404)
            return
        content_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"
        self._send_headers(200, content_type)
        with audio_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _stream_file(self, file_path: Path, mime: str) -> None:
        file_size = file_path.stat().st_size
        range_header = self.headers.get("Range", "")

        if range_header.startswith("bytes="):
            range_spec = range_header[6:].strip()
            start_str, _, end_str = range_spec.partition("-")
            start = int(start_str) if start_str.strip() else 0
            end = int(end_str) if end_str.strip() else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", CORS_ORIGIN)
            self.end_headers()

            with file_path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", CORS_ORIGIN)
            self.end_headers()

            with file_path.open("rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def _serve_video_media(self, path: str) -> None:

        parts = path.split("/api/video/media/", 1)
        if len(parts) < 2 or not parts[1]:
            self._json({"error": "Ungültiger Pfad"}, 400)
            return
        rel_path = unquote(parts[1])
        base_dir = LOFI_SCENARIOS_DIR.resolve()
        file_path = (base_dir / rel_path).resolve()
        if not file_path.is_relative_to(base_dir):
            self._json({"error": "Ungültiger Pfad"}, 400)
            return
        if not file_path.exists() or not file_path.is_file():
            self._json({"error": "Datei nicht gefunden"}, 404)
            return
        suffix = file_path.suffix.lower()
        mime = {"mp4": "video/mp4", "gif": "image/gif", "webm": "video/webm"}.get(suffix.lstrip("."), "application/octet-stream")
        self._stream_file(file_path, mime)

    def _serve_gallery_file(self, path: str) -> None:
        gallery_id = path.split("/")[-2]
        file_path = _video_gallery_map().get(gallery_id)
        if not file_path or not file_path.exists():
            self._json({"error": "Datei nicht gefunden"}, 404)
            return
        ext = file_path.suffix.lower().lstrip(".")
        mime = {"gif": "image/gif", "mp4": "video/mp4", "webm": "video/webm"}.get(ext, "application/octet-stream")
        self._stream_file(file_path, mime)

    def _serve_endprodukt_file(self, path: str) -> None:
        endprodukt_id = path.split("/")[-2]
        run_dir = _endprodukt_map().get(endprodukt_id)
        if not run_dir:
            self._json({"error": "Datei nicht gefunden"}, 404)
            return
        self._stream_file(run_dir / "final.mp4", "video/mp4")

    def _download_report(self, path: str) -> None:
        audio_id = path.split("/")[-2]
        run_dir = _audio_map().get(audio_id)
        if not run_dir:
            self._json({"error": "Report nicht gefunden"}, 404)
            return
        report = run_dir / "finale_audio_info.json"
        if not report.exists():
            report = run_dir / "generation_report.json"
        self._json(read_json(report, {}))

def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    if "done" in job:

        done = bool(job.get("done"))
        rc = job.get("returncode")
        failed = done and rc is not None and int(rc) != 0
        status = "completed" if (done and not failed) else ("failed" if failed else "running")
        return {
            "jobId": job.get("jobId") or "",
            "isVideoJob": True,
            "status": status,
            "kind": job.get("kind") or "",
            "scenarioId": job.get("scenarioId") or "",
            "phase": job.get("phase") or "",
            "trainStep": job.get("trainStep"),
            "trainTotal": job.get("trainTotal"),
            "videoUrl": job.get("videoUrl"),
            "round": int(job.get("round") or 1),
            "output": job.get("output") or "",
            "progress": float(job.get("progress") or 0) / 100.0,
            "startedAt": job.get("startedAt"),
            "finishedAt": job.get("finishedAt"),

            "promptProfile": "video",
            "targetDurationMin": 0,
            "outputFormat": "mp4",
            "audioId": None,
            "logPath": None,
            "lastLines": [job.get("output") or ""],
            "error": job.get("output") if failed else None,
        }


    return {
        "jobId": job.get("jobId"),
        "isVideoJob": False,
        "status": job.get("status"),
        "promptProfile": job.get("promptProfile") or "",
        "targetDurationMin": job.get("targetDurationMin") or 0,
        "outputFormat": job.get("outputFormat") or "wav_mp3",
        "startedAt": job.get("startedAt"),
        "finishedAt": job.get("finishedAt"),
        "progress": float(job.get("progress") or 0.0),
        "step": job.get("step") or "model_loading",
        "audioId": job.get("audioId"),
        "error": job.get("error"),
        "kind": job.get("kind"),
        "logPath": job.get("logPath"),
        "lastLines": job.get("lastLines") or [],
        "commandIndex": job.get("commandIndex"),
        "commandTotal": job.get("commandTotal"),
        "commandLabel": job.get("commandLabel"),
    }


def _youtube_job(job: dict[str, Any]) -> dict[str, Any]:

    progress = round(float(job.get("progress") or 0.0) * 100, 1)
    status = str(job.get("status") or "queued")
    video_id = str(job.get("videoId") or "")

    audio_exists = False
    audio_path = ""
    file_size_bytes = 0
    if video_id:
        source = _merge_source_entries().get(video_id)
        if source:
            path = Path(str(source.get("audio_path") or ""))
            audio_exists = path.exists()
            if audio_exists:
                audio_path = rel(path)
                file_size_bytes = int(source.get("file_size_bytes") or path.stat().st_size)

    return {
        "job_id": job.get("jobId"),
        "status": status,
        "stage": "done" if status == "completed" else "failed" if status == "failed" else "downloading",
        "progress_percent": progress,
        "clip_total": 0,
        "clip_current": 0,
        "clip_accepted": 0,
        "clip_rejected": 0,
        "build_clips": "clips-5000" in " ".join(" ".join(cmd) for cmd in job.get("commands") or []),
        "created_at": job.get("startedAt"),
        "started_at": job.get("startedAt"),
        "finished_at": job.get("finishedAt"),
        "error": job.get("error"),
        "video_id": video_id,
        "audio_exists": audio_exists,
        "audio_path": audio_path,
        "file_size_bytes": file_size_bytes,
    }


def main() -> int:

    host = os.environ.get("LOFILAB_API_HOST", "127.0.0.1")
    port = int(os.environ.get("LOFILAB_API_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), ApiHandler)
    print("LofiLab API", flush=True)
    print("===========", flush=True)
    print(f"URL: http://{host}:{port}", flush=True)
    print("Start: Website kann jetzt lokale Pipeline-Funktionen nutzen.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
