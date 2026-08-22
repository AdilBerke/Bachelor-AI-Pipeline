#!/usr/bin/env python3
"""Zentraler Starter fuer MusicGen-LoRA-Training.

Diese Datei startet nicht eine neue Trainingslogik, sondern prueft zuerst die
Trainingsdaten und ruft danach den vorhandenen Trainer in `musicgen_steuerung.py`
auf. Dadurch bleiben Checkpoint-Format, VRAM-Schutz und LoRA-Parameter an einer
zentralen Stelle.

Wichtig fuer die Bachelorarbeit:
- MP3-Rohdaten werden vor dem Training auf Duplikate geprueft.
- Manifest-Clips werden vor dem Training auf doppelte Pfade und doppelte
  Quell-Zeitfenster geprueft.
- Bei zusaetzlichen Dataset-Wurzeln wird ein dedupliziertes Manifest-Dataset
  erstellt, ohne WAV-Dateien zu kopieren.
- Wenn kritische Duplikate gefunden werden, startet das Training nicht.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import wave
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJEKTWURZEL = Path(__file__).resolve().parents[3]
PYTHON = PROJEKTWURZEL / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

TRAINER = PROJEKTWURZEL / "code" / "src" / "Training" / "musicgen_steuerung.py"
STANDARD_DATASET = PROJEKTWURZEL / "daten" / "processed" / "lora_training"
STANDARD_ROH_AUDIO = PROJEKTWURZEL / "daten" / "raw" / "audio"
STANDARD_RUN_ROOT = PROJEKTWURZEL / "training" / "musicgen"
STANDARD_RUN_NAME = "lora_training"
STANDARD_ZIEL_STEP = 625
STANDARD_FREIGEGEBENER_ADAPTER = PROJEKTWURZEL / "training" / "musicgen" / "lora_training" / "adapter.pt"
STANDARD_MODELL = PROJEKTWURZEL / "daten" / "modelle" / "musicgen" / "facebook_musicgen_melody_large"
STANDARD_BESTER_ADAPTER = STANDARD_FREIGEGEBENER_ADAPTER
LORA_STAND = STANDARD_RUN_ROOT / STANDARD_RUN_NAME / "stand.json"
SPLITS = ("train", "valid", "test")
MODELL_DATEIEN = ("state_dict.bin", "compression_state_dict.bin", "config.json")
YOUTUBE_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


def jetzt_utc() -> str:
    """UTC-Zeitstempel fuer reproduzierbare Reports."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    """Gibt Projektpfade kurz aus, damit Reports lesbar bleiben."""
    try:
        return str(path.resolve().relative_to(PROJEKTWURZEL))
    except Exception:
        return str(path)


def _policy_pfad(value: Any) -> Path:
    """Loest relative Pfade aus der Checkpoint-Policy gegen die Projektwurzel auf."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJEKTWURZEL / path
    return path.resolve()


def lade_lora_stand(run_dir: Path | None = None) -> dict[str, Any]:
    """Liest den lokalen LoRA-Stand eines Trainingslaufs (Standard: der geteilte)."""
    stand_path = (run_dir / "stand.json") if run_dir is not None else LORA_STAND
    if not stand_path.exists():
        return {}
    try:
        payload = json.loads(stand_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def policy_bester_adapter() -> Path:
    """Gibt den freigegebenen besten Adapter zurueck; Fallback ist LoRA."""
    payload = lade_lora_stand()
    best = payload.get("best_adapter") if payload.get("status") != "freigegeben" else payload.get("best_adapter") or payload.get("adapter")
    if best:
        path = _policy_pfad(best)
        if path.exists():
            return path
    return STANDARD_BESTER_ADAPTER.resolve()


def policy_verworfene_runs() -> set[str]:
    """Run-Namen, die nicht automatisch als Resume-Basis genutzt werden sollen."""
    payload = lade_lora_stand()
    runs = payload.get("rejected_runs", [])
    return {str(item).strip().lower() for item in runs if str(item).strip()}


def policy_verworfene_adapter() -> set[Path]:
    """Einzelne Adapter, die nicht automatisch genutzt werden duerfen."""
    payload = lade_lora_stand()
    adapters = payload.get("rejected_adapters", [])
    result: set[Path] = set()
    for item in adapters:
        try:
            result.add(_policy_pfad(item))
        except Exception:
            continue
    return result


def slug(text: str) -> str:
    """Erzeugt kurze deutsche Dateinamen ohne Sonderzeichenprobleme."""
    value = str(text or "").strip().lower()
    value = value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    value = re.sub(r"[^a-z0-9_.-]+", "_", value).strip("._-")
    return value or datetime.now().strftime("lora_%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    """Definiert die sichere Bedienung des LoRA-Starters."""
    parser = argparse.ArgumentParser(description="Prueft Daten und startet danach MusicGen-LoRA-Training.")
    parser.add_argument("--dataset-root", default=str(STANDARD_DATASET))
    parser.add_argument(
        "--zusatz-dataset-root",
        action="append",
        default=[],
        help="Optional weitere Dataset-Wurzeln. Daraus wird ein dedupliziertes Manifest gebaut.",
    )
    parser.add_argument("--kombiniert-name", default="", help="Name fuer ein dedupliziertes Kombi-Dataset.")
    parser.add_argument("--roh-audio-root", action="append", default=[], help="Ordner fuer MP3-Duplikatpruefung.")
    parser.add_argument("--run-root", default=str(STANDARD_RUN_ROOT))
    parser.add_argument("--run-name", default=STANDARD_RUN_NAME)
    parser.add_argument("--model-dir", default=str(STANDARD_MODELL))
    parser.add_argument("--remote-modell-erlauben", action="store_true")
    parser.add_argument(
        "--resume-from",
        default="auto",
        help=(
            "Checkpoint fuer Fortsetzung. Standard auto = letzter Checkpoint im LoRA-Lauf; "
            "lora/freigegeben nutzt bewusst nur den freigegebenen Adapter."
        ),
    )
    parser.add_argument("--ohne-resume", action="store_true", help="Von Basismodell starten, keinen Adapter laden.")
    parser.add_argument("--steps-weiter", type=int, default=625, help="Interne Trainingsschritte ab Resume-Checkpoint.")
    parser.add_argument(
        "--ziel-step",
        type=int,
        default=STANDARD_ZIEL_STEP,
        help="Interner Ziel-Step fuer LoRA. 625 entspricht bei batch=1 und accumulation=8 etwa 5000 Trainingsbeispielen.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Absolute Ziel-Step-Zahl. Ueberschreibt --ziel-step.")
    parser.add_argument(
        "--weitere-runde",
        action="store_true",
        help="Nach erreichtem Ziel bewusst weitere --steps-weiter trainieren.",
    )
    parser.add_argument(
        "--train-clips",
        type=int,
        default=5000,
        help="Maximale Anzahl Trainingsclips fuer diesen Lauf, z.B. 5000.",
    )
    parser.add_argument(
        "--valid-clips",
        type=int,
        default=500,
        help="Maximale Anzahl Validierungsclips fuer schnelle Zwischenpruefung.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Konservative LoRA-Lernrate; 5e-5 war fuer die kleine Quellenvielfalt zu aggressiv.",
    )
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--vram-limit-fraction", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("none", "torch"),
        default="none",
        help="Standard ist none, weil torch-checkpointing bei MusicGen hier instabil abbrechen kann.",
    )
    parser.add_argument(
        "--ausgabe",
        choices=("kompakt", "voll"),
        default="kompakt",
        help="kompakt zeigt nur Fortschritt; voll zeigt die komplette Trainer-Ausgabe.",
    )
    parser.add_argument("--volle-clip-hash-pruefung", action="store_true")
    parser.add_argument("--mp3-pruefung-aus", action="store_true")
    parser.add_argument("--duplikate-erlauben", action="store_true")
    parser.add_argument("--review-sample", action="store_true", help="Nach Training ein Review-Sample erzeugen.")
    parser.add_argument("--nur-pruefen", action="store_true", help="Nur Daten pruefen und Befehl schreiben.")
    parser.add_argument(
        "--freigeben-checkpoint",
        default="",
        help="Bewerteten Checkpoint als aktiven LoRA-Adapter freigeben.",
    )
    parser.add_argument("--freigabe-notiz", default="", help="Kurze Begruendung fuer die Checkpoint-Freigabe.")
    return parser.parse_args()


def datei_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Berechnet SHA256, um echte Datei-Duplikate zu erkennen."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def youtube_id(path: Path) -> str:
    """Liest eine YouTube-ID aus Dateinamen wie `Titel [VIDEO_ID].mp3`."""
    match = YOUTUBE_ID_RE.search(path.name)
    return match.group(1) if match else ""


def gruppen_mit_duplikaten(items: Iterable[tuple[str, Any]]) -> dict[str, list[Any]]:
    """Sammelt nur Schluessel, die mehrfach vorkommen."""
    gruppen: dict[str, list[Any]] = defaultdict(list)
    for key, item in items:
        if key:
            gruppen[key].append(item)
    return {key: value for key, value in gruppen.items() if len(value) > 1}


def training_laeuft_bereits() -> bool:
    """Verhindert doppelte LoRA-Starts aus VS Code oder einem zweiten Terminal."""

    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    except (OSError, subprocess.SubprocessError):
        return False

    eigener_pid = os.getpid()
    parent_pid = os.getppid()
    marker = ("code/src/Training/lora.py", "musicgen_steuerung.py train")
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid in {eigener_pid, parent_pid}:
            continue
        command = parts[1]
        if any(item in command for item in marker):
            return True
    return False


def pruefe_mp3_duplikate(audio_roots: list[Path]) -> dict[str, Any]:
    """Prueft MP3-Rohdaten auf doppelte Inhalte und doppelte YouTube-IDs."""
    mp3s: list[Path] = []
    for root in audio_roots:
        if root.exists():
            mp3s.extend(sorted(root.rglob("*.mp3")))

    hash_rows = []
    id_rows = []
    for path in mp3s:
        hash_rows.append((datei_hash(path), rel(path)))
        video_id = youtube_id(path)
        if video_id:
            id_rows.append((video_id, rel(path)))

    return {
        "mp3_count": len(mp3s),
        "duplicate_hashes": gruppen_mit_duplikaten(hash_rows),
        "duplicate_youtube_ids": gruppen_mit_duplikaten(id_rows),
    }


def manifest_roots(root: Path) -> list[Path]:
    """Findet Dataset-Wurzeln mit train/valid/test-Manifests.

    Wenn `root` selbst ein Dataset ist, wird nur dieses genutzt. Wenn `root`
    ein Sammelordner ist, werden darunter alle Dataset-Unterordner gefunden.
    """
    root = root.expanduser().resolve()
    if all((root / split / "data.jsonl").exists() for split in SPLITS):
        return [root]
    found = sorted({path.parents[1] for path in root.glob("**/train/data.jsonl")})
    return [item for item in found if all((item / split / "data.jsonl").exists() for split in SPLITS)]


def dataset_training_ready(root: Path) -> tuple[bool, str]:
    """Blockiert Trainingsstarts auf bewusst unvollstaendigen Ziel-Datasets."""
    summary_path = root / "dataset_summary.json"
    if not summary_path.exists():
        return True, ""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True, ""
    if summary.get("training_ready") is False:
        missing = summary.get("missing_total")
        target = summary.get("target_total")
        selected = summary.get("selected_total")
        return (
            False,
            f"Dataset ist noch nicht trainingsbereit: {selected}/{target} Clips vorhanden, {missing} fehlen.",
        )
    if "target_total" in summary and not summary.get("source_disjoint"):
        return (
            False,
            "Dataset nutzt noch die alte Clip-Aufteilung: Ursprungs-MP3s sind "
            "nicht sicher zwischen Train, Valid und Test getrennt.",
        )
    if "target_total" in summary and not summary.get("minimum_independent_sources_met"):
        return (
            False,
            "Dataset hat fuer mindestens ein Genre zu wenige unabhaengige MP3-Quellen.",
        )
    return True, ""


def lese_jsonl(path: Path) -> list[dict[str, Any]]:
    """Liest eine JSONL-Manifestdatei."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} ist kein JSON-Objekt")
            row["_manifest_path"] = str(path)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def lade_manifest_rows(roots: list[Path]) -> list[dict[str, Any]]:
    """Laedt alle Manifest-Zeilen aus einer oder mehreren Dataset-Wurzeln."""
    rows: list[dict[str, Any]] = []
    for root in roots:
        for split in SPLITS:
            for row in lese_jsonl(root / split / "data.jsonl"):
                row["_dataset_root"] = str(root)
                row["_manifest_split"] = split
                rows.append(row)
    return rows


def nummer(value: Any) -> str:
    """Normalisiert Sekundenwerte fuer Duplikat-Schluessel."""
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def clip_quellschluessel(row: dict[str, Any]) -> str:
    """Erkennt doppelte Clips ueber Quelle und Zeitfenster.

    Alte 60s-Splits und neue YouTube-MP3-Clips nutzen unterschiedliche
    Feldnamen. Diese Funktion fuehrt sie in einen gemeinsamen Schluessel.
    """
    # Bei 60s-zu-30s-Datasets ist `source_60s_path` der verlaesslichste
    # Ursprung. Manche alte Manifeste setzen `clip_start_sec` pro 60s-Datei
    # erneut auf 0/30; nur source_video_id + Sekunden wuerde dort falsche
    # Duplikate melden.
    source_60s_path = row.get("source_60s_path")
    if source_60s_path:
        teil = row.get("teil") or ""
        teil_start = row.get("teil_start_sec")
        teil_end = row.get("teil_end_sec")
        return f"60s|{source_60s_path}|{teil}|{nummer(teil_start)}|{nummer(teil_end)}"

    # Neue MP3/Youtube-Imports haben ein Roh-Audio plus absolutes Zeitfenster.
    source_audio_path = row.get("source_audio_path")
    start_time = row.get("start_time_sec")
    end_time = row.get("end_time_sec")
    if source_audio_path and start_time is not None and end_time is not None:
        return f"audio|{source_audio_path}|{nummer(start_time)}|{nummer(end_time)}"

    source_id = (
        row.get("source_id")
        or row.get("source_video_id")
        or row.get("source_webpage_url")
        or row.get("source_url")
    )
    start = row.get("start_time_sec", row.get("clip_start_sec"))
    end = row.get("end_time_sec", row.get("clip_end_sec"))
    if source_id and start is not None and end is not None:
        return f"{source_id}|{nummer(start)}|{nummer(end)}"
    path = str(row.get("path") or "")
    return path


def wav_header(path: Path) -> dict[str, Any]:
    """Prueft schnell, ob WAV-Header zum MusicGen-Standard passt."""
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        frames = int(handle.getnframes())
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": frames / float(sample_rate or 1),
    }


def pruefe_manifest_clips(rows: list[dict[str, Any]], *, volle_hash_pruefung: bool) -> dict[str, Any]:
    """Prueft Clip-Manifeste auf Duplikate und offensichtliche Fehler."""
    path_items = []
    name_items = []
    source_items = []
    hash_items = []
    fehler: list[dict[str, Any]] = []

    for row in rows:
        raw_path = str(row.get("path") or "").strip()
        path = Path(raw_path).expanduser() if raw_path else Path()
        info = {
            "split": row.get("_manifest_split"),
            "line": row.get("_line_number"),
            "path": raw_path,
            "manifest": row.get("_manifest_path"),
        }
        if not raw_path:
            fehler.append({**info, "problem": "pfad_fehlt"})
            continue
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = raw_path
        path_items.append((resolved, info))
        name_items.append((path.name, info))
        source_items.append((clip_quellschluessel(row), info))

        if not path.exists():
            fehler.append({**info, "problem": "datei_fehlt"})
            continue
        if path.suffix.lower() != ".wav":
            fehler.append({**info, "problem": "kein_wav"})
        if not str(row.get("caption") or row.get("text") or row.get("description") or "").strip():
            fehler.append({**info, "problem": "caption_fehlt"})
        try:
            header = wav_header(path)
            if header["sample_rate"] != 32000:
                fehler.append({**info, "problem": "sample_rate_falsch", "wert": header["sample_rate"]})
            if header["channels"] != 1:
                fehler.append({**info, "problem": "kanaele_falsch", "wert": header["channels"]})
            if abs(float(header["duration"]) - 30.0) > 0.05:
                fehler.append({**info, "problem": "dauer_falsch", "wert": round(float(header["duration"]), 4)})
        except Exception as exc:
            fehler.append({**info, "problem": "wav_nicht_lesbar", "fehler": f"{type(exc).__name__}: {exc}"})
        if volle_hash_pruefung and path.exists() and path.suffix.lower() == ".wav":
            hash_items.append((datei_hash(path), info))

    report = {
        "clip_count": len(rows),
        "duplicate_paths": gruppen_mit_duplikaten(path_items),
        "duplicate_names": gruppen_mit_duplikaten(name_items),
        "duplicate_source_windows": gruppen_mit_duplikaten(source_items),
        "duplicate_audio_hashes": gruppen_mit_duplikaten(hash_items) if volle_hash_pruefung else {},
        "errors": fehler,
    }
    return report


def dedupliziere_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Entfernt doppelte Clip-Zeilen fuer ein neues Kombi-Manifest."""
    gesehen: set[str] = set()
    sauber: list[dict[str, Any]] = []
    uebersprungen: list[dict[str, Any]] = []
    for row in rows:
        key = clip_quellschluessel(row)
        path_key = str(Path(str(row.get("path") or "")).expanduser().resolve())
        combined_key = f"{key}|{path_key}"
        if key in gesehen or path_key in gesehen or combined_key in gesehen:
            uebersprungen.append(
                {
                    "split": row.get("_manifest_split"),
                    "line": row.get("_line_number"),
                    "path": row.get("path"),
                    "source_key": key,
                    "grund": "duplikat",
                }
            )
            continue
        gesehen.add(key)
        gesehen.add(path_key)
        gesehen.add(combined_key)
        clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
        sauber.append(clean_row)
    return sauber, uebersprungen


def schreibe_kombi_dataset(rows: list[dict[str, Any]], name: str) -> Path:
    """Schreibt ein dedupliziertes Manifest-Dataset ohne Audiodateien zu kopieren."""
    ziel = PROJEKTWURZEL / "daten" / "processed" / slug(name)
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in rows:
        split = str(row.get("split") or row.get("_manifest_split") or "train")
        if split not in split_rows:
            split = "train"
        row["split"] = split
        split_rows[split].append(row)

    for split, items in split_rows.items():
        manifest = ziel / split / "data.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", encoding="utf-8") as handle:
            for row in items:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "created_at": jetzt_utc(),
        "status": "ready",
        "dataset_root": rel(ziel),
        "mode": "dedupliziertes_manifest_ohne_audio_kopien",
        "split_counts": {split: len(items) for split, items in split_rows.items()},
        "total": sum(len(items) for items in split_rows.values()),
        "duration_sec": 30.0,
        "sample_rate": 32000,
        "channels": 1,
    }
    (ziel / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ziel


def checkpoint_step(path: Path) -> int:
    """Liest die technische Step-Zahl aus dem Ordner oder Adapter-Payload."""
    for part in reversed(path.parts):
        if part.startswith("step_") and part.split("step_", 1)[1].isdigit():
            return int(part.split("step_", 1)[1])
    return 0


def finde_neuesten_checkpoint(run_root: Path, *, max_step: int = 0) -> Path | None:
    """Findet den hoechsten gespeicherten LoRA-Step im Trainingsordner.

    Der Starter soll nicht jedes Mal manuell einen neuen Resume-Pfad brauchen.
    Deshalb wird standardmaessig der Checkpoint mit der hoechsten Step-Zahl
    genutzt. Bei gleicher Step-Zahl gewinnt die zuletzt geaenderte Datei.
    """

    candidates = []
    rejected_runs = policy_verworfene_runs()
    rejected_adapters = policy_verworfene_adapter()
    patterns = (
        "checkpoints/step_*/lora_adapter.pt",
        "*/checkpoints/step_*/lora_adapter.pt",
    )
    for pattern in patterns:
        paths = run_root.glob(pattern)
        for path in paths:
            try:
                relative_parts = path.relative_to(run_root).parts
            except Exception:
                relative_parts = ()
            run_name = relative_parts[0].lower() if "checkpoints" not in relative_parts[:1] else run_root.name.lower()
            if any(marker in run_name for marker in ("test", "smoke", "pruefung", "prüfung", "check")):
                continue
            if run_name in rejected_runs:
                continue
            step = checkpoint_step(path)
            if step <= 0 or not path.is_file():
                continue
            if max_step > 0 and step > max_step:
                continue
            resolved = path.resolve()
            if resolved in rejected_adapters:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates.append((step, mtime, resolved))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def finde_checkpoint_mit_step(run_dir: Path, target_step: int) -> Path | None:
    """Findet den Checkpoint des geplanten Zielstands im aktiven Run."""

    direct = run_dir / "checkpoints" / f"step_{target_step:06d}" / "lora_adapter.pt"
    if direct.is_file():
        return direct.resolve()
    candidates = []
    for path in run_dir.glob("checkpoints/step_*/lora_adapter.pt"):
        if checkpoint_step(path) != target_step:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, path.resolve()))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def finde_neuesten_checkpoint_im_run(run_root: Path, run_name: str) -> Path | None:
    """Findet zuerst den letzten Checkpoint des gewuenschten Run-Ordners."""

    run_dir = run_root / slug(run_name)
    latest = finde_neuesten_checkpoint(run_dir, max_step=geplanter_ziel_step(run_dir))
    if latest:
        return latest
    return None


def lese_run_plan(run_dir: Path) -> dict[str, Any]:
    """Liest den gespeicherten Zielplan eines LoRA-Runs."""

    plan_path = run_dir / "training_plan.json"
    if not plan_path.exists():
        return {}
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def geplanter_ziel_step(run_dir: Path) -> int:
    """Gibt das bisher geplante Ziel zurueck, falls ein Run fortgesetzt wird."""

    plan = lese_run_plan(run_dir)
    try:
        return int(plan.get("target_step") or 0)
    except (TypeError, ValueError):
        return 0


def loese_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    """Loest `--resume-from auto` in einen echten Checkpoint-Pfad auf."""

    if args.ohne_resume:
        return None
    normalized = str(args.resume_from).strip().lower()
    if normalized in {"lora", "freigegeben", "sicher", "approved"}:
        if STANDARD_FREIGEGEBENER_ADAPTER.exists():
            adapter = STANDARD_FREIGEGEBENER_ADAPTER.resolve()
            args.resume_from = str(adapter)
            return adapter
        best = policy_bester_adapter()
        args.resume_from = str(best)
        return best
    if normalized in {"best", "bester", "stabil"}:
        best = policy_bester_adapter()
        args.resume_from = str(best)
        return best
    if normalized in {"auto", "latest", "neueste", "neuster"}:
        run_root = Path(args.run_root).expanduser().resolve()
        latest = finde_neuesten_checkpoint_im_run(run_root, args.run_name)
        if latest:
            args.resume_from = str(latest)
            return latest
        # Ein neuer Run darf nicht unbemerkt einen Adapter aus einem anderen
        # Experiment erben. Ohne Checkpoint im explizit gewaehlten Run startet
        # ``auto`` sauber vom Basismodell. Fuer eine bewusste Uebernahme gibt es
        # weiterhin ``--resume-from best`` oder einen konkreten Pfad.
        args.ohne_resume = True
        args.resume_from = ""
        return None
    return Path(args.resume_from).expanduser().resolve()


def naechster_run_name(run_root: Path, start_step: int, max_steps: int) -> str:
    """Erzeugt einen kurzen, verstaendlichen Run-Namen ohne manuelle Nummern."""

    base = f"training_{start_step}_bis_{max_steps}"
    candidate = base
    index = 2
    while run_ordner_hat_training(run_root / candidate):
        candidate = f"{base}_{index:02d}"
        index += 1
    return candidate


def run_ordner_hat_training(path: Path) -> bool:
    """Prueft, ob ein Run-Ordner bereits echtes Training enthaelt.

    Ein Ordner aus `--nur-pruefen` soll den naechsten echten Start nicht zu
    `_02` verschieben. Erst Checkpoints, Metriken oder Trainingslogs zaehlen als
    belegter Trainingslauf.
    """

    if not path.exists():
        return False
    if any(path.glob("checkpoints/step_*/lora_adapter.pt")) or (path / "abschluss.json").exists():
        return True
    metrics = path / "metrics.csv"
    if metrics.exists():
        try:
            return len(metrics.read_text(encoding="utf-8").splitlines()) > 1
        except Exception:
            return True
    return False


def modell_vorhanden(model_dir: Path) -> bool:
    """Prueft, ob das lokale MusicGen-Modell vollstaendig vorhanden ist."""
    return model_dir.is_dir() and all((model_dir / name).exists() for name in MODELL_DATEIEN)


def flache_duplikate(groups: dict[str, list[Any]], typ: str) -> list[dict[str, Any]]:
    """Macht Duplikatgruppen als CSV lesbar."""
    rows: list[dict[str, Any]] = []
    for key, items in groups.items():
        for item in items:
            if isinstance(item, dict):
                rows.append({"typ": typ, "schluessel": key, **item})
            else:
                rows.append({"typ": typ, "schluessel": key, "wert": item})
    return rows


def baue_trainingsbefehl(args: argparse.Namespace, dataset_root: Path, run_name: str) -> list[str]:
    """Erstellt den eigentlichen `musicgen_steuerung.py train`-Befehl."""
    resume = Path(args.resume_from).expanduser().resolve()
    start_step = 0 if args.ohne_resume else checkpoint_step(resume)
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else start_step + int(args.steps_weiter)
    command = [
        str(PYTHON),
        str(TRAINER.relative_to(PROJEKTWURZEL)),
        "train",
        "--mode",
        "train",
        "--dataset-root",
        str(dataset_root),
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--run-name",
        run_name,
        "--model-dir",
        str(Path(args.model_dir).expanduser().resolve()),
        "--auto-vram",
        "--auto-reduce-on-oom",
        "--max-steps",
        str(max_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--save-steps",
        str(args.save_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--vram-limit-fraction",
        str(args.vram_limit_fraction),
        "--precision",
        args.precision,
        "--gradient-checkpointing",
        args.gradient_checkpointing,
    ]
    if not args.ohne_resume:
        command.extend(["--resume-from", str(resume)])
    if args.remote_modell_erlauben:
        command.append("--allow-remote-model")
    if args.batch_size > 0:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.gradient_accumulation > 0:
        command.extend(["--gradient-accumulation", str(args.gradient_accumulation)])
    if args.lora_rank > 0:
        command.extend(["--lora-rank", str(args.lora_rank)])
    if args.lora_alpha > 0:
        command.extend(["--lora-alpha", str(args.lora_alpha)])
    if args.train_clips > 0:
        command.extend(["--train-limit", str(args.train_clips)])
    if args.valid_clips > 0:
        command.extend(["--valid-limit", str(args.valid_clips)])
    if not args.review_sample:
        command.append("--no-generate-review-sample")
    return command


def zaehle_manifest_rows(dataset_root: Path, split: str) -> int:
    """Zaehlt Manifest-Zeilen, ohne die Audiodateien erneut zu lesen."""

    path = dataset_root / split / "data.jsonl"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def trainings_start_und_ziel(args: argparse.Namespace, run_dir: Path) -> tuple[int, int]:
    """Berechnet Start- und Zielstep passend zum Resume-Checkpoint."""
    resume = Path(args.resume_from).expanduser().resolve()
    start_step = 0 if args.ohne_resume else checkpoint_step(resume)
    ziel_step = int(args.max_steps) if int(args.max_steps) > 0 else int(args.ziel_step)
    if ziel_step > start_step:
        max_steps = ziel_step
    else:
        planned_target = geplanter_ziel_step(run_dir)
        if planned_target > start_step:
            max_steps = planned_target
        elif (planned_target > 0 or ziel_step > 0) and not args.weitere_runde:
            max_steps = start_step
        else:
            max_steps = start_step + int(args.steps_weiter)
    return start_step, max_steps


def schreibe_training_plan(
    run_dir: Path,
    *,
    status: str,
    dataset_root: Path,
    start_step: int,
    target_step: int,
    command: list[str],
    returncode: int | None = None,
) -> Path:
    """Hält den aktuellen LoRA-Lauf in einer einzigen kleinen Statusdatei fest."""

    path = run_dir / "training_plan.json"
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}
    resume_source = "base_model"
    if "--resume-from" in command:
        index = command.index("--resume-from")
        if index + 1 < len(command):
            resume_source = rel(Path(command[index + 1]).expanduser())
    payload.update(
        {
            "name": "LoRA",
            "status": status,
            "updated_at": jetzt_utc(),
            "run_name": run_dir.name,
            "dataset_root": rel(dataset_root),
            "current_step": start_step,
            "target_step": target_step,
            "steps_this_run": max(0, target_step - start_step),
            "checkpointing": {
                "save_steps": 25,
                "automatic_resume": True,
                "resume_source": resume_source,
            },
            "command": command,
        }
    )
    payload.setdefault("created_at", jetzt_utc())
    if status == "running":
        payload["started_at"] = jetzt_utc()
        payload.pop("finished_at", None)
        payload.pop("returncode", None)
    if returncode is not None:
        payload["finished_at"] = jetzt_utc()
        payload["returncode"] = returncode
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def freigeben_checkpoint(checkpoint: Path, notiz: str = "", run_dir: Path | None = None) -> Path:
    """Gibt einen menschlich bewerteten Checkpoint als aktiven LoRA-Adapter frei.

    ``run_dir`` erlaubt die Freigabe fuer einen eigenen (z.B. genre-spezifischen)
    Trainingslauf, ohne den geteilten Basis-Adapter/-Stand zu beruehren. Ohne
    Angabe bleibt das Verhalten identisch zum bisherigen, einzigen Ablauf.
    """

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"Checkpoint nicht gefunden: {checkpoint}")

    active_run_dir = run_dir if run_dir is not None else (STANDARD_RUN_ROOT / STANDARD_RUN_NAME)
    active_run_dir.mkdir(parents=True, exist_ok=True)
    active_adapter = active_run_dir / "adapter.pt"
    if active_adapter.exists() or active_adapter.is_symlink():
        active_adapter.unlink()
    active_adapter.symlink_to(os.path.relpath(checkpoint, active_adapter.parent))

    previous = lade_lora_stand(active_run_dir if run_dir is not None else None)
    stand = {
        "name": "LoRA",
        "beschreibung": "Menschlich bewerteter und freigegebener MusicGen-LoRA-Stand.",
        "adapter": rel(active_adapter),
        "best_adapter": rel(active_adapter),
        "checkpoint": rel(checkpoint),
        "adapter_sha256": datei_hash(checkpoint),
        "checkpoint_step": checkpoint_step(checkpoint),
        "dataset": previous.get("dataset", ""),
        "clips_gesamt": previous.get("clips_gesamt", 0),
        "genres": previous.get("genres", 0),
        "clips_pro_genre": previous.get("clips_pro_genre", {}),
        "status": "freigegeben",
        "review_required": False,
        "freigabe_notiz": notiz,
        "rejected_runs": previous.get("rejected_runs", []),
        "rejected_adapters": previous.get("rejected_adapters", []),
        "aktualisiert_am": jetzt_utc(),
    }
    (active_run_dir / "stand.json").write_text(
        json.dumps(stand, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return active_adapter


def veroeffentliche_lora_stand(run_dir: Path, dataset_root: Path, target_step: int) -> Path:
    """Registriert einen erfolgreichen Checkpoint zuerst als Review-Kandidat.

    Frueher wurde ein fertig trainierter Checkpoint direkt als aktiver Adapter
    benutzt. Genau dadurch konnten schlechtere spaetere Checkpoints die
    Audioerstellung verschlechtern. Jetzt bleibt der bisher freigegebene Adapter
    aktiv, bis der neue Checkpoint menschlich bewertet und explizit freigegeben
    wurde.
    """

    checkpoint = finde_checkpoint_mit_step(run_dir, target_step)
    if checkpoint is None:
        raise RuntimeError(f"Fertiger LoRA-Checkpoint fuer Step {target_step} fehlt.")

    kandidat_adapter = run_dir / "kandidat_adapter.pt"
    if kandidat_adapter.exists() or kandidat_adapter.is_symlink():
        kandidat_adapter.unlink()
    kandidat_adapter.symlink_to(checkpoint.relative_to(run_dir))

    summary: dict[str, Any] = {}
    summary_path = dataset_root / "dataset_summary.json"
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = loaded
        except (OSError, json.JSONDecodeError):
            summary = {}

    selected_by_genre = summary.get("selected_by_genre") or {}
    ist_geteilter_lauf = run_dir == (STANDARD_RUN_ROOT / STANDARD_RUN_NAME)
    vorheriger_stand = lade_lora_stand(None if ist_geteilter_lauf else run_dir)
    vorheriger_adapter = vorheriger_stand.get("best_adapter") or vorheriger_stand.get("adapter") or ""

    stand = {
        "name": "LoRA",
        "beschreibung": "Neu trainierter MusicGen-LoRA-Kandidat. Vor Nutzung menschlich bewerten.",
        "candidate_adapter": rel(kandidat_adapter),
        "adapter": rel(kandidat_adapter),
        "best_adapter": vorheriger_adapter,
        "checkpoint": rel(checkpoint),
        "adapter_sha256": datei_hash(checkpoint),
        "checkpoint_step": checkpoint_step(checkpoint),
        "dataset": rel(dataset_root),
        "clips_gesamt": int(summary.get("selected_total") or 0),
        "genres": len(selected_by_genre),
        "clips_pro_genre": selected_by_genre,
        "status": "bewertung_offen",
        "review_required": True,
        "use_for_audio_generation": False,
        "naechster_schritt": "Testaudios erzeugen, bewerten und danach bewusst freigeben.",
        "freigabe_befehl": f".venv/bin/python code/src/Training/lora.py --freigeben-checkpoint {rel(checkpoint)}",
        "rejected_runs": vorheriger_stand.get("rejected_runs", []),
        "rejected_adapters": vorheriger_stand.get("rejected_adapters", []),
        "aktualisiert_am": jetzt_utc(),
    }
    (run_dir / "stand.json").write_text(
        json.dumps(stand, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if ist_geteilter_lauf:
        # Nur beim geteilten Basis-Lauf zusaetzlich den Standard-Zeiger aktualisieren.
        # Ein genre-eigener Lauf (anderer run_dir) darf den geteilten Status nie
        # ueberschreiben - sonst wuerde ein neues, noch unbewertetes Genre-Training
        # den bewaehrten Basis-Adapter-Status verdraengen.
        active_run_dir = STANDARD_RUN_ROOT / STANDARD_RUN_NAME
        active_run_dir.mkdir(parents=True, exist_ok=True)
        active_stand = dict(stand)
        active_stand["aktiver_run"] = run_dir.name
        (active_run_dir / "stand.json").write_text(
            json.dumps(active_stand, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return kandidat_adapter


def schreibe_terminal_fortschritt(
    *,
    step: int,
    max_steps: int,
    start_step: int,
    clip_count: int,
    batch_size: int,
    gradient_accumulation: int,
) -> None:
    """Aktualisiert eine einzige Terminalzeile mit minimalem Trainingsstand."""
    effektive_clips_pro_step = max(1, batch_size * gradient_accumulation)
    trainierte_clips = max(0, step - start_step) * effektive_clips_pro_step
    ziel_clips = max(1, (max_steps - start_step) * effektive_clips_pro_step)
    rest_prozent = min(100.0, max(0.0, 100.0 - ((step - start_step) / max(1, max_steps - start_step) * 100.0)))
    print(
        "\r"
        f"Trainingsbeispiele: {trainierte_clips}/{ziel_clips} | "
        f"Rest: {rest_prozent:6.2f}%",
        end="",
        flush=True,
    )


def fuehre_training_kompakt_aus(
    command: list[str],
    *,
    run_dir: Path,
    start_step: int,
    max_steps: int,
    clip_count: int,
    batch_size: int,
    gradient_accumulation: int,
    voll: bool,
) -> int:
    """Startet den Trainer und filtert die Terminalausgabe auf einen Live-Stand."""
    if voll:
        return subprocess.call(command, cwd=PROJEKTWURZEL)

    log_path = run_dir / "training_konsole.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONFAULTHANDLER"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
    env["XFORMERS_MORE_DETAILS"] = "0"
    step_re = re.compile(r"LoRA Step\s+(\d+)/(\d+)")
    fehlerzeilen: list[str] = []
    last_step = start_step
    status_path = run_dir / "training_status.json"
    status_started_at = jetzt_utc()

    status_path.write_text(
        json.dumps(
            {
                "status": "running",
                "started_at": status_started_at,
                "start_step": start_step,
                "last_seen_step": start_step,
                "max_steps": max_steps,
                "train_manifest_count": clip_count,
                "examples_per_optimizer_step": max(1, batch_size * gradient_accumulation),
                "log_path": rel(log_path),
                "command": command,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    schreibe_terminal_fortschritt(
        step=start_step,
        max_steps=max_steps,
        start_step=start_step,
        clip_count=clip_count,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[starter] LoRA-Training gestartet: {jetzt_utc()}\n")
        log_handle.write(f"[starter] start_step={start_step} max_steps={max_steps}\n")
        log_handle.write(f"[starter] command={' '.join(subprocess.list2cmdline([part]) for part in command)}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJEKTWURZEL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            match = step_re.search(line)
            if match:
                step = int(match.group(1))
                last_step = step
                max_steps = int(match.group(2))
                schreibe_terminal_fortschritt(
                    step=step,
                    max_steps=max_steps,
                    start_step=start_step,
                    clip_count=clip_count,
                    batch_size=batch_size,
                    gradient_accumulation=gradient_accumulation,
                )
                status_path.write_text(
                    json.dumps(
                        {
                            "status": "running",
                            "started_at": status_started_at,
                            "start_step": start_step,
                            "last_seen_step": last_step,
                            "max_steps": max_steps,
                            "train_manifest_count": clip_count,
                            "examples_per_optimizer_step": max(1, batch_size * gradient_accumulation),
                            "processed_training_examples_this_process": max(
                                0, last_step - start_step
                            )
                            * max(1, batch_size * gradient_accumulation),
                            "log_path": rel(log_path),
                            "command": command,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            known_xformers_noise = (
                "WARNING[XFORMERS]" in line
                or "xformers" in line.lower()
                or "xFormers can't load C++/CUDA extensions" in line
            )
            if (
                not known_xformers_noise
                and ("Traceback" in line or "RuntimeError" in line or "Error" in line or "AttributeError" in line)
            ):
                fehlerzeilen.append(line.strip())
                fehlerzeilen = fehlerzeilen[-8:]
        returncode = process.wait()
        log_handle.write(f"\n[starter] returncode={returncode}\n")
        if returncode < 0:
            signal_number = -returncode
            signal_name = signal.Signals(signal_number).name if signal_number in [item.value for item in signal.Signals] else "unbekannt"
            log_handle.write(f"[starter] signal={signal_name} ({signal_number})\n")
        log_handle.flush()

    print()
    status_payload = {
        "status": "finished" if returncode == 0 else "failed",
        "finished_at": jetzt_utc(),
        "returncode": returncode,
        "start_step": start_step,
        "last_seen_step": last_step,
        "max_steps": max_steps,
        "train_manifest_count": clip_count,
        "examples_per_optimizer_step": max(1, batch_size * gradient_accumulation),
        "log_path": rel(log_path),
        "command": command,
        "xformers_warning_seen": "WARNING[XFORMERS]" in log_path.read_text(encoding="utf-8", errors="replace"),
        "xformers_warning_ignored": True,
    }
    if returncode < 0:
        signal_number = -returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = "unbekannt"
        status_payload["signal"] = signal_name
        status_payload["signal_number"] = signal_number
        status_payload["hinweis"] = (
            "Der Trainingsprozess wurde nativ beendet. Wenn vorher nur eine xFormers-Warnung im Log steht, "
            "ist die lokale xFormers-Version wahrscheinlich inkompatibel oder der Prozess wurde vom System beendet."
        )
    status_path.write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if returncode != 0:
        print(f"Status: abgebrochen mit Code {returncode}. Log: {rel(log_path)}")
        print(f"Statusreport: {rel(run_dir / 'training_status.json')}")
        if status_payload.get("xformers_warning_seen"):
            print("Hinweis: xFormers ist lokal inkompatibel. Training nutzt/erzwingt jetzt Diagnose- und Stabilitaets-Flags.")
        if fehlerzeilen:
            print("Letzter Fehler:")
            for line in fehlerzeilen:
                print(f"- {line}")
    return returncode


def main() -> int:
    """Fuehrt Vorpruefung aus und startet bei sauberem Zustand das LoRA-Training."""
    args = parse_args()
    if args.freigeben_checkpoint:
        try:
            freigabe_run_dir = Path(args.run_root).expanduser().resolve() / args.run_name
            adapter = freigeben_checkpoint(
                Path(args.freigeben_checkpoint), args.freigabe_notiz, run_dir=freigabe_run_dir
            )
        except RuntimeError as exc:
            print(f"Status: Freigabe fehlgeschlagen. {exc}", file=sys.stderr)
            return 4
        print("LoRA-Freigabe")
        print("=============")
        print("Status: freigegeben")
        print(f"Adapter: {rel(adapter)}")
        return 0
    if not args.nur_pruefen and training_laeuft_bereits():
        print("Status: LoRA-Training laeuft bereits. Kein zweiter Prozess wurde gestartet.")
        return 6
    run_root = Path(args.run_root).expanduser().resolve()
    run_name = slug(args.run_name or STANDARD_RUN_NAME)
    run_dir = run_root / run_name
    resume_checkpoint = loese_resume_checkpoint(args)
    start_step, max_steps = trainings_start_und_ziel(args, run_dir)
    if not args.run_name:
        run_name = slug(naechster_run_name(run_root, start_step, max_steps))
        run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if max_steps <= start_step and not args.nur_pruefen:
        print("Status: LoRA-Training ist fuer diesen Lauf bereits abgeschlossen.")
        print(f"Interner Checkpoint: {start_step}")
        print("Weitere Trainingsrunde nur bewusst mit --weitere-runde starten.")
        return 0
    args.max_steps = max_steps

    primary_root = Path(args.dataset_root).expanduser().resolve()
    ready, ready_message = dataset_training_ready(primary_root)
    if not ready:
        print(ready_message, file=sys.stderr)
        print(f"Summary: {rel(primary_root / 'dataset_summary.json')}", file=sys.stderr)
        return 5
    extra_roots = [Path(item).expanduser().resolve() for item in args.zusatz_dataset_root]
    all_manifest_roots = manifest_roots(primary_root)
    for extra in extra_roots:
        all_manifest_roots.extend(manifest_roots(extra))
    all_manifest_roots = sorted(set(all_manifest_roots))
    if not all_manifest_roots:
        print(f"Kein gueltiges Dataset gefunden: {primary_root}", file=sys.stderr)
        return 2

    rows = lade_manifest_rows(all_manifest_roots)
    clip_report = pruefe_manifest_clips(rows, volle_hash_pruefung=args.volle_clip_hash_pruefung)
    mp3_roots = [Path(item).expanduser().resolve() for item in args.roh_audio_root] or [STANDARD_ROH_AUDIO]
    mp3_report = {"mp3_count": 0, "duplicate_hashes": {}, "duplicate_youtube_ids": {}}
    if not args.mp3_pruefung_aus:
        mp3_report = pruefe_mp3_duplikate(mp3_roots)

    cleaned_rows, skipped_rows = dedupliziere_rows(rows)
    dataset_for_training = primary_root
    if extra_roots:
        kombi_name = args.kombiniert_name or f"musicgen_training_{run_name}"
        dataset_for_training = schreibe_kombi_dataset(cleaned_rows, kombi_name)

    command = baue_trainingsbefehl(args, dataset_for_training, run_name)
    train_manifest_count = zaehle_manifest_rows(dataset_for_training, "train")
    valid_manifest_count = zaehle_manifest_rows(dataset_for_training, "valid")
    train_clip_limit = min(train_manifest_count, int(args.train_clips)) if int(args.train_clips) > 0 else train_manifest_count
    valid_clip_limit = min(valid_manifest_count, int(args.valid_clips)) if int(args.valid_clips) > 0 else valid_manifest_count

    duplikat_rows: list[dict[str, Any]] = []
    duplikat_rows.extend(flache_duplikate(mp3_report.get("duplicate_hashes", {}), "mp3_hash"))
    duplikat_rows.extend(flache_duplikate(mp3_report.get("duplicate_youtube_ids", {}), "mp3_youtube_id"))
    duplikat_rows.extend(flache_duplikate(clip_report.get("duplicate_paths", {}), "clip_pfad"))
    duplikat_rows.extend(flache_duplikate(clip_report.get("duplicate_source_windows", {}), "clip_quellfenster"))
    duplikat_rows.extend(flache_duplikate(clip_report.get("duplicate_audio_hashes", {}), "clip_audio_hash"))

    report = {
        "created_at": jetzt_utc(),
        "run_name": run_name,
        "status": "geprueft",
        "dataset_roots": [rel(path) for path in all_manifest_roots],
        "dataset_for_training": rel(dataset_for_training),
        "manifest_clip_count": len(rows),
        "deduplicated_clip_count": len(cleaned_rows),
        "train_manifest_count": train_manifest_count,
        "valid_manifest_count": valid_manifest_count,
        "train_clip_limit": train_clip_limit,
        "valid_clip_limit": valid_clip_limit,
        "skipped_duplicate_rows": len(skipped_rows),
        "mp3_report": mp3_report,
        "clip_report_summary": {
            "clip_count": clip_report["clip_count"],
            "duplicate_paths": len(clip_report["duplicate_paths"]),
            "duplicate_names": len(clip_report["duplicate_names"]),
            "duplicate_source_windows": len(clip_report["duplicate_source_windows"]),
            "duplicate_audio_hashes": len(clip_report["duplicate_audio_hashes"]),
            "errors": len(clip_report["errors"]),
        },
        "duplikate": duplikat_rows,
        "clip_fehler": clip_report["errors"],
        "uebersprungene_duplikate": skipped_rows,
        "training_command": command,
        "nur_pruefen": bool(args.nur_pruefen),
    }
    pruefung_path = run_dir / "pruefung.json"
    pruefung_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    hat_duplikate = bool(duplikat_rows)
    hat_fehler = bool(clip_report["errors"])
    modell_ok = modell_vorhanden(Path(args.model_dir).expanduser().resolve()) or args.remote_modell_erlauben
    resume_ok = args.ohne_resume or (resume_checkpoint is not None and resume_checkpoint.exists())
    effektive_batch_size = args.batch_size if args.batch_size > 0 else 1
    effektive_gradient_accumulation = args.gradient_accumulation if args.gradient_accumulation > 0 else 8

    if args.ausgabe == "voll":
        print("LoRA-Training")
        print("=============")
        print("LoRA:       LoRA")
        print(f"Ordner:     {run_name}")
        print(f"Dataset:    {rel(dataset_for_training)}")
        print(f"Clips:      Train {train_clip_limit}/{train_manifest_count} | Valid {valid_clip_limit}/{valid_manifest_count}")
        print(f"MP3s:       {mp3_report.get('mp3_count', 0)}")
        print(f"Duplikate:  {len(duplikat_rows)}")
        print(f"Clipfehler: {len(clip_report['errors'])}")
        print(f"Report:     {rel(pruefung_path)}")
    elif args.nur_pruefen:
        print("LoRA: LoRA")
        print(f"Clips: Train {train_clip_limit}/{train_manifest_count} | Steps: {start_step}->{max_steps}")

    if not modell_ok:
        print(f"Lokales Modell fehlt oder ist unvollstaendig: {args.model_dir}", file=sys.stderr)
        return 4
    if not resume_ok:
        print(f"Resume-Checkpoint nicht gefunden: {args.resume_from}", file=sys.stderr)
        return 4
    if (hat_duplikate or hat_fehler) and not args.duplikate_erlauben:
        print("Training wird nicht gestartet: Duplikate oder Clipfehler gefunden.", file=sys.stderr)
        print(f"Details: {rel(pruefung_path)}", file=sys.stderr)
        return 3
    if args.nur_pruefen:
        print("Nur Pruefung: Training wurde nicht gestartet.")
        if args.ausgabe == "voll":
            print("Befehl:")
            print(" ".join(subprocess.list2cmdline([part]) for part in command))
        else:
            print(f"Report: {rel(pruefung_path)}")
        return 0

    schreibe_training_plan(
        run_dir,
        status="running",
        dataset_root=dataset_for_training,
        start_step=start_step,
        target_step=max_steps,
        command=command,
    )
    returncode = fuehre_training_kompakt_aus(
        command,
        run_dir=run_dir,
        start_step=start_step,
        max_steps=max_steps,
        clip_count=train_clip_limit,
        batch_size=effektive_batch_size,
        gradient_accumulation=effektive_gradient_accumulation,
        voll=args.ausgabe == "voll",
    )
    schreibe_training_plan(
        run_dir,
        status="finished" if returncode == 0 else "interrupted_or_failed",
        dataset_root=dataset_for_training,
        start_step=max_steps if returncode == 0 else start_step,
        target_step=max_steps,
        command=command,
        returncode=returncode,
    )
    if returncode == 0:
        try:
            veroeffentliche_lora_stand(run_dir, dataset_for_training, max_steps)
        except RuntimeError as exc:
            print(f"Status: Training beendet, Adapter aber nicht freigegeben: {exc}", file=sys.stderr)
            return 7
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
