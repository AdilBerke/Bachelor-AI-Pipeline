#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import select
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Merkmale.audio_merkmale import extract_features

PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "daten" / "processed" / "lora_training"
DEFAULT_PIPELINE_ROOT = PROJECT_ROOT / "training" / "musicgen" / "pipeline_runs"
DEFAULT_BEST_ADAPTER = (
    PROJECT_ROOT
    / "training"
    / "musicgen"
    / "lora_training"
    / "adapter.pt"
)
DEFAULT_LORA_RUN = PROJECT_ROOT / "training" / "musicgen" / "lora_training"
DEFAULT_LORA_PLAN = DEFAULT_LORA_RUN / "training_plan.json"
DEFAULT_TRAINING_ENTRY = PROJECT_ROOT / "code" / "src" / "Training" / "musicgen_steuerung.py"
DEFAULT_DATASET_ENTRY = PROJECT_ROOT / "code" / "src" / "Dataset" / "datensatz.py"
DEFAULT_POSTPROCESSING_ENTRY = PROJECT_ROOT / "code" / "src" / "Training" / "audio_nachbearbeitung.py"
SPLITS = ("train", "valid", "test")
DEFAULT_RHYTHMUS_ANTEIL_PROZENT = 5.0
DEFAULT_MIN_RHYTHMUS_SEKUNDEN = 90.0


def adapter_fuer_audio_freigegeben(adapter_path: Path) -> bool:
    if not adapter_path.is_file():
        return False
    try:
        adapter_path.resolve().relative_to(DEFAULT_LORA_RUN.resolve())
    except ValueError:
        return True
    if not DEFAULT_LORA_PLAN.is_file():
        return False
    stand_path = DEFAULT_LORA_RUN / "stand.json"
    if stand_path.is_file():
        stand = read_json_optional(stand_path)
        if stand and stand.get("status") != "freigegeben":
            best = stand.get("best_adapter")
            if best:
                best_path = Path(str(best)).expanduser()
                if not best_path.is_absolute():
                    best_path = PROJECT_ROOT / best_path
                try:
                    return adapter_path.resolve() == best_path.resolve()
                except Exception:
                    return False
            return False
    try:
        plan = json.loads(DEFAULT_LORA_PLAN.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return plan.get("status") == "finished" and int(plan.get("returncode") or 0) == 0


BAD_REVIEW_WORDS = (
    "schlecht",
    "kein ton",
    "ohne ton",
    "kippt",
    "signalton",
    "explosion",
    "rauschen",
    "monoton",
    "nicht gut",
    "nicht bewerten",
    "fehlgeschlagen",
    "uebersteuert",
    "übersteuert",
)
GOOD_REVIEW_WORDS = (
    "gute audio",
    "sehr gute audio",
    "top audio",
    "gut eingesetzt",
    "guter uebergang",
    "guter übergang",
    "guter rhythmus",
    "gute harmonie",
    "hat mir sehr gefallen",
)


@dataclass
class StageResult:

    name: str
    status: str
    started_at: str
    finished_at: str
    command: List[str]
    log_path: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MusicGen-Audio-Pipeline ohne Crawler/Suche.")
    parser.add_argument(
        "--modus",
        choices=("audio_generieren", "pruefung", "end_audio", "komplett"),
        default="audio_generieren",
        help=(
            "audio_generieren: frische MusicGen-Abschnitte erzeugen. "
            "pruefung: Features/Validierung/Audit. "
            "end_audio: alte Loop-Pipeline aus Bewertungsclips. "
            "komplett: optional auch LoRA-Training und Review."
        ),
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--pipeline-root", default=str(DEFAULT_PIPELINE_ROOT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--nur-plan", action="store_true", help="Nur Plan/Commands schreiben, nichts ausfuehren.")
    parser.add_argument(
        "--sparmodus",
        action="store_true",
        help="Reduziert Kandidatenzahl und setzt ein hartes Generierungslimit.",
    )
    parser.add_argument(
        "--freigabe-langer-lauf",
        action="store_true",
        help="Erlaubt echte Generierung ueber 5 Minuten. Ohne diese Freigabe wird nur geplant.",
    )
    parser.add_argument(
        "--live-status-sekunden",
        type=float,
        default=1.0,
        help="Abstand fuer Live-Fortschritt im Terminal. Standard: 1 Sekunde. 0 deaktiviert die Zusatzanzeige.",
    )
    parser.add_argument(
        "--debug-ausgabe",
        action="store_true",
        help="Zeigt rohe MusicGen-Ausgabe zusaetzlich im Terminal. Standard: nur Logdatei + Live-Prozent.",
    )
    parser.add_argument(
        "--stufen",
        default="",
        help=(
            "Optionale kommaseparierte Stufen: struktur,features,validate,audit,"
            "train,review,final_audio,audio_generieren"
        ),
    )

    parser.add_argument("--feature-limit", type=int, default=0, help="0 = alle Manifest-Eintraege.")
    parser.add_argument("--deep-audit", action="store_true", help="Audit liest alle WAV-Samples erneut.")

    parser.add_argument("--training-aktivieren", action="store_true")
    parser.add_argument("--training-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--vram-limit-fraction", type=float, default=0.80)
    parser.add_argument("--training-run-name", default="")
    parser.add_argument("--best-adapter", default=str(DEFAULT_BEST_ADAPTER))
    parser.add_argument(
        "--review-adapter-erlauben",
        action="store_true",
        help="Erlaubt einen bewertung_offen-LoRA-Kandidaten nur fuer Test-/Review-Audios.",
    )

    parser.add_argument("--review-aktivieren", action="store_true")
    parser.add_argument("--review-run-name", default="")
    parser.add_argument("--prompt-set", choices=("lofi_standard", "jazz_lofi"), default="lofi_standard")

    parser.add_argument("--dauer", default="20m", help="Finale Audio-Laenge, z.B. 20m, 30m, 1h, 3h.")
    parser.add_argument("--genre", default="Lofi", help="Genre fuer neu generierte Longform, z.B. Jazz Lofi.")
    parser.add_argument("--stimmung", default="", help="Optionale Stimmung fuer die neue Longform.")
    parser.add_argument("--instrumente", default="", help="Optionale Instrumente fuer die neue Longform.")
    parser.add_argument("--abschnitt-sekunden", type=float, default=30.0)
    parser.add_argument(
        "--kandidaten-pro-abschnitt",
        type=int,
        default=3,
        help="Drei Kandidaten je Musikblock sind der Qualitaetsstandard.",
    )
    parser.add_argument(
        "--max-generierte-kandidaten",
        type=int,
        default=0,
        help="0 = kein hartes Limit. Stoppt sonst nach dieser Anzahl echter MusicGen-Kandidaten.",
    )
    parser.add_argument("--max-abschnitte", type=int, default=0, help="0 = alle benoetigten Abschnitte.")
    parser.add_argument("--fallback-pool", action="store_true", help="Erlaubt alte Clips als Notfallquelle.")
    parser.add_argument("--final-name", default="")
    parser.add_argument("--loop-source", action="append", default=[], help="Audio-Datei oder Ordner. Wiederholbar.")
    parser.add_argument("--seed", type=int, default=0, help="0 = jedes Mal neuer Zufall.")
    parser.add_argument("--ziel-bpm", type=float, default=78.0)
    parser.add_argument("--bpm-toleranz", type=float, default=4.0)
    parser.add_argument(
        "--uebergang-bpm-toleranz",
        type=float,
        default=2.5,
        help="Maximal bevorzugter BPM-Unterschied zwischen zwei benachbarten Clips.",
    )
    parser.add_argument("--bpm-pruefung-aktiv", dest="bpm_pruefung_aktiv", action="store_true", default=True)
    parser.add_argument("--bpm-pruefung-deaktivieren", dest="bpm_pruefung_aktiv", action="store_false")
    parser.add_argument("--crossfade-sekunden", type=float, default=3.0)
    parser.add_argument("--fade-in-sekunden", type=float, default=5.0)
    parser.add_argument("--fade-out-sekunden", type=float, default=5.0)
    parser.add_argument(
        "--anschluss-conditioning-aktiv",
        dest="anschluss_conditioning_aktiv",
        action="store_true",
        default=False,
        help="Nutzt das Ende des vorherigen Clips als Melody-Referenz fuer den naechsten Clip.",
    )
    parser.add_argument("--anschluss-conditioning-deaktivieren", dest="anschluss_conditioning_aktiv", action="store_false")
    parser.add_argument("--anschluss-sekunden", type=float, default=8.0)
    parser.add_argument(
        "--genre-melodie-conditioning-aktiv",
        dest="genre_melodie_conditioning_aktiv",
        action="store_true",
        default=False,
        help="Nutzt einen echten Referenzclip des Zielgenres als Melody-Vorgabe (generate_with_chroma).",
    )
    parser.add_argument(
        "--genre-melodie-conditioning-deaktivieren",
        dest="genre_melodie_conditioning_aktiv",
        action="store_false",
    )
    parser.add_argument("--sekunden-pruefung-aktiv", dest="sekunden_pruefung_aktiv", action="store_true", default=True)
    parser.add_argument("--sekunden-pruefung-deaktivieren", dest="sekunden_pruefung_aktiv", action="store_false")
    parser.add_argument("--min-sekunden-rms-db", type=float, default=-52.0)
    parser.add_argument("--temperature", type=float, default=0.72)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--cfg-coef", type=float, default=4.0)
    parser.add_argument(
        "--block-sekunden",
        type=float,
        default=0.0,
        help="0 = automatisch prozentual berechnen; positiver Wert erzwingt eine feste Blockdauer.",
    )
    parser.add_argument(
        "--rhythmus-anteil-prozent",
        type=float,
        default=DEFAULT_RHYTHMUS_ANTEIL_PROZENT,
        help="Anteil der Gesamtdauer pro Rhythmusblock. 5 Prozent ergeben bei 1h genau 3 Minuten.",
    )
    parser.add_argument(
        "--min-rhythmus-sekunden",
        type=float,
        default=DEFAULT_MIN_RHYTHMUS_SEKUNDEN,
        help="Kuerzeste erlaubte Rhythmusdauer, sofern die gesamte Audio nicht kuerzer ist.",
    )
    parser.add_argument("--block-variation-sekunden", type=float, default=0.0)
    parser.add_argument(
        "--loop-crossfade-sekunden",
        type=float,
        default=0.0,
        help="0 = automatisch ein Beat; betrifft nur interne Wiederholungen eines Loop-Blocks.",
    )
    parser.add_argument(
        "--tempo-variation-prozent",
        type=float,
        default=1.5,
        help="Maximale pitch-erhaltende Tempoabweichung innerhalb eines Loop-Blocks.",
    )
    parser.add_argument("--tempo-phase-sekunden", type=float, default=60.0)
    parser.add_argument("--block-looping-aktiv", dest="block_looping_aktiv", action="store_true", default=True)
    parser.add_argument("--block-looping-deaktivieren", dest="block_looping_aktiv", action="store_false")
    parser.add_argument(
        "--kontinuierliche-bloecke-aktiv",
        dest="kontinuierliche_bloecke_aktiv",
        action="store_true",
        default=False,
        help="Erzeugt jeden Rhythmusblock als fortgesetzte MusicGen-Phrase.",
    )
    parser.add_argument(
        "--kontinuierliche-bloecke-deaktivieren",
        dest="kontinuierliche_bloecke_aktiv",
        action="store_false",
    )
    parser.add_argument("--erweiterungs-schritt-sekunden", type=float, default=12.0)
    parser.add_argument(
        "--genre-pruefung-aktiv",
        dest="genre_pruefung_aktiv",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--genre-pruefung-deaktivieren",
        dest="genre_pruefung_aktiv",
        action="store_false",
    )
    parser.add_argument(
        "--clap-modell-pfad",
        default=str(
            PROJECT_ROOT
            / "daten"
            / "modelle"
            / "audio_analyse"
            / "clap_htsat_unfused"
        ),
    )
    parser.add_argument(
        "--genre-referenz-ordner",
        default=str(
            PROJECT_ROOT
            / "training"
            / "bewertungen"
            / "musicgen"
            / "lora_review_001"
        ),
    )
    parser.add_argument("--min-genre-aehnlichkeit", type=float, default=0.62)
    parser.add_argument("--min-genre-abstand", type=float, default=0.02)
    parser.add_argument("--min-qualitaets-abstand", type=float, default=0.0)
    parser.add_argument(
        "--mp3-referenz-pruefung-aktiv",
        dest="mp3_referenz_pruefung_aktiv",
        action="store_true",
        default=True,
        help="Vergleicht Kandidaten technisch mit guten MP3-Trainingsclips.",
    )
    parser.add_argument(
        "--mp3-referenz-pruefung-deaktivieren",
        dest="mp3_referenz_pruefung_aktiv",
        action="store_false",
    )
    parser.add_argument(
        "--referenz-dataset-root",
        default=str(PROJECT_ROOT / "daten" / "processed" / "lora_training"),
    )
    parser.add_argument("--referenzen-pro-genre", type=int, default=20)
    parser.add_argument("--referenz-score-limit", type=float, default=18.0)
    parser.add_argument(
        "--referenzvergleich-aktiv",
        dest="referenzvergleich_aktiv",
        action="store_true",
        default=True,
        help="Nach fertiger Audio automatisch MP3-Referenzvergleich schreiben.",
    )
    parser.add_argument(
        "--referenzvergleich-deaktivieren",
        dest="referenzvergleich_aktiv",
        action="store_false",
    )
    parser.add_argument("--referenzvergleich-segment-sekunden", type=float, default=30.0)
    parser.add_argument("--kein-mp3", action="store_true")
    parser.add_argument(
        "--github-push",
        dest="github_push",
        action="store_true",
        default=True,
        help="Pusht die finale Audio nach erfolgreicher Generierung isoliert nach main.",
    )
    parser.add_argument(
        "--kein-github-push",
        dest="github_push",
        action="store_false",
        help="Laesst die finale Audio ausschliesslich lokal.",
    )
    parser.add_argument(
        "--postprocessing",
        action="store_true",
        help="Nach der Generierung eine optionale Audio-Nachbearbeitung starten.",
    )
    parser.add_argument("--denoise", action="store_true", help="DeepFilterNet nutzen, falls lokal vorhanden.")
    parser.add_argument(
        "--denoise-max-dauer-sec",
        type=float,
        default=180.0,
        help="DeepFilterNet nur bis zu dieser Audiolaenge nutzen. 0 = kein Limit.",
    )
    parser.add_argument(
        "--denoise-lange-audio-erlauben",
        action="store_true",
        help="DeepFilterNet auch fuer lange Audios erzwingen. Kann sehr langsam sein.",
    )
    parser.add_argument("--mastering", action="store_true", help="Matchering nutzen, falls lokal vorhanden.")
    parser.add_argument("--mastering-reference", default="", help="Referenz-Audio fuer optionales Mastering.")
    parser.add_argument("--audio-sr", action="store_true", help="Experimentelle AudioSR-Stufe aktivieren, falls vorhanden.")
    parser.add_argument(
        "--postprocessing-intensitaet",
        choices=("vorsichtig", "normal", "stark"),
        default="vorsichtig",
        help="Staerke der Nachbearbeitung. Standard: vorsichtig.",
    )
    parser.add_argument(
        "--postprocessing-nur-pruefen",
        action="store_true",
        help="Nur Analyse-Reports schreiben, keine bearbeitete Audio erzeugen.",
    )
    return parser.parse_args()


def default_stages(mode: str) -> List[str]:
    if mode == "audio_generieren":
        return ["audio_generieren"]
    if mode == "pruefung":
        return ["struktur", "features", "validate", "audit"]
    if mode == "komplett":
        return ["struktur", "features", "validate", "audit", "train", "review", "final_audio"]
    return ["struktur", "features", "validate", "audit", "final_audio"]


def planned_stages(args: argparse.Namespace) -> List[str]:
    if args.stufen.strip():
        return [item.strip() for item in args.stufen.split(",") if item.strip()]
    return default_stages(args.modus)


def run_subprocess(stage: str, command: List[str], run_dir: Path, dry_run: bool) -> StageResult:
    started = now()
    log_path = run_dir / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nStufe: {stage}", flush=True)
    if dry_run:
        log_path.write_text(
            "Nur Plan: Befehl wurde nicht ausgefuehrt.\n"
            + "Befehl: "
            + " ".join(subprocess.list2cmdline([part]) for part in command)
            + "\n",
            encoding="utf-8",
        )
        return StageResult(stage, "planned", started, now(), command, rel(log_path))

    with log_path.open("w", encoding="utf-8") as log:
        log.write("Befehl: " + " ".join(subprocess.list2cmdline([part]) for part in command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
        log.write(f"\n[returncode] {code}\n")
    status = "ok" if code == 0 else "failed"
    result = StageResult(stage, status, started, now(), command, rel(log_path), {"returncode": code})
    if code != 0:
        raise RuntimeError(f"Stufe {stage} fehlgeschlagen mit Return-Code {code}. Log: {log_path}")
    return result


def read_json_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_duration_seconds(value: str) -> float:
    text = str(value).strip().lower().replace(",", ".")
    if not text:
        raise ValueError("Dauer fehlt.")
    unit = text[-1]
    number_text = text[:-1] if unit in {"s", "m", "h"} else text
    number = float(number_text)
    if unit == "h":
        return number * 3600.0
    if unit == "m":
        return number * 60.0
    return number


def resolve_rhythm_block_seconds(
    duration_sec: float,
    requested_block_sec: float,
    percentage: float,
    minimum_sec: float,
) -> float:

    if duration_sec <= 0:
        raise ValueError("Die Audio-Dauer muss groesser als 0 sein.")
    minimum = max(1.0, float(minimum_sec))
    if requested_block_sec > 0:
        target = max(minimum, float(requested_block_sec))
    else:
        if percentage <= 0 or percentage > 100:
            raise ValueError("rhythmus-anteil-prozent muss zwischen 0 und 100 liegen.")
        target = max(minimum, duration_sec * float(percentage) / 100.0)
    return min(duration_sec, target)


def estimate_rhythm_blocks(duration_sec: float, target_sec: float, minimum_sec: float) -> int:

    if duration_sec <= 0:
        return 0
    minimum = max(1.0, min(float(minimum_sec), duration_sec))
    maximum_count = max(1, int(math.floor(duration_sec / minimum)))
    ideal_count = max(1, int(round(duration_sec / max(1.0, target_sec))))
    return min(maximum_count, ideal_count)


def estimate_sections(duration_sec: float, section_sec: float, crossfade_sec: float) -> int:
    if duration_sec <= 0:
        return 0
    if section_sec <= 0:
        raise ValueError("abschnitt-sekunden muss groesser als 0 sein.")
    if duration_sec <= section_sec:
        return 1
    useful = max(1.0, section_sec - max(0.0, crossfade_sec))
    return int(math.ceil((duration_sec - max(0.0, crossfade_sec)) / useful))


def automatic_candidate_limit(estimated_sections: int, candidates_per_section: int) -> int:

    sections = max(1, int(estimated_sections))
    candidates = max(1, int(candidates_per_section))
    planned = sections * candidates
    reserve = max(sections * 4, planned, 12)
    return planned + reserve


def audio_stage_requested(stages: List[str]) -> bool:
    return "audio_generieren" in stages


def apply_safety_mode(args: argparse.Namespace, stages: List[str]) -> Dict[str, Any]:
    duration_sec = parse_duration_seconds(args.dauer)
    crossfade_sec = min(max(0.0, args.crossfade_sekunden), max(1.0, args.abschnitt_sekunden * 0.20))
    requested_block_seconds = float(args.block_sekunden)
    resolved_block_seconds = resolve_rhythm_block_seconds(
        duration_sec,
        requested_block_seconds,
        args.rhythmus_anteil_prozent,
        args.min_rhythmus_sekunden,
    )
    rhythm_block_count = estimate_rhythm_blocks(
        duration_sec,
        resolved_block_seconds,
        args.min_rhythmus_sekunden,
    )
    effective_rhythm_block_seconds = duration_sec / max(1, rhythm_block_count)
    if args.block_looping_aktiv or args.kontinuierliche_bloecke_aktiv:
        estimated = rhythm_block_count
    else:
        estimated = estimate_sections(duration_sec, args.abschnitt_sekunden, crossfade_sec)
    original_candidates = args.kandidaten_pro_abschnitt

    if args.sparmodus:
        args.kandidaten_pro_abschnitt = max(1, min(args.kandidaten_pro_abschnitt, 2))
        if not args.max_generierte_kandidaten:
            args.max_generierte_kandidaten = automatic_candidate_limit(
                estimated,
                args.kandidaten_pro_abschnitt,
            )

    total_candidates = estimated * max(1, args.kandidaten_pro_abschnitt)
    candidate_reserve = (
        max(0, int(args.max_generierte_kandidaten) - total_candidates)
        if args.max_generierte_kandidaten
        else None
    )
    long_run_blocked = (
        audio_stage_requested(stages)
        and duration_sec > 300.0
        and not args.freigabe_langer_lauf
        and not args.review_adapter_erlauben
        and not args.nur_plan
    )
    if long_run_blocked:
        args.nur_plan = True

    return {
        "duration_sec": duration_sec,
        "duration_input": args.dauer,
        "audio_stage_requested": audio_stage_requested(stages),
        "estimated_sections": estimated,
        "section_seconds": args.abschnitt_sekunden,
        "crossfade_seconds": crossfade_sec,
        "rhythm_block_mode": "percentage" if requested_block_seconds <= 0 else "fixed",
        "rhythm_share_percent": args.rhythmus_anteil_prozent,
        "minimum_rhythm_seconds": args.min_rhythmus_sekunden,
        "rhythm_target_seconds": resolved_block_seconds,
        "rhythm_block_seconds": effective_rhythm_block_seconds,
        "rhythm_block_count": rhythm_block_count,
        "continuous_musicgen_blocks": bool(args.kontinuierliche_bloecke_aktiv),
        "candidates_per_section_original": original_candidates,
        "candidates_per_section_effective": args.kandidaten_pro_abschnitt,
        "estimated_candidate_generations": total_candidates,
        "candidate_generation_reserve": candidate_reserve,
        "max_generated_candidates": args.max_generierte_kandidaten or None,
        "sparmodus": bool(args.sparmodus),
        "long_run_release": bool(args.freigabe_langer_lauf),
        "long_run_blocked_to_plan": bool(long_run_blocked),
        "hinweis": (
            "Langer Lauf wurde aus Sicherheitsgruenden in einen Planlauf umgewandelt. "
            "Zum echten Start --freigabe-langer-lauf angeben."
            if long_run_blocked
            else "Sicherheitsplan erstellt."
        ),
    }


def write_generation_budget(run_dir: Path, budget: Dict[str, Any]) -> None:
    path = run_dir / "audio_generierung" / "generierungs_plan.json"
    write_json(path, {"created_at": now(), **budget})
    md = [
        "# Audio-Generierungsplan",
        "",
        f"- Dauer: {budget['duration_input']} ({budget['duration_sec']:.1f}s)",
        f"- Geschaetzte Abschnitte: {budget['estimated_sections']}",
        (
            f"- Rhythmusblock: {budget['rhythm_block_seconds']:.1f}s "
            f"({budget['rhythm_share_percent']:.1f}% der Gesamtdauer, "
            f"Minimum {budget['minimum_rhythm_seconds']:.1f}s)"
        ),
        f"- Rhythmusbloecke: {budget['rhythm_block_count']}",
        f"- Kandidaten pro Abschnitt: {budget['candidates_per_section_effective']}",
        f"- Geschaetzte MusicGen-Generierungen: {budget['estimated_candidate_generations']}",
        f"- Kandidatenreserve: {budget['candidate_generation_reserve']}",
        f"- Hartes Kandidatenlimit: {budget['max_generated_candidates'] or 'kein Limit'}",
        f"- Sparmodus: {budget['sparmodus']}",
        f"- Langer Lauf freigegeben: {budget['long_run_release']}",
        f"- Hinweis: {budget['hinweis']}",
        "",
    ]
    (run_dir / "audio_generierung" / "generierungs_plan.md").write_text("\n".join(md), encoding="utf-8")


def yes_no(value: Any) -> str:
    return "ja" if bool(value) else "nein"


def percent_from_progress(progress: Dict[str, Any]) -> Optional[float]:
    value = progress.get("progress_percent", progress.get("percent"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def progress_value(progress: Dict[str, Any], key: str, fallback: Any = "?") -> Any:
    value = progress.get(key)
    return fallback if value in (None, "") else value


def format_live_dashboard(
    progress: Dict[str, Any],
    attempt: int,
    max_attempts: int,
    budget: Dict[str, Any],
    waiting: bool = False,
) -> str:
    percent = percent_from_progress(progress)
    percent_text = "  0.00%" if waiting else "  ?.??%"
    if percent is not None:
        percent_text = f"{percent:6.2f}%"

    accepted = progress.get("accepted_clips", "?")
    total = progress.get("total_sections") or budget.get("estimated_sections") or "?"
    rejected = progress.get("rejected_clips", 0)
    candidate = progress_value(progress, "current_candidate", "-")
    total_candidates = progress.get("total_candidates") or budget.get("candidates_per_section_effective") or "?"
    status = "warte auf ersten Fortschritt" if waiting else progress.get("status", "running")
    message = progress.get("message") or ("MusicGen startet..." if waiting else "")
    return "\n".join(
        [
            "Live-Fortschritt",
            "-----------------",
            f"Status:      {status}",
            f"Versuch:     {attempt}/{max_attempts}",
            f"Gesamt:      {percent_text}",
            f"Clips:       {accepted} / {total}",
            f"Kandidat:    {candidate} / {total_candidates}",
            f"Verworfen:   {rejected}",
            f"Nachricht:   {message}",
        ]
    )


def print_live_block(block: str, previous_lines: int) -> int:
    lines = block.splitlines()
    if previous_lines and sys.stdout.isatty():
        sys.stdout.write(f"\033[{previous_lines}F")
    terminal_width = shutil.get_terminal_size((110, 20)).columns
    for line in lines:
        if sys.stdout.isatty():
            sys.stdout.write("\r\033[K" + line[:terminal_width] + "\n")
        else:
            sys.stdout.write(line[:terminal_width] + "\n")
    sys.stdout.flush()
    return len(lines)


def finish_live_block(previous_lines: int) -> None:
    if previous_lines > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()


def final_audio_from_results(results: List[StageResult]) -> Optional[str]:
    for result in reversed(results):
        detail = result.detail or {}
        final_audio = detail.get("final_audio")
        if final_audio:
            return str(final_audio)
    return None


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_postprocessing_command(final_audio: str, args: argparse.Namespace) -> List[str]:
    input_path = project_path(final_audio)
    output_dir = input_path.parent / "postprocessing"
    command = [
        str(PYTHON),
        "code/src/Training/audio_nachbearbeitung.py",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--intensitaet",
        args.postprocessing_intensitaet,
        "--nur-lokale-tools",
    ]
    if args.denoise:
        command.append("--denoise")
        command.extend(["--denoise-max-dauer-sec", str(args.denoise_max_dauer_sec)])
    if args.denoise_lange_audio_erlauben:
        command.append("--denoise-lange-audio-erlauben")
    if args.mastering:
        command.append("--mastering")
    if args.mastering_reference:
        command.extend(["--mastering-reference", args.mastering_reference])
    if args.audio_sr:
        command.append("--audio-sr")
    if args.postprocessing_nur_pruefen:
        command.append("--nur-pruefen")
    return command


def run_postprocessing_stage(
    final_audio: str,
    args: argparse.Namespace,
    run_dir: Path,
    dry_run: bool,
) -> StageResult:
    command = build_postprocessing_command(final_audio, args)
    output_dir = project_path(value_after(command, "--output-dir") or "")
    result = run_subprocess("postprocessing", command, run_dir, dry_run)
    report_path = output_dir / "postprocessing_report.json"
    report = read_json_optional(report_path)
    final_post_audio = report.get("output_mp3") or report.get("output_wav")
    detail = {
        "output_dir": rel(output_dir),
        "report": rel(report_path),
        "postprocessing_intensitaet": args.postprocessing_intensitaet,
        "nur_pruefen": bool(args.postprocessing_nur_pruefen),
    }
    if final_post_audio:
        detail["final_audio"] = str(final_post_audio)
    elif dry_run:
        detail["planned"] = True
        detail["final_audio"] = rel(output_dir / "lange_audio_postprocessed.mp3")
    result.detail = {**(result.detail or {}), **detail}
    return result


def build_referenzvergleich_command(final_audio: str, args: argparse.Namespace, run_dir: Path) -> List[str]:
    input_path = project_path(final_audio)
    output_dir = run_dir / "referenzvergleich"
    return [
        str(PYTHON),
        "code/src/Training/referenz_vergleich.py",
        "--referenz-root",
        str(Path(args.referenz_dataset_root).expanduser().resolve()),
        "--generiert-root",
        str(input_path),
        "--ausgabe-dir",
        str(output_dir),
        "--referenzen-pro-genre",
        str(args.referenzen_pro_genre),
        "--segment-sekunden",
        str(args.referenzvergleich_segment_sekunden),
    ]


def run_referenzvergleich_stage(
    final_audio: str,
    args: argparse.Namespace,
    run_dir: Path,
    dry_run: bool,
) -> StageResult:
    command = build_referenzvergleich_command(final_audio, args, run_dir)
    result = run_subprocess("referenzvergleich", command, run_dir, dry_run)
    output_dir = run_dir / "referenzvergleich"
    summary_path = output_dir / "zusammenfassung.json"
    summary = read_json_optional(summary_path)
    detail = {
        "output_dir": rel(output_dir),
        "summary": rel(summary_path),
        "vergleich": rel(output_dir / "vergleich.csv"),
        "diagramm_daten": rel(output_dir / "diagramm_daten.csv"),
        "bericht": rel(output_dir / "bericht.txt"),
    }
    if summary:
        detail["score_median"] = summary.get("score_median")
        detail["score_mittelwert"] = summary.get("score_mittelwert")
        detail["haeufigste_probleme"] = summary.get("haeufigste_probleme", [])
    result.detail = {**(result.detail or {}), **detail}
    return result


def print_pipeline_finish(report: Dict[str, Any], results: List[StageResult], run_dir: Path) -> None:
    status = str(report.get("status") or "unbekannt")
    print("", flush=True)
    print("Abschluss", flush=True)
    print("---------", flush=True)
    print(f"Status: {status}", flush=True)
    if report.get("error"):
        print(f"Fehler: {report['error']}", flush=True)
    final_audio = final_audio_from_results(results)
    if final_audio:
        print(f"Audio:  {final_audio}", flush=True)
    print(f"Report: {rel(run_dir / 'pipeline_report.json')}", flush=True)


def value_after(command: List[str], flag: str) -> Optional[str]:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def replace_or_append_flag(command: List[str], flag: str, value: str) -> List[str]:
    updated = list(command)
    try:
        index = updated.index(flag)
    except ValueError:
        updated.extend([flag, value])
        return updated
    if index + 1 < len(updated):
        updated[index + 1] = value
    else:
        updated.append(value)
    return updated


def audio_output_dir(command: List[str]) -> Path:
    name = value_after(command, "--name") or "audio"
    return PROJECT_ROOT / "training" / "ausgaben" / "musicgen_generiert" / name


def audio_final_exists(command: List[str], output_dir: Path) -> bool:
    wav_path = output_dir / "lange_audio.wav"
    mp3_path = output_dir / "lange_audio.mp3"
    info_path = output_dir / "finale_audio_info.json"
    needs_mp3 = "--kein-mp3" not in command
    if not wav_path.exists() or (needs_mp3 and not mp3_path.exists()):
        return False
    info = read_json_optional(info_path)
    return bool(info) or wav_path.stat().st_size > 0


def run_audio_with_resume(
    stage: str,
    command: List[str],
    run_dir: Path,
    dry_run: bool,
    live_interval_sec: float = 5.0,
    debug_output: bool = False,
    args: Optional[argparse.Namespace] = None,
    budget: Optional[Dict[str, Any]] = None,
) -> StageResult:

    started = now()
    output_dir = audio_output_dir(command)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if debug_output:
        print(f"\n[{stage}] {' '.join(command)}", flush=True)

    if dry_run:
        log_path = log_dir / f"{stage}.log"
        log_path.write_text("Nur Plan: Befehl wurde nicht ausgefuehrt.\n", encoding="utf-8")
        final_audio = output_dir / ("lange_audio.wav" if "--kein-mp3" in command else "lange_audio.mp3")
        return StageResult(
            stage,
            "planned",
            started,
            now(),
            command,
            rel(log_path),
            {
                "planned": True,
                "output_dir": rel(output_dir),
                "final_audio": rel(final_audio),
            },
        )

    max_attempts = 8
    attempts: List[Dict[str, Any]] = []
    last_log_path = log_dir / f"{stage}.log"

    stall_timeout_sec = 210.0
    live_interval_sec = max(0.0, float(live_interval_sec))
    if args is None:
        raise RuntimeError("Interner Fehler: Pipeline-Argumente fuer Live-Status fehlen.")
    if budget is None:
        budget = apply_safety_mode(args, [stage])

    for attempt in range(1, max_attempts + 1):
        attempt_command = command
        progress_before = read_json_optional(output_dir / "fortschritt.json")
        accepted_before = int(progress_before.get("accepted_clips") or 0)
        last_log_path = log_dir / f"{stage}_versuch_{attempt:02d}.log"
        if debug_output:
            print(
                f"\n[{stage}] Versuch {attempt}/{max_attempts} "
                f"(bereits akzeptiert: {accepted_before}, "
                f"Kandidaten pro Abschnitt: {value_after(attempt_command, '--kandidaten-pro-abschnitt')})",
                flush=True,
            )

        with last_log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                attempt_command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            assert process.stdout is not None
            stdout_fd = process.stdout.fileno()
            os.set_blocking(stdout_fd, False)
            progress_path = output_dir / "fortschritt.json"
            last_activity = time.monotonic()
            last_live_print = 0.0
            live_block_lines = 0
            last_progress_mtime = progress_path.stat().st_mtime if progress_path.exists() else 0.0
            last_progress_signature = (
                json.dumps(progress_before, sort_keys=True, ensure_ascii=False)
                if progress_before
                else ""
            )
            waiting_printed = False
            progress_changed_during_attempt = False
            while True:
                ready, _, _ = select.select([stdout_fd], [], [], 1.0)
                if ready:
                    try:
                        chunk = os.read(stdout_fd, 4096)
                    except BlockingIOError:
                        chunk = b""
                    if chunk:
                        text = chunk.decode("utf-8", errors="replace")
                        if debug_output:
                            if live_block_lines:
                                finish_live_block(live_block_lines)
                                live_block_lines = 0
                            print(text, end="")
                        log.write(text)
                        last_activity = time.monotonic()
                    elif process.poll() is not None:
                        break
                if process.poll() is not None:
                    break
                now_monotonic = time.monotonic()
                if progress_path.exists():
                    current_mtime = progress_path.stat().st_mtime
                    if current_mtime != last_progress_mtime:
                        last_progress_mtime = current_mtime
                        last_activity = time.monotonic()
                        progress_changed_during_attempt = True
                    progress_now = read_json_optional(progress_path)
                    progress_signature = json.dumps(progress_now, sort_keys=True, ensure_ascii=False)
                    should_print_live = (
                        live_interval_sec > 0
                        and progress_now
                        and (
                            progress_signature != last_progress_signature
                            or (sys.stdout.isatty() and now_monotonic - last_live_print >= live_interval_sec)
                        )
                    )
                    if should_print_live:
                        live_block = format_live_dashboard(
                            progress_now,
                            attempt,
                            max_attempts,
                            budget,
                        )
                        live_block_lines = print_live_block(live_block, live_block_lines)
                        if progress_signature != last_progress_signature:
                            log.write(live_block + "\n\n")
                        last_live_print = now_monotonic
                        last_progress_signature = progress_signature
                elif (
                    live_interval_sec > 0
                    and now_monotonic - last_live_print >= live_interval_sec
                    and (sys.stdout.isatty() or not waiting_printed)
                ):
                    live_block = format_live_dashboard(
                        {},
                        attempt,
                        max_attempts,
                        budget,
                        waiting=True,
                    )
                    live_block_lines = print_live_block(live_block, live_block_lines)
                    last_live_print = now_monotonic
                    waiting_printed = True
                if time.monotonic() - last_activity > stall_timeout_sec:
                    finish_live_block(live_block_lines)
                    live_block_lines = 0
                    log.write(
                        f"\n[watchdog] Keine Ausgabe/Fortschritt seit "
                        f"{stall_timeout_sec:.0f}s. Prozess wird beendet.\n"
                    )
                    print(
                        f"[{stage}] Watchdog: keine Ausgabe/Fortschritt seit "
                        f"{stall_timeout_sec:.0f}s. Prozess wird beendet.",
                        flush=True,
                    )
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
            code = process.wait()
            finish_live_block(live_block_lines)
            log.write(f"\n[returncode] {code}\n")

        progress_after = read_json_optional(output_dir / "fortschritt.json")
        accepted_after = int(progress_after.get("accepted_clips") or 0)
        native_start_crash = (
            code == -11
            and not progress_changed_during_attempt
            and accepted_after == accepted_before
        )
        attempt_info = {
            "attempt": attempt,
            "returncode": code,
            "log_path": rel(last_log_path),
            "accepted_before": accepted_before,
            "accepted_after": accepted_after,
            "progress_percent": progress_after.get("progress_percent"),
            "candidates_per_section": value_after(attempt_command, "--kandidaten-pro-abschnitt"),
            "progress_changed": progress_changed_during_attempt,
            "failure_class": "native_model_start_crash" if native_start_crash else None,
        }
        attempts.append(attempt_info)

        if code == 0 and audio_final_exists(attempt_command, output_dir):
            return StageResult(
                stage,
                "ok",
                started,
                now(),
                attempt_command,
                rel(last_log_path),
                {
                    "returncode": code,
                    "attempts": attempts,
                    "output_dir": rel(output_dir),
                    "final_audio": rel(output_dir / "lange_audio.mp3")
                    if "--kein-mp3" not in command
                    else rel(output_dir / "lange_audio.wav"),
                },
            )

        if attempt < max_attempts:
            if native_start_crash:
                wait_seconds = min(15, 4 + attempt * 2)
                print(
                    f"[{stage}] GPU-Modellstart fehlgeschlagen (-11). "
                    f"Kandidatenzahl bleibt unveraendert; neuer Versuch in "
                    f"{wait_seconds}s.",
                    flush=True,
                )
                time.sleep(wait_seconds)
            else:
                print(
                    f"[{stage}] Abbruch erkannt. Fortsetzung am letzten guten Abschnitt "
                    f"({accepted_after} akzeptiert) startet gleich neu.",
                    flush=True,
                )
                time.sleep(3)
            continue

    raise RuntimeError(
        f"Stufe {stage} konnte nach {max_attempts} Versuchen nicht fertiggestellt werden. "
        f"Letztes Log: {last_log_path}"
    )


def write_structure_report(run_dir: Path) -> Dict[str, Any]:

    relevant = [
        "daten/raw",
        "daten/processed",
        "daten/features",
        "daten/modelle/musicgen",
        "training/musicgen",
        "training/ausgaben/musicgen_loops",
        "training/bewertungen/musicgen",
        "code/src/Dataset",
        "code/src/Training",
        "code/src/Crawler",
    ]
    rows: List[Dict[str, Any]] = []
    for item in relevant:
        path = PROJECT_ROOT / item
        rows.append(
            {
                "pfad": item,
                "existiert": path.exists(),
                "typ": "ordner" if path.is_dir() else "datei" if path.is_file() else "fehlt",
            }
        )
    payload = {
        "created_at": now(),
        "hinweis": "Crawler/Suche ist bewusst separat und wird von dieser Pipeline nicht gestartet.",
        "ordner": rows,
        "pipeline_start": "Merkmal-Extraktion aus vorhandenen lokalen Dateien/Manifesten",
        "pipeline_ende": "Generierung einer finalen langen Audio",
    }
    write_json(run_dir / "struktur" / "projektstruktur.json", payload)
    md = [
        "# Projektstruktur fuer die MusicGen-Audio-Pipeline",
        "",
        "Crawler und Suchfunktion sind separat. Diese Pipeline startet erst bei vorhandenen lokalen Daten.",
        "",
        "| Pfad | Zweck |",
        "|---|---|",
        "| `daten/raw` | Rohdaten/Importe, nicht die Suche selbst |",
        "| `daten/processed` | MusicGen-Datasets und Manifeste |",
        "| `daten/features` | extrahierte Audio-Merkmale |",
        "| `daten/modelle/musicgen` | lokale MusicGen-Modelle |",
        "| `training/musicgen` | LoRA-Laeufe, Pipeline-Reports und Checkpoints |",
        "| `training/ausgaben/musicgen_loops` | finale lange Audios |",
        "| `training/bewertungen/musicgen` | menschliche Bewertungen und Review-Audios |",
        "| `code/src/Pipeline` | Ablaufsteuerung und technische Merkmal-Extraktion |",
        "| `code/src/Dataset` | Validierung, Qualitaetspruefung und Dataset-Bau |",
        "| `code/src/Training` | LoRA-Training, Review-Generierung und Longform |",
        "| `code/src/Crawler` | separat: Suche, Download, Import |",
        "",
    ]
    (run_dir / "struktur" / "projektstruktur.md").write_text("\n".join(md), encoding="utf-8")
    return payload


def parse_review_line(line: str) -> Optional[Dict[str, str]]:
    if not line.strip() or line.lower().startswith("sample,"):
        return None
    parts = line.rstrip("\n").split(",", 4)
    if len(parts) < 5:
        return None
    return {
        "sample": parts[0].strip(),
        "file": parts[1].strip(),
        "genre": parts[2].strip(),
        "status": parts[3].strip(),
        "note": parts[4].strip(),
    }


def review_is_positive(note: str) -> bool:
    text = note.lower()
    if any(word in text for word in BAD_REVIEW_WORDS):
        return False
    return any(word in text for word in GOOD_REVIEW_WORDS)


def discover_positive_review_audio() -> List[Path]:
    files: List[Path] = []
    for csv_path in sorted((PROJECT_ROOT / "training" / "bewertungen" / "musicgen").glob("durchgang_*/bewertung.csv")):
        for line in csv_path.read_text(encoding="utf-8", errors="replace").splitlines():
            row = parse_review_line(line)
            if not row or not review_is_positive(row["note"]):
                continue
            audio_path = (csv_path.parent / row["file"]).resolve()
            if audio_path.exists() and audio_path.suffix.lower() in {".mp3", ".wav", ".flac"}:
                files.append(audio_path)
    return sorted(dict.fromkeys(files))


def fallback_loop_sources() -> List[Path]:
    candidates = [
        PROJECT_ROOT / "training" / "bewertungen" / "musicgen" / "beste_referenz" / "audio",
        PROJECT_ROOT / "training" / "bewertungen" / "musicgen" / "durchgang_010" / "audio",
        PROJECT_ROOT / "training" / "bewertungen" / "musicgen" / "durchgang_012" / "audio",
        PROJECT_ROOT / "training" / "bewertungen" / "musicgen" / "durchgang_013" / "audio",
    ]
    return [path for path in candidates if path.exists()]


def resolve_loop_sources(args: argparse.Namespace, run_dir: Path) -> List[Path]:
    if args.loop_source:
        sources = [Path(item).expanduser() for item in args.loop_source]
        sources = [(PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve() for path in sources]
    else:
        sources = discover_positive_review_audio()
        if not sources:
            sources = fallback_loop_sources()
    payload = {"created_at": now(), "source_count": len(sources), "sources": [rel(path) for path in sources]}
    write_json(run_dir / "final_audio" / "ausgewaehlte_quellen.json", payload)
    return sources


def adapter_after_training(args: argparse.Namespace, training_run_name: str) -> Path:
    report = PROJECT_ROOT / "training" / "musicgen" / "melody_large_lora" / training_run_name / "abschluss.json"
    if report.exists():
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
            final_checkpoint = payload.get("final_checkpoint")
            if final_checkpoint and Path(final_checkpoint).exists():
                return Path(final_checkpoint)
        except Exception:
            pass
    return Path(args.best_adapter).expanduser().resolve()


def checkpoint_step_from_path(path: Path) -> int:

    for part in reversed(path.parts):
        if part.startswith("step_"):
            digits = part.split("step_", 1)[1]
            if digits.isdigit():
                return int(digits)
    return 0


def build_command_for_stage(
    stage: str,
    args: argparse.Namespace,
    run_dir: Path,
    training_run_name: str,
    adapter_path: Path,
) -> Optional[List[str]]:

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if stage == "validate":
        return [
            str(PYTHON),
            str(DEFAULT_DATASET_ENTRY.relative_to(PROJECT_ROOT)),
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--expected-duration-sec",
            "30.0",
            "--expected-sample-rate",
            "32000",
            "--expected-channels",
            "1",
        ]
    if stage == "audit":
        command = [
            str(PYTHON),
            str(DEFAULT_DATASET_ENTRY.relative_to(PROJECT_ROOT)),
            "audit",
            "--dataset-root",
            str(dataset_root),
            "--report-path",
            str(run_dir / "audit" / "musicgen_clip_qualitaetspruefung.json"),
            "--issues-path",
            str(run_dir / "audit" / "musicgen_clip_probleme.jsonl"),
        ]
        if args.deep_audit:
            command.append("--deep-audio-check")
        return command
    if stage == "train":
        if not args.training_aktivieren:
            return None
        resume_adapter = Path(args.best_adapter).expanduser().resolve()
        start_step = checkpoint_step_from_path(resume_adapter)
        ziel_step = start_step + int(args.training_steps)
        return [
            str(PYTHON),
            str(DEFAULT_TRAINING_ENTRY.relative_to(PROJECT_ROOT)),
            "train",
            "--mode",
            "train",
            "--dataset-root",
            str(dataset_root),
            "--run-name",
            training_run_name,
            "--resume-from",
            str(resume_adapter),
            "--auto-vram",
            "--max-steps",
            str(ziel_step),
            "--learning-rate",
            str(args.learning_rate),
            "--save-steps",
            str(args.save_steps),
            "--eval-steps",
            str(args.eval_steps),
            "--vram-limit-fraction",
            str(args.vram_limit_fraction),
            "--min-sekunden-rms-db",
            str(args.min_sekunden_rms_db),
        ]
    if stage == "review":
        if not args.review_aktivieren and not args.training_aktivieren:
            return None
        review_run = args.review_run_name or f"{run_dir.name}_review"
        return [
            str(PYTHON),
            str(DEFAULT_TRAINING_ENTRY.relative_to(PROJECT_ROOT)),
            "review",
            "--adapter-path",
            str(adapter_path),
            "--run-name",
            review_run,
            "--prompt-set",
            args.prompt_set,
            "--seed",
            str(args.seed),
        ]
    if stage == "final_audio":
        sources = resolve_loop_sources(args, run_dir)
        final_name = args.final_name or f"{run_dir.name}_final_audio"
        rhythm_block_seconds = resolve_rhythm_block_seconds(
            parse_duration_seconds(args.dauer),
            args.block_sekunden,
            args.rhythmus_anteil_prozent,
            args.min_rhythmus_sekunden,
        )
        command = [
            str(PYTHON),
            "code/src/Training/clips_loopen.py",
            "--dauer",
            args.dauer,
            "--name",
            final_name,
            "--seed",
            str(args.seed),
            "--crossfade-sekunden",
            str(args.crossfade_sekunden),
            "--fade-in-sekunden",
            str(args.fade_in_sekunden),
            "--fade-out-sekunden",
            str(args.fade_out_sekunden),
            "--ziel-bpm",
            str(args.ziel_bpm),
            "--block-sekunden",
            str(rhythm_block_seconds),
            "--block-variation-sekunden",
            str(args.block_variation_sekunden),
            "--ausgabe",
            "kompakt",
        ]
        for source in sources:
            command.extend(["--quelle", str(source)])
        if args.kein_mp3:
            command.append("--kein-mp3")
        if args.github_push:
            command.append("--github-push")
        else:
            command.append("--kein-github-push")
        return command
    if stage == "audio_generieren":
        final_name = args.final_name or f"{run_dir.name}_audio"
        rhythm_block_seconds = resolve_rhythm_block_seconds(
            parse_duration_seconds(args.dauer),
            args.block_sekunden,
            args.rhythmus_anteil_prozent,
            args.min_rhythmus_sekunden,
        )
        command = [
            str(PYTHON),
            "code/src/Training/audio_erstellen.py",
            "--dauer",
            args.dauer,
            "--genre",
            args.genre,
            "--name",
            final_name,
            "--seed",
            str(args.seed),
            "--adapter-path",
            str(adapter_path),
            "--crossfade-sekunden",
            str(args.crossfade_sekunden),
            "--fade-in-sekunden",
            str(args.fade_in_sekunden),
            "--fade-out-sekunden",
            str(args.fade_out_sekunden),
            "--anschluss-sekunden",
            str(args.anschluss_sekunden),
            "--ziel-bpm",
            str(args.ziel_bpm),
            "--bpm-toleranz",
            str(args.bpm_toleranz),
            "--uebergang-bpm-toleranz",
            str(args.uebergang_bpm_toleranz),
            "--block-sekunden",
            str(rhythm_block_seconds),
            "--rhythmus-anteil-prozent",
            str(args.rhythmus_anteil_prozent),
            "--min-rhythmus-sekunden",
            str(args.min_rhythmus_sekunden),
            "--block-variation-sekunden",
            str(args.block_variation_sekunden),
            "--loop-crossfade-sekunden",
            str(args.loop_crossfade_sekunden),
            "--tempo-variation-prozent",
            str(args.tempo_variation_prozent),
            "--tempo-phase-sekunden",
            str(args.tempo_phase_sekunden),
            "--abschnitt-sekunden",
            str(args.abschnitt_sekunden),
            "--kandidaten-pro-abschnitt",
            str(args.kandidaten_pro_abschnitt),
            "--vram-limit-fraction",
            str(args.vram_limit_fraction),
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            "--top-p",
            str(args.top_p),
            "--cfg-coef",
            str(args.cfg_coef),
            "--erweiterungs-schritt-sekunden",
            str(args.erweiterungs_schritt_sekunden),
            "--clap-modell-pfad",
            str(Path(args.clap_modell_pfad).expanduser().resolve()),
            "--genre-referenz-ordner",
            str(Path(args.genre_referenz_ordner).expanduser().resolve()),
            "--min-genre-aehnlichkeit",
            str(args.min_genre_aehnlichkeit),
            "--min-genre-abstand",
            str(args.min_genre_abstand),
            "--min-qualitaets-abstand",
            str(args.min_qualitaets_abstand),
            "--referenz-dataset-root",
            str(Path(args.referenz_dataset_root).expanduser().resolve()),
            "--referenzen-pro-genre",
            str(args.referenzen_pro_genre),
            "--referenz-score-limit",
            str(args.referenz_score_limit),
        ]
        if args.block_sekunden <= 0:
            command.append("--blockmodus-prozentual")
        if args.max_generierte_kandidaten:
            command.extend(["--max-generierte-kandidaten", str(args.max_generierte_kandidaten)])
        if args.review_adapter_erlauben:
            command.append("--review-adapter-erlauben")
        if args.stimmung:
            command.extend(["--stimmung", args.stimmung])
        if args.instrumente:
            command.extend(["--instrumente", args.instrumente])
        if args.max_abschnitte:
            command.extend(["--max-abschnitte", str(args.max_abschnitte)])
        if args.fallback_pool:
            command.append("--fallback-pool")
        if args.anschluss_conditioning_aktiv:
            command.append("--anschluss-conditioning-aktiv")
        else:
            command.append("--anschluss-conditioning-deaktivieren")
        if args.genre_melodie_conditioning_aktiv:
            command.append("--genre-melodie-conditioning-aktiv")
        else:
            command.append("--genre-melodie-conditioning-deaktivieren")
        if args.block_looping_aktiv:
            command.append("--block-looping-aktiv")
        else:
            command.append("--block-looping-deaktivieren")
        if args.kontinuierliche_bloecke_aktiv:
            command.append("--kontinuierliche-bloecke-aktiv")
        else:
            command.append("--kontinuierliche-bloecke-deaktivieren")
        if args.genre_pruefung_aktiv:
            command.append("--genre-pruefung-aktiv")
        else:
            command.append("--genre-pruefung-deaktivieren")
        if args.mp3_referenz_pruefung_aktiv:
            command.append("--mp3-referenz-pruefung-aktiv")
        else:
            command.append("--mp3-referenz-pruefung-deaktivieren")
        if args.bpm_pruefung_aktiv:
            command.append("--bpm-pruefung-aktiv")
        else:
            command.append("--bpm-pruefung-deaktivieren")
        if args.sekunden_pruefung_aktiv:
            command.append("--sekunden-pruefung-aktiv")
        else:
            command.append("--sekunden-pruefung-deaktivieren")
        if args.kein_mp3:
            command.append("--kein-mp3")
        if args.github_push:
            command.append("--github-push")
        else:
            command.append("--kein-github-push")
        return command
    return None


def main() -> int:
    args = parse_args()
    run_name = args.run_name or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.pipeline_root).expanduser()
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir = run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    stages = planned_stages(args)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    training_run_name = args.training_run_name or f"{run_name}_lora"
    adapter_path = Path(args.best_adapter).expanduser().resolve()
    results: List[StageResult] = []
    budget = apply_safety_mode(args, stages)
    if audio_stage_requested(stages):
        write_generation_budget(run_dir, budget)
        if not args.nur_plan and not args.review_adapter_erlauben and not adapter_fuer_audio_freigegeben(adapter_path):
            report = {
                "status": "failed",
                "created_at": now(),
                "finished_at": now(),
                "error": (
                    "LoRA-Adapter aus dem LoRA-Clip-Training fehlt oder der Lauf ist noch nicht abgeschlossen: "
                    f"{rel(adapter_path)}. Training fortsetzen mit: "
                    ".venv/bin/python code/start.py --lora-fortsetzen"
                ),
                "run_dir": rel(run_dir),
                "stages": [],
            }
            write_json(run_dir / "pipeline_report.json", report)
            print("Status:    Audioerzeugung nicht gestartet.", flush=True)
            print(f"Adapter:   noch nicht freigegeben ({rel(adapter_path)})", flush=True)
            print("Naechster Schritt:", flush=True)
            print(".venv/bin/python code/start.py --lora-fortsetzen", flush=True)
            return 5

    print("Pipeline", flush=True)
    print("========", flush=True)
    print(f"Run:           {rel(run_dir)}", flush=True)
    print(f"Stufen:        {', '.join(stages)}", flush=True)
    print("Crawler/Suche: separat, wird nicht gestartet.", flush=True)
    if audio_stage_requested(stages):
        print("", flush=True)
        print("Audio-Ziel", flush=True)
        print("----------", flush=True)
        print(f"Genre:         {args.genre}", flush=True)
        print(f"Dauer:         {args.dauer} ({budget['duration_sec']:.0f}s)", flush=True)
        print(f"BPM:           {args.ziel_bpm:.1f} (+/-{args.bpm_toleranz:.1f})", flush=True)
        print(f"Uebergang-BPM: +/-{args.uebergang_bpm_toleranz:.1f}", flush=True)
        print(f"Crossfade:     {args.crossfade_sekunden:.1f}s", flush=True)
        print(
            f"Rhythmus:      {budget['rhythm_block_seconds']:.1f}s pro Block "
            f"({budget['rhythm_share_percent']:.1f}%, "
            f"Minimum {budget['minimum_rhythm_seconds']:.0f}s)",
            flush=True,
        )
        print(
            f"Sekundencheck: {yes_no(args.sekunden_pruefung_aktiv)} "
            f"(min. {args.min_sekunden_rms_db:.1f} dBFS)",
            flush=True,
        )
        print(
            f"MP3-Referenz:  {yes_no(args.mp3_referenz_pruefung_aktiv)} "
            f"({args.referenzen_pro_genre} Clips/Genre, Limit {args.referenz_score_limit:.1f})",
            flush=True,
        )
        print(f"Vergleich:     {yes_no(args.referenzvergleich_aktiv)} nach fertiger Audio", flush=True)
        print(
            f"Postprocess:  {yes_no(args.postprocessing)}"
            + (f" ({args.postprocessing_intensitaet})" if args.postprocessing else ""),
            flush=True,
        )
        print(f"VRAM-Limit:    {args.vram_limit_fraction * 100:.0f}%", flush=True)
        kandidaten_limit = (
            f"Reserve {budget['candidate_generation_reserve']}"
            if budget["candidate_generation_reserve"] is not None
            else "keine harte Grenze"
        )
        print(
            f"Plan:          {budget['estimated_sections']} Abschnitt(e), "
            f"{budget['estimated_candidate_generations']} Kandidaten, "
            f"{kandidaten_limit}, Sparmodus={yes_no(budget['sparmodus'])}.",
            flush=True,
        )
        if budget["long_run_blocked_to_plan"]:
            print(
                "Sicherheitsstopp: Lauf ist laenger als 5 Minuten. "
                "Es wird nur geplant. Fuer echten Start --freigabe-langer-lauf angeben.",
                flush=True,
            )

    started = now()
    try:
        for stage in stages:
            if stage == "struktur":
                stage_started = now()
                detail = write_structure_report(run_dir)
                results.append(StageResult(stage, "ok", stage_started, now(), [], detail=detail))
            elif stage == "features":
                stage_started = now()
                if args.nur_plan:
                    detail = {"planned": True, "dataset_root": rel(dataset_root), "limit": args.feature_limit}
                    results.append(StageResult(stage, "planned", stage_started, now(), [], detail=detail))
                else:
                    detail = extract_features(dataset_root, run_dir, args.feature_limit)
                    results.append(StageResult(stage, "ok", stage_started, now(), [], detail=detail))
            else:
                if stage == "review":
                    adapter_path = adapter_after_training(args, training_run_name)
                command = build_command_for_stage(stage, args, run_dir, training_run_name, adapter_path)
                if command is None:
                    results.append(
                        StageResult(
                            stage,
                            "skipped",
                            now(),
                            now(),
                            [],
                            detail={"reason": "nicht aktiviert oder fuer diesen Modus nicht noetig"},
                        )
                    )
                    continue
                if stage == "audio_generieren":
                    audio_result = run_audio_with_resume(
                        stage,
                        command,
                        run_dir,
                        args.nur_plan,
                        args.live_status_sekunden,
                        args.debug_ausgabe,
                        args,
                        budget,
                    )
                    results.append(audio_result)
                    final_audio = (audio_result.detail or {}).get("final_audio")
                    if args.postprocessing and final_audio:
                        results.append(
                            run_postprocessing_stage(
                                str(final_audio),
                                args,
                                run_dir,
                                args.nur_plan,
                            )
                        )
                        final_audio = (results[-1].detail or {}).get("final_audio") or final_audio
                    if args.referenzvergleich_aktiv and final_audio:
                        results.append(
                            run_referenzvergleich_stage(
                                str(final_audio),
                                args,
                                run_dir,
                                args.nur_plan,
                            )
                        )
                else:
                    results.append(run_subprocess(stage, command, run_dir, args.nur_plan))
                if stage == "train" and not args.nur_plan:
                    adapter_path = adapter_after_training(args, training_run_name)
    except Exception as exc:
        report = {
            "status": "failed",
            "created_at": started,
            "finished_at": now(),
            "error": f"{type(exc).__name__}: {exc}",
            "run_dir": rel(run_dir),
            "stages": [asdict(item) for item in results],
        }
        write_json(run_dir / "pipeline_report.json", report)
        print_pipeline_finish(report, results, run_dir)
        return 1

    report = {
        "status": "planned" if args.nur_plan else "finished",
        "created_at": started,
        "finished_at": now(),
        "run_dir": rel(run_dir),
        "mode": args.modus,
        "dataset_root": rel(dataset_root),
        "crawler_und_suche": "separat, nicht Teil dieser Pipeline",
        "best_adapter": rel(Path(args.best_adapter).expanduser().resolve()),
        "final_adapter_used": rel(adapter_path),
        "audio_generierungs_plan": budget if audio_stage_requested(stages) else None,
        "stages": [asdict(item) for item in results],
    }
    write_json(run_dir / "pipeline_report.json", report)
    print_pipeline_finish(report, results, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
