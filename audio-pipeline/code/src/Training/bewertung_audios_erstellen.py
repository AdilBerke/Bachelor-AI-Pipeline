#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJEKTWURZEL = Path(__file__).resolve().parents[3]
PYTHON = PROJEKTWURZEL / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

TRAINING_DIR = PROJEKTWURZEL / "code" / "src" / "Training"
STEUERUNG = TRAINING_DIR / "musicgen_steuerung.py"
BEWERTUNGEN_ROOT = PROJEKTWURZEL / "training" / "bewertungen" / "musicgen"
STANDARD_LORA_ADAPTER = (
    PROJEKTWURZEL
    / "training"
    / "musicgen"
    / "lora_training"
    / "adapter.pt"
)

sys.path.insert(0, str(TRAINING_DIR))
from lora import (
    STANDARD_BESTER_ADAPTER,
    STANDARD_RUN_ROOT,
    checkpoint_step,
    finde_neuesten_checkpoint,
    lade_lora_stand,
    policy_bester_adapter,
    rel,
)
from audio_veroeffentlichen import audio_dateien_aus_eingabe, veroeffentliche_audios


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Erstellt 8 feste MusicGen-LoRA-Bewertungsaudios.")
    parser.add_argument(
        "--adapter-path",
        default="best",
        help=(
            "best/lora = aktueller LoRA-Stand, latest/auto = neuester Checkpoint, "
            "sonst direkter Pfad zu lora_adapter.pt."
        ),
    )
    parser.add_argument(
        "--prompt-set",
        choices=("lofi_standard", "jazz_lofi"),
        default="lofi_standard",
        help="Feste 8-Prompt-Gruppe fuer die Bewertung.",
    )
    parser.add_argument("--run-name", default="", help="Optional z.B. durchgang_015 oder vergleich_step_3050.")
    parser.add_argument("--output-root", default=str(BEWERTUNGEN_ROOT))
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--kandidaten-pro-genre", type=int, default=3)
    parser.add_argument("--min-attempts", type=int, default=3)
    parser.add_argument("--min-ok-score", type=float, default=75.0)
    parser.add_argument("--seed", type=int, default=0, help="0 = bei jedem Lauf ein neuer Zufallsseed.")
    parser.add_argument("--vram-limit-fraction", type=float, default=0.75)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--remote-modell-erlauben", action="store_true")
    parser.add_argument(
        "--github-push",
        dest="github_push",
        action="store_true",
        default=True,
        help="Pusht nach dem Lauf alle finalen Testaudios aus dem audio-Ordner nach main.",
    )
    parser.add_argument(
        "--kein-github-push",
        dest="github_push",
        action="store_false",
        help="Laesst die Testaudios nur lokal.",
    )
    parser.add_argument("--nur-plan", action="store_true", help="Nur anzeigen, was gestartet wuerde.")
    return parser.parse_args()


def loese_adapter(value: str) -> Path:

    normalized = str(value or "").strip().lower()
    if normalized in {"", "best", "bester", "lora", "stabil"}:
        stand = lade_lora_stand()
        keys = (
            ("adapter", "best_adapter", "checkpoint")
            if stand.get("status") == "freigegeben"
            else ("candidate_adapter", "adapter", "checkpoint", "best_adapter")
        )
        for key in keys:
            value_from_stand = stand.get(key)
            if not value_from_stand:
                continue
            path = Path(str(value_from_stand)).expanduser()
            if not path.is_absolute():
                path = PROJEKTWURZEL / path
            if path.exists():
                return path.resolve()
        return policy_bester_adapter()
    if normalized in {"5000", "5000_clips", "lora_5000_clips"}:
        if STANDARD_LORA_ADAPTER.exists():
            return STANDARD_LORA_ADAPTER.resolve()
        return policy_bester_adapter()
    if normalized in {"auto", "latest", "neueste", "neuster"}:
        latest = finde_neuesten_checkpoint(STANDARD_RUN_ROOT)
        return (latest or policy_bester_adapter()).resolve()
    return Path(value).expanduser().resolve()


def standard_run_name(adapter_path: Path) -> str:

    try:
        if adapter_path.resolve() == STANDARD_LORA_ADAPTER.resolve():
            return "lora_bewertung_001"
    except FileNotFoundError:
        pass
    step = checkpoint_step(adapter_path)
    if step > 0:
        return f"vergleich_step_{step:04d}"
    return ""


def freier_run_name(output_root: Path, run_name: str) -> str:

    if not run_name or not (output_root / run_name).exists():
        return run_name
    for index in range(2, 1000):
        candidate = f"{run_name}_{index:03d}"
        if not (output_root / candidate).exists():
            return candidate
    zeit = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{run_name}_{zeit}"


def baue_befehl(args: argparse.Namespace, adapter_path: Path) -> list[str]:

    command = [
        str(PYTHON),
        str(STEUERUNG.relative_to(PROJEKTWURZEL)),
        "review",
        "--adapter-path",
        str(adapter_path),
        "--output-root",
        str(Path(args.output_root).expanduser().resolve()),
        "--prompt-set",
        args.prompt_set,
        "--duration-sec",
        str(args.duration_sec),
        "--candidates-per-genre",
        str(args.kandidaten_pro_genre),
        "--min-attempts",
        str(args.min_attempts),
        "--min-ok-score",
        str(args.min_ok_score),
        "--min-final-samples",
        "8",
        "--seed",
        str(args.seed),
        "--vram-limit-fraction",
        str(args.vram_limit_fraction),
    ]
    if args.run_name:
        command.extend(["--run-name", args.run_name])
    if args.keep_temp:
        command.append("--keep-temp")
    if args.overwrite:
        command.append("--overwrite")
    if args.remote_modell_erlauben:
        command.append("--allow-remote-model")
    return command


def naechster_log_pfad(output_root: Path, run_name: str) -> Path:

    zeit = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", run_name.strip()) if run_name else "bewertung_audios"
    return output_root / "_logs" / f"{label}_{zeit}.log"


def parse_finales_json(text: str) -> dict[str, object]:

    marker = '{\n  "pack_root"'
    start = text.rfind(marker)
    if start < 0:
        return {}
    snippet = text[start:]
    end = snippet.rfind("}")
    if end < 0:
        return {}
    try:
        payload = json.loads(snippet[: end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def fuehre_minimal_aus(command: list[str], *, output_root: Path, run_name: str) -> tuple[int, Path, dict[str, object]]:

    log_path = naechster_log_pfad(output_root, run_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kandidat_re = re.compile(r"^(sample_(\d+)) Kandidat (\d+):")
    auswahl_re = re.compile(r"^=> ausgewaehlt: sample_(\d+)")
    skip_re = re.compile(r"^=> uebersprungen: sample_(\d+)")
    sample = 0
    kandidat = 0
    ausgewaehlt = 0
    uebersprungen = 0

    print("Status: Modell wird geladen ...", flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("XFORMERS_MORE_DETAILS", "0")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
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
            stripped = line.strip()
            match = kandidat_re.search(stripped)
            if match:
                sample = int(match.group(2))
                kandidat = int(match.group(3))
            elif auswahl_re.search(stripped):
                ausgewaehlt += 1
            elif skip_re.search(stripped):
                uebersprungen += 1
            if match or stripped.startswith("=> "):
                print(
                    "\r"
                    f"Status: laeuft | Sample {sample}/8 | Kandidat {kandidat} | "
                    f"fertig {ausgewaehlt}/8 | verworfen {uebersprungen}",
                    end="",
                    flush=True,
                )
        returncode = process.wait()
    print()

    text = log_path.read_text(encoding="utf-8", errors="replace")
    return returncode, log_path, parse_finales_json(text)


def push_testaudios(pack_root: str | object, run_name: str) -> dict[str, object]:

    if not pack_root:
        return {"status": "skipped", "reason": "kein Bewertungsordner im Report"}
    audio_dir = Path(str(pack_root)).expanduser().resolve() / "audio"
    if not audio_dir.exists():
        return {"status": "skipped", "reason": f"kein audio-Ordner: {rel(audio_dir)}"}
    try:
        audio_paths = audio_dateien_aus_eingabe(audio_dir)
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    if not audio_paths:
        return {"status": "skipped", "reason": "keine Audios gefunden"}
    return veroeffentliche_audios(
        audio_paths,
        branch="main",
        commit_text=f"Testaudios: {run_name}",
    )


def main() -> int:

    args = parse_args()
    if args.seed <= 0:
        args.seed = int(time.time_ns() ^ (os.getpid() << 16)) & 0x7FFFFFFF
    adapter_path = loese_adapter(args.adapter_path)
    output_root = Path(args.output_root).expanduser().resolve()
    auto_run_name = not args.run_name
    if not args.run_name:
        args.run_name = standard_run_name(adapter_path)
    if auto_run_name:
        args.run_name = freier_run_name(output_root, args.run_name)
    if not adapter_path.exists():
        print(f"Adapter nicht gefunden: {adapter_path}", file=sys.stderr)
        return 2
    if not STEUERUNG.exists():
        print(f"MusicGen-Steuerung nicht gefunden: {STEUERUNG}", file=sys.stderr)
        return 2

    step = checkpoint_step(adapter_path)
    command = baue_befehl(args, adapter_path)
    print("Bewertungs-Audios")
    print("=================")
    print(f"Adapter:   {rel(adapter_path)}")
    print(f"Step:      {step if step > 0 else 'unbekannt'}")
    print(f"Promptset: {args.prompt_set}")
    print(f"Audios:    8 x {args.duration_sec:.0f}s")
    print(f"Kandidaten:{args.kandidaten_pro_genre} pro Genre")
    if args.run_name:
        print(f"Run:       {args.run_name}")
    print(f"Ausgabe:   {rel(output_root)}")

    if args.nur_plan:
        print("Status:    nur Plan, keine Generierung gestartet")
        print("Befehl:")
        print(" ".join(subprocess.list2cmdline([part]) for part in command))
        return 0

    returncode, log_path, result_payload = fuehre_minimal_aus(command, output_root=output_root, run_name=args.run_name)
    status = {
        "returncode": returncode,
        "adapter": rel(adapter_path),
        "adapter_step": step,
        "prompt_set": args.prompt_set,
        "duration_sec": args.duration_sec,
        "candidates_per_genre": args.kandidaten_pro_genre,
        "seed": args.seed,
        "log_path": rel(log_path),
        "result": result_payload,
    }
    status_path = output_root / "letzter_review_start.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pack_root = result_payload.get("pack_root")
    bewertung = result_payload.get("bewertung")
    push_result: dict[str, object] = {"status": "skipped", "reason": "nicht gestartet"}
    if args.github_push and pack_root:
        push_result = push_testaudios(pack_root, args.run_name)
        status["github_push"] = push_result
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if returncode == 0:
        print("Status: fertig")
        if pack_root:
            print(f"Ordner: {rel(Path(str(pack_root)))}")
        if bewertung:
            print(f"Bewertung: {rel(Path(str(bewertung)))}")
        if push_result.get("status") == "pushed":
            print(f"GitHub: {push_result.get('url')}")
        elif args.github_push:
            print(f"GitHub: {push_result.get('status')} ({push_result.get('reason', 'siehe Report')})")
        print(f"Log: {rel(log_path)}")
    elif returncode == 3 and result_payload:
        print("Status: nicht bewertbar")
        print(f"Akzeptiert: {result_payload.get('accepted_samples', 0)}/8")
        print(f"Uebersprungen: {result_payload.get('skipped_samples', 0)}/8")
        if pack_root:
            print(f"Ordner: {rel(Path(str(pack_root)))}")
        if push_result.get("status") == "pushed":
            print(f"GitHub: {push_result.get('url')}")
        elif args.github_push:
            print(f"GitHub: {push_result.get('status')} ({push_result.get('reason', 'siehe Report')})")
        print(f"Log: {rel(log_path)}")
        print("Hinweis: Dieser Checkpoint sollte nicht als neuer bester Checkpoint verwendet werden.")
    else:
        print(f"Status: abgebrochen mit Code {returncode}")
        print(f"Log: {rel(log_path)}")
        print(f"Report: {rel(status_path)}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
