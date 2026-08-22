#!/usr/bin/env python3
"""Interaktives Terminal-Skript fuer lokale Audio-Generierung."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

MOODS = ["random", "relaxed", "melancholic", "uplifting", "dreamy", "nostalgic", "cozy"]
INSTRUMENTS = ["piano", "guitar", "vinyl crackle", "soft drums", "bass", "synth pad", "bells", "saxophone"]
RENDER_ENGINES = ["foundation", "hybrid"]
QUALITIES = ["fast", "medium", "ultra", "final_master"]
TRAINED_MODEL_OPTIONS = ["ja", "ja inkl probe", "nein"]

QUALITY_INFO = {
    "fast": "musicgen-small  | ~15-25 Min pro 30 Min Audio",
    "medium": "musicgen-medium | ~30-60 Min pro 30 Min Audio",
    "ultra": "musicgen-large  | ~1-3 Std pro 30 Min Audio",
    "final_master": "musicgen-large  | bis zu 24 Std pro 30 Min Audio",
}


def get_project_python() -> str:
    script_path = Path(__file__).resolve()
    for root in (script_path.parents[2], script_path.parents[1]):
        venv_python = root / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
    return sys.executable


def print_cuda_status(python_exe: str) -> None:
    probe = (
        "import torch; "
        "print(torch.cuda.is_available()); "
        "print(torch.cuda.device_count()); "
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    )
    try:
        result = subprocess.run(
            [python_exe, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"  CUDA Check:    fehlgeschlagen ({exc})")
        return

    lines = [line.strip() for line in result.stdout.splitlines()]
    cuda_available = len(lines) >= 1 and lines[0] == "True"
    device_count = lines[1] if len(lines) >= 2 else "0"
    device_name = lines[2] if len(lines) >= 3 else ""
    if cuda_available:
        print(f"  CUDA:         aktiv ({device_count} GPU, {device_name})")
    else:
        print("  CUDA:         nicht sichtbar fuer PyTorch")


def ask_choice(prompt: str, options: list[str], default: str | None = None,
               info: dict[str, str] | None = None) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if opt == default else ""
        extra = f"  -- {info[opt]}" if info and opt in info else ""
        print(f"  [{i}] {opt}{marker}{extra}")
    while True:
        raw = input("Auswahl: ").strip()
        if not raw and default:
            return default
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            if raw.lower() in options:
                return raw.lower()
        print(f"Bitte 1-{len(options)} eingeben.")


def ask_multi_choice(prompt: str, options: list[str]) -> list[str]:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    print("  Mehrere mit Komma trennen (z.B. 1,3,5) oder Enter fuer auto")
    while True:
        raw = input("Auswahl: ").strip()
        if not raw:
            return []
        try:
            indices = [int(x.strip()) for x in raw.split(",")]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(options):
                    selected.append(options[idx - 1])
                else:
                    raise ValueError
            return selected
        except ValueError:
            print(f"Bitte gueltige Nummern 1-{len(options)} eingeben.")


def ask_float(prompt: str, min_val: float, max_val: float, default: float | None = None) -> float | None:
    default_hint = f" (default: {default})" if default is not None else " (Enter = auto)"
    while True:
        raw = input(f"\n{prompt}{default_hint}: ").strip()
        if not raw:
            return default
        raw = raw.replace(",", ".")
        try:
            val = float(raw)
            if min_val <= val <= max_val:
                return val
            print(f"Bitte einen Wert zwischen {min_val} und {max_val} eingeben.")
        except ValueError:
            print("Ungueltige Eingabe.")


def estimate_duration(minutes: float, quality: str, engine: str) -> str:
    ratios = {"fast": (0.5, 0.85), "medium": (1.0, 2.0), "ultra": (2.0, 6.0), "final_master": (6.0, 48.0)}
    low, high = ratios.get(quality, (1.0, 2.0))
    if engine == "hybrid":
        low += 0.3
        high += 1.0
    est_low = minutes * low
    est_high = minutes * high
    def fmt(m: float) -> str:
        if m < 60:
            return f"{m:.0f} Min"
        return f"{m / 60:.1f} Std"
    return f"~{fmt(est_low)} - {fmt(est_high)}"


def run_with_output(cmd: list[str], cwd: str, label: str) -> int:
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"  Befehl: {' '.join(cmd)}")
    print(f"{'=' * 50}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0
    if result.returncode == 0:
        minutes_elapsed = elapsed / 60
        print(f"\n  Fertig in {minutes_elapsed:.1f} Minuten.")
    else:
        print(f"\n  FEHLER (exit code {result.returncode})")
    return result.returncode


def export_trained_musicgen(python_exe: str, repo_root: str, include_probe: bool) -> str:
    script = os.path.join(repo_root, "src", "Training", "export_musicgen_checkpoint.py")
    cmd = [
        python_exe,
        script,
        "--checkpoint",
        "latest",
        "--name",
        "latest_with_probe" if include_probe else "latest",
        "--print-path",
    ]
    if include_probe:
        cmd.append("--include-probe-checkpoints")
    print("\nTrainiertes MusicGen wird vorbereitet...")
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        raise RuntimeError("Konnte trainiertes MusicGen nicht exportieren.")
    model_dir = result.stdout.strip().splitlines()[-1].strip()
    if not model_dir:
        raise RuntimeError("Export hat keinen Modellpfad zurueckgegeben.")
    print(f"  Trainiertes Modell: {model_dir}")
    return model_dir


def main() -> None:
    print("=" * 50)
    print("  Lofi Audio Generator - Lokaler Modus")
    print("=" * 50)

    minutes = ask_float("Wie viele Minuten soll die Audio gehen?", 1, 300, default=5)
    bpm = ask_float("BPM (Tempo)", 60, 220, default=None)
    mood = ask_choice("Mood waehlen:", MOODS, default="random")
    instruments = ask_multi_choice("Instrumente waehlen:", INSTRUMENTS)
    engine = ask_choice("Render Engine:", RENDER_ENGINES, default="foundation")
    quality = ask_choice("Generierungsqualitaet:", QUALITIES, default="fast", info=QUALITY_INFO)
    trained_model_choice = "nein"
    if engine == "foundation":
        trained_model_choice = ask_choice(
            "Trainiertes MusicGen verwenden?",
            TRAINED_MODEL_OPTIONS,
            default="ja",
            info={
                "ja": "neuester stabiler Fine-Tuning-Checkpoint",
                "ja inkl probe": "auch Probe-Checkpoints wie train_013_probe_once erlauben",
                "nein": "originales facebook/musicgen-* Modell",
            },
        )
    else:
        print("\nHinweis: Hybrid nutzt musicgen-melody. Unser aktuelles Fine-Tuning ist textbasiert und wird dort noch nicht geladen.")

    est = estimate_duration(minutes, quality, engine)
    python_exe = get_project_python()

    print("\n" + "=" * 50)
    print("  Zusammenfassung")
    print("=" * 50)
    print(f"  Laenge:       {minutes} Minuten")
    print(f"  BPM:          {bpm if bpm else 'auto'}")
    print(f"  Mood:         {mood}")
    print(f"  Instrumente:  {', '.join(instruments) if instruments else 'auto'}")
    print(f"  Engine:       {engine}")
    print(f"  Qualitaet:    {quality}")
    print(f"  Fine-Tuning:  {trained_model_choice if engine == 'foundation' else 'nicht im Hybrid-Modus'}")
    print(f"  Geschaetzte Dauer: {est}")
    print(f"  Python:       {python_exe}")
    print_cuda_status(python_exe)
    print("=" * 50)

    confirm = input("\nStarten? [J/n] ").strip().lower()
    if confirm and confirm not in ("j", "ja", "y", "yes"):
        print("Abgebrochen.")
        return

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(repo_root, "archiv", "hybrid_ap4", "ap4_prompting_musikgenerierung")
    work_dir = repo_root
    features = os.path.join("daten", "features", "audio_features.jsonl")

    common_args = []
    if bpm:
        common_args += ["--bpm", str(bpm)]
    if mood:
        common_args += ["--mood", mood]
    if instruments:
        common_args += ["--instruments", ",".join(instruments)]

    trained_model_dir = ""
    if engine == "foundation" and trained_model_choice != "nein":
        trained_model_dir = export_trained_musicgen(
            python_exe=python_exe,
            repo_root=repo_root,
            include_probe=trained_model_choice == "ja inkl probe",
        )

    if engine == "foundation":
        script = os.path.join(src_dir, "foundation_prompting.py")
        cmd = [python_exe, "-u", script,
               "--features", features,
               "--minutes", str(minutes),
               "--quality", quality,
               "--device", "cuda",
               "--log-level", "INFO"] + common_args
        if trained_model_dir:
            cmd += ["--model-id", trained_model_dir]
        rc = run_with_output(cmd, work_dir, "Foundation Model Generierung")
        sys.exit(rc)

    elif engine == "hybrid":
        script_proc = os.path.join(src_dir, "prompting.py")
        script_found = os.path.join(src_dir, "foundation_prompting.py")
        skeleton_wav = os.path.join("ausgaben", "generated_audio", "skeleton_tmp.wav")

        cmd_skeleton = [python_exe, "-u", script_proc,
                        "--features", features,
                        "--minutes", str(minutes),
                        "--out-wav", skeleton_wav,
                        "--log-level", "INFO"] + common_args
        rc = run_with_output(cmd_skeleton, work_dir, "Phase 1/2: Prozedurales Skelett")
        if rc != 0:
            print("Abbruch: Skelett-Generierung fehlgeschlagen.")
            sys.exit(rc)

        cmd_found = [python_exe, "-u", script_found,
                     "--features", features,
                     "--minutes", str(minutes),
                     "--quality", quality,
                     "--device", "cuda",
                     "--melody-wav", skeleton_wav,
                     "--log-level", "INFO"] + common_args
        rc = run_with_output(cmd_found, work_dir, "Phase 2/2: Foundation Model mit Skelett")
        sys.exit(rc)


if __name__ == "__main__":
    main()
