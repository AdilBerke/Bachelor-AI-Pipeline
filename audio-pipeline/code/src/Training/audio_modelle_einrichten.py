#!/usr/bin/env python3
"""Lokales Setup fuer optionale Audio-Qualitaetsmodelle.

Die normale MusicGen-Pipeline laedt absichtlich keine externen Modelle. Dieses
Skript ist der bewusste Einmal-Schritt, wenn die Nachbearbeitung komplett lokal
mit DeepFilterNet, Demucs, Matchering oder AudioSR vorbereitet werden soll.
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "code").exists() and (parent / "training").exists()
)
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)
VENV_BIN = PROJECT_ROOT / ".venv" / "bin"
REPORT_DIR = PROJECT_ROOT / "training" / "musicgen" / "audio_modelle"
MODEL_ROOT = PROJECT_ROOT / "daten" / "modelle" / "audio_analyse"
CLAP_REPO_ID = "laion/clap-htsat-unfused"
CLAP_DIR = MODEL_ROOT / "clap_htsat_unfused"
DEMUCS_MODEL_NAME = "htdemucs"
DEMUCS_CACHE = MODEL_ROOT / "demucs"


@dataclass(frozen=True)
class ModellSetup:
    """Beschreibt ein optionales lokales Audio-Tool."""

    name: str
    pip_pakete: List[str]
    python_imports: List[str]
    cli_befehle: List[str]
    zweck: str
    hinweis: str
    empfohlen: bool = True


MODELLE: Dict[str, ModellSetup] = {
    "deepfilternet": ModellSetup(
        name="DeepFilterNet",
        pip_pakete=["deepfilternet"],
        python_imports=["df"],
        cli_befehle=["deepFilter", "deep-filter-py", "deep-filter"],
        zweck="vorsichtige Rauschreduzierung",
        hinweis="Gut fuer Rauschen/Zischen. Fuer Musik vorsichtig einsetzen.",
    ),
    "demucs": ModellSetup(
        name="Demucs",
        pip_pakete=["demucs"],
        python_imports=["demucs"],
        cli_befehle=["demucs"],
        zweck="Stem-Analyse und Trennung von Drums/Bass/Rest",
        hinweis="Hilft bei Analyse, ist aber kein automatischer Qualitaetsfilter.",
    ),
    "matchering": ModellSetup(
        name="Matchering",
        pip_pakete=["matchering"],
        python_imports=["matchering"],
        cli_befehle=["matchering"],
        zweck="optionales Referenz-Mastering",
        hinweis="Braucht spaeter eine passende Referenz-Audio.",
    ),
    "audiosr": ModellSetup(
        name="AudioSR",
        pip_pakete=["audiosr"],
        python_imports=["audiosr"],
        cli_befehle=["audiosr"],
        zweck="experimentelle Audio-Super-Resolution",
        hinweis="Gross und experimentell. Erst testen, wenn die Basis stabil ist.",
        empfohlen=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optionale Audio-Modelle lokal einrichten.")
    parser.add_argument(
        "--install",
        nargs="+",
        choices=tuple(MODELLE) + ("all",),
        default=[],
        help="Zu installierende Tools. Ohne --install wird nur der Status geprueft.",
    )
    parser.add_argument(
        "--auch-experimentell",
        action="store_true",
        help="Erlaubt AudioSR bei --install all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur anzeigen, welche Befehle ausgefuehrt wuerden.",
    )
    parser.add_argument(
        "--download",
        nargs="+",
        choices=("clap", "demucs", "all"),
        default=[],
        help="Modellgewichte lokal nach daten/modelle/audio_analyse herunterladen.",
    )
    return parser.parse_args()


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


def run(command: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def run_mit_env(command: List[str], env_update: Dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_update)
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )


def python_package_available(package_name: str) -> bool:
    result = run([str(PYTHON), "-c", f"import {package_name.replace('-', '_')}"])
    return result.returncode == 0


def find_command(names: List[str]) -> str | None:
    for name in names:
        direct = shutil.which(name)
        if direct:
            return direct
        local = VENV_BIN / name
        if local.exists() and local.is_file():
            return str(local)
    return None


def modell_status(setup: ModellSetup) -> Dict[str, Any]:
    cli_path = find_command(setup.cli_befehle)
    return {
        "name": setup.name,
        "zweck": setup.zweck,
        "cli_befehl": setup.cli_befehle[0],
        "cli_befehle": setup.cli_befehle,
        "cli_verfuegbar": bool(cli_path),
        "cli_pfad": cli_path,
        "pakete": setup.pip_pakete,
        "python_imports": setup.python_imports,
        "python_paket_verfuegbar": all(python_package_available(pkg) for pkg in setup.python_imports),
        "empfohlen": setup.empfohlen,
        "hinweis": setup.hinweis,
    }


def resolve_install_targets(args: argparse.Namespace) -> List[str]:
    targets = list(args.install)
    if "all" in targets:
        targets = [name for name, setup in MODELLE.items() if setup.empfohlen or args.auch_experimentell]
    return sorted(dict.fromkeys(targets))


def resolve_download_targets(args: argparse.Namespace) -> List[str]:
    targets = list(args.download)
    if "all" in targets:
        targets = ["clap", "demucs"]
    return sorted(dict.fromkeys(targets))


def install_modelle(targets: List[str], dry_run: bool) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for target in targets:
        setup = MODELLE[target]
        command = [str(PYTHON), "-m", "pip", "install", *setup.pip_pakete]
        print(f"Installiere: {setup.name}", flush=True)
        print("Befehl:      " + " ".join(command), flush=True)
        if dry_run:
            results.append(
                {
                    "target": target,
                    "name": setup.name,
                    "status": "planned",
                    "command": command,
                }
            )
            continue
        completed = run(command)
        results.append(
            {
                "target": target,
                "name": setup.name,
                "status": "ok" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "command": command,
                "output_tail": completed.stdout[-4000:],
            }
        )
        if completed.returncode != 0:
            print(f"Fehler bei {setup.name}. Details im Report.", flush=True)
        else:
            print(f"OK:          {setup.name}", flush=True)
    return results


def download_clap(dry_run: bool) -> Dict[str, Any]:
    """Laedt CLAP lokal fuer Text-Audio-Aehnlichkeitspruefung."""
    command = [
        str(PYTHON),
        "-c",
        (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id={CLAP_REPO_ID!r}, local_dir={str(CLAP_DIR)!r}, "
            "local_dir_use_symlinks=False)"
        ),
    ]
    print("Download: CLAP", flush=True)
    print(f"Ziel:     {rel(CLAP_DIR)}", flush=True)
    if dry_run:
        return {"target": "clap", "status": "planned", "repo_id": CLAP_REPO_ID, "local_dir": rel(CLAP_DIR)}
    CLAP_DIR.parent.mkdir(parents=True, exist_ok=True)
    completed = run(command)
    return {
        "target": "clap",
        "name": "CLAP HTSAT unfused",
        "status": "ok" if completed.returncode == 0 else "failed",
        "repo_id": CLAP_REPO_ID,
        "local_dir": rel(CLAP_DIR),
        "returncode": completed.returncode,
        "output_tail": completed.stdout[-4000:],
    }


def download_demucs(dry_run: bool) -> Dict[str, Any]:
    """Laedt Demucs-Gewichte lokal fuer Bass/Drums/Stem-Analyse."""
    command = [
        str(PYTHON),
        "-c",
        (
            "from demucs.pretrained import get_model; "
            f"model = get_model({DEMUCS_MODEL_NAME!r}); "
            "print(type(model).__name__)"
        ),
    ]
    env_update = {
        "TORCH_HOME": str(DEMUCS_CACHE / "torch"),
        "XDG_CACHE_HOME": str(DEMUCS_CACHE / "xdg"),
    }
    print("Download: Demucs", flush=True)
    print(f"Ziel:     {rel(DEMUCS_CACHE)}", flush=True)
    if dry_run:
        return {"target": "demucs", "status": "planned", "model_name": DEMUCS_MODEL_NAME, "local_dir": rel(DEMUCS_CACHE)}
    DEMUCS_CACHE.mkdir(parents=True, exist_ok=True)
    completed = run_mit_env(command, env_update)
    return {
        "target": "demucs",
        "name": f"Demucs {DEMUCS_MODEL_NAME}",
        "status": "ok" if completed.returncode == 0 else "failed",
        "model_name": DEMUCS_MODEL_NAME,
        "local_dir": rel(DEMUCS_CACHE),
        "returncode": completed.returncode,
        "output_tail": completed.stdout[-4000:],
    }


def download_modelle(targets: List[str], dry_run: bool) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for target in targets:
        if target == "clap":
            results.append(download_clap(dry_run))
        elif target == "demucs":
            results.append(download_demucs(dry_run))
    return results


def ordner_groesse_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return round(total / 1024 / 1024, 2)


def main() -> int:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = resolve_install_targets(args)
    download_targets = resolve_download_targets(args)

    print("Audio-Modelle lokal", flush=True)
    print("===================", flush=True)
    print(f"Python: {PYTHON}", flush=True)
    if targets:
        print("Install: " + ", ".join(MODELLE[item].name for item in targets), flush=True)
    else:
        print("Install: keine, nur Status", flush=True)
    if download_targets:
        print("Download: " + ", ".join(download_targets), flush=True)
    else:
        print("Download: keine", flush=True)

    install_results = install_modelle(targets, args.dry_run) if targets else []
    download_results = download_modelle(download_targets, args.dry_run) if download_targets else []
    status = {name: modell_status(setup) for name, setup in MODELLE.items()}
    report = {
        "created_at": now(),
        "python": str(PYTHON),
        "dry_run": bool(args.dry_run),
        "install_targets": targets,
        "download_targets": download_targets,
        "install_results": install_results,
        "download_results": download_results,
        "local_model_dirs": {
            "clap": {"path": rel(CLAP_DIR), "size_mb": ordner_groesse_mb(CLAP_DIR), "exists": CLAP_DIR.exists()},
            "demucs": {"path": rel(DEMUCS_CACHE), "size_mb": ordner_groesse_mb(DEMUCS_CACHE), "exists": DEMUCS_CACHE.exists()},
        },
        "status": status,
        "wichtiger_hinweis": (
            "Dieses Setup veraendert keine LoRA-Checkpoints und keine Datasets. "
            "Die Tools werden nur fuer lokale Audio-Nachbearbeitung vorbereitet."
        ),
    }
    report_path = REPORT_DIR / "setup_report.json"
    write_json(report_path, report)

    print("", flush=True)
    print("Status", flush=True)
    print("------", flush=True)
    for key, item in status.items():
        marker = "ok" if item["cli_verfuegbar"] or item["python_paket_verfuegbar"] else "fehlt"
        print(f"{MODELLE[key].name}: {marker}", flush=True)
    if download_results:
        print("", flush=True)
        print("Downloads", flush=True)
        print("---------", flush=True)
        for item in download_results:
            print(f"{item['target']}: {item['status']} -> {item.get('local_dir')}", flush=True)
    print(f"Report: {rel(report_path)}", flush=True)
    all_results = install_results + download_results
    return 0 if all(item.get("status") != "failed" for item in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
