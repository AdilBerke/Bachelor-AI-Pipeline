#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote


PROJEKTWURZEL = Path(__file__).resolve().parents[3]
GITHUB_DIREKT_LIMIT = 95 * 1024 * 1024
ERLAUBTE_ENDUNGEN = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def git_befehl(
    argumente: list[str],
    *,
    arbeitsordner: Path,
    pruefen: bool = True,
) -> subprocess.CompletedProcess[str]:

    umgebung = os.environ.copy()
    umgebung["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["git", *argumente],
        cwd=arbeitsordner,
        env=umgebung,
        text=True,
        capture_output=True,
        check=pruefen,
    )


def projektpfad(audio_path: Path) -> Path:

    path = audio_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio nicht gefunden: {path}")
    if path.suffix.lower() not in ERLAUBTE_ENDUNGEN:
        raise ValueError(f"Nicht unterstuetzte Audio-Endung: {path.suffix}")
    try:
        return path.relative_to(PROJEKTWURZEL)
    except ValueError as exc:
        raise ValueError("Es werden nur Audios innerhalb des Projekts veroeffentlicht.") from exc


def audio_dateien_aus_eingabe(path: str | Path) -> list[Path]:

    eingabe = Path(path).expanduser().resolve()
    if eingabe.is_file():
        projektpfad(eingabe)
        return [eingabe]
    if not eingabe.is_dir():
        raise FileNotFoundError(f"Audio-Datei oder Ordner nicht gefunden: {eingabe}")
    dateien = [
        item
        for item in sorted(eingabe.rglob("*"))
        if item.is_file() and item.suffix.lower() in ERLAUBTE_ENDUNGEN
    ]
    if not dateien:
        raise FileNotFoundError(f"Keine Audiodateien im Ordner gefunden: {eingabe}")
    for item in dateien:
        projektpfad(item)
    return dateien


def github_basis_url(remote_url: str) -> str:

    value = remote_url.strip()
    if value.endswith(".git"):
        value = value[:-4]
    if value.startswith("git@github.com:"):
        return "https://github.com/" + value.split(":", 1)[1]
    if value.startswith("ssh://git@github.com/"):
        return "https://github.com/" + value.split("github.com/", 1)[1]
    return value


def datei_verknuepfen_oder_kopieren(source: Path, target: Path) -> None:

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def lfs_fuer_endung_aktivieren(worktree: Path, endung: str) -> None:

    git_befehl(["lfs", "version"], arbeitsordner=worktree)
    git_befehl(["lfs", "install", "--local"], arbeitsordner=worktree)
    git_befehl(["lfs", "track", f"*{endung.lower()}"], arbeitsordner=worktree)
    git_befehl(["add", "--", ".gitattributes"], arbeitsordner=worktree)


def veroeffentliche_audio(
    audio: str | Path,
    *,
    remote: str = "origin",
    branch: str = "main",
    commit_text: str = "",
    nur_plan: bool = False,
    versuche: int = 3,
) -> dict[str, Any]:

    return veroeffentliche_audios(
        [Path(audio).expanduser().resolve()],
        remote=remote,
        branch=branch,
        commit_text=commit_text,
        nur_plan=nur_plan,
        versuche=versuche,
    )


def veroeffentliche_audios(
    audios: list[str | Path],
    *,
    remote: str = "origin",
    branch: str = "main",
    commit_text: str = "",
    nur_plan: bool = False,
    versuche: int = 3,
) -> dict[str, Any]:

    audio_paths = [Path(item).expanduser().resolve() for item in audios]
    if not audio_paths:
        raise FileNotFoundError("Keine Audios zum Veroeffentlichen angegeben.")
    relative_paths = [projektpfad(item) for item in audio_paths]
    file_sizes = [item.stat().st_size for item in audio_paths]
    lfs_suffixes = sorted(
        {
            item.suffix.lower()
            for item, size in zip(audio_paths, file_sizes)
            if size > GITHUB_DIREKT_LIMIT
        }
    )
    lfs_required = bool(lfs_suffixes)
    total_size = sum(file_sizes)
    remote_url = git_befehl(
        ["remote", "get-url", remote],
        arbeitsordner=PROJEKTWURZEL,
    ).stdout.strip()
    first_audio = audio_paths[0]
    message = commit_text.strip() or (
        f"Audios: {first_audio.parent.name}" if len(audio_paths) > 1 else f"Audio: {first_audio.parent.name}"
    )
    plan = {
        "status": "planned" if nur_plan else "running",
        "audio": str(relative_paths[0]),
        "audios": [str(item) for item in relative_paths],
        "count": len(audio_paths),
        "size_bytes": total_size,
        "remote": remote,
        "branch": branch,
        "git_lfs_required": lfs_required,
        "git_lfs_suffixes": lfs_suffixes,
        "commit_message": message,
    }
    if nur_plan:
        return plan

    last_error = ""
    for attempt in range(1, max(1, versuche) + 1):
        temp_root = Path(tempfile.mkdtemp(prefix="musicgen-github-"))
        worktree = temp_root / "repo"
        worktree_registered = False
        try:
            git_befehl(["fetch", remote, branch], arbeitsordner=PROJEKTWURZEL)
            git_befehl(
                ["worktree", "add", "--detach", str(worktree), f"{remote}/{branch}"],
                arbeitsordner=PROJEKTWURZEL,
            )
            worktree_registered = True

            for source, relative_path in zip(audio_paths, relative_paths):
                target = worktree / relative_path
                datei_verknuepfen_oder_kopieren(source, target)
            for suffix in lfs_suffixes:
                lfs_fuer_endung_aktivieren(worktree, suffix)
            git_befehl(
                ["add", "-f", "--", *[str(item) for item in relative_paths]],
                arbeitsordner=worktree,
            )

            staged = {
                line.strip()
                for line in git_befehl(
                    ["diff", "--cached", "--name-only"],
                    arbeitsordner=worktree,
                ).stdout.splitlines()
                if line.strip()
            }
            erlaubt = {str(item) for item in relative_paths}
            if lfs_required:
                erlaubt.add(".gitattributes")
            unerwartet = staged - erlaubt
            if unerwartet:
                raise RuntimeError(
                    "Sicherheitsabbruch: unerwartete Dateien im Audio-Commit: "
                    + ", ".join(sorted(unerwartet))
                )
            fehlend = erlaubt - staged
            fehlend.discard(".gitattributes")
            if fehlend:
                raise RuntimeError(
                    "Nicht alle Audios wurden fuer den Commit erfasst: "
                    + ", ".join(sorted(fehlend))
                )

            git_befehl(["commit", "-m", message], arbeitsordner=worktree)
            commit = git_befehl(["rev-parse", "HEAD"], arbeitsordner=worktree).stdout.strip()
            push = git_befehl(
                ["push", remote, f"HEAD:{branch}"],
                arbeitsordner=worktree,
                pruefen=False,
            )
            if push.returncode != 0:
                last_error = (push.stderr or push.stdout).strip()
                continue

            common_parent = Path(os.path.commonpath([str(item.parent) for item in relative_paths]))
            web_url = f"{github_basis_url(remote_url)}/tree/{quote(branch)}/{quote(str(common_parent), safe='/')}"
            return {
                **plan,
                "status": "pushed",
                "attempt": attempt,
                "commit": commit,
                "url": web_url,
                "pushed_at": datetime.now().isoformat(timespec="seconds"),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        finally:
            if worktree_registered:
                git_befehl(
                    ["worktree", "remove", "--force", str(worktree)],
                    arbeitsordner=PROJEKTWURZEL,
                    pruefen=False,
                )
            shutil.rmtree(temp_root, ignore_errors=True)
            git_befehl(["worktree", "prune"], arbeitsordner=PROJEKTWURZEL, pruefen=False)

    return {
        **plan,
        "status": "failed",
        "attempts": max(1, versuche),
        "error": last_error or "Unbekannter Git-Push-Fehler.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pusht Audiodateien isoliert nach GitHub.")
    parser.add_argument("audio", help="Audiodatei oder Ordner mit Audiodateien.")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--commit-text", default="")
    parser.add_argument("--nur-plan", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_paths = audio_dateien_aus_eingabe(args.audio)
    result = veroeffentliche_audios(
        audio_paths,
        remote=args.remote,
        branch=args.branch,
        commit_text=args.commit_text,
        nur_plan=args.nur_plan,
    )
    print(f"Status: {result['status']}")
    print(f"Audios: {result.get('count', 1)}")
    print(f"Pfad:   {result['audio']}")
    if result.get("url"):
        print(f"GitHub: {result['url']}")
    if result.get("error"):
        print(f"Fehler: {result['error']}")
    return 0 if result["status"] in {"planned", "pushed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
