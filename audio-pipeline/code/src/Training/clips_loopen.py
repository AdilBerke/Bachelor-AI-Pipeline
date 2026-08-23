#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from audio_loopen import LoopAnalyse, baue_loop_block


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (
        (parent / "code" / "configs" / "konfiguration.yaml").exists()
        or (parent / "konfiguration.yaml").exists()
    )
    and (parent / "code").exists()
)
SAMPLE_RATE = 32000
EPS = 1e-12


@dataclass(frozen=True)
class ClipInfo:

    path: Path
    genre: str
    slug: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baut eine lange Lofi-Audio aus 30s-Clips.")
    parser.add_argument(
        "--quelle",
        action="append",
        default=[],
        help="Audio-Datei oder Ordner mit MP3/WAV/FLAC-Clips. Kann mehrfach angegeben werden.",
    )
    parser.add_argument(
        "--genre",
        default="Alle Lofi Genres",
        help="Filtert die Quellen, z.B. Jazz Lofi, Dreamy Lofi oder Alle Lofi Genres.",
    )
    parser.add_argument("--dauer", default="2m", help="Zieldauer, z.B. 30s, 10m, 1h, 3h.")
    parser.add_argument("--name", default="")
    parser.add_argument(
        "--ausgabe-root",
        default=str(PROJECT_ROOT / "training" / "ausgaben" / "musicgen_loops"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="0 = jedes Mal neuer Zufall; positiver Wert = reproduzierbare Reihenfolge.",
    )
    parser.add_argument("--modus", choices=("zufall", "sortiert"), default="zufall")
    parser.add_argument("--genre-modus", choices=("genre_bloecke", "gemischt"), default="genre_bloecke")
    parser.add_argument("--block-sekunden", type=float, default=300.0)
    parser.add_argument("--block-variation-sekunden", type=float, default=30.0)
    parser.add_argument(
        "--wiederholung-erlauben",
        action="store_true",
        help="Erlaubt Clip-Wiederholungen, wenn die Zieldauer laenger als der Quellenpool ist.",
    )
    parser.add_argument("--crossfade-sekunden", type=float, default=5.0)
    parser.add_argument(
        "--interner-crossfade-sekunden",
        type=float,
        default=0.0,
        help="0 = automatisch ein Beat; sonst gewuenschter interner Loop-Crossfade.",
    )
    parser.add_argument("--ziel-bpm", type=float, default=78.0)
    parser.add_argument("--min-loop-sekunden", type=float, default=18.0)
    parser.add_argument("--max-loop-sekunden", type=float, default=29.0)
    parser.add_argument("--loop-suchfenster-sekunden", type=float, default=7.0)
    parser.add_argument(
        "--tempo-variation-prozent",
        type=float,
        default=1.0,
        help="Maximale pitch-erhaltende Tempoabweichung. 0 deaktiviert die Variation.",
    )
    parser.add_argument(
        "--tempo-phase-sekunden",
        type=float,
        default=300.0,
        help="Mindestabstand zwischen kleinen Tempowechseln.",
    )
    parser.add_argument("--fade-out-sekunden", type=float, default=3.0)
    parser.add_argument("--fade-in-sekunden", type=float, default=5.0)
    parser.add_argument("--ziel-rms-db", type=float, default=-22.0)
    parser.add_argument("--peak-limit", type=float, default=0.92)
    parser.add_argument("--kein-mp3", action="store_true")
    parser.add_argument(
        "--github-push",
        dest="github_push",
        action="store_true",
        default=True,
        help="Pusht die finale Audio nach erfolgreicher Erstellung isoliert nach main.",
    )
    parser.add_argument(
        "--kein-github-push",
        dest="github_push",
        action="store_false",
        help="Speichert die finale Audio nur lokal.",
    )
    parser.add_argument("--ausgabe", choices=("kompakt", "json"), default="kompakt")
    return parser.parse_args()


def parse_duration_seconds(value: str) -> float:
    text = value.strip().lower().replace(",", ".")
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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def actual_seed(requested_seed: int) -> int:

    if requested_seed > 0:
        return requested_seed
    return int(time.time_ns() ^ (os.getpid() << 16)) & 0xFFFFFFFF


def slug_to_genre(slug: str) -> str:
    words = [word for word in slug.replace("-", "_").split("_") if word]
    if not words:
        return "Unbekannt"
    keep_upper = {"lofi": "Lofi", "musicgen": "MusicGen"}
    return " ".join(keep_upper.get(word.lower(), word.capitalize()) for word in words)


def infer_genre_from_path(path: Path) -> tuple[str, str]:

    stem = path.stem
    if "__" in stem:
        slug = stem.split("__", 1)[1]
    else:
        slug = path.parent.parent.name if path.parent.name == "audio" else path.parent.name
    slug = slug.strip("_").lower() or "unbekannt"
    return slug_to_genre(slug), slug


def normalize_genre(value: str) -> str:

    text = value.strip().lower().replace("-", " ").replace("_", " ")
    replacements = {
        "chill lofi": "chillhop lofi",
        "jazz": "jazz lofi",
        "dreamy": "dreamy lofi",
        "study": "study lofi",
        "guitar": "guitar lofi",
    }
    text = replacements.get(text, text)
    return " ".join(text.split())


def collect_audio_files(sources: List[str], genre: str = "") -> List[ClipInfo]:
    requested_genre = genre
    if not sources:
        current_review = (
            PROJECT_ROOT
            / "training"
            / "bewertungen"
            / "musicgen"
            / "lora_review_001"
            / "audio"
        )
        old_review = PROJECT_ROOT / "training" / "bewertungen" / "musicgen" / "durchgang_010" / "audio"
        sources = [str(current_review if current_review.exists() else old_review)]

    files: List[Path] = []
    for source in sources:
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".flac"}:
            files.append(path.resolve())
        elif path.is_dir():
            for suffix in ("*.mp3", "*.wav", "*.flac"):
                files.extend(item.resolve() for item in path.rglob(suffix))
        else:
            raise FileNotFoundError(f"Quelle nicht gefunden: {path}")

    unique = sorted(dict.fromkeys(files))
    if not unique:
        raise FileNotFoundError("Keine MP3/WAV/FLAC-Dateien in den Quellen gefunden.")

    result: List[ClipInfo] = []
    for path in unique:
        inferred_genre, slug = infer_genre_from_path(path)
        result.append(ClipInfo(path=path, genre=inferred_genre, slug=slug))

    requested = normalize_genre(requested_genre)
    if requested and requested not in {"alle", "alle lofi genres", "lofi"}:
        filtered = [item for item in result if normalize_genre(item.genre) == requested]
        if not filtered:
            available = ", ".join(sorted({item.genre for item in result}))
            raise ValueError(
                f"Keine Quellen fuer Genre '{requested_genre}' gefunden. Verfuegbar: {available}"
            )
        result = filtered
    return result


def decode_audio(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ],
        check=True,
        capture_output=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    audio = audio[np.isfinite(audio)]
    if len(audio) == 0:
        raise ValueError(f"Audio ist leer oder nicht lesbar: {path}")
    return audio


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


def db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), EPS))


def normalize_clip(audio: np.ndarray, target_rms_db: float, peak_limit: float) -> np.ndarray:
    current_rms = rms(audio)
    if current_rms <= EPS:
        return audio.astype(np.float32)
    target_rms = 10.0 ** (target_rms_db / 20.0)
    gain = target_rms / current_rms
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak * gain > peak_limit:
        gain = peak_limit / max(peak, EPS)
    return (audio * gain).astype(np.float32)


def append_with_crossfade(base: np.ndarray, clip: np.ndarray, crossfade_frames: int) -> np.ndarray:
    if len(base) == 0:
        return clip.copy()
    fade_frames = min(crossfade_frames, len(base) // 2, len(clip) // 2)
    if fade_frames <= 0:
        return np.concatenate([base, clip])
    adjusted_clip = clip.astype(np.float32, copy=True)
    base_rms = rms(base[-fade_frames:])
    clip_rms = rms(adjusted_clip[:fade_frames])
    if base_rms > EPS and clip_rms > EPS:


        start_gain = min(10.0 ** (3.0 / 20.0), max(10.0 ** (-3.0 / 20.0), base_rms / clip_rms))
        release_frames = min(len(adjusted_clip), max(fade_frames, SAMPLE_RATE * 20))
        gain_curve = np.linspace(start_gain, 1.0, release_frames, dtype=np.float32)
        adjusted_clip[:release_frames] *= gain_curve
    curve = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
    fade_out = np.cos(curve * math.pi / 2.0)
    fade_in = np.sin(curve * math.pi / 2.0)
    overlap = base[-fade_frames:] * fade_out + adjusted_clip[:fade_frames] * fade_in
    return np.concatenate([base[:-fade_frames], overlap, adjusted_clip[fade_frames:]])


def extend_clip_to_block(
    clip: np.ndarray,
    block_frames: int,
    args: argparse.Namespace,
    seed: int,
    tempo_start_phase: int,
) -> tuple[np.ndarray, LoopAnalyse, int]:

    return baue_loop_block(
        clip,
        block_frames / SAMPLE_RATE,
        sample_rate=SAMPLE_RATE,
        ziel_bpm=args.ziel_bpm,
        min_loop_sec=args.min_loop_sekunden,
        max_loop_sec=args.max_loop_sekunden,
        suchfenster_sec=args.loop_suchfenster_sekunden,
        crossfade_sec=args.interner_crossfade_sekunden,
        tempo_variation_percent=args.tempo_variation_prozent,
        tempo_phase_sec=args.tempo_phase_sekunden,
        tempo_start_phase=tempo_start_phase,
        seed=seed,
    )


def apply_final_fade(audio: np.ndarray, fade_out_frames: int) -> np.ndarray:
    if fade_out_frames <= 0 or len(audio) <= fade_out_frames:
        return audio
    result = audio.copy()
    curve = np.linspace(1.0, 0.0, fade_out_frames, dtype=np.float32)
    result[-fade_out_frames:] *= curve
    return result


def apply_initial_fade(audio: np.ndarray, fade_in_frames: int) -> np.ndarray:

    if fade_in_frames <= 0 or len(audio) <= fade_in_frames:
        return audio
    result = audio.copy()
    curve = np.sin(
        np.linspace(0.0, math.pi / 2.0, fade_in_frames, dtype=np.float32)
    )
    result[:fade_in_frames] *= curve
    return result


def limit_peak(audio: np.ndarray, peak_limit: float) -> np.ndarray:

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= peak_limit or peak <= EPS:
        return audio
    return (audio * (peak_limit / peak)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def encode_mp3(wav_path: Path, mp3_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(mp3_path),
        ],
        check=True,
    )


def prepare_pools(files: List[ClipInfo], rng: random.Random) -> Dict[str, List[ClipInfo]]:
    pools: Dict[str, List[ClipInfo]] = {}
    for item in files:
        pools.setdefault(item.slug, []).append(item)
    for items in pools.values():
        rng.shuffle(items)
    return pools


def choose_genre(pools: Dict[str, List[ClipInfo]], rng: random.Random, previous_slug: Optional[str]) -> str:
    available = [slug for slug, items in pools.items() if items]
    if not available:
        raise RuntimeError("Keine verfuegbaren Clips mehr.")
    if len(available) > 1 and previous_slug in available:
        available = [slug for slug in available if slug != previous_slug]
    return rng.choice(available)


def choose_next_clip(
    pools: Dict[str, List[ClipInfo]],
    all_files: List[ClipInfo],
    rng: random.Random,
    previous_slug: Optional[str],
    previous_path: Optional[Path],
    modus: str,
    genre_modus: str,
    block_index: int,
) -> tuple[ClipInfo, int]:

    pool_reset = 0
    if not any(pools.values()):
        pools.update(prepare_pools(all_files, rng))
        pool_reset = 1

    if modus == "sortiert":
        available = [item for items in pools.values() for item in items]
        candidates = [item for item in available if item.path != previous_path] or available
        selected = candidates[block_index % len(candidates)]
        pools[selected.slug].remove(selected)
        return selected, pool_reset

    if genre_modus == "gemischt":
        available = [item for items in pools.values() for item in items]
        candidates = [item for item in available if item.path != previous_path] or available
        selected = rng.choice(candidates)
        pools[selected.slug].remove(selected)
        return selected, pool_reset

    slug = choose_genre(pools, rng, previous_slug)
    candidates = [item for item in pools[slug] if item.path != previous_path] or pools[slug]
    selected = rng.choice(candidates)
    pools[slug].remove(selected)
    return selected, pool_reset


def plan_block_net_frames(
    args: argparse.Namespace,
    rng: random.Random,
    target_frames: int,
    source_count: int,
) -> tuple[List[int], float]:

    requested_min_sec = max(1.0, args.block_sekunden)
    if not args.wiederholung_erlauben and source_count > 0:
        needed_sec_without_reuse = target_frames / SAMPLE_RATE / source_count
        effective_min_sec = max(requested_min_sec, needed_sec_without_reuse)
    else:
        effective_min_sec = requested_min_sec

    min_frames = int(round(effective_min_sec * SAMPLE_RATE))
    variation = max(0.0, args.block_variation_sekunden)
    if variation <= 0.0:


        block_count = max(1, int(round(target_frames / max(1, min_frames))))
        basis = target_frames // block_count
        rest = target_frames % block_count
        durations = [basis + (1 if index < rest else 0) for index in range(block_count)]
        return durations, effective_min_sec

    durations: List[int] = []
    remaining_frames = target_frames

    while remaining_frames > 0:
        if not durations and remaining_frames <= min_frames:
            durations.append(remaining_frames)
            break

        if durations and remaining_frames < min_frames:
            durations[-1] += remaining_frames
            break

        if remaining_frames <= min_frames * 2:
            durations.append(remaining_frames)
            break

        if variation > 0:
            block_seconds = effective_min_sec + rng.uniform(0.0, variation)
            block_frames = int(round(block_seconds * SAMPLE_RATE))
        else:
            block_frames = min_frames

        if remaining_frames - block_frames < min_frames:
            durations.append(remaining_frames)
            break

        durations.append(block_frames)
        remaining_frames -= block_frames

    return durations, effective_min_sec


def write_sequence(run_dir: Path, sequence: List[Dict[str, object]]) -> None:
    if not sequence:
        return
    with (run_dir / "ablauf.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence[0].keys()))
        writer.writeheader()
        writer.writerows(sequence)


def main() -> int:
    args = parse_args()
    duration_sec = parse_duration_seconds(args.dauer)
    if duration_sec <= 0:
        raise ValueError("Dauer muss groesser als 0 sein.")
    if args.block_sekunden < 30:
        raise ValueError("block-sekunden sollte mindestens 30 sein.")

    seed = actual_seed(args.seed)
    rng = random.Random(seed)
    files = collect_audio_files(args.quelle, args.genre)
    pools = prepare_pools(files, rng)

    run_name = args.name or f"lange_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.ausgabe_root).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    target_frames = int(round(duration_sec * SAMPLE_RATE))
    crossfade_frames = int(round(args.crossfade_sekunden * SAMPLE_RATE))
    fade_out_frames = int(round(args.fade_out_sekunden * SAMPLE_RATE))
    fade_in_frames = int(round(args.fade_in_sekunden * SAMPLE_RATE))
    planned_net_blocks, effective_block_seconds = plan_block_net_frames(
        args=args,
        rng=rng,
        target_frames=target_frames,
        source_count=len(files),
    )

    built = np.zeros(0, dtype=np.float32)
    sequence: List[Dict[str, object]] = []
    previous_slug: Optional[str] = None
    previous_path: Optional[Path] = None
    pool_resets = 0
    used_paths: set[Path] = set()
    loop_cache: Dict[Path, tuple[np.ndarray, np.ndarray, LoopAnalyse]] = {}
    loop_analyses: Dict[Path, Dict[str, Any]] = {}

    for block_index, net_block_frames in enumerate(planned_net_blocks):


        audio_block_frames = net_block_frames if len(built) == 0 else net_block_frames + crossfade_frames
        selected, reset_count = choose_next_clip(
            pools=pools,
            all_files=files,
            rng=rng,
            previous_slug=previous_slug,
            previous_path=previous_path,
            modus=args.modus,
            genre_modus=args.genre_modus,
            block_index=block_index,
        )
        pool_resets += reset_count
        if selected.path in loop_cache:
            raw, clip, loop_analysis = loop_cache[selected.path]
            block, loop_analysis, loops = baue_loop_block(
                clip,
                audio_block_frames / SAMPLE_RATE,
                sample_rate=SAMPLE_RATE,
                ziel_bpm=args.ziel_bpm,
                min_loop_sec=args.min_loop_sekunden,
                max_loop_sec=args.max_loop_sekunden,
                suchfenster_sec=args.loop_suchfenster_sekunden,
                crossfade_sec=args.interner_crossfade_sekunden,
                tempo_variation_percent=args.tempo_variation_prozent,
                tempo_phase_sec=args.tempo_phase_sekunden,
                tempo_start_phase=block_index,
                seed=seed,
                analyse=loop_analysis,
            )
        else:
            raw = decode_audio(selected.path)
            clip = normalize_clip(raw, args.ziel_rms_db, args.peak_limit)
            block, loop_analysis, loops = extend_clip_to_block(
                clip,
                audio_block_frames,
                args,
                seed,
                block_index,
            )
            loop_cache[selected.path] = (raw, clip, loop_analysis)
            loop_analyses[selected.path] = {
                "datei": rel(selected.path),
                "genre": selected.genre,
                "source_duration_sec": round(len(raw) / SAMPLE_RATE, 3),
                **loop_analysis.als_dict(),
            }

        previous_end_sec = len(built) / SAMPLE_RATE
        start_sec = (
            0.0
            if len(built) == 0
            else max(0.0, (len(built) - crossfade_frames) / SAMPLE_RATE)
        )
        built = append_with_crossfade(built, block, crossfade_frames)
        end_sec = len(built) / SAMPLE_RATE
        sequence.append(
            {
                "block": block_index + 1,
                "genre": selected.genre,
                "datei": rel(selected.path),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "block_duration_sec": round((end_sec - start_sec), 3),
                "net_contribution_sec": round(end_sec - previous_end_sec, 3),
                "audio_material_duration_sec": round(audio_block_frames / SAMPLE_RATE, 3),
                "source_clip_duration_sec": round(len(raw) / SAMPLE_RATE, 3),
                "interne_loops": loops,
                "loop_start_sec": loop_analysis.start_sec,
                "loop_end_sec": loop_analysis.end_sec,
                "loop_duration_sec": loop_analysis.loop_duration_sec,
                "loop_crossfade_sec": loop_analysis.crossfade_sec,
                "loop_bpm": loop_analysis.bpm,
                "loop_seam_score": loop_analysis.seam_score,
                "loop_status": loop_analysis.status,
                "tempo_variation_percent": loop_analysis.tempo_variation_percent,
                "tempo_phase_sec": loop_analysis.tempo_phase_sec,
                "tempo_factors": loop_analysis.tempo_factors,
                "clip_rms_db_after_norm": round(db(rms(clip)), 3),
                "wiederverwendet_nach_pool_reset": selected.path in used_paths,
            }
        )
        used_paths.add(selected.path)
        previous_slug = selected.slug
        previous_path = selected.path
        progress = (block_index + 1) / max(1, len(planned_net_blocks)) * 100.0
        print(
            f"\rLooping: {progress:6.2f}% | Block {block_index + 1}/{len(planned_net_blocks)}",
            end="",
            flush=True,
        )

    if planned_net_blocks:
        print("", flush=True)
    built = built[:target_frames]
    built = apply_initial_fade(built, fade_in_frames)
    built = apply_final_fade(built, fade_out_frames)
    built = limit_peak(built, args.peak_limit)
    wav_path = run_dir / "lange_audio.wav"
    write_wav(wav_path, built)
    mp3_path = None
    if not args.kein_mp3:
        mp3_path = run_dir / "lange_audio.mp3"
        encode_mp3(wav_path, mp3_path)

    github_result: Dict[str, Any] = {"status": "disabled"}
    if args.github_push:
        try:
            from audio_veroeffentlichen import veroeffentliche_audio

            github_result = veroeffentliche_audio(
                mp3_path or wav_path,
                branch="main",
                commit_text=f"Audio: {args.genre} {args.dauer}",
            )
            if github_result.get("status") == "pushed":
                print(f"GitHub: {github_result.get('url', 'Audio gepusht')}", flush=True)
            else:
                print(
                    "Warnung: Audio lokal fertig, GitHub-Push aber fehlgeschlagen: "
                    f"{github_result.get('error', github_result.get('status'))}",
                    flush=True,
                )
        except Exception as exc:
            github_result = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                "Warnung: Audio lokal fertig, GitHub-Push aber fehlgeschlagen: "
                f"{github_result['error']}",
                flush=True,
            )

    genre_counts: Dict[str, int] = {}
    for row in sequence:
        genre = str(row["genre"])
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": round(len(built) / SAMPLE_RATE, 3),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "requested_seed": args.seed,
        "seed": seed,
        "modus": args.modus,
        "genre_modus": args.genre_modus,
        "requested_genre": args.genre,
        "source_count": len(files),
        "unique_sources_used": len(used_paths),
        "source_pool_resets": pool_resets,
        "sources": [rel(item.path) for item in files],
        "crossfade_sec": args.crossfade_sekunden,
        "internal_crossfade_requested_sec": args.interner_crossfade_sekunden,
        "internal_crossfade_strategy": "0 = automatisch ein Beat, maximal 1.25 Sekunden",
        "loop_strategy": "taktnahe Grenzensuche mit technischer Nahtbewertung",
        "target_bpm": args.ziel_bpm,
        "min_loop_seconds": args.min_loop_sekunden,
        "max_loop_seconds": args.max_loop_sekunden,
        "loop_search_window_seconds": args.loop_suchfenster_sekunden,
        "tempo_variation_percent": args.tempo_variation_prozent,
        "tempo_phase_seconds": args.tempo_phase_sekunden,
        "tempo_variation_strategy": "pitch-erhaltend und schrittweise",
        "loop_analysis_ok": sum(1 for row in loop_analyses.values() if row.get("status") == "OK"),
        "loop_analysis_warning": sum(
            1 for row in loop_analyses.values() if row.get("status") == "WARNUNG"
        ),
        "block_seconds_min": args.block_sekunden,
        "effective_block_seconds_min": round(effective_block_seconds, 3),
        "block_variation_seconds": args.block_variation_sekunden,
        "wiederholung_erlauben": args.wiederholung_erlauben,
        "fade_in_sec": args.fade_in_sekunden,
        "fade_out_sec": args.fade_out_sekunden,
        "target_rms_db": args.ziel_rms_db,
        "peak_limit": args.peak_limit,
        "final_peak": round(float(np.max(np.abs(built))) if len(built) else 0.0, 6),
        "genre_counts": genre_counts,
        "output_wav": rel(wav_path),
        "output_mp3": rel(mp3_path) if mp3_path else None,
        "github_push": github_result,
        "sequence_count": len(sequence),
        "hinweis": (
            "Longform nutzt laengere Rhythmus-/Genre-Bloecke. Ein Quellclip wird "
            "innerhalb eines Blocks geloopt, aber als eigener Block nicht wiederholt, "
            "solange genuegend Quellen vorhanden sind."
        ),
    }
    (run_dir / "bericht.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_sequence(run_dir, sequence)

    loop_rows = list(loop_analyses.values())
    if loop_rows:
        with (run_dir / "loop_analyse.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(loop_rows[0].keys()))
            writer.writeheader()
            writer.writerows(loop_rows)

    if args.ausgabe == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    else:
        print("Status: fertig", flush=True)
        print(f"Dauer:  {report['duration_sec']:.1f}s", flush=True)
        print(f"Bloecke: {len(sequence)}", flush=True)
        print(f"Loops:   {sum(int(row['interne_loops']) for row in sequence)}", flush=True)
        print(f"Audio:   {report['output_mp3'] or report['output_wav']}", flush=True)
        print(f"Report:  {rel(run_dir / 'bericht.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
