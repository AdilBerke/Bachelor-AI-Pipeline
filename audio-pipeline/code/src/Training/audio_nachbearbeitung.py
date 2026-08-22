#!/usr/bin/env python3
"""Optionale Nachbearbeitung fuer generierte MusicGen-Audios.

Diese Stufe laeuft bewusst nach der MusicGen-/LoRA-Generierung. Sie veraendert
keine LoRA-Adapter, keine Checkpoints und keine Trainingsdaten. Das Original-
Audio bleibt erhalten; bearbeitete Dateien und Reports werden in einen eigenen
Ausgabeordner geschrieben.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "code").exists() and (parent / "training").exists()
)
VENV_BIN = PROJECT_ROOT / ".venv" / "bin"
EPS = 1e-12


@dataclass(frozen=True)
class FilterProfil:
    """Filterwerte fuer eine vorsichtige, reproduzierbare Nachbearbeitung."""

    highpass_hz: float
    lowpass_hz: float
    bass_gain_db: float
    high_gain_db: float
    compressor_threshold_db: float
    compressor_ratio: float
    loudness_i: float
    true_peak_db: float
    limiter: float


FILTER_PROFILE: Dict[str, FilterProfil] = {
    "vorsichtig": FilterProfil(
        highpass_hz=30.0,
        lowpass_hz=13500.0,
        bass_gain_db=-1.3,
        high_gain_db=-2.4,
        compressor_threshold_db=-18.0,
        compressor_ratio=1.35,
        loudness_i=-16.0,
        true_peak_db=-1.5,
        limiter=0.92,
    ),
    "normal": FilterProfil(
        highpass_hz=35.0,
        lowpass_hz=12500.0,
        bass_gain_db=-2.3,
        high_gain_db=-4.0,
        compressor_threshold_db=-20.0,
        compressor_ratio=1.7,
        loudness_i=-16.0,
        true_peak_db=-1.5,
        limiter=0.90,
    ),
    "stark": FilterProfil(
        highpass_hz=40.0,
        lowpass_hz=11500.0,
        bass_gain_db=-3.5,
        high_gain_db=-6.0,
        compressor_threshold_db=-22.0,
        compressor_ratio=2.1,
        loudness_i=-17.0,
        true_peak_db=-1.8,
        limiter=0.88,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio analysieren und optional nachbearbeiten.")
    parser.add_argument("--input", required=True, help="Eingabe-Audio, z.B. lange_audio.mp3.")
    parser.add_argument("--output-dir", required=True, help="Ordner fuer bearbeitete Audios und Reports.")
    parser.add_argument(
        "--intensitaet",
        choices=tuple(FILTER_PROFILE),
        default="vorsichtig",
        help="Staerke der Standardfilter.",
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
    parser.add_argument("--audio-sr", action="store_true", help="Experimentelle AudioSR-Stufe, falls lokal vorhanden.")
    parser.add_argument(
        "--nur-pruefen",
        action="store_true",
        help="Nur Analyse/Reports schreiben, keine bearbeitete Audio erzeugen.",
    )
    parser.add_argument(
        "--nur-lokale-tools",
        action="store_true",
        help="Dokumentiert, dass keine Downloads oder Online-Dienste verwendet werden.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_command(command: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd or PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def require_tool(name: str) -> str:
    path = find_command([name])
    if not path:
        raise RuntimeError(f"Pflichttool fehlt: {name}")
    return str(path)


def find_command(names: Iterable[str]) -> Optional[Path]:
    """Findet CLI-Tools auch dann, wenn `.venv/bin` nicht im PATH liegt."""

    for name in names:
        direct = shutil.which(name)
        if direct:
            return Path(direct)
        local = VENV_BIN / name
        if local.exists() and local.is_file():
            return local
    return None


def optional_tools() -> Dict[str, Dict[str, Any]]:
    """Erkennt lokale Zusatztools, ohne etwas herunterzuladen."""

    mapping = {
        "ffmpeg": (["ffmpeg"], []),
        "ffprobe": (["ffprobe"], []),
        "deepfilternet": (["deepFilter", "deep-filter-py", "deep-filter"], ["df"]),
        "demucs": (["demucs"], ["demucs"]),
        "matchering": (["matchering"], ["matchering"]),
        "audiosr": (["audiosr"], ["audiosr"]),
    }
    result: Dict[str, Dict[str, Any]] = {}
    for label, (commands, import_names) in mapping.items():
        detected = find_command(commands)
        python_available = all(importlib.util.find_spec(name) is not None for name in import_names)
        result[label] = {
            "command": commands[0],
            "command_candidates": commands,
            "available": bool(detected) or python_available,
            "cli_available": bool(detected),
            "path": str(detected) if detected else None,
            "python_imports": import_names,
            "python_available": python_available,
        }
    return result


def ffprobe_info(path: Path) -> Dict[str, Any]:
    ffprobe = require_tool("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=sample_rate,channels,codec_name",
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe konnte Audio nicht lesen: {result.stdout.strip()}")
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    first_audio = streams[0] if streams else {}
    duration = None
    try:
        duration = float((payload.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    return {
        "duration_sec": duration,
        "sample_rate": int(first_audio.get("sample_rate") or 0),
        "channels": int(first_audio.get("channels") or 0),
        "codec_name": first_audio.get("codec_name"),
    }


def decode_to_wav(input_path: Path, wav_path: Path, sample_rate: int = 32000) -> None:
    ffmpeg = require_tool("ffmpeg")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode fehlgeschlagen: {result.stdout.strip()}")


def load_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise RuntimeError(f"Nur 16-bit WAV wird erwartet, erhalten: {width * 8} bit")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), EPS))


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def window_rms_db(audio: np.ndarray, sample_rate: int, seconds: float = 1.0) -> List[float]:
    size = max(1, int(round(sample_rate * seconds)))
    values: List[float] = []
    for start in range(0, len(audio), size):
        segment = audio[start : start + size]
        if len(segment) < size // 2:
            continue
        values.append(db(rms(segment)))
    return values


def spectral_ratio(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    if audio.size < sample_rate:
        return 0.0
    segment = audio[: min(len(audio), sample_rate * 30)]
    window = np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(segment * window))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sample_rate)
    mask_total = (freqs >= 20.0) & (freqs <= min(sample_rate / 2.0, 16000.0))
    mask_band = (freqs >= low_hz) & (freqs <= high_hz)
    total = float(np.sum(spectrum[mask_total]))
    if total <= EPS:
        return 0.0
    return float(np.sum(spectrum[mask_band]) / total)


def quality_warnings(metrics: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if metrics["peak"] >= 0.98 or metrics["clipping_ratio"] > 0.001:
        warnings.append("Clipping/Peak sehr hoch")
    if metrics["rms_db"] < -34.0:
        warnings.append("Audio insgesamt leise")
    if metrics["silent_second_count"] > 0:
        warnings.append("Sekunden mit wenig/keinem Ton erkannt")
    if metrics["longest_silent_seconds"] >= 2:
        warnings.append("Laengere stille Stelle erkannt")
    if metrics["second_half_drop_db"] > 8.0:
        warnings.append("Zweite Haelfte deutlich leiser")
    if metrics["bass_ratio"] > 0.55:
        warnings.append("Bass sehr dominant")
    if metrics["high_ratio"] > 0.16:
        warnings.append("Hoehen/Shaker auffaellig stark")
    if metrics["loudness_range_db"] > 14.0:
        warnings.append("Starke Lautheitsschwankungen")
    if metrics["noise_floor_db"] > -42.0 and metrics["rms_db"] < -20.0:
        warnings.append("Moegliches Rauschen in leisen Passagen")
    return warnings


def analyse_audio(input_path: Path, work_dir: Path, label: str) -> Dict[str, Any]:
    decoded = work_dir / f"{label}_analyse.wav"
    probe = ffprobe_info(input_path)
    decode_to_wav(input_path, decoded, 32000)
    audio, sample_rate = load_wav_mono(decoded)
    duration = len(audio) / float(sample_rate) if sample_rate else 0.0
    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio)) if audio.size else 0.0
    rms_value = rms(audio)
    per_second = window_rms_db(audio, sample_rate, 1.0)
    silent_flags = [value < -52.0 for value in per_second]
    longest_silent = 0
    current_silent = 0
    for item in silent_flags:
        if item:
            current_silent += 1
            longest_silent = max(longest_silent, current_silent)
        else:
            current_silent = 0
    half = len(audio) // 2
    first_half_db = db(rms(audio[:half])) if half else -240.0
    second_half_db = db(rms(audio[half:])) if half else -240.0
    noise_floor_db = float(np.percentile(per_second, 15)) if per_second else -240.0
    loudness_range = float(np.percentile(per_second, 95) - np.percentile(per_second, 5)) if len(per_second) >= 3 else 0.0
    metrics = {
        "input": rel(input_path),
        "duration_sec": round(duration, 3),
        "sample_rate": sample_rate,
        "source_sample_rate": probe.get("sample_rate"),
        "source_channels": probe.get("channels"),
        "source_codec": probe.get("codec_name"),
        "peak": round(peak, 6),
        "peak_db": round(db(peak), 3),
        "rms": round(rms_value, 8),
        "rms_db": round(db(rms_value), 3),
        "clipping_ratio": round(float(np.mean(abs_audio >= 0.98)) if audio.size else 0.0, 8),
        "bass_ratio": round(spectral_ratio(audio, sample_rate, 20.0, 180.0), 5),
        "low_mid_ratio": round(spectral_ratio(audio, sample_rate, 180.0, 500.0), 5),
        "high_ratio": round(spectral_ratio(audio, sample_rate, 6000.0, 14000.0), 5),
        "noise_floor_db": round(noise_floor_db, 3),
        "loudness_range_db": round(loudness_range, 3),
        "silent_second_count": int(sum(silent_flags)),
        "longest_silent_seconds": int(longest_silent),
        "first_half_rms_db": round(first_half_db, 3),
        "second_half_rms_db": round(second_half_db, 3),
        "second_half_drop_db": round(max(0.0, first_half_db - second_half_db), 3),
    }
    metrics["warnings"] = quality_warnings(metrics)
    return metrics


def build_filter_chain(profile: FilterProfil) -> str:
    """Erzeugt eine konservative ffmpeg-Filterkette fuer Musik."""

    return ",".join(
        [
            f"highpass=f={profile.highpass_hz:.1f}",
            f"lowpass=f={profile.lowpass_hz:.1f}",
            f"equalizer=f=95:t=q:w=1:g={profile.bass_gain_db:.2f}",
            f"equalizer=f=8500:t=q:w=1.1:g={profile.high_gain_db:.2f}",
            (
                "acompressor="
                f"threshold={profile.compressor_threshold_db:.1f}dB:"
                f"ratio={profile.compressor_ratio:.2f}:"
                "attack=25:release=250:makeup=1"
            ),
            f"alimiter=limit={profile.limiter:.3f}:level=false",
            f"loudnorm=I={profile.loudness_i:.1f}:TP={profile.true_peak_db:.1f}:LRA=11",
        ]
    )


def render_standard_postprocessing(input_path: Path, wav_out: Path, mp3_out: Path, profile: FilterProfil) -> None:
    ffmpeg = require_tool("ffmpeg")
    filter_chain = build_filter_chain(profile)
    wav_command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "32000",
        "-af",
        filter_chain,
        "-c:a",
        "pcm_s16le",
        str(wav_out),
    ]
    result = run_command(wav_command)
    if result.returncode != 0:
        raise RuntimeError(f"Standard-Postprocessing WAV fehlgeschlagen: {result.stdout.strip()}")

    mp3_command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_out),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(mp3_out),
    ]
    result = run_command(mp3_command)
    if result.returncode != 0:
        raise RuntimeError(f"MP3-Export fehlgeschlagen: {result.stdout.strip()}")


def try_deepfilter(input_path: Path, output_dir: Path, tools: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Optionaler DeepFilterNet-Haken. Standardfilter bleiben die sichere Basis."""

    if not tools["deepfilternet"]["available"]:
        return {"used": False, "reason": "DeepFilterNet CLI 'deep-filter' nicht lokal gefunden."}
    command_path = tools["deepfilternet"].get("path")
    if not command_path:
        return {
            "used": False,
            "reason": "DeepFilterNet Python-Paket gefunden, aber kein lokaler CLI-Befehl in PATH oder .venv/bin.",
        }
    work_dir = output_dir / "deepfilternet"
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared = work_dir / "input_48k.wav"
    decode_to_wav(input_path, prepared, 48000)
    before = {path.resolve() for path in work_dir.glob("*.wav")}
    command = [str(command_path), "-o", str(work_dir), str(prepared)]
    result = run_command(command)
    candidates = [
        path
        for path in work_dir.glob("*.wav")
        if path.resolve() not in before and path.name != prepared.name and path.stat().st_size > 0
    ]
    if result.returncode == 0 and candidates:
        output_path = max(candidates, key=lambda path: path.stat().st_mtime)
        return {
            "used": True,
            "returncode": result.returncode,
            "command": command,
            "output_path": rel(output_path),
            "output": result.stdout[-2000:],
            "reason": None,
        }
    return {
        "used": False,
        "returncode": result.returncode,
        "command": command,
        "output": result.stdout[-2000:],
        "reason": "DeepFilterNet-Ausgabe nicht eindeutig gefunden, Standardfilter bleiben aktiv.",
    }


def skipped_optional_model(name: str, requested: bool, tools: Dict[str, Dict[str, Any]], reason: str = "") -> Dict[str, Any]:
    key = name.lower()
    info = tools.get(key, {})
    return {
        "requested": bool(requested),
        "available": bool(info.get("available")),
        "used": False,
        "reason": reason or ("nicht angefordert" if not requested else f"{name} nicht lokal verfuegbar oder nicht sicher konfiguriert."),
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tools = optional_tools()
    profile = FILTER_PROFILE[args.intensitaet]
    started = now()
    wav_out = output_dir / "lange_audio_postprocessed.wav"
    mp3_out = output_dir / "lange_audio_postprocessed.mp3"

    print("Audio-Nachbearbeitung", flush=True)
    print("=====================", flush=True)
    print(f"Eingabe:    {rel(input_path)}", flush=True)
    print(f"Ausgabe:    {rel(output_dir)}", flush=True)
    print(f"Intensitaet:{args.intensitaet}", flush=True)

    if not input_path.exists():
        raise SystemExit(f"Eingabe nicht gefunden: {input_path}")

    with tempfile.TemporaryDirectory(prefix="audio_analyse_") as tmp:
        work_dir = Path(tmp)
        before = analyse_audio(input_path, work_dir, "vorher")
        write_json(output_dir / "audio_analyse_vorher.json", before)

        processing_steps: List[str] = []
        optional_usage: Dict[str, Any] = {
            "tools_detected": tools,
            "deepfilternet": skipped_optional_model("deepfilternet", args.denoise, tools),
            "demucs": skipped_optional_model(
                "demucs",
                False,
                tools,
                "Demucs wird in dieser Stufe nur als spaetere Analyseoption erkannt, nicht automatisch ausgefuehrt.",
            ),
            "matchering": skipped_optional_model("matchering", args.mastering, tools),
            "audiosr": skipped_optional_model("audiosr", args.audio_sr, tools, "AudioSR bleibt experimentell und standardmaessig deaktiviert."),
        }

        processing_input = input_path
        if args.denoise:
            duration = float(before.get("duration_sec") or 0.0)
            denoise_limit = max(0.0, float(args.denoise_max_dauer_sec))
            if (
                denoise_limit > 0
                and duration > denoise_limit
                and not args.denoise_lange_audio_erlauben
            ):
                optional_usage["deepfilternet"] = {
                    "requested": True,
                    "available": bool(tools["deepfilternet"]["available"]),
                    "used": False,
                    "reason": (
                        f"Audio ist {duration:.1f}s lang. DeepFilterNet wird ohne "
                        f"--denoise-lange-audio-erlauben nur bis {denoise_limit:.1f}s genutzt."
                    ),
                }
            else:
                optional_usage["deepfilternet"] = try_deepfilter(input_path, output_dir, tools)
                if optional_usage["deepfilternet"].get("used"):
                    processing_steps.append("deepfilternet")
                    processing_input = project_path(str(optional_usage["deepfilternet"]["output_path"]))

        if args.mastering and not args.mastering_reference:
            optional_usage["matchering"]["reason"] = "Mastering angefordert, aber keine --mastering-reference angegeben."
        elif args.mastering and not tools["matchering"]["available"]:
            optional_usage["matchering"]["reason"] = "Matchering CLI nicht lokal gefunden."
        elif args.mastering:
            optional_usage["matchering"]["reason"] = "Matchering wird erkannt, aber aus Sicherheitsgruenden nicht automatisch aufgerufen."

        if args.nur_pruefen:
            after = before
            output_audio = None
            processing_steps.append("nur_pruefen")
        else:
            render_standard_postprocessing(processing_input, wav_out, mp3_out, profile)
            processing_steps.extend(
                [
                    "standard_highpass",
                    "standard_bass_control",
                    "standard_high_control",
                    "standard_compressor",
                    "standard_limiter",
                    "standard_loudness_normalization",
                ]
            )
            after = analyse_audio(mp3_out, work_dir, "nachher")
            output_audio = rel(mp3_out)

        write_json(output_dir / "audio_analyse_nachher.json", after)

    report = {
        "status": "ok",
        "created_at": started,
        "finished_at": now(),
        "input": rel(input_path),
        "output_dir": rel(output_dir),
        "output_wav": rel(wav_out) if wav_out.exists() else None,
        "output_mp3": rel(mp3_out) if mp3_out.exists() else None,
        "output_audio": output_audio,
        "intensitaet": args.intensitaet,
        "nur_pruefen": bool(args.nur_pruefen),
        "nur_lokale_tools": bool(args.nur_lokale_tools),
        "processing_steps": processing_steps,
        "optional_tools": optional_usage,
        "lora_sicherheit": {
            "lora_adapter_veraendert": False,
            "checkpoints_veraendert": False,
            "datasets_veraendert": False,
            "postprocessing_nach_generierung": True,
            "postprocessed_audio_automatisch_trainingsdaten": False,
            "hinweis": "LoRA bleibt unveraendert; diese Audio ist nur ein Ausgabeprodukt.",
        },
        "analyse_vorher": rel(output_dir / "audio_analyse_vorher.json"),
        "analyse_nachher": rel(output_dir / "audio_analyse_nachher.json"),
    }
    write_json(output_dir / "postprocessing_report.json", report)

    print("Status:     ok", flush=True)
    if output_audio:
        print(f"Audio:      {output_audio}", flush=True)
    print(f"Report:     {rel(output_dir / 'postprocessing_report.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
