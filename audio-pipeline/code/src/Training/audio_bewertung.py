#!/usr/bin/env python3
"""Bewertung und Auswertung fuer MusicGen-Audios.

Diese Datei buendelt zwei aktive Hilfsfunktionen:
- technische Bewertung einer erzeugten MP3/WAV-Datei
- Analyse der menschlichen Bewertungs-CSVs

Die technische Bewertung ersetzt keine menschliche Musikbewertung. Sie filtert
nur typische technische Fehler wie Stille, Clipping, Signaltoene, dominante
Bassbereiche oder harte Lautheitswechsel.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


SAMPLE_RATE = 32000
EPS = 1e-12

SCORE_KATEGORIEN = [
    "Technische Audioqualität",
    "Musikalische Kohärenz",
    "Genre-Treue",
    "Übergangsqualität",
    "Referenzähnlichkeit",
]

GENRE_STANDARD_LABELS = {
    "lofi_hiphop": "Lo-Fi Hip-Hop",
    "jazz_lofi": "Jazz Lo-Fi",
    "chillhop": "Chillhop",
    "chillhop_lofi": "Chillhop",
    "study_lofi": "Study Lo-Fi",
    "ambient_lofi": "Ambient Lo-Fi",
    "dreamy_lofi": "Dreamy Lo-Fi",
    "guitar_lofi": "Guitar Lo-Fi",
}


def projektwurzel() -> Path:
    """Findet die Projektwurzel ueber die vier Hauptordner."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "code").exists() and (parent / "training").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = projektwurzel()


def rel(path: Path) -> str:
    """Gibt Pfade in Reports moeglichst relativ zur Projektwurzel aus."""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def db(value: float) -> float:
    """Wandelt lineare Amplitude in Dezibel um."""

    return 20.0 * math.log10(max(float(value), EPS))


def rms(audio: np.ndarray) -> float:
    """Berechnet die mittlere Energie eines Audiosignals."""

    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


def decode_audio(path: Path) -> np.ndarray:
    """Dekodiert MP3/WAV mit ffmpeg als mono 32-kHz-Float-Audio."""

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
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def seconds_label(seconds: float) -> str:
    """Formatiert Sekunden als mm:ss oder hh:mm:ss."""

    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, sec = divmod(rest, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def spectral_ratios(audio: np.ndarray) -> Dict[str, float]:
    """Schaetzt Bass-, Snare-, Hoehen- und Signalton-Anteile."""

    if len(audio) < SAMPLE_RATE:
        return {"bass_ratio": 0.0, "snare_ratio": 0.0, "high_ratio": 0.0, "tone_ratio": 0.0}
    sample = audio[: min(len(audio), SAMPLE_RATE * 30)]
    window = np.hanning(len(sample)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(sample * window))
    freqs = np.fft.rfftfreq(len(sample), d=1.0 / SAMPLE_RATE)
    total = float(np.sum(spectrum[(freqs >= 20) & (freqs <= 14000)])) + EPS
    return {
        "bass_ratio": float(np.sum(spectrum[(freqs >= 20) & (freqs <= 180)])) / total,
        "snare_ratio": float(np.sum(spectrum[(freqs >= 1800) & (freqs <= 6000)])) / total,
        "high_ratio": float(np.sum(spectrum[(freqs >= 8000) & (freqs <= 14000)])) / total,
        "tone_ratio": float(np.max(spectrum)) / total,
    }


def normalisiere_genre(value: Any) -> str:
    """Macht freie Genre-Namen fuer gespeicherte Profile vergleichbar."""

    text = str(value or "").lower().strip()
    text = text.replace("lo-fi", "lofi").replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if not text:
        return "lofi_hiphop"
    mapping = {
        "lofi": "lofi_hiphop",
        "lofi hiphop": "lofi_hiphop",
        "lofi hip hop": "lofi_hiphop",
        "lofi hip-hop": "lofi_hiphop",
        "jazz lofi": "jazz_lofi",
        "lofi jazz": "jazz_lofi",
        "chillhop": "chillhop",
        "chillhop lofi": "chillhop",
        "chill lofi": "chillhop",
        "study lofi": "study_lofi",
        "focus lofi": "study_lofi",
        "ambient lofi": "ambient_lofi",
        "dreamy lofi": "dreamy_lofi",
        "dream lofi": "dreamy_lofi",
        "guitar lofi": "guitar_lofi",
        "gitarre lofi": "guitar_lofi",
    }
    if text in mapping:
        return mapping[text]
    if "jazz" in text:
        return "jazz_lofi"
    if "chillhop" in text or "chill" in text:
        return "chillhop"
    if "study" in text or "focus" in text:
        return "study_lofi"
    if "ambient" in text:
        return "ambient_lofi"
    if "dream" in text:
        return "dreamy_lofi"
    if "guitar" in text or "gitarre" in text:
        return "guitar_lofi"
    return "lofi_hiphop"


def genre_label(value: Any) -> str:
    """Gibt einen lesbaren Genrenamen fuer Reports zurueck."""

    key = normalisiere_genre(value)
    return GENRE_STANDARD_LABELS.get(key, key.replace("_", " ").title())


def clamp_score(value: float) -> float:
    """Begrenzt Scores auf die Skala 0 bis 100."""

    return round(max(0.0, min(100.0, float(value))), 1)


def median_float(values: List[float]) -> float:
    """Robuster Median fuer Listen mit Messwerten."""

    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(np.median(clean)) if clean else 0.0


def schaetze_bpm(audio: np.ndarray) -> float:
    """Schaetzt das Tempo grob ueber Energie-Impulse."""

    if len(audio) < SAMPLE_RATE * 8:
        return 0.0
    frame = 1024
    hop = 512
    envelope: List[float] = []
    for start in range(0, len(audio) - frame, hop):
        chunk = audio[start : start + frame]
        envelope.append(float(np.sqrt(np.mean(np.square(chunk)))))
    if len(envelope) < 16:
        return 0.0
    env = np.asarray(envelope, dtype=np.float32)
    env = np.diff(env, prepend=env[0])
    env = np.maximum(env, 0.0)
    env = env - float(np.mean(env))
    if float(np.max(np.abs(env))) <= EPS:
        return 0.0
    corr = np.correlate(env, env, mode="full")[len(env) - 1 :]
    tempo_min = 55.0
    tempo_max = 110.0
    lag_min = int(round((60.0 / tempo_max) * SAMPLE_RATE / hop))
    lag_max = int(round((60.0 / tempo_min) * SAMPLE_RATE / hop))
    lag_min = max(1, min(lag_min, len(corr) - 1))
    lag_max = max(lag_min + 1, min(lag_max, len(corr)))
    lag = int(np.argmax(corr[lag_min:lag_max]) + lag_min)
    if lag <= 0:
        return 0.0
    return round(60.0 * SAMPLE_RATE / (hop * lag), 2)


def chroma_vector(audio: np.ndarray) -> np.ndarray:
    """Schaetzt eine einfache chromatische Verteilung fuer Harmonie-Spruenge."""

    if len(audio) < SAMPLE_RATE:
        return np.zeros(12, dtype=np.float32)
    sample = audio[: min(len(audio), SAMPLE_RATE * 8)]
    window = np.hanning(len(sample)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(sample * window))
    freqs = np.fft.rfftfreq(len(sample), d=1.0 / SAMPLE_RATE)
    chroma = np.zeros(12, dtype=np.float64)
    mask = (freqs >= 60.0) & (freqs <= 4000.0)
    for freq, mag in zip(freqs[mask], spectrum[mask]):
        if freq <= 0:
            continue
        midi = int(round(69 + 12 * math.log2(float(freq) / 440.0)))
        chroma[midi % 12] += float(mag)
    total = float(np.sum(chroma)) + EPS
    return (chroma / total).astype(np.float32)


def _spectral_centroid(audio: np.ndarray) -> float:
    """Berechnet eine grobe spektrale Helligkeit."""

    if len(audio) < SAMPLE_RATE:
        return 0.0
    sample = audio[: min(len(audio), SAMPLE_RATE * 30)]
    window = np.hanning(len(sample)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(sample * window))
    freqs = np.fft.rfftfreq(len(sample), d=1.0 / SAMPLE_RATE)
    return float(np.sum(freqs * spectrum) / (np.sum(spectrum) + EPS))


def _section_bpm_values(audio: np.ndarray, section_seconds: float = 30.0) -> List[float]:
    """BPM-Liste fuer laengere Audios, ohne Rohkurve im Hauptreport zu zeigen."""

    frames = int(round(section_seconds * SAMPLE_RATE))
    values: List[float] = []
    if frames <= 0:
        return values
    for start in range(0, len(audio), frames):
        chunk = audio[start : start + frames]
        if len(chunk) >= SAMPLE_RATE * 8:
            bpm = schaetze_bpm(chunk)
            if bpm > 0:
                values.append(float(bpm))
    return values


def score_kennwerte(audio: np.ndarray) -> Dict[str, float]:
    """Berechnet interne Kennwerte fuer die fuenf Score-Kategorien."""

    duration = len(audio) / float(SAMPLE_RATE)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.995)) if len(audio) else 0.0
    rms_values = segment_rms_values(audio, step_seconds=1.0)
    active_ratio = sum(value >= -50.0 for value in rms_values) / len(rms_values) if rms_values else 0.0
    quiet_ratio = 1.0 - active_ratio
    half = len(audio) // 2
    half_drop_db = max(0.0, db(rms(audio[:half])) - db(rms(audio[half:]))) if half else 0.0
    explosion_db = max(rms_values) - float(np.median(rms_values)) if rms_values else 0.0
    rms_range_db = max(rms_values) - min(rms_values) if rms_values else 0.0
    spec = spectral_ratios(audio)
    bpm_values = _section_bpm_values(audio)
    bpm = float(np.median(bpm_values)) if bpm_values else schaetze_bpm(audio)
    bpm_std = float(np.std(bpm_values)) if len(bpm_values) >= 2 else 0.0

    seam_seconds = min(8.0, max(2.0, duration * 0.1)) if duration > 0 else 2.0
    seam_frames = int(round(seam_seconds * SAMPLE_RATE))
    start_audio = audio[:seam_frames]
    end_audio = audio[-seam_frames:] if len(audio) >= seam_frames else audio
    start_spec = spectral_ratios(start_audio)
    end_spec = spectral_ratios(end_audio)
    seam_loudness_jump_db = abs(db(rms(start_audio)) - db(rms(end_audio))) if len(start_audio) and len(end_audio) else 0.0
    seam_spectral_jump = (
        abs(start_spec["bass_ratio"] - end_spec["bass_ratio"])
        + abs(start_spec["snare_ratio"] - end_spec["snare_ratio"])
        + abs(start_spec["high_ratio"] - end_spec["high_ratio"])
        + abs(start_spec["tone_ratio"] - end_spec["tone_ratio"])
    )
    seam_click = float(abs(float(start_audio[0]) - float(end_audio[-1]))) if len(start_audio) and len(end_audio) else 0.0
    start_chroma = chroma_vector(start_audio)
    end_chroma = chroma_vector(end_audio)
    harmonic_jump = float(np.linalg.norm(start_chroma - end_chroma))

    return {
        "duration_sec": round(duration, 3),
        "rms_db": round(db(rms(audio)), 3),
        "peak_db": round(db(peak), 3),
        "clipping_ratio": round(clipping_ratio, 6),
        "active_ratio": round(active_ratio, 4),
        "quiet_ratio": round(quiet_ratio, 4),
        "half_drop_db": round(half_drop_db, 3),
        "explosion_db": round(float(explosion_db), 3),
        "rms_range_db": round(float(rms_range_db), 3),
        "bass_ratio": round(float(spec["bass_ratio"]), 5),
        "snare_ratio": round(float(spec["snare_ratio"]), 5),
        "high_ratio": round(float(spec["high_ratio"]), 5),
        "tone_ratio": round(float(spec["tone_ratio"]), 5),
        "spectral_centroid": round(_spectral_centroid(audio), 3),
        "bpm": round(float(bpm), 3),
        "bpm_std": round(float(bpm_std), 3),
        "seam_loudness_jump_db": round(float(seam_loudness_jump_db), 3),
        "seam_spectral_jump": round(float(seam_spectral_jump), 5),
        "seam_click": round(float(seam_click), 5),
        "seam_harmonic_jump": round(float(harmonic_jump), 5),
    }


def _audio_pfad(row: Dict[str, Any]) -> Path:
    """Loest Pfade aus Dataset-Manifesten auf."""

    path = Path(str(row.get("path") or "")).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _manifest_rows(referenz_root: Path) -> List[Dict[str, Any]]:
    """Liest alle Split-Manifeste eines Referenzdatasets."""

    rows: List[Dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        path = referenz_root / split / "data.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _score_technisch(metrics: Dict[str, float]) -> float:
    """Technische Qualitaet ohne harte Strafe fuer gewolltes Vinylrauschen."""

    score = 100.0
    score -= min(35.0, max(0.0, 0.98 - metrics["active_ratio"]) * 140.0)
    score -= min(25.0, metrics["clipping_ratio"] * 2500.0)
    score -= max(0.0, metrics["peak_db"] + 0.4) * 8.0
    score -= max(0.0, metrics["half_drop_db"] - 5.0) * 2.0
    score -= max(0.0, metrics["explosion_db"] - 8.0) * 2.2
    score -= max(0.0, metrics["tone_ratio"] - 0.20) * 80.0
    score -= max(0.0, metrics["high_ratio"] - 0.28) * 35.0
    return clamp_score(score)


def _score_kohaerenz(metrics: Dict[str, float]) -> float:
    """Schaetzt rhythmische und dynamische Stabilitaet."""

    score = 100.0
    score -= min(30.0, metrics["bpm_std"] * 4.0)
    score -= max(0.0, metrics["explosion_db"] - 7.0) * 2.0
    score -= max(0.0, metrics["half_drop_db"] - 6.0) * 2.0
    score -= max(0.0, 2.0 - metrics["rms_range_db"]) * 4.0
    score -= max(0.0, metrics["rms_range_db"] - 18.0) * 1.5
    score -= max(0.0, metrics["tone_ratio"] - 0.18) * 70.0
    return clamp_score(score)


def _distanz_score(value: float, ref: float, scale: float) -> float:
    """Normalisiert Messwert-Abstand auf eine 0-bis-100-Teilwertung."""

    if scale <= 0:
        return 100.0
    return clamp_score(100.0 - min(100.0, abs(value - ref) / scale * 100.0))


def _score_genre(metrics: Dict[str, float], ref: Dict[str, Any]) -> float:
    """Bewertet Naehe zum gespeicherten Klangprofil des Genres."""

    parts = [
        _distanz_score(metrics["bpm"], float(ref.get("bpm") or metrics["bpm"]), 10.0),
        _distanz_score(metrics["bass_ratio"], float(ref.get("bass_ratio") or metrics["bass_ratio"]), 0.18),
        _distanz_score(metrics["snare_ratio"], float(ref.get("snare_ratio") or metrics["snare_ratio"]), 0.16),
        _distanz_score(metrics["high_ratio"], float(ref.get("high_ratio") or metrics["high_ratio"]), 0.12),
        _distanz_score(metrics["spectral_centroid"], float(ref.get("spectral_centroid") or metrics["spectral_centroid"]), 1400.0),
        _distanz_score(metrics["rms_range_db"], float(ref.get("rms_range_db") or metrics["rms_range_db"]), 8.0),
    ]
    return clamp_score(float(np.mean(parts)))


def _score_uebergang(metrics: Dict[str, float]) -> float:
    """Bewertet den Loop-Punkt vom Ende zurueck zum Anfang."""

    score = 100.0
    score -= max(0.0, metrics["seam_loudness_jump_db"] - 3.0) * 5.0
    score -= max(0.0, metrics["seam_spectral_jump"] - 0.12) * 120.0
    score -= max(0.0, metrics["seam_harmonic_jump"] - 0.28) * 60.0
    score -= max(0.0, metrics["seam_click"] - 0.08) * 180.0
    return clamp_score(score)


def _score_referenz(metrics: Dict[str, float], ref: Dict[str, Any]) -> float:
    """Beschreibt messbare Naehe zur Referenz, nicht objektive Musikqualitaet."""

    vergleich = [
        _distanz_score(metrics["rms_db"], float(ref.get("rms_db") or metrics["rms_db"]), 8.0),
        _distanz_score(metrics["peak_db"], float(ref.get("peak_db") or metrics["peak_db"]), 5.0),
        _distanz_score(metrics["active_ratio"], float(ref.get("active_ratio") or metrics["active_ratio"]), 0.12),
        _distanz_score(metrics["bass_ratio"], float(ref.get("bass_ratio") or metrics["bass_ratio"]), 0.20),
        _distanz_score(metrics["high_ratio"], float(ref.get("high_ratio") or metrics["high_ratio"]), 0.14),
        _distanz_score(metrics["tone_ratio"], float(ref.get("tone_ratio") or metrics["tone_ratio"]), 0.12),
        _distanz_score(metrics["bpm"], float(ref.get("bpm") or metrics["bpm"]), 12.0),
    ]
    return clamp_score(float(np.mean(vergleich)))


def berechne_fuenf_scores(metrics: Dict[str, float], referenz: Dict[str, Any]) -> Dict[str, float]:
    """Berechnet genau die fuenf sichtbaren Bewertungsscores."""

    return {
        SCORE_KATEGORIEN[0]: _score_technisch(metrics),
        SCORE_KATEGORIEN[1]: _score_kohaerenz(metrics),
        SCORE_KATEGORIEN[2]: _score_genre(metrics, referenz),
        SCORE_KATEGORIEN[3]: _score_uebergang(metrics),
        SCORE_KATEGORIEN[4]: _score_referenz(metrics, referenz),
    }


def _profil_ordner() -> Path:
    """Speicherort fuer feste Genrestandard-Profile."""

    return PROJECT_ROOT / "daten" / "metadata" / "genrestandards"


def _baue_genreprofile(referenz_root: Path, profile_dir: Path, pro_genre: int = 20) -> None:
    """Erstellt gespeicherte Genreprofile aus guten Trainingsclips."""

    profile_dir.mkdir(parents=True, exist_ok=True)
    gruppen: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in _manifest_rows(referenz_root):
        path = _audio_pfad(row)
        if not path.exists():
            continue
        genre = normalisiere_genre(row.get("lora_genre") or row.get("primary_genre") or row.get("genre"))
        gruppen[genre].append(row)

    for genre, rows in gruppen.items():
        rows = sorted(
            rows,
            key=lambda item: (
                -float(item.get("quality_score") or 0.0),
                str(item.get("source_file") or item.get("source_audio_path") or ""),
                str(item.get("path") or ""),
            ),
        )[: max(1, pro_genre)]
        metrics_rows: List[Dict[str, float]] = []
        for row in rows:
            try:
                metrics_rows.append(score_kennwerte(decode_audio(_audio_pfad(row))))
            except Exception:
                continue
        if not metrics_rows:
            continue
        median_metrics = {
            key: round(median_float([float(item.get(key, 0.0)) for item in metrics_rows]), 5)
            for key in metrics_rows[0]
        }
        technical_scores = [_score_technisch(item) for item in metrics_rows]
        coherence_scores = [_score_kohaerenz(item) for item in metrics_rows]
        transition_scores = [_score_uebergang(item) for item in metrics_rows]
        genre_scores = [_score_genre(item, median_metrics) for item in metrics_rows]
        reference_scores = [_score_referenz(item, median_metrics) for item in metrics_rows]
        standard_scores = {
            SCORE_KATEGORIEN[0]: clamp_score(median_float(technical_scores)),
            SCORE_KATEGORIEN[1]: clamp_score(median_float(coherence_scores)),
            SCORE_KATEGORIEN[2]: clamp_score(median_float(genre_scores)),
            SCORE_KATEGORIEN[3]: clamp_score(median_float(transition_scores)),
            SCORE_KATEGORIEN[4]: clamp_score(median_float(reference_scores)),
        }
        payload = {
            "genre_key": genre,
            "genre": GENRE_STANDARD_LABELS.get(genre, genre.replace("_", " ").title()),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": rel(referenz_root),
            "reference_count": len(metrics_rows),
            "scores": standard_scores,
            "metrics": median_metrics,
            "hinweis": "Genrestandard aus geprueften lokalen Trainingsclips. Scores dienen als Vergleichsstandard.",
        }
        write_json(profile_dir / f"{genre}.json", payload)


def _profil_gueltig(path: Path) -> bool:
    """Prueft, ob ein gespeichertes Profil die aktuelle 5-Score-Struktur hat."""

    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    scores = payload.get("scores")
    return isinstance(scores, dict) and all(category in scores for category in SCORE_KATEGORIEN)


def lade_genrestandard(genre: Any, referenz_root: Path, pro_genre: int = 20) -> Dict[str, Any]:
    """Laedt das gespeicherte Referenzprofil des Genres oder erstellt es lokal."""

    genre_key = normalisiere_genre(genre)
    profile_dir = _profil_ordner()
    fallback_map = {
        "lofi_hiphop": ["lofi_hiphop", "chillhop", "study_lofi"],
        "ambient_lofi": ["ambient_lofi", "dreamy_lofi", "chillhop"],
        "chillhop_lofi": ["chillhop", "chillhop_lofi"],
    }
    profile_path = profile_dir / f"{genre_key}.json"
    if not _profil_gueltig(profile_path):
        _baue_genreprofile(referenz_root, profile_dir, pro_genre)
    if not _profil_gueltig(profile_path):
        for fallback_key in fallback_map.get(genre_key, ["lofi_hiphop", "chillhop", "study_lofi"]):
            fallback_path = profile_dir / f"{fallback_key}.json"
            if _profil_gueltig(fallback_path):
                profile_path = fallback_path
                break
    if not _profil_gueltig(profile_path):
        return {
            "genre_key": genre_key,
            "genre": genre_label(genre_key),
            "scores": {key: 80.0 for key in SCORE_KATEGORIEN},
            "metrics": {},
            "reference_count": 0,
            "profile_path": "",
            "hinweis": "Kein gespeichertes Profil gefunden; neutraler Fallback verwendet.",
        }
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["profile_path"] = rel(profile_path)
    payload["requested_genre"] = genre_label(genre)
    return payload


def _score_rows(scores: Dict[str, float]) -> List[Dict[str, Any]]:
    """Macht Score-Dictionaries fuer CSV/HTML sortiert."""

    return [{"Kategorie": key, "Score": scores.get(key, 0.0)} for key in SCORE_KATEGORIEN]


def _groesste_unterschiede(standard: Dict[str, float], generiert: Dict[str, float]) -> List[Dict[str, Any]]:
    """Findet die groessten sichtbaren Score-Abweichungen."""

    rows = []
    for key in SCORE_KATEGORIEN:
        diff = round(float(generiert.get(key, 0.0)) - float(standard.get(key, 0.0)), 1)
        rows.append(
            {
                "Kategorie": key,
                "Genrestandard": standard.get(key, 0.0),
                "Generierte Audio": generiert.get(key, 0.0),
                "Differenz": diff,
                "Betrag": abs(diff),
            }
        )
    return sorted(rows, key=lambda item: item["Betrag"], reverse=True)


def _erklaerung(unterschiede: List[Dict[str, Any]]) -> str:
    """Erstellt eine kurze Auswertung fuer die sichtbare UI."""

    if not unterschiede:
        return "Keine Abweichung berechnet."
    top = unterschiede[0]
    richtung = "unter" if float(top["Differenz"]) < 0 else "ueber"
    text = (
        f"Die groesste Abweichung liegt bei {top['Kategorie']}: "
        f"die generierte Audio liegt {abs(float(top['Differenz'])):.1f} Punkte {richtung} dem Genrestandard."
    )
    if len(unterschiede) > 1 and float(unterschiede[1]["Betrag"]) >= 8.0:
        second = unterschiede[1]
        richtung_2 = "unter" if float(second["Differenz"]) < 0 else "ueber"
        text += (
            f" Zusaetzlich faellt {second['Kategorie']} auf "
            f"({abs(float(second['Differenz'])):.1f} Punkte {richtung_2} Standard)."
        )
    else:
        text += " Die uebrigen Kategorien liegen naeher am Referenzprofil."
    return text


def _svg_chart(title: str, scores: Dict[str, float]) -> str:
    """Erstellt ein simples Punkt-Liniendiagramm mit fester 0-bis-100-Skala."""

    width = 820
    height = 280
    left = 54
    right = 26
    top = 24
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    points: List[tuple[float, float, str, float]] = []
    for index, category in enumerate(SCORE_KATEGORIEN):
        x = left + (plot_w / (len(SCORE_KATEGORIEN) - 1)) * index
        score = float(scores.get(category, 0.0))
        y = top + (100.0 - score) / 100.0 * plot_h
        points.append((x, y, category, score))
    line_points = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
    grid = []
    for value in (0, 25, 50, 75, 100):
        y = top + (100.0 - value) / 100.0 * plot_h
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>'
            f'<text x="12" y="{y+4:.1f}" class="axis">{value}</text>'
        )
    labels = []
    for x, y, category, score in points:
        safe_category = html.escape(category)
        labels.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="point"/>')
        labels.append(f'<text x="{x:.1f}" y="{y-12:.1f}" class="score">{score:.1f}</text>')
        labels.append(
            f'<text x="{x:.1f}" y="{height-42}" class="xlabel">{safe_category}</text>'
        )
    return f"""
<section class="graph">
  <h2>{html.escape(title)}</h2>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
    <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis-line"/>
    <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis-line"/>
    {''.join(grid)}
    <polyline points="{line_points}" class="score-line"/>
    {''.join(labels)}
  </svg>
</section>
"""


def _score_liste(title: str, scores: Dict[str, float]) -> str:
    """Sichtbare fuenf Scores als kompakte Liste."""

    items = "\n".join(
        f"<li><span>{html.escape(category)}</span><strong>{float(scores.get(category, 0.0)):.1f}</strong></li>"
        for category in SCORE_KATEGORIEN
    )
    return f"<section><h2>{html.escape(title)}</h2><ul class=\"scores\">{items}</ul></section>"


def schreibe_score_html(
    path: Path,
    *,
    audio_path: Path,
    genre: str,
    standard_scores: Dict[str, float],
    generated_scores: Dict[str, float],
    auswertung: str,
) -> None:
    """Schreibt die sichtbare Bewertung mit genau zwei getrennten Graphen."""

    rel_audio = html.escape(os.path.relpath(audio_path, path.parent))
    content = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Audio-Bewertung</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101114; --fg:#f2f2ee; --muted:#a8aaa5; --line:#7dd3c7; --grid:#30343a; }}
    body {{ margin:0; background:var(--bg); color:var(--fg); font-family:Inter, Arial, sans-serif; }}
    main {{ max-width:980px; margin:0 auto; padding:32px 24px 48px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    h2 {{ margin:24px 0 12px; font-size:18px; }}
    .genre {{ color:var(--muted); margin-bottom:20px; }}
    audio {{ width:100%; margin:18px 0 10px; }}
    .graph {{ margin-top:22px; }}
    svg {{ width:100%; height:auto; background:#15181d; border:1px solid #272b31; border-radius:8px; }}
    .grid {{ stroke:var(--grid); stroke-width:1; }}
    .axis, .xlabel {{ fill:var(--muted); font-size:12px; text-anchor:middle; }}
    .axis {{ text-anchor:start; }}
    .axis-line {{ stroke:#666b73; stroke-width:1; }}
    .score-line {{ fill:none; stroke:var(--line); stroke-width:3; }}
    .point {{ fill:var(--line); stroke:#101114; stroke-width:2; }}
    .score {{ fill:var(--fg); font-size:13px; font-weight:700; text-anchor:middle; }}
    .scores {{ display:grid; gap:8px; list-style:none; padding:0; margin:0; }}
    .scores li {{ display:flex; justify-content:space-between; gap:16px; border-bottom:1px solid #272b31; padding:8px 0; }}
    .auswertung {{ margin-top:22px; color:var(--fg); line-height:1.55; }}
  </style>
</head>
<body>
<main>
  <h1>Audio-Bewertung</h1>
  <div class="genre">Genre: {html.escape(genre)}</div>
  <audio controls src="{rel_audio}"></audio>
  {_svg_chart("Genrestandard", standard_scores)}
  {_score_liste("Scores des Genrestandards", standard_scores)}
  {_svg_chart("Generierte Audio", generated_scores)}
  {_score_liste("Scores der generierten Audio", generated_scores)}
  <section class="auswertung"><h2>Auswertung</h2><p>{html.escape(auswertung)}</p></section>
</main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def bewerte_audio_mit_genrestandard(
    audio_path: Path,
    genre: str,
    ausgabe_dir: Optional[Path] = None,
    referenz_root: Optional[Path] = None,
    referenzen_pro_genre: int = 20,
) -> Dict[str, Any]:
    """Bewertet eine fertige Audio anhand fuenf Scores und schreibt die Visualisierung."""

    audio_path = audio_path.expanduser().resolve()
    output_dir = (ausgabe_dir or audio_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    referenz_root = (referenz_root or PROJECT_ROOT / "daten" / "processed" / "lora_training").expanduser().resolve()
    standard = lade_genrestandard(genre, referenz_root, referenzen_pro_genre)
    standard_scores = {key: float((standard.get("scores") or {}).get(key, 0.0)) for key in SCORE_KATEGORIEN}
    audio = decode_audio(audio_path)
    metrics = score_kennwerte(audio)
    generated_scores = berechne_fuenf_scores(metrics, dict(standard.get("metrics") or {}))
    unterschiede = _groesste_unterschiede(standard_scores, generated_scores)
    auswertung = _erklaerung(unterschiede)

    score_rows = []
    for category in SCORE_KATEGORIEN:
        score_rows.append(
            {
                "Kategorie": category,
                "Genrestandard": standard_scores.get(category, 0.0),
                "Generierte Audio": generated_scores.get(category, 0.0),
                "Differenz": round(generated_scores.get(category, 0.0) - standard_scores.get(category, 0.0), 1),
            }
        )
    csv_path = output_dir / "score_bewertung.csv"
    json_path = output_dir / "score_bewertung.json"
    html_path = output_dir / "score_bewertung.html"
    write_csv(csv_path, score_rows)
    schreibe_score_html(
        html_path,
        audio_path=audio_path,
        genre=genre_label(genre),
        standard_scores=standard_scores,
        generated_scores=generated_scores,
        auswertung=auswertung,
    )
    report = {
        "status": "finished",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audio": rel(audio_path),
        "genre": genre_label(genre),
        "genre_key": normalisiere_genre(genre),
        "standard_profile": standard.get("profile_path", ""),
        "standard_reference_count": standard.get("reference_count", 0),
        "kategorien": SCORE_KATEGORIEN,
        "genrestandard_scores": standard_scores,
        "generierte_audio_scores": generated_scores,
        "groesste_unterschiede": [{k: v for k, v in row.items() if k != "Betrag"} for row in unterschiede],
        "auswertung": auswertung,
        "csv": rel(csv_path),
        "html": rel(html_path),
        "hinweis": "Referenzaehnlichkeit ist eine messbare Naehe zum Genrestandard, keine objektive musikalische Qualitaet.",
        "interne_kennwerte": metrics,
    }
    write_json(json_path, report)
    return {**report, "json": rel(json_path)}


def segment_rms_values(audio: np.ndarray, step_seconds: float = 3.0) -> List[float]:
    """Berechnet Lautstaerke-Werte fuer kurze Zeitfenster."""

    step = int(round(step_seconds * SAMPLE_RATE))
    values: List[float] = []
    for start in range(0, len(audio), step):
        chunk = audio[start : start + step]
        if len(chunk) >= step * 0.5:
            values.append(db(rms(chunk)))
    return values


def transition_note(previous: Optional[np.ndarray], current: np.ndarray) -> str:
    """Bewertet, ob Anfang und vorheriges Ende technisch zusammenpassen."""

    if previous is None:
        return "Startabschnitt"
    frames = int(round(10.0 * SAMPLE_RATE))
    left = previous[-frames:] if len(previous) >= frames else previous
    right = current[:frames] if len(current) >= frames else current
    rms_gap = abs(db(rms(left)) - db(rms(right)))
    left_spec = spectral_ratios(left)
    right_spec = spectral_ratios(right)
    bass_gap = abs(left_spec["bass_ratio"] - right_spec["bass_ratio"])
    high_gap = abs(left_spec["high_ratio"] - right_spec["high_ratio"])
    if rms_gap > 9.0 or bass_gap > 0.28 or high_gap > 0.18:
        return "Pruefen: Uebergang wirkt technisch abrupt"
    if rms_gap > 6.0:
        return "Warnung: Lautheit springt am Uebergang"
    return "OK: technisch weicher Uebergang"


def classify_section(audio: np.ndarray, previous: Optional[np.ndarray]) -> Dict[str, Any]:
    """Bewertet einen Abschnitt mit einfachen technischen Regeln."""

    duration = len(audio) / float(SAMPLE_RATE)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.995)) if len(audio) else 0.0
    total_rms = rms(audio)
    rms_db = db(total_rms)
    rms_values = segment_rms_values(audio)
    active_ratio = sum(value >= -45.0 for value in rms_values) / len(rms_values) if rms_values else 0.0
    half = len(audio) // 2
    half_drop_db = max(0.0, db(rms(audio[:half])) - db(rms(audio[half:]))) if half else 0.0
    explosion_db = (max(rms_values) - float(np.median(rms_values))) if rms_values else 0.0
    rms_range = (max(rms_values) - min(rms_values)) if rms_values else 0.0
    spec = spectral_ratios(audio)

    probleme: List[str] = []
    warnungen: List[str] = []
    if duration < 5.0:
        probleme.append("Abschnitt zu kurz")
    if rms_db < -42.0:
        probleme.append("zu leise oder fast still")
    if active_ratio < 0.75:
        probleme.append("nicht durchgehend hoerbar")
    elif active_ratio < 0.90:
        warnungen.append("einige leise Stellen")
    if half_drop_db > 14.0:
        probleme.append("zweite Haelfte deutlich leiser")
    elif half_drop_db > 9.0:
        warnungen.append("Lautheit faellt ab")
    if clipping_ratio > 0.025:
        probleme.append("moegliche Uebersteuerung")
    elif peak > 0.98:
        warnungen.append("Peak sehr nah an 0 dB")
    if explosion_db > 14.0:
        probleme.append("Energie-Explosion")
    elif explosion_db > 10.0:
        warnungen.append("auffaelliger Pegelsprung")
    if spec["tone_ratio"] > 0.28:
        probleme.append("Signalton oder sehr monotone Schwingung")
    if spec["bass_ratio"] > 0.50:
        probleme.append("Bass deutlich zu dominant")
    elif spec["bass_ratio"] > 0.38:
        warnungen.append("Bass eher dominant")
    if spec["high_ratio"] > 0.32:
        probleme.append("Hoehen/Shaker sehr scharf")
    elif spec["high_ratio"] > 0.22:
        warnungen.append("Shaker/Hoehen auffaellig")
    if spec["snare_ratio"] > 0.38:
        warnungen.append("Snare-Bereich auffaellig stark")
    if rms_range < 2.2 and spec["tone_ratio"] > 0.13:
        warnungen.append("moeglich monoton")

    status = "Problem" if probleme else "Pruefen" if warnungen else "Gut"
    uebergang = transition_note(previous, audio)
    if uebergang.startswith("Pruefen") and status == "Gut":
        status = "Pruefen"

    return {
        "Status": status,
        "Uebergaenge": uebergang,
        "BPM_Stabilitaet": "Nicht automatisch sicher messbar",
        "Bass": "Problem: zu dominant" if spec["bass_ratio"] > 0.50 else "Pruefen" if spec["bass_ratio"] > 0.38 else "OK",
        "Shaker": "Problem: zu scharf/laut" if spec["high_ratio"] > 0.32 else "Pruefen" if spec["high_ratio"] > 0.22 else "OK",
        "Snare": "Pruefen" if spec["snare_ratio"] > 0.38 else "OK",
        "Harmonie": "Nicht automatisch musikalisch bewertbar",
        "Monotonie": "Problem" if spec["tone_ratio"] > 0.28 else "Pruefen" if rms_range < 2.2 else "OK",
        "Stoergeraeusche": "Problem" if spec["tone_ratio"] > 0.28 else "OK",
        "Lofi_Charakter": "Menschlich pruefen",
        "Gesamteindruck": status,
        "Technische_Notiz": "; ".join(probleme + warnungen) if probleme or warnungen else "technisch unauffaellig",
        "rms_db": round(rms_db, 3),
        "peak_db": round(db(peak), 3),
        "clipping_ratio": round(clipping_ratio, 5),
        "active_ratio": round(active_ratio, 3),
        "half_drop_db": round(half_drop_db, 3),
        "explosion_db": round(explosion_db, 3),
        "bass_ratio": round(spec["bass_ratio"], 4),
        "snare_ratio": round(spec["snare_ratio"], 4),
        "high_ratio": round(spec["high_ratio"], 4),
        "tone_ratio": round(spec["tone_ratio"], 4),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Schreibt eine CSV mit allen vorkommenden Spalten."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_bewertung_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Schreibt die normale Longform-Bewertung ohne technische Messspalten."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Abschnitt",
        "Start",
        "Ende",
        "Datei",
        "Status",
        "Uebergaenge",
        "BPM_Stabilitaet",
        "Bass",
        "Shaker",
        "Snare",
        "Harmonie",
        "Monotonie",
        "Stoergeraeusche",
        "Lofi_Charakter",
        "Gesamteindruck",
        "Notiz",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Schreibt einen JSON-Report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fasst Abschnittsbewertungen fuer den JSON-Report zusammen."""

    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("Status") or "Unbekannt")
        counts[status] = counts.get(status, 0) + 1
    problem_rows = [row for row in rows if row.get("Status") == "Problem"]
    check_rows = [row for row in rows if row.get("Status") == "Pruefen"]
    return {
        "sections": len(rows),
        "status_counts": counts,
        "problem_sections": [row.get("Abschnitt") for row in problem_rows],
        "check_sections": [row.get("Abschnitt") for row in check_rows],
        "overall_status": "Problem" if problem_rows else "Pruefen" if check_rows else "Gut",
    }


def bewerte_audio_datei(
    audio_path: Path,
    ausgabe_dir: Optional[Path] = None,
    abschnitt_sekunden: float = 300.0,
    bewertung_csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Bewertet eine Audio-Datei abschnittsweise und schreibt CSV/JSON.

    `automatische_bewertung.csv` bleibt der technische Detailreport. Wenn
    `bewertung_csv_path` gesetzt ist, wird zusaetzlich die normale
    Longform-Bewertungsdatei geschrieben, damit der menschliche Review direkt
    an derselben Stelle weiterarbeiten kann.
    """

    audio_path = audio_path.expanduser().resolve()
    output_dir = (ausgabe_dir or audio_path.parent).expanduser().resolve()
    audio = decode_audio(audio_path)
    duration = len(audio) / float(SAMPLE_RATE)
    section_frames = int(round(max(10.0, abschnitt_sekunden) * SAMPLE_RATE))
    rows: List[Dict[str, Any]] = []
    previous: Optional[np.ndarray] = None
    index = 1
    for start in range(0, len(audio), section_frames):
        end = min(len(audio), start + section_frames)
        section = audio[start:end]
        if len(section) < SAMPLE_RATE:
            continue
        start_sec = start / float(SAMPLE_RATE)
        end_sec = end / float(SAMPLE_RATE)
        section_rating = classify_section(section, previous)
        rows.append(
            {
                "Abschnitt": f"abschnitt_{index:03d}",
                "Start": seconds_label(start_sec),
                "Ende": seconds_label(end_sec),
                "Datei": rel(audio_path),
                **section_rating,
                "Notiz": f"Automatisch: {section_rating.get('Technische_Notiz', '')}",
            }
        )
        previous = section
        index += 1

    csv_path = output_dir / "automatische_bewertung.csv"
    json_path = output_dir / "automatische_bewertung.json"
    write_csv(csv_path, rows)
    bewertung_csv: Optional[Path] = None
    if bewertung_csv_path is not None:
        bewertung_csv = bewertung_csv_path.expanduser().resolve()
        write_bewertung_csv(bewertung_csv, rows)
    report = {
        "status": "finished",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audio": rel(audio_path),
        "duration_sec": round(duration, 3),
        "sample_rate": SAMPLE_RATE,
        "section_seconds": abschnitt_sekunden,
        "csv": rel(csv_path),
        "bewertung_csv": rel(bewertung_csv) if bewertung_csv is not None else None,
        "summary": summarize(rows),
        "hinweis": "Automatische technische Bewertung; menschliche musikalische Bewertung bleibt massgeblich.",
    }
    write_json(json_path, report)
    return {**report, "json": rel(json_path)}


PROBLEM_KEYWORDS: Dict[str, List[str]] = {
    "bass_stark": ["bass zu stark", "bass ist zu stark", "bass dominant", "zu viel bass", "bass viel zu stark"],
    "shaker_stark": ["shaker zu stark", "shaker zu laut", "shaker viel zu stark", "shaker muss", "shaker reduzieren"],
    "ton_kippt": ["kippt", "kein ton ab", "ton bricht", "ton verschwindet", "kein ton oder melodie"],
    "kein_ton": ["kein ton", "ohne ton", "keine melodie", "stille", "kein klang"],
    "monoton": ["monoton", "zu gleichfoermig", "zu einfach", "langweilig", "keine variation"],
    "snare_stark": ["snare zu laut", "snare zu stark", "snair zu laut", "snair zu stark"],
    "uebersteuerung": ["uebersteuert", "übersteuert", "clipping", "zu laut insgesamt"],
    "rauschen": ["rauschen", "stoergeraeusch", "noisy", "artefakt"],
    "gitarre_passt_nicht": ["gitarre harmoniert nicht", "gitarrenakzente", "gitarre passt"],
    "kreativitaet_fehlt": ["mehr kreativitaet", "mehr variation", "mehr rythmik", "mehr rhythmik", "mehr abwechslung"],
}

POSITIVE_KEYWORDS: Dict[str, List[str]] = {
    "sehr_gut": ["sehr gute audio", "hat mir sehr gefallen", "top audio", "ausgezeichnet"],
    "gut": ["gute audio", "gut eingesetzt", "guter uebergang", "guter übergang"],
    "rhythmik_gut": ["gute rythmik", "guter rhythmus", "rhythmisch gut", "rhythmik gut", "guter beat"],
    "relaxed": ["schoen relaxed", "schön relaxed", "angenehm", "entspannend"],
}


def lese_bewertungs_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Liest eine Bewertungs-CSV robust ein."""

    rows: List[Dict[str, str]] = []
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(text.splitlines())
    header = next(reader, None)
    if not header:
        return rows
    header = [item.strip() for item in header]
    for line_parts in reader:
        if not any(part.strip() for part in line_parts):
            continue
        rows.append({header[i]: line_parts[i].strip() if i < len(line_parts) else "" for i in range(len(header))})
    return rows


def notiz_aus_zeile(row: Dict[str, str]) -> str:
    """Extrahiert den eigentlichen Bewertungstext aus einer CSV-Zeile."""

    for key, value in row.items():
        key_lower = key.lower()
        if ("bewertung" in key_lower or "notiz" in key_lower) and value.strip():
            return value.lower()
    skip = {"sample", "datei", "genre"}
    return max((value.strip() for key, value in row.items() if key.lower() not in skip), key=len, default="").lower()


def zaehle_keywords(notiz: str, keyword_dict: Dict[str, List[str]]) -> List[str]:
    """Ordnet eine Notiz anhand einfacher Suchwoerter Kategorien zu."""

    return [kategorie for kategorie, woerter in keyword_dict.items() if any(wort in notiz for wort in woerter)]


def _berechne_trend(pro_durchgang: List[Dict[str, Any]], problem_key: str) -> bool:
    """Vergleicht erste und zweite Haelfte der Bewertungsdurchgaenge."""

    if len(pro_durchgang) < 4:
        return False
    haelfte = len(pro_durchgang) // 2

    def rate(gruppe: List[Dict[str, Any]]) -> float:
        total = sum(d["bewertungen"] for d in gruppe)
        count = sum(d["probleme"].get(problem_key, 0) for d in gruppe)
        return count / max(1, total)

    return rate(pro_durchgang[haelfte:]) < rate(pro_durchgang[:haelfte])


def _erstelle_fazit(probleme: List[Tuple[str, int]], gesamt: int, trend_bass: bool, trend_shaker: bool) -> str:
    """Erstellt ein kurzes Fazit fuer die Bachelorarbeits-Auswertung."""

    if not probleme:
        return "Keine Probleme gefunden."
    top = [f"{key} ({value}x, {round(value / max(1, gesamt) * 100, 0):.0f}%)" for key, value in probleme[:3]]
    fazit = f"Haeufigste Probleme: {', '.join(top)}. "
    fazit += "Bass-Dominanz zeigt Verbesserungstrend. " if trend_bass else "Bass-Dominanz hat sich nicht klar verbessert. "
    fazit += "Shaker-Intensitaet zeigt Verbesserungstrend." if trend_shaker else "Shaker-Intensitaet hat sich nicht klar verbessert."
    return fazit


def analysiere_alle_durchgaenge(bewertungs_root: Path) -> Dict[str, Any]:
    """Analysiert alle `durchgang_*/bewertung.csv` unter einem Ordner."""

    gesamt_probleme: Dict[str, int] = defaultdict(int)
    gesamt_positiv: Dict[str, int] = defaultdict(int)
    pro_durchgang: List[Dict[str, Any]] = []
    gesamt_zeilen = 0

    for csv_path in sorted(bewertungs_root.glob("durchgang_*/bewertung.csv")):
        zeilen = lese_bewertungs_csv(csv_path)
        durchgang_probleme: Dict[str, int] = defaultdict(int)
        durchgang_positiv: Dict[str, int] = defaultdict(int)
        bewertungen = 0
        for row in zeilen:
            notiz = notiz_aus_zeile(row)
            if not notiz:
                continue
            bewertungen += 1
            for kategorie in zaehle_keywords(notiz, PROBLEM_KEYWORDS):
                durchgang_probleme[kategorie] += 1
                gesamt_probleme[kategorie] += 1
            for kategorie in zaehle_keywords(notiz, POSITIVE_KEYWORDS):
                durchgang_positiv[kategorie] += 1
                gesamt_positiv[kategorie] += 1
        if not bewertungen:
            continue
        gesamt_zeilen += bewertungen
        pro_durchgang.append(
            {
                "durchgang": csv_path.parent.name,
                "bewertungen": bewertungen,
                "probleme": dict(sorted(durchgang_probleme.items(), key=lambda item: -item[1])),
                "positiv": dict(sorted(durchgang_positiv.items(), key=lambda item: -item[1])),
                "problem_rate": round(sum(durchgang_probleme.values()) / max(1, bewertungen), 2),
            }
        )

    probleme_sortiert = sorted(gesamt_probleme.items(), key=lambda item: -item[1])
    positiv_sortiert = sorted(gesamt_positiv.items(), key=lambda item: -item[1])
    trend_bass = _berechne_trend(pro_durchgang, "bass_stark")
    trend_shaker = _berechne_trend(pro_durchgang, "shaker_stark")
    return {
        "erstellt_am": datetime.now().isoformat(timespec="seconds"),
        "analysierte_durchgaenge": len(pro_durchgang),
        "gesamt_bewertungen": gesamt_zeilen,
        "haeufigste_probleme": [
            {"kategorie": key, "nennungen": value, "anteil_prozent": round(value / max(1, gesamt_zeilen) * 100, 1)}
            for key, value in probleme_sortiert
        ],
        "haeufigste_positiv": [
            {"kategorie": key, "nennungen": value, "anteil_prozent": round(value / max(1, gesamt_zeilen) * 100, 1)}
            for key, value in positiv_sortiert
        ],
        "trend_bass_verbessert": trend_bass,
        "trend_shaker_verbessert": trend_shaker,
        "pro_durchgang": pro_durchgang,
        "fazit": _erstelle_fazit(probleme_sortiert, gesamt_zeilen, trend_bass, trend_shaker),
    }


def schreibe_analyse_report(report: Dict[str, Any], ausgabe_dir: Path) -> Tuple[Path, Path]:
    """Speichert JSON- und CSV-Bericht der Bewertungsanalyse."""

    ausgabe_dir.mkdir(parents=True, exist_ok=True)
    json_path = ausgabe_dir / "bewertungs_analyse.json"
    csv_path = ausgabe_dir / "bewertungs_analyse_pro_durchgang.csv"
    write_json(json_path, report)

    rows: List[Dict[str, Any]] = []
    for item in report["pro_durchgang"]:
        row: Dict[str, Any] = {
            "Durchgang": item["durchgang"],
            "Bewertungen": item["bewertungen"],
            "Problem-Rate": item["problem_rate"],
        }
        for key in PROBLEM_KEYWORDS:
            row[f"P_{key}"] = item["probleme"].get(key, 0)
        for key in POSITIVE_KEYWORDS:
            row[f"G_{key}"] = item["positiv"].get(key, 0)
        rows.append(row)
    write_csv(csv_path, rows)
    return json_path, csv_path


def parse_audio_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Erzeugte MusicGen-Audio technisch bewerten.")
    parser.add_argument("--audio", required=True, help="Pfad zur MP3/WAV-Datei.")
    parser.add_argument("--ausgabe-dir", default="", help="Ordner fuer automatische_bewertung.csv/json.")
    parser.add_argument("--genre", default="", help="Optional: erzeugt zusaetzlich die 5-Score-Visualisierung.")
    parser.add_argument(
        "--referenz-root",
        default=str(PROJECT_ROOT / "daten" / "processed" / "lora_training"),
        help="Dataset fuer gespeicherte Genrestandard-Profile.",
    )
    parser.add_argument("--referenzen-pro-genre", type=int, default=20)
    parser.add_argument(
        "--bewertung-csv",
        default="",
        help="Optional: normale Longform-bewertung.csv direkt automatisch ausfuellen.",
    )
    parser.add_argument("--abschnitt-sekunden", type=float, default=300.0)
    return parser.parse_args(args)


def parse_analyse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Menschliche MusicGen-Bewertungen analysieren.")
    parser.add_argument("--bewertungs-root", default=str(PROJECT_ROOT / "training" / "bewertungen" / "musicgen"))
    parser.add_argument("--ausgabe-dir", default=str(PROJECT_ROOT / "training" / "bewertungen" / "analyse"))
    return parser.parse_args(args)


def run_audio(args: Optional[List[str]] = None) -> int:
    parsed = parse_audio_args(args)
    output_dir = Path(parsed.ausgabe_dir) if parsed.ausgabe_dir else None
    bewertung_csv = Path(parsed.bewertung_csv) if parsed.bewertung_csv else None
    report = bewerte_audio_datei(Path(parsed.audio), output_dir, parsed.abschnitt_sekunden, bewertung_csv)
    if parsed.genre:
        report["score_rating"] = bewerte_audio_mit_genrestandard(
            Path(parsed.audio),
            parsed.genre,
            output_dir,
            Path(parsed.referenz_root),
            parsed.referenzen_pro_genre,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def run_analyse(args: Optional[List[str]] = None) -> int:
    parsed = parse_analyse_args(args)
    bewertungs_root = Path(parsed.bewertungs_root).expanduser().resolve()
    ausgabe_dir = Path(parsed.ausgabe_dir).expanduser().resolve()
    print(f"Analysiere Bewertungen in: {bewertungs_root}", flush=True)
    report = analysiere_alle_durchgaenge(bewertungs_root)
    json_path, csv_path = schreibe_analyse_report(report, ausgabe_dir)

    print("\n=== ANALYSE-ERGEBNIS ===", flush=True)
    print(f"Durchgaenge: {report['analysierte_durchgaenge']}", flush=True)
    print(f"Bewertungen gesamt: {report['gesamt_bewertungen']}", flush=True)
    print("\nHaeufigste Probleme:", flush=True)
    for item in report["haeufigste_probleme"]:
        print(f"  {item['kategorie']:25s} {item['nennungen']:3d}x ({item['anteil_prozent']:.1f}%)", flush=True)
    print("\nHaeufigste positive Merkmale:", flush=True)
    for item in report["haeufigste_positiv"]:
        print(f"  {item['kategorie']:25s} {item['nennungen']:3d}x ({item['anteil_prozent']:.1f}%)", flush=True)
    print(f"\nFazit: {report['fazit']}", flush=True)
    print(f"\nReport gespeichert: {json_path}", flush=True)
    print(f"CSV gespeichert:    {csv_path}", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("MusicGen-Bewertung und Bewertungsanalyse")
        print("\nBefehle:")
        print("  audio    technische Bewertung und optional 5-Score-Visualisierung")
        print("  analyse  menschliche Bewertungs-CSVs auswerten")
        return 0
    befehl = sys.argv[1]
    rest = sys.argv[2:]
    if befehl == "audio":
        return run_audio(rest)
    if befehl == "analyse":
        return run_analyse(rest)
    print(f"Unbekannter Befehl: {befehl}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
