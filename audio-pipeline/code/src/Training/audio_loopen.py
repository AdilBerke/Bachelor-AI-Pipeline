#!/usr/bin/env python3
"""Musikalisch passende Loops aus kurzen MusicGen-Audios erzeugen.

MusicGen erzeugt in diesem Projekt 30-Sekunden-Clips. Dieses Modul veraendert
das Modell nicht, sondern verlaengert einen fertigen Clip in der
Nachbearbeitung. Dafuer werden moegliche Taktgrenzen gesucht und technisch
bewertet. Der beste Ausschnitt wird anschliessend mit einem kurzen,
lautstaerkekonstanten Crossfade wiederholt.
"""

from __future__ import annotations

import math
import random
import subprocess
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class LoopAnalyse:
    """Dokumentiert, warum eine bestimmte Loop-Grenze gewaehlt wurde."""

    methode: str
    bpm: Optional[float]
    beat_count: int
    start_sec: float
    end_sec: float
    loop_duration_sec: float
    crossfade_sec: float
    seam_score: float
    loudness_jump_db: float
    peak_jump_db: float
    bass_jump: float
    high_jump: float
    envelope_correlation: float
    candidate_count: int
    status: str
    warnung: str
    tempo_variation_percent: float = 0.0
    tempo_phase_sec: float = 0.0
    tempo_min_factor: float = 1.0
    tempo_max_factor: float = 1.0
    tempo_factors: str = "1.0000"

    def als_dict(self) -> dict[str, Any]:
        """Konvertiert die Analyse fuer CSV- und JSON-Reports."""

        return asdict(self)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), EPS))


def _spektralanteile(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Berechnet Bass- und Hoehenanteil eines kurzen Audiobereichs."""

    if len(audio) < 32:
        return 0.0, 0.0
    window = np.hanning(len(audio)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    relevant = (freqs >= 20.0) & (freqs <= 14000.0)
    total = float(np.sum(spectrum[relevant])) + EPS
    bass = float(np.sum(spectrum[(freqs >= 20.0) & (freqs <= 180.0)])) / total
    high = float(np.sum(spectrum[(freqs >= 7000.0) & (freqs <= 14000.0)])) / total
    return bass, high


def _energieverlauf(audio: np.ndarray, teile: int = 24) -> np.ndarray:
    """Verdichtet einen Audiobereich zu einem normierten Energieverlauf."""

    if len(audio) == 0:
        return np.zeros(teile, dtype=np.float32)
    grenzen = np.linspace(0, len(audio), teile + 1, dtype=int)
    werte = np.asarray(
        [_rms(audio[grenzen[index] : grenzen[index + 1]]) for index in range(teile)],
        dtype=np.float32,
    )
    mittel = float(np.mean(werte))
    standard = float(np.std(werte))
    if standard <= EPS:
        return werte - mittel
    return (werte - mittel) / standard


def _korrelation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    if float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else 0.0


def _aktive_grenzen(audio: np.ndarray, sample_rate: int) -> tuple[int, int]:
    """Blendet fast stille Intros und Outros aus der Loop-Suche aus."""

    fenster = max(1, int(round(0.25 * sample_rate)))
    pegel: list[float] = []
    starts: list[int] = []
    for start in range(0, len(audio), fenster):
        segment = audio[start : start + fenster]
        if len(segment) < fenster // 2:
            continue
        starts.append(start)
        pegel.append(_db(_rms(segment)))
    if not pegel:
        return 0, len(audio)

    gesamt_db = _db(_rms(audio))
    schwelle = max(-55.0, gesamt_db - 28.0)
    aktiv = [index for index, value in enumerate(pegel) if value >= schwelle]
    if not aktiv:
        return 0, len(audio)

    start_frame = starts[aktiv[0]]
    end_frame = min(len(audio), starts[aktiv[-1]] + fenster)
    if end_frame - start_frame < sample_rate * 8:
        return 0, len(audio)
    return start_frame, end_frame


def _tempo_und_beats(
    audio: np.ndarray,
    sample_rate: int,
    ziel_bpm: float,
) -> tuple[Optional[float], np.ndarray]:
    """Schaetzt Tempo und Beat-Positionen mit librosa.

    Falls librosa auf einem anderen Rechner nicht verfuegbar ist, bleibt die
    Loop-Funktion nutzbar und faellt auf eine zeitbasierte Suche zurueck.
    """

    try:
        import librosa

        tempo_raw, beat_times = librosa.beat.beat_track(
            y=audio,
            sr=sample_rate,
            start_bpm=max(40.0, ziel_bpm),
            units="time",
            trim=False,
        )
        tempo = float(np.asarray(tempo_raw).reshape(-1)[0])
        beats = np.asarray(beat_times, dtype=np.float64)
    except Exception:
        return None, np.zeros(0, dtype=np.float64)

    # Halb- oder Doppeltempo wird auf den Lofi-Zielbereich abgebildet. Wenn
    # halbiert wird, wird auch das Beat-Raster entsprechend ausgeduennt.
    while tempo > ziel_bpm * 1.45 and len(beats) >= 4:
        tempo /= 2.0
        beats = beats[::2]
    while tempo < ziel_bpm * 0.70 and tempo > 0.0:
        tempo *= 2.0
    return tempo, beats


def _naht_metriken(
    audio: np.ndarray,
    start_frame: int,
    end_frame: int,
    fenster_frames: int,
    sample_rate: int,
) -> dict[str, float]:
    before = audio[max(start_frame, end_frame - fenster_frames) : end_frame]
    after = audio[start_frame : min(end_frame, start_frame + fenster_frames)]
    size = min(len(before), len(after))
    if size < 32:
        return {
            "loudness_jump_db": 99.0,
            "peak_jump_db": 99.0,
            "bass_jump": 1.0,
            "high_jump": 1.0,
            "envelope_correlation": -1.0,
            "score": 999.0,
        }
    before = before[-size:]
    after = after[:size]

    loudness_jump = abs(_db(_rms(before)) - _db(_rms(after)))
    before_peak = float(np.max(np.abs(before)))
    after_peak = float(np.max(np.abs(after)))
    peak_jump = abs(_db(before_peak) - _db(after_peak))
    before_bass, before_high = _spektralanteile(before, sample_rate)
    after_bass, after_high = _spektralanteile(after, sample_rate)
    bass_jump = abs(before_bass - after_bass)
    high_jump = abs(before_high - after_high)
    correlation = _korrelation(_energieverlauf(before), _energieverlauf(after))

    # Niedriger ist besser. Lautheit, Klangfarbe und rhythmischer
    # Energieverlauf werden gemeinsam bewertet.
    score = (
        loudness_jump * 0.75
        + peak_jump * 0.25
        + bass_jump * 14.0
        + high_jump * 18.0
        + (1.0 - correlation) * 2.5
    )
    return {
        "loudness_jump_db": loudness_jump,
        "peak_jump_db": peak_jump,
        "bass_jump": bass_jump,
        "high_jump": high_jump,
        "envelope_correlation": correlation,
        "score": score,
    }


def _zeitkandidaten(
    active_start_sec: float,
    active_end_sec: float,
    min_loop_sec: float,
    max_loop_sec: float,
    suchfenster_sec: float,
) -> list[tuple[float, float, str]]:
    """Fallback-Kandidaten, wenn kein stabiles Beat-Raster messbar ist."""

    candidates: list[tuple[float, float, str]] = []
    start_limit = min(active_end_sec - min_loop_sec, active_start_sec + suchfenster_sec)
    end_limit = max(active_start_sec + min_loop_sec, active_end_sec - suchfenster_sec)
    starts = np.arange(active_start_sec, start_limit + 0.001, 0.25)
    ends = np.arange(end_limit, active_end_sec + 0.001, 0.25)
    for start in starts:
        for end in ends:
            duration = float(end - start)
            if min_loop_sec <= duration <= max_loop_sec:
                candidates.append((float(start), float(end), "zeitfenster"))
    return candidates


def analysiere_loop(
    audio: np.ndarray,
    *,
    sample_rate: int = 32000,
    ziel_bpm: float = 78.0,
    min_loop_sec: float = 18.0,
    max_loop_sec: float = 29.0,
    suchfenster_sec: float = 7.0,
    crossfade_sec: float = 0.0,
) -> LoopAnalyse:
    """Findet eine taktnahe, technisch moeglichst unauffaellige Loop-Naht."""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration_sec = len(audio) / float(sample_rate)
    if duration_sec < 4.0:
        raise ValueError("Der Quellclip ist fuer einen musikalischen Loop zu kurz.")

    active_start, active_end = _aktive_grenzen(audio, sample_rate)
    active_start_sec = active_start / sample_rate
    active_end_sec = active_end / sample_rate
    max_loop_sec = min(max_loop_sec, active_end_sec - active_start_sec)
    min_loop_sec = min(min_loop_sec, max_loop_sec)
    if max_loop_sec < 3.0:
        raise ValueError("Im Quellclip wurde kein ausreichend langer aktiver Bereich gefunden.")

    bpm, beats = _tempo_und_beats(audio, sample_rate, ziel_bpm)
    beat_sec = 60.0 / bpm if bpm and bpm > 0.0 else 60.0 / max(40.0, ziel_bpm)
    effective_crossfade = crossfade_sec if crossfade_sec > 0.0 else beat_sec
    effective_crossfade = min(max(0.25, effective_crossfade), max(0.25, beat_sec * 1.25), 1.25)
    fenster_frames = max(32, int(round(effective_crossfade * sample_rate)))

    candidates: list[tuple[float, float, str]] = []
    if len(beats) >= 8:
        for start_index, start in enumerate(beats):
            if start < active_start_sec or start > active_start_sec + suchfenster_sec:
                continue
            for end_index in range(start_index + 4, len(beats)):
                end = float(beats[end_index])
                beat_distance = end_index - start_index
                duration = end - float(start)
                if end > active_end_sec:
                    break
                if duration > max_loop_sec:
                    break
                if (
                    end >= active_end_sec - suchfenster_sec
                    and duration >= min_loop_sec
                    and beat_distance % 4 == 0
                ):
                    candidates.append((float(start), end, "taktgrenzen"))

    if not candidates:
        candidates = _zeitkandidaten(
            active_start_sec,
            active_end_sec,
            min_loop_sec,
            max_loop_sec,
            suchfenster_sec,
        )
    if not candidates:
        candidates = [(active_start_sec, active_end_sec, "aktive_grenzen")]

    best: Optional[tuple[float, float, str, dict[str, float], float]] = None
    for start_sec, end_sec, methode in candidates:
        start_frame = max(0, min(len(audio) - 1, int(round(start_sec * sample_rate))))
        end_frame = max(start_frame + 1, min(len(audio), int(round(end_sec * sample_rate))))
        metrics = _naht_metriken(audio, start_frame, end_frame, fenster_frames, sample_rate)
        # Bei aehnlicher Nahtqualitaet wird ein laengerer Ausschnitt bevorzugt,
        # weil sich das musikalische Material dadurch seltener wiederholt.
        laengen_malus = max(0.0, max_loop_sec - (end_sec - start_sec)) * 0.08
        score = float(metrics["score"] + laengen_malus)
        if best is None or score < best[4]:
            best = (start_sec, end_sec, methode, metrics, score)

    assert best is not None
    start_sec, end_sec, methode, metrics, score = best
    warnungen: list[str] = []
    if metrics["loudness_jump_db"] > 4.0:
        warnungen.append("deutlicher Lautheitssprung")
    if metrics["bass_jump"] > 0.12:
        warnungen.append("deutlicher Basssprung")
    if metrics["high_jump"] > 0.10:
        warnungen.append("deutlicher Hoehensprung")
    if metrics["envelope_correlation"] < -0.20:
        warnungen.append("unterschiedlicher Rhythmus an der Naht")
    status = "WARNUNG" if warnungen else "OK"

    return LoopAnalyse(
        methode=methode,
        bpm=round(bpm, 3) if bpm is not None else None,
        beat_count=int(len(beats)),
        start_sec=round(start_sec, 6),
        end_sec=round(end_sec, 6),
        loop_duration_sec=round(end_sec - start_sec, 6),
        crossfade_sec=round(effective_crossfade, 6),
        seam_score=round(score, 6),
        loudness_jump_db=round(metrics["loudness_jump_db"], 6),
        peak_jump_db=round(metrics["peak_jump_db"], 6),
        bass_jump=round(metrics["bass_jump"], 6),
        high_jump=round(metrics["high_jump"], 6),
        envelope_correlation=round(metrics["envelope_correlation"], 6),
        candidate_count=len(candidates),
        status=status,
        warnung="; ".join(warnungen),
    )


def _tempo_strecken(audio: np.ndarray, faktor: float, sample_rate: int) -> np.ndarray:
    """Veraendert das Tempo mit FFmpeg, ohne die Tonhoehe zu verschieben."""

    if abs(faktor - 1.0) < 0.0005:
        return audio.astype(np.float32, copy=True)
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "f32le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-filter:a",
                f"atempo={float(faktor):.6f}",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "pipe:1",
            ],
            input=audio.astype(np.float32, copy=False).tobytes(),
            check=True,
            capture_output=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Pitch-erhaltende Tempovariation ist fehlgeschlagen. "
            "Bitte pruefen, ob FFmpeg lokal installiert ist."
        ) from exc
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _tempo_faktor_fuer_phase(
    rng: random.Random,
    phase_index: int,
    variation_percent: float,
    erste_richtung: int,
) -> float:
    """Wechselt kontrolliert zwischen normal, langsamer und schneller."""

    breite = max(0.0, variation_percent) / 100.0
    if breite <= 0.0 or phase_index % 2 == 0:
        return 1.0

    # Ungerade Phasen wechseln die Richtung. Dazwischen liegt immer eine
    # neutrale Phase, wodurch kein abrupter Sprung von langsam zu schnell
    # entstehen kann.
    ungerade_phase = (phase_index + 1) // 2
    richtung = erste_richtung if ungerade_phase % 2 == 1 else -erste_richtung
    max_abweichung = min(0.01, breite)
    min_abweichung = min(max_abweichung, max(0.0025, max_abweichung * 0.45))
    abweichung = rng.uniform(min_abweichung, max_abweichung)
    return round(1.0 + richtung * abweichung, 4)


def baue_loop_block(
    audio: np.ndarray,
    ziel_dauer_sec: float,
    *,
    sample_rate: int = 32000,
    ziel_bpm: float = 78.0,
    min_loop_sec: float = 18.0,
    max_loop_sec: float = 29.0,
    suchfenster_sec: float = 7.0,
    crossfade_sec: float = 0.0,
    tempo_variation_percent: float = 0.0,
    tempo_phase_sec: float = 60.0,
    tempo_start_phase: int = 0,
    seed: int = 0,
    analyse: Optional[LoopAnalyse] = None,
) -> tuple[np.ndarray, LoopAnalyse, int]:
    """Verlaengert einen Clip und gibt Block, Analyse und Wiederholungen zurueck."""

    if ziel_dauer_sec <= 0.0:
        raise ValueError("Die Ziel-Dauer des Loop-Blocks muss groesser als 0 sein.")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    analyse = analyse or analysiere_loop(
        audio,
        sample_rate=sample_rate,
        ziel_bpm=ziel_bpm,
        min_loop_sec=min_loop_sec,
        max_loop_sec=max_loop_sec,
        suchfenster_sec=suchfenster_sec,
        crossfade_sec=crossfade_sec,
    )
    start = int(round(analyse.start_sec * sample_rate))
    end = int(round(analyse.end_sec * sample_rate))
    segment = audio[start:end].astype(np.float32, copy=True)
    if len(segment) < sample_rate:
        raise ValueError("Die gefundene Loop-Region ist ungueltig oder zu kurz.")

    target_frames = int(round(ziel_dauer_sec * sample_rate))
    crossfade_frames = int(round(analyse.crossfade_sec * sample_rate))
    crossfade_frames = min(crossfade_frames, len(segment) // 2)
    # Ein zusaetzliches Segment reicht als Reserve, weil der Aufbau beendet
    # wird, sobald die angeforderte Ziel-Dauer abgedeckt ist.
    reserve = int(math.ceil(len(segment) * (1.0 + max(0.0, tempo_variation_percent) / 100.0)))
    block = np.zeros(target_frames + reserve, dtype=np.float32)
    cursor = 0
    verwendungen = 0
    phase_frames = max(1, int(round(max(10.0, tempo_phase_sec) * sample_rate)))
    rng = random.Random(seed)
    phase_factors: list[float] = [1.0]
    erste_richtung = -1 if rng.random() < 0.5 else 1
    verwendete_faktoren: list[float] = []
    varianten_cache: dict[float, np.ndarray] = {1.0: segment}

    while cursor < target_frames:
        phase_index = max(0, tempo_start_phase) + cursor // phase_frames
        while len(phase_factors) <= phase_index:
            phase_factors.append(
                _tempo_faktor_fuer_phase(
                    rng,
                    len(phase_factors),
                    tempo_variation_percent,
                    erste_richtung,
                )
            )
        faktor = phase_factors[phase_index]
        verwendete_faktoren.append(faktor)
        if faktor not in varianten_cache:
            varianten_cache[faktor] = _tempo_strecken(segment, faktor, sample_rate)
        variante = varianten_cache[faktor]
        fade_frames = min(crossfade_frames, len(variante) // 2, cursor // 2)

        if verwendungen == 0:
            block[: len(variante)] = variante
            cursor = len(variante)
        else:
            start = cursor - fade_frames
            if fade_frames > 0:
                fade_in = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
                block[start:cursor] = (
                    block[start:cursor] * (1.0 - fade_in)
                    + variante[:fade_frames] * fade_in
                )
            end = start + len(variante)
            if end > len(block):
                block = np.pad(block, (0, end - len(block)))
            block[cursor:end] = variante[fade_frames:]
            cursor = end
        verwendungen += 1

    eindeutige_faktoren = list(dict.fromkeys(verwendete_faktoren))
    analyse = replace(
        analyse,
        tempo_variation_percent=round(max(0.0, tempo_variation_percent), 3),
        tempo_phase_sec=round(max(10.0, tempo_phase_sec), 3),
        tempo_min_factor=min(eindeutige_faktoren),
        tempo_max_factor=max(eindeutige_faktoren),
        tempo_factors=" ".join(f"{value:.4f}" for value in eindeutige_faktoren),
    )
    return block[:target_frames].astype(np.float32), analyse, verwendungen
