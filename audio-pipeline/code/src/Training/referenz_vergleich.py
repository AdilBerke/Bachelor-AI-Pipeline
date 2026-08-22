#!/usr/bin/env python3
"""Vergleicht erzeugte MusicGen-Audios mit guten MP3-Referenzclips.

Zweck der Datei:
- gute Trainingsclips aus den lokalen MP3s als Referenz auswerten
- erzeugte MusicGen-/LoRA-Audios dagegen messen
- sichtbar machen, warum die generierte Qualitaet nicht wie die MP3s klingt

Die Datei startet kein Training und erzeugt keine Audios. Sie ist eine reine
Diagnose fuer die Bachelorarbeit und fuer die naechsten Modellentscheidungen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audio_bewertung import (  # noqa: E402
    EPS,
    SAMPLE_RATE,
    db,
    decode_audio,
    rel,
    rms,
    segment_rms_values,
    spectral_ratios,
    write_csv,
    write_json,
)


STANDARD_GENRES = (
    "jazz_lofi",
    "chillhop_lofi",
    "dreamy_lofi",
    "study_lofi",
    "guitar_lofi",
)

GENRE_LABELS = {
    "jazz_lofi": "Jazz Lofi",
    "chillhop_lofi": "Chillhop Lofi",
    "dreamy_lofi": "Dreamy Lofi",
    "study_lofi": "Study Lofi",
    "guitar_lofi": "Guitar Lofi",
    "general_lofi": "General Lofi",
}

METRIK_FELDER = (
    "duration_sec",
    "rms_db",
    "peak_db",
    "active_ratio",
    "quiet_ratio",
    "half_drop_db",
    "explosion_db",
    "rms_range_db",
    "bass_ratio",
    "snare_ratio",
    "high_ratio",
    "tone_ratio",
    "clipping_ratio",
    "bpm",
)


@dataclass
class AudioEintrag:
    """Beschreibt eine echte Datei oder einen Abschnitt einer langen Datei."""

    path: Path
    genre: str
    quelle: str
    label: str
    audio: np.ndarray | None = None
    start_sec: float = 0.0
    end_sec: float = 0.0
    metadata: dict[str, Any] | None = None


def projektwurzel() -> Path:
    """Findet die Projektwurzel ueber die vier Hauptordner."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "code").exists() and (parent / "daten").exists() and (parent / "training").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = projektwurzel()


def normalisiere_genre(value: Any) -> str:
    """Macht aus freien Genre-Namen stabile interne Genre-Schluessel."""

    text = str(value or "").lower().strip()
    text = text.replace("lo-fi", "lofi").replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if not text:
        return "general_lofi"
    mapping = {
        "jazz lofi": "jazz_lofi",
        "lofi jazz": "jazz_lofi",
        "chillhop lofi": "chillhop_lofi",
        "chill lofi": "chillhop_lofi",
        "dreamy lofi": "dreamy_lofi",
        "dream lofi": "dreamy_lofi",
        "study lofi": "study_lofi",
        "focus lofi": "study_lofi",
        "guitar lofi": "guitar_lofi",
        "gitarre lofi": "guitar_lofi",
    }
    if text in mapping:
        return mapping[text]
    for genre in STANDARD_GENRES:
        readable = genre.replace("_", " ")
        if readable in text:
            return genre
    if "jazz" in text:
        return "jazz_lofi"
    if "chillhop" in text or "chill" in text:
        return "chillhop_lofi"
    if "dream" in text:
        return "dreamy_lofi"
    if "study" in text or "focus" in text:
        return "study_lofi"
    if "guitar" in text or "gitarre" in text:
        return "guitar_lofi"
    return "general_lofi"


def loese_pfad(path_text: Any) -> Path:
    """Loest relative Pfade gegen die Projektwurzel auf."""

    path = Path(str(path_text or "")).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def lese_jsonl(path: Path) -> list[dict[str, Any]]:
    """Liest ein JSONL-Manifest und ignoriert defekte Leerzeilen."""

    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def sammle_referenzen(referenz_root: Path, pro_genre: int) -> list[AudioEintrag]:
    """Waehlt gute Referenzclips aus dem geprueften Trainingsdataset."""

    rows: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        rows.extend(lese_jsonl(referenz_root / split / "data.jsonl"))

    gruppen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        path = loese_pfad(row.get("path"))
        if not path.exists():
            continue
        genre = normalisiere_genre(row.get("lora_genre") or row.get("primary_genre") or row.get("genre"))
        gruppen[genre].append(row)

    eintraege: list[AudioEintrag] = []
    for genre in sorted(gruppen):
        # Deterministisch und verteilt: erst hohe Scores, dann verschiedene Quellen.
        sortierte = sorted(
            gruppen[genre],
            key=lambda item: (
                -float(item.get("quality_score") or 0.0),
                str(item.get("source_file") or item.get("source_audio_path") or ""),
                str(item.get("path") or ""),
            ),
        )
        genutzte_quellen: set[str] = set()
        auswahl: list[dict[str, Any]] = []
        for row in sortierte:
            quelle = str(row.get("source_split_key") or row.get("source_audio_path") or row.get("source_file") or "")
            if quelle in genutzte_quellen and len(auswahl) < max(1, pro_genre // 2):
                continue
            auswahl.append(row)
            genutzte_quellen.add(quelle)
            if len(auswahl) >= pro_genre:
                break
        if len(auswahl) < pro_genre:
            for row in sortierte:
                if row in auswahl:
                    continue
                auswahl.append(row)
                if len(auswahl) >= pro_genre:
                    break

        for index, row in enumerate(auswahl, start=1):
            path = loese_pfad(row.get("path"))
            eintraege.append(
                AudioEintrag(
                    path=path,
                    genre=genre,
                    quelle="referenz",
                    label=f"{genre}_referenz_{index:03d}",
                    metadata=row,
                )
            )
    return eintraege


def csv_bewertung_mapping(root: Path) -> dict[str, str]:
    """Liest Genre-Namen aus einer passenden bewertung.csv, falls vorhanden."""

    mapping: dict[str, str] = {}
    kandidaten = [root / "bewertung.csv", root.parent / "bewertung.csv"]
    for csv_path in kandidaten:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                datei = str(row.get("Datei") or row.get("datei") or "").strip()
                sample = str(row.get("Sample") or row.get("sample") or "").strip()
                genre = normalisiere_genre(row.get("Genre") or row.get("genre"))
                if datei:
                    mapping[Path(datei).name] = genre
                if sample:
                    mapping[sample] = genre
    return mapping


def inferiere_genre(path: Path, mapping: dict[str, str]) -> str:
    """Erkennt das Genre aus Dateiname, Bewertung oder Ordnername."""

    name = path.name
    stem = path.stem
    if name in mapping:
        return mapping[name]
    if stem in mapping:
        return mapping[stem]
    return normalisiere_genre(" ".join(path.parts[-5:]))


def audio_dateien(root: Path) -> list[Path]:
    """Sammelt Audio-Dateien unter einem Ordner oder gibt die Datei selbst."""

    if root.is_file() and root.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a"}:
        return [root]
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a"}
        and "__pycache__" not in path.parts
    )


def finde_neuesten_audio_ordner() -> Path | None:
    """Findet den neuesten Bewertungs-Audioordner mit echten Audiodateien."""

    root = PROJECT_ROOT / "training" / "bewertungen" / "musicgen"
    if not root.exists():
        return None
    kandidaten: list[tuple[float, Path]] = []
    for audio_dir in root.rglob("audio"):
        files = audio_dateien(audio_dir)
        if not files:
            continue
        try:
            mtime = max(path.stat().st_mtime for path in files)
        except OSError:
            continue
        kandidaten.append((mtime, audio_dir))
    if not kandidaten:
        return None
    return sorted(kandidaten)[-1][1]


def sammle_generierte(generiert_root: Path | None, limit: int, segment_sekunden: float) -> list[AudioEintrag]:
    """Sammelt erzeugte Audios und teilt lange Dateien in pruefbare Segmente."""

    root = generiert_root or finde_neuesten_audio_ordner()
    if root is None:
        return []
    mapping = csv_bewertung_mapping(root)
    files = audio_dateien(root)
    if limit > 0:
        files = files[:limit]

    eintraege: list[AudioEintrag] = []
    for path in files:
        genre = inferiere_genre(path, mapping)
        try:
            audio = decode_audio(path)
        except Exception:
            continue
        duration = len(audio) / float(SAMPLE_RATE)
        if duration <= max(75.0, segment_sekunden * 1.5):
            eintraege.append(
                AudioEintrag(
                    path=path,
                    genre=genre,
                    quelle="generiert",
                    label=path.stem,
                    audio=audio,
                    start_sec=0.0,
                    end_sec=duration,
                )
            )
            continue

        frames = int(round(segment_sekunden * SAMPLE_RATE))
        index = 1
        for start in range(0, len(audio), frames):
            segment = audio[start : start + frames]
            if len(segment) < SAMPLE_RATE * 10:
                continue
            start_sec = start / float(SAMPLE_RATE)
            end_sec = (start + len(segment)) / float(SAMPLE_RATE)
            eintraege.append(
                AudioEintrag(
                    path=path,
                    genre=genre,
                    quelle="generiert",
                    label=f"{path.stem}_abschnitt_{index:03d}",
                    audio=segment.copy(),
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
            )
            index += 1
    return eintraege


def schaetze_bpm(audio: np.ndarray) -> float:
    """Schaetzt Tempo grob aus Energie-Impulsen ohne externe Abhaengigkeiten."""

    if len(audio) < SAMPLE_RATE * 8:
        return 0.0
    hop = 512
    frame = 2048
    if len(audio) < frame * 4:
        return 0.0
    values: list[float] = []
    for start in range(0, len(audio) - frame, hop):
        chunk = audio[start : start + frame]
        values.append(rms(chunk))
    envelope = np.asarray(values, dtype=np.float32)
    if len(envelope) < 20:
        return 0.0
    envelope = np.maximum(0.0, np.diff(envelope))
    envelope -= float(np.mean(envelope))
    if float(np.std(envelope)) <= EPS:
        return 0.0
    corr = np.correlate(envelope, envelope, mode="full")[len(envelope) - 1 :]
    frames_per_second = SAMPLE_RATE / hop
    min_lag = int(round(frames_per_second * 60.0 / 150.0))
    max_lag = int(round(frames_per_second * 60.0 / 55.0))
    max_lag = min(max_lag, len(corr) - 1)
    if max_lag <= min_lag:
        return 0.0
    window = corr[min_lag:max_lag]
    if len(window) == 0 or float(np.max(window)) <= EPS:
        return 0.0
    lag = int(np.argmax(window)) + min_lag
    bpm = 60.0 * frames_per_second / max(1, lag)
    # Lofi wird oft halb/doppelt erkannt. In einen sinnvollen Bereich falten.
    while bpm < 65.0:
        bpm *= 2.0
    while bpm > 120.0:
        bpm /= 2.0
    return round(float(bpm), 2)


def audio_kennwerte(audio: np.ndarray) -> dict[str, float]:
    """Berechnet technische Kennwerte fuer Referenz- und generierte Clips."""

    duration = len(audio) / float(SAMPLE_RATE)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.995)) if len(audio) else 0.0
    rms_values = segment_rms_values(audio, step_seconds=1.0)
    active_ratio = sum(value >= -50.0 for value in rms_values) / len(rms_values) if rms_values else 0.0
    quiet_ratio = sum(value < -50.0 for value in rms_values) / len(rms_values) if rms_values else 1.0
    half = len(audio) // 2
    half_drop_db = max(0.0, db(rms(audio[:half])) - db(rms(audio[half:]))) if half else 0.0
    explosion_db = max(rms_values) - float(np.median(rms_values)) if rms_values else 0.0
    rms_range_db = max(rms_values) - min(rms_values) if rms_values else 0.0
    spec = spectral_ratios(audio)
    return {
        "duration_sec": round(duration, 3),
        "rms_db": round(db(rms(audio)), 3),
        "peak_db": round(db(peak), 3),
        "active_ratio": round(active_ratio, 4),
        "quiet_ratio": round(quiet_ratio, 4),
        "half_drop_db": round(half_drop_db, 3),
        "explosion_db": round(float(explosion_db), 3),
        "rms_range_db": round(float(rms_range_db), 3),
        "bass_ratio": round(float(spec["bass_ratio"]), 5),
        "snare_ratio": round(float(spec["snare_ratio"]), 5),
        "high_ratio": round(float(spec["high_ratio"]), 5),
        "tone_ratio": round(float(spec["tone_ratio"]), 5),
        "clipping_ratio": round(clipping_ratio, 6),
        "bpm": schaetze_bpm(audio),
    }


def analysiere_eintraege(eintraege: Iterable[AudioEintrag]) -> list[dict[str, Any]]:
    """Dekodiert Audios und gibt flache Report-Zeilen zurueck."""

    rows: list[dict[str, Any]] = []
    for eintrag in eintraege:
        try:
            audio = eintrag.audio if eintrag.audio is not None else decode_audio(eintrag.path)
            kennwerte = audio_kennwerte(audio)
            status = "ok"
            fehler = ""
        except Exception as exc:
            kennwerte = {feld: 0.0 for feld in METRIK_FELDER}
            status = "fehler"
            fehler = str(exc)

        metadata = eintrag.metadata or {}
        rows.append(
            {
                "label": eintrag.label,
                "quelle": eintrag.quelle,
                "genre": eintrag.genre,
                "genre_name": GENRE_LABELS.get(eintrag.genre, eintrag.genre.replace("_", " ").title()),
                "datei": rel(eintrag.path),
                "start_sec": round(eintrag.start_sec, 3),
                "end_sec": round(eintrag.end_sec, 3),
                "status": status,
                "fehler": fehler,
                "quality_score": metadata.get("quality_score", ""),
                "review_status": metadata.get("review_status", ""),
                "source_file": metadata.get("source_file", ""),
                **kennwerte,
            }
        )
    return rows


def median(values: list[float]) -> float:
    """Robuste Median-Hilfe fuer leere Listen."""

    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return round(float(np.median(clean)), 5) if clean else 0.0


def referenz_statistik(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Berechnet Referenz-Mediane pro Genre."""

    stats: dict[str, dict[str, float]] = {}
    gruppen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            gruppen[str(row.get("genre") or "general_lofi")].append(row)
    for genre, genre_rows in gruppen.items():
        stats[genre] = {
            "anzahl": float(len(genre_rows)),
            **{feld: median([float(row.get(feld) or 0.0) for row in genre_rows]) for feld in METRIK_FELDER},
        }
    return stats


def problem_text(probleme: list[str]) -> str:
    """Gibt eine kurze lesbare Problemzusammenfassung aus."""

    return "; ".join(probleme) if probleme else "technisch nahe an Referenz"


def vergleiche_mit_referenz(row: dict[str, Any], stats: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Vergleicht eine generierte Zeile mit dem passenden Genre-Referenzprofil."""

    genre = str(row.get("genre") or "general_lofi")
    ref = stats.get(genre) or stats.get("general_lofi") or {}
    probleme: list[str] = []
    empfehlungen: list[str] = []
    score = 100.0

    if row.get("status") != "ok":
        return {
            **row,
            "referenz_gefunden": "nein",
            "score": 0,
            "hauptproblem": "Audio konnte nicht analysiert werden",
            "empfehlung": row.get("fehler") or "Datei pruefen",
        }
    if not ref:
        return {
            **row,
            "referenz_gefunden": "nein",
            "score": 50,
            "hauptproblem": "keine passende Referenz gefunden",
            "empfehlung": "Genre-Zuordnung pruefen",
        }

    def wert(name: str) -> float:
        return float(row.get(name) or 0.0)

    def refwert(name: str) -> float:
        return float(ref.get(name) or 0.0)

    if wert("active_ratio") < 0.96:
        probleme.append("zu viele leise oder stille Sekunden")
        empfehlungen.append("Kandidaten mit Sekundentest strenger filtern")
        score -= 18.0
    if wert("half_drop_db") > max(6.0, refwert("half_drop_db") + 3.0):
        probleme.append("zweite Haelfte faellt gegenueber Referenz ab")
        empfehlungen.append("Clips mit schwachem Ende verwerfen")
        score -= 14.0
    if wert("explosion_db") > max(8.0, refwert("explosion_db") + 4.0):
        probleme.append("Energie-Spitze oder Drop zu hart")
        empfehlungen.append("Explosion-Filter und Pegelangleichung verschaerfen")
        score -= 14.0
    if wert("rms_db") > refwert("rms_db") + 5.0:
        probleme.append("insgesamt lauter als MP3-Referenz")
        empfehlungen.append("Lautheit vor dem finalen Mix niedriger normalisieren")
        score -= 8.0
    elif wert("rms_db") < refwert("rms_db") - 7.0:
        probleme.append("insgesamt leiser als MP3-Referenz")
        empfehlungen.append("zu leise Kandidaten aussortieren")
        score -= 8.0
    if wert("peak_db") > -0.5 or wert("clipping_ratio") > 0.002:
        probleme.append("Peak zu nah an 0 dB")
        empfehlungen.append("Limiter/Headroom nach MusicGen-Ausgabe nutzen")
        score -= 10.0
    if wert("bass_ratio") > max(0.55, refwert("bass_ratio") + 0.12):
        probleme.append("Bass staerker als Referenz")
        empfehlungen.append("Bass-Filter oder Prompt gegen dominanten Bass nutzen")
        score -= 12.0
    if wert("high_ratio") > max(0.18, refwert("high_ratio") + 0.07):
        probleme.append("Hoehen/Shaker staerker als Referenz")
        empfehlungen.append("Shaker/Snare-Prompts und Hoehenfilter kontrollieren")
        score -= 10.0
    if wert("tone_ratio") > max(0.20, refwert("tone_ratio") + 0.08):
        probleme.append("Signalton- oder Monotonie-Risiko")
        empfehlungen.append("monotone Kandidaten direkt verwerfen")
        score -= 16.0
    if wert("rms_range_db") < max(1.4, refwert("rms_range_db") * 0.45):
        probleme.append("Dynamik wirkt zu flach")
        empfehlungen.append("Prompt-Variation und Kandidatenauswahl erhoehen")
        score -= 6.0
    if wert("bpm") and refwert("bpm") and abs(wert("bpm") - refwert("bpm")) > 8.0:
        probleme.append("BPM weicht deutlich vom Genre-Referenztempo ab")
        empfehlungen.append("BPM-Filter pro Genre enger setzen")
        score -= 8.0

    return {
        **row,
        "referenz_gefunden": "ja",
        "ref_rms_db": ref.get("rms_db", ""),
        "ref_peak_db": ref.get("peak_db", ""),
        "ref_active_ratio": ref.get("active_ratio", ""),
        "ref_bass_ratio": ref.get("bass_ratio", ""),
        "ref_high_ratio": ref.get("high_ratio", ""),
        "ref_tone_ratio": ref.get("tone_ratio", ""),
        "ref_bpm": ref.get("bpm", ""),
        "score": round(max(0.0, min(100.0, score)), 1),
        "hauptproblem": problem_text(probleme),
        "empfehlung": "; ".join(dict.fromkeys(empfehlungen)) if empfehlungen else "keine technische Aenderung zwingend",
    }


def diagramm_daten(referenz_rows: list[dict[str, Any]], vergleich_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Erstellt kompakte Diagrammdaten fuer Referenz vs. Generierung."""

    rows: list[dict[str, Any]] = []
    gruppen: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in referenz_rows:
        if row.get("status") == "ok":
            gruppen[(str(row["genre"]), "referenz")].append(row)
    for row in vergleich_rows:
        if row.get("status") == "ok":
            gruppen[(str(row["genre"]), "generiert")].append(row)

    for (genre, quelle), items in sorted(gruppen.items()):
        data = {
            "genre": genre,
            "genre_name": GENRE_LABELS.get(genre, genre.replace("_", " ").title()),
            "quelle": quelle,
            "anzahl": len(items),
        }
        for feld in ("rms_db", "active_ratio", "bass_ratio", "high_ratio", "tone_ratio", "explosion_db", "bpm"):
            data[feld] = median([float(item.get(feld) or 0.0) for item in items])
        rows.append(data)
    return rows


def schreibe_optional_diagramm(rows: list[dict[str, Any]], output_path: Path) -> str:
    """Schreibt ein kleines PNG, wenn matplotlib lokal vorhanden ist."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return ""
    if not rows:
        return ""

    genres = sorted({str(row["genre"]) for row in rows})
    metrics = ["bass_ratio", "high_ratio", "tone_ratio", "active_ratio"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 9), sharex=True)
    x = np.arange(len(genres))
    width = 0.36
    for axis, metric in zip(axes, metrics):
        ref_values = []
        gen_values = []
        for genre in genres:
            ref = next((row for row in rows if row["genre"] == genre and row["quelle"] == "referenz"), {})
            gen = next((row for row in rows if row["genre"] == genre and row["quelle"] == "generiert"), {})
            ref_values.append(float(ref.get(metric) or 0.0))
            gen_values.append(float(gen.get(metric) or 0.0))
        axis.bar(x - width / 2, ref_values, width, label="Referenz")
        axis.bar(x + width / 2, gen_values, width, label="Generiert")
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([GENRE_LABELS.get(genre, genre) for genre in genres], rotation=25, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return rel(output_path)


def zusammenfassung(
    referenzen: list[dict[str, Any]],
    generiert: list[dict[str, Any]],
    vergleich: list[dict[str, Any]],
    stats: dict[str, dict[str, float]],
    output_dir: Path,
    referenz_root: Path,
    generiert_root: Path | None,
) -> dict[str, Any]:
    """Erstellt einen JSON-kompatiblen Ergebnisbericht."""

    problem_counter: Counter[str] = Counter()
    for row in vergleich:
        for problem in str(row.get("hauptproblem") or "").split(";"):
            problem = problem.strip()
            if problem and problem != "technisch nahe an Referenz":
                problem_counter[problem] += 1
    scores = [float(row.get("score") or 0.0) for row in vergleich if row.get("status") == "ok"]
    return {
        "status": "finished",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "referenz_root": rel(referenz_root),
        "generiert_root": rel(generiert_root) if generiert_root else "automatisch erkannt",
        "output_dir": rel(output_dir),
        "referenzen": len(referenzen),
        "generierte_abschnitte": len(generiert),
        "referenz_profile": stats,
        "score_mittelwert": round(float(np.mean(scores)), 2) if scores else 0.0,
        "score_median": round(float(np.median(scores)), 2) if scores else 0.0,
        "haeufigste_probleme": [
            {"problem": key, "anzahl": value} for key, value in problem_counter.most_common()
        ],
        "hinweis": (
            "Technischer Referenzvergleich. Er ersetzt keine menschliche Bewertung, "
            "zeigt aber messbar, wo generierte Audios von den MP3-Referenzen abweichen."
        ),
    }


def schreibe_bericht(report: dict[str, Any], output_dir: Path) -> Path:
    """Schreibt eine kurze lesbare Textzusammenfassung."""

    lines = [
        "Referenzvergleich",
        "=================",
        f"Referenzen: {report['referenzen']}",
        f"Generierte Abschnitte: {report['generierte_abschnitte']}",
        f"Score Median: {report['score_median']}",
        "",
        "Haeufigste Probleme:",
    ]
    probleme = report.get("haeufigste_probleme") or []
    if not probleme:
        lines.append("- keine technischen Hauptprobleme erkannt")
    else:
        for item in probleme[:8]:
            lines.append(f"- {item['problem']}: {item['anzahl']}x")
    lines.extend(
        [
            "",
            "Einordnung:",
            "Wenn generierte Audios deutlich schlechter wirken als die MP3s,",
            "sollten zuerst Bass, stille Sekunden, Energie-Spitzen, Hoehen/Shaker",
            "und monotone Signalton-Anteile gegen die Referenzen geprueft werden.",
        ]
    )
    path = output_dir / "bericht.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI fuer den lokalen Vergleich."""

    default_output = PROJECT_ROOT / "training" / "musicgen" / "referenzvergleich" / datetime.now().strftime(
        "vergleich_%Y%m%d_%H%M%S"
    )
    parser = argparse.ArgumentParser(description="Vergleicht generierte MusicGen-Audios mit MP3-Referenzen.")
    parser.add_argument(
        "--referenz-root",
        default=str(PROJECT_ROOT / "daten" / "processed" / "lora_training"),
        help="Geprueftes Trainingsdataset mit train/valid/test/data.jsonl.",
    )
    parser.add_argument(
        "--generiert-root",
        default="",
        help="Ordner oder Datei mit erzeugten Audios. Leer bedeutet: neuester Bewertungs-audio-Ordner.",
    )
    parser.add_argument("--ausgabe-dir", default=str(default_output), help="Report-Ordner.")
    parser.add_argument("--referenzen-pro-genre", type=int, default=20)
    parser.add_argument("--limit-generiert", type=int, default=0)
    parser.add_argument("--segment-sekunden", type=float, default=30.0)
    parser.add_argument("--nur-plan", action="store_true", help="Nur anzeigen, was analysiert wuerde.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Fuehrt den Vergleich aus und schreibt CSV/JSON/Diagrammdaten."""

    args = parse_args(argv)
    referenz_root = loese_pfad(args.referenz_root)
    generiert_root = loese_pfad(args.generiert_root) if args.generiert_root else finde_neuesten_audio_ordner()
    output_dir = loese_pfad(args.ausgabe_dir)

    referenz_eintraege = sammle_referenzen(referenz_root, max(1, args.referenzen_pro_genre))
    generierte_eintraege = sammle_generierte(generiert_root, max(0, args.limit_generiert), max(10.0, args.segment_sekunden))

    print("Referenzvergleich", flush=True)
    print("=================", flush=True)
    print(f"Referenzen: {len(referenz_eintraege)} Clips", flush=True)
    print(f"Generiert:  {len(generierte_eintraege)} Abschnitt(e)", flush=True)
    print(f"Ausgabe:    {rel(output_dir)}", flush=True)
    if generiert_root:
        print(f"Quelle:     {rel(generiert_root)}", flush=True)
    else:
        print("Quelle:     keine generierten Audios gefunden", flush=True)
    print("", flush=True)

    if args.nur_plan:
        print("Status: Plan erstellt, keine Analyse gestartet.", flush=True)
        return 0
    if not referenz_eintraege:
        print("Status: Fehler, keine Referenzclips gefunden.", flush=True)
        return 2
    if not generierte_eintraege:
        print("Status: Fehler, keine generierten Audios gefunden.", flush=True)
        return 3

    output_dir.mkdir(parents=True, exist_ok=True)
    referenz_rows = analysiere_eintraege(referenz_eintraege)
    generierte_rows = analysiere_eintraege(generierte_eintraege)
    stats = referenz_statistik(referenz_rows)
    vergleich_rows = [vergleiche_mit_referenz(row, stats) for row in generierte_rows]
    diagramm_rows = diagramm_daten(referenz_rows, vergleich_rows)

    write_csv(output_dir / "referenzen.csv", referenz_rows)
    write_csv(output_dir / "generierte_abschnitte.csv", generierte_rows)
    write_csv(output_dir / "vergleich.csv", vergleich_rows)
    write_csv(output_dir / "diagramm_daten.csv", diagramm_rows)
    diagramm_png = schreibe_optional_diagramm(diagramm_rows, output_dir / "diagramm.png")
    report = zusammenfassung(
        referenz_rows,
        generierte_rows,
        vergleich_rows,
        stats,
        output_dir,
        referenz_root,
        generiert_root,
    )
    if diagramm_png:
        report["diagramm_png"] = diagramm_png
    write_json(output_dir / "zusammenfassung.json", report)
    bericht_path = schreibe_bericht(report, output_dir)

    print("Status: finished", flush=True)
    print(f"Score Median: {report['score_median']}", flush=True)
    if report["haeufigste_probleme"]:
        top = report["haeufigste_probleme"][0]
        print(f"Hauptproblem: {top['problem']} ({top['anzahl']}x)", flush=True)
    else:
        print("Hauptproblem: keine technische Hauptabweichung", flush=True)
    print(f"Bericht: {rel(bericht_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
