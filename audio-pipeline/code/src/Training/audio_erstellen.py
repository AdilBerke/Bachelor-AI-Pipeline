#!/usr/bin/env python3
"""Erzeugt Longform-Audio aus frisch generierten MusicGen-Abschnitten.

Diese aktive Pipeline ersetzt im Standardlauf die alte Clip-Pool-Pipeline. Die alte Pipeline
setzt vorhandene bewertete Clips zusammen. Diese Version laedt MusicGen mit dem
besten LoRA-Checkpoint, erzeugt neue kurze Abschnitte, prueft jeden Kandidaten
technisch und baut daraus erst danach eine lange Audio.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import random
import re
import runpy
import shutil
import subprocess
import sys
import time
import types
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

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
TRANSITION_SCORE_WEIGHT = 3.0
MAX_AKZEPTIERTER_UEBERGANG_SCORE = 24.0
DEFAULT_RHYTHMUS_ANTEIL_PROZENT = 5.0
DEFAULT_MIN_RHYTHMUS_SEKUNDEN = 90.0
DEFAULT_ADAPTER = (
    PROJECT_ROOT
    / "training"
    / "musicgen"
    / "lora_training"
    / "adapter.pt"
)
DEFAULT_LORA_RUN = PROJECT_ROOT / "training" / "musicgen" / "lora_training"
DEFAULT_LORA_PLAN = DEFAULT_LORA_RUN / "training_plan.json"
DEFAULT_CLAP_MODELL = (
    PROJECT_ROOT / "daten" / "modelle" / "audio_analyse" / "clap_htsat_unfused"
)
DEFAULT_GENRE_REFERENZEN = (
    PROJECT_ROOT
    / "training"
    / "bewertungen"
    / "musicgen"
    / "lora_review_001"
)
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


def adapter_fuer_generierung_freigegeben(adapter_path: Path) -> bool:
    """Laesst den aktuellen LoRA-Run erst nach seinem Abschluss zu."""
    if not adapter_path.is_file():
        return False
    try:
        adapter_path.resolve().relative_to(DEFAULT_LORA_RUN.resolve())
    except ValueError:
        return True
    if not DEFAULT_LORA_PLAN.is_file():
        return False
    try:
        plan = json.loads(DEFAULT_LORA_PLAN.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return plan.get("status") == "finished" and int(plan.get("returncode") or 0) == 0
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
class GeplanterAbschnitt:
    """Ein frisch zu erzeugender Abschnitt innerhalb eines Longform-Blocks."""

    block: int
    abschnitt: int
    start_sec: float
    end_sec: float
    prompt: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frische MusicGen-Longform-Audio erzeugen.")
    parser.add_argument("--dauer", default="20m", help="Zieldauer, z.B. 20m, 30m, 1h, 3h.")
    parser.add_argument("--genre", default="Lofi", help="Genre, z.B. Jazz Lofi oder Dreamy Lofi.")
    parser.add_argument("--stimmung", default="", help="Optionale Stimmung, z.B. calm night mood.")
    parser.add_argument("--instrumente", default="", help="Optionale Instrumente, z.B. mellow piano, rhodes.")
    parser.add_argument("--name", default="")
    parser.add_argument(
        "--ausgabe-root",
        default=str(PROJECT_ROOT / "training" / "ausgaben" / "musicgen_generiert"),
    )
    parser.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER))
    parser.add_argument(
        "--review-adapter-erlauben",
        action="store_true",
        help="Erlaubt einen noch nicht freigegebenen LoRA-Kandidaten nur fuer Test-/Review-Audios.",
    )
    parser.add_argument("--model-id", default="facebook/musicgen-melody-large")
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "daten" / "modelle" / "musicgen" / "facebook_musicgen_melody_large"),
    )
    parser.add_argument("--allow-remote-model", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="0 = jedes Mal neuer Zufall.")
    parser.add_argument("--abschnitt-sekunden", type=float, default=30.0)
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
    )
    parser.add_argument(
        "--min-rhythmus-sekunden",
        type=float,
        default=DEFAULT_MIN_RHYTHMUS_SEKUNDEN,
    )
    parser.add_argument(
        "--blockmodus-prozentual",
        action="store_true",
        help="Dokumentiert, dass die uebergebene Blockdauer prozentual berechnet wurde.",
    )
    parser.add_argument("--block-variation-sekunden", type=float, default=0.0)
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
    parser.add_argument("--kandidaten-pro-abschnitt", type=int, default=5)
    parser.add_argument(
        "--max-generierte-kandidaten",
        type=int,
        default=0,
        help="0 = kein hartes Limit. Stoppt sonst nach dieser Anzahl echter MusicGen-Kandidaten.",
    )
    parser.add_argument("--max-abschnitte", type=int, default=0, help="0 = alle benoetigten Abschnitte.")
    parser.add_argument(
        "--block-looping-aktiv",
        dest="block_looping_aktiv",
        action="store_true",
        default=True,
        help="Ein frisch erzeugter guter 30s-Clip wird pro Musikblock weich verlaengert.",
    )
    parser.add_argument("--block-looping-deaktivieren", dest="block_looping_aktiv", action="store_false")
    parser.add_argument(
        "--kontinuierliche-bloecke-aktiv",
        dest="kontinuierliche_bloecke_aktiv",
        action="store_true",
        default=False,
        help=(
            "Erzeugt jeden 90s- bis 3min-Rhythmusblock als echte MusicGen-Fortsetzung "
            "statt einen 30s-Clip zu wiederholen."
        ),
    )
    parser.add_argument(
        "--kontinuierliche-bloecke-deaktivieren",
        dest="kontinuierliche_bloecke_aktiv",
        action="store_false",
    )
    parser.add_argument(
        "--erweiterungs-schritt-sekunden",
        type=float,
        default=12.0,
        help="MusicGen-Schritt fuer Fortsetzungen; kleiner bedeutet mehr Kontext und mehr Rechenzeit.",
    )
    parser.add_argument(
        "--loop-crossfade-sekunden",
        type=float,
        default=0.0,
        help="0 = automatisch ein Beat; sonst gewuenschter Crossfade innerhalb eines Loop-Blocks.",
    )
    parser.add_argument("--tempo-variation-prozent", type=float, default=1.5)
    parser.add_argument("--tempo-phase-sekunden", type=float, default=60.0)
    parser.add_argument("--fallback-pool", action="store_true", help="Erlaubt spaeter bewertete Clips als Notfallquelle.")
    parser.add_argument(
        "--nicht-fortsetzen",
        dest="resume_existing",
        action="store_false",
        default=True,
        help="Vorhandene akzeptierte Clips im Run-Ordner ignorieren und neu beginnen.",
    )
    parser.add_argument("--temperature", type=float, default=0.72)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--cfg-coef", type=float, default=4.0)
    parser.add_argument("--vram-limit-fraction", type=float, default=0.80)
    parser.add_argument("--disable-vram-limit", action="store_true")
    parser.add_argument("--crossfade-sekunden", type=float, default=3.0)
    parser.add_argument("--fade-in-sekunden", type=float, default=5.0)
    parser.add_argument("--fade-out-sekunden", type=float, default=5.0)
    parser.add_argument("--interner-crossfade-sekunden", type=float, default=2.0)
    parser.add_argument(
        "--anschluss-conditioning-aktiv",
        dest="anschluss_conditioning_aktiv",
        action="store_true",
        default=False,
        help="Nutzt das Ende des vorherigen Clips als Melody-Referenz fuer den Anfang des naechsten Clips.",
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
    parser.add_argument("--ziel-rms-db", type=float, default=-22.0)
    parser.add_argument("--peak-limit", type=float, default=0.92)
    parser.add_argument(
        "--sekunden-pruefung-aktiv",
        dest="sekunden_pruefung_aktiv",
        action="store_true",
        default=True,
        help="Prueft jeden 30s-Kandidaten sekundenweise auf durchgehend hoerbaren Ton.",
    )
    parser.add_argument("--sekunden-pruefung-deaktivieren", dest="sekunden_pruefung_aktiv", action="store_false")
    parser.add_argument(
        "--min-sekunden-rms-db",
        type=float,
        default=-52.0,
        help="Mindestlautheit pro 1s-Fenster. Darunter gilt eine Sekunde als ohne Ton.",
    )
    parser.add_argument(
        "--genre-pruefung-aktiv",
        dest="genre_pruefung_aktiv",
        action="store_true",
        default=True,
        help="Vergleicht neue Kandidaten lokal mit menschlich bewerteten Genre-Referenzen.",
    )
    parser.add_argument(
        "--genre-pruefung-deaktivieren",
        dest="genre_pruefung_aktiv",
        action="store_false",
    )
    parser.add_argument(
        "--mp3-referenz-pruefung-aktiv",
        dest="mp3_referenz_pruefung_aktiv",
        action="store_true",
        default=True,
        help="Vergleicht neue Kandidaten technisch mit guten MP3-Trainingsclips.",
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
    parser.add_argument(
        "--referenz-score-limit",
        type=float,
        default=18.0,
        help="Maximal erlaubte technische Abweichung zur MP3-Referenz.",
    )
    parser.add_argument("--clap-modell-pfad", default=str(DEFAULT_CLAP_MODELL))
    parser.add_argument("--genre-referenz-ordner", default=str(DEFAULT_GENRE_REFERENZEN))
    parser.add_argument("--min-genre-aehnlichkeit", type=float, default=0.62)
    parser.add_argument("--min-genre-abstand", type=float, default=0.02)
    parser.add_argument("--min-qualitaets-abstand", type=float, default=0.0)
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
        help="Speichert die finale Audio nur lokal.",
    )
    parser.add_argument("--nur-plan", action="store_true", help="Nur Plan/Prompts/Reports schreiben, kein Modell laden.")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


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


def resolve_rhythm_block_seconds(
    duration_sec: float,
    requested_block_sec: float,
    percentage: float,
    minimum_sec: float,
) -> float:
    """Berechnet die Ziel-Rhythmusdauer mit prozentualem Standard."""

    minimum = max(1.0, float(minimum_sec))
    if requested_block_sec > 0:
        target = max(minimum, float(requested_block_sec))
    else:
        if percentage <= 0 or percentage > 100:
            raise ValueError("rhythmus-anteil-prozent muss zwischen 0 und 100 liegen.")
        target = max(minimum, duration_sec * float(percentage) / 100.0)
    return min(duration_sec, target)


def actual_seed(requested_seed: int) -> int:
    if requested_seed > 0:
        return requested_seed
    return int(time.time_ns() ^ (os.getpid() << 16)) & 0xFFFFFFFF


def section_seed(base_seed: int, section_index: int, attempt: int = 0) -> int:
    """Erzeugt reproduzierbare, aber unterschiedliche Seeds pro Abschnitt."""

    return int((base_seed + section_index * 1009 + attempt * 37) % (2**31 - 1)) or 1


def effective_crossfade_seconds(section_seconds: float, requested_seconds: float) -> float:
    """Begrenzt Crossfade auf einen musikalisch sinnvollen Segmentanteil."""

    if requested_seconds <= 0:
        return 0.0
    max_by_section = max(1.0, section_seconds * 0.20)
    return round(min(float(requested_seconds), max_by_section), 3)


def estimated_assembled_duration(
    section_count: int,
    section_seconds: float,
    crossfade_seconds: float,
) -> float:
    """Schaetzt die Endlaenge nach dem Zusammenbau mit Crossfades."""

    if section_count <= 0:
        return 0.0
    overlap = max(0.0, crossfade_seconds)
    return section_count * section_seconds - max(0, section_count - 1) * overlap


def required_section_count(
    target_duration_sec: float,
    section_seconds: float,
    crossfade_seconds: float,
) -> int:
    """Berechnet, wie viele Clips fuer die Zielzeit trotz Crossfade noetig sind."""

    if target_duration_sec <= 0:
        return 0
    if section_seconds <= 0:
        raise ValueError("abschnitt-sekunden muss groesser als 0 sein.")
    if target_duration_sec <= section_seconds:
        return 1
    useful_seconds = max(1.0, section_seconds - max(0.0, crossfade_seconds))
    return int(math.ceil((target_duration_sec - max(0.0, crossfade_seconds)) / useful_seconds))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json_optional(path: Path) -> Dict[str, Any]:
    """Liest JSON, falls vorhanden; bei defekten Dateien leer weiterarbeiten."""

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def cleanup_cuda_cache() -> None:
    """Raeumt nach Kandidaten auf, um lange Generierungslaeufe stabiler zu halten."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def write_progress(
    run_dir: Path,
    *,
    status: str,
    section_index: int,
    total_sections: int,
    candidate_index: Any = "",
    total_candidates: Any = "",
    accepted_count: int = 0,
    rejected_count: int = 0,
    message: str = "",
    explicit_percent: Optional[float] = None,
    candidate_progress_percent: Optional[float] = None,
) -> None:
    """Schreibt den aktuellen Longform-Fortschritt fuer Terminal und Website."""

    if explicit_percent is None:
        percent = 0.0
        if total_sections > 0:
            percent = section_index / total_sections * 100.0
    else:
        percent = explicit_percent
    percent = min(100.0, max(0.0, percent))
    write_json(
        run_dir / "fortschritt.json",
        {
            "status": status,
            "progress_percent": round(percent, 2),
            "current_section": section_index,
            "total_sections": total_sections,
            "current_candidate": candidate_index,
            "total_candidates": total_candidates,
            "accepted_clips": accepted_count,
            "rejected_clips": rejected_count,
            "message": message,
            "candidate_progress_percent": (
                round(candidate_progress_percent, 2)
                if candidate_progress_percent is not None
                else None
            ),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def load_lora_helpers() -> types.ModuleType:
    """Laedt die vorhandenen LoRA-Hilfsfunktionen aus `musicgen_steuerung.py`.

    Die kompakte Projektstruktur buendelt Training und LoRA-Hilfen aktuell in
    einer Datei. Damit hier keine zweite grosse LoRA-Implementierung entsteht,
    wird das vorhandene Hilfsmodul dynamisch wiederverwendet.
    """

    module_name = "lora_utils"
    if module_name in sys.modules:
        return sys.modules[module_name]
    namespace = runpy.run_path(str(PROJECT_ROOT / "code" / "src" / "Training" / "musicgen_steuerung.py"), run_name="not_main")
    source = namespace["MODULES"][module_name]
    module = types.ModuleType(module_name)
    module.__file__ = str(PROJECT_ROOT / "code" / "src" / "Training" / "musicgen_steuerung.py")
    sys.modules[module_name] = module
    exec(compile(source, "musicgen_steuerung.py:lora_utils", "exec"), module.__dict__)
    return module


def load_musicgen_stabil(model_source: str, device: str) -> Any:
    """Laedt MusicGen mit deaktivierter memory_efficient-Attention.

    Die lokal installierte xFormers-Version passt nicht zur aktuellen
    Torch/CUDA-Version. AudioCraft importiert xFormers trotzdem und kann bei
    `memory_efficient=True` nativ abstuerzen. Diese Ladefunktion nutzt dieselben
    Checkpoints, setzt aber nur fuer den laufenden Prozess
    `transformer_lm.memory_efficient=False`.
    """

    from omegaconf import OmegaConf

    from audiocraft.models import builders
    from audiocraft.models.loaders import _delete_param, load_compression_model, load_lm_model_ckpt
    from audiocraft.models.musicgen import MusicGen

    pkg = load_lm_model_ckpt(model_source)
    cfg = OmegaConf.create(pkg["xp.cfg"])
    OmegaConf.set_struct(cfg, False)
    cfg.device = str(device)
    cfg.dtype = "float32" if str(device) == "cpu" else "float16"
    if "transformer_lm" in cfg:
        cfg.transformer_lm.memory_efficient = False
        cfg.transformer_lm.checkpointing = "none"
    OmegaConf.set_struct(cfg, True)
    _delete_param(cfg, "conditioners.self_wav.chroma_stem.cache_path")
    _delete_param(cfg, "conditioners.args.merge_text_conditions_p")
    _delete_param(cfg, "conditioners.args.drop_desc_p")

    lm = builders.get_lm_model(cfg)
    lm.load_state_dict(pkg["best_state"])
    lm.eval()
    lm.cfg = cfg
    compression_model = load_compression_model(model_source, device=device)
    if "self_wav" in lm.condition_provider.conditioners:
        lm.condition_provider.conditioners["self_wav"].match_len_on_eval = True
        lm.condition_provider.conditioners["self_wav"]._use_masking = False
    return MusicGen(model_source, compression_model, lm)


def load_lora_checkpoint_kompatibel(model: Any, adapter_path: Path, helpers: types.ModuleType, device: str) -> Dict[str, Any]:
    """Laedt LoRA-Adapter auch bei stabiler Attention-Modulstruktur.

    Ohne memory_efficient-Attention liegt das Output-Projection-Modul unter
    `self_attn.mha.out_proj`. Aeltere Adapter wurden aber mit
    `self_attn.out_proj` gespeichert. Die Gewichte sind kompatibel; nur der
    Modulpfad wird beim Laden uebersetzt.
    """

    import torch

    # Den Adapter zuerst auf CPU laden. Direktes Laden auf CUDA kann bei
    # grossen Checkpoints und instabiler CUDA-Umgebung native Loader-Fehler
    # ausloesen; die Tensoren werden unten kontrolliert auf das Zielgeraet
    # verschoben.
    payload = torch.load(adapter_path, map_location="cpu")
    if payload.get("format") != "musicgen_lora_adapter_v1":
        raise ValueError(f"Unsupported LoRA checkpoint format: {adapter_path}")
    config = helpers.LoraConfig.from_dict(payload["lora_config"])
    modules = dict(model.named_modules())
    mapped = 0
    merged = 0
    for name, state in payload["adapter_state"].items():
        target_name = name
        if target_name not in modules:
            target_name = name.replace(".self_attn.out_proj", ".self_attn.mha.out_proj")
        if target_name not in modules:
            raise KeyError(f"LoRA module missing in current model: {name}")
        target_module = modules[target_name]
        if not hasattr(target_module, "weight"):
            raise TypeError(f"LoRA target has no weight tensor: {target_name}")
        lora_a = state["lora_a.weight"].to(device=device, dtype=torch.float32)
        lora_b = state["lora_b.weight"].to(device=device, dtype=torch.float32)
        update = (lora_b @ lora_a) * (float(config.alpha) / float(config.rank))
        with torch.no_grad():
            target_module.weight.data.add_(update.to(dtype=target_module.weight.dtype))
        if target_name != name:
            mapped += 1
        merged += 1
    payload["runtime_adapter_mapping"] = {
        "self_attn.out_proj_to_self_attn.mha.out_proj": mapped,
        "merged_linear_layers": merged,
        "hinweis": "LoRA wird nur im laufenden Prozess in die Gewichte gemergt; Checkpoint-Datei bleibt unveraendert.",
    }
    return payload


def db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), EPS))


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


_LIBROSA: Any = None
_LIBROSA_CHECKED = False


def optional_librosa() -> Any:
    """Laedt librosa nur, wenn es lokal vorhanden ist."""

    global _LIBROSA, _LIBROSA_CHECKED
    if _LIBROSA_CHECKED:
        return _LIBROSA
    _LIBROSA_CHECKED = True
    try:
        import librosa  # type: ignore

        _LIBROSA = librosa
    except Exception:
        _LIBROSA = None
    return _LIBROSA


def estimate_bpm(audio: np.ndarray, target_bpm: float) -> tuple[Optional[float], str]:
    """Schaetzt den BPM-Wert vorsichtig und gibt auch die Methode zurueck."""

    if len(audio) < SAMPLE_RATE * 8 or rms(audio) <= EPS:
        return None, "unavailable"

    librosa = optional_librosa()
    if librosa is not None:
        try:
            tempo = librosa.feature.tempo(y=audio.astype(np.float32), sr=SAMPLE_RATE)
            if len(tempo):
                value = float(np.asarray(tempo).ravel()[0])
                if math.isfinite(value) and value > 0:
                    return normalize_bpm_to_target(value, target_bpm), "librosa"
        except Exception:
            pass

    estimated = estimate_bpm_from_rms_onsets(audio, target_bpm)
    if estimated is None:
        return None, "unavailable"
    return estimated, "rms_autocorrelation"


def normalize_bpm_to_target(value: float, target_bpm: float) -> float:
    """Korrigiert einfache Half-/Double-Time-Schaetzungen Richtung Ziel-BPM."""

    candidates = [value]
    current = value
    while current < target_bpm * 0.75:
        current *= 2.0
        candidates.append(current)
    current = value
    while current > target_bpm * 1.5:
        current /= 2.0
        candidates.append(current)
    return float(min(candidates, key=lambda item: abs(item - target_bpm)))


def estimate_bpm_from_rms_onsets(audio: np.ndarray, target_bpm: float) -> Optional[float]:
    """Einfache BPM-Schaetzung ueber RMS-Onsets und Autokorrelation."""

    frame = 2048
    hop = 512
    if len(audio) < frame * 4:
        return None
    energies: List[float] = []
    for start in range(0, len(audio) - frame, hop):
        chunk = audio[start : start + frame]
        energies.append(math.log(max(rms(chunk), EPS)))
    if len(energies) < 16:
        return None
    envelope = np.asarray(energies, dtype=np.float32)
    envelope = envelope - float(np.mean(envelope))
    onset = np.diff(envelope, prepend=envelope[0])
    onset = np.maximum(onset, 0.0)
    if float(np.sum(onset)) <= EPS:
        return None
    onset = onset / (float(np.max(onset)) + EPS)

    bpm_min = max(45.0, target_bpm - 35.0)
    bpm_max = min(140.0, target_bpm + 35.0)
    best_bpm: Optional[float] = None
    best_score = 0.0
    for bpm in np.arange(bpm_min, bpm_max + 0.001, 0.5):
        lag = int(round((60.0 / float(bpm)) * SAMPLE_RATE / hop))
        if lag <= 1 or lag >= len(onset) - 2:
            continue
        left = onset[:-lag]
        right = onset[lag:]
        score = float(np.dot(left, right)) / (float(np.linalg.norm(left) * np.linalg.norm(right)) + EPS)
        # Bei Lofi ist die Onset-Schaetzung oft verrauscht. Eine leichte
        # Zielnaehe-Priorisierung stabilisiert Half-/Double-Time-Ausreisser.
        proximity = max(0.75, 1.0 - abs(float(bpm) - target_bpm) / max(target_bpm, EPS) * 0.2)
        score *= proximity
        if score > best_score:
            best_score = score
            best_bpm = float(bpm)
    if best_bpm is None or best_score < 0.05:
        return None
    return normalize_bpm_to_target(best_bpm, target_bpm)


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
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def export_mp3(wav_path: Path, mp3_path: Path) -> None:
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


def apply_post_eq(wav_path: Path) -> None:
    """Wendet sanfte EQ-Kette auf die fertige Longform-Audio an.

    Reduziert Bass und den Shaker-/Rauschbereich deutlich und begrenzt den
    Peak danach per Soft-Limiter. Behebt damit
    die haeufigsten Kritikpunkte aus den menschlichen Bewertungen, die
    MusicGen per Prompt allein nicht zuverlaessig loesen kann.
    """
    tmp = wav_path.with_suffix(".eq_tmp.wav")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(wav_path),
                "-af",
                (
                    "equalizer=f=100:width_type=o:width=2:g=-3,"
                    "equalizer=f=7500:width_type=o:width=1.4:g=-3,"
                    "equalizer=f=10500:width_type=o:width=2:g=-6,"
                    "lowpass=f=12500,"
                    "alimiter=limit=0.95:level=true"
                ),
                str(tmp),
            ],
            check=True,
        )
        shutil.move(str(tmp), str(wav_path))
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def convert_wav(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            str(target),
        ],
        check=True,
    )


def normalize_audio(audio: np.ndarray, target_rms_db: float, peak_limit: float) -> np.ndarray:
    current_rms = rms(audio)
    if current_rms <= EPS:
        return audio.astype(np.float32)
    target_rms = 10.0 ** (target_rms_db / 20.0)
    gain = target_rms / current_rms
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak * gain > peak_limit:
        gain = peak_limit / max(peak, EPS)
    return (audio * gain).astype(np.float32)


def melody_reference_from_previous(
    previous_clip: Path,
    duration_sec: float,
    reference_sec: float,
) -> Any:
    """Baut eine Melody-Referenz aus dem Ende des vorherigen Clips.

    MusicGen Melody kann mit `generate_with_chroma` eine harmonische Kontur
    bekommen. Fuer weichere Longform-Uebergaenge legen wir die letzten Sekunden
    des vorherigen Clips an den Anfang der Referenzspur. Der Rest bleibt leise,
    damit MusicGen nur den Einstieg an das vorherige Ende anlehnt und danach
    wieder frei variieren kann.
    """

    import torch

    audio = decode_audio(previous_clip)
    total_frames = max(1, int(round(duration_sec * SAMPLE_RATE)))
    reference_frames = max(1, min(len(audio), int(round(reference_sec * SAMPLE_RATE)), total_frames))
    tail = audio[-reference_frames:].astype(np.float32, copy=True)
    if len(tail) > 16:
        fade_frames = min(len(tail), SAMPLE_RATE)
        tail[:fade_frames] *= np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
        tail[-fade_frames:] *= np.linspace(1.0, 0.0, fade_frames, dtype=np.float32)
    peak = float(np.max(np.abs(tail))) if len(tail) else 0.0
    if peak > 0.65:
        tail *= 0.65 / peak
    reference = np.zeros(total_frames, dtype=np.float32)
    reference[: len(tail)] = tail
    return torch.from_numpy(reference).view(1, 1, -1)


_GENRE_MELODIE_POOL_CACHE: Dict[str, Dict[str, List[Path]]] = {}


def genre_melodie_pool(dataset_root: Path) -> Dict[str, List[Path]]:
    """Gruppiert echte Trainingsclips nach Genre, fuer Melody-Conditioning.

    Wird pro Prozess einmal aus dem Trainingsdataset gebaut (dieselbe Quelle
    wie der MP3-Referenzvergleich) und danach zwischengespeichert, damit
    nicht bei jedem Kandidaten neu durchs Manifest gelesen wird.
    """

    cache_key = str(dataset_root)
    cached = _GENRE_MELODIE_POOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    grouped: Dict[str, List[Path]] = {}
    for split in ("train", "valid", "test"):
        for row in read_jsonl_rows(dataset_root / split / "data.jsonl"):
            path = resolve_dataset_audio_path(row.get("path"))
            if not path.is_file():
                continue
            genre = referenz_genre_key(row.get("lora_genre") or row.get("primary_genre") or row.get("genre"))
            grouped.setdefault(genre, []).append(path)

    for genre in grouped:
        grouped[genre] = sorted(grouped[genre])
    _GENRE_MELODIE_POOL_CACHE[cache_key] = grouped
    return grouped


def melody_reference_from_genre(
    genre: str,
    seed: int,
    duration_sec: float,
    dataset_root: Path,
) -> Any:
    """Baut eine Melody-Referenz aus einem echten Clip des Zielgenres.

    Anders als `melody_reference_from_previous` (nur Anfangs-Anker fuer
    weiche Uebergaenge) liefert diese Funktion die volle Referenzspur, damit
    MusicGen ueber den gesamten Abschnitt einer echten harmonischen Kontur
    des Zielgenres folgen kann statt nur dem Text-Prompt.
    """

    import torch

    pool = genre_melodie_pool(dataset_root)
    ziel = referenz_genre_key(genre)
    kandidaten = pool.get(ziel) or []
    if not kandidaten:
        return None

    clip_path = kandidaten[seed % len(kandidaten)]
    try:
        audio = decode_audio(clip_path)
    except Exception:
        return None
    if len(audio) == 0:
        return None

    total_frames = max(1, int(round(duration_sec * SAMPLE_RATE)))
    if len(audio) >= total_frames:
        reference = audio[:total_frames].astype(np.float32, copy=True)
    else:
        wiederholungen = int(np.ceil(total_frames / len(audio)))
        reference = np.tile(audio, wiederholungen)[:total_frames].astype(np.float32, copy=True)

    peak = float(np.max(np.abs(reference))) if len(reference) else 0.0
    if peak > 0.9:
        reference *= 0.9 / peak
    return torch.from_numpy(reference).view(1, 1, -1)


def soften_clip_edges(audio: np.ndarray, fade_seconds: float) -> np.ndarray:
    """Macht Clip-Anfang und -Ende leicht weicher fuer bessere Crossfades."""

    fade_frames = int(round(max(0.0, fade_seconds) * SAMPLE_RATE))
    fade_frames = min(fade_frames, len(audio) // 3)
    if fade_frames <= 0:
        return audio.astype(np.float32)
    out = audio.astype(np.float32).copy()
    curve = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
    out[:fade_frames] *= np.sin(curve * math.pi / 2.0)
    out[-fade_frames:] *= np.cos(curve * math.pi / 2.0)
    return out


def append_with_crossfade(base: np.ndarray, clip: np.ndarray, crossfade_frames: int) -> np.ndarray:
    if len(base) == 0:
        return clip.copy()
    fade_frames = min(crossfade_frames, len(base) // 2, len(clip) // 2)
    if fade_frames <= 0:
        return np.concatenate([base, clip])
    curve = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
    fade_out = np.cos(curve * math.pi / 2.0)
    fade_in = np.sin(curve * math.pi / 2.0)
    overlap = base[-fade_frames:] * fade_out + clip[:fade_frames] * fade_in
    return np.concatenate([base[:-fade_frames], overlap, clip[fade_frames:]])


def build_looped_block(
    audio: np.ndarray,
    target_duration_sec: float,
    loop_crossfade_sec: float,
    target_bpm: float,
    tempo_variation_percent: float,
    tempo_phase_sec: float,
    tempo_start_phase: int,
    seed: int,
) -> tuple[np.ndarray, LoopAnalyse, int]:
    """Verlaengert einen guten Clip ueber automatisch gefundene Taktgrenzen."""

    return baue_loop_block(
        audio,
        target_duration_sec,
        sample_rate=SAMPLE_RATE,
        ziel_bpm=target_bpm,
        min_loop_sec=18.0,
        max_loop_sec=min(29.0, max(18.0, len(audio) / SAMPLE_RATE - 0.25)),
        suchfenster_sec=7.0,
        crossfade_sec=loop_crossfade_sec,
        tempo_variation_percent=tempo_variation_percent,
        tempo_phase_sec=tempo_phase_sec,
        tempo_start_phase=tempo_start_phase,
        seed=seed,
    )


def audio_metrics(
    path: Path,
    expected_duration: float,
    target_bpm: float = 78.0,
    bpm_tolerance: float = 4.0,
    bpm_check_enabled: bool = True,
    second_check_enabled: bool = True,
    min_second_rms_db: float = -52.0,
) -> Dict[str, Any]:
    audio = decode_audio(path)
    duration = len(audio) / float(SAMPLE_RATE)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.995)) if len(audio) else 0.0
    total_rms = rms(audio)
    segment_frames = int(round(3.0 * SAMPLE_RATE))
    segment_rms_db: List[float] = []
    for start in range(0, len(audio), segment_frames):
        segment = audio[start : start + segment_frames]
        if len(segment) >= segment_frames * 0.5:
            segment_rms_db.append(db(rms(segment)))

    second_frames = int(round(1.0 * SAMPLE_RATE))
    second_rms_db: List[float] = []
    for start in range(0, len(audio), second_frames):
        segment = audio[start : start + second_frames]
        if len(segment) >= second_frames * 0.75:
            second_rms_db.append(db(rms(segment)))
    active_seconds = [value >= min_second_rms_db for value in second_rms_db]
    active_second_ratio = sum(active_seconds) / len(active_seconds) if active_seconds else 0.0
    silent_second_count = sum(1 for item in active_seconds if not item)
    longest_silent_seconds = 0
    current_silent_seconds = 0
    for item in active_seconds:
        if item:
            current_silent_seconds = 0
        else:
            current_silent_seconds += 1
            longest_silent_seconds = max(longest_silent_seconds, current_silent_seconds)
    min_second_db = min(second_rms_db) if second_rms_db else -240.0

    active = [value >= -45.0 for value in segment_rms_db]
    active_ratio = sum(active) / len(active) if active else 0.0
    longest_quiet = 0
    current_quiet = 0
    for item in active:
        if item:
            current_quiet = 0
        else:
            current_quiet += 1
            longest_quiet = max(longest_quiet, current_quiet)

    half = len(audio) // 2
    first_half_db = db(rms(audio[:half])) if half else -240.0
    second_half_db = db(rms(audio[half:])) if half else -240.0
    half_drop_db = max(0.0, first_half_db - second_half_db)
    explosion_db = (max(segment_rms_db) - float(np.median(segment_rms_db))) if segment_rms_db else 0.0

    bass_ratio, high_ratio, tone_ratio = spectral_ratios(audio)
    band_metrics = spectral_band_ratios(audio)
    movement_metrics = audio_movement_metrics(audio)
    estimated_bpm, bpm_method = estimate_bpm(audio, target_bpm)
    bpm_diff: Optional[float] = None
    bpm_status = "UNAVAILABLE"
    bpm_problem_reason = False
    hard_reasons_before_bpm = 0
    if estimated_bpm is not None:
        bpm_diff = abs(float(estimated_bpm) - float(target_bpm))
        if bpm_diff <= bpm_tolerance:
            bpm_status = "OK"
        elif bpm_diff <= bpm_tolerance * 2.0:
            bpm_status = "WARN"
        else:
            bpm_status = "PROBLEM"

    # Die harten Gruende entfernen einen Kandidaten. Warnungen bleiben im
    # Report, damit musikalisch brauchbare MusicGen-Clips nicht nur wegen der
    # ueblichen Lautheitsnormalisierung verworfen werden.
    reasons: List[str] = []
    warnings_list: List[str] = []
    if abs(duration - expected_duration) > 1.0:
        reasons.append("Dauer nicht korrekt")
    if second_check_enabled:
        expected_seconds = max(1, int(math.floor(expected_duration)))
        if len(second_rms_db) < expected_seconds - 1:
            reasons.append("zu wenige vollstaendige Sekunden messbar")
        if silent_second_count > 0:
            reasons.append("mindestens eine Sekunde ohne hoerbaren Ton")
        if active_second_ratio < 0.999:
            reasons.append("nicht jede Sekunde ist aktiv")
    if db(total_rms) < -42.0:
        reasons.append("insgesamt zu leise")
    elif db(total_rms) < -34.0:
        warnings_list.append("insgesamt eher leise")
    if active_ratio < 0.65:
        reasons.append("nicht durchgehend hoerbar")
    elif active_ratio < 0.85:
        warnings_list.append("einige leise Abschnitte")
    if longest_quiet * 3.0 > 9.0:
        reasons.append("stiller/leiser Abschnitt zu lang")
    elif longest_quiet * 3.0 >= 6.0:
        warnings_list.append("laengerer leiser Abschnitt")
    if half_drop_db > 18.0:
        reasons.append("starker Lautheitsabfall")
    elif half_drop_db > 12.0:
        warnings_list.append("Lautheit faellt ab")
    if clipping_ratio > 0.03:
        reasons.append("echte Uebersteuerung moeglich")
    elif peak >= 0.98:
        warnings_list.append("Peak nahe 0 dB")
    if explosion_db > 18.0:
        reasons.append("Energie-Explosion")
    elif explosion_db > 12.0:
        warnings_list.append("Energiesprung auffaellig")
    if bass_ratio > 0.72:
        reasons.append("Bass zu dominant")
    elif bass_ratio > 0.56:
        warnings_list.append("Bass auffaellig")
    if high_ratio > 0.16:
        reasons.append("scharfe Hoehen / Shaker zu stark")
    elif high_ratio > 0.10:
        warnings_list.append("Shaker/Hoehen auffaellig")
    if tone_ratio > 0.30:
        reasons.append("Signalton oder monotone Schwingung")
    movement_score = float(movement_metrics.get("musical_movement_score", 0.0))
    movement_windows = int(movement_metrics.get("movement_window_count", 0))
    if duration >= 20.0 and movement_windows >= 4:
        if movement_score < 0.18:
            warnings_list.append("30s-Clip wirkt sehr statisch")
        elif movement_score < 0.40:
            warnings_list.append("wenig musikalische Bewegung im 30s-Clip")
    hard_reasons_before_bpm = len(reasons)
    if bpm_check_enabled:
        if estimated_bpm is None:
            warnings_list.append("BPM nicht sicher messbar")
        elif bpm_diff is not None and bpm_diff > max(bpm_tolerance * 2.0, bpm_tolerance + 4.0):
            reasons.append("BPM deutlich ausserhalb Zielbereich")
            bpm_problem_reason = True
        elif bpm_diff is not None and bpm_diff > bpm_tolerance:
            warnings_list.append("BPM ausserhalb Toleranz")

    bpm_fallback_ok = bpm_problem_reason and hard_reasons_before_bpm == 0

    return {
        "status": "OK" if not reasons else "PROBLEM",
        "gruende": "; ".join(reasons),
        "warnungen": "; ".join(warnings_list),
        "target_bpm": round(target_bpm, 3),
        "estimated_bpm": round(estimated_bpm, 3) if estimated_bpm is not None else "",
        "bpm_diff": round(bpm_diff, 3) if bpm_diff is not None else "",
        "bpm_status": bpm_status,
        "bpm_method": bpm_method,
        "bpm_tolerance": round(bpm_tolerance, 3),
        "bpm_filter_enabled": bool(bpm_check_enabled),
        "bpm_fallback_ok": bpm_fallback_ok,
        "duration_sec": round(duration, 3),
        "rms_db": round(db(total_rms), 3),
        "peak_db": round(db(peak), 3),
        "clipping_ratio": round(clipping_ratio, 5),
        "active_segment_ratio": round(active_ratio, 3),
        "quiet_run_sec": round(longest_quiet * 3.0, 3),
        "second_check_enabled": bool(second_check_enabled),
        "min_second_rms_db_threshold": round(min_second_rms_db, 3),
        "active_second_ratio": round(active_second_ratio, 3),
        "silent_second_count": int(silent_second_count),
        "longest_silent_sec": round(float(longest_silent_seconds), 3),
        "min_second_rms_db": round(min_second_db, 3),
        "half_drop_db": round(half_drop_db, 3),
        "explosion_db": round(explosion_db, 3),
        "bass_ratio": round(bass_ratio, 4),
        "snare_ratio": round(float(band_metrics.get("snare_ratio", 0.0)), 4),
        "high_ratio": round(high_ratio, 4),
        "tone_ratio": round(tone_ratio, 4),
        **movement_metrics,
    }


def spectral_ratios(audio: np.ndarray) -> tuple[float, float, float]:
    if len(audio) < SAMPLE_RATE:
        return 0.0, 0.0, 0.0
    # Bei langen MusicGen-Fortsetzungen darf nicht nur der Anfang entscheiden.
    # Vier gleichmaessig verteilte 5s-Fenster erfassen auch spaetere Bass-,
    # Shaker- oder Signalton-Probleme, ohne eine riesige FFT zu erzeugen.
    maximum_frames = SAMPLE_RATE * 20
    if len(audio) <= maximum_frames:
        samples = [audio]
    else:
        window_frames = SAMPLE_RATE * 5
        starts = np.linspace(
            0,
            max(0, len(audio) - window_frames),
            4,
        ).astype(int)
        samples = [audio[start : start + window_frames] for start in starts]
    segment_length = min(len(sample) for sample in samples)
    window = np.hanning(segment_length).astype(np.float32)
    spectra = [
        np.abs(np.fft.rfft(sample[:segment_length] * window))
        for sample in samples
    ]
    spectrum = np.mean(spectra, axis=0)
    freqs = np.fft.rfftfreq(segment_length, d=1.0 / SAMPLE_RATE)
    total = float(np.sum(spectrum[(freqs >= 20) & (freqs <= 15000)])) + EPS
    bass = float(np.sum(spectrum[(freqs >= 20) & (freqs <= 180)])) / total
    high = float(np.sum(spectrum[(freqs >= 8000) & (freqs <= 15000)])) / total
    tone = float(np.max(spectrum)) / total
    return bass, high, tone


def spectral_band_ratios(audio: np.ndarray) -> Dict[str, float]:
    """Misst Bass, Mitten, Snare-/Shaker-Bereiche und tonale Naehe.

    Die Werte sind bewusst einfache, reproduzierbare Signalmerkmale. Sie
    ersetzen keine harmonische Analyse, liefern aber eine stabile lokale
    Naeherung fuer die Uebergangsauswahl ohne zusaetzliche Modell-Downloads.
    """

    if len(audio) < SAMPLE_RATE // 4:
        return {
            "bass_ratio": 0.0,
            "low_mid_ratio": 0.0,
            "mid_ratio": 0.0,
            "snare_ratio": 0.0,
            "high_ratio": 0.0,
            "tone_ratio": 0.0,
            "spectral_centroid_hz": 0.0,
        }
    sample = audio[: min(len(audio), SAMPLE_RATE * 6)]
    window = np.hanning(len(sample)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(sample * window))
    freqs = np.fft.rfftfreq(len(sample), d=1.0 / SAMPLE_RATE)
    relevant = (freqs >= 20) & (freqs <= 14000)
    total = float(np.sum(spectrum[relevant])) + EPS

    def band(lo: float, hi: float) -> float:
        return float(np.sum(spectrum[(freqs >= lo) & (freqs < hi)])) / total

    centroid = float(np.sum(freqs[relevant] * spectrum[relevant]) / total)
    return {
        "bass_ratio": band(20, 180),
        "low_mid_ratio": band(180, 700),
        "mid_ratio": band(700, 1800),
        "snare_ratio": band(1800, 6000),
        "high_ratio": band(8000, 14000),
        "tone_ratio": float(np.max(spectrum[relevant])) / total if np.any(relevant) else 0.0,
        "spectral_centroid_hz": centroid,
    }


def spectral_fingerprint(audio: np.ndarray) -> List[float]:
    """Erzeugt einen kleinen spektralen Fingerabdruck fuer Harmonie-/Timbre-Naehe."""

    if len(audio) < SAMPLE_RATE // 4:
        return [0.0] * 9
    sample = audio[: min(len(audio), SAMPLE_RATE * 6)]
    window = np.hanning(len(sample)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(sample * window))
    freqs = np.fft.rfftfreq(len(sample), d=1.0 / SAMPLE_RATE)
    bands = (
        (20, 60),
        (60, 120),
        (120, 250),
        (250, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
        (8000, 14000),
    )
    values = []
    for lo, hi in bands:
        values.append(float(np.sum(spectrum[(freqs >= lo) & (freqs < hi)])))
    vector = np.asarray(values, dtype=np.float32)
    total = float(np.sum(vector)) + EPS
    return [round(float(value / total), 8) for value in vector]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=np.float32)
    b = np.asarray(list(right), dtype=np.float32)
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    value = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, value)) if math.isfinite(value) else 0.0


def audio_movement_metrics(audio: np.ndarray, window_seconds: float = 3.0) -> Dict[str, Any]:
    """Misst kleine Binnenbewegungen, ohne ruhige Lofi-Clips zu bestrafen."""

    window_frames = int(round(max(1.0, window_seconds) * SAMPLE_RATE))
    if len(audio) < window_frames * 2:
        return {
            "movement_window_sec": round(float(window_seconds), 3),
            "movement_window_count": 0,
            "spectral_movement": 0.0,
            "rms_movement_db": 0.0,
            "centroid_movement_hz": 0.0,
            "musical_movement_score": 0.0,
        }

    fingerprints: List[List[float]] = []
    rms_values: List[float] = []
    centroids: List[float] = []
    for start in range(0, len(audio), window_frames):
        segment = audio[start : start + window_frames]
        if len(segment) < window_frames * 0.75:
            continue
        fingerprints.append(spectral_fingerprint(segment))
        rms_values.append(db(rms(segment)))
        centroids.append(float(spectral_band_ratios(segment).get("spectral_centroid_hz", 0.0)))

    if len(fingerprints) < 2:
        return {
            "movement_window_sec": round(float(window_seconds), 3),
            "movement_window_count": len(fingerprints),
            "spectral_movement": 0.0,
            "rms_movement_db": 0.0,
            "centroid_movement_hz": 0.0,
            "musical_movement_score": 0.0,
        }

    spectral_steps = [
        max(0.0, 1.0 - cosine_similarity(left, right))
        for left, right in zip(fingerprints, fingerprints[1:])
    ]
    spectral_movement = float(np.mean(spectral_steps)) if spectral_steps else 0.0
    rms_movement_db = float(np.std(rms_values)) if rms_values else 0.0
    centroid_movement_hz = float(np.std(centroids)) if centroids else 0.0

    score = (
        spectral_movement * 14.0
        + min(0.30, rms_movement_db / 10.0)
        + min(0.30, centroid_movement_hz / 2500.0)
    )
    score = max(0.0, min(1.0, score))
    return {
        "movement_window_sec": round(float(window_seconds), 3),
        "movement_window_count": len(fingerprints),
        "spectral_movement": round(spectral_movement, 5),
        "rms_movement_db": round(rms_movement_db, 3),
        "centroid_movement_hz": round(centroid_movement_hz, 3),
        "musical_movement_score": round(score, 4),
    }


def energieverlauf(audio: np.ndarray, teile: int = 24) -> np.ndarray:
    """Verdichtet einen Uebergangsbereich zu einem normierten Energieverlauf."""

    if len(audio) == 0:
        return np.zeros(teile, dtype=np.float32)
    grenzen = np.linspace(0, len(audio), teile + 1, dtype=int)
    werte = np.asarray(
        [rms(audio[grenzen[index] : grenzen[index + 1]]) for index in range(teile)],
        dtype=np.float32,
    )
    mittel = float(np.mean(werte))
    standard = float(np.std(werte))
    if standard <= EPS:
        return werte - mittel
    return (werte - mittel) / standard


def korrelation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    if float(np.std(left)) <= EPS or float(np.std(right)) <= EPS:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def click_risiko(audio: np.ndarray) -> float:
    """Schaetzt Impuls-/Klickrisiko an einem kurzen Cliprand."""

    if len(audio) < 8:
        return 0.0
    diff = np.abs(np.diff(audio.astype(np.float32, copy=False)))
    if len(diff) == 0:
        return 0.0
    basis = float(np.percentile(diff, 95)) + EPS
    return round(float(np.max(diff) / basis), 4)


def aktive_randquote(audio: np.ndarray, min_db: float = -52.0) -> float:
    """Anteil aktiver 250-ms-Fenster an einem Randbereich."""

    frames = max(1, int(round(0.25 * SAMPLE_RATE)))
    werte: List[bool] = []
    for start in range(0, len(audio), frames):
        segment = audio[start : start + frames]
        if len(segment) >= frames * 0.5:
            werte.append(db(rms(segment)) >= min_db)
    return round(sum(werte) / len(werte), 4) if werte else 0.0


_KANTEN_PROFIL_CACHE: Dict[str, Dict[str, Any]] = {}


def public_profile_values(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Entfernt interne Vektoren, bevor Profile in CSV/JSON-Reports landen."""

    return {key: value for key, value in profile.items() if not key.startswith("_")}


def optional_float(value: Any) -> Optional[float]:
    """Konvertiert Report-Werte vorsichtig in float."""

    if value in ("", None):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def project_path(value: Any) -> Path:
    """Erzeugt aus Report-Pfaden einen sicher lesbaren Projektpfad."""

    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def summarize_optional(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    numbers = [item for item in (optional_float(value) for value in values) if item is not None]
    if not numbers:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(min(numbers), 3),
        "mean": round(sum(numbers) / len(numbers), 3),
        "max": round(max(numbers), 3),
    }


def referenz_genre_key(value: Any) -> str:
    """Vereinheitlicht Genre-Namen fuer den MP3-Referenzvergleich."""

    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if "jazz" in text:
        return "jazz_lofi"
    if "chillhop" in text or text == "chill lofi":
        return "chillhop_lofi"
    if "dream" in text:
        return "dreamy_lofi"
    if "study" in text or "focus" in text:
        return "study_lofi"
    if "guitar" in text or "gitarre" in text:
        return "guitar_lofi"
    return "lofi_gesamt"


def read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    """Liest Manifestzeilen aus dem Trainingsdataset."""

    rows: List[Dict[str, Any]] = []
    if not path.is_file():
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


def resolve_dataset_audio_path(value: Any) -> Path:
    """Loest Audio-Pfade aus Dataset-Manifesten auf."""

    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def median_dict(rows: List[Dict[str, Any]], fields: Iterable[str]) -> Dict[str, Any]:
    """Berechnet Medianwerte fuer ein Referenzprofil."""

    result: Dict[str, Any] = {"anzahl": len(rows)}
    for field in fields:
        values = [optional_float(row.get(field)) for row in rows]
        numbers = [value for value in values if value is not None]
        result[field] = round(float(np.median(numbers)), 5) if numbers else ""
    return result


def collect_reference_rows(args: argparse.Namespace, run_dir: Path) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Baut technische MP3-Referenzprofile pro Genre.

    Die Profile stammen aus den geprueften Trainingsdaten. Dadurch vergleichen
    wir neue MusicGen-Kandidaten mit genau dem Material, das als guter Standard
    fuer LoRA dient: echte MP3-Clips, 30 Sekunden, genrebalanciert.
    """

    report: Dict[str, Any] = {
        "enabled": bool(getattr(args, "mp3_referenz_pruefung_aktiv", True)),
        "available": False,
        "dataset_root": rel(Path(getattr(args, "referenz_dataset_root", "") or ".")),
        "referenzen_pro_genre": int(getattr(args, "referenzen_pro_genre", 20) or 20),
        "score_limit": float(getattr(args, "referenz_score_limit", 18.0) or 18.0),
    }
    if not report["enabled"]:
        report["hinweis"] = "MP3-Referenzpruefung deaktiviert."
        return {}, report

    dataset_root = Path(str(args.referenz_dataset_root)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    dataset_root = dataset_root.resolve()
    report["dataset_root"] = rel(dataset_root)
    if not dataset_root.exists():
        report["error"] = f"Referenzdataset fehlt: {dataset_root}"
        return {}, report

    manifest_rows: List[Dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        manifest_rows.extend(read_jsonl_rows(dataset_root / split / "data.jsonl"))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in manifest_rows:
        path = resolve_dataset_audio_path(row.get("path"))
        if not path.is_file():
            continue
        genre = referenz_genre_key(row.get("lora_genre") or row.get("primary_genre") or row.get("genre"))
        grouped.setdefault(genre, []).append(row)

    reference_rows: List[Dict[str, Any]] = []
    selected_per_genre: Dict[str, int] = {}
    limit = max(1, int(args.referenzen_pro_genre))
    for genre, rows in sorted(grouped.items()):
        sorted_rows = sorted(
            rows,
            key=lambda item: (
                -float(item.get("quality_score") or 0.0),
                str(item.get("source_file") or item.get("source_audio_path") or ""),
                str(item.get("path") or ""),
            ),
        )
        selected: List[Dict[str, Any]] = []
        used_sources: set[str] = set()
        for row in sorted_rows:
            source = str(row.get("source_split_key") or row.get("source_audio_path") or row.get("source_file") or "")
            if source in used_sources and len(selected) < max(1, limit // 2):
                continue
            selected.append(row)
            used_sources.add(source)
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            for row in sorted_rows:
                if row in selected:
                    continue
                selected.append(row)
                if len(selected) >= limit:
                    break
        selected_per_genre[genre] = len(selected)
        for index, row in enumerate(selected, start=1):
            path = resolve_dataset_audio_path(row.get("path"))
            try:
                metrics = audio_metrics(
                    path,
                    float(row.get("duration") or args.abschnitt_sekunden),
                    args.ziel_bpm,
                    args.bpm_toleranz,
                    args.bpm_pruefung_aktiv,
                    True,
                    args.min_sekunden_rms_db,
                )
                status = "OK"
                error = ""
            except Exception as exc:
                metrics = {}
                status = "FEHLER"
                error = f"{type(exc).__name__}: {exc}"
            metric_status = metrics.get("status", "")
            reference_rows.append(
                {
                    **metrics,
                    "referenz": f"{genre}_{index:03d}",
                    "genre": genre,
                    "path": rel(path),
                    "source_file": row.get("source_file", ""),
                    "quality_score": row.get("quality_score", ""),
                    "status": status,
                    "metric_status": metric_status,
                    "fehler": error,
                }
            )

    ok_rows = [row for row in reference_rows if row.get("status") == "OK"]
    metric_fields = (
        "rms_db",
        "peak_db",
        "active_second_ratio",
        "silent_second_count",
        "half_drop_db",
        "explosion_db",
        "bass_ratio",
        "high_ratio",
        "tone_ratio",
        "musical_movement_score",
        "spectral_movement",
        "rms_movement_db",
        "centroid_movement_hz",
        "estimated_bpm",
        "bpm_diff",
        "min_second_rms_db",
    )
    profiles: Dict[str, Dict[str, Any]] = {}
    for genre in sorted({str(row["genre"]) for row in ok_rows}):
        genre_rows = [row for row in ok_rows if row.get("genre") == genre]
        profiles[genre] = median_dict(genre_rows, metric_fields)
    if ok_rows:
        profiles["lofi_gesamt"] = median_dict(ok_rows, metric_fields)

    write_csv(run_dir / "mp3_referenz_profile.csv", ok_rows)
    write_json(run_dir / "mp3_referenz_profile.json", profiles)
    report.update(
        {
            "available": bool(profiles),
            "profile_path": rel(run_dir / "mp3_referenz_profile.json"),
            "reference_rows_path": rel(run_dir / "mp3_referenz_profile.csv"),
            "reference_count": len(ok_rows),
            "selected_per_genre": selected_per_genre,
            "profiles": profiles,
        }
    )
    if not profiles:
        report["error"] = "Keine analysierbaren MP3-Referenzen gefunden."
    return profiles, report


def score_mp3_referenz(
    metrics: Dict[str, Any],
    genre: str,
    profiles: Dict[str, Dict[str, Any]],
    score_limit: float,
) -> Dict[str, Any]:
    """Bewertet, wie nah ein Kandidat technisch am MP3-Referenzprofil liegt."""

    if not profiles:
        return {
            "referenz_status": "DISABLED",
            "referenz_score": 0.0,
            "referenz_gruende": "",
        }

    profile_key = referenz_genre_key(genre)
    profile = profiles.get(profile_key) or profiles.get("lofi_gesamt") or {}
    if not profile:
        return {
            "referenz_status": "UNAVAILABLE",
            "referenz_score": 0.0,
            "referenz_gruende": "Keine passende MP3-Referenz gefunden",
        }

    def value(name: str, default: float = 0.0) -> float:
        current = optional_float(metrics.get(name))
        return default if current is None else current

    def ref(name: str, default: float = 0.0) -> float:
        current = optional_float(profile.get(name))
        return default if current is None else current

    score = 0.0
    reasons: List[str] = []
    hard_problem = False

    silent_seconds = value("silent_second_count")
    active_seconds = value("active_second_ratio")
    if silent_seconds > 0 or active_seconds < 0.999:
        reasons.append("nicht jede Sekunde hoerbar")
        score += 22.0 + silent_seconds * 4.0
        hard_problem = True

    rms_gap = abs(value("rms_db") - ref("rms_db", value("rms_db")))
    if rms_gap > 5.0:
        reasons.append("Lautheit weicht von MP3-Referenz ab")
        score += (rms_gap - 5.0) * 1.6

    if value("peak_db") > max(-0.4, ref("peak_db", -1.0) + 2.0):
        reasons.append("Peak/Headroom schlechter als Referenz")
        score += 8.0
        if value("peak_db") > -0.15:
            hard_problem = True

    half_limit = max(5.0, ref("half_drop_db", 0.0) + 4.0)
    if value("half_drop_db") > half_limit:
        reasons.append("zweite Haelfte schwach gegenueber Referenz")
        score += (value("half_drop_db") - half_limit) * 1.8

    explosion_limit = max(8.0, ref("explosion_db", 2.0) + 4.0)
    if value("explosion_db") > explosion_limit:
        reasons.append("Energie-Spitze staerker als Referenz")
        score += (value("explosion_db") - explosion_limit) * 1.4

    bass_ref = ref("bass_ratio", 0.30)
    bass_limit = max(0.26, min(0.36, bass_ref + 0.05))
    bass_problem_limit = max(0.40, min(0.48, bass_ref + 0.18))
    if value("bass_ratio") > bass_limit:
        reasons.append("Bass staerker als MP3-Referenz")
        score += (value("bass_ratio") - bass_limit) * 180.0
        if value("bass_ratio") > bass_problem_limit:
            hard_problem = True

    high_limit = max(0.10, ref("high_ratio", 0.04) + 0.04)
    if value("high_ratio") > high_limit:
        reasons.append("Hoehen/Shaker staerker als MP3-Referenz")
        score += (value("high_ratio") - high_limit) * 70.0

    tone_limit = max(0.20, ref("tone_ratio", 0.002) + 0.08)
    if value("tone_ratio") > tone_limit:
        reasons.append("Signalton-/Monotonie-Risiko gegenueber Referenz")
        score += (value("tone_ratio") - tone_limit) * 45.0
        hard_problem = True

    movement_ref = ref("musical_movement_score", 0.0)
    if movement_ref > 0.0:
        movement_gap = movement_ref - value("musical_movement_score", movement_ref)
        if movement_gap > 0.28:
            reasons.append("weniger musikalische Bewegung als MP3-Referenz")
            score += (movement_gap - 0.28) * 10.0

    spectral_ref = ref("spectral_movement", 0.0)
    if spectral_ref > 0.0:
        spectral_gap = spectral_ref - value("spectral_movement", spectral_ref)
        if spectral_gap > 0.035:
            reasons.append("Klangfarbe entwickelt sich weniger als Referenz")
            score += (spectral_gap - 0.035) * 35.0

    bpm = value("estimated_bpm", 0.0)
    ref_bpm = ref("estimated_bpm", 0.0)
    if bpm > 0.0 and ref_bpm > 0.0:
        bpm_gap = abs(bpm - ref_bpm)
        if bpm_gap > 8.0:
            reasons.append("BPM weicht von MP3-Referenz ab")
            score += min(18.0, (bpm_gap - 8.0) * 0.7)

    status = "OK"
    if hard_problem or score > float(score_limit):
        status = "PROBLEM"
    elif reasons:
        status = "PRUEFEN"
    return {
        "referenz_status": status,
        "referenz_score": round(score, 4),
        "referenz_score_limit": round(float(score_limit), 4),
        "referenz_genre": profile_key,
        "referenz_anzahl": profile.get("anzahl", ""),
        "referenz_gruende": "; ".join(reasons),
        "referenz_rms_db": profile.get("rms_db", ""),
        "referenz_peak_db": profile.get("peak_db", ""),
        "referenz_bass_ratio": profile.get("bass_ratio", ""),
        "referenz_high_ratio": profile.get("high_ratio", ""),
        "referenz_tone_ratio": profile.get("tone_ratio", ""),
        "referenz_musical_movement_score": profile.get("musical_movement_score", ""),
        "referenz_spectral_movement": profile.get("spectral_movement", ""),
        "referenz_bpm": profile.get("estimated_bpm", ""),
    }


def order_accepted_clips(accepted: List[Dict[str, Any]], target_bpm: float) -> List[Dict[str, Any]]:
    """Behaelt die geplante Abschnittsreihenfolge fuer stabile Uebergaenge."""

    return sorted(accepted, key=lambda row: int(optional_float(row.get("abschnitt")) or 0))


def score_kandidat(metrics: Dict[str, Any]) -> float:
    """Bewertet einen Kandidaten; niedrigerer Score = bessere Qualitaet.

    Die Gewichtung basiert auf den haeufigsten Kritikpunkten aus den
    menschlichen Bewertungen (13 Durchgaenge): Bass-Dominanz und Shaker/
    Hoehen-Intensitaet sind die meistgenannten Probleme und werden
    entsprechend stark bewertet.
    """
    bass = optional_float(metrics.get("bass_ratio")) or 0.5
    snare = optional_float(metrics.get("snare_ratio")) or 0.0
    high = optional_float(metrics.get("high_ratio")) or 0.0
    tone = optional_float(metrics.get("tone_ratio")) or 0.0
    bpm_diff = optional_float(metrics.get("bpm_diff")) or 0.0
    half_drop = optional_float(metrics.get("half_drop_db")) or 0.0
    quiet_run = optional_float(metrics.get("quiet_run_sec")) or 0.0
    active = optional_float(metrics.get("active_segment_ratio")) or 0.0
    active_seconds = optional_float(metrics.get("active_second_ratio")) or 0.0
    silent_seconds = optional_float(metrics.get("silent_second_count")) or 0.0
    movement = optional_float(metrics.get("musical_movement_score")) or 0.0
    spectral_movement = optional_float(metrics.get("spectral_movement")) or 0.0
    return (
        bass * 4.2        # Bass-Dominanz: groesster Kritikpunkt
        + max(0.0, bass - 0.36) * 12.0
        + snare * 1.8     # Snare-/Shaker-Bereich: hart an Uebergaengen
        + high * 4.0      # Shaker/Hoehen: zweithaeufigster Kritikpunkt
        + max(0.0, high - 0.10) * 16.0
        + tone * 1.5      # Signalton / Monotonie
        + max(0.0, 0.35 - movement) * 6.0  # zu statische 30s-Clips vermeiden
        + max(0.0, 0.020 - spectral_movement) * 30.0
        + half_drop * 0.08  # Ton-Einbruch in zweiter Haelfte
        + quiet_run * 0.35  # lange leise Stellen zerstoeren spaetere Uebergaenge
        + bpm_diff * 0.04   # BPM-Abweichung
        + max(0.0, 0.95 - active) * 4.0  # Kandidaten sollen bis zum Ende tragen
        + max(0.0, 1.0 - active_seconds) * 8.0  # jede einzelne Sekunde soll hoerbar sein
        + silent_seconds * 2.0
        - active * 0.3    # Bonus fuer durchgehend aktive Audio
    )


def score_block_konsistenz(
    section: GeplanterAbschnitt,
    candidate_prompt: str,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Bewertet, ob ein Kandidat musikalisch zum aktuellen Block passt."""

    prompt_lower = candidate_prompt.lower()
    basis_lower = section.prompt.lower()
    warnings_list: List[str] = []
    score = 0.0

    if "lofi" not in prompt_lower.replace("-", ""):
        warnings_list.append("Prompt enthaelt keinen Lofi-Rahmen")
        score += 8.0
    if "calm controlled" not in prompt_lower:
        warnings_list.append("ruhig-kontrollierter Charakter fehlt im Prompt")
        score += 2.0
    if "aggressive" in prompt_lower and "no aggressive" not in prompt_lower:
        warnings_list.append("aggressive Prompt-Richtung")
        score += 5.0

    bpm_diff = optional_float(metrics.get("bpm_diff"))
    if bpm_diff is None:
        warnings_list.append("BPM fuer Block-Konsistenz nicht sicher messbar")
        score += 1.5
    elif bpm_diff > args.bpm_toleranz:
        warnings_list.append("BPM weicht vom Block-Ziel ab")
        score += min(8.0, (bpm_diff - args.bpm_toleranz) * 0.8)

    bass = optional_float(metrics.get("bass_ratio")) or 0.0
    snare = optional_float(metrics.get("snare_ratio")) or 0.0
    high = optional_float(metrics.get("high_ratio")) or 0.0
    tone = optional_float(metrics.get("tone_ratio")) or 0.0
    movement = optional_float(metrics.get("musical_movement_score")) or 0.0
    spectral_movement = optional_float(metrics.get("spectral_movement")) or 0.0
    if bass > 0.42:
        warnings_list.append("Bass fuer ruhigen Lofi-Block zu dominant")
        score += (bass - 0.42) * 28.0
    if snare > 0.34:
        warnings_list.append("Snare-/Praesenzbereich fuer ruhigen Lofi-Block zu stark")
        score += (snare - 0.34) * 9.0
    if high > 0.12:
        warnings_list.append("Hoehen/Shaker fuer ruhigen Lofi-Block zu stark")
        score += (high - 0.12) * 24.0
    if tone > 0.22:
        warnings_list.append("monotone oder signalartige Komponente")
        score += (tone - 0.22) * 10.0

    genre_lower = str(args.genre or "").lower()
    if "chillhop" in genre_lower:
        if movement < 0.35:
            warnings_list.append("Chillhop-Clip wirkt zu statisch gegenueber den Referenzen")
            score += (0.35 - movement) * 18.0
        if movement < 0.55:
            warnings_list.append("Chillhop braucht mehr natuerliche Sample-Bewegung")
            score += (0.55 - movement) * 6.0
        if spectral_movement < 0.025:
            warnings_list.append("Chillhop-Klangfarbe entwickelt sich zu wenig")
            score += (0.025 - spectral_movement) * 60.0
    elif movement < 0.18:
        warnings_list.append("30s-Clip hat sehr wenig musikalische Binnenbewegung")
        score += (0.18 - movement) * 8.0
    elif movement < 0.30:
        warnings_list.append("30s-Clip koennte musikalisch mehr passieren lassen")
        score += (0.30 - movement) * 4.0

    active_seconds = optional_float(metrics.get("active_second_ratio")) or 0.0
    if active_seconds < 1.0:
        warnings_list.append("nicht jede Sekunde im Block aktiv")
        score += (1.0 - active_seconds) * 12.0

    same_prompt_family = candidate_prompt.startswith(section.prompt)
    if not same_prompt_family and basis_lower[:80] not in prompt_lower:
        warnings_list.append("Prompt weicht vom Block-Kern ab")
        score += 2.5

    return {
        "block_id": section.block,
        "block_genre": lofi_genre_label(args.genre),
        "block_prompt_basis": section.prompt,
        "block_bpm": round(float(args.ziel_bpm), 3),
        "block_mood": args.stimmung,
        "block_instrumente": args.instrumente,
        "block_consistency_score": round(score, 4),
        "block_consistency_status": "OK" if not warnings_list else "PRUEFEN",
        "block_consistency_warning": "; ".join(warnings_list),
        "lofi_style_score": round(score, 4),
    }


def kanten_profil(path: Path, sekunden: float = 4.0) -> Dict[str, Any]:
    """Misst Anfang und Ende eines Clips fuer den Uebergangsvergleich."""

    resolved = path.resolve()
    try:
        stat_key = f"{resolved}:{resolved.stat().st_mtime_ns}:{sekunden:.3f}"
    except OSError:
        stat_key = f"{resolved}:missing:{sekunden:.3f}"
    cached = _KANTEN_PROFIL_CACHE.get(stat_key)
    if cached is not None:
        return cached

    audio = decode_audio(path)
    frames = int(round(max(1.0, sekunden) * SAMPLE_RATE))
    start_audio = audio[: min(len(audio), frames)]
    end_audio = audio[max(0, len(audio) - frames) :]
    short_frames = int(round(1.0 * SAMPLE_RATE))
    very_start = audio[: min(len(audio), short_frames)]
    very_end = audio[max(0, len(audio) - short_frames) :]
    click_frames = int(round(0.12 * SAMPLE_RATE))
    start_click_audio = audio[: min(len(audio), click_frames)]
    end_click_audio = audio[max(0, len(audio) - click_frames) :]
    start_bands = spectral_band_ratios(start_audio)
    end_bands = spectral_band_ratios(end_audio)
    result = {
        "start_rms_db": round(db(rms(start_audio)), 3),
        "end_rms_db": round(db(rms(end_audio)), 3),
        "start_peak_db": round(db(float(np.max(np.abs(start_audio))) if len(start_audio) else 0.0), 3),
        "end_peak_db": round(db(float(np.max(np.abs(end_audio))) if len(end_audio) else 0.0), 3),
        "very_start_rms_db": round(db(rms(very_start)), 3),
        "very_end_rms_db": round(db(rms(very_end)), 3),
        "start_bass_ratio": round(float(start_bands["bass_ratio"]), 4),
        "end_bass_ratio": round(float(end_bands["bass_ratio"]), 4),
        "start_snare_ratio": round(float(start_bands["snare_ratio"]), 4),
        "end_snare_ratio": round(float(end_bands["snare_ratio"]), 4),
        "start_high_ratio": round(float(start_bands["high_ratio"]), 4),
        "end_high_ratio": round(float(end_bands["high_ratio"]), 4),
        "start_spectral_centroid_hz": round(float(start_bands["spectral_centroid_hz"]), 3),
        "end_spectral_centroid_hz": round(float(end_bands["spectral_centroid_hz"]), 3),
        "start_active_ratio": aktive_randquote(start_audio),
        "end_active_ratio": aktive_randquote(end_audio),
        "start_click_risk": click_risiko(start_click_audio),
        "end_click_risk": click_risiko(end_click_audio),
        "first_sample": round(float(audio[0]), 6) if len(audio) else 0.0,
        "last_sample": round(float(audio[-1]), 6) if len(audio) else 0.0,
        "_start_fingerprint": spectral_fingerprint(start_audio),
        "_end_fingerprint": spectral_fingerprint(end_audio),
        "_start_energy": energieverlauf(start_audio).tolist(),
        "_end_energy": energieverlauf(end_audio).tolist(),
    }
    _KANTEN_PROFIL_CACHE[stat_key] = result
    return result


def score_uebergang(
    vorheriger_clip: Optional[Dict[str, Any]],
    kandidat_path: Path,
    kandidat_metrics: Dict[str, Any],
    bpm_toleranz: float,
) -> Dict[str, Any]:
    """Bewertet, wie gut ein neuer Kandidat zum vorherigen Clip passt."""

    kandidat_profil = kanten_profil(kandidat_path)
    if vorheriger_clip is None:
        return {
            "transition_score": 0.0,
            "transition_status": "START",
            "transition_gruende": "",
            "transition_bpm_diff": "",
            "transition_bpm_status": "START",
            "previous_bpm": "",
            "candidate_bpm": kandidat_metrics.get("estimated_bpm", ""),
            "transition_bpm_tolerance": round(float(bpm_toleranz), 3),
            "transition_rms_jump_db": "",
            "transition_snare_jump": "",
            "transition_bass_jump": "",
            "transition_high_jump": "",
            "transition_harmonic_similarity": "",
            "transition_harmonic_distance": "",
            "transition_rhythm_correlation": "",
            "transition_click_risk": "",
            "transition_harte_gruende": "",
            **public_profile_values(kandidat_profil),
        }

    vorheriger_path = project_path(vorheriger_clip.get("path", ""))
    vorheriges_profil = kanten_profil(vorheriger_path)
    prev_bpm = optional_float(vorheriger_clip.get("estimated_bpm"))
    next_bpm = optional_float(kandidat_metrics.get("estimated_bpm"))
    bpm_known = prev_bpm is not None and next_bpm is not None
    bpm_diff = abs(prev_bpm - next_bpm) if bpm_known else 0.0
    bpm_toleranz = max(0.1, float(bpm_toleranz))
    rms_jump = abs(vorheriges_profil["end_rms_db"] - kandidat_profil["start_rms_db"])
    energy_jump = abs(vorheriges_profil["end_peak_db"] - kandidat_profil["start_peak_db"])
    bass_jump = abs(vorheriges_profil["end_bass_ratio"] - kandidat_profil["start_bass_ratio"])
    snare_jump = abs(vorheriges_profil["end_snare_ratio"] - kandidat_profil["start_snare_ratio"])
    high_jump = abs(vorheriges_profil["end_high_ratio"] - kandidat_profil["start_high_ratio"])
    start_impulse = max(0.0, kandidat_profil["very_start_rms_db"] - kandidat_profil["start_rms_db"])
    start_silence = max(0.0, kandidat_profil["start_rms_db"] - kandidat_profil["very_start_rms_db"])
    previous_end_impulse = max(0.0, vorheriges_profil["very_end_rms_db"] - vorheriges_profil["end_rms_db"])
    previous_end_cut = max(0.0, vorheriges_profil["end_rms_db"] - vorheriges_profil["very_end_rms_db"])
    candidate_end_impulse = max(0.0, kandidat_profil["very_end_rms_db"] - kandidat_profil["end_rms_db"])
    candidate_end_cut = max(0.0, kandidat_profil["end_rms_db"] - kandidat_profil["very_end_rms_db"])
    harmonic_similarity = cosine_similarity(
        vorheriges_profil.get("_end_fingerprint", []),
        kandidat_profil.get("_start_fingerprint", []),
    )
    harmonic_distance = 1.0 - harmonic_similarity
    rhythm_correlation = korrelation(
        np.asarray(vorheriges_profil.get("_end_energy", []), dtype=np.float32),
        np.asarray(kandidat_profil.get("_start_energy", []), dtype=np.float32),
    )
    click_risk = max(
        float(vorheriges_profil.get("end_click_risk") or 0.0),
        float(kandidat_profil.get("start_click_risk") or 0.0),
    )
    boundary_jump = abs(
        float(vorheriges_profil.get("last_sample") or 0.0)
        - float(kandidat_profil.get("first_sample") or 0.0)
    )

    gruende: List[str] = []
    harte_gruende: List[str] = []
    if not bpm_known:
        gruende.append("BPM nicht vergleichbar")
        bpm_status = "UNBEKANNT"
    elif bpm_diff > bpm_toleranz * 1.6:
        reason = "deutlicher BPM-Sprung"
        gruende.append(reason)
        harte_gruende.append(reason)
        bpm_status = "PROBLEM"
    elif bpm_diff > bpm_toleranz:
        gruende.append("BPM leicht unterschiedlich")
        bpm_status = "PRUEFEN"
    else:
        bpm_status = "OK"
    for condition, reason, hard in (
        (rms_jump > 4.5, "Lautheitssprung", rms_jump > 7.5),
        (energy_jump > 6.0, "Energiesprung", energy_jump > 9.0),
        (bass_jump > 0.14, "Basssprung", bass_jump > 0.22),
        (snare_jump > 0.12, "Snare-/Praesenzsprung", snare_jump > 0.18),
        (high_jump > 0.08, "Hoehen-/Shakersprung", high_jump > 0.14),
        (harmonic_distance > 0.35, "harmonische/timbrale Distanz", harmonic_distance > 0.55),
        (rhythm_correlation < 0.05, "Rhythmus-/Energieverlauf passt schlecht", rhythm_correlation < -0.25),
        (click_risk > 10.0, "Klickrisiko am Uebergang", click_risk > 18.0),
        (boundary_jump > 0.55, "Wellenform-Sprung an Clipkante", boundary_jump > 0.80),
        (kandidat_profil["start_rms_db"] < -42.0, "leiser Einstieg", True),
        (kandidat_profil["end_rms_db"] < -35.0 or kandidat_profil["very_end_rms_db"] < -38.0, "Clip-Ende zu leise", True),
        (kandidat_profil["start_rms_db"] > -15.0 or kandidat_profil["start_peak_db"] > -2.5, "zu starker Einstieg/Drop", True),
        (vorheriges_profil["end_rms_db"] > -15.0 or vorheriges_profil["end_peak_db"] > -2.5, "vorheriger Clip endet zu stark", False),
        (vorheriges_profil["end_rms_db"] < -35.0 or vorheriges_profil["very_end_rms_db"] < -38.0, "vorheriger Clip endet zu leise", True),
        (start_impulse > 5.0, "Startimpuls faellt schnell ab", start_impulse > 9.0),
        (start_silence > 5.0, "Start beginnt mit Stille/Anschnitt", start_silence > 8.0),
        (previous_end_impulse > 5.0, "vorheriges Ende hat harten Schlussimpuls", previous_end_impulse > 9.0),
        (previous_end_cut > 5.0, "vorheriges Ende bricht ab", previous_end_cut > 8.0),
        (candidate_end_impulse > 5.0, "Kandidatenende hat harten Schlussimpuls", candidate_end_impulse > 9.0),
        (candidate_end_cut > 5.0, "Kandidatenende bricht ab", candidate_end_cut > 8.0),
        (kandidat_profil["start_active_ratio"] < 0.90, "zu wenig aktive Audio am Anfang", True),
        (vorheriges_profil["end_active_ratio"] < 0.90, "zu wenig aktive Audio am vorherigen Ende", True),
    ):
        if condition:
            gruende.append(reason)
            if hard:
                harte_gruende.append(reason)

    raw_score = (
        bpm_diff * 0.65
        + rms_jump * 0.42
        + energy_jump * 0.25
        + bass_jump * 10.0
        + snare_jump * 9.0
        + high_jump * 11.0
        + harmonic_distance * 7.0
        + max(0.0, 0.25 - rhythm_correlation) * 3.0
        + max(0.0, start_impulse - 3.0) * 0.30
        + max(0.0, start_silence - 3.0) * 0.40
        + max(0.0, previous_end_impulse - 3.0) * 0.25
        + max(0.0, previous_end_cut - 3.0) * 0.40
        + max(0.0, candidate_end_impulse - 3.0) * 0.25
        + max(0.0, candidate_end_cut - 3.0) * 0.45
        + max(0.0, click_risk - 10.0) * 0.25
        + boundary_jump * 4.0
        + (2.0 if not bpm_known else 0.0)
        + max(0.0, bpm_diff - bpm_toleranz) * 1.25
        + max(0.0, bpm_diff - bpm_toleranz * 1.6) * 1.75
        + (2.0 if kandidat_profil["start_rms_db"] < -42.0 else 0.0)
        + (2.5 if kandidat_profil["end_rms_db"] < -35.0 or kandidat_profil["very_end_rms_db"] < -38.0 else 0.0)
        + (1.8 if kandidat_profil["start_rms_db"] > -15.0 or kandidat_profil["start_peak_db"] > -2.5 else 0.0)
        + (1.2 if vorheriges_profil["end_rms_db"] > -15.0 or vorheriges_profil["end_peak_db"] > -2.5 else 0.0)
        + (2.5 if vorheriges_profil["end_rms_db"] < -35.0 or vorheriges_profil["very_end_rms_db"] < -38.0 else 0.0)
    )
    score = raw_score * TRANSITION_SCORE_WEIGHT
    if score > MAX_AKZEPTIERTER_UEBERGANG_SCORE and "Uebergangsscore ueber Annahmelimit" not in harte_gruende:
        harte_gruende.append("Uebergangsscore ueber Annahmelimit")
        gruende.append("Uebergangsscore ueber Annahmelimit")
    status = "PROBLEM" if harte_gruende else "PRUEFEN" if gruende else "OK"
    return {
        "transition_score": round(score, 4),
        "transition_score_raw": round(raw_score, 4),
        "transition_score_weight": TRANSITION_SCORE_WEIGHT,
        "transition_acceptance_limit": MAX_AKZEPTIERTER_UEBERGANG_SCORE,
        "transition_status": status,
        "transition_gruende": "; ".join(gruende),
        "transition_warning": "; ".join(gruende),
        "transition_harte_gruende": "; ".join(dict.fromkeys(harte_gruende)),
        "transition_bpm_diff": round(bpm_diff, 3),
        "transition_bpm_status": bpm_status,
        "previous_bpm": round(prev_bpm, 3) if prev_bpm is not None else "",
        "candidate_bpm": round(next_bpm, 3) if next_bpm is not None else "",
        "transition_bpm_tolerance": round(bpm_toleranz, 3),
        "transition_rms_jump_db": round(rms_jump, 3),
        "transition_bass_jump": round(bass_jump, 4),
        "transition_snare_jump": round(snare_jump, 4),
        "transition_high_jump": round(high_jump, 4),
        "transition_energy_jump_db": round(energy_jump, 3),
        "transition_harmonic_similarity": round(harmonic_similarity, 4),
        "transition_harmonic_distance": round(harmonic_distance, 4),
        "transition_rhythm_correlation": round(rhythm_correlation, 4),
        "transition_click_risk": round(click_risk, 4),
        "transition_boundary_jump": round(boundary_jump, 6),
        "transition_start_impulse_db": round(start_impulse, 3),
        "transition_start_silence_db": round(start_silence, 3),
        "transition_previous_end_impulse_db": round(previous_end_impulse, 3),
        "transition_previous_end_cut_db": round(previous_end_cut, 3),
        "transition_candidate_end_impulse_db": round(candidate_end_impulse, 3),
        "transition_candidate_end_cut_db": round(candidate_end_cut, 3),
        "bpm_diff": round(bpm_diff, 3),
        "loudness_jump_db": round(rms_jump, 3),
        "bass_jump": round(bass_jump, 4),
        "snare_jump": round(snare_jump, 4),
        "high_jump": round(high_jump, 4),
        "harmonic_similarity": round(harmonic_similarity, 4),
        "harmonic_distance": round(harmonic_distance, 4),
        "rhythm_correlation": round(rhythm_correlation, 4),
        "click_risk": round(click_risk, 4),
        "start_silence_db": round(start_silence, 3),
        "previous_end_cut_db": round(previous_end_cut, 3),
        "candidate_end_cut_db": round(candidate_end_cut, 3),
        "energy_jump_db": round(energy_jump, 3),
        "previous_end_rms_db": vorheriges_profil["end_rms_db"],
        "previous_end_peak_db": vorheriges_profil["end_peak_db"],
        "previous_end_bass_ratio": vorheriges_profil["end_bass_ratio"],
        "previous_end_snare_ratio": vorheriges_profil["end_snare_ratio"],
        "previous_end_high_ratio": vorheriges_profil["end_high_ratio"],
        **public_profile_values(kandidat_profil),
    }


def split_prompt_terms(text: str) -> List[str]:
    """Zerlegt frei eingegebene Instrumente/Stimmung in kurze Prompt-Bausteine."""

    items = [item.strip() for item in text.replace(";", ",").split(",")]
    return [item for item in items if item]


GENERATION_PROMPT_BLOCKLIST = (
    "shader",
    "shaker",
    "hi hat",
    "hi-hat",
    "hat",
    "hats",
    "percussion",
    "vinyl",
    "tape",
    "noise",
    "texture",
    "crackle",
    "hiss",
    "rustle",
    "airy",
    "air",
    "dusty",
    "top end",
    "highs",
)


def generation_prompt_allowed(part: str) -> bool:
    lowered = f" {part.lower()} "
    for blocked in GENERATION_PROMPT_BLOCKLIST:
        if re.search(rf"(?<![a-z]){re.escape(blocked)}(?![a-z])", lowered):
            return False
    return True


def clean_generation_prompt(prompt: str) -> str:
    parts = [part.strip() for part in prompt.split(",")]
    cleaned: List[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.lower()
        if not part or key in seen or not generation_prompt_allowed(part):
            continue
        seen.add(key)
        cleaned.append(part)
    return ", ".join(cleaned)


def harmony_prompt_terms(items: List[str], *, guitar_only: bool = False) -> List[str]:
    """Filtert Begleit-/Mix-Begriffe aus der Hauptinstrument-Auswahl."""

    blocked = (
        "bass",
        "low end",
        "sub",
        "drum",
        "snare",
        "kick",
        "hat",
        "shaker",
        "percussion",
        "vinyl",
        "tape",
        "texture",
        "noise",
    )
    result: List[str] = []
    for item in items:
        lowered = item.lower()
        if any(term in lowered for term in blocked):
            continue
        if guitar_only and "guitar" not in lowered and "gitarre" not in lowered:
            continue
        result.append(item)
    return result


def lofi_genre_label(genre: str) -> str:
    """Haelt jede Genre-Eingabe im Lofi-Rahmen."""

    cleaned = (genre or "Lofi").strip()
    lowered = cleaned.lower().replace("-", "")
    if "lofi" in lowered:
        return cleaned
    return f"{cleaned} Lofi"


def genre_prompt_profile(genre: str) -> Dict[str, List[str]]:
    """Gibt pro Zielgenre engere Prompt-Bausteine vor.

    Die Longform-Generierung soll nicht nur technisch gute Clips erzeugen,
    sondern im gleichen Zielstil bleiben wie die positiv bewerteten
    LoRA-/Trainingsbeispiele. Darum sind die Genre-Profile bewusst enger als
    die frueheren allgemeinen Lofi-Prompts.
    """

    lowered = (genre or "").lower()
    if "jazz" in lowered:
        return {
            "labels": ["Jazz Lofi"],
            "identity": [
                "authentic jazz lofi instrumental",
                "smoky late night jazz lofi",
                "calm jazz harmony, not party music",
            ],
            "required_harmony": [
                "jazz chord voicings clearly present",
                "soft swing feel with warm brushed drums",
                "gentle chord movement across the section",
                "brief soft piano answer phrases add variation",
            ],
            "harmony": [
                "mellow jazz piano chords",
                "mellow jazz piano notes between the chords",
                "warm rhodes jazz chords",
                "soft seventh chords",
                "gentle jazz guitar comping",
                "soft walking bass tucked behind chords",
            ],
            "drums": [
                "brushed drums",
                "soft swing groove",
                "quiet brushed snare",
                "laid back jazz drums",
            ],
            "moods": [
                "relaxed cafe mood",
                "calm midnight lounge",
                "warm study jazz atmosphere",
                "slow reflective jazz mood",
            ],
            "avoid": [
                "no EDM",
                "no trap drums",
                "no party energy",
                "no aggressive brass",
                "soft brushed drums",
            ],
        }
    if "chillhop" in lowered:
        return {
            "labels": ["Chillhop Lofi"],
            "identity": [
                "sample-based chillhop lofi instrumental",
                "understated head nod groove",
                "warm relaxed chillhop beat",
            ],
            "required_harmony": [
                "laid back sampled chord groove",
                "mellow loop stays central with natural feel",
                "tiny natural voicing drift inside the chords",
            ],
            "harmony": [
                "warm piano sample with soft chord movement",
                "muted rhodes sample with loose feel",
                "soft sampled keys tucked into the groove",
                "short understated motif inside the chords",
            ],
            "drums": ["loose boom bap drums", "controlled snare", "laid back kick", "soft dusty drum groove"],
            "moods": ["relaxed study mood", "warm evening groove", "calm urban lofi mood"],
            "avoid": [
                "no aggressive drop",
                "no fast party beat",
                "relaxed boom bap drums only",
                "no forced fills",
                "no dramatic chord reveal",
                "no showy melody",
            ],
        }
    if "dreamy" in lowered:
        return {
            "labels": ["Dreamy Lofi"],
            "identity": [
                "clean dreamy lofi instrumental",
                "soft floating dreamy lofi",
                "smooth clean dreamy beat",
            ],
            "required_harmony": [
                "clean soft pad layer remains audible",
                "gentle melody without hard lead",
                "layered soft pads and small melodic movement",
                "smooth pad movement across the full thirty seconds",
                "clean quiet background behind the chords",
            ],
            "harmony": [
                "gentle pad chords with slow movement",
                "mellow piano with soft melodic variation",
                "clean soft rhodes chords",
                "soft pad harmony under the melody",
            ],
            "drums": ["soft kick", "quiet snare", "very soft drum bed"],
            "moods": ["dreamy night mood", "clean reflective mood", "warm nostalgic feeling"],
            "avoid": ["clean quiet background", "soft kick and snare only", "smooth clean mix"],
        }
    if "study" in lowered:
        return {
            "labels": ["Study Lofi"],
            "identity": ["focused study lofi instrumental", "steady calm study beat"],
            "required_harmony": [
                "stable background harmony",
                "no distracting lead instrument",
                "quiet chord movement keeps the loop alive",
                "subtle phrase variation inside each thirty second loop",
            ],
            "harmony": [
                "mellow piano chords with small variations",
                "warm rhodes with gentle chord movement",
                "simple soft melody",
                "gentle guitar accents answering the chords",
            ],
            "drums": ["soft steady drums", "controlled snare", "quiet groove"],
            "moods": ["calm focus mood", "warm background music", "relaxed concentration"],
            "avoid": ["no distracting lead", "warm muted percussion", "steady full-length audio", "no busy percussion"],
        }
    if "guitar" in lowered:
        return {
            "labels": ["Guitar Lofi"],
            "identity": [
                "guitar-led lofi instrumental",
                "clean electric guitar lofi beat",
                "mellow acoustic guitar lofi with guitar in foreground",
            ],
            "required_harmony": [
                "guitar clearly audible in the foreground",
                "clean guitar carries the main loop",
                "warm simple guitar phrase",
                "gentle chord changes under the guitar",
            ],
            "harmony": [
                "clean electric guitar arpeggios",
                "soft fingerpicked guitar loop",
                "mellow nylon string guitar chords",
                "gentle guitar melody",
                "warm guitar harmonics",
            ],
            "drums": ["soft drums", "controlled snare", "quiet groove"],
            "moods": ["warm intimate room", "relaxed evening guitar mood", "calm nostalgic mood"],
            "avoid": [
                "no rock energy",
                "no harsh guitar",
                "no party rhythm",
                "no piano-led arrangement",
                "no rhodes-led arrangement",
            ],
        }
    if "alle" in lowered:
        return {
            "labels": ["Jazz Lofi", "Chillhop Lofi", "Dreamy Lofi", "Study Lofi", "Guitar Lofi"],
            "identity": ["calm controlled lofi instrumental", "authentic relaxed lofi"],
            "harmony": ["mellow piano chords", "warm rhodes chords", "soft guitar accents"],
            "drums": ["soft drums", "controlled snare", "quiet groove"],
            "moods": ["relaxed study mood", "warm evening atmosphere", "calm late night mood"],
            "avoid": ["no aggressive drop", "soft drums only", "no party energy"],
        }
    return {
        "labels": [lofi_genre_label(genre)],
        "identity": ["calm controlled lofi instrumental", "authentic relaxed lofi"],
        "harmony": ["mellow piano chords", "warm rhodes chords", "soft guitar accents", "gentle melodic phrases"],
        "drums": ["soft drums", "controlled snare", "quiet groove", "subtle brushed drums"],
        "moods": ["relaxed late night mood", "calm evening atmosphere", "warm nostalgic mood"],
        "avoid": ["no aggressive drops", "smooth clean mix", "no party energy"],
    }


def optional_quality_helpers_report() -> Dict[str, Any]:
    """Dokumentiert lokal vorhandene Hilfsmodelle und ihre aktive Rolle."""

    return {
        "clap": {
            "purpose": "semantische Genre- und Referenz-Aehnlichkeit neuer Kandidaten",
            "available": bool(
                DEFAULT_CLAP_MODELL.is_dir()
                and importlib.util.find_spec("transformers")
            ),
            "local_model": rel(DEFAULT_CLAP_MODELL),
            "auto_load": True,
            "implementation": "transformers.ClapModel, nur lokale Dateien",
        },
        "mert": {
            "purpose": "musikalische Stil-/Genre-Konsistenz und Aehnlichkeit",
            "available": False,
            "auto_load": False,
            "hinweis": "Nur vorbereiteter Reporteintrag; MERT wird nicht automatisch geladen.",
        },
        "essentia": {
            "purpose": "BPM-, Beat- und Rhythmuspruefung",
            "available": bool(importlib.util.find_spec("essentia")),
            "auto_load": False,
        },
        "demucs": {
            "purpose": "optionale Bass-/Drums-/Shaker-Trennung fuer genauere Analyse",
            "available": bool(importlib.util.find_spec("demucs") or shutil.which("demucs")),
            "auto_load": False,
        },
    }


def prompt_variants(
    genre: str,
    stimmung: str,
    instrumente: str,
    rng: random.Random,
    ziel_bpm: float = 78.0,
) -> List[str]:
    profile = genre_prompt_profile(genre)
    mix_color = [
        "clean balanced mix",
        "smooth clean mix",
        "soft analog warmth",
        "warm rounded sound",
    ]
    user_instruments = split_prompt_terms(instrumente)
    user_moods = split_prompt_terms(stimmung)
    guitar_profile = "guitar" in (genre or "").lower() or "gitarre" in (genre or "").lower()
    user_harmony = harmony_prompt_terms(user_instruments, guitar_only=guitar_profile)
    instrument_pool = user_harmony + profile["harmony"]
    mood_pool = user_moods + profile["moods"]
    required_harmony = profile.get("required_harmony", [])
    bpm_phrase = f"steady {int(round(ziel_bpm))} BPM tempo"
    variants: List[str] = []
    used: set[str] = set()
    for _ in range(24):
        genre_label = rng.choice(profile["labels"])
        identity = rng.choice(profile["identity"])
        main_harmony = rng.choice(instrument_pool)
        support_harmony = rng.choice([item for item in profile["harmony"] if item != main_harmony] or profile["harmony"])
        percussion = rng.choice(profile["drums"])
        selected_mood = rng.choice(mood_pool)
        parts = [
            genre_label,
            identity,
            bpm_phrase,
            "constant BPM",
            "calm controlled lofi arrangement",
            "relaxed mellow groove",
            percussion,
            *required_harmony,
            main_harmony,
            support_harmony if rng.random() < 0.65 else "",
            "layered but soft arrangement, not empty",
            "subtle natural variation across the loop",
            "organic understated phrasing",
            "natural unforced loop feel",
            rng.choice(mix_color),
            selected_mood,
            "light low end, bass tucked behind chords, no booming sub bass",
            "smooth clean mix",
            "soft kick and snare",
            rng.choice(profile["avoid"]),
            "no aggressive drops",
            "complete thirty second musical phrase",
            "music remains audible for the full thirty seconds",
        ]
        prompt = clean_generation_prompt(", ".join(part for part in parts if part))
        if prompt in used:
            prompt = clean_generation_prompt(f"{prompt}, subtle arrangement variation")
        used.add(prompt)
        variants.append(prompt)
    return variants


def prompt_for_candidate(base_prompt: str, candidate_index: int) -> str:
    """Varriert den Prompt leicht, wenn ein Abschnitt erneut versucht wird.

    Unterschiedliche Seeds reichen bei MusicGen nicht immer aus. Wenn ein
    Kandidat still oder instabil wird, bekommt der naechste Versuch eine klare
    technische Richtung: durchgehend spielen, keine Stille, stabile Lautheit.
    """

    stabilizers = [
        "",
        "continuous calm lofi performance, same genre identity, stable tempo, stable volume, audible from first to final second",
        "steady musical phrase, same BPM through the whole section, consistent groove, balanced low end, smooth ending, keep the requested genre",
        "active but calm arrangement, complete second half, balanced mix, natural ending, no style drift",
        "steady lofi groove with chords, quiet drums and light low end through the whole section, keep the requested genre",
        "complete quiet controlled thirty second lofi section, consistent rhythm, stable harmony, soft drums only, no party energy",
    ]
    extra = stabilizers[min(candidate_index - 1, len(stabilizers) - 1)]
    if not extra:
        return clean_generation_prompt(base_prompt)
    return clean_generation_prompt(f"{base_prompt}, {extra}")


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


def discover_fallback_audio() -> List[Path]:
    """Findet positive alte Review-Clips nur fuer den ausdruecklichen Notfall."""

    files: List[Path] = []
    review_root = PROJECT_ROOT / "training" / "bewertungen" / "musicgen"
    for csv_path in sorted(review_root.glob("durchgang_*/bewertung.csv")):
        for line in csv_path.read_text(encoding="utf-8", errors="replace").splitlines():
            row = parse_review_line(line)
            if not row or not review_is_positive(row["note"]):
                continue
            audio_path = (csv_path.parent / row["file"]).resolve()
            if audio_path.exists() and audio_path.suffix.lower() in {".mp3", ".wav", ".flac"}:
                files.append(audio_path)
    return sorted(dict.fromkeys(files))


def plan_blocks(
    duration_sec: float,
    block_sec: float,
    variation_sec: float,
    rng: random.Random,
    minimum_sec: float,
) -> List[float]:
    """Verteilt Rhythmusbloecke ohne einen zu kurzen Restblock.

    Die Zielzeit bestimmt die ungefaehre Blockanzahl. Anschliessend wird die
    Gesamtdauer gleichmaessig auf diese Anzahl verteilt. Dadurch entstehen bei
    einer Stunde und fuenf Prozent exakt 20 Bloecke zu je drei Minuten.
    """

    if duration_sec <= 0:
        return []
    minimum = max(1.0, min(float(minimum_sec), duration_sec))
    target = max(minimum, float(block_sec))
    maximum_count = max(1, int(math.floor(duration_sec / minimum)))
    ideal_count = max(1, int(round(duration_sec / target)))
    block_count = min(maximum_count, ideal_count)
    blocks: List[float] = []
    remaining = duration_sec
    variation = max(0.0, float(variation_sec))
    for index in range(block_count):
        remaining_blocks = block_count - index
        if remaining_blocks == 1:
            blocks.append(remaining)
            break
        ideal = remaining / remaining_blocks
        low = max(minimum, ideal - variation)
        high = min(
            ideal + variation,
            remaining - minimum * (remaining_blocks - 1),
        )
        current = rng.uniform(low, high) if variation > 0 and high > low else ideal
        blocks.append(current)
        remaining -= current
    return blocks


def planned_section_duration(section: GeplanterAbschnitt, fallback_seconds: float) -> float:
    """Dauer eines musikalischen Blocks, mindestens eine Clip-Laenge."""

    return max(float(fallback_seconds), float(section.end_sec) - float(section.start_sec))


def estimated_block_assembled_duration(sections: List[GeplanterAbschnitt], fallback_seconds: float, crossfade: float) -> float:
    """Schaetzt die Laenge nach dem Zusammenbau mit Block-Crossfades."""

    if not sections:
        return 0.0
    total = sum(planned_section_duration(section, fallback_seconds) for section in sections)
    return max(0.0, total - max(0, len(sections) - 1) * max(0.0, crossfade))


def planned_rhythm_summary(
    sections: List[GeplanterAbschnitt],
    duration_sec: float,
) -> Dict[str, Any]:
    """Fasst die tatsaechlichen Rhythmusgrenzen des Plans zusammen."""

    block_starts: Dict[int, float] = {}
    for section in sections:
        block_starts.setdefault(int(section.block), float(section.start_sec))
    starts = sorted(max(0.0, min(duration_sec, value)) for value in block_starts.values())
    durations: List[float] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else duration_sec
        if end > start:
            durations.append(end - start)
    if not durations:
        durations = [duration_sec]
    return {
        "count": len(durations),
        "min_seconds": round(min(durations), 3),
        "mean_seconds": round(sum(durations) / len(durations), 3),
        "max_seconds": round(max(durations), 3),
    }


def row_block_duration(row: Dict[str, Any], fallback_seconds: float) -> float:
    """Dauer eines akzeptierten Blocks aus Report-Zeilen rekonstruieren."""

    start = optional_float(row.get("start_sec"))
    end = optional_float(row.get("end_sec"))
    if start is not None and end is not None and end > start:
        return max(float(fallback_seconds), end - start)
    return float(fallback_seconds)


def nutzt_einen_clip_pro_block(args: argparse.Namespace) -> bool:
    """Ein Planabschnitt entspricht einem ganzen Rhythmusblock."""

    return bool(
        getattr(args, "block_looping_aktiv", False)
        or getattr(args, "kontinuierliche_bloecke_aktiv", False)
    )


def plan_sections(
    args: argparse.Namespace,
    duration_sec: float,
    rng: random.Random,
    base_seed: int,
    crossfade_seconds: float,
) -> List[GeplanterAbschnitt]:
    variants = prompt_variants(args.genre, args.stimmung, args.instrumente, rng, args.ziel_bpm)
    sections: List[GeplanterAbschnitt] = []
    cursor = 0.0
    for block_index, block_duration in enumerate(
        plan_blocks(
            duration_sec,
            args.block_sekunden,
            args.block_variation_sekunden,
            rng,
            args.min_rhythmus_sekunden,
        ),
        start=1,
    ):
        block_end = min(duration_sec, cursor + block_duration)
        prompt_offset = rng.randrange(len(variants))
        block_prompt = variants[prompt_offset]
        if nutzt_einen_clip_pro_block(args):
            # Ab Block 2 wird die Crossfade-Zeit zusaetzlich erzeugt. Dadurch
            # traegt jeder Block netto seine geplante Dauer bei und ein
            # Zwei-Stunden-Lauf bleibt bei genau 24 Fuenf-Minuten-Bloecken.
            material_duration = block_duration + (crossfade_seconds if block_index > 1 else 0.0)
            section_index = len(sections) + 1
            sections.append(
                GeplanterAbschnitt(
                    block=block_index,
                    abschnitt=section_index,
                    start_sec=round(cursor, 3),
                    end_sec=round(cursor + material_duration, 3),
                    prompt=block_prompt,
                    seed=section_seed(base_seed, section_index),
                )
            )
            cursor = block_end
            continue
        abschnitt_im_block = 0
        while cursor < block_end - 0.001:
            # Die Blockgrenze ist musikalische Planung, aber jeder MusicGen-
            # Kandidat bleibt ein voller 30s-Clip. Kurze Restclips klingen
            # in Longform besonders oft wie harte Wechsel.
            end = cursor + args.abschnitt_sekunden
            section_index = len(sections) + 1
            sections.append(
                GeplanterAbschnitt(
                    block=block_index,
                    abschnitt=section_index,
                    start_sec=round(cursor, 3),
                    end_sec=round(end, 3),
                    prompt=block_prompt,
                    seed=section_seed(base_seed, section_index),
                )
            )
            cursor = end
            abschnitt_im_block += 1
    if nutzt_einen_clip_pro_block(args):
        while estimated_block_assembled_duration(sections, args.abschnitt_sekunden, crossfade_seconds) < duration_sec:
            section_index = len(sections) + 1
            block_duration = max(args.block_sekunden, args.abschnitt_sekunden)
            sections.append(
                GeplanterAbschnitt(
                    block=sections[-1].block + 1 if sections else 1,
                    abschnitt=section_index,
                    start_sec=round(cursor, 3),
                    end_sec=round(cursor + block_duration, 3),
                    prompt=variants[section_index % len(variants)],
                    seed=section_seed(base_seed, section_index),
                )
            )
            cursor += block_duration
    minimum_sections = len(sections) if nutzt_einen_clip_pro_block(args) else required_section_count(
        duration_sec, args.abschnitt_sekunden, crossfade_seconds
    )
    while len(sections) < minimum_sections:
        prompt = variants[len(sections) % len(variants)]
        section_index = len(sections) + 1
        sections.append(
            GeplanterAbschnitt(
                block=sections[-1].block if sections else 1,
                abschnitt=section_index,
                start_sec=round(cursor, 3),
                end_sec=round(cursor + args.abschnitt_sekunden, 3),
                prompt=prompt,
                seed=section_seed(base_seed, section_index),
            )
        )
        cursor += args.abschnitt_sekunden
    if args.max_abschnitte and args.max_abschnitte > 0:
        return sections[: args.max_abschnitte]
    return sections


def create_extra_section(
    args: argparse.Namespace,
    rng: random.Random,
    base_seed: int,
    section_number: int,
    start_sec: float,
) -> GeplanterAbschnitt:
    """Plant einen zusaetzlichen frischen Abschnitt, wenn vorherige scheitern."""

    variants = prompt_variants(args.genre, args.stimmung, args.instrumente, rng, args.ziel_bpm)
    rhythm_block_seconds = max(
        float(getattr(args, "effective_rhythm_block_seconds", args.block_sekunden)),
        args.abschnitt_sekunden,
    )
    return GeplanterAbschnitt(
        block=max(1, int(start_sec // rhythm_block_seconds) + 1),
        abschnitt=section_number,
        start_sec=round(start_sec, 3),
        end_sec=round(
            start_sec + (
                rhythm_block_seconds
                if nutzt_einen_clip_pro_block(args)
                else args.abschnitt_sekunden
            ),
            3,
        ),
        prompt=rng.choice(variants),
        seed=section_seed(base_seed, section_number),
    )


def write_plan(path: Path, rows: List[GeplanterAbschnitt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block", "abschnitt", "start_sec", "end_sec", "seed", "prompt"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "block": row.block,
                    "abschnitt": row.abschnitt,
                    "start_sec": row.start_sec,
                    "end_sec": row.end_sec,
                    "seed": row.seed,
                    "prompt": row.prompt,
                }
            )


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Liest vorhandene CSV-Reports, falls ein Lauf fortgesetzt wird."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_existing_accepted_clips(
    run_dir: Path,
    planned: List[GeplanterAbschnitt],
    args: argparse.Namespace,
    base_seed: int,
) -> List[Dict[str, Any]]:
    """Rekonstruiert bereits akzeptierte Clips nach einem abgebrochenen Lauf."""

    existing_rows = read_csv_rows(run_dir / "akzeptierte_clips.csv")
    if existing_rows:
        valid_rows = []
        for row in existing_rows:
            path = PROJECT_ROOT / str(row.get("path", ""))
            if path.exists():
                valid_rows.append(row)
        if valid_rows:
            return sorted(valid_rows, key=lambda row: int(float(row.get("abschnitt") or 0)))

    planned_by_section = {section.abschnitt: section for section in planned}
    rows: List[Dict[str, Any]] = []
    for clip_path in sorted((run_dir / "clips").glob("abschnitt_*.wav")):
        try:
            section_number = int(clip_path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        section = planned_by_section.get(section_number)
        if section is None:
            continue
        metrics = audio_metrics(
            clip_path,
            (
                planned_section_duration(section, args.abschnitt_sekunden)
                if args.kontinuierliche_bloecke_aktiv
                else args.abschnitt_sekunden
            ),
            args.ziel_bpm,
            args.bpm_toleranz,
            args.bpm_pruefung_aktiv,
            args.sekunden_pruefung_aktiv,
            args.min_sekunden_rms_db,
        )
        rows.append(
            {
                "block": section.block,
                "transition_group": f"block_{section.block:03d}",
                "abschnitt": section.abschnitt,
                "start_sec": section.start_sec,
                "end_sec": section.end_sec,
                "kandidat": "resume",
                "path": rel(clip_path),
                "seed": section.seed,
                "base_seed": base_seed,
                "section_seed": section.seed,
                "seed_strategy": "base_seed + section_index * 1009 + attempt * 37",
                "prompt": section.prompt,
                "resumed_from_existing_clip": True,
                **metrics,
            }
        )
    return sorted(rows, key=lambda row: int(row.get("abschnitt") or 0))


def generate_candidate(
    model: Any,
    prompt: str,
    seed: int,
    duration_sec: float,
    args: argparse.Namespace,
    target: Path,
    previous_clip: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.set_generation_params(
        use_sampling=True,
        duration=duration_sec,
        extend_stride=min(
            max(1.0, float(getattr(args, "erweiterungs_schritt_sekunden", 12.0))),
            max(1.0, float(getattr(model, "max_duration", 30.0)) - 0.5),
        ),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_coef=args.cfg_coef,
    )
    model.set_custom_progress_callback(progress_callback)
    try:
        genre_melody = None
        if getattr(args, "genre_melodie_conditioning_aktiv", False):
            genre_melody = melody_reference_from_genre(
                args.genre,
                seed=seed,
                duration_sec=duration_sec,
                dataset_root=Path(getattr(args, "referenz_dataset_root", "")).expanduser(),
            )
        if previous_clip is not None and getattr(args, "anschluss_conditioning_aktiv", False):
            melody = melody_reference_from_previous(
                previous_clip,
                duration_sec=duration_sec,
                reference_sec=float(getattr(args, "anschluss_sekunden", 8.0)),
            )
            wav = model.generate_with_chroma([prompt], melody, SAMPLE_RATE, progress=True)[0].cpu()
        elif genre_melody is not None:
            wav = model.generate_with_chroma([prompt], genre_melody, SAMPLE_RATE, progress=True)[0].cpu()
        else:
            wav = model.generate([prompt], progress=True)[0].cpu()
    finally:
        model.set_custom_progress_callback(None)
    # audiocrafts eigene "loudness"-Normalisierung zielt auf ein lauteres
    # Profil (kaum Headroom, Peak nahe 0 dB) als unsere Referenz-Tracks.
    # Dadurch wurden Kandidaten unten in audio_metrics()/score_mp3_referenz()
    # gegen ein Lautheitsprofil geprueft, das sie mangels Normalisierung
    # praktisch nie treffen konnten. Deshalb hier direkt mit demselben
    # normalize_audio()-Ziel schreiben, das auch build_longform() fuer die
    # finale Audio verwendet - Pruefung und Endergebnis sehen dann dieselbe
    # Lautstaerke.
    audio = normalize_audio(wav.numpy().reshape(-1), args.ziel_rms_db, args.peak_limit)
    write_wav(target, audio)
    return target


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_longform(accepted: List[Dict[str, Any]], args: argparse.Namespace, run_dir: Path, target_duration_sec: float) -> Path:
    built = np.zeros(0, dtype=np.float32)
    loop_rows: List[Dict[str, Any]] = []
    crossfade_seconds = effective_crossfade_seconds(args.abschnitt_sekunden, args.crossfade_sekunden)
    crossfade = int(round(crossfade_seconds * SAMPLE_RATE))
    last_source_audio: Optional[np.ndarray] = None
    last_row: Optional[Dict[str, Any]] = None
    for block_index, row in enumerate(accepted):
        audio = decode_audio(project_path(row["path"]))
        audio = normalize_audio(audio, args.ziel_rms_db, args.peak_limit)
        last_source_audio = audio.astype(np.float32, copy=True)
        last_row = row
        if args.block_looping_aktiv:
            block_duration = row_block_duration(row, args.abschnitt_sekunden)
            audio, loop_analysis, loop_uses = build_looped_block(
                audio,
                block_duration,
                args.loop_crossfade_sekunden,
                args.ziel_bpm,
                args.tempo_variation_prozent,
                args.tempo_phase_sekunden,
                block_index,
                int(row.get("base_seed") or row.get("seed") or 0),
            )
            row.update(
                {
                    "loop_start_sec": loop_analysis.start_sec,
                    "loop_end_sec": loop_analysis.end_sec,
                    "loop_duration_sec": loop_analysis.loop_duration_sec,
                    "loop_crossfade_sec": loop_analysis.crossfade_sec,
                    "loop_bpm": loop_analysis.bpm,
                    "loop_seam_score": loop_analysis.seam_score,
                    "loop_status": loop_analysis.status,
                    "loop_warnung": loop_analysis.warnung,
                    "loop_verwendungen": loop_uses,
                    "tempo_variation_percent": loop_analysis.tempo_variation_percent,
                    "tempo_phase_sec": loop_analysis.tempo_phase_sec,
                    "tempo_min_factor": loop_analysis.tempo_min_factor,
                    "tempo_max_factor": loop_analysis.tempo_max_factor,
                    "tempo_factors": loop_analysis.tempo_factors,
                }
            )
            loop_rows.append(
                {
                    "block": row.get("block", ""),
                    "abschnitt": row.get("abschnitt", ""),
                    "datei": row.get("path", ""),
                    **loop_analysis.als_dict(),
                    "loop_verwendungen": loop_uses,
                }
            )
            audio = soften_clip_edges(audio, args.interner_crossfade_sekunden)
        else:
            audio = soften_clip_edges(audio, args.interner_crossfade_sekunden)
        built = append_with_crossfade(built, audio, crossfade)
    if loop_rows:
        write_csv(run_dir / "loop_analyse.csv", loop_rows)
    target_frames = int(round(target_duration_sec * SAMPLE_RATE))
    if len(built) < target_frames:
        if last_source_audio is None or last_row is None:
            raise RuntimeError("Zu wenige akzeptierte Clips fuer die gewuenschte Longform-Dauer.")
        if args.kontinuierliche_bloecke_aktiv:
            raise RuntimeError(
                "Kontinuierliche MusicGen-Bloecke sind kuerzer als geplant. "
                "Der Lauf wird nicht durch eine hoerbare 30s-Wiederholung aufgefuellt."
            )
        missing_sec = max(0.0, (target_frames - len(built)) / float(SAMPLE_RATE))
        filler_sec = missing_sec + (crossfade_seconds if len(built) else 0.0) + 0.25
        filler_audio, filler_analysis, filler_uses = build_looped_block(
            last_source_audio,
            filler_sec,
            args.loop_crossfade_sekunden,
            args.ziel_bpm,
            args.tempo_variation_prozent,
            args.tempo_phase_sekunden,
            len(accepted),
            int(last_row.get("base_seed") or last_row.get("seed") or 0) + 991,
        )
        filler_audio = soften_clip_edges(filler_audio, args.interner_crossfade_sekunden)
        built = append_with_crossfade(built, filler_audio, crossfade)
        loop_rows.append(
            {
                "block": last_row.get("block", ""),
                "abschnitt": last_row.get("abschnitt", ""),
                "datei": last_row.get("path", ""),
                "filler": True,
                "missing_seconds_before_filler": round(missing_sec, 3),
                **filler_analysis.als_dict(),
                "loop_verwendungen": filler_uses,
            }
        )
        write_csv(run_dir / "loop_analyse.csv", loop_rows)
    if len(built) < target_frames:
        raise RuntimeError("Zu wenige akzeptierte Clips fuer die gewuenschte Longform-Dauer.")
    built = built[:target_frames]
    fade_in_frames = min(len(built), int(round(max(0.0, args.fade_in_sekunden) * SAMPLE_RATE)))
    fade_out_frames = min(len(built), int(round(max(0.0, args.fade_out_sekunden) * SAMPLE_RATE)))
    if fade_in_frames > 0:
        built[:fade_in_frames] *= np.sin(
            np.linspace(0.0, math.pi / 2.0, fade_in_frames, dtype=np.float32)
        )
    if fade_out_frames > 0:
        built[-fade_out_frames:] *= np.cos(
            np.linspace(0.0, math.pi / 2.0, fade_out_frames, dtype=np.float32)
        )
    wav_path = run_dir / "lange_audio.wav"
    write_wav(wav_path, built)
    try:
        apply_post_eq(wav_path)
        print("Post-EQ angewendet: Bass -3 dB, Rausch-/Shakerbereich stark reduziert, Limiter.", flush=True)
    except Exception as exc:
        print(f"Warnung: Post-EQ fehlgeschlagen ({exc}). Originaldatei wird behalten.", flush=True)
    if not args.kein_mp3:
        export_mp3(wav_path, run_dir / "lange_audio.mp3")
    return wav_path


def write_transition_quality_report(
    wav_path: Path,
    timeline: List[Dict[str, Any]],
    run_dir: Path,
    crossfade_seconds: float,
    bpm_tolerance: float,
) -> Dict[str, Any]:
    """Prueft Pegel- und Frequenzspruenge rund um Longform-Uebergaenge."""

    audio = decode_audio(wav_path)
    rows: List[Dict[str, Any]] = []
    window_frames = int(round(2.0 * SAMPLE_RATE))
    for index, row in enumerate(timeline):
        if index == 0:
            continue
        transition_sec = float(row.get("start_sec") or 0.0)
        center = int(round(transition_sec * SAMPLE_RATE))
        before = audio[max(0, center - window_frames) : center]
        after = audio[center : min(len(audio), center + window_frames)]
        if len(before) < SAMPLE_RATE * 0.5 or len(after) < SAMPLE_RATE * 0.5:
            continue
        before_bass, before_high, _ = spectral_ratios(before)
        after_bass, after_high, _ = spectral_ratios(after)
        before_bands = spectral_band_ratios(before)
        after_bands = spectral_band_ratios(after)
        rms_diff = abs(db(rms(before)) - db(rms(after)))
        peak_diff = abs(
            db(float(np.max(np.abs(before))) if len(before) else 0.0)
            - db(float(np.max(np.abs(after))) if len(after) else 0.0)
        )
        bass_diff = abs(before_bass - after_bass)
        snare_diff = abs(float(before_bands["snare_ratio"]) - float(after_bands["snare_ratio"]))
        high_diff = abs(before_high - after_high)
        harmonic_similarity = cosine_similarity(spectral_fingerprint(before), spectral_fingerprint(after))
        harmonic_distance = 1.0 - harmonic_similarity
        rhythm_correlation = korrelation(energieverlauf(before), energieverlauf(after))
        click_frames = int(round(0.12 * SAMPLE_RATE))
        click_risk = max(
            click_risiko(before[-click_frames:]),
            click_risiko(after[:click_frames]),
        )
        boundary_jump = abs(float(before[-1]) - float(after[0])) if len(before) and len(after) else 0.0
        previous_bpm = optional_float(timeline[index - 1].get("estimated_bpm"))
        next_bpm = optional_float(row.get("estimated_bpm"))
        bpm_diff = abs(previous_bpm - next_bpm) if previous_bpm is not None and next_bpm is not None else None
        transition_score = (
            rms_diff * 0.42
            + peak_diff * 0.25
            + bass_diff * 10.0
            + snare_diff * 9.0
            + high_diff * 11.0
            + harmonic_distance * 7.0
            + max(0.0, 0.25 - rhythm_correlation) * 3.0
            + max(0.0, click_risk - 10.0) * 0.25
            + boundary_jump * 4.0
            + ((bpm_diff or 0.0) * 0.65)
            + max(0.0, (bpm_diff or 0.0) - bpm_tolerance) * 1.25
        ) * TRANSITION_SCORE_WEIGHT
        reasons: List[str] = []
        hard_reasons: List[str] = []
        if bpm_diff is None:
            reasons.append("BPM nicht vergleichbar")
        elif bpm_diff > bpm_tolerance * 1.6:
            reasons.append("deutlicher BPM-Sprung")
            hard_reasons.append("deutlicher BPM-Sprung")
        elif bpm_diff > bpm_tolerance:
            reasons.append("BPM leicht unterschiedlich")
        for condition, reason, hard in (
            (rms_diff > 4.5, "Lautheitssprung", rms_diff > 7.5),
            (peak_diff > 6.0, "Energiesprung", peak_diff > 9.0),
            (bass_diff > 0.14, "Basssprung", bass_diff > 0.22),
            (snare_diff > 0.12, "Snare-/Praesenzsprung", snare_diff > 0.18),
            (high_diff > 0.08, "Hoehensprung", high_diff > 0.14),
            (harmonic_distance > 0.35, "harmonische/timbrale Distanz", harmonic_distance > 0.55),
            (rhythm_correlation < 0.05, "Rhythmus-/Energieverlauf passt schlecht", rhythm_correlation < -0.25),
            (click_risk > 10.0, "Klickrisiko", click_risk > 18.0),
            (boundary_jump > 0.55, "Wellenform-Sprung", boundary_jump > 0.80),
        ):
            if condition:
                reasons.append(reason)
                if hard:
                    hard_reasons.append(reason)
        if db(rms(after)) < -42.0:
            reasons.append("Dropout nach Uebergang")
            hard_reasons.append("Dropout nach Uebergang")
        if transition_score > MAX_AKZEPTIERTER_UEBERGANG_SCORE:
            reasons.append("Uebergangsscore ueber Annahmelimit")
            hard_reasons.append("Uebergangsscore ueber Annahmelimit")
        status = "PROBLEM" if hard_reasons else "PRUEFEN" if reasons else "OK"
        rows.append(
            {
                "uebergang": index,
                "time_sec": round(transition_sec, 3),
                "time_label": seconds_label(transition_sec),
                "von_abschnitt": timeline[index - 1].get("abschnitt", ""),
                "zu_abschnitt": row.get("abschnitt", ""),
                "crossfade_seconds": crossfade_seconds,
                "previous_bpm": round(previous_bpm, 3) if previous_bpm is not None else "",
                "next_bpm": round(next_bpm, 3) if next_bpm is not None else "",
                "bpm_diff": round(bpm_diff, 3) if bpm_diff is not None else "",
                "bpm_tolerance": round(bpm_tolerance, 3),
                "transition_score": round(transition_score, 4),
                "transition_acceptance_limit": MAX_AKZEPTIERTER_UEBERGANG_SCORE,
                "rms_diff_db": round(rms_diff, 3),
                "loudness_jump_db": round(rms_diff, 3),
                "energy_jump_db": round(peak_diff, 3),
                "bass_diff": round(bass_diff, 4),
                "bass_jump": round(bass_diff, 4),
                "snare_diff": round(snare_diff, 4),
                "snare_jump": round(snare_diff, 4),
                "high_diff": round(high_diff, 4),
                "high_jump": round(high_diff, 4),
                "harmonic_similarity": round(harmonic_similarity, 4),
                "harmonic_distance": round(harmonic_distance, 4),
                "rhythm_correlation": round(rhythm_correlation, 4),
                "click_risk": round(click_risk, 4),
                "boundary_jump": round(boundary_jump, 6),
                "status": status,
                "gruende": "; ".join(reasons),
                "harte_gruende": "; ".join(dict.fromkeys(hard_reasons)),
                "transition_warning": "; ".join(reasons),
            }
        )
    path = run_dir / "uebergangs_pruefung.csv"
    write_csv(path, rows)
    return {
        "path": rel(path),
        "transition_count": len(rows),
        "problem_count": sum(1 for row in rows if row.get("status") != "OK"),
        "hard_problem_count": sum(1 for row in rows if row.get("status") == "PROBLEM"),
    }


def next_review_dir() -> Path:
    root = PROJECT_ROOT / "training" / "bewertungen" / "musicgen"
    indices: List[int] = []
    for path in root.glob("longform_*/bewertung.csv"):
        try:
            indices.append(int(path.parent.name.split("_", 1)[1]))
        except Exception:
            continue
    next_index = max(indices, default=0) + 1
    return root / f"longform_{next_index:03d}"


def write_rating_template(audio_path: Path, duration_sec: float) -> Path:
    review_dir = next_review_dir()
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / "bewertung.csv"
    rows: List[Dict[str, Any]] = []
    section_sec = 60.0 if duration_sec <= 600.0 else 300.0
    cursor = 0.0
    index = 1
    while cursor < duration_sec - 0.001:
        end = min(duration_sec, cursor + section_sec)
        rows.append(
            {
                "Abschnitt": f"abschnitt_{index:03d}",
                "Start": seconds_label(cursor),
                "Ende": seconds_label(end),
                "Datei": rel(audio_path),
                "Status": "Offen",
                "Uebergaenge": "",
                "BPM_Stabilitaet": "",
                "Bass": "",
                "Shaker": "",
                "Snare": "",
                "Harmonie": "",
                "Monotonie": "",
                "Stoergeraeusche": "",
                "Lofi_Charakter": "",
                "Gesamteindruck": "",
                "Notiz": "",
            }
        )
        cursor = end
        index += 1
    write_csv(path, rows)
    return path


def haeufige_gruende(rows: List[Dict[str, Any]], key: str = "gruende", limit: int = 6) -> List[tuple[str, int]]:
    """Zaehlt Begruendungen aus CSV-Reportzeilen fuer den Kurzbericht."""

    counts: Dict[str, int] = {}
    for row in rows:
        text = str(row.get(key) or row.get("transition_gruende") or row.get("referenz_gruende") or "")
        for part in text.split(";"):
            reason = part.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def score_mittelwert(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [value for value in (optional_float(row.get(key)) for row in rows) if value is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def write_scientific_audio_report(
    run_dir: Path,
    final_info: Dict[str, Any],
    accepted: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    skipped_sections: List[Dict[str, Any]],
    transition_report: Dict[str, Any],
) -> Path:
    """Schreibt einen sachlichen Kurzbericht fuer die Bachelorarbeit."""

    output_audio = final_info.get("output_mp3") or final_info.get("output_wav") or ""
    transition_problem_count = int(transition_report.get("problem_count") or 0)
    hard_transition_problem_count = int(transition_report.get("hard_problem_count") or 0)
    accepted_transition_mean = score_mittelwert(accepted, "transition_score")
    reference_mean = score_mittelwert(accepted, "referenz_score")
    top_rejections = haeufige_gruende(rejected)
    transition_rejections = haeufige_gruende(
        [row for row in rejected if str(row.get("auswahlstatus") or "") == "UEBERGANG_VERWORFEN"],
        "gruende",
    )
    recommendation = "Audio ist fuer die naechste Bewertung geeignet."
    if hard_transition_problem_count:
        recommendation = "Audio sollte vor der naechsten Bewertung wegen harter Uebergangsprobleme erneut erzeugt werden."
    elif transition_problem_count:
        recommendation = "Audio ist pruefbar, die markierten Uebergaenge sollten menschlich nachgehoert werden."
    elif final_info.get("score_rating", {}).get("status") == "failed":
        recommendation = "Audio ist technisch erzeugt, aber die automatische Fuenf-Score-Bewertung muss nachgeholt werden."

    lines = [
        "# Prozessbericht Longform-Audio",
        "",
        "## Was wurde gemacht?",
        (
            f"Es wurde eine Longform-Audio fuer {final_info.get('genre', 'Lofi')} "
            f"mit einer ZielLaenge von {final_info.get('duration_sec', 0)} Sekunden erzeugt. "
            "MusicGen erzeugte kurze Kandidaten; pro Rhythmusblock wurde der am besten passende "
            "Kandidat gewaehlt und bei aktivem Block-Looping musikalisch verlaengert."
        ),
        "",
        "## Warum wurde es gemacht?",
        (
            "Ziel war eine lokal reproduzierbare Audio-Erzeugung, bei der nicht der isoliert "
            "beste Clip gewinnt, sondern der Clip mit dem musikalisch saubersten Anschluss "
            "an den vorherigen Block."
        ),
        "",
        "## Eingaben",
        f"- Genre: {final_info.get('genre', '')}",
        f"- Dauer: {seconds_label(float(final_info.get('duration_sec') or 0.0))}",
        f"- Ziel-BPM: {final_info.get('target_bpm', '')}",
        f"- LoRA-Adapter: {final_info.get('adapter_path', '')}",
        f"- Rhythmusbloecke: {final_info.get('rhythm_block_count', '')} "
        f"a ca. {final_info.get('rhythm_block_seconds', '')} Sekunden",
        f"- Kandidaten erzeugt: {final_info.get('generated_candidate_count', '')}",
        "",
        "## Erzeugte Dateien",
        f"- Finale Audio: `{output_audio}`",
        f"- Ablauf: `{rel(run_dir / 'ablauf.csv')}`",
        f"- Kandidatenbewertung: `{rel(run_dir / 'kandidaten_score.csv')}`",
        f"- Uebergangsprüfung: `{transition_report.get('path', '')}`",
        f"- Bewertungsvorlage: `{final_info.get('rating_template', '')}`",
        f"- Fuenf-Score-Bewertung: `{final_info.get('score_rating', {}).get('html', '')}`",
        "",
        "## Gepruefte Qualitaetskriterien",
        "- Technische Audioqualitaet: Lautheit, Stille, Clipping, Energie, Bass, Hoehen und Signaltonrisiko.",
        "- Musikalische Kohaerenz: BPM-Stabilitaet, aktive Sekunden und Blockkonsistenz.",
        "- Genre-Treue: lokaler Genre-/Referenzvergleich, sofern verfuegbar.",
        "- Uebergangsqualitaet: BPM-Nahe, Lautheit, Bass, Snare/Praesenz, Hoehen, Energie, spektrale Naehe, Rhythmuskorrelation, Dropouts, abgeschnittene Enden und Klickrisiko.",
        "- Referenzaehnlichkeit: Abstand zum gespeicherten Genrestandard.",
        "",
        "## Gute Ergebnisse",
        f"- Akzeptierte Clips/Bloecke: {len(accepted)}",
        f"- Mittlerer Uebergangsscore der akzeptierten Kandidaten: {accepted_transition_mean if accepted_transition_mean is not None else 'nicht berechnet'}",
        f"- Mittlerer Referenzscore der akzeptierten Kandidaten: {reference_mean if reference_mean is not None else 'nicht berechnet'}",
        f"- Harte Uebergangsprobleme in der finalen Pruefung: {hard_transition_problem_count}",
        "",
        "## Schlechte oder verworfene Ergebnisse",
        f"- Verworfene Kandidaten: {len(rejected)}",
        f"- Uebersprungene Abschnitte: {len(skipped_sections)}",
        f"- Final markierte Uebergaenge: {transition_problem_count}",
    ]
    if top_rejections:
        lines.append("- Haeufigste Verwerfungsgruende:")
        lines.extend(f"  - {reason}: {count}x" for reason, count in top_rejections)
    if transition_rejections:
        lines.append("- Haeufigste reine Uebergangsverwerfungen:")
        lines.extend(f"  - {reason}: {count}x" for reason, count in transition_rejections)
    lines.extend(
        [
            "",
            "## Entscheidung",
            (
                "Die finale Audio wurde aus den akzeptierten Kandidaten zusammengesetzt. "
                "Kandidaten mit harten Uebergangsproblemen wurden nicht als bester Clip uebernommen."
            ),
            "",
            "## Empfehlung",
            recommendation,
            "",
        ]
    )
    path = run_dir / "bericht.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def seconds_label(seconds: float) -> str:
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    return f"{minutes:02d}:{sec:02d}"


def erstelle_zufaellige_musicgen_audio(
    *,
    dauer: str = "20m",
    genre: str = "Lofi",
    stimmung: str = "",
    instrumente: str = "",
    name: str = "",
    seed: int = 0,
    abschnitt_sekunden: float = 30.0,
    crossfade_sekunden: float = 3.0,
    fade_in_sekunden: float = 5.0,
    fade_out_sekunden: float = 5.0,
    interner_crossfade_sekunden: float = 2.0,
    anschluss_conditioning_aktiv: bool = False,
    anschluss_sekunden: float = 8.0,
    ziel_bpm: float = 78.0,
    bpm_toleranz: float = 4.0,
    bpm_pruefung_aktiv: bool = True,
    kandidaten_pro_abschnitt: int = 5,
    max_generierte_kandidaten: int = 0,
    block_looping_aktiv: bool = True,
    block_sekunden: float = 0.0,
    rhythmus_anteil_prozent: float = DEFAULT_RHYTHMUS_ANTEIL_PROZENT,
    min_rhythmus_sekunden: float = DEFAULT_MIN_RHYTHMUS_SEKUNDEN,
    loop_crossfade_sekunden: float = 0.0,
    tempo_variation_prozent: float = 1.5,
    tempo_phase_sekunden: float = 60.0,
    fallback_pool: bool = False,
    nur_plan: bool = False,
    kein_mp3: bool = False,
    vram_limit_fraction: float = 0.80,
    sekunden_pruefung_aktiv: bool = True,
    min_sekunden_rms_db: float = -52.0,
    github_push: bool = True,
) -> Dict[str, Any]:
    """Automatisiert eine zufaellige MusicGen-Longform-Generierung.

    Diese Funktion ist der programmatische Einstieg fuer Pipeline, Backend oder
    spaetere Website. Bei `seed=0` wird jedes Mal ein neuer Zufallsseed genutzt.
    Alte bewertete Clips werden nur verwendet, wenn `fallback_pool=True` gesetzt
    ist. Standardmaessig erzeugt MusicGen frische Abschnitte.
    """

    args = argparse.Namespace(
        dauer=dauer,
        genre=genre,
        stimmung=stimmung,
        instrumente=instrumente,
        name=name,
        ausgabe_root=str(PROJECT_ROOT / "training" / "ausgaben" / "musicgen_generiert"),
        adapter_path=str(DEFAULT_ADAPTER),
        model_id="facebook/musicgen-melody-large",
        model_dir=str(PROJECT_ROOT / "daten" / "modelle" / "musicgen" / "facebook_musicgen_melody_large"),
        allow_remote_model=False,
        seed=seed,
        abschnitt_sekunden=abschnitt_sekunden,
        block_sekunden=block_sekunden,
        rhythmus_anteil_prozent=rhythmus_anteil_prozent,
        min_rhythmus_sekunden=min_rhythmus_sekunden,
        blockmodus_prozentual=block_sekunden <= 0,
        block_variation_sekunden=0.0,
        ziel_bpm=ziel_bpm,
        bpm_toleranz=bpm_toleranz,
        uebergang_bpm_toleranz=2.5,
        bpm_pruefung_aktiv=bpm_pruefung_aktiv,
        kandidaten_pro_abschnitt=kandidaten_pro_abschnitt,
        max_generierte_kandidaten=max_generierte_kandidaten,
        max_abschnitte=0,
        block_looping_aktiv=block_looping_aktiv,
        kontinuierliche_bloecke_aktiv=False,
        erweiterungs_schritt_sekunden=12.0,
        loop_crossfade_sekunden=loop_crossfade_sekunden,
        tempo_variation_prozent=tempo_variation_prozent,
        tempo_phase_sekunden=tempo_phase_sekunden,
        fallback_pool=fallback_pool,
        resume_existing=True,
        temperature=0.78,
        top_k=120,
        top_p=0.0,
        cfg_coef=3.5,
        vram_limit_fraction=vram_limit_fraction,
        disable_vram_limit=False,
        crossfade_sekunden=crossfade_sekunden,
        fade_in_sekunden=fade_in_sekunden,
        fade_out_sekunden=fade_out_sekunden,
        interner_crossfade_sekunden=interner_crossfade_sekunden,
        anschluss_conditioning_aktiv=anschluss_conditioning_aktiv,
        anschluss_sekunden=anschluss_sekunden,
        ziel_rms_db=-22.0,
        peak_limit=0.92,
        sekunden_pruefung_aktiv=sekunden_pruefung_aktiv,
        min_sekunden_rms_db=min_sekunden_rms_db,
        mp3_referenz_pruefung_aktiv=True,
        referenz_dataset_root=str(PROJECT_ROOT / "daten" / "processed" / "lora_training"),
        referenzen_pro_genre=20,
        referenz_score_limit=18.0,
        genre_pruefung_aktiv=True,
        clap_modell_pfad=str(DEFAULT_CLAP_MODELL),
        genre_referenz_ordner=str(DEFAULT_GENRE_REFERENZEN),
        min_genre_aehnlichkeit=0.62,
        min_genre_abstand=0.02,
        min_qualitaets_abstand=0.0,
        kein_mp3=kein_mp3,
        github_push=github_push,
        nur_plan=nur_plan,
    )
    return fuehre_zufaellige_musicgen_generierung_aus(args)


def fuehre_zufaellige_musicgen_generierung_aus(args: argparse.Namespace) -> Dict[str, Any]:
    """Fuehrt die zufaellige MusicGen-Generierung mit fertigen Argumenten aus."""

    args.genre = lofi_genre_label(args.genre)
    duration_sec = parse_duration_seconds(args.dauer)
    requested_block_seconds = float(getattr(args, "block_sekunden", 0.0) or 0.0)
    rhythm_share_percent = float(
        getattr(args, "rhythmus_anteil_prozent", DEFAULT_RHYTHMUS_ANTEIL_PROZENT)
    )
    minimum_rhythm_seconds = float(
        getattr(args, "min_rhythmus_sekunden", DEFAULT_MIN_RHYTHMUS_SEKUNDEN)
    )
    percentage_mode = bool(
        getattr(args, "blockmodus_prozentual", False) or requested_block_seconds <= 0
    )
    args.rhythmus_anteil_prozent = rhythm_share_percent
    args.min_rhythmus_sekunden = minimum_rhythm_seconds
    args.blockmodus_prozentual = percentage_mode
    args.block_sekunden = resolve_rhythm_block_seconds(
        duration_sec,
        0.0 if percentage_mode else requested_block_seconds,
        rhythm_share_percent,
        minimum_rhythm_seconds,
    )
    if args.abschnitt_sekunden < 10:
        raise ValueError("abschnitt-sekunden ist zu kurz.")
    run_name = args.name or f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.ausgabe_root).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_report = read_json_optional(run_dir / "generation_report.json")
    mp3_reference_profiles: Dict[str, Dict[str, Any]] = {}
    mp3_reference_report: Dict[str, Any] = {
        "enabled": bool(getattr(args, "mp3_referenz_pruefung_aktiv", True)),
        "available": False,
        "hinweis": "Wird beim echten Lauf vor der Kandidatenauswahl geladen.",
    }
    if not args.nur_plan:
        mp3_reference_profiles, mp3_reference_report = collect_reference_rows(args, run_dir)
        if args.mp3_referenz_pruefung_aktiv and not mp3_reference_profiles:
            raise RuntimeError(
                "MP3-Referenzpruefung ist aktiv, aber es wurden keine Referenzprofile gefunden: "
                f"{mp3_reference_report.get('error', 'unbekannter Fehler')}"
            )
        if mp3_reference_profiles:
            print(
                "MP3-Referenz: aktiv "
                f"({mp3_reference_report.get('reference_count', 0)} Trainingsclips).",
                flush=True,
            )
    if args.seed == 0 and getattr(args, "resume_existing", True) and existing_report.get("base_seed"):
        seed = int(existing_report["base_seed"])
    else:
        seed = actual_seed(args.seed)
    rng = random.Random(seed)
    crossfade_seconds = effective_crossfade_seconds(args.abschnitt_sekunden, args.crossfade_sekunden)

    planned = plan_sections(args, duration_sec, rng, seed, crossfade_seconds)
    rhythm_summary = planned_rhythm_summary(planned, duration_sec)
    args.effective_rhythm_block_seconds = rhythm_summary["mean_seconds"]
    needed_sections = len(planned) if nutzt_einen_clip_pro_block(args) else required_section_count(
        duration_sec, args.abschnitt_sekunden, crossfade_seconds
    )
    estimated_duration = (
        estimated_block_assembled_duration(planned, args.abschnitt_sekunden, crossfade_seconds)
        if nutzt_einen_clip_pro_block(args)
        else estimated_assembled_duration(
            len(planned),
            args.abschnitt_sekunden,
            crossfade_seconds,
        )
    )
    if estimated_duration + 0.001 < duration_sec:
        raise ValueError(
            "Zu wenige Abschnitte geplant: "
            f"{len(planned)} Abschnitte ergeben mit Crossfade nur ca. {estimated_duration:.1f}s. "
            f"Fuer {duration_sec:.1f}s werden mindestens {needed_sections} Abschnitte benoetigt."
        )
    write_plan(run_dir / "verwendete_prompts.csv", planned)
    write_json(
        run_dir / "generation_report.json",
        {
            "status": "planned" if args.nur_plan else "running",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "genre": args.genre,
            "stimmung": args.stimmung,
            "instrumente": args.instrumente,
            "duration_sec": duration_sec,
            "base_seed": seed,
            "seed_strategy": "segment_seed = base_seed + section_index * 1009 + attempt * 37",
            "target_bpm": args.ziel_bpm,
            "bpm_tolerance": args.bpm_toleranz,
            "bpm_filter_enabled": args.bpm_pruefung_aktiv,
            "second_by_second_quality_check_enabled": args.sekunden_pruefung_aktiv,
            "min_second_rms_db_threshold": args.min_sekunden_rms_db,
            "generation_sampling": {
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "cfg_coef": args.cfg_coef,
            },
            "style_prompt_profile": genre_prompt_profile(args.genre),
            "requested_crossfade_seconds": args.crossfade_sekunden,
            "crossfade_seconds": crossfade_seconds,
            "fade_in_seconds": args.fade_in_sekunden,
            "fade_out_seconds": args.fade_out_sekunden,
            "internal_crossfade_seconds": args.interner_crossfade_sekunden,
            "block_looping_enabled": args.block_looping_aktiv,
            "continuous_musicgen_blocks_enabled": args.kontinuierliche_bloecke_aktiv,
            "musicgen_extend_stride_seconds": args.erweiterungs_schritt_sekunden,
            "rhythm_block_mode": "percentage" if args.blockmodus_prozentual else "fixed",
            "rhythm_share_percent": args.rhythmus_anteil_prozent,
            "minimum_rhythm_seconds": args.min_rhythmus_sekunden,
            "rhythm_target_seconds": args.block_sekunden,
            "rhythm_block_seconds": rhythm_summary["mean_seconds"],
            "rhythm_block_count": rhythm_summary["count"],
            "rhythm_block_min_seconds": rhythm_summary["min_seconds"],
            "rhythm_block_max_seconds": rhythm_summary["max_seconds"],
            "loop_crossfade_requested_seconds": args.loop_crossfade_sekunden,
            "loop_crossfade_strategy": "0 = automatisch ein Beat, maximal 1.25 Sekunden",
            "tempo_variation_percent": args.tempo_variation_prozent,
            "tempo_phase_seconds": args.tempo_phase_sekunden,
            "tempo_variation_strategy": "pitch-erhaltend und schrittweise",
            "transition_strategy": (
                "BPM-nahe Kandidaten, technischer Kandidaten-Score, hart gewichteter "
                "Uebergangs-Score zum vorherigen Abschnitt, Pruefung von Lautheit, "
                "Bass, Snare/Praesenz, Hoehen, Energie, harmonischer/timbraler Naehe, "
                "Rhythmuskorrelation, Dropouts, Anschnitten und Klickrisiko, kurzer "
                "effektiver Crossfade, optionales Anschluss-Conditioning"
            ),
            "anschluss_conditioning_enabled": args.anschluss_conditioning_aktiv,
            "anschluss_reference_seconds": args.anschluss_sekunden if args.anschluss_conditioning_aktiv else 0.0,
            "transition_score_weight": TRANSITION_SCORE_WEIGHT,
            "beat_sync_enabled_or_false": bool(args.block_looping_aktiv),
            "sections_planned": len(planned),
            "sections_minimum_for_crossfade": needed_sections,
            "estimated_assembled_duration_sec": round(estimated_duration, 3),
            "candidates_per_section": args.kandidaten_pro_abschnitt,
            "max_generated_candidates": args.max_generierte_kandidaten or None,
            "adapter_path": rel(Path(args.adapter_path).expanduser()),
            "model_dir": rel(Path(args.model_dir).expanduser()),
            "model_strategy": {
                "base_model": args.model_id,
                "local_model_dir": rel(Path(args.model_dir).expanduser()),
                "allow_remote_model": bool(args.allow_remote_model),
                "adapter_path": rel(Path(args.adapter_path).expanduser()),
                "adapter_run": Path(args.adapter_path).expanduser().parent.name,
                "reference_checkpoint": Path(args.adapter_path).expanduser().name,
                "anschluss_conditioning_enabled": args.anschluss_conditioning_aktiv,
                "melody_conditioning_usage": (
                    "Anfang des naechsten Clips kann mit Ende des vorherigen Clips konditioniert werden."
                    if args.anschluss_conditioning_aktiv
                    else "Text-only Generierung ohne Anschluss-Conditioning."
                ),
            },
            "optional_quality_helpers": optional_quality_helpers_report(),
            "semantic_genre_check": {
                "enabled": bool(args.genre_pruefung_aktiv),
                "model_path": rel(Path(args.clap_modell_pfad).expanduser()),
                "reference_dir": rel(Path(args.genre_referenz_ordner).expanduser()),
                "minimum_target_similarity": args.min_genre_aehnlichkeit,
                "minimum_genre_margin": args.min_genre_abstand,
                "minimum_quality_margin": args.min_qualitaets_abstand,
            },
            "mp3_reference_check": mp3_reference_report,
            "fallback_pool": args.fallback_pool,
            "memory_efficient_attention_disabled": True,
            "hinweis": (
                "Neue MusicGen-Abschnitte werden erzeugt; alte Clips sind nicht Hauptquelle. "
                "Ein akzeptierter 30s-Clip wird pro Musikblock weich auf die geplante Blocklaenge verlaengert."
                if args.block_looping_aktiv
                else (
                    "Jeder Rhythmusblock wird als echte MusicGen-Fortsetzung erzeugt; "
                    "30s-Wiederholungen sind deaktiviert."
                    if args.kontinuierliche_bloecke_aktiv
                    else "Neue MusicGen-Abschnitte werden erzeugt; alte Clips sind nicht Hauptquelle."
                )
            ),
        },
    )

    if args.nur_plan:
        payload = {"status": "planned", "run_dir": rel(run_dir), "sections": len(planned)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    genre_pruefer: Any = None
    genre_pruefung_report: Dict[str, Any] = {
        "enabled": bool(args.genre_pruefung_aktiv),
        "available": False,
    }
    if args.genre_pruefung_aktiv:
        try:
            from genre_pruefung import GenrePruefer

            genre_pruefer = GenrePruefer(
                modell_pfad=Path(args.clap_modell_pfad),
                bewertungs_ordner=Path(args.genre_referenz_ordner),
                min_ziel_aehnlichkeit=args.min_genre_aehnlichkeit,
                min_genre_abstand=args.min_genre_abstand,
                min_qualitaets_abstand=args.min_qualitaets_abstand,
                device="cpu",
            )
            genre_pruefung_report = genre_pruefer.bericht()
            if not genre_pruefer.verfuegbar:
                print(
                    "Genre-Pruefung: nicht verfuegbar, technische Pruefung laeuft weiter "
                    f"({genre_pruefung_report.get('error')}).",
                    flush=True,
                )
                genre_pruefer = None
            else:
                print(
                    "Genre-Pruefung: lokal aktiv "
                    f"({sum(genre_pruefer.referenz_anzahl.values())} gute Referenzen).",
                    flush=True,
                )
        except Exception as exc:
            genre_pruefung_report = {
                "enabled": True,
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            genre_pruefer = None
            print(
                "Genre-Pruefung: nicht verfuegbar, technische Pruefung laeuft weiter "
                f"({genre_pruefung_report['error']}).",
                flush=True,
            )

    helpers = load_lora_helpers()
    helpers.setup_musicgen_environment(PROJECT_ROOT)
    import torch

    vram_limit = (
        {"enabled": False, "message": "VRAM-Limit deaktiviert."}
        if args.disable_vram_limit
        else helpers.apply_cuda_vram_limit(args.vram_limit_fraction)
    )
    if vram_limit.get("enabled"):
        print(vram_limit["message"], flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA ist fuer frische MusicGen-Longform empfohlen und aktuell nicht verfuegbar.")
    model_source = helpers.resolve_model_source(
        project_root=PROJECT_ROOT,
        model_id=args.model_id,
        model_dir=Path(args.model_dir).expanduser().resolve(),
        allow_remote=args.allow_remote_model,
    )
    adapter_path = Path(args.adapter_path).expanduser().resolve()
    if not args.review_adapter_erlauben and not adapter_fuer_generierung_freigegeben(adapter_path):
        raise FileNotFoundError(
            "LoRA-Adapter fehlt oder das zugehoerige LoRA-Training ist noch nicht abgeschlossen: "
            f"{adapter_path}. Den neuen LoRA-Clip-Lauf zuerst lokal fortsetzen: "
            ".venv/bin/python code/start.py --lora-fortsetzen"
        )
    model = load_musicgen_stabil(model_source, device=device)
    adapter_payload = load_lora_checkpoint_kompatibel(model.lm, adapter_path, helpers, device)
    model.lm.eval()

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = read_csv_rows(run_dir / "rejected_clips.csv")
    all_checks: List[Dict[str, Any]] = read_csv_rows(run_dir / "technische_pruefung.csv")
    candidate_scores: List[Dict[str, Any]] = read_csv_rows(run_dir / "kandidaten_score.csv")
    skipped_sections: List[Dict[str, Any]] = read_csv_rows(run_dir / "uebersprungene_abschnitte.csv")
    generated_candidate_count = len(all_checks)
    max_generated_candidates = int(getattr(args, "max_generierte_kandidaten", 0) or 0)
    fallback_files = discover_fallback_audio() if args.fallback_pool else []
    rng.shuffle(fallback_files)
    fallback_index = 0
    target_sections = needed_sections
    max_generation_sections = target_sections + max(20, target_sections // 2)
    if getattr(args, "resume_existing", True):
        accepted = load_existing_accepted_clips(run_dir, planned, args, seed)
        if accepted:
            write_csv(run_dir / "akzeptierte_clips.csv", accepted)
            print(
                f"Fortsetzen: {len(accepted)}/{target_sections} vorhandene gute Clips gefunden.",
                flush=True,
            )
    accepted_sections = {int(row.get("abschnitt") or 0) for row in accepted}

    write_progress(
        run_dir,
        status="running",
        section_index=len(accepted),
        total_sections=target_sections,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        message="Longform-Generierung gestartet oder fortgesetzt.",
    )
    section_cursor = 0
    while len(accepted) < target_sections:
        if section_cursor >= len(planned):
            if len(planned) >= max_generation_sections:
                write_csv(run_dir / "technische_pruefung.csv", all_checks)
                write_csv(run_dir / "rejected_clips.csv", rejected)
                write_csv(run_dir / "kandidaten_score.csv", candidate_scores)
                write_csv(run_dir / "uebersprungene_abschnitte.csv", skipped_sections)
                write_progress(
                    run_dir,
                    status="failed",
                    section_index=len(accepted),
                    total_sections=target_sections,
                    accepted_count=len(accepted),
                    rejected_count=len(rejected),
                    message=(
                        "Zu viele Abschnitte waren technisch unbrauchbar. "
                        f"{len(accepted)}/{target_sections} gute Clips erreicht."
                    ),
                )
                raise RuntimeError(
                    "Zu viele unbrauchbare Abschnitte: "
                    f"{len(accepted)}/{target_sections} gute Clips erreicht."
                )
            planned.append(
                create_extra_section(
                    args=args,
                    rng=rng,
                    base_seed=seed,
                    section_number=len(planned) + 1,
                    start_sec=float(planned[-1].end_sec if planned else 0.0),
                )
            )
            write_plan(run_dir / "verwendete_prompts.csv", planned)

        section = planned[section_cursor]
        section_cursor += 1
        if section.abschnitt in accepted_sections:
            continue
        accepted_path: Optional[Path] = None
        ok_candidates: List[Any] = []
        best_bpm_fallback_row: Optional[Dict[str, Any]] = None
        best_bpm_fallback_path: Optional[Path] = None
        for candidate_index in range(1, max(1, args.kandidaten_pro_abschnitt) + 1):
            if max_generated_candidates and generated_candidate_count >= max_generated_candidates:
                write_csv(run_dir / "technische_pruefung.csv", all_checks)
                write_csv(run_dir / "rejected_clips.csv", rejected)
                write_csv(run_dir / "kandidaten_score.csv", candidate_scores)
                write_csv(run_dir / "uebersprungene_abschnitte.csv", skipped_sections)
                write_progress(
                    run_dir,
                    status="failed",
                    section_index=len(accepted),
                    total_sections=target_sections,
                    accepted_count=len(accepted),
                    rejected_count=len(rejected),
                    message=(
                        "Hartes Generierungslimit erreicht: "
                        f"{generated_candidate_count}/{max_generated_candidates} Kandidaten."
                    ),
                )
                raise RuntimeError(
                    "Hartes Generierungslimit erreicht: "
                    f"{generated_candidate_count}/{max_generated_candidates} Kandidaten. "
                    "Erhoehe --max-generierte-kandidaten oder nutze --nur-plan fuer eine Vorschau."
                )
            candidate_seed = section_seed(seed, section.abschnitt, candidate_index - 1)
            candidate_prompt = prompt_for_candidate(section.prompt, candidate_index)
            candidate_duration = (
                planned_section_duration(section, args.abschnitt_sekunden)
                if args.kontinuierliche_bloecke_aktiv
                else args.abschnitt_sekunden
            )
            candidate_path = run_dir / "kandidaten" / f"abschnitt_{section.abschnitt:04d}_kandidat_{candidate_index:02d}.wav"
            previous_clip_path = (
                project_path(accepted[-1]["path"])
                if accepted and getattr(args, "anschluss_conditioning_aktiv", False)
                else None
            )
            progress_before = len(accepted) / max(1, target_sections) * 100.0
            write_progress(
                run_dir,
                status="running",
                section_index=len(accepted),
                total_sections=target_sections,
                candidate_index=candidate_index,
                total_candidates=args.kandidaten_pro_abschnitt,
                accepted_count=len(accepted),
                rejected_count=len(rejected),
                message=(
                    f"Abschnitt {section.abschnitt}, guter Clip "
                    f"{len(accepted) + 1}/{target_sections}, "
                    f"Kandidat {candidate_index}/{args.kandidaten_pro_abschnitt}"
                ),
            )
            print(
                f"Fortschritt {progress_before:.1f}% | guter Clip "
                f"{len(accepted) + 1}/{target_sections} | Abschnitt "
                f"{section.abschnitt} - Kandidat "
                f"{candidate_index}/{args.kandidaten_pro_abschnitt}",
                flush=True,
            )
            last_progress_write = 0.0
            expected_tokens = max(
                1,
                int(
                    round(
                        candidate_duration
                        * float(getattr(model, "frame_rate", 50.0))
                    )
                ),
            )

            def update_generation_progress(
                generated_tokens: int,
                _tokens_to_generate: int,
            ) -> None:
                """Aktualisiert dieselbe Terminalanzeige waehrend MusicGen rechnet."""

                nonlocal last_progress_write
                now = time.monotonic()
                candidate_fraction = min(
                    1.0,
                    max(0.0, generated_tokens / expected_tokens),
                )
                if now - last_progress_write < 0.5 and candidate_fraction < 1.0:
                    return
                last_progress_write = now
                candidates_in_section = max(1, args.kandidaten_pro_abschnitt)
                section_fraction = (
                    (candidate_index - 1) + candidate_fraction
                ) / candidates_in_section
                overall_percent = (
                    len(accepted) + section_fraction
                ) / max(1, target_sections) * 100.0
                write_progress(
                    run_dir,
                    status="running",
                    section_index=len(accepted),
                    total_sections=target_sections,
                    candidate_index=candidate_index,
                    total_candidates=args.kandidaten_pro_abschnitt,
                    accepted_count=len(accepted),
                    rejected_count=len(rejected),
                    explicit_percent=overall_percent,
                    candidate_progress_percent=candidate_fraction * 100.0,
                    message=(
                        f"MusicGen erzeugt Abschnitt {section.abschnitt}: "
                        f"Kandidat {candidate_index}/"
                        f"{args.kandidaten_pro_abschnitt}"
                    ),
                )

            generated = generate_candidate(
                model=model,
                prompt=candidate_prompt,
                seed=candidate_seed,
                duration_sec=candidate_duration,
                args=args,
                target=candidate_path,
                previous_clip=previous_clip_path,
                progress_callback=update_generation_progress,
            )
            generated_candidate_count += 1
            metrics = audio_metrics(
                generated,
                candidate_duration,
                args.ziel_bpm,
                args.bpm_toleranz,
                args.bpm_pruefung_aktiv,
                args.sekunden_pruefung_aktiv,
                args.min_sekunden_rms_db,
            )
            reference_check = score_mp3_referenz(
                metrics,
                args.genre,
                mp3_reference_profiles,
                args.referenz_score_limit,
            )
            semantic = (
                genre_pruefer.pruefe(generated, args.genre)
                if genre_pruefer is not None
                else {
                    "semantic_status": "DISABLED",
                    "semantic_penalty": 0.0,
                    "semantic_warning": "",
                }
            )
            row = {
                "block": section.block,
                "transition_group": f"block_{section.block:03d}",
                "abschnitt": section.abschnitt,
                "start_sec": section.start_sec,
                "end_sec": section.end_sec,
                "generated_duration_target_sec": round(candidate_duration, 3),
                "kandidat": candidate_index,
                "path": rel(generated),
                "seed": candidate_seed,
                "base_seed": seed,
                "section_seed": section.seed,
                "seed_strategy": "base_seed + section_index * 1009 + attempt * 37",
                "prompt": candidate_prompt,
                "anschluss_conditioning": bool(previous_clip_path),
                "anschluss_quelle": rel(previous_clip_path) if previous_clip_path else "",
                "anschluss_sekunden": args.anschluss_sekunden if previous_clip_path else "",
                **metrics,
                **semantic,
                **reference_check,
            }
            technical_score = score_kandidat(metrics)
            transition_details = score_uebergang(
                accepted[-1] if accepted else None,
                generated,
                metrics,
                args.uebergang_bpm_toleranz,
            )
            block_details = score_block_konsistenz(section, candidate_prompt, metrics, args)
            row.update(block_details)
            all_checks.append(row)
            write_csv(run_dir / "technische_pruefung.csv", all_checks)
            transition_score = optional_float(transition_details.get("transition_score")) or 0.0
            block_score = optional_float(block_details.get("block_consistency_score")) or 0.0
            semantic_penalty = optional_float(semantic.get("semantic_penalty")) or 0.0
            semantic_target = optional_float(semantic.get("semantic_target_similarity")) or 0.0
            semantic_margin = optional_float(semantic.get("semantic_genre_margin")) or 0.0
            semantic_quality = optional_float(semantic.get("semantic_quality_margin")) or 0.0
            semantic_match = optional_float(semantic.get("semantic_genre_match_ratio")) or 0.0
            semantic_score = (
                semantic_penalty
                - semantic_target * 1.2
                - semantic_margin * 3.0
                - semantic_quality * 2.0
                - semantic_match * 0.4
            )
            reference_score = optional_float(reference_check.get("referenz_score")) or 0.0
            total_score = technical_score + transition_score + block_score + semantic_score + reference_score
            score_row = {
                "block": section.block,
                "abschnitt": section.abschnitt,
                "kandidat": candidate_index,
                "path": rel(generated),
                "status": metrics["status"],
                "auswahlstatus": "NICHT_GEWAEHLT",
                "technical_score": round(technical_score, 4),
                "transition_score": round(transition_score, 4),
                "block_consistency_score": round(block_score, 4),
                "semantic_score": round(semantic_score, 4),
                "referenz_score": round(reference_score, 4),
                "referenz_status": reference_check.get("referenz_status", ""),
                "referenz_gruende": reference_check.get("referenz_gruende", ""),
                "gesamt_score": round(total_score, 4),
                "target_bpm_diff": metrics.get("bpm_diff", ""),
                "rms_db": metrics.get("rms_db", ""),
                "bass_ratio": metrics.get("bass_ratio", ""),
                "snare_ratio": metrics.get("snare_ratio", ""),
                "high_ratio": metrics.get("high_ratio", ""),
                "tone_ratio": metrics.get("tone_ratio", ""),
                "musical_movement_score": metrics.get("musical_movement_score", ""),
                "spectral_movement": metrics.get("spectral_movement", ""),
                "rms_movement_db": metrics.get("rms_movement_db", ""),
                "centroid_movement_hz": metrics.get("centroid_movement_hz", ""),
                "active_second_ratio": metrics.get("active_second_ratio", ""),
                "silent_second_count": metrics.get("silent_second_count", ""),
                "longest_silent_sec": metrics.get("longest_silent_sec", ""),
                "min_second_rms_db": metrics.get("min_second_rms_db", ""),
                "anschluss_conditioning": bool(previous_clip_path),
                "anschluss_quelle": rel(previous_clip_path) if previous_clip_path else "",
                "anschluss_sekunden": args.anschluss_sekunden if previous_clip_path else "",
                "gruende": metrics.get("gruende", ""),
                "warnungen": metrics.get("warnungen", ""),
                "prompt": candidate_prompt,
                **transition_details,
                **block_details,
                **semantic,
                **reference_check,
            }
            candidate_scores.append(score_row)
            write_csv(run_dir / "kandidaten_score.csv", candidate_scores)
            semantic_ok = semantic.get("semantic_status") in {"OK", "DISABLED"}
            reference_ok = reference_check.get("referenz_status") in {"OK", "PRUEFEN", "DISABLED"}
            transition_ok = transition_details.get("transition_status") in {"START", "OK", "PRUEFEN"}
            if metrics["status"] == "OK" and block_score < 8.0 and semantic_ok and reference_ok and transition_ok:
                ok_candidates.append((total_score, row, generated, score_row))
                print(
                    f"  Kandidat {candidate_index}: OK "
                    f"(Score {total_score:.3f}, "
                    f"Technik {technical_score:.3f}, "
                    f"Uebergang {transition_score:.3f}, "
                    f"Block {block_score:.3f}, "
                    f"Referenz {reference_score:.3f}, "
                    f"Genre {semantic.get('semantic_status')} "
                    f"{semantic.get('semantic_genre_margin', '')}, "
                    f"BPM-Wechsel {transition_details.get('transition_bpm_diff', '')}, "
                    f"Bass {metrics.get('bass_ratio', 0):.3f}, "
                    f"Shaker {metrics.get('high_ratio', 0):.3f}, "
                    f"Bewegung {metrics.get('musical_movement_score', 0):.3f})",
                    flush=True,
                )
            elif metrics["status"] == "OK" and not transition_ok:
                rejected.append(
                    {
                        **row,
                        **transition_details,
                        "status": "PROBLEM",
                        "auswahlstatus": "UEBERGANG_VERWORFEN",
                        "gruende": "Uebergangsproblem: "
                        + str(
                            transition_details.get("transition_harte_gruende")
                            or transition_details.get("transition_gruende")
                            or "Anschluss zum vorherigen Clip ist zu hart"
                        ),
                    }
                )
            elif metrics["status"] == "OK" and not reference_ok:
                rejected.append(
                    {
                        **row,
                        "status": "PROBLEM",
                        "gruende": "MP3-Referenzpruefung: "
                        + str(reference_check.get("referenz_gruende", "Kandidat weicht zu stark von Referenz ab")),
                    }
                )
            elif metrics["status"] == "OK" and not semantic_ok:
                rejected.append(
                    {
                        **row,
                        "status": "PROBLEM",
                        "gruende": "Genre-Pruefung: "
                        + str(semantic.get("semantic_reasons", "Genre passt nicht")),
                    }
                )
            elif metrics["status"] == "OK":
                rejected.append(
                    {
                        **row,
                        "status": "PROBLEM",
                        "gruende": "Block-Konsistenzproblem: "
                        + str(block_details.get("block_consistency_warning", "")),
                    }
                )
            elif metrics.get("bpm_fallback_ok") and semantic_ok and reference_ok:
                current_diff = optional_float(metrics.get("bpm_diff")) or 999.0
                previous_diff = (
                    optional_float(best_bpm_fallback_row.get("bpm_diff")) if best_bpm_fallback_row else None
                )
                if best_bpm_fallback_row is None or current_diff < (previous_diff or 999.0):
                    best_bpm_fallback_row = row
                    best_bpm_fallback_path = generated
                rejected.append(row)
            else:
                rejected.append(row)
            cleanup_cuda_cache()
        if ok_candidates:
            ok_candidates.sort(key=lambda item: item[0])
            _, best_row, best_path, best_score_row = ok_candidates[0]
            best_score_row["auswahlstatus"] = "AKZEPTIERT"
            for _, rej_row, _, rej_score_row in ok_candidates[1:]:
                rej_score_row["auswahlstatus"] = "OK_ABER_SCHLECHTERER_SCORE"
                rejected.append({**rej_row, "auswahlstatus": "OK_ABER_SCHLECHTERER_SCORE"})
            accepted_path = run_dir / "clips" / f"abschnitt_{section.abschnitt:04d}.wav"
            accepted_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_path, accepted_path)
            accepted.append(
                {
                    **best_row,
                    "path": rel(accepted_path),
                    "gewaehlter_kandidat": best_row.get("kandidat", ""),
                    "technical_score": best_score_row.get("technical_score", ""),
                    "transition_score": best_score_row.get("transition_score", ""),
                    "block_consistency_score": best_score_row.get("block_consistency_score", ""),
                    "block_consistency_status": best_score_row.get("block_consistency_status", ""),
                    "block_consistency_warning": best_score_row.get("block_consistency_warning", ""),
                    "block_prompt_basis": best_score_row.get("block_prompt_basis", ""),
                    "lofi_style_score": best_score_row.get("lofi_style_score", ""),
                    "transition_score_raw": best_score_row.get("transition_score_raw", ""),
                    "transition_score_weight": best_score_row.get("transition_score_weight", ""),
                    "transition_acceptance_limit": best_score_row.get("transition_acceptance_limit", ""),
                    "gesamt_score": best_score_row.get("gesamt_score", ""),
                    "transition_status": best_score_row.get("transition_status", ""),
                    "transition_gruende": best_score_row.get("transition_gruende", ""),
                    "transition_warning": best_score_row.get("transition_warning", ""),
                    "transition_harte_gruende": best_score_row.get("transition_harte_gruende", ""),
                    "transition_bpm_diff": best_score_row.get("transition_bpm_diff", ""),
                    "transition_bpm_status": best_score_row.get("transition_bpm_status", ""),
                    "previous_bpm": best_score_row.get("previous_bpm", ""),
                    "candidate_bpm": best_score_row.get("candidate_bpm", ""),
                    "transition_bpm_tolerance": best_score_row.get("transition_bpm_tolerance", ""),
                    "loudness_jump_db": best_score_row.get("loudness_jump_db", ""),
                    "bass_jump": best_score_row.get("bass_jump", ""),
                    "snare_jump": best_score_row.get("snare_jump", ""),
                    "high_jump": best_score_row.get("high_jump", ""),
                    "harmonic_similarity": best_score_row.get("harmonic_similarity", ""),
                    "harmonic_distance": best_score_row.get("harmonic_distance", ""),
                    "rhythm_correlation": best_score_row.get("rhythm_correlation", ""),
                    "click_risk": best_score_row.get("click_risk", ""),
                    "start_silence_db": best_score_row.get("start_silence_db", ""),
                    "previous_end_cut_db": best_score_row.get("previous_end_cut_db", ""),
                    "candidate_end_cut_db": best_score_row.get("candidate_end_cut_db", ""),
                    "energy_jump_db": best_score_row.get("energy_jump_db", ""),
                }
            )
            accepted_sections.add(section.abschnitt)
            write_csv(run_dir / "akzeptierte_clips.csv", accepted)
            write_csv(run_dir / "technische_pruefung.csv", all_checks)
            write_csv(run_dir / "rejected_clips.csv", rejected)
            write_csv(run_dir / "kandidaten_score.csv", candidate_scores)
            progress_after = len(accepted) / max(1, target_sections) * 100.0
            write_progress(
                run_dir,
                status="running",
                section_index=len(accepted),
                total_sections=target_sections,
                candidate_index=args.kandidaten_pro_abschnitt,
                total_candidates=args.kandidaten_pro_abschnitt,
                accepted_count=len(accepted),
                rejected_count=len(rejected),
                message=(
                    f"Abschnitt {section.abschnitt} akzeptiert "
                    f"(bester von {len(ok_candidates)} OK-Kandidaten). "
                    f"{len(accepted)}/{target_sections} gute Clips."
                ),
            )
            print(
                f"Fortschritt {progress_after:.1f}% | guter Clip "
                f"{len(accepted)}/{target_sections} akzeptiert "
                f"(Abschnitt {section.abschnitt}, bester von {len(ok_candidates)} OK-Kandidaten, "
                f"BPM {best_row.get('estimated_bpm') or 'unbekannt'}, "
                f"Score {best_score_row.get('gesamt_score')})",
                flush=True,
            )
        if accepted_path is None and best_bpm_fallback_row is not None and best_bpm_fallback_path is not None:
            accepted_path = run_dir / "clips" / f"abschnitt_{section.abschnitt:04d}.wav"
            accepted_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_bpm_fallback_path, accepted_path)
            accepted.append(
                {
                    **best_bpm_fallback_row,
                    "path": rel(accepted_path),
                    "status": "OK_MIT_BPM_WARNUNG",
                    "bpm_warnung_akzeptiert": True,
                }
            )
            accepted_sections.add(section.abschnitt)
            write_csv(run_dir / "akzeptierte_clips.csv", accepted)
            write_csv(run_dir / "technische_pruefung.csv", all_checks)
            write_csv(run_dir / "rejected_clips.csv", rejected)
            progress_after = len(accepted) / max(1, target_sections) * 100.0
            write_progress(
                run_dir,
                status="running",
                section_index=len(accepted),
                total_sections=target_sections,
                accepted_count=len(accepted),
                rejected_count=len(rejected),
                message=(
                    f"Abschnitt {section.abschnitt} mit BPM-Warnung akzeptiert. "
                    f"{len(accepted)}/{target_sections} gute Clips."
                ),
            )
            print(
                f"Fortschritt {progress_after:.1f}% | Abschnitt {section.abschnitt} "
                "mit BPM-Warnung akzeptiert",
                flush=True,
            )
        if accepted_path is None and args.fallback_pool:
            while fallback_index < len(fallback_files):
                source = fallback_files[fallback_index]
                fallback_index += 1
                fallback_path = run_dir / "clips" / f"abschnitt_{section.abschnitt:04d}_fallback.wav"
                convert_wav(source, fallback_path)
                metrics = audio_metrics(
                    fallback_path,
                    args.abschnitt_sekunden,
                    args.ziel_bpm,
                    args.bpm_toleranz,
                    args.bpm_pruefung_aktiv,
                    args.sekunden_pruefung_aktiv,
                    args.min_sekunden_rms_db,
                )
                row = {
                    "block": section.block,
                    "transition_group": f"block_{section.block:03d}",
                    "abschnitt": section.abschnitt,
                    "start_sec": section.start_sec,
                    "end_sec": section.end_sec,
                    "kandidat": "fallback",
                    "path": rel(fallback_path),
                    "quelle_alt": rel(source),
                    "seed": section_seed(seed, section.abschnitt, 99 + fallback_index),
                    "base_seed": seed,
                    "section_seed": section.seed,
                    "prompt": section.prompt,
                    **metrics,
                }
                all_checks.append(row)
                if metrics["status"] == "OK":
                    accepted.append(row)
                    accepted_sections.add(section.abschnitt)
                    accepted_path = fallback_path
                    write_csv(run_dir / "akzeptierte_clips.csv", accepted)
                    write_csv(run_dir / "technische_pruefung.csv", all_checks)
                    write_csv(run_dir / "rejected_clips.csv", rejected)
                    break
                rejected.append(row)
        if accepted_path is None:
            skipped_sections.append(
                {
                    "abschnitt": section.abschnitt,
                    "block": section.block,
                    "start_sec": section.start_sec,
                    "end_sec": section.end_sec,
                    "prompt": section.prompt,
                    "seed": section.seed,
                    "grund": "Kein Kandidat hat die technische Pruefung bestanden.",
                }
            )
            write_csv(run_dir / "technische_pruefung.csv", all_checks)
            write_csv(run_dir / "rejected_clips.csv", rejected)
            write_csv(run_dir / "kandidaten_score.csv", candidate_scores)
            write_csv(run_dir / "uebersprungene_abschnitte.csv", skipped_sections)
            write_progress(
                run_dir,
                status="running",
                section_index=len(accepted),
                total_sections=target_sections,
                accepted_count=len(accepted),
                rejected_count=len(rejected),
                message=(
                    f"Abschnitt {section.abschnitt} uebersprungen. "
                    "Es wird ein neuer Abschnitt geplant."
                ),
            )
            print(
                f"Abschnitt {section.abschnitt} uebersprungen: "
                "kein brauchbarer Kandidat. Neuer Abschnitt wird spaeter erzeugt.",
                flush=True,
            )
            continue

    write_csv(run_dir / "technische_pruefung.csv", all_checks)
    write_csv(run_dir / "rejected_clips.csv", rejected)
    write_csv(run_dir / "kandidaten_score.csv", candidate_scores)
    write_csv(run_dir / "uebersprungene_abschnitte.csv", skipped_sections)
    ordered_accepted = order_accepted_clips(accepted, args.ziel_bpm)
    write_csv(run_dir / "akzeptierte_clips.csv", ordered_accepted)
    wav_path = build_longform(ordered_accepted, args, run_dir, duration_sec)
    audio_path = run_dir / "lange_audio.mp3" if not args.kein_mp3 else wav_path
    final_semantic_check: Dict[str, Any] = {"semantic_status": "DISABLED"}
    if genre_pruefer is not None:
        try:
            final_semantic_check = genre_pruefer.pruefe(wav_path, args.genre)
        finally:
            genre_pruefer.schliessen()
    rating_path = write_rating_template(audio_path, duration_sec)
    auto_rating: Dict[str, Any]
    try:
        from audio_bewertung import bewerte_audio_datei

        rating_section_sec = 60.0 if duration_sec <= 600.0 else 300.0
        auto_rating = bewerte_audio_datei(
            audio_path,
            run_dir,
            abschnitt_sekunden=rating_section_sec,
            bewertung_csv_path=rating_path,
        )
    except Exception as exc:
        auto_rating = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "hinweis": "Automatische technische Bewertung konnte nicht erstellt werden.",
        }
        write_json(run_dir / "automatische_bewertung_fehler.json", auto_rating)

    score_rating: Dict[str, Any]
    try:
        from audio_bewertung import bewerte_audio_mit_genrestandard

        score_rating = bewerte_audio_mit_genrestandard(
            audio_path,
            args.genre,
            run_dir,
            Path(args.referenz_dataset_root).expanduser(),
            int(args.referenzen_pro_genre),
        )
    except Exception as exc:
        score_rating = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "hinweis": "Fuenf-Score-Bewertung konnte nicht erstellt werden.",
        }
        write_json(run_dir / "score_bewertung_fehler.json", score_rating)

    timeline: List[Dict[str, Any]] = []
    assembled_end = 0.0
    for index, row in enumerate(ordered_accepted):
        block_duration = (
            row_block_duration(row, args.abschnitt_sekunden)
            if nutzt_einen_clip_pro_block(args)
            else args.abschnitt_sekunden
        )
        start = 0.0 if index == 0 else max(0.0, assembled_end - crossfade_seconds)
        end = min(duration_sec, start + block_duration)
        timeline.append(
            {
                "block": row["block"],
                "abschnitt": row["abschnitt"],
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "datei": row["path"],
                "prompt": row["prompt"],
                "seed": row["seed"],
                "base_seed": row.get("base_seed", seed),
                "section_seed": row.get("section_seed", ""),
                "estimated_bpm": row.get("estimated_bpm", ""),
                "bpm_diff": row.get("bpm_diff", ""),
                "bpm_status": row.get("bpm_status", ""),
                "rms_db": row.get("rms_db", ""),
                "bass_ratio": row.get("bass_ratio", ""),
                "high_ratio": row.get("high_ratio", ""),
                "active_second_ratio": row.get("active_second_ratio", ""),
                "silent_second_count": row.get("silent_second_count", ""),
                "min_second_rms_db": row.get("min_second_rms_db", ""),
                "block_consistency_score": row.get("block_consistency_score", ""),
                "block_consistency_status": row.get("block_consistency_status", ""),
                "block_consistency_warning": row.get("block_consistency_warning", ""),
                "block_prompt_basis": row.get("block_prompt_basis", row.get("prompt", "")),
                "lofi_style_score": row.get("lofi_style_score", ""),
                "semantic_status": row.get("semantic_status", ""),
                "semantic_target_similarity": row.get("semantic_target_similarity", ""),
                "semantic_genre_margin": row.get("semantic_genre_margin", ""),
                "semantic_quality_margin": row.get("semantic_quality_margin", ""),
                "semantic_genre_match_ratio": row.get("semantic_genre_match_ratio", ""),
                "referenz_status": row.get("referenz_status", ""),
                "referenz_score": row.get("referenz_score", ""),
                "referenz_gruende": row.get("referenz_gruende", ""),
                "referenz_genre": row.get("referenz_genre", ""),
                "transition_score": row.get("transition_score", ""),
                "transition_status": row.get("transition_status", ""),
                "transition_gruende": row.get("transition_gruende", ""),
                "transition_harte_gruende": row.get("transition_harte_gruende", ""),
                "transition_bpm_diff": row.get("transition_bpm_diff", ""),
                "loudness_jump_db": row.get("loudness_jump_db", ""),
                "bass_jump": row.get("bass_jump", ""),
                "snare_jump": row.get("snare_jump", ""),
                "high_jump": row.get("high_jump", ""),
                "energy_jump_db": row.get("energy_jump_db", ""),
                "harmonic_similarity": row.get("harmonic_similarity", ""),
                "harmonic_distance": row.get("harmonic_distance", ""),
                "rhythm_correlation": row.get("rhythm_correlation", ""),
                "click_risk": row.get("click_risk", ""),
                "start_silence_db": row.get("start_silence_db", ""),
                "previous_end_cut_db": row.get("previous_end_cut_db", ""),
                "candidate_end_cut_db": row.get("candidate_end_cut_db", ""),
                "crossfade_seconds": 0.0 if index == 0 else crossfade_seconds,
                "loop_crossfade_seconds": row.get("loop_crossfade_sec", 0.0),
                "loop_start_seconds": row.get("loop_start_sec", ""),
                "loop_end_seconds": row.get("loop_end_sec", ""),
                "loop_duration_seconds": row.get("loop_duration_sec", ""),
                "loop_bpm": row.get("loop_bpm", ""),
                "loop_seam_score": row.get("loop_seam_score", ""),
                "loop_status": row.get("loop_status", ""),
                "loop_warning": row.get("loop_warnung", ""),
                "loop_uses": row.get("loop_verwendungen", ""),
                "tempo_variation_percent": row.get("tempo_variation_percent", ""),
                "tempo_phase_seconds": row.get("tempo_phase_sec", ""),
                "tempo_min_factor": row.get("tempo_min_factor", ""),
                "tempo_max_factor": row.get("tempo_max_factor", ""),
                "tempo_factors": row.get("tempo_factors", ""),
                "block_looping": args.block_looping_aktiv,
                "continuous_musicgen_block": args.kontinuierliche_bloecke_aktiv,
                "block_duration_sec": round(block_duration, 3),
                "transition_type": (
                    "start"
                    if index == 0
                    else (
                        "looped_block_crossfade"
                        if args.block_looping_aktiv
                        else (
                            "continuous_musicgen_block_crossfade"
                            if args.kontinuierliche_bloecke_aktiv
                            else "fresh_clip_score_crossfade"
                        )
                    )
                ),
                "beat_sync_used": bool(args.block_looping_aktiv and row.get("loop_bpm")),
            }
        )
        assembled_end = start + block_duration
    write_csv(run_dir / "ablauf.csv", timeline)
    transition_report = write_transition_quality_report(
        wav_path,
        timeline,
        run_dir,
        crossfade_seconds,
        args.uebergang_bpm_toleranz,
    )
    write_csv(run_dir / "technische_pruefung.csv", all_checks)
    write_csv(run_dir / "rejected_clips.csv", rejected)
    write_csv(run_dir / "uebersprungene_abschnitte.csv", skipped_sections)
    github_result: Dict[str, Any] = {"status": "disabled"}
    if args.github_push:
        try:
            from audio_veroeffentlichen import veroeffentliche_audio

            github_result = veroeffentliche_audio(
                audio_path,
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

    final_info = {
        "status": "finished",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_source": model_source,
        "adapter_path": rel(adapter_path),
        "adapter_step": adapter_payload.get("step"),
        "model_strategy": {
            "base_model": args.model_id,
            "model_source": str(model_source),
            "local_model_dir": rel(Path(args.model_dir).expanduser()),
            "allow_remote_model": bool(args.allow_remote_model),
            "adapter_path": rel(adapter_path),
            "adapter_step": adapter_payload.get("step"),
            "adapter_run": adapter_path.parent.name,
            "reference_checkpoint": f"step_{int(adapter_payload.get('step') or 0):06d}",
            "anschluss_conditioning_enabled": args.anschluss_conditioning_aktiv,
            "melody_conditioning_usage": (
                "Anfang des naechsten Clips wird bei vorhandener Vorclip-Datei mit Melody-Referenz konditioniert."
                if args.anschluss_conditioning_aktiv
                else "Text-only Generierung ohne Anschluss-Conditioning."
            ),
        },
        "optional_quality_helpers": optional_quality_helpers_report(),
        "semantic_genre_check": genre_pruefung_report,
        "final_semantic_check": final_semantic_check,
        "mp3_reference_check": mp3_reference_report,
        "genre": args.genre,
        "duration_sec": duration_sec,
        "base_seed": seed,
        "seed_strategy": "segment_seed = base_seed + section_index * 1009 + attempt * 37",
        "target_bpm": args.ziel_bpm,
        "bpm_tolerance": args.bpm_toleranz,
        "transition_bpm_tolerance": args.uebergang_bpm_toleranz,
        "bpm_filter_enabled": args.bpm_pruefung_aktiv,
        "second_by_second_quality_check_enabled": args.sekunden_pruefung_aktiv,
        "min_second_rms_db_threshold": args.min_sekunden_rms_db,
        "generation_sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "cfg_coef": args.cfg_coef,
        },
        "style_prompt_profile": genre_prompt_profile(args.genre),
        "active_second_ratio_summary": summarize_optional(row.get("active_second_ratio") for row in ordered_accepted),
        "silent_second_count_total": sum(int(optional_float(row.get("silent_second_count")) or 0) for row in ordered_accepted),
        "min_second_rms_db_summary": summarize_optional(row.get("min_second_rms_db") for row in ordered_accepted),
        "semantic_target_similarity_summary": summarize_optional(
            row.get("semantic_target_similarity") for row in ordered_accepted
        ),
        "semantic_genre_margin_summary": summarize_optional(
            row.get("semantic_genre_margin") for row in ordered_accepted
        ),
        "semantic_quality_margin_summary": summarize_optional(
            row.get("semantic_quality_margin") for row in ordered_accepted
        ),
        "referenz_score_summary": summarize_optional(row.get("referenz_score") for row in ordered_accepted),
        "referenz_problem_clips": sum(
            1 for row in ordered_accepted if row.get("referenz_status") == "PROBLEM"
        ),
        "bpm_summary": summarize_optional(row.get("estimated_bpm") for row in ordered_accepted),
        "bpm_diff_summary": summarize_optional(row.get("bpm_diff") for row in ordered_accepted),
        "transition_strategy": (
            "konstantes Ziel-BPM, frische Clip-Kandidaten, technischer Score, "
            "hart gewichteter Uebergangs-Score, strenge BPM-Aehnlichkeit "
            "zwischen benachbarten Clips, Lautheit, Bass, Snare/Praesenz, Hoehen, "
            "Energie, harmonische/timbrale Naehe, Rhythmuskorrelation, Dropouts, "
            "Anschnitte und Klickrisiko, kurzer effektiver Crossfade, optionales "
            "Anschluss-Conditioning"
        ),
        "anschluss_conditioning_enabled": args.anschluss_conditioning_aktiv,
        "anschluss_reference_seconds": args.anschluss_sekunden if args.anschluss_conditioning_aktiv else 0.0,
        "transition_score_weight": TRANSITION_SCORE_WEIGHT,
        "requested_crossfade_seconds": args.crossfade_sekunden,
        "crossfade_seconds": crossfade_seconds,
        "fade_in_seconds": args.fade_in_sekunden,
        "fade_out_seconds": args.fade_out_sekunden,
        "internal_crossfade_seconds": args.interner_crossfade_sekunden,
        "block_looping_enabled": args.block_looping_aktiv,
        "continuous_musicgen_blocks_enabled": args.kontinuierliche_bloecke_aktiv,
        "musicgen_extend_stride_seconds": args.erweiterungs_schritt_sekunden,
        "rhythm_block_mode": "percentage" if args.blockmodus_prozentual else "fixed",
        "rhythm_share_percent": args.rhythmus_anteil_prozent,
        "minimum_rhythm_seconds": args.min_rhythmus_sekunden,
        "rhythm_target_seconds": args.block_sekunden,
        "rhythm_block_seconds": rhythm_summary["mean_seconds"],
        "rhythm_block_count": rhythm_summary["count"],
        "rhythm_block_min_seconds": rhythm_summary["min_seconds"],
        "rhythm_block_max_seconds": rhythm_summary["max_seconds"],
        "loop_crossfade_requested_seconds": args.loop_crossfade_sekunden,
        "loop_crossfade_strategy": "0 = automatisch ein Beat, maximal 1.25 Sekunden",
        "tempo_variation_percent": args.tempo_variation_prozent,
        "tempo_phase_seconds": args.tempo_phase_sekunden,
        "tempo_variation_strategy": "pitch-erhaltend und schrittweise",
        "block_seconds": args.block_sekunden,
        "block_variation_seconds": args.block_variation_sekunden,
        "beat_sync_enabled_or_false": bool(args.block_looping_aktiv),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "accepted_clips": len(ordered_accepted),
        "rejected_clips": len(rejected),
        "skipped_sections": len(skipped_sections),
        "planned_sections_total": len(planned),
        "generated_candidate_count": generated_candidate_count,
        "max_generated_candidates": max_generated_candidates or None,
        "sections_minimum_for_crossfade": needed_sections,
        "estimated_assembled_duration_sec": round(estimated_duration, 3),
        "fallback_pool_enabled": args.fallback_pool,
        "memory_efficient_attention_disabled": True,
        "fallback_pool_size": len(fallback_files),
        "fallback_clips_used": sum(1 for row in ordered_accepted if row.get("kandidat") == "fallback"),
        "bpm_warning_clips_used": sum(1 for row in ordered_accepted if row.get("bpm_warnung_akzeptiert")),
        "output_wav": rel(wav_path),
        "output_mp3": rel(run_dir / "lange_audio.mp3") if not args.kein_mp3 else None,
        "github_push": github_result,
        "rating_template": rel(rating_path),
        "automatic_rating": auto_rating,
        "score_rating": score_rating,
        "transition_quality_report": transition_report,
        "candidate_score_report": rel(run_dir / "kandidaten_score.csv"),
        "vram_limit": vram_limit,
    }
    scientific_report = write_scientific_audio_report(
        run_dir,
        final_info,
        ordered_accepted,
        rejected,
        skipped_sections,
        transition_report,
    )
    final_info["scientific_report"] = rel(scientific_report)
    write_json(run_dir / "finale_audio_info.json", final_info)
    write_json(run_dir / "generation_report.json", {**final_info, "used_prompts": rel(run_dir / "verwendete_prompts.csv")})
    write_progress(
        run_dir,
        status="finished",
        section_index=len(accepted),
        total_sections=target_sections,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        message="Longform-Audio fertig erzeugt.",
    )
    print("Audio fertig", flush=True)
    print(f"Audio: {rel(audio_path)}", flush=True)
    if score_rating.get("status") == "finished":
        print(f"Bewertung: {score_rating.get('html')}", flush=True)
    return final_info


def main() -> int:
    fuehre_zufaellige_musicgen_generierung_aus(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
