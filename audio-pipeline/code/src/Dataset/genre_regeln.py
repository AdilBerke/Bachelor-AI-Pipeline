#!/usr/bin/env python3

from __future__ import annotations

import random
import re
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


STANDARD_GENRES = (
    "jazz_lofi",
    "chillhop_lofi",
    "dreamy_lofi",
    "study_lofi",
    "guitar_lofi",
)

QUELLEN_REGEL_VERSION = "lofi_quellen_v4_guitar_breiter_2026_08_11"

GENRE_LABELS = {
    "jazz_lofi": "Jazz Lofi",
    "chillhop_lofi": "Chillhop Lofi",
    "dreamy_lofi": "Dreamy Lofi",
    "study_lofi": "Study Lofi",
    "guitar_lofi": "Guitar Lofi",
}

GENRE_CAPTIONS = {
    "jazz_lofi": (
        "jazz lofi instrumental, warm rhodes or piano chords, mellow bass, "
        "soft brushed drums, relaxed groove, no vocals"
    ),
    "chillhop_lofi": (
        "chillhop lofi instrumental, smooth drums, warm bass, mellow chords, "
        "relaxed head nod groove, no vocals"
    ),
    "dreamy_lofi": (
        "dreamy lofi instrumental, soft warm texture, mellow piano or pads, "
        "gentle drums, relaxed mood, no vocals"
    ),
    "study_lofi": (
        "study lofi instrumental, calm focus mood, soft drums, warm bass, "
        "mellow chords, stable groove, no vocals"
    ),
    "guitar_lofi": (
        "guitar lofi instrumental, mellow guitar accents, warm chords, "
        "soft drums, controlled bass, no vocals"
    ),
}

GENRE_PREFIXE = {
    "jazz_lofi": ("jazz lofi", "jazzlofi"),
    "chillhop_lofi": ("chillhop lofi", "chill lofi", "chillhop"),
    "dreamy_lofi": ("dreamy lofi", "dream lofi"),
    "study_lofi": ("study lofi", "focus lofi"),
    "guitar_lofi": ("guitar lofi", "gitarre lofi"),
}

GENRE_KEYWORDS = {
    "jazz_lofi": ("jazz lofi", "jazzhop", "jazzy", "rhodes", "saxophone", "sax"),
    "guitar_lofi": (
        "guitar lofi",
        "lofi guitar",
        "guitar vibe",
        "chill guitar beats",
        "acoustic lofi",
        "lofi guitar beats",
        "gitarre",
        "acoustic guitar",
        "guitar",
    ),
    "dreamy_lofi": ("dreamy lofi", "dreamy", "night lofi", "ambient lofi"),
    "study_lofi": ("study lofi", "study", "focus", "coding lofi", "work lofi"),
    "chillhop_lofi": ("chillhop", "chill lofi", "lofi hip hop", "chill beat"),
}

LOFI_ANCHOR_TERMS = (
    "lofi",
    "lo fi",
    "lowfi",
    "chillhop",
)

SOURCE_BLOCK_TERMS = (
    "math rock",
    "neo soul",
    "r b",
    "rnb",
    "cover",
    "covers",
    "popular songs",
    "pop songs",
    "party",
    "edm",
    "house",
    "techno",
    "dubstep",
    "trap",
    "country",
    "healing guitar",
    "healing music",
    "chillstep",
    "vlog",
    "vlogs",
    "tutorial",
    "reaction",
    "podcast",
    "interview",
    "speech",
    "spoken",
    "spoken word",
    "talk",
    "talking",
    "commentary",
    "narration",
    "narrator",
    "voice over",
    "voiceover",
    "vocal",
    "vocals",
    "lyrics",
    "lyric",
    "singing",
    "singer",
    "rap",
    "rapper",
    "freestyle",
    "acapella",
    "a cappella",
    "karaoke",
    "reaction",
    "review",
    "lesson",
    "explained",
    "subscribe",
    "announcement",
    "live performance",
    "music video",
    "michael jackson",
    "christian",
    "worship",
    "gospel",
    "trip hop",
    "downtempo",
    "dub",
    "acid jazz",
    "future jazz",
    "abstract beats",
)

GENRE_REQUIRED_TERMS = {
    "jazz_lofi": ("jazz", "jazzhop", "jazzy", "rhodes", "sax", "trumpet"),
    "chillhop_lofi": ("chillhop", "lofi hip hop", "hip hop", "hiphop", "beat", "beats", "groove"),
    "dreamy_lofi": ("dreamy", "dream", "night", "calm", "soft", "warm", "relax", "cute", "ambient lofi"),
    "study_lofi": ("study", "focus", "work", "working", "relax", "coding"),
    "guitar_lofi": ("guitar", "gitarre", "acoustic", "strings"),
}

GENRE_BLOCK_TERMS = {
    "jazz_lofi": (
        "oud",
        "arabic",
        "oriental",
        "world music",
        "bossa",
        "bossa nova",
        "brazil",
        "brazilian",
        "samba",
        "smooth jazz",
        "jazz cafe",
        "cafe jazz",
        "coffee jazz",
        "relaxing jazz",
        "sleep jazz",
        "night jazz",
        "dinner jazz",
        "jazz bgm",
        "piano jazz",
        "jazz piano music",
    ),
    "guitar_lofi": (
        "country",
        "healing guitar",
        "healing music",
        "flamenco",
        "rock guitar",
        "guitar solo",
    ),
    "dreamy_lofi": (
        "ambient only",
        "rain sounds",
        "nature sounds",
        "thunderstorm",
        "fireplace",
    ),
}


def _genre_config_path() -> Path:

    return Path(__file__).resolve().parents[3] / "code" / "configs" / "lora_genres.json"


def _str_liste(value: Any) -> tuple[str, ...]:

    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _lade_genre_config() -> None:

    global STANDARD_GENRES, GENRE_LABELS, GENRE_CAPTIONS, GENRE_PREFIXE, GENRE_KEYWORDS, GENRE_REQUIRED_TERMS, GENRE_BLOCK_TERMS
    path = _genre_config_path()
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    raw_genres = payload.get("genres") if isinstance(payload, dict) else None
    if not isinstance(raw_genres, dict) or not raw_genres:
        return

    labels: dict[str, str] = {}
    captions: dict[str, str] = {}
    prefixes: dict[str, tuple[str, ...]] = {}
    keywords: dict[str, tuple[str, ...]] = {}
    required: dict[str, tuple[str, ...]] = {}
    block_terms: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_info in raw_genres.items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        info = raw_info if isinstance(raw_info, dict) else {}
        label = str(info.get("label") or key.replace("_", " ").title()).strip()
        caption = str(info.get("caption") or "calm lofi instrumental, no vocals").strip()
        labels[key] = label
        captions[key] = caption
        prefixes[key] = _str_liste(info.get("prefixes")) or (label.lower().replace("-", " "),)
        keywords[key] = _str_liste(info.get("keywords")) or prefixes[key]
        required[key] = _str_liste(info.get("required_terms")) or keywords[key]
        block_terms[key] = _str_liste(info.get("block_terms")) or GENRE_BLOCK_TERMS.get(key, ())

    if labels:
        STANDARD_GENRES = tuple(labels)
        GENRE_LABELS = labels
        GENRE_CAPTIONS = captions
        GENRE_PREFIXE = prefixes
        GENRE_KEYWORDS = keywords
        GENRE_REQUIRED_TERMS = required
        GENRE_BLOCK_TERMS = block_terms


_lade_genre_config()


def normalisiere(text: Any) -> str:

    value = str(text or "").lower()
    value = value.replace("lo-fi", "lofi")
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def quellentext(row: dict[str, Any]) -> str:

    parts: list[str] = []
    for key in (
        "source_title",
        "source_file",
        "source_audio_path",
        "title",
        "titel",
        "caption",
        "text",
        "description",
        "path",
        "name",
    ):
        value = row.get(key)
        if value:
            parts.append(str(value))
    return normalisiere(" ".join(parts))


def _block_prueftext(text: str) -> str:

    result = f" {text} "
    for phrase in (
        " no vocals ",
        " no vocal ",
        " without vocals ",
        " without vocal ",
        " non vocal ",
        " instrumental vocals free ",
        " vocals free ",
        " vocal free ",
    ):
        result = result.replace(phrase, " ")
    return re.sub(r"\s+", " ", result).strip()


def _enthaelt_begriff(text: str, term: str) -> bool:

    term_norm = normalisiere(term)
    if not term_norm:
        return False
    pattern = r"\b" + re.escape(term_norm).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, text) is not None


def quelle_hat_lofi_bezug(row: dict[str, Any]) -> tuple[bool, str]:

    text = quellentext(row)
    if not text:
        return False, "quellentext_fehlt"
    if not any(term in text for term in LOFI_ANCHOR_TERMS):
        return False, "kein_lofi_bezug"
    block_text = _block_prueftext(text)
    for term in SOURCE_BLOCK_TERMS:
        if _enthaelt_begriff(block_text, term):
            return False, f"falscher_stil:{term}"
    return True, ""


def quelle_passt_zum_genre(row: dict[str, Any], genre: str | None = None) -> tuple[bool, str]:

    ok, reason = quelle_hat_lofi_bezug(row)
    if not ok:
        return False, reason
    genre_key = genre or erkenne_genre(row)
    text = quellentext(row)
    for term in GENRE_BLOCK_TERMS.get(str(genre_key), ()):
        if _enthaelt_begriff(text, term):
            return False, f"falsches_untergenre:{genre_key}:{term}"
    required = GENRE_REQUIRED_TERMS.get(str(genre_key), ())
    if required and not any(term in text for term in required):
        return False, f"genrebezug_fehlt:{genre_key}"
    return True, ""


def _direktes_genre(row: dict[str, Any]) -> str:

    for key in ("lora_genre", "primary_genre"):
        value = normalisiere(row.get(key)).replace(" ", "_")
        if value in STANDARD_GENRES:
            return value

    value = normalisiere(row.get("genre")).replace(" ", "_")
    if value in STANDARD_GENRES:
        return value
    return ""


def _basename_texte(row: dict[str, Any]) -> list[str]:

    values: list[str] = []
    for key in ("source_audio_path", "source_mp3", "source_file", "path", "name"):
        raw = str(row.get(key) or "").strip()
        if raw:
            values.append(normalisiere(Path(raw).name))
    return values


def _prefix_genre(text: str) -> str:

    for genre, prefixes in GENRE_PREFIXE.items():
        if any(text == prefix or text.startswith(prefix + " ") for prefix in prefixes):
            return genre
    return ""


def erkenne_genre(row: dict[str, Any]) -> str:

    direct = _direktes_genre(row)
    if direct:
        return direct

    for text in _basename_texte(row):
        detected = _prefix_genre(text)
        if detected:
            return detected

    for key in ("caption", "text", "description", "source_title"):
        detected = _prefix_genre(normalisiere(row.get(key)))
        if detected:
            return detected

    fallback_parts = []
    for key in ("caption", "text", "description", "source_title"):
        value = row.get(key)
        if isinstance(value, str):
            fallback_parts.append(value)
    fallback = normalisiere(" ".join(fallback_parts))
    for genre in STANDARD_GENRES:
        if any(token in fallback for token in GENRE_KEYWORDS[genre]):
            return genre
    return "general_lofi"


def quellen_schluessel(row: dict[str, Any], projektwurzel: Path | None = None) -> str:

    raw_audio = row.get("source_audio_path") or row.get("source_mp3")
    if raw_audio:
        path = Path(str(raw_audio)).expanduser()
        if not path.is_absolute() and projektwurzel is not None:
            path = projektwurzel / path
        try:
            return str(path.resolve())
        except OSError:
            return str(path)

    value = (
        row.get("source_id")
        or row.get("source_key")
        or row.get("source_video_id")
        or row.get("source_file")
        or row.get("source_url")
        or row.get("path")
        or "unknown"
    )
    return str(value)


def _ziel_je_split(target_count: int) -> dict[str, int]:

    valid = max(1, round(target_count * 0.05)) if target_count >= 20 else 0
    test = max(1, round(target_count * 0.05)) if target_count >= 20 else 0
    return {"train": max(0, target_count - valid - test), "valid": valid, "test": test}


def _round_robin(
    per_source: dict[str, list[dict[str, Any]]],
    sources: list[str],
    target: int,
    max_per_source: int,
    counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:

    selected: list[dict[str, Any]] = []
    counts = counts if counts is not None else {source: 0 for source in sources}
    for source in sources:
        counts.setdefault(source, 0)
    while len(selected) < target:
        progress = False
        for source in sources:
            if len(selected) >= target:
                break
            if counts[source] >= max_per_source or not per_source[source]:
                continue
            selected.append(per_source[source].pop())
            counts[source] += 1
            progress = True
        if not progress:
            break
    return selected


def waehle_quellengetrennt(
    rows: list[dict[str, Any]],
    *,
    genre: str,
    target_count: int,
    max_per_source: int,
    seed: int,
    projektwurzel: Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:

    rng = random.Random(seed)
    per_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if erkenne_genre(row) == genre:
            per_source[quellen_schluessel(row, projektwurzel)].append(row)

    for items in per_source.values():
        rng.shuffle(items)
    sources = sorted(per_source)
    rng.shuffle(sources)
    targets = _ziel_je_split(target_count)
    capacities = {
        source: min(len(items), max_per_source)
        for source, items in per_source.items()
    }
    available_clips = sum(len(items) for items in per_source.values())

    best_pair: tuple[str, str] | None = None
    best_score: tuple[int, int, int, int] | None = None
    for valid_source in sources:
        for test_source in sources:
            if valid_source == test_source:
                continue
            train_sources = [
                source for source in sources if source not in {valid_source, test_source}
            ]
            train_available = sum(capacities[source] for source in train_sources)
            valid_available = capacities[valid_source]
            test_available = capacities[test_source]
            train_selected = min(targets["train"], train_available)
            valid_selected = min(targets["valid"], valid_available)
            test_selected = min(targets["test"], test_available)
            total_selected = train_selected + valid_selected + test_selected
            complete = int(
                train_selected == targets["train"]
                and valid_selected == targets["valid"]
                and test_selected == targets["test"]
            )
            reserve_waste = max(0, valid_available - targets["valid"]) + max(
                0, test_available - targets["test"]
            )
            score = (complete, total_selected, len(train_sources), -reserve_waste)
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (valid_source, test_source)

    result = {"train": [], "valid": [], "test": []}
    split_sources: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    split_counts: dict[str, dict[str, int]] = {"train": {}, "valid": {}, "test": {}}
    if best_pair is not None:
        valid_source, test_source = best_pair
        train_sources = [
            source for source in sources if source not in {valid_source, test_source}
        ]
        result["train"] = _round_robin(
            per_source, train_sources, targets["train"], max_per_source, split_counts["train"]
        )
        result["valid"] = _round_robin(
            per_source, [valid_source], targets["valid"], max_per_source, split_counts["valid"]
        )
        result["test"] = _round_robin(
            per_source, [test_source], targets["test"], max_per_source, split_counts["test"]
        )
        split_sources = {
            "train": sorted(train_sources),
            "valid": [valid_source],
            "test": [test_source],
        }
    else:
        result["train"] = _round_robin(
            per_source, sources, targets["train"], max_per_source, split_counts["train"]
        )
        split_sources["train"] = sorted(sources)


    backfill_by_split = {"train": 0, "valid": 0, "test": 0}
    while sum(len(items) for items in result.values()) < target_count:
        progress = False
        for split in ("train", "valid", "test"):
            remaining = target_count - sum(len(items) for items in result.values())
            if remaining <= 0:
                break
            extra = _round_robin(
                per_source,
                split_sources[split],
                remaining,
                max_per_source,
                split_counts[split],
            )
            if extra:
                result[split].extend(extra)
                backfill_by_split[split] += len(extra)
                progress = True
        if not progress:
            break

    overlaps = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        intersection = sorted(set(split_sources[left]) & set(split_sources[right]))
        overlaps[f"{left}_{right}"] = intersection

    diagnostics = {
        "genre": genre,
        "target": target_count,
        "targets_by_split": targets,
        "available_clips": available_clips,
        "source_count": len(sources),
        "selected_by_split": {split: len(items) for split, items in result.items()},
        "selected_total": sum(len(items) for items in result.values()),
        "sources_by_split": {
            split: len(items) for split, items in split_sources.items()
        },
        "source_keys_by_split": split_sources,
        "backfill_by_split": backfill_by_split,
        "source_overlap": overlaps,
        "source_disjoint": not any(overlaps.values()),
        "minimum_independent_sources_met": len(sources) >= 5,
    }
    return result, diagnostics


def bereinige_manifestzeile(
    row: dict[str, Any],
    *,
    genre: str,
    split: str,
    projektwurzel: Path | None = None,
    dataset_marker: str,
) -> dict[str, Any]:

    clean = {key: value for key, value in row.items() if not key.startswith("_")}
    clean["split"] = split
    clean["primary_genre"] = genre
    clean["lora_genre"] = genre
    clean["genre"] = genre
    clean["caption"] = GENRE_CAPTIONS.get(
        genre, str(clean.get("caption") or "calm lofi instrumental, no vocals")
    )
    clean["text"] = clean["caption"]
    clean["description"] = clean["caption"]
    clean[dataset_marker] = True
    clean["source_split_key"] = quellen_schluessel(clean, projektwurzel)
    if dataset_marker == "lora_balanced_dataset":
        clean["lora_source_key"] = clean["source_split_key"]
    return clean
