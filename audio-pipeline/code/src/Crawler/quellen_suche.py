#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

DATASET_SRC = Path(__file__).resolve().parents[1] / "Dataset"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))
try:
    from genre_regeln import (
        GENRE_KEYWORDS,
        GENRE_LABELS,
        GENRE_REQUIRED_TERMS,
        QUELLEN_REGEL_VERSION,
        STANDARD_GENRES,
        quelle_passt_zum_genre,
    )
except Exception:
    QUELLEN_REGEL_VERSION = "lofi_quellen_v4_guitar_breiter_2026_08_11"
    STANDARD_GENRES = ("jazz_lofi", "chillhop_lofi", "dreamy_lofi", "study_lofi", "guitar_lofi")
    GENRE_LABELS = {
        "jazz_lofi": "Jazz Lofi",
        "chillhop_lofi": "Chillhop Lofi",
        "dreamy_lofi": "Dreamy Lofi",
        "study_lofi": "Study Lofi",
        "guitar_lofi": "Guitar Lofi",
    }
    GENRE_KEYWORDS = {genre: (label,) for genre, label in GENRE_LABELS.items()}
    GENRE_REQUIRED_TERMS = GENRE_KEYWORDS

    def quelle_passt_zum_genre(row: dict[str, Any], genre: str | None = None) -> tuple[bool, str]:
        return True, ""


MODULES = {
    'suche': '"""Monolithischer YouTube-LoFi-Crawler.\n\nDieses Skript vereint saemtliche Funktionen und Klassen des urspruenglichen Crawlers in\nin einer Datei. Es enthaelt Konfiguration, API-Client, Hilfsfunktionen, Filterregeln,\nSpeicherlogik und den CLI-Einstiegspunkt. Dadurch laesst sich der Crawler als\neinzelne Datei ausfuehren, ohne mehrere Module anlegen zu muessen.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport logging\nimport os\nimport re\nimport sys\nimport time\nfrom dataclasses import dataclass, asdict, field\nfrom datetime import datetime, timezone\nfrom typing import Any, Dict, Generator, Iterable, List, Optional, Set, TypeVar\n\nimport requests\n\n\n###############################################################################\n# Abschnitt: Hilfsfunktionen (utils)\n###############################################################################\n\nT = TypeVar("T")\n\n\ndef now_utc_iso() -> str:\n    """Gibt die aktuelle Zeit in UTC als ISO-8601-String zurueck.\n\n    Der zurueckgegebene String enthaelt immer ein ``Z``-Suffix, das UTC kennzeichnet.\n    """\n    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(\n        "+00:00", "Z"\n    )\n\n\ndef parse_iso8601_duration(duration: str) -> int:\n    """Parst eine ISO-8601-Dauerzeichenfolge in Gesamtsekunden.\n\n    Dieser Parser unterstuetzt den Teil der ISO-8601-Dauern, den die\n    YouTube-API zurueckliefert (``PnDTnHnMnS``). Wochen und Monate werden bei\n    Videolaengen nicht erwartet und daher nicht verarbeitet. Unbekannte\n    Formate liefern 0 Sekunden zurueck.\n\n    Args:\n        duration: Eine Dauerzeichenfolge wie ``"PT1H30M"`` oder ``"P1DT2H"``.\n\n    Returns:\n        Die Gesamtdauer in Sekunden als Ganzzahl.\n    """\n    pattern = re.compile(\n        r"P"\n        r"(?:(?P<days>\\d+)D)?"\n        r"(?:T"\n        r"(?:(?P<hours>\\d+)H)?"\n        r"(?:(?P<minutes>\\d+)M)?"\n        r"(?:(?P<seconds>\\d+)S)?"\n        r")?"\n    )\n    match = pattern.fullmatch(duration)\n    if not match:\n        return 0\n    days = int(match.group("days") or 0)\n    hours = int(match.group("hours") or 0)\n    minutes = int(match.group("minutes") or 0)\n    seconds = int(match.group("seconds") or 0)\n    return days * 86400 + hours * 3600 + minutes * 60 + seconds\n\n\ndef chunked(iterable: Iterable[T], size: int) -> Generator[List[T], None, None]:\n    """Liefert sukzessive Teilstuecke aus einem Iterable."""\n    chunk: List[T] = []\n    for item in iterable:\n        chunk.append(item)\n        if len(chunk) >= size:\n            yield chunk\n            chunk = []\n    if chunk:\n        yield chunk\n\n\ndef extract_video_id(value: str) -> Optional[str]:\n    """Extrahiert eine YouTube-Video-ID aus einer URL oder gibt die ID unveraendert zurueck."""\n    value = value.strip()\n    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):\n        return value\n    patterns = [\n        r"[?&]v=([A-Za-z0-9_-]{11})",\n        r"youtu\\.be/([A-Za-z0-9_-]{11})",\n        r"/shorts/([A-Za-z0-9_-]{11})",\n        r"/live/([A-Za-z0-9_-]{11})",\n        r"/embed/([A-Za-z0-9_-]{11})",\n    ]\n    for pat in patterns:\n        match = re.search(pat, value)\n        if match:\n            return match.group(1)\n    return None\n\n\ndef load_candidate_ids(path: str) -> List[str]:\n    """Laedt Video-IDs aus einer JSON- oder JSONL-Datei."""\n    if not path or not os.path.exists(path):\n        return []\n    candidates: List[str] = []\n    try:\n        if path.lower().endswith(".jsonl"):\n            with open(path, "r", encoding="utf-8") as f:\n                for line in f:\n                    line = line.strip()\n                    if not line:\n                        continue\n                    try:\n                        item = json.loads(line)\n                    except json.JSONDecodeError:\n                        continue\n                    vid = None\n                    if isinstance(item, dict):\n                        vid = item.get("video_id") or item.get("videoId")\n                        if not vid:\n                            url = item.get("webpage_url") or item.get("url") or ""\n                            vid = extract_video_id(url)\n                    elif isinstance(item, str):\n                        vid = extract_video_id(item)\n                    if vid:\n                        candidates.append(vid)\n        else:\n            with open(path, "r", encoding="utf-8") as f:\n                data = json.load(f)\n            if isinstance(data, dict):\n                items = data.get("videos", [])\n            elif isinstance(data, list):\n                items = data\n            else:\n                items = []\n            for item in items:\n                vid = None\n                if isinstance(item, dict):\n                    vid = item.get("video_id") or item.get("videoId")\n                    if not vid:\n                        url = item.get("webpage_url") or item.get("url") or ""\n                        vid = extract_video_id(url)\n                elif isinstance(item, str):\n                    vid = extract_video_id(item)\n                if vid:\n                    candidates.append(vid)\n    except Exception as exc:\n        logging.warning("Fehler beim Laden der Kandidaten aus %s: %s", path, exc)\n    return list(dict.fromkeys(candidates))\n\n\ndef load_existing_video_ids(path: str) -> Set[str]:\n    """Laedt bereits gespeicherte Video-IDs, um Duplikate zu vermeiden."""\n    if not os.path.exists(path):\n        return set()\n    ids: Set[str] = set()\n    with open(path, "r", encoding="utf-8") as f:\n        for line in f:\n            line = line.strip()\n            if not line:\n                continue\n            try:\n                item = json.loads(line)\n            except json.JSONDecodeError:\n                continue\n            if isinstance(item, dict):\n                vid = item.get("video_id") or item.get("videoId")\n                if not vid:\n                    url = item.get("webpage_url") or item.get("url") or ""\n                    vid = extract_video_id(url)\n                if vid:\n                    ids.add(vid)\n    return ids\n\n\n###############################################################################\n# Abschnitt: Konfiguration (config)\n###############################################################################\n\nDEFAULT_VIDEO_URLS: List[str] = []\n\n\ndef load_env_file(path: str = "code/configs/env/.env") -> None:\n    if not os.path.exists(path):\n        return\n    try:\n        with open(path, "r", encoding="utf-8") as f:\n            for line in f:\n                raw = line.strip()\n                if not raw or raw.startswith("#") or "=" not in raw:\n                    continue\n                key, value = raw.split("=", 1)\n                key = key.strip()\n                value = value.strip()\n                if value.startswith(("\'", "\\"")) and value.endswith(("\'", "\\"")):\n                    value = value[1:-1]\n                if key and key not in os.environ:\n                    os.environ[key] = value\n    except Exception as exc:\n        logging.warning("Konnte .env nicht laden: %s", exc)\n\n\n@dataclass\nclass Config:\n    """Enthaelt die Konfigurationswerte des Crawlers."""\n\n    api_key: str = field(default_factory=lambda: os.getenv("YOUTUBE_API_KEY", ""))\n    keywords: List[str] = field(\n        default_factory=lambda: [\n            kw.strip()\n            for kw in os.getenv(\n                "KEYWORDS", "lofi, lo-fi, chillhop, chill, hip hop"\n            ).split(",")\n            if kw.strip()\n        ]\n    )\n    min_duration_sec: int = int(os.getenv("MIN_DURATION_SEC", "600"))\n    min_view_count: int = int(os.getenv("MIN_VIEW_COUNT", "10000"))\n    max_results_per_query: int = int(os.getenv("MAX_RESULTS_PER_QUERY", "50"))\n    fetch_comments: bool = os.getenv("FETCH_COMMENTS", "false").lower() in {"1", "true", "yes"}\n    max_comments_per_video: int = int(os.getenv("MAX_COMMENTS_PER_VIDEO", "50"))\n\n    @classmethod\n    def load(cls) -> "Config":\n        for env_path in ("code/configs/env/.env", "../code/configs/env/.env", ".env"):\n            load_env_file(env_path)\n        cfg = cls()\n        if not cfg.api_key:\n            if sys.stdin is None or not sys.stdin.isatty():\n                raise ValueError(\n                    "YOUTUBE_API_KEY ist nicht gesetzt und kein interaktives Terminal verfuegbar. "\n                    "Bitte lege code/configs/env/.env an oder setze die Umgebungsvariable."\n                )\n            try:\n                cfg.api_key = input("YouTube API Key eingeben: ").strip()\n            except EOFError:\n                raise ValueError(\n                    "YOUTUBE_API_KEY ist nicht gesetzt und die Eingabe konnte nicht gelesen werden. "\n                    "Bitte lege code/configs/env/.env an oder setze die Umgebungsvariable."\n                )\n        if not cfg.api_key:\n            raise ValueError(\n                "YOUTUBE_API_KEY ist nicht gesetzt. Bitte erstelle code/configs/env/.env oder exportiere die Variable."\n            )\n        return cfg\n\n\n###############################################################################\n# Abschnitt: API-Client (youtube_client)\n###############################################################################\n\n\nclass YouTubeClient:\n    """Einfacher Client fuer einen Teil der YouTube Data API v3."""\n\n    API_BASE = "https://www.googleapis.com/youtube/v3"\n\n    def __init__(\n        self,\n        api_key: str,\n        session: Optional[requests.Session] = None,\n        max_retries: int = 3,\n        backoff_factor: float = 1.0,\n    ) -> None:\n        self.api_key = api_key\n        self.session = session or requests.Session()\n        # Ignore global proxy env vars so local broken defaults do not block API calls.\n        self.session.trust_env = False\n        self.max_retries = max_retries\n        self.backoff_factor = backoff_factor\n\n    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:\n        url = f"{self.API_BASE}/{endpoint}"\n        params = dict(params)\n        params.setdefault("key", self.api_key)\n        for attempt in range(1, self.max_retries + 1):\n            try:\n                response = self.session.get(url, params=params, timeout=15)\n                response.raise_for_status()\n                return response.json()\n            except requests.RequestException as exc:\n                logging.warning(\n                    "Anfrage an %s scheiterte beim Versuch %d/%d: %s",\n                    url,\n                    attempt,\n                    self.max_retries,\n                    exc,\n                )\n                if attempt == self.max_retries:\n                    raise\n                sleep_for = self.backoff_factor * (2 ** (attempt - 1))\n                time.sleep(sleep_for)\n        raise RuntimeError("Unerwarteter Abbruch der Retry-Schleife in _request")\n\n    def search_video_ids(self, query: str, max_results: int = 50) -> List[str]:\n        collected: List[str] = []\n        page_token: Optional[str] = None\n        while len(collected) < max_results:\n            batch_size = min(max_results - len(collected), 50)\n            params = {\n                "part": "id",\n                "q": query,\n                "type": "video",\n                "maxResults": batch_size,\n            }\n            if page_token:\n                params["pageToken"] = page_token\n            data = self._request("search", params)\n            items = data.get("items", [])\n            for item in items:\n                vid = item.get("id", {}).get("videoId")\n                if vid:\n                    collected.append(vid)\n                    if len(collected) >= max_results:\n                        break\n            page_token = data.get("nextPageToken")\n            if not page_token:\n                break\n        return collected\n\n    def fetch_video_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:\n        videos: List[Dict[str, Any]] = []\n        for chunk in chunked(video_ids, 50):\n            params = {\n                "part": "snippet,contentDetails,statistics",\n                "id": ",".join(chunk),\n            }\n            data = self._request("videos", params)\n            for item in data.get("items", []):\n                snippet = item.get("snippet", {})\n                content_details = item.get("contentDetails", {})\n                statistics = item.get("statistics", {})\n                duration_str = content_details.get("duration")\n                duration_sec = (\n                    parse_iso8601_duration(duration_str) if duration_str else None\n                )\n\n                def to_int(value: Optional[str]) -> Optional[int]:\n                    return int(value) if value is not None else None\n\n                videos.append(\n                    {\n                        "video_id": item.get("id"),\n                        "title": snippet.get("title"),\n                        "channel_title": snippet.get("channelTitle"),\n                        "channel_id": snippet.get("channelId"),\n                        "upload_date": snippet.get("publishedAt"),\n                        "duration_sec": duration_sec,\n                        "view_count": to_int(statistics.get("viewCount")),\n                        "like_count": to_int(statistics.get("likeCount")),\n                        "comment_count": to_int(statistics.get("commentCount")),\n                        "tags": snippet.get("tags", []),\n                    }\n                )\n        return videos\n\n    def fetch_top_comments(self, video_id: str, max_comments: int) -> List[Dict[str, Any]]:\n        comments: List[Dict[str, Any]] = []\n        page_token: Optional[str] = None\n        while len(comments) < max_comments:\n            batch_size = min(max_comments - len(comments), 100)\n            params = {\n                "part": "snippet",\n                "videoId": video_id,\n                "maxResults": batch_size,\n                "textFormat": "plainText",\n            }\n            if page_token:\n                params["pageToken"] = page_token\n            data = self._request("commentThreads", params)\n            for item in data.get("items", []):\n                snippet = item.get("snippet", {})\n                top_comment = snippet.get("topLevelComment", {})\n                comment_snippet = top_comment.get("snippet", {})\n                comments.append(\n                    {\n                        "comment_id": top_comment.get("id"),\n                        "video_id": video_id,\n                        "author": comment_snippet.get("authorDisplayName"),\n                        "text": comment_snippet.get("textDisplay")\n                        or comment_snippet.get("textOriginal"),\n                        "like_count": (\n                            int(comment_snippet.get("likeCount"))\n                            if comment_snippet.get("likeCount") is not None\n                            else None\n                        ),\n                        "published_at": comment_snippet.get("publishedAt"),\n                    }\n                )\n                if len(comments) >= max_comments:\n                    break\n            page_token = data.get("nextPageToken")\n            if not page_token:\n                break\n        return comments\n\n\n###############################################################################\n# Abschnitt: Filterregeln (rules)\n###############################################################################\n\n\ndef is_relevant_video(video: Dict[str, object], cfg: Config) -> bool:\n    """Gibt ``True`` zurueck, wenn das gegebene Video alle Relevanzpruefungen besteht."""\n    duration = video.get("duration_sec")\n    views = video.get("view_count")\n    if duration is None or views is None:\n        return False\n    if duration < cfg.min_duration_sec:\n        return False\n    if views < cfg.min_view_count:\n        return False\n    title = (video.get("title") or "").lower()\n    tags = video.get("tags") or []\n    text = title + " " + " ".join(t.lower() for t in tags)\n    for kw in cfg.keywords:\n        if kw.lower() in text:\n            return True\n    return False\n\n\n###############################################################################\n# Abschnitt: JSONL-Speicher (storage_jsonl)\n###############################################################################\n\n\n@dataclass\nclass State:\n    """Speichert IDs, die bereits verarbeitet wurden."""\n\n    seen_video_ids: Set[str] = field(default_factory=set)\n    seen_comment_ids: Set[str] = field(default_factory=set)\n\n\ndef load_state(path: str) -> State:\n    if not os.path.exists(path):\n        return State()\n    try:\n        with open(path, "r", encoding="utf-8") as f:\n            data = json.load(f)\n        return State(\n            seen_video_ids=set(data.get("seen_video_ids", [])),\n            seen_comment_ids=set(data.get("seen_comment_ids", [])),\n        )\n    except Exception as exc:\n        logging.warning("Fehler beim Laden der Zustandsdatei %s: %s", path, exc)\n        return State()\n\n\ndef save_state(state: State, path: str) -> None:\n    os.makedirs(os.path.dirname(path), exist_ok=True)\n    data = {\n        "seen_video_ids": sorted(state.seen_video_ids),\n        "seen_comment_ids": sorted(state.seen_comment_ids),\n    }\n    with open(path, "w", encoding="utf-8") as f:\n        json.dump(data, f, ensure_ascii=False, indent=2)\n\n\ndef append_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:\n    os.makedirs(os.path.dirname(path), exist_ok=True)\n    with open(path, "a", encoding="utf-8") as f:\n        for record in records:\n            json_line = json.dumps(record, ensure_ascii=False)\n            f.write(json_line + "\\n")\n\n\ndef write_run_meta(\n    run_id: str, cfg: Any, stats: Dict[str, Any], errors: List[str]\n) -> None:\n    base_dir = os.path.join("daten", "raw", "youtube")\n    runs_dir = os.path.join(base_dir, "runs")\n    os.makedirs(runs_dir, exist_ok=True)\n    meta_path = os.path.join(runs_dir, f"{run_id}_meta.json")\n    try:\n        config_dict = asdict(cfg)\n    except Exception:\n        config_dict = cfg.__dict__.copy()\n    config_safe = {k: v for k, v in config_dict.items() if k != "api_key"}\n    meta: Dict[str, Any] = {\n        "run_id": run_id,\n        "config": config_safe,\n        "stats": stats,\n        "errors": errors,\n        "created_at": now_utc_iso(),\n    }\n    with open(meta_path, "w", encoding="utf-8") as f:\n        json.dump(meta, f, ensure_ascii=False, indent=2)\n\n\n###############################################################################\n# Abschnitt: Crawler-Orchestrierung (crawler)\n###############################################################################\n\n\ndef run_crawler(args: argparse.Namespace) -> None:\n    cfg = Config.load()\n\n    max_results = args.max_results if args.max_results is not None else cfg.max_results_per_query\n    fetch_comments = args.fetch_comments or cfg.fetch_comments\n    max_comments = args.max_comments if args.max_comments is not None else cfg.max_comments_per_video\n\n    client = YouTubeClient(cfg.api_key)\n    base_dir = os.path.join("daten", "raw", "youtube")\n    state_path = os.path.join(base_dir, "state.json")\n    state = load_state(state_path)\n    output_path = (\n        args.output_path\n        if getattr(args, "output_path", None)\n        else os.path.join(base_dir, "candidates.jsonl")\n    )\n    state.seen_video_ids.update(load_existing_video_ids(output_path))\n\n    run_id = now_utc_iso().replace("-", "").replace(":", "").replace("T", "")\n\n    stats: Dict[str, int] = {\n        "searched_videos": 0,\n        "fetched_videos": 0,\n        "relevant_videos": 0,\n        "written_videos": 0,\n        "written_comments": 0,\n    }\n    errors: List[str] = []\n    start_time = now_utc_iso()\n\n    def process_videos(videos: List[Dict[str, Any]], search_label: str) -> None:\n        """Gemeinsame Verarbeitungslinie fuer gefundene oder manuell angegebene Videos."""\n        for video in videos:\n            video_id = video.get("video_id")\n            if not video_id or video_id in state.seen_video_ids:\n                continue\n            if not is_relevant_video(video, cfg):\n                continue\n            stats["relevant_videos"] += 1\n            record = {\n                "video_id": video_id,\n                "title": video.get("title"),\n                "channel_title": video.get("channel_title"),\n                "channel_id": video.get("channel_id"),\n                "upload_date": video.get("upload_date"),\n                "duration_sec": video.get("duration_sec"),\n                "view_count": video.get("view_count"),\n                "like_count": video.get("like_count"),\n                "comment_count": video.get("comment_count"),\n                "tags": video.get("tags"),\n                "search_query": search_label,\n                "webpage_url": f"https://www.youtube.com/watch?v={video_id}",\n                "crawled_at": now_utc_iso(),\n                "run_id": run_id,\n            }\n            try:\n                append_jsonl(output_path, [record])\n                state.seen_video_ids.add(video_id)\n                stats["written_videos"] += 1\n            except Exception as write_err:\n                errors.append(f"Video schreiben {video_id}: {write_err}")\n                logging.error("Fehler beim Schreiben des Videos %s: %s", video_id, write_err)\n                continue\n            if fetch_comments:\n                try:\n                    comments = client.fetch_top_comments(video_id, max_comments)\n                    for comment in comments:\n                        cid = comment.get("comment_id")\n                        if not cid or cid in state.seen_comment_ids:\n                            continue\n                        comment_record = {\n                            "comment_id": cid,\n                            "video_id": video_id,\n                            "author": comment.get("author"),\n                            "text": comment.get("text"),\n                            "like_count": comment.get("like_count"),\n                            "published_at": comment.get("published_at"),\n                            "crawled_at": now_utc_iso(),\n                            "run_id": run_id,\n                        }\n                        append_jsonl(os.path.join(base_dir, "comments.jsonl"), [comment_record])\n                        state.seen_comment_ids.add(cid)\n                        stats["written_comments"] += 1\n                except Exception as cm_err:\n                    errors.append(f"Kommentare fuer {video_id}: {cm_err}")\n                    logging.warning(\n                        "Fehler beim Abrufen/Schreiben von Kommentaren fuer Video %s: %s",\n                        video_id,\n                        cm_err,\n                    )\n\n    manual_video_ids: List[str] = []\n    if getattr(args, "candidates_path", None):\n        manual_video_ids.extend(load_candidate_ids(args.candidates_path))\n    if getattr(args, "video_ids", None):\n        manual_video_ids.extend(args.video_ids)\n    if getattr(args, "video_urls", None):\n        for url in args.video_urls:\n            vid = extract_video_id(url)\n            if vid:\n                manual_video_ids.append(vid)\n            else:\n                errors.append(f"Ungueltige Video-URL: {url}")\n                logging.warning("Konnte keine Video-ID aus URL extrahieren: %s", url)\n    manual_video_ids = list(dict.fromkeys(manual_video_ids))\n    if manual_video_ids:\n        try:\n            videos = client.fetch_video_details(manual_video_ids)\n            stats["fetched_videos"] += len(videos)\n            process_videos(videos, "manual")\n        except Exception as manual_err:\n            errors.append(f"Manuelle Videos: {manual_err}")\n            logging.error("Fehler beim Abrufen manueller Videos: %s", manual_err, exc_info=True)\n\n    for query in args.queries or []:\n        logging.info("Suche nach Suchbegriff: %s", query)\n        try:\n            video_ids = client.search_video_ids(query, max_results)\n            stats["searched_videos"] += len(video_ids)\n            if not video_ids:\n                continue\n            videos = client.fetch_video_details(video_ids)\n            stats["fetched_videos"] += len(videos)\n            process_videos(videos, query)\n        except Exception as q_err:\n            errors.append(f"Suchanfrage {query}: {q_err}")\n            logging.error(\n                "Fehler bei der Verarbeitung der Suchanfrage %s: %s",\n                query,\n                q_err,\n                exc_info=True,\n            )\n\n    save_state(state, state_path)\n    stats["start_time"] = start_time\n    stats["end_time"] = now_utc_iso()\n    write_run_meta(run_id, cfg, stats, errors)\n\n    logging.info(\n        "Lauf %s abgeschlossen. Geschriebene Videos: %d, Geschriebene Kommentare: %d, Fehler: %d",\n        run_id,\n        stats["written_videos"],\n        stats["written_comments"],\n        len(errors),\n    )\n\n\ndef main(argv: List[str] | None = None) -> None:\n    parser = argparse.ArgumentParser(description="Monolithischer YouTube-LoFi-Crawler")\n    parser.add_argument(\n        "--candidates",\n        dest="candidates_path",\n        help="Pfad zu candidates.json oder candidates.jsonl (Video-IDs als Input)",\n    )\n    parser.add_argument(\n        "--queries",\n        nargs="+",\n        help="Ein oder mehrere Suchbegriffe, die ausgefuehrt werden sollen",\n    )\n    parser.add_argument(\n        "--video-ids",\n        nargs="+",\n        help="Optionale Liste von Video-IDs, die direkt verarbeitet werden sollen",\n    )\n    parser.add_argument(\n        "--video-urls",\n        nargs="+",\n        help="Optionale Liste von YouTube-URLs, aus denen die Video-IDs extrahiert werden",\n    )\n    parser.add_argument(\n        "--max-results",\n        type=int,\n        help="Maximale Anzahl der Suchergebnisse pro Suchbegriff (Standard aus der Konfiguration)",\n    )\n    parser.add_argument(\n        "--fetch-comments",\n        action="store_true",\n        help="Rufe fuer jedes relevante Video die Top-Kommentare ab",\n    )\n    parser.add_argument(\n        "--max-comments",\n        type=int,\n        help="Maximale Anzahl von Kommentaren pro Video (Standard aus der Konfiguration)",\n    )\n    parser.add_argument(\n        "--output-path",\n        help="Zielpfad fuer die JSONL-Ausgabe (Standard: daten/raw/youtube/candidates.jsonl)",\n    )\n    parser.add_argument(\n        "--log-level",\n        default="INFO",\n        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],\n        help="Legt das Logging-Level fest",\n    )\n    args = parser.parse_args(argv)\n    if not args.queries and not args.video_ids and not args.video_urls and not args.candidates_path:\n        if os.path.exists("candidates.jsonl"):\n            args.candidates_path = "candidates.jsonl"\n        elif os.path.exists("candidates.json"):\n            args.candidates_path = "candidates.json"\n    if not args.queries and not args.video_ids and not args.video_urls and not args.candidates_path:\n        if DEFAULT_VIDEO_URLS:\n            args.video_urls = DEFAULT_VIDEO_URLS.copy()\n        else:\n            url = input("YouTube-Link eingeben: ").strip()\n            if url:\n                args.video_urls = [url]\n            else:\n                parser.error(\n                    "Gib mindestens einen Suchbegriff (--queries) oder Video-IDs/URLs (--video-ids/--video-urls) an."\n                )\n    logging.basicConfig(\n        level=getattr(logging, args.log_level),\n        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",\n    )\n    try:\n        run_crawler(args)\n    except Exception as exc:\n        logging.exception("Unbehandelte Ausnahme: %s", exc)\n        sys.exit(1)\n\n\nif __name__ == "__main__":\n    main()\n',
    'download': '#!/usr/bin/env python3\n"""\nEigenstaendiger Download-Schritt fuer YouTube-Audio.\n\nDieses Skript liest `candidates.jsonl`, laedt fuer jedes Video die Audiospur mit\n`yt-dlp` herunter, extrahiert sie als MP3 (oder ein anderes von FFmpeg\nunterstuetztes Zielformat) und schreibt begleitende JSONL-Metadaten.\n\nZiel ist ein sauberer Zwischenschritt:\n    quellen_suche.py -> download_audio.py -> dataset prep\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport logging\nimport os\nimport re\nimport shutil\nimport sys\nimport time\nimport unicodedata\nfrom dataclasses import dataclass\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Any, Dict, Iterable, List, Optional\n\ntry:\n    import yt_dlp\nexcept ModuleNotFoundError as exc:\n    raise SystemExit(\n        "yt-dlp is not installed for this interpreter. "\n        "Use the project virtualenv, e.g. `.venv/bin/python ...`."\n    ) from exc\n\ntry:\n    from tqdm import tqdm\nexcept ModuleNotFoundError as exc:\n    raise SystemExit(\n        "tqdm is not installed for this interpreter. "\n        "Use the project virtualenv, e.g. `.venv/bin/python ...`."\n    ) from exc\n\n\nPROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if ((parent / "code" / "configs" / "konfiguration.yaml").exists() or (parent / "konfiguration.yaml").exists()) and (parent / "code").exists())\nWORKSPACE_ROOT = PROJECT_ROOT\nDEFAULT_CANDIDATES_PATH = PROJECT_ROOT / "daten" / "raw" / "youtube" / "candidates.jsonl"\nDEFAULT_OUTPUT_AUDIO_DIR = PROJECT_ROOT / "daten" / "raw" / "audio"\nDEFAULT_OUTPUT_METADATA_PATH = (\n    PROJECT_ROOT / "daten" / "metadata" / "downloads" / "downloaded_videos.jsonl"\n)\nDENO_BIN_DIR = PROJECT_ROOT / "code" / "tools" / "deno" / "bin"\n\nWINDOWS_INVALID_CHARS_RE = re.compile(r\'[<>:"/\\\\|?*\\x00-\\x1F]\')\nWHITESPACE_RE = re.compile(r"\\s+")\nVIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")\nWINDOWS_RESERVED_NAMES = {\n    "CON",\n    "PRN",\n    "AUX",\n    "NUL",\n    "COM1",\n    "COM2",\n    "COM3",\n    "COM4",\n    "COM5",\n    "COM6",\n    "COM7",\n    "COM8",\n    "COM9",\n    "LPT1",\n    "LPT2",\n    "LPT3",\n    "LPT4",\n    "LPT5",\n    "LPT6",\n    "LPT7",\n    "LPT8",\n    "LPT9",\n}\n\n\n@dataclass(frozen=True)\nclass Candidate:\n    video_id: str\n    title: str\n    webpage_url: str\n\n\n@dataclass(frozen=True)\nclass DownloadStrategy:\n    name: str\n    extractor_args: Dict[str, Dict[str, List[str]]]\n\n\nDOWNLOAD_STRATEGIES: List[DownloadStrategy] = [\n    DownloadStrategy(\n        name="web_safari",\n        extractor_args={"youtube": {"player_client": ["web_safari"]}},\n    ),\n    DownloadStrategy(\n        name="mweb",\n        extractor_args={"youtube": {"player_client": ["mweb"]}},\n    ),\n    DownloadStrategy(\n        name="ios",\n        extractor_args={"youtube": {"player_client": ["ios"]}},\n    ),\n    DownloadStrategy(\n        name="tv_simply",\n        extractor_args={"youtube": {"player_client": ["tv_simply"]}},\n    ),\n    DownloadStrategy(\n        name="tv_downgraded",\n        extractor_args={"youtube": {"player_client": ["tv_downgraded"]}},\n    ),\n    DownloadStrategy(\n        name="web_embedded",\n        extractor_args={"youtube": {"player_client": ["web_embedded"]}},\n    ),\n    DownloadStrategy(\n        name="android_vr",\n        extractor_args={"youtube": {"player_client": ["android_vr"]}},\n    ),\n    DownloadStrategy(\n        name="tv",\n        extractor_args={"youtube": {"player_client": ["tv"]}},\n    ),\n    DownloadStrategy(\n        name="android",\n        extractor_args={"youtube": {"player_client": ["android"]}},\n    ),\n    DownloadStrategy(name="default", extractor_args={}),\n]\n\n\ndef ensure_js_runtime_on_path() -> None:\n    """Make the bundled Deno visible to yt-dlp when the shell PATH misses it."""\n    deno_bin = DENO_BIN_DIR / "deno"\n    if not deno_bin.exists():\n        return\n    current_path = os.environ.get("PATH", "")\n    paths = current_path.split(os.pathsep) if current_path else []\n    deno_dir = str(DENO_BIN_DIR)\n    if deno_dir not in paths:\n        os.environ["PATH"] = deno_dir + (os.pathsep + current_path if current_path else "")\n\n\ndef now_utc_iso() -> str:\n    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(\n        "+00:00", "Z"\n    )\n\n\ndef parse_log_level(value: str) -> int:\n    level_name = str(value).upper()\n    if level_name not in logging._nameToLevel:\n        raise argparse.ArgumentTypeError(f"Unsupported log level: {value}")\n    return logging._nameToLevel[level_name]\n\n\ndef extract_video_id(value: str | None) -> Optional[str]:\n    if not value:\n        return None\n    raw = str(value).strip()\n    if VIDEO_ID_RE.fullmatch(raw):\n        return raw\n    patterns = [\n        r"[?&]v=([A-Za-z0-9_-]{11})",\n        r"youtu\\.be/([A-Za-z0-9_-]{11})",\n        r"/shorts/([A-Za-z0-9_-]{11})",\n        r"/live/([A-Za-z0-9_-]{11})",\n        r"/embed/([A-Za-z0-9_-]{11})",\n    ]\n    for pattern in patterns:\n        match = re.search(pattern, raw)\n        if match:\n            return match.group(1)\n    return None\n\n\ndef sanitize_title_for_filename(title: str, max_length: int) -> str:\n    normalized = unicodedata.normalize("NFKC", str(title or ""))\n    sanitized = WINDOWS_INVALID_CHARS_RE.sub(" ", normalized)\n    sanitized = WHITESPACE_RE.sub(" ", sanitized).strip(" .")\n    if not sanitized:\n        sanitized = "untitled"\n    if sanitized.upper() in WINDOWS_RESERVED_NAMES:\n        sanitized = f"file_{sanitized}"\n    sanitized = sanitized[:max_length].rstrip(" .")\n    return sanitized or "untitled"\n\n\ndef build_audio_filename(title: str, video_id: str, audio_format: str) -> str:\n    audio_format = audio_format.lower()\n    suffix = f" [{video_id}]"\n    max_title_length = max(16, 180 - len(suffix) - len(audio_format) - 1)\n    safe_title = sanitize_title_for_filename(title, max_title_length)\n    return f"{safe_title}{suffix}.{audio_format}"\n\n\ndef append_jsonl_record(path: Path, record: Dict[str, Any]) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    with path.open("a", encoding="utf-8") as handle:\n        handle.write(json.dumps(record, ensure_ascii=False) + "\\n")\n\n\ndef read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:\n    with path.open("r", encoding="utf-8") as handle:\n        for line_number, raw_line in enumerate(handle, start=1):\n            line = raw_line.strip()\n            if not line:\n                continue\n            try:\n                item = json.loads(line)\n            except json.JSONDecodeError as exc:\n                logging.warning("Skipping invalid JSONL line %d in %s: %s", line_number, path, exc)\n                continue\n            if isinstance(item, dict):\n                yield item\n            else:\n                logging.warning(\n                    "Skipping non-object JSONL line %d in %s", line_number, path\n                )\n\n\ndef load_candidates(path: Path, max_videos: Optional[int]) -> List[Candidate]:\n    if not path.exists():\n        raise FileNotFoundError(f"Candidates file not found: {path}")\n\n    candidates: List[Candidate] = []\n    seen_video_ids: set[str] = set()\n    skipped_rows = 0\n\n    for item in read_jsonl(path):\n        video_id = (\n            item.get("video_id")\n            or item.get("videoId")\n            or extract_video_id(item.get("webpage_url"))\n            or extract_video_id(item.get("url"))\n        )\n        if not video_id:\n            skipped_rows += 1\n            logging.warning("Skipping candidate without video_id: %s", item)\n            continue\n        if video_id in seen_video_ids:\n            continue\n\n        title = str(item.get("title") or video_id).strip() or video_id\n        webpage_url = str(\n            item.get("webpage_url")\n            or item.get("url")\n            or f"https://www.youtube.com/watch?v={video_id}"\n        ).strip()\n\n        candidates.append(\n            Candidate(video_id=video_id, title=title, webpage_url=webpage_url)\n        )\n        seen_video_ids.add(video_id)\n\n        if max_videos is not None and len(candidates) >= max_videos:\n            break\n\n    logging.info(\n        "Loaded %d candidate(s) from %s%s",\n        len(candidates),\n        path,\n        f"; skipped rows={skipped_rows}" if skipped_rows else "",\n    )\n    return candidates\n\n\ndef load_success_index(metadata_path: Path) -> Dict[str, Dict[str, Any]]:\n    if not metadata_path.exists():\n        return {}\n\n    success_index: Dict[str, Dict[str, Any]] = {}\n    for item in read_jsonl(metadata_path):\n        video_id = item.get("video_id")\n        status = item.get("status")\n        if not video_id or status not in {"downloaded", "skipped_existing"}:\n            continue\n        success_index[str(video_id)] = item\n    return success_index\n\n\ndef build_existing_audio_index(\n    output_audio_dir: Path,\n    audio_format: str,\n    success_index: Dict[str, Dict[str, Any]],\n) -> Dict[str, Path]:\n    audio_index: Dict[str, Path] = {}\n    expected_extension = f".{audio_format.lower()}"\n\n    for video_id, record in success_index.items():\n        audio_path = record.get("audio_path")\n        if not audio_path:\n            continue\n        candidate = Path(str(audio_path))\n        if candidate.exists():\n            audio_index[video_id] = candidate\n\n    if output_audio_dir.exists():\n        for child in output_audio_dir.iterdir():\n            if not child.is_file() or child.suffix.lower() != expected_extension:\n                continue\n            match = re.search(r"\\[([A-Za-z0-9_-]{11})\\]$", child.stem)\n            if match:\n                audio_index[match.group(1)] = child\n\n    return audio_index\n\n\ndef find_existing_audio_path(\n    video_id: str,\n    existing_audio_index: Dict[str, Path],\n) -> Optional[Path]:\n    candidate = existing_audio_index.get(video_id)\n    if candidate and candidate.exists():\n        return candidate\n    return None\n\n\ndef resolve_ffmpeg_location() -> Optional[str]:\n    ffmpeg_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")\n    if ffmpeg_bin:\n        return str(Path(ffmpeg_bin).parent)\n\n    candidates = [\n        PROJECT_ROOT / "code" / "werkzeuge" / "ffmpeg" / "bin",\n    ]\n    for directory in candidates:\n        if (directory / "ffmpeg").exists() or (directory / "ffmpeg.exe").exists():\n            return str(directory)\n\n    for pattern in ("ffmpeg", "ffmpeg.exe"):\n        for match in PROJECT_ROOT.rglob(pattern):\n            return str(match.parent)\n    return None\n\n\nclass YtDlpLogger:\n    def __init__(self, logger: logging.Logger) -> None:\n        self._logger = logger\n\n    def debug(self, message: str) -> None:\n        if self._logger.isEnabledFor(logging.DEBUG):\n            self._logger.debug(message)\n\n    def warning(self, message: str) -> None:\n        self._logger.warning(message)\n\n    def error(self, message: str) -> None:\n        self._logger.error(message)\n\n\ndef build_ydl_opts(\n    target_path: Path,\n    audio_format: str,\n    ffmpeg_location: str,\n    overwrite: bool,\n    strategy: DownloadStrategy,\n) -> Dict[str, Any]:\n    ensure_js_runtime_on_path()\n    stem_path = target_path.parent / target_path.stem\n    opts: Dict[str, Any] = {\n        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",\n        "outtmpl": str(stem_path) + ".%(ext)s",\n        "noplaylist": True,\n        "quiet": True,\n        "no_warnings": True,\n        "continuedl": True,\n        "retries": 3,\n        "fragment_retries": 3,\n        "socket_timeout": 30,\n        "overwrites": overwrite,\n        "logger": YtDlpLogger(logging.getLogger("yt_dlp")),\n        "extractor_args": strategy.extractor_args,\n        "postprocessors": [\n            {\n                "key": "FFmpegExtractAudio",\n                "preferredcodec": audio_format,\n                "preferredquality": "192",\n            }\n        ],\n        "ffmpeg_location": ffmpeg_location,\n        "final_ext": audio_format,\n    }\n    cookie_file = os.environ.get("YTDLP_COOKIE_FILE") or os.environ.get("YTDLP_COOKIES")\n    if cookie_file:\n        opts["cookiefile"] = str(Path(cookie_file).expanduser())\n    cookies_from_browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER")\n    if cookies_from_browser:\n        opts["cookiesfrombrowser"] = (cookies_from_browser,)\n    opts["remote_components"] = ["ejs:github"]\n    return opts\n\n\ndef resolve_downloaded_audio_path(\n    target_path: Path,\n    video_id: str,\n    existing_audio_index: Dict[str, Path],\n) -> Path:\n    if target_path.exists():\n        return target_path\n\n    existing = find_existing_audio_path(video_id=video_id, existing_audio_index=existing_audio_index)\n    if existing:\n        return existing\n\n    raise FileNotFoundError(\n        f"Downloaded audio file not found for {video_id}; expected {target_path}"\n    )\n\n\ndef download_candidate_audio(\n    candidate: Candidate,\n    output_audio_dir: Path,\n    metadata_path: Path,\n    audio_format: str,\n    skip_existing: bool,\n    ffmpeg_location: str,\n    success_index: Dict[str, Dict[str, Any]],\n    existing_audio_index: Dict[str, Path],\n) -> Dict[str, Any]:\n    output_audio_dir.mkdir(parents=True, exist_ok=True)\n    target_filename = build_audio_filename(candidate.title, candidate.video_id, audio_format)\n    target_path = output_audio_dir / target_filename\n\n    existing_path = find_existing_audio_path(\n        video_id=candidate.video_id,\n        existing_audio_index=existing_audio_index,\n    )\n    if skip_existing and existing_path:\n        record = {\n            "video_id": candidate.video_id,\n            "title": candidate.title,\n            "webpage_url": candidate.webpage_url,\n            "status": "skipped_existing",\n            "audio_format": audio_format,\n            "audio_path": str(existing_path.resolve()),\n            "file_size_bytes": existing_path.stat().st_size,\n            "expected_audio_path": str(target_path.resolve()),\n            "started_at": now_utc_iso(),\n            "finished_at": now_utc_iso(),\n            "elapsed_sec": 0.0,\n        }\n        append_jsonl_record(metadata_path, record)\n        success_index[candidate.video_id] = record\n        existing_audio_index[candidate.video_id] = existing_path\n        return record\n\n    started_at = now_utc_iso()\n    started_perf = time.perf_counter()\n\n    try:\n        used_strategy = "default"\n        strategy_errors: List[str] = []\n        last_error: Optional[Exception] = None\n\n        for strategy in DOWNLOAD_STRATEGIES:\n            used_strategy = strategy.name\n            try:\n                ydl_opts = build_ydl_opts(\n                    target_path=target_path,\n                    audio_format=audio_format,\n                    ffmpeg_location=ffmpeg_location,\n                    overwrite=not skip_existing,\n                    strategy=strategy,\n                )\n                with yt_dlp.YoutubeDL(ydl_opts) as ydl:\n                    ydl.extract_info(candidate.webpage_url, download=True)\n                break\n            except Exception as exc:\n                last_error = exc\n                strategy_errors.append(f"{strategy.name}: {type(exc).__name__}: {exc}")\n                logging.warning(\n                    "Download strategy \'%s\' failed for %s: %s",\n                    strategy.name,\n                    candidate.video_id,\n                    exc,\n                )\n        else:\n            error_tail = "; ".join(strategy_errors) if strategy_errors else "unknown error"\n            raise RuntimeError(\n                f"All yt-dlp strategies failed for {candidate.video_id}: {error_tail}"\n            ) from last_error\n\n        final_path = resolve_downloaded_audio_path(\n            target_path=target_path,\n            video_id=candidate.video_id,\n            existing_audio_index=existing_audio_index,\n        )\n        elapsed_sec = time.perf_counter() - started_perf\n        record = {\n            "video_id": candidate.video_id,\n            "title": candidate.title,\n            "webpage_url": candidate.webpage_url,\n            "status": "downloaded",\n            "audio_format": audio_format,\n            "audio_path": str(final_path.resolve()),\n            "file_size_bytes": final_path.stat().st_size,\n            "expected_audio_path": str(target_path.resolve()),\n            "ffmpeg_location": ffmpeg_location,\n            "download_strategy": used_strategy,\n            "started_at": started_at,\n            "finished_at": now_utc_iso(),\n            "elapsed_sec": round(elapsed_sec, 3),\n        }\n        append_jsonl_record(metadata_path, record)\n        success_index[candidate.video_id] = record\n        existing_audio_index[candidate.video_id] = final_path\n        return record\n    except Exception as exc:\n        elapsed_sec = time.perf_counter() - started_perf\n        error_message = f"{type(exc).__name__}: {exc}"\n        record = {\n            "video_id": candidate.video_id,\n            "title": candidate.title,\n            "webpage_url": candidate.webpage_url,\n            "status": "error",\n            "audio_format": audio_format,\n            "audio_path": str(target_path.resolve()),\n            "expected_audio_path": str(target_path.resolve()),\n            "started_at": started_at,\n            "finished_at": now_utc_iso(),\n            "elapsed_sec": round(elapsed_sec, 3),\n            "error": error_message,\n        }\n        append_jsonl_record(metadata_path, record)\n        logging.exception("Download failed for %s (%s)", candidate.video_id, candidate.webpage_url)\n        return record\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description="Download YouTube audio for crawler candidates")\n    parser.add_argument(\n        "--candidates-path",\n        default=str(DEFAULT_CANDIDATES_PATH),\n        help="Path to candidates.jsonl",\n    )\n    parser.add_argument(\n        "--output-audio-dir",\n        default=str(DEFAULT_OUTPUT_AUDIO_DIR),\n        help="Directory for downloaded audio files",\n    )\n    parser.add_argument(\n        "--output-metadata-path",\n        default=str(DEFAULT_OUTPUT_METADATA_PATH),\n        help="JSONL path for download metadata",\n    )\n    parser.add_argument(\n        "--audio-format",\n        default="mp3",\n        help="Target audio format for FFmpegExtractAudio (default: mp3)",\n    )\n    parser.add_argument(\n        "--max-videos",\n        type=int,\n        default=None,\n        help="Limit number of candidate videos for this run",\n    )\n    parser.add_argument(\n        "--skip-existing",\n        action=argparse.BooleanOptionalAction,\n        default=True,\n        help="Skip videos whose target audio already exists (default: true)",\n    )\n    parser.add_argument(\n        "--log-level",\n        default="INFO",\n        help="Logging level (DEBUG, INFO, WARNING, ERROR)",\n    )\n    return parser.parse_args()\n\n\ndef main() -> int:\n    args = parse_args()\n    log_level = parse_log_level(args.log_level)\n    logging.basicConfig(\n        level=log_level,\n        format="%(asctime)s [%(levelname)s] %(message)s",\n    )\n\n    candidates_path = Path(args.candidates_path).expanduser().resolve()\n    output_audio_dir = Path(args.output_audio_dir).expanduser().resolve()\n    metadata_path = Path(args.output_metadata_path).expanduser().resolve()\n    audio_format = str(args.audio_format).strip().lower()\n\n    ffmpeg_location = resolve_ffmpeg_location()\n    if not ffmpeg_location:\n        logging.error(\n            "FFmpeg not found. Cannot extract audio as %s. Install FFmpeg or provide it in PATH.",\n            audio_format,\n        )\n        return 2\n\n    candidates = load_candidates(path=candidates_path, max_videos=args.max_videos)\n    if not candidates:\n        logging.warning("No candidates to process.")\n        return 0\n\n    success_index = load_success_index(metadata_path)\n    existing_audio_index = build_existing_audio_index(\n        output_audio_dir=output_audio_dir,\n        audio_format=audio_format,\n        success_index=success_index,\n    )\n    stats = {\n        "total": len(candidates),\n        "downloaded": 0,\n        "skipped_existing": 0,\n        "failed": 0,\n    }\n    run_started_perf = time.perf_counter()\n\n    for candidate in tqdm(candidates, desc="Audio downloads", unit="video"):\n        record = download_candidate_audio(\n            candidate=candidate,\n            output_audio_dir=output_audio_dir,\n            metadata_path=metadata_path,\n            audio_format=audio_format,\n            skip_existing=args.skip_existing,\n            ffmpeg_location=ffmpeg_location,\n            success_index=success_index,\n            existing_audio_index=existing_audio_index,\n        )\n        status = str(record.get("status"))\n        if status == "downloaded":\n            stats["downloaded"] += 1\n        elif status == "skipped_existing":\n            stats["skipped_existing"] += 1\n        else:\n            stats["failed"] += 1\n\n    elapsed_sec = time.perf_counter() - run_started_perf\n    logging.info(\n        (\n            "Download summary: total=%d downloaded=%d skipped_existing=%d "\n            "failed=%d elapsed=%.2fs metadata=%s"\n        ),\n        stats["total"],\n        stats["downloaded"],\n        stats["skipped_existing"],\n        stats["failed"],\n        elapsed_sec,\n        metadata_path,\n    )\n    return 1 if stats["failed"] else 0\n\n\nif __name__ == "__main__":\n    sys.exit(main())\n',
    'import': '#!/usr/bin/env python3\n"""Importiert eine erlaubte YouTube-Quelle lokal als MP3.\n\nDer Import ist nur fuer eigene, freigegebene, Creative-Commons- oder sonstig\nrechtlich erlaubte Quellen gedacht. Das Skript nutzt lokal yt-dlp und ffmpeg;\nkeine externen Converter-Websites.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport os\nimport sys\nimport time\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Any, Dict\n\n\nPROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if ((parent / "code" / "configs" / "konfiguration.yaml").exists() or (parent / "konfiguration.yaml").exists()) and (parent / "code").exists())\nCRAWLER_DIR = Path(__file__).resolve().parent\nif str(CRAWLER_DIR) not in sys.path:\n    sys.path.insert(0, str(CRAWLER_DIR))\n\nfrom download import (  # noqa: E402\n    Candidate,\n    build_audio_filename,\n    build_existing_audio_index,\n    download_candidate_audio,\n    extract_video_id,\n    load_success_index,\n    resolve_ffmpeg_location,\n)\n\n\nDEFAULT_OUTPUT_AUDIO_DIR = PROJECT_ROOT / "daten" / "raw" / "audio" / "youtube_imports"\nDEFAULT_DOWNLOAD_METADATA_PATH = PROJECT_ROOT / "daten" / "metadata" / "downloads" / "downloaded_videos.jsonl"\nDEFAULT_IMPORT_METADATA_PATH = PROJECT_ROOT / "daten" / "metadata" / "downloads" / "youtube_manual_mp3_imports.jsonl"\n\n\ndef utc_now() -> str:\n    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")\n\n\ndef append_jsonl(path: Path, row: Dict[str, Any]) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    with path.open("a", encoding="utf-8") as handle:\n        handle.write(json.dumps(row, ensure_ascii=False) + "\\n")\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description="Importiert eine erlaubte YouTube-Quelle als MP3.")\n    parser.add_argument("--url", required=True, help="YouTube-URL oder Video-ID.")\n    parser.add_argument("--titel", default="", help="Optionaler lokaler Titel fuer Dateiname/Metadaten.")\n    parser.add_argument("--audio-format", default="mp3", choices=["mp3", "wav", "m4a"])\n    parser.add_argument("--output-audio-dir", default=str(DEFAULT_OUTPUT_AUDIO_DIR))\n    parser.add_argument("--metadata-path", default=str(DEFAULT_DOWNLOAD_METADATA_PATH))\n    parser.add_argument("--import-metadata-path", default=str(DEFAULT_IMPORT_METADATA_PATH))\n    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)\n    parser.add_argument("--cookies", default="", help="Optional: yt-dlp Cookies-Datei.")\n    parser.add_argument("--cookies-from-browser", default="", help="Optional: z.B. firefox oder chrome.")\n    parser.add_argument("--dry-run", action="store_true", help="Nur pruefen, nichts herunterladen.")\n    parser.add_argument(\n        "--ich-habe-rechte",\n        action="store_true",\n        help="Bestaetigung: Quelle darf fuer dieses Projekt heruntergeladen/umgewandelt werden.",\n    )\n    return parser.parse_args()\n\n\ndef configure_cookies(args: argparse.Namespace) -> None:\n    if args.cookies:\n        cookie_path = Path(args.cookies).expanduser().resolve()\n        if not cookie_path.exists():\n            raise FileNotFoundError(f"Cookies-Datei nicht gefunden: {cookie_path}")\n        os.environ["YTDLP_COOKIE_FILE"] = str(cookie_path)\n    if args.cookies_from_browser:\n        os.environ["YTDLP_COOKIES_FROM_BROWSER"] = str(args.cookies_from_browser).strip()\n\n\ndef main() -> int:\n    args = parse_args()\n    video_id = extract_video_id(args.url)\n    if not video_id:\n        print(json.dumps({"status": "error", "error": "Ungueltige YouTube-URL oder Video-ID."}, ensure_ascii=False))\n        return 2\n    if not args.ich_habe_rechte:\n        print(\n            json.dumps(\n                {\n                    "status": "blocked",\n                    "video_id": video_id,\n                    "error": (\n                        "Import blockiert. Nutze --ich-habe-rechte nur, wenn du die Quelle "\n                        "rechtlich fuer dieses Projekt verwenden darfst."\n                    ),\n                },\n                ensure_ascii=False,\n            )\n        )\n        return 3\n\n    output_audio_dir = Path(args.output_audio_dir).expanduser().resolve()\n    metadata_path = Path(args.metadata_path).expanduser().resolve()\n    import_metadata_path = Path(args.import_metadata_path).expanduser().resolve()\n    audio_format = str(args.audio_format).lower().strip()\n    title = str(args.titel or f"youtube_{video_id}").strip() or f"youtube_{video_id}"\n    webpage_url = args.url if args.url.startswith("http") else f"https://www.youtube.com/watch?v={video_id}"\n    expected_path = output_audio_dir / build_audio_filename(title, video_id, audio_format)\n\n    if args.dry_run:\n        print(\n            json.dumps(\n                {\n                    "status": "dry_run",\n                    "video_id": video_id,\n                    "webpage_url": webpage_url,\n                    "expected_audio_path": str(expected_path),\n                    "output_audio_dir": str(output_audio_dir),\n                    "metadata_path": str(metadata_path),\n                    "rights_confirmed": True,\n                },\n                ensure_ascii=False,\n                indent=2,\n            )\n        )\n        return 0\n\n    configure_cookies(args)\n    ffmpeg_location = resolve_ffmpeg_location()\n    if not ffmpeg_location:\n        print(json.dumps({"status": "error", "error": "FFmpeg nicht gefunden."}, ensure_ascii=False))\n        return 4\n\n    started = time.perf_counter()\n    success_index = load_success_index(metadata_path)\n    existing_audio_index = build_existing_audio_index(\n        output_audio_dir=output_audio_dir,\n        audio_format=audio_format,\n        success_index=success_index,\n    )\n    record = download_candidate_audio(\n        candidate=Candidate(video_id=video_id, title=title, webpage_url=webpage_url),\n        output_audio_dir=output_audio_dir,\n        metadata_path=metadata_path,\n        audio_format=audio_format,\n        skip_existing=bool(args.skip_existing),\n        ffmpeg_location=ffmpeg_location,\n        success_index=success_index,\n        existing_audio_index=existing_audio_index,\n    )\n    import_record = {\n        **record,\n        "importer": "manual_youtube_mp3",\n        "rights_confirmed": True,\n        "legal_note": "Nur fuer eigene, freigegebene oder anderweitig erlaubte Quellen verwenden.",\n        "import_metadata_path": str(import_metadata_path),\n        "finished_import_at": utc_now(),\n        "total_elapsed_sec": round(time.perf_counter() - started, 3),\n    }\n    append_jsonl(import_metadata_path, import_record)\n    print(json.dumps(import_record, ensure_ascii=False, indent=2))\n    return 0 if str(record.get("status")) in {"downloaded", "skipped_existing"} else 1\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n',
    'status': '#!/usr/bin/env python3\n"""\nMonitor the crawler audio download progress in the terminal.\n\nThis script reads the crawler candidates and the downloader metadata JSONL and\nrenders a live progress view with percent complete, remaining item count and an\nETA estimate.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport statistics\nimport sys\nimport time\nfrom dataclasses import dataclass\nfrom pathlib import Path\nfrom typing import Any, Dict, Iterable, Optional\n\n\nPROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if ((parent / "code" / "configs" / "konfiguration.yaml").exists() or (parent / "konfiguration.yaml").exists()) and (parent / "code").exists())\nDEFAULT_CANDIDATES_PATH = PROJECT_ROOT / "daten" / "raw" / "youtube" / "candidates.jsonl"\nDEFAULT_METADATA_PATH = PROJECT_ROOT / "daten" / "metadata" / "downloads" / "downloaded_videos.jsonl"\nDEFAULT_AUDIO_DIR = PROJECT_ROOT / "daten" / "raw" / "audio"\nDONE_STATUSES = {"downloaded", "skipped_existing", "error"}\n\n\n@dataclass(frozen=True)\nclass Snapshot:\n    total: int\n    processed: int\n    remaining: int\n    downloaded: int\n    skipped_existing: int\n    errors: int\n    avg_elapsed_sec: Optional[float]\n    median_elapsed_sec: Optional[float]\n    eta_sec: Optional[float]\n    current_part_name: Optional[str]\n    current_part_size_bytes: Optional[int]\n    metadata_rows: int\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description="Monitor audio download progress")\n    parser.add_argument(\n        "--candidates-path",\n        default=str(DEFAULT_CANDIDATES_PATH),\n        help="Path to candidates.jsonl",\n    )\n    parser.add_argument(\n        "--metadata-path",\n        default=str(DEFAULT_METADATA_PATH),\n        help="Path to downloaded_videos.jsonl",\n    )\n    parser.add_argument(\n        "--audio-dir",\n        default=str(DEFAULT_AUDIO_DIR),\n        help="Directory containing downloaded audio files and partial downloads",\n    )\n    parser.add_argument(\n        "--poll-interval-sec",\n        type=float,\n        default=5.0,\n        help="Refresh interval in seconds",\n    )\n    parser.add_argument(\n        "--bar-width",\n        type=int,\n        default=40,\n        help="Width of the ASCII progress bar",\n    )\n    parser.add_argument(\n        "--once",\n        action="store_true",\n        help="Print one snapshot and exit",\n    )\n    parser.add_argument(\n        "--events",\n        action="store_true",\n        help="Print one line whenever a new download result appears instead of rendering a full-screen monitor",\n    )\n    return parser.parse_args()\n\n\ndef read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:\n    if not path.exists():\n        return\n    with path.open("r", encoding="utf-8") as handle:\n        for raw_line in handle:\n            line = raw_line.strip()\n            if not line:\n                continue\n            try:\n                item = json.loads(line)\n            except json.JSONDecodeError:\n                continue\n            if isinstance(item, dict):\n                yield item\n\n\ndef load_candidate_ids(path: Path) -> list[str]:\n    ids: list[str] = []\n    seen: set[str] = set()\n    for row in read_jsonl(path):\n        video_id = row.get("video_id")\n        if not video_id:\n            continue\n        value = str(video_id)\n        if value in seen:\n            continue\n        ids.append(value)\n        seen.add(value)\n    return ids\n\n\ndef latest_status_by_video(path: Path, allowed_ids: set[str]) -> tuple[Dict[str, Dict[str, Any]], int]:\n    latest: Dict[str, Dict[str, Any]] = {}\n    row_count = 0\n    for row in read_jsonl(path):\n        row_count += 1\n        video_id = row.get("video_id")\n        if not video_id:\n            continue\n        video_id = str(video_id)\n        if allowed_ids and video_id not in allowed_ids:\n            continue\n        latest[video_id] = row\n    return latest, row_count\n\n\ndef all_rows(path: Path, allowed_ids: set[str]) -> list[Dict[str, Any]]:\n    rows: list[Dict[str, Any]] = []\n    for row in read_jsonl(path):\n        video_id = row.get("video_id")\n        if not video_id:\n            continue\n        if allowed_ids and str(video_id) not in allowed_ids:\n            continue\n        rows.append(row)\n    return rows\n\n\ndef find_current_part(audio_dir: Path) -> tuple[Optional[str], Optional[int]]:\n    newest_path: Optional[Path] = None\n    newest_mtime = -1.0\n    if audio_dir.exists():\n        for path in audio_dir.glob("*.part"):\n            try:\n                stat = path.stat()\n            except OSError:\n                continue\n            if stat.st_mtime >= newest_mtime:\n                newest_path = path\n                newest_mtime = stat.st_mtime\n    if newest_path is None:\n        return None, None\n    return newest_path.name, newest_path.stat().st_size\n\n\ndef build_snapshot(candidates_path: Path, metadata_path: Path, audio_dir: Path) -> Snapshot:\n    candidate_ids = load_candidate_ids(candidates_path)\n    total = len(candidate_ids)\n    latest, row_count = latest_status_by_video(metadata_path, set(candidate_ids))\n\n    downloaded = 0\n    skipped_existing = 0\n    errors = 0\n    processed = 0\n    elapsed_values: list[float] = []\n\n    for row in latest.values():\n        status = str(row.get("status") or "")\n        if status not in DONE_STATUSES:\n            continue\n        processed += 1\n        if status == "downloaded":\n            downloaded += 1\n        elif status == "skipped_existing":\n            skipped_existing += 1\n        elif status == "error":\n            errors += 1\n\n        try:\n            elapsed = float(row.get("elapsed_sec") or 0.0)\n        except (TypeError, ValueError):\n            elapsed = 0.0\n        if elapsed > 0:\n            elapsed_values.append(elapsed)\n\n    remaining = max(0, total - processed)\n    avg_elapsed = (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else None\n    median_elapsed = statistics.median(elapsed_values) if elapsed_values else None\n    eta_sec = (avg_elapsed * remaining) if avg_elapsed is not None and remaining > 0 else 0.0 if remaining == 0 else None\n    current_part_name, current_part_size_bytes = find_current_part(audio_dir)\n\n    return Snapshot(\n        total=total,\n        processed=processed,\n        remaining=remaining,\n        downloaded=downloaded,\n        skipped_existing=skipped_existing,\n        errors=errors,\n        avg_elapsed_sec=avg_elapsed,\n        median_elapsed_sec=median_elapsed,\n        eta_sec=eta_sec,\n        current_part_name=current_part_name,\n        current_part_size_bytes=current_part_size_bytes,\n        metadata_rows=row_count,\n    )\n\n\ndef format_seconds(seconds: Optional[float]) -> str:\n    if seconds is None:\n        return "n/a"\n    total = max(0, int(round(seconds)))\n    hours, remainder = divmod(total, 3600)\n    minutes, secs = divmod(remainder, 60)\n    if hours > 0:\n        return f"{hours}h {minutes:02d}m"\n    if minutes > 0:\n        return f"{minutes}m {secs:02d}s"\n    return f"{secs}s"\n\n\ndef format_bytes(num_bytes: Optional[int]) -> str:\n    if num_bytes is None:\n        return "n/a"\n    value = float(num_bytes)\n    for unit in ("B", "KB", "MB", "GB", "TB"):\n        if value < 1024.0 or unit == "TB":\n            if unit == "B":\n                return f"{int(value)} {unit}"\n            return f"{value:.1f} {unit}"\n        value /= 1024.0\n    return f"{value:.1f} TB"\n\n\ndef render_bar(processed: int, total: int, width: int) -> str:\n    if total <= 0:\n        return "[" + ("-" * width) + "]"\n    ratio = min(1.0, max(0.0, processed / total))\n    filled = int(round(width * ratio))\n    return "[" + ("#" * filled) + ("-" * max(0, width - filled)) + "]"\n\n\ndef render_snapshot(snapshot: Snapshot, bar_width: int, metadata_path: Path, audio_dir: Path) -> str:\n    percent = (snapshot.processed / snapshot.total * 100.0) if snapshot.total else 0.0\n    lines = [\n        "Audio Download Monitor",\n        f"{render_bar(snapshot.processed, snapshot.total, bar_width)} "\n        f"{snapshot.processed}/{snapshot.total} ({percent:.1f}%)",\n        (\n            f"Downloaded: {snapshot.downloaded} | "\n            f"Skipped: {snapshot.skipped_existing} | "\n            f"Errors: {snapshot.errors} | "\n            f"Remaining: {snapshot.remaining}"\n        ),\n        (\n            f"Avg/item: {format_seconds(snapshot.avg_elapsed_sec)} | "\n            f"Median/item: {format_seconds(snapshot.median_elapsed_sec)} | "\n            f"ETA: {format_seconds(snapshot.eta_sec)}"\n        ),\n        (\n            f"Current partial: {snapshot.current_part_name or \'none\'}"\n            + (\n                f" ({format_bytes(snapshot.current_part_size_bytes)})"\n                if snapshot.current_part_name\n                else ""\n            )\n        ),\n        f"Metadata rows: {snapshot.metadata_rows} | metadata={metadata_path}",\n        f"Audio dir: {audio_dir}",\n        "Press Ctrl+C to stop monitoring.",\n    ]\n    return "\\n".join(lines)\n\n\ndef format_event_row(row: Dict[str, Any]) -> str:\n    status = str(row.get("status") or "unknown")\n    video_id = str(row.get("video_id") or "-")\n    title = str(row.get("title") or video_id)\n    elapsed = format_seconds(float(row.get("elapsed_sec") or 0.0))\n    audio_path = str(row.get("audio_path") or "")\n    if status == "downloaded":\n        size_bytes = row.get("file_size_bytes")\n        size_text = format_bytes(int(size_bytes)) if isinstance(size_bytes, int) else "n/a"\n        return f"[downloaded] {title} [{video_id}] | {elapsed} | {size_text} | {audio_path}"\n    if status == "skipped_existing":\n        return f"[skipped] {title} [{video_id}] | existing file | {audio_path}"\n    if status == "error":\n        error = str(row.get("error") or "unknown error")\n        return f"[error] {title} [{video_id}] | {elapsed} | {error}"\n    return f"[{status}] {title} [{video_id}]"\n\n\ndef run_events_mode(candidates_path: Path, metadata_path: Path, poll_interval_sec: float) -> int:\n    candidate_ids = set(load_candidate_ids(candidates_path))\n    seen_signatures: set[str] = set()\n\n    for row in all_rows(metadata_path, candidate_ids):\n        signature = json.dumps(row, ensure_ascii=False, sort_keys=True)\n        seen_signatures.add(signature)\n\n    sys.stdout.write("Download event watcher running. Press Ctrl+C to stop.\\n")\n    sys.stdout.flush()\n\n    try:\n        while True:\n            time.sleep(max(0.2, poll_interval_sec))\n            for row in all_rows(metadata_path, candidate_ids):\n                signature = json.dumps(row, ensure_ascii=False, sort_keys=True)\n                if signature in seen_signatures:\n                    continue\n                seen_signatures.add(signature)\n                sys.stdout.write(format_event_row(row) + "\\n")\n                sys.stdout.flush()\n    except KeyboardInterrupt:\n        return 130\n\n\ndef main() -> int:\n    args = parse_args()\n    candidates_path = Path(args.candidates_path).expanduser().resolve()\n    metadata_path = Path(args.metadata_path).expanduser().resolve()\n    audio_dir = Path(args.audio_dir).expanduser().resolve()\n\n    if args.events:\n        return run_events_mode(\n            candidates_path=candidates_path,\n            metadata_path=metadata_path,\n            poll_interval_sec=args.poll_interval_sec,\n        )\n\n    try:\n        while True:\n            snapshot = build_snapshot(candidates_path, metadata_path, audio_dir)\n            output = render_snapshot(\n                snapshot=snapshot,\n                bar_width=args.bar_width,\n                metadata_path=metadata_path,\n                audio_dir=audio_dir,\n            )\n            sys.stdout.write("\\x1b[2J\\x1b[H")\n            sys.stdout.write(output + "\\n")\n            sys.stdout.flush()\n\n            if args.once or (snapshot.total > 0 and snapshot.remaining == 0 and snapshot.current_part_name is None):\n                break\n            time.sleep(max(0.2, args.poll_interval_sec))\n    except KeyboardInterrupt:\n        return 130\n    return 0\n\n\nif __name__ == "__main__":\n    sys.exit(main())\n',
    'cleanup': '#!/usr/bin/env python3\n"""Safely delete raw MusicGen MP3 files after their clips were prepared.\n\nThe script only deletes MP3s when the matching per-source clip dataset exists\nand contains at least one manifest row. By default it is a dry-run.\n\nRun from the workspace root:\n\n    .venv/bin/python code/src/Crawler/cleanup_raw_musicgen_mp3s_with_clips.py\n\nActually delete safe MP3s:\n\n    .venv/bin/python code/src/Crawler/cleanup_raw_musicgen_mp3s_with_clips.py --apply\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport re\nimport shutil\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Any, Dict, List\n\n\nPROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if ((parent / "code" / "configs" / "konfiguration.yaml").exists() or (parent / "konfiguration.yaml").exists()) and (parent / "code").exists())\nDEFAULT_AUDIO_DIR = PROJECT_ROOT / "daten" / "raw" / "audio"\nDEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "daten" / "processed" / "musicgen_per_source_60s"\nDEFAULT_REPORT_PATH = PROJECT_ROOT / "training" / "downloads" / "raw_mp3_cleanup_report.json"\n\nVIDEO_ID_IN_FILENAME_RE = re.compile(r"\\[([A-Za-z0-9_-]{11})\\]$")\n\n\ndef utc_now() -> str:\n    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")\n\n\ndef manifest_row_count(path: Path) -> int:\n    if not path.exists():\n        return 0\n    try:\n        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())\n    except OSError:\n        return 0\n\n\ndef clip_row_count(source_root: Path) -> int:\n    return sum(manifest_row_count(source_root / split / "data.jsonl") for split in ("train", "valid", "test"))\n\n\ndef video_id_from_mp3(path: Path) -> str | None:\n    match = VIDEO_ID_IN_FILENAME_RE.search(path.stem)\n    return match.group(1) if match else None\n\n\ndef gb(bytes_value: int) -> float:\n    return round(bytes_value / (1024 ** 3), 3)\n\n\ndef write_report(path: Path, payload: Dict[str, Any]) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description="Delete raw MP3s only when finished MusicGen clips exist.")\n    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))\n    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))\n    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))\n    parser.add_argument("--apply", action="store_true", help="Actually delete safe MP3 files.")\n    return parser.parse_args()\n\n\ndef main() -> int:\n    args = parse_args()\n    audio_dir = Path(args.audio_dir).expanduser().resolve()\n    processed_root = Path(args.processed_root).expanduser().resolve()\n    report_path = Path(args.report_path).expanduser().resolve()\n\n    safe: List[Dict[str, Any]] = []\n    unsafe: List[Dict[str, Any]] = []\n\n    for mp3_path in sorted(audio_dir.glob("*.mp3")):\n        video_id = video_id_from_mp3(mp3_path)\n        size_bytes = mp3_path.stat().st_size\n        row: Dict[str, Any] = {\n            "video_id": video_id,\n            "path": str(mp3_path),\n            "size_bytes": size_bytes,\n            "size_gb": gb(size_bytes),\n        }\n        if not video_id:\n            row["reason"] = "missing_video_id_in_filename"\n            unsafe.append(row)\n            continue\n\n        source_root = processed_root / video_id\n        rows = clip_row_count(source_root)\n        has_summary = (source_root / "dataset_summary.json").exists()\n        row["processed_root"] = str(source_root)\n        row["clip_rows"] = rows\n        row["has_dataset_summary"] = has_summary\n        if has_summary and rows > 0:\n            safe.append(row)\n        else:\n            row["reason"] = "no_finished_clip_dataset"\n            unsafe.append(row)\n\n    deleted: List[Dict[str, Any]] = []\n    if args.apply:\n        for row in safe:\n            path = Path(str(row["path"]))\n            try:\n                path.unlink()\n                deleted.append(row)\n            except OSError as exc:\n                failed = dict(row)\n                failed["reason"] = f"delete_failed:{type(exc).__name__}:{exc}"\n                unsafe.append(failed)\n\n    payload = {\n        "created_at": utc_now(),\n        "applied": bool(args.apply),\n        "audio_dir": str(audio_dir),\n        "processed_root": str(processed_root),\n        "safe_count": len(safe),\n        "unsafe_count": len(unsafe),\n        "deleted_count": len(deleted),\n        "safe_size_gb": gb(sum(int(row["size_bytes"]) for row in safe)),\n        "unsafe_size_gb": gb(sum(int(row["size_bytes"]) for row in unsafe)),\n        "deleted_size_gb": gb(sum(int(row["size_bytes"]) for row in deleted)),\n        "free_space_after_gb": round(shutil.disk_usage(audio_dir).free / (1024 ** 3), 3),\n        "safe": safe,\n        "unsafe": unsafe,\n    }\n    write_report(report_path, payload)\n\n    action = "Geloescht" if args.apply else "Loeschbar"\n    print(\n        f"{action}: {len(deleted) if args.apply else len(safe)} MP3s "\n        f"({payload[\'deleted_size_gb\'] if args.apply else payload[\'safe_size_gb\']} GB)",\n        flush=True,\n    )\n    print(f"Nicht geloescht: {len(unsafe)} MP3s ({payload[\'unsafe_size_gb\']} GB)", flush=True)\n    print(f"Report: {report_path}", flush=True)\n    if not args.apply:\n        print("Dry-run. Zum echten Loeschen erneut mit --apply starten.", flush=True)\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n',
}

DEPENDENCIES = {'import': ['download']}
TOP_AUSWAHL_LIMIT = 5


def _projektwurzel() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "code").exists() and (parent / "daten").exists():
            return parent
    return Path.cwd()


def _jetzt_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug(text: str) -> str:
    value = text.strip().lower()
    value = value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "top10"


def _video_id(value: str) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    patterns = [
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"/shorts/([A-Za-z0-9_-]{11})",
        r"/live/([A-Za-z0-9_-]{11})",
        r"/embed/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


def _youtube_url(video_id: str | None, fallback: str = "") -> str:
    return f"https://www.youtube.com/watch?v={video_id}" if video_id else fallback


def _text_norm(text: str) -> str:
    value = str(text or "").lower()
    value = value.replace("lo-fi", "lofi")
    value = value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _woerter(text: str) -> set[str]:
    stopp = {"and", "or", "the", "for", "mit", "und", "oder", "der", "die", "das", "ein", "eine"}
    return {part for part in _text_norm(text).split() if len(part) > 1 and part not in stopp}


def _genre_key_aus_input(text: str) -> str:
    value = _text_norm(text)
    value_key = value.replace(" ", "_")
    for genre in STANDARD_GENRES:
        label = _text_norm(GENRE_LABELS.get(genre, genre.replace("_", " ")))
        terms = [label, genre.replace("_", " ")]
        terms.extend(str(item) for item in GENRE_KEYWORDS.get(genre, ()))
        terms.extend(str(item) for item in GENRE_REQUIRED_TERMS.get(genre, ()))
        if value_key == genre or any(_text_norm(term) and _text_norm(term) in value for term in terms):
            return genre
    if "jazz" in value:
        return "jazz_lofi"
    if "guitar" in value or "gitarre" in value or "acoustic" in value:
        return "guitar_lofi"
    if "dream" in value or "ambient" in value:
        return "dreamy_lofi"
    if "study" in value or "focus" in value or "work" in value:
        return "study_lofi"
    return "chillhop_lofi"


def _float_wert(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _dauer_minuten(seconds: Any) -> float | None:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return round(value / 60.0, 2)


def _lese_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _bekannte_videos(verlauf_path: Path, wurzel: Path) -> list[dict[str, Any]]:
    bekannte: list[dict[str, Any]] = []
    quellen = [
        wurzel / "daten" / "metadata" / "downloads" / "heruntergeladene_videos.jsonl",
        wurzel / "daten" / "metadata" / "downloads" / "downloaded_videos.jsonl",
        wurzel / "daten" / "metadata" / "downloads" / "youtube_manuelle_mp3_importe.jsonl",
    ]
    for path in quellen:
        for row in _lese_jsonl(path):
            vid = _video_id(str(row.get("video_id") or row.get("videoId") or row.get("url") or row.get("webpage_url") or ""))
            url = str(row.get("url") or row.get("webpage_url") or _youtube_url(vid))
            bekannte.append(
                {
                    "video_id": vid,
                    "url": url,
                    "titel": row.get("titel") or row.get("title") or "",
                    "kanal": row.get("kanal") or row.get("channel") or row.get("uploader") or "",
                    "status": row.get("status") or "bekannt",
                    "quelle": str(path),
                }
            )
    return bekannte


def _ist_aehnlich_bekannt(kandidat: dict[str, Any], bekannte: list[dict[str, Any]]) -> tuple[bool, str]:
    vid = str(kandidat.get("video_id") or "")
    url = str(kandidat.get("url") or "")
    titel = _text_norm(str(kandidat.get("titel") or ""))
    kanal = _text_norm(str(kandidat.get("kanal") or ""))
    for row in bekannte:
        known_vid = str(row.get("video_id") or "")
        known_url = str(row.get("url") or "")
        if vid and known_vid and vid == known_vid:
            return True, "video_id_bekannt"
        if url and known_url and url == known_url:
            return True, "url_bekannt"
        known_titel = _text_norm(str(row.get("titel") or ""))
        known_kanal = _text_norm(str(row.get("kanal") or ""))
        if titel and known_titel and kanal and known_kanal and kanal == known_kanal:
            if SequenceMatcher(None, titel, known_titel).ratio() >= 0.86:
                return True, "aehnlicher_titel_gleicher_kanal"
    return False, ""


def _suchanfragen(args: argparse.Namespace) -> list[str]:
    genre = args.genre.strip()
    stimmung = args.stimmung.strip()
    lizenz = args.lizenz.strip()
    lofi_genre = genre if "lofi" in _text_norm(genre) else f"{genre} lofi"
    dauer_text = ""
    if args.dauer_minuten_min and args.dauer_minuten_max:
        dauer_text = f"{int(args.dauer_minuten_min)}-{int(args.dauer_minuten_max)} minutes"

    genre_key = _genre_key_aus_input(genre)
    kern_limit = 6 if genre_key == "guitar_lofi" else 1
    kernbegriffe = list(GENRE_KEYWORDS.get(genre_key, ())[:kern_limit]) or [lofi_genre]
    vorlagen = [
        "{kern} mix 1 hour instrumental {lizenz}",
        "{kern} beats instrumental {lizenz} long mix",
        "{kern} instrumental full album no vocals",
        "{kern} playlist instrumental {stimmung} {lizenz}",
        "calm {kern} study mix instrumental no vocals",
        "mellow {kern} beats long mix",
        "royalty free {kern} instrumental mix",
        "no copyright {kern} instrumental beats",
        "{kern} radio mix instrumental no vocals",
        "{kern} {dauer_text} instrumental {lizenz}",
    ]

    queries: list[str] = []
    for kern in kernbegriffe:
        for vorlage in vorlagen:
            queries.append(
                vorlage.format(kern=kern, lizenz=lizenz, stimmung=stimmung, dauer_text=dauer_text)
            )
    queries.extend(
        [
            f"{lofi_genre} instrumental {stimmung} {lizenz}",
            f"{lofi_genre} beats {lizenz} long mix",
            f"{lofi_genre} instrumental full album no vocals",
            f"{lofi_genre} playlist instrumental {lizenz}",
            f"{lofi_genre} {dauer_text} instrumental {lizenz}".strip(),
        ]
    )


    for kern in kernbegriffe:
        queries.extend(
            [
                f"{kern} instrumental mix no vocals",
                f"{kern} instrumental playlist",
                f"{kern} study beats instrumental",
                f"best {kern} instrumental mix",
                f"{kern} hip hop instrumental mix no vocals",
            ]
        )

    seen: set[str] = set()
    result: list[str] = []
    for query in queries:
        normalized = " ".join(query.split())
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _yt_dlp_suche(queries: list[str], max_suchergebnisse: int) -> tuple[list[dict[str, Any]], str | None]:
    try:
        import yt_dlp
    except Exception as exc:
        return [], f"yt-dlp nicht verfuegbar: {type(exc).__name__}: {exc}"

    kandidaten: list[dict[str, Any]] = []
    if not queries:
        return kandidaten, None


    pro_query = max(1, min(50, max_suchergebnisse))
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            for index, query in enumerate(queries, start=1):
                print(f"Suche {index}/{len(queries)}", flush=True)
                info = ydl.extract_info(f"ytsearch{pro_query}:{query}", download=False)
                for entry in (info or {}).get("entries") or []:
                    if not isinstance(entry, dict):
                        continue
                    vid = _video_id(str(entry.get("id") or entry.get("url") or ""))
                    url = _youtube_url(vid, str(entry.get("url") or ""))
                    kandidaten.append(
                        {
                            "video_id": vid,
                            "url": url,
                            "titel": entry.get("title") or "",
                            "kanal": entry.get("uploader") or entry.get("channel") or "",
                            "beschreibung": entry.get("description") or "",
                            "dauer": entry.get("duration"),
                            "dauer_minuten": _dauer_minuten(entry.get("duration")),
                            "view_count": entry.get("view_count"),
                            "like_count": entry.get("like_count"),
                            "suchquery": query,
                            "quelle": "ytsearch",
                        }
                    )
    except Exception as exc:
        return kandidaten, f"ytsearch fehlgeschlagen: {type(exc).__name__}: {exc}"
    return kandidaten, None


def _yt_dlp_detail_json(url: str) -> dict[str, Any] | None:
    if not url:
        return None
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--dump-json",
        "--skip-download",
        "--no-playlist",
        "--no-warnings",
        url,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=35)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _erganze_yt_dlp_details(kandidaten: list[dict[str, Any]], max_details: int) -> list[dict[str, Any]]:
    if max_details <= 0 or not kandidaten:
        return kandidaten
    result: list[dict[str, Any]] = []
    anzahl_details = min(max_details, len(kandidaten))
    for index, kandidat in enumerate(kandidaten):
        row = dict(kandidat)
        if index < max_details:
            print(f"Details {index + 1}/{anzahl_details}", flush=True)
            info = _yt_dlp_detail_json(str(row.get("url") or ""))
            if isinstance(info, dict):
                row["titel"] = row.get("titel") or info.get("title") or ""
                row["kanal"] = row.get("kanal") or info.get("uploader") or info.get("channel") or ""
                row["beschreibung"] = row.get("beschreibung") or info.get("description") or ""
                row["dauer"] = row.get("dauer") or info.get("duration")
                row["dauer_minuten"] = row.get("dauer_minuten") or _dauer_minuten(info.get("duration"))
                row["view_count"] = row.get("view_count") or info.get("view_count")
                row["like_count"] = row.get("like_count") or info.get("like_count")
                row["tags"] = info.get("tags") or row.get("tags") or []
                row["detail_metadaten_status"] = "ok"
            else:
                row["detail_metadaten_status"] = "fehlt"
        result.append(row)
    return result


def _urls_aus_lokalen_reports(wurzel: Path) -> list[dict[str, Any]]:
    suchorte = [
        wurzel / "daten" / "metadata",
        wurzel / "daten" / "raw" / "youtube",
        wurzel / "training" / "downloads",
    ]
    pattern = re.compile(r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[A-Za-z0-9_-]{11}[^\s\"'<>)]*")
    kandidaten: list[dict[str, Any]] = []
    for base in suchorte:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".csv", ".json", ".jsonl", ".md"}:
                continue
            try:
                if path.stat().st_size > 10_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for url in pattern.findall(text):
                vid = _video_id(url)
                kandidaten.append(
                    {
                        "video_id": vid,
                        "url": _youtube_url(vid, url),
                        "titel": "",
                        "kanal": "",
                        "beschreibung": "",
                        "dauer": None,
                        "dauer_minuten": None,
                        "view_count": None,
                        "like_count": None,
                        "suchquery": "lokaler_report",
                        "quelle": str(path),
                    }
                )
    return kandidaten


def _score_keywords(text: str, terms: set[str], extra: set[str] | None = None) -> float:
    if extra:
        terms = set(terms) | set(extra)
    if not terms:
        return 50.0
    normalized = _text_norm(text)
    hits = sum(1 for term in terms if term in normalized)
    return min(100.0, round(100.0 * hits / max(1, min(len(terms), 5)), 2))


def _dauer_score(dauer_minuten: float | None, minimum: float, maximum: float) -> float:
    if dauer_minuten is None:
        return 45.0
    if minimum <= dauer_minuten <= maximum:
        return 100.0
    if dauer_minuten < minimum:
        return round(max(0.0, 100.0 * (dauer_minuten / max(1.0, minimum))), 2)
    return round(max(0.0, 100.0 * (maximum / max(dauer_minuten, 1.0))), 2)


def _audio_eignung_score(text: str) -> float:
    normalized = _text_norm(text)
    block_text = f" {normalized} "
    for phrase in (
        " no vocals ",
        " no vocal ",
        " without vocals ",
        " without vocal ",
        " non vocal ",
        " vocals free ",
        " vocal free ",
    ):
        block_text = block_text.replace(phrase, " ")
    score = 60.0
    positive = ["instrumental", "mix", "beat", "beats", "lofi", "lo fi", "chillhop", "study", "relax", "album", "playlist"]
    negative = [
        "tutorial",
        "reaction",
        "vlog",
        "podcast",
        "interview",
        "speech",
        "spoken",
        "spoken word",
        "talk",
        "talking",
        "commentary",
        "narration",
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
        "review",
        "lesson",
        "explained",
        "math rock",
        "cover",
        "covers",
        "party",
        "edm",
        "techno",
        "house",
    ]
    score += min(30.0, 6.0 * sum(1 for word in positive if word in normalized))
    score -= min(60.0, 12.0 * sum(1 for word in negative if word in block_text))
    return round(max(0.0, min(100.0, score)), 2)


def _log_score(value: Any, low: float, high: float, missing: float = 45.0) -> float:
    number = _float_wert(value)
    if number is None:
        return missing
    if number <= low:
        return max(0.0, round(number / max(1.0, low) * 40.0, 2))
    scale = math.log10(max(high, low + 1.0) / max(1.0, low))
    score = 40.0 + 60.0 * (math.log10(min(number, high) / max(1.0, low)) / max(scale, 1e-6))
    return round(max(0.0, min(100.0, score)), 2)


def _popularitaet_score(kandidat: dict[str, Any]) -> float:
    views = _float_wert(kandidat.get("view_count"))
    likes = _float_wert(kandidat.get("like_count"))
    view_score = _log_score(views, low=10_000, high=5_000_000)
    like_score = _log_score(likes, low=250, high=100_000, missing=45.0)
    if views and likes:
        like_rate = min(0.08, likes / max(1.0, views))
        like_rate_score = round(like_rate / 0.04 * 100.0, 2)
    else:
        like_rate_score = 45.0
    return round(view_score * 0.55 + like_score * 0.25 + min(100.0, like_rate_score) * 0.20, 2)


_TITEL_KERNBEGRIFFE: dict[str, tuple[str, ...]] = {
    "jazz_lofi": ("jazz", "jazzy", "jazzhop"),
    "chillhop_lofi": ("chillhop",),
    "dreamy_lofi": ("dreamy", "dream", "sleep", "soft", "calm", "soothing", "gentle", "peaceful"),
    "study_lofi": ("study", "studies", "focus"),
    "guitar_lofi": ("guitar", "gitarre", "acoustic"),
}


_TITEL_SELBSTGENUEGSAM = {"chillhop_lofi"}


def _titel_zeigt_genre(titel: str, genre_key: str) -> bool:

    kernbegriffe = _TITEL_KERNBEGRIFFE.get(genre_key)
    if not kernbegriffe:
        return True
    titel_norm = _text_norm(titel).replace("-", " ")
    titel_norm = " ".join(titel_norm.split())
    if genre_key in _TITEL_SELBSTGENUEGSAM:
        return any(wort in titel_norm for wort in kernbegriffe)


    titel_norm = re.sub(r"\blo fi\b", "lofi", titel_norm)
    woerter = titel_norm.split()


    if genre_key == "guitar_lofi":
        hat_gitarre = any(
            wort == kern or wort.startswith(kern)
            for wort in woerter
            for kern in kernbegriffe
        )
        hat_lofi = "lofi" in woerter or "chillhop" in woerter
        hat_musiksignal = any(
            wort.startswith(("beat", "hiphop", "study", "relax", "mix", "vibe", "chill"))
            for wort in woerter
        )
        if hat_gitarre and (hat_lofi or hat_musiksignal):
            return True

    for i, wort in enumerate(woerter):
        if not any(wort == kern or wort.startswith(kern) for kern in kernbegriffe):
            continue
        if (i > 0 and woerter[i - 1] == "lofi") or (i + 1 < len(woerter) and woerter[i + 1] == "lofi"):
            return True
    return False


def _titel_grob_fuer_details_ok(kandidat: dict[str, Any], args: argparse.Namespace) -> bool:

    genre_key = _genre_key_aus_input(args.genre)
    titel = str(kandidat.get("titel") or "")
    if _titel_zeigt_genre(titel, genre_key):
        return True
    if genre_key != "guitar_lofi":
        return False
    titel_norm = _text_norm(titel)
    hat_gitarre = any(term in titel_norm for term in ("guitar", "gitarre", "acoustic"))
    hat_musiksignal = any(
        term in titel_norm
        for term in ("lofi", "lo fi", "chill", "beat", "beats", "hip hop", "hiphop", "study", "relax", "mix", "vibe")
    )
    return hat_gitarre and hat_musiksignal


def _lofi_quelle_ok(kandidat: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    genre_key = _genre_key_aus_input(args.genre)
    titel = str(kandidat.get("titel") or "")
    if not _titel_zeigt_genre(titel, genre_key):
        return False, f"titel_ohne_genrebezug:{genre_key}"

    text = " ".join(str(kandidat.get(key) or "") for key in ("titel", "kanal", "beschreibung", "tags"))
    row = {
        "source_title": text,
        "source_file": text,
        "titel": text,
        "description": text,
        "genre": genre_key,
    }
    return quelle_passt_zum_genre(row, genre_key)


def _bewerte_kandidat(kandidat: dict[str, Any], args: argparse.Namespace, duplikat_score: float) -> dict[str, Any]:
    text = " ".join(
        str(kandidat.get(key) or "")
        for key in ("titel", "kanal", "beschreibung", "tags")
    )
    genre_terms = _woerter(args.genre)
    if "lofi" in _text_norm(args.genre):
        genre_terms |= {"lofi", "lo-fi", "chillhop", "beat", "beats"}
    lizenz_terms = _woerter(args.lizenz)
    if "copyright" in _text_norm(args.lizenz) or "no copyright" in _text_norm(args.lizenz):
        lizenz_terms |= {"copyright", "free", "royalty", "creative", "commons", "nocopyright"}
    if "no copyright" in _text_norm(text):
        text += " nocopyright"
    stimmung_terms = _woerter(args.stimmung)

    genre_score = _score_keywords(text, genre_terms)
    lizenz_score = _score_keywords(text, lizenz_terms)
    dauer_score = _dauer_score(kandidat.get("dauer_minuten"), args.dauer_minuten_min, args.dauer_minuten_max)
    stimmung_score = _score_keywords(text, stimmung_terms)
    audio_score = _audio_eignung_score(text)
    popularitaet_score = _popularitaet_score(kandidat)
    gesamt = (
        genre_score * 0.30
        + popularitaet_score * 0.25
        + dauer_score * 0.15
        + lizenz_score * 0.10
        + audio_score * 0.10
        + stimmung_score * 0.05
        + duplikat_score * 0.05
    )
    kandidat.update(
        {
            "genre_score": round(genre_score, 2),
            "lizenz_score": round(lizenz_score, 2),
            "dauer_score": round(dauer_score, 2),
            "stimmung_score": round(stimmung_score, 2),
            "audio_eignung_score": round(audio_score, 2),
            "popularitaet_score": round(popularitaet_score, 2),
            "duplikat_score": round(duplikat_score, 2),
            "gesamt_score": round(gesamt, 2),
            "begruendung": _begruendung(
                genre_score,
                lizenz_score,
                dauer_score,
                stimmung_score,
                audio_score,
                popularitaet_score,
                duplikat_score,
            ),
        }
    )
    return kandidat


def _begruendung(
    genre: float,
    lizenz: float,
    dauer: float,
    stimmung: float,
    audio: float,
    popularitaet: float,
    duplikat: float,
) -> str:
    teile = []
    if genre >= 70:
        teile.append("Genre passt gut")
    if lizenz >= 70:
        teile.append("Lizenzhinweise passen")
    if dauer >= 90:
        teile.append("Dauer im Zielbereich")
    if stimmung >= 60:
        teile.append("Stimmung passt")
    if audio >= 75:
        teile.append("wahrscheinlich gute Audioquelle")
    if popularitaet >= 75:
        teile.append("hohe Reichweite")
    if duplikat < 100:
        teile.append("Duplikat-/Aehnlichkeitsrisiko")
    return "; ".join(teile) or "solider Kandidat mit gemischten Signalen"


def _dedupliziere(kandidaten: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for kandidat in kandidaten:
        key = str(kandidat.get("video_id") or kandidat.get("url") or kandidat.get("titel"))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(kandidat)
    return result


def _schreibe_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bevorzugt = [
        "rang",
        "status",
        "video_id",
        "url",
        "titel",
        "kanal",
        "beschreibung",
        "view_count",
        "like_count",
        "dauer",
        "dauer_minuten",
        "suchquery",
        "gesamt_score",
        "genre_score",
        "popularitaet_score",
        "lizenz_score",
        "dauer_score",
        "stimmung_score",
        "audio_eignung_score",
        "duplikat_score",
        "ausschlussgrund",
        "begruendung",
        "quelle",
        "eingabe_genre",
        "eingabe_stimmung",
        "eingabe_lizenz",
        "eingabe_dauer_minuten_min",
        "eingabe_dauer_minuten_max",
        "quellen_regel_version",
        "erstellt_am",
    ]
    extras = sorted({key for row in rows for key in row.keys()} - set(bevorzugt))
    fieldnames = bevorzugt + extras
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _schreibe_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _verlauf_anhaengen(path: Path, rows: list[dict[str, Any]], run_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            item = {
                "erstellt_am": _jetzt_utc(),
                "status": "top10",
                "run_name": run_name,
                "video_id": row.get("video_id"),
                "url": row.get("url"),
                "titel": row.get("titel"),
                "kanal": row.get("kanal"),
                "gesamt_score": row.get("gesamt_score"),
                "genre": row.get("eingabe_genre"),
                "stimmung": row.get("eingabe_stimmung"),
                "lizenz": row.get("eingabe_lizenz"),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _sortwert(row: dict[str, Any], key: str) -> float:

    value = _float_wert(row.get(key))
    return value if value is not None else -1.0


def _beste_sortierung(row: dict[str, Any]) -> tuple[float, float, float, float, float]:

    return (
        _sortwert(row, "view_count"),
        _sortwert(row, "gesamt_score"),
        _sortwert(row, "like_count"),
        _sortwert(row, "genre_score"),
        _sortwert(row, "audio_eignung_score"),
    )


def _kandidaten_sortierung(row: dict[str, Any]) -> tuple[int, float, float, float, float, float]:

    status = str(row.get("status") or "")
    return (
        1 if status == "kandidat" else 0,
        *_beste_sortierung(row),
    )


def _tabelle_text(value: Any, width: int) -> str:

    text = " ".join(str(value or "").split())
    if len(text) > width:
        text = text[: max(0, width - 3)] + "..."
    return text.ljust(width)


def _tabelle_zahl(value: Any, width: int) -> str:

    number = _float_wert(value)
    if number is None:
        return "-".rjust(width)
    return f"{int(number):,}".replace(",", ".").rjust(width)


def _tabelle_dauer(row: dict[str, Any]) -> str:

    seconds = _float_wert(row.get("dauer"))
    if seconds is None:
        minutes = _float_wert(row.get("dauer_minuten"))
        seconds = minutes * 60.0 if minutes is not None else None
    if seconds is None:
        return "-".rjust(8)
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        text = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        text = f"{minutes:02d}:{secs:02d}"
    return text.rjust(8)


def _schreibe_top10_tabelle(path: Path, rows: list[dict[str, Any]]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"{'Rang':>4} | {'Score':>6} | {'Aufrufe':>12} | {'Likes':>10} | "
        f"{'Dauer':>8} | {'Video-ID':11} | {'Titel':58} | {'Kanal':24} | URL"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{_tabelle_zahl(row.get('rang'), 4)} | "
            f"{_sortwert(row, 'gesamt_score'):6.1f} | "
            f"{_tabelle_zahl(row.get('view_count'), 12)} | "
            f"{_tabelle_zahl(row.get('like_count'), 10)} | "
            f"{_tabelle_dauer(row)} | "
            f"{_tabelle_text(row.get('video_id'), 11)} | "
            f"{_tabelle_text(row.get('titel'), 58)} | "
            f"{_tabelle_text(row.get('kanal'), 24)} | "
            f"{row.get('url') or ''}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_top10_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="quellen_suche.py top10",
        description="Erstellt eine Top-10-Liste passender YouTube-Videos, ohne Audio herunterzuladen.",
    )
    parser.add_argument("--genre", required=True, help="Gewuenschtes Genre, z.B. Jazz Lofi.")
    parser.add_argument("--stimmung", default="", help="Gewuenschte Stimmung, z.B. relaxed calm.")
    parser.add_argument("--dauer-minuten-min", type=float, default=30.0)
    parser.add_argument("--dauer-minuten-max", type=float, default=180.0)
    parser.add_argument("--lizenz", default="no copyright", help="Lizenzhinweis, der in Titel/Beschreibung gesucht wird.")
    parser.add_argument("--max-suchergebnisse", type=int, default=150)
    parser.add_argument("--min-aufrufe", type=int, default=10000, help="Mindestaufrufe, falls diese Metadaten verfuegbar sind.")
    parser.add_argument("--min-likes", type=int, default=0, help="Mindestlikes, falls diese Metadaten verfuegbar sind.")
    parser.add_argument(
        "--ohne-detail-metadaten",
        action="store_true",
        help="Keine zusaetzlichen yt-dlp-Detaildaten fuer Aufrufe/Likes laden.",
    )
    parser.add_argument("--ausschliessen-bekannte", action="store_true", help="Bekannte Videos und sehr aehnliche Treffer ausschliessen.")
    parser.add_argument("--output-name", default=None, help="Kurzname fuer Reportdateien.")
    parser.add_argument("--url", action="append", default=[], help="Optionaler manueller YouTube-Link als Kandidat.")
    parser.add_argument("--url-datei", default=None, help="Optionale Textdatei mit YouTube-Links.")
    parser.add_argument("--nur-lokale-reports", action="store_true", help="Keine Websuche, nur vorhandene lokale Reports/Links nutzen.")
    return parser.parse_args(argv)


def _manuelle_urls(args: argparse.Namespace) -> list[dict[str, Any]]:
    urls = list(args.url or [])
    if args.url_datei:
        path = Path(args.url_datei).expanduser()
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            urls.extend(re.findall(r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[A-Za-z0-9_-]{11}[^\s\"'<>)]*", text))
    kandidaten: list[dict[str, Any]] = []
    for url in urls:
        vid = _video_id(url)
        kandidaten.append(
            {
                "video_id": vid,
                "url": _youtube_url(vid, url),
                "titel": "",
                "kanal": "",
                "beschreibung": "",
                "dauer": None,
                "dauer_minuten": None,
                "view_count": None,
                "like_count": None,
                "suchquery": "manuelle_url",
                "quelle": "cli",
            }
        )
    return kandidaten


def _run_top10(argv: list[str]) -> int:
    args = _parse_top10_args(argv)
    wurzel = _projektwurzel()
    report_dir = wurzel / "daten" / "metadata" / "crawler"
    verlauf_path = report_dir / "verlauf.jsonl"
    run_name = _slug(args.output_name or f"{args.genre}_{_jetzt_utc()[:10]}")
    erstellt_am = _jetzt_utc()
    queries = _suchanfragen(args)

    kandidaten: list[dict[str, Any]] = []
    fehler: list[str] = []
    if not args.nur_lokale_reports:
        web_kandidaten, web_fehler = _yt_dlp_suche(queries, args.max_suchergebnisse)
        kandidaten.extend(web_kandidaten)
        if web_fehler:
            fehler.append(web_fehler)
    kandidaten.extend(_urls_aus_lokalen_reports(wurzel))
    kandidaten.extend(_manuelle_urls(args))
    kandidaten = _dedupliziere(kandidaten)
    if not args.ohne_detail_metadaten:


        braucht_details: list[dict[str, Any]] = []
        ohne_details: list[dict[str, Any]] = []
        for kandidat in kandidaten:
            if kandidat.get("quelle") == "ytsearch":
                if not _titel_grob_fuer_details_ok(kandidat, args):
                    ohne_details.append(kandidat)
                    continue
            braucht_details.append(kandidat)
        braucht_details = _erganze_yt_dlp_details(braucht_details, len(braucht_details))
        kandidaten = braucht_details + ohne_details

    bekannte = _bekannte_videos(verlauf_path, wurzel)
    kandidaten_report: list[dict[str, Any]] = []
    auswahlfaehig: list[dict[str, Any]] = []
    for kandidat in kandidaten:
        bekannt, grund = _ist_aehnlich_bekannt(kandidat, bekannte)
        status = "kandidat"
        duplikat_score = 100.0
        ausschlussgrund = ""
        if bekannt:
            duplikat_score = 0.0 if grund in {"video_id_bekannt", "url_bekannt"} else 35.0
            if args.ausschliessen_bekannte:
                status = "bekannt_ausgeschlossen" if duplikat_score == 0.0 else "aehnlich_ausgeschlossen"
                ausschlussgrund = grund
        if status == "kandidat":
            lofi_ok, lofi_grund = _lofi_quelle_ok(kandidat, args)
            if not lofi_ok:
                status = "lofi_ausgeschlossen"
                ausschlussgrund = lofi_grund
        views = _float_wert(kandidat.get("view_count"))
        likes = _float_wert(kandidat.get("like_count"))
        titel_text = _text_norm(str(kandidat.get("titel") or ""))
        if status == "kandidat" and not titel_text:
            status = "metadaten_ausgeschlossen"
            ausschlussgrund = "titel_fehlt"
        if status == "kandidat" and args.min_aufrufe > 0 and views is None:
            status = "metadaten_ausgeschlossen"
            ausschlussgrund = "aufrufe_unbekannt"
        if status == "kandidat" and views is not None and views < float(args.min_aufrufe):
            status = "reichweite_ausgeschlossen"
            ausschlussgrund = f"zu_wenig_aufrufe:{int(views)}"
        if status == "kandidat" and args.min_likes > 0 and likes is None:
            status = "metadaten_ausgeschlossen"
            ausschlussgrund = "likes_unbekannt"
        if status == "kandidat" and likes is not None and likes < float(args.min_likes):
            status = "reichweite_ausgeschlossen"
            ausschlussgrund = f"zu_wenig_likes:{int(likes)}"
        row = dict(kandidat)
        row.update(
            {
                "status": status,
                "ausschlussgrund": ausschlussgrund,
                "eingabe_genre": args.genre,
                "eingabe_stimmung": args.stimmung,
                "eingabe_lizenz": args.lizenz,
                "eingabe_dauer_minuten_min": args.dauer_minuten_min,
                "eingabe_dauer_minuten_max": args.dauer_minuten_max,
                "quellen_regel_version": QUELLEN_REGEL_VERSION,
                "erstellt_am": erstellt_am,
            }
        )
        row = _bewerte_kandidat(row, args, duplikat_score)
        kandidaten_report.append(row)
        if status == "kandidat":
            auswahlfaehig.append(row)

    auswahlfaehig.sort(key=_beste_sortierung, reverse=True)
    kandidaten_report.sort(key=_kandidaten_sortierung, reverse=True)
    top10 = []
    for index, row in enumerate(auswahlfaehig[:TOP_AUSWAHL_LIMIT], start=1):
        chosen = dict(row)
        chosen["rang"] = index
        chosen["status"] = "top5"
        top10.append(chosen)

    top10_json = report_dir / f"top10_{run_name}.json"
    top10_csv = report_dir / f"top10_{run_name}.csv"
    top10_txt = report_dir / f"top10_{run_name}.txt"
    kandidaten_json = report_dir / f"kandidaten_{run_name}.json"
    kandidaten_csv = report_dir / f"kandidaten_{run_name}.csv"

    payload = {
        "run_name": run_name,
        "erstellt_am": erstellt_am,
        "quellen_regel_version": QUELLEN_REGEL_VERSION,
        "eingabe": {
            "genre": args.genre,
            "stimmung": args.stimmung,
            "dauer_minuten_min": args.dauer_minuten_min,
            "dauer_minuten_max": args.dauer_minuten_max,
            "lizenz": args.lizenz,
            "max_suchergebnisse": args.max_suchergebnisse,
            "min_aufrufe": args.min_aufrufe,
            "min_likes": args.min_likes,
            "ausschliessen_bekannte": bool(args.ausschliessen_bekannte),
            "detail_metadaten": not bool(args.ohne_detail_metadaten),
        },
        "suchanfragen": queries,
        "fehler": fehler,
        "auswahl_limit": TOP_AUSWAHL_LIMIT,
        "kandidaten_gesamt": len(kandidaten_report),
        "kandidaten_auswahlfaehig": len(auswahlfaehig),
        "top10_count": len(top10),
        "top10": top10,
    }
    _schreibe_json(top10_json, payload)
    _schreibe_csv(top10_csv, top10)
    _schreibe_top10_tabelle(top10_txt, top10)
    _schreibe_json(
        kandidaten_json,
        {
            "run_name": run_name,
            "quellen_regel_version": QUELLEN_REGEL_VERSION,
            "kandidaten": kandidaten_report,
            "fehler": fehler,
        },
    )
    _schreibe_csv(kandidaten_csv, kandidaten_report)
    _verlauf_anhaengen(verlauf_path, top10, run_name)

    print("Top-5-Suche abgeschlossen.")
    print(f"Kandidaten gesamt: {len(kandidaten_report)}")
    print(f"Auswahlfaehig: {len(auswahlfaehig)}")
    print(f"Top-5: {len(top10)}")
    if fehler:
        print("Hinweise/Fehler:")
        for item in fehler:
            print(f"- {item}")
    print(f"Top-5 CSV: {top10_csv}")
    print(f"Top-5 Tabelle: {top10_txt}")
    print(f"Top-5 JSON: {top10_json}")
    print(f"Kandidaten CSV: {kandidaten_csv}")
    print(f"Verlauf: {verlauf_path}")
    return 0 if top10 else 1


GENRE_PLAN: list[dict[str, Any]] = [
    {
        "genre": "Chill Lofi",
        "anteil_prozent": 25,
        "stimmung": "calm relaxed warm study mood",
        "zweck": "Basisstil und ruhige Drums",
    },
    {
        "genre": "Jazz Lofi",
        "anteil_prozent": 20,
        "stimmung": "relaxed calm piano sax rhodes",
        "zweck": "Harmonie, Akkorde, Sax/Piano",
    },
    {
        "genre": "Piano Lofi",
        "anteil_prozent": 15,
        "stimmung": "soft mellow piano study",
        "zweck": "Melodie und harmonische Stabilitaet",
    },
    {
        "genre": "Study Lofi",
        "anteil_prozent": 15,
        "stimmung": "focus warm mellow no vocals",
        "zweck": "lange stabile Passagen",
    },
    {
        "genre": "Dreamy Lofi",
        "anteil_prozent": 10,
        "stimmung": "soft warm night calm dreamy",
        "zweck": "Atmosphaere und weiche Uebergaenge",
    },
    {
        "genre": "Chillhop",
        "anteil_prozent": 10,
        "stimmung": "warm drums mellow groove",
        "zweck": "mehr Rhythmik ohne zu aggressiv zu werden",
    },
    {
        "genre": "Boom Bap Lofi",
        "anteil_prozent": 3,
        "stimmung": "soft boom bap drums mellow groove",
        "zweck": "dosierte Drum-Variation",
    },
    {
        "genre": "Ambient Lofi",
        "anteil_prozent": 2,
        "stimmung": "dreamy atmospheric calm texture",
        "zweck": "ruhige Flaechen fuer Longform",
    },
]


def _parse_genres_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="quellen_suche.py genres",
        description="Sucht Top-10-Quellen fuer mehrere Lofi-Untergenres.",
    )
    parser.add_argument("--run-name", default=f"lofi_genres_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--genres", default="", help="Optional: kommaseparierte Genre-Auswahl.")
    parser.add_argument("--lizenz", default="no copyright creative commons royalty free")
    parser.add_argument("--dauer-minuten-min", type=float, default=20.0)
    parser.add_argument("--dauer-minuten-max", type=float, default=180.0)
    parser.add_argument("--max-suchergebnisse", type=int, default=60)
    parser.add_argument("--bekannte-erlauben", action="store_true", help="Bekannte Videos nicht ausschliessen.")
    parser.add_argument("--nur-plan", action="store_true", help="Nur Genre-Plan schreiben, keine Suche starten.")
    return parser.parse_args(argv)


def _ausgewaehlte_genres(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.genres.strip():
        return GENRE_PLAN
    wanted = {part.strip().lower() for part in args.genres.split(",") if part.strip()}
    return [item for item in GENRE_PLAN if str(item["genre"]).lower() in wanted]


def _schreibe_genre_plan(report_dir: Path, run_name: str, genres: list[dict[str, Any]]) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"genre_plan_{_slug(run_name)}.csv"
    json_path = report_dir / f"genre_plan_{_slug(run_name)}.json"
    fieldnames = ["genre", "anteil_prozent", "stimmung", "zweck"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(genres)
    payload = {
        "run_name": run_name,
        "erstellt_am": datetime.now().isoformat(timespec="seconds"),
        "hinweis": "Nur Quellensuche, keine MP3-Downloads und kein Training.",
        "genres": genres,
    }
    _schreibe_json(json_path, payload)
    return csv_path, json_path


def _lese_top10_csv(path: Path, genre: dict[str, Any]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["strategie_genre"] = str(genre["genre"])
            row["strategie_anteil_prozent"] = str(genre["anteil_prozent"])
            rows.append(row)
    return rows


def _schreibe_top10_gesamt(report_dir: Path, run_name: str, rows: list[dict[str, str]]) -> Path:
    output = report_dir / f"top10_gesamt_{_slug(run_name)}.csv"
    if not rows:
        output.write_text("", encoding="utf-8")
        return output
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _run_genres(argv: list[str]) -> int:
    args = _parse_genres_args(argv)
    wurzel = _projektwurzel()
    report_dir = wurzel / "daten" / "metadata" / "crawler"
    genres = _ausgewaehlte_genres(args)
    if not genres:
        print("Keine passenden Genres ausgewaehlt.", file=sys.stderr)
        return 2

    plan_csv, plan_json = _schreibe_genre_plan(report_dir, args.run_name, genres)
    print("Genre-Plan")
    print("----------")
    print(f"CSV:  {plan_csv}")
    print(f"JSON: {plan_json}")
    for item in genres:
        print(f"- {item['genre']}: {item['anteil_prozent']}% | {item['stimmung']}")
    if args.nur_plan:
        print("\nNur Plan: Es wurde keine Suche gestartet.")
        return 0

    alle_rows: list[dict[str, str]] = []
    fehler = 0
    run_slug = _slug(args.run_name)
    for genre in genres:
        print(f"\nSuche: {genre['genre']}")
        output_name = f"{run_slug}_{_slug(str(genre['genre']))}"
        top10_args = [
            "--genre",
            str(genre["genre"]),
            "--stimmung",
            str(genre["stimmung"]),
            "--lizenz",
            args.lizenz,
            "--dauer-minuten-min",
            str(args.dauer_minuten_min),
            "--dauer-minuten-max",
            str(args.dauer_minuten_max),
            "--max-suchergebnisse",
            str(args.max_suchergebnisse),
            "--output-name",
            output_name,
        ]
        if not args.bekannte_erlauben:
            top10_args.append("--ausschliessen-bekannte")
        result = _run_top10(top10_args)
        if result != 0:
            fehler += 1
            print(f"Warnung: Suche fuer {genre['genre']} endete mit {result}.")
        alle_rows.extend(_lese_top10_csv(report_dir / f"top10_{output_name}.csv", genre))

    gesamt_csv = _schreibe_top10_gesamt(report_dir, args.run_name, alle_rows)
    print("\nAbschluss")
    print("---------")
    print(f"Gesamt-CSV: {gesamt_csv}")
    print(f"Top-10-Eintraege gesamt: {len(alle_rows)}")
    print("MP3-Download danach z.B.:")
    print(f".venv/bin/python code/src/Crawler/quellen_suche.py mp3 --url-datei {gesamt_csv}")
    return 0 if fehler == 0 else 1


def _parse_mp3_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="quellen_suche.py mp3",
        description="Laedt erlaubte YouTube-Quellen als MP3 herunter.",
    )
    wurzel = _projektwurzel()
    parser.add_argument("--url", action="append", default=[], help="YouTube-Link. Mehrfach nutzbar.")
    parser.add_argument("--url-datei", default="", help="Text- oder CSV-Datei mit Links.")
    parser.add_argument("--titel", default="", help="Optionaler Titel fuer einen einzelnen Link.")
    parser.add_argument("--genre", default="", help="Optionales Genre, z.B. Jazz Lofi.")
    parser.add_argument("--audio-ordner", default=str(wurzel / "daten" / "raw" / "audio" / "youtube_imports"))
    parser.add_argument(
        "--metadata",
        default=str(wurzel / "daten" / "metadata" / "downloads" / "youtube_manual_mp3_imports.jsonl"),
    )
    parser.add_argument("--format", default="mp3", choices=("mp3", "wav", "m4a"))
    parser.add_argument("--cookies", default="", help="Optional: yt-dlp Cookies-Datei.")
    parser.add_argument("--cookies-browser", default="", help="Optional: z.B. firefox oder chrome.")
    parser.add_argument("--erneut-laden", action="store_true", help="Vorhandene Dateien nicht ueberspringen.")
    parser.add_argument("--nur-test", action="store_true", help="Nur pruefen, nichts herunterladen.")
    return parser.parse_args(argv)


def _lese_url_datei_mp3(path: Path, fallback_genre: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"URL-Datei nicht gefunden: {path}")
    if path.suffix.lower() == ".csv":
        entries: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                url = str(row.get("url") or row.get("URL") or "").strip()
                if not url or url.startswith("#"):
                    continue
                entries.append(
                    {
                        "url": url,
                        "titel": str(row.get("titel") or row.get("title") or "").strip(),
                        "genre": str(row.get("genre") or row.get("eingabe_genre") or fallback_genre or "").strip(),
                    }
                )
        return entries

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        entries.append({"url": url, "titel": "", "genre": fallback_genre})
    return entries


def _mp3_titel(entry: dict[str, str]) -> str:
    titel = entry.get("titel", "").strip()
    genre = entry.get("genre", "").strip()
    if genre and titel:
        return f"{genre} - {titel}"
    return titel or genre


def _mp3_report_path() -> Path:
    wurzel = _projektwurzel()
    report_dir = wurzel / "daten" / "metadata" / "downloads"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"mp3_download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def _run_mp3(argv: list[str]) -> int:
    args = _parse_mp3_args(argv)
    wurzel = _projektwurzel()
    standard_cookies = wurzel / "code" / "configs" / "youtube_cookies.txt"
    cookie_datei = args.cookies.strip()
    if not cookie_datei and standard_cookies.exists():
        cookie_datei = str(standard_cookies)
    entries = [{"url": url, "titel": args.titel, "genre": args.genre} for url in args.url]
    if args.url_datei:
        entries.extend(_lese_url_datei_mp3(Path(args.url_datei).expanduser(), args.genre))
    if not entries:
        print("Keine URL angegeben. Nutze --url oder --url-datei.", file=sys.stderr)
        return 2

    Path(args.audio_ordner).mkdir(parents=True, exist_ok=True)
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        print(f"Import {index}/{len(entries)}", flush=True)
        titel = _mp3_titel(entry)
        import_args = [
            "--url",
            entry["url"],
            "--audio-format",
            args.format,
            "--output-audio-dir",
            args.audio_ordner,
            "--metadata-path",
            args.metadata,
            "--import-metadata-path",
            args.metadata,
            "--ich-habe-rechte",
        ]
        if titel:
            import_args.extend(["--titel", titel])
        if cookie_datei:
            import_args.extend(["--cookies", cookie_datei])
        if args.cookies_browser:
            import_args.extend(["--cookies-from-browser", args.cookies_browser])
        if args.erneut_laden:
            import_args.append("--no-skip-existing")
        if args.nur_test:
            import_args.append("--dry-run")

        print("\nMP3-Import")
        print("----------")
        print(f"URL:   {entry['url']}")
        print(f"Genre: {entry.get('genre') or '-'}")
        print(f"Titel: {titel or '-'}")
        print(f"Ziel:  {args.audio_ordner}\n")
        if index > 1:
            time.sleep(20)
        returncode = _run_mit_zeitlimit("import", import_args, IMPORT_ZEITLIMIT_SEKUNDEN)
        results.append(
            {
                "url": entry["url"],
                "titel": titel,
                "genre": entry.get("genre", ""),
                "status": "ok" if returncode == 0 else "fehler",
                "returncode": returncode,
                "audio_ordner": args.audio_ordner,
            }
        )

    report_path = _mp3_report_path()
    report_path.write_text(
        json.dumps(
            {
                "erstellt_am": _jetzt_utc(),
                "ergebnisse": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ok_count = sum(1 for item in results if item["status"] == "ok")
    print("\nAbschluss")
    print("---------")
    print(f"Erfolgreich: {ok_count}/{len(results)}")
    print(f"Report:      {report_path}")
    return 0 if ok_count == len(results) else 1


def _load_module(name: str) -> None:
    if name in sys.modules:
        return
    source = MODULES[name]
    module = types.ModuleType(name)
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(source, f"{Path(__file__).name}:{name}", "exec"), module.__dict__)

IMPORT_ZEITLIMIT_SEKUNDEN = 1800


def _run_mit_zeitlimit(command: str, args: list[str], zeitlimit_sekunden: int) -> int:
    befehl = [sys.executable, str(Path(__file__).resolve()), command, *args]
    try:
        result = subprocess.run(befehl, timeout=zeitlimit_sekunden)
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"Zeitlimit ({zeitlimit_sekunden}s) ueberschritten, ueberspringe.", flush=True)
        return 1


def _run(command: str, args: list[str]) -> int:
    for dep in DEPENDENCIES.get(command, []):
        _load_module(dep)
    old_argv = sys.argv[:]
    sys.argv = [f"{Path(__file__).name} {command}", *args]
    namespace = {
        "__name__": "__main__",
        "__file__": str(Path(__file__).resolve()),
        "__package__": "",
    }
    try:
        exec(compile(MODULES[command], f"{Path(__file__).name}:{command}", "exec"), namespace)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 0
    finally:
        sys.argv = old_argv
    return 0

def _help() -> None:
    print('Crawler-Befehle:')
    for name in MODULES:
        print(f"  {name}")
    print("  top10")
    print("  genres")
    print("  mp3")
    print("\nBeispiele:")
    print('.venv/bin/python code/src/Crawler/quellen_suche.py suche --help')
    print('.venv/bin/python code/src/Crawler/quellen_suche.py import --url <link> --ich-habe-rechte')
    print('.venv/bin/python code/src/Crawler/quellen_suche.py top10 --genre "Jazz Lofi" --stimmung "relaxed calm" --lizenz "no copyright"')
    print('.venv/bin/python code/src/Crawler/quellen_suche.py genres --nur-plan')
    print('.venv/bin/python code/src/Crawler/quellen_suche.py mp3 --url <link> --genre "Jazz Lofi"')

def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _help()
        return 0
    command = sys.argv[1]
    if command == "top10":
        return _run_top10(sys.argv[2:])
    if command == "genres":
        return _run_genres(sys.argv[2:])
    if command == "mp3":
        return _run_mp3(sys.argv[2:])
    if command not in MODULES:
        print(f"Unbekannter Befehl: {command}", file=sys.stderr)
        _help()
        return 2
    return _run(command, sys.argv[2:])

if __name__ == "__main__":
    raise SystemExit(main())
