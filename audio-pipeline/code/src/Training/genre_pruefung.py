#!/usr/bin/env python3

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np


GENRE_NAMEN = {
    "jazz lofi": "jazz_lofi",
    "jazz_lofi": "jazz_lofi",
    "chillhop lofi": "chillhop_lofi",
    "chillhop_lofi": "chillhop_lofi",
    "dreamy lofi": "dreamy_lofi",
    "dreamy_lofi": "dreamy_lofi",
    "study lofi": "study_lofi",
    "study_lofi": "study_lofi",
    "guitar lofi": "guitar_lofi",
    "guitar_lofi": "guitar_lofi",
}


def genre_schluessel(value: str) -> str:

    text = str(value or "").strip().lower().replace("-", " ")
    text = " ".join(text.split())
    if text in GENRE_NAMEN:
        return GENRE_NAMEN[text]
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
    return text.replace(" ", "_") or "general_lofi"


def _status(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"gut", "ok", "sehr gut"}:
        return "Gut"
    if text in {"schlecht", "ablehnen", "nicht gut"}:
        return "Schlecht"
    return "Pruefen"


def _normalisiere(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)


class GenrePruefer:

    def __init__(
        self,
        *,
        modell_pfad: Path,
        bewertungs_ordner: Path,
        min_ziel_aehnlichkeit: float = 0.62,
        min_genre_abstand: float = 0.02,
        min_qualitaets_abstand: float = 0.0,
        fenster_sekunden: float = 10.0,
        max_fenster: int = 6,
        device: str = "cpu",
    ) -> None:
        self.modell_pfad = modell_pfad.expanduser().resolve()
        self.bewertungs_ordner = bewertungs_ordner.expanduser().resolve()
        self.min_ziel_aehnlichkeit = float(min_ziel_aehnlichkeit)
        self.min_genre_abstand = float(min_genre_abstand)
        self.min_qualitaets_abstand = float(min_qualitaets_abstand)
        self.fenster_sekunden = float(fenster_sekunden)
        self.max_fenster = max(1, int(max_fenster))
        self.device_name = str(device)
        self.verfuegbar = False
        self.fehler = ""
        self.genre_zentren: dict[str, np.ndarray] = {}
        self.positives_zentrum: np.ndarray | None = None
        self.negatives_zentrum: np.ndarray | None = None
        self.referenz_anzahl: dict[str, int] = {}
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._lade()

    def _lade(self) -> None:

        try:
            import torch
            from transformers import ClapModel, ClapProcessor

            if not self.modell_pfad.is_dir():
                raise FileNotFoundError(f"Lokales CLAP-Modell fehlt: {self.modell_pfad}")
            self._torch = torch
            self._processor = ClapProcessor.from_pretrained(
                str(self.modell_pfad), local_files_only=True
            )
            self._model = ClapModel.from_pretrained(
                str(self.modell_pfad), local_files_only=True
            ).to(self.device_name)
            self._model.eval()
            self._baue_referenzen()
            if len(self.genre_zentren) < 2:
                raise RuntimeError(
                    "Zu wenige positiv bewertete Genres fuer einen Genrevergleich."
                )
            self.verfuegbar = True
        except Exception as exc:
            self.fehler = f"{type(exc).__name__}: {exc}"
            self.verfuegbar = False
            self.schliessen()

    def _bewertungszeilen(self) -> list[dict[str, str]]:
        csv_path = self.bewertungs_ordner / "bewertung.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Bewertung fehlt: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _audio_pfad(self, row: dict[str, str]) -> Path:
        path = Path(str(row.get("Datei") or "").strip()).expanduser()
        if not path.is_absolute():
            path = self.bewertungs_ordner / path
        return path.resolve()

    def _baue_referenzen(self) -> None:
        positive: dict[str, list[np.ndarray]] = {}
        alle_guten: list[np.ndarray] = []
        alle_schlechten: list[np.ndarray] = []
        for row in self._bewertungszeilen():
            status = _status(row.get("Status", ""))
            if status not in {"Gut", "Schlecht"}:
                continue
            path = self._audio_pfad(row)
            if not path.is_file():
                continue
            embedding = self._audio_embedding(path, max_fenster=3)
            if status == "Gut":
                genre = genre_schluessel(row.get("Genre", ""))
                positive.setdefault(genre, []).append(embedding)
                alle_guten.append(embedding)
            else:
                alle_schlechten.append(embedding)

        self.genre_zentren = {
            genre: _normalisiere(np.mean(items, axis=0))
            for genre, items in positive.items()
            if items
        }
        self.referenz_anzahl = {genre: len(items) for genre, items in positive.items()}
        if alle_guten:
            self.positives_zentrum = _normalisiere(np.mean(alle_guten, axis=0))
        if alle_schlechten:
            self.negatives_zentrum = _normalisiere(np.mean(alle_schlechten, axis=0))

    def _fenster(self, path: Path, max_fenster: int | None = None) -> list[np.ndarray]:

        import librosa

        sample_rate = 48000
        audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
        frames = max(1, int(round(self.fenster_sekunden * sample_rate)))
        if len(audio) < frames:
            audio = np.pad(audio, (0, frames - len(audio)))
        possible = max(1, int(np.ceil(len(audio) / frames)))
        count = min(possible, max_fenster or self.max_fenster)
        if count == 1:
            starts = [0]
        else:
            starts = np.linspace(0, max(0, len(audio) - frames), count).astype(int).tolist()
        windows = []
        for start in starts:
            window = np.asarray(audio[start : start + frames], dtype=np.float32)
            if len(window) < frames:
                window = np.pad(window, (0, frames - len(window)))
            windows.append(window)
        return windows

    def _fenster_embeddings(
        self, path: Path, max_fenster: int | None = None
    ) -> np.ndarray:
        windows = self._fenster(path, max_fenster=max_fenster)
        inputs = self._processor(
            audios=windows,
            sampling_rate=48000,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device_name) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._torch.no_grad():
            output = self._model.get_audio_features(**inputs)
        embeddings = output.detach().float().cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def _audio_embedding(self, path: Path, max_fenster: int | None = None) -> np.ndarray:
        return _normalisiere(
            np.mean(self._fenster_embeddings(path, max_fenster=max_fenster), axis=0)
        )

    def pruefe(self, path: Path, ziel_genre: str) -> dict[str, Any]:

        if not self.verfuegbar:
            return {
                "semantic_status": "UNAVAILABLE",
                "semantic_warning": self.fehler or "CLAP-Pruefung nicht verfuegbar",
                "semantic_penalty": 0.0,
            }

        ziel = genre_schluessel(ziel_genre)
        if ziel not in self.genre_zentren:
            return {
                "semantic_status": "UNAVAILABLE",
                "semantic_warning": f"Keine guten Referenzen fuer {ziel}",
                "semantic_penalty": 0.0,
            }

        windows = self._fenster_embeddings(path)
        target_center = self.genre_zentren[ziel]
        target_values = windows @ target_center
        mean_embedding = _normalisiere(np.mean(windows, axis=0))
        genre_scores = {
            genre: float(mean_embedding @ center)
            for genre, center in self.genre_zentren.items()
        }
        competitors = [
            score for genre, score in genre_scores.items() if genre != ziel
        ]
        target_similarity = float(genre_scores[ziel])
        competitor_similarity = max(competitors) if competitors else target_similarity
        genre_margin = target_similarity - competitor_similarity
        predicted = []
        for embedding in windows:
            scores = {
                genre: float(embedding @ center)
                for genre, center in self.genre_zentren.items()
            }
            predicted.append(max(scores, key=scores.get))
        match_ratio = sum(item == ziel for item in predicted) / len(predicted)

        quality_margin = 0.0
        if self.positives_zentrum is not None and self.negatives_zentrum is not None:
            quality_margin = float(
                mean_embedding @ self.positives_zentrum
                - mean_embedding @ self.negatives_zentrum
            )

        reasons = []
        if target_similarity < self.min_ziel_aehnlichkeit:
            reasons.append("zu geringe Aehnlichkeit zu guten Genre-Referenzen")
        if genre_margin < self.min_genre_abstand:
            reasons.append("anderem Lofi-Genre naeher als dem Zielgenre")
        if quality_margin < self.min_qualitaets_abstand:
            reasons.append("naeher an schlecht bewerteten Referenzen")
        if match_ratio < 0.5:
            reasons.append("Genre nicht ueber die gesamte Audio stabil")

        penalty = (
            max(0.0, self.min_ziel_aehnlichkeit - target_similarity) * 10.0
            + max(0.0, self.min_genre_abstand - genre_margin) * 12.0
            + max(0.0, self.min_qualitaets_abstand - quality_margin) * 8.0
            + max(0.0, 0.5 - match_ratio) * 3.0
        )
        return {
            "semantic_status": "OK" if not reasons else "PROBLEM",
            "semantic_target_genre": ziel,
            "semantic_target_similarity": round(target_similarity, 5),
            "semantic_competitor_similarity": round(competitor_similarity, 5),
            "semantic_genre_margin": round(genre_margin, 5),
            "semantic_quality_margin": round(quality_margin, 5),
            "semantic_genre_match_ratio": round(match_ratio, 5),
            "semantic_predicted_genres": ";".join(predicted),
            "semantic_window_count": len(windows),
            "semantic_penalty": round(penalty, 5),
            "semantic_reasons": "; ".join(reasons),
            "semantic_warning": "",
        }

    def bericht(self) -> dict[str, Any]:

        return {
            "enabled": True,
            "available": self.verfuegbar,
            "error": self.fehler or None,
            "model_path": str(self.modell_pfad),
            "review_dir": str(self.bewertungs_ordner),
            "positive_references_by_genre": self.referenz_anzahl,
            "minimum_target_similarity": self.min_ziel_aehnlichkeit,
            "minimum_genre_margin": self.min_genre_abstand,
            "minimum_quality_margin": self.min_qualitaets_abstand,
            "window_seconds": self.fenster_sekunden,
            "maximum_windows": self.max_fenster,
            "device": self.device_name,
            "role": "automatischer Vorfilter und Ranking; Mensch entscheidet final",
        }

    def schliessen(self) -> None:

        self._model = None
        self._processor = None
        gc.collect()
        try:
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> int:

    import argparse

    projekt = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Audio lokal mit Genre-Referenzen vergleichen.")
    parser.add_argument("audio")
    parser.add_argument("--genre", required=True)
    parser.add_argument(
        "--modell",
        default=str(projekt / "daten" / "modelle" / "audio_analyse" / "clap_htsat_unfused"),
    )
    parser.add_argument(
        "--bewertung",
        default=str(
            projekt
            / "training"
            / "bewertungen"
            / "musicgen"
            / "lora_review_001"
        ),
    )
    args = parser.parse_args()
    pruefer = GenrePruefer(
        modell_pfad=Path(args.modell),
        bewertungs_ordner=Path(args.bewertung),
    )
    result = {"configuration": pruefer.bericht()}
    if pruefer.verfuegbar:
        result["result"] = pruefer.pruefe(Path(args.audio).expanduser().resolve(), args.genre)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if pruefer.verfuegbar else 2


if __name__ == "__main__":
    raise SystemExit(main())
