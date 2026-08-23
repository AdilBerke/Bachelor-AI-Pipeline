#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import json
import math
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "code" / "src" / "Pipeline" / "projekt_ablauf.py"
CLIPS_LORA_PIPELINE = ROOT / "code" / "src" / "Pipeline" / "clips_vorbereiten.py"
QUELLEN_LORA_SUCHE = ROOT / "code" / "src" / "Crawler" / "quellen_finden.py"
ZIELDATENSATZ = ROOT / "code" / "src" / "Dataset" / "zieldatensatz.py"
LORA_GENRES_CONFIG = ROOT / "code" / "configs" / "lora_genres.json"
CLIPS_LORA_SUMMARY = ROOT / "daten" / "processed" / "lora_training" / "dataset_summary.json"
QUALITAETS_DATASET = ROOT / "daten" / "processed" / "lora_training_geprueft"
QUALITAETS_SUMMARY = QUALITAETS_DATASET / "dataset_summary.json"
QUALITAETS_PIPELINE = ROOT / "code" / "src" / "Dataset" / "trainingsdaten_pruefen.py"
FEATURE_EXTRACT = ROOT / "code" / "src" / "Merkmale" / "audio_merkmale.py"
LORA_TRAINING = ROOT / "code" / "src" / "Training" / "lora.py"
TESTAUDIO_BEWERTUNG = ROOT / "code" / "src" / "Training" / "bewertung_audios_erstellen.py"
TRAININGSCLIP_BEWERTUNG = ROOT / "code" / "src" / "Training" / "trainingsclips_bewertung.py"
AUDIO_LOOPING = ROOT / "code" / "src" / "Training" / "clips_loopen.py"
REFERENZ_VERGLEICH = ROOT / "code" / "src" / "Training" / "referenz_vergleich.py"
LORA_RUN = ROOT / "training" / "musicgen" / "lora_training"
LORA_ADAPTER = LORA_RUN / "adapter.pt"
LORA_PLAN = LORA_RUN / "training_plan.json"
LORA_LOG = LORA_RUN / "training_konsole.log"
LORA_STAND = LORA_RUN / "stand.json"
FREIGABE_ARGUMENT = "--generierung-starten"
RHYTHMUS_ANTEIL_PROZENT = 5.0
MIN_RHYTHMUS_SEKUNDEN = 90.0


STANDARD_BASIS_ARGUMENTE = [
    "--stufen",
    "audio_generieren",
    "--dauer",
    "5m",
    "--genre",
    "Chill Lofi",
    "--ziel-bpm",
    "78",
    "--bpm-toleranz",
    "4",
    "--uebergang-bpm-toleranz",
    "2.5",
    "--stimmung",
    "calm controlled relaxed lofi mood",
    "--instrumente",
    "mellow piano, light low end, soft drums, subtle vinyl texture",
    "--kandidaten-pro-abschnitt",
    "3",
    "--block-sekunden",
    "0",
    "--rhythmus-anteil-prozent",
    str(RHYTHMUS_ANTEIL_PROZENT),
    "--min-rhythmus-sekunden",
    str(MIN_RHYTHMUS_SEKUNDEN),
    "--block-variation-sekunden",
    "0",
    "--block-looping-aktiv",
    "--kontinuierliche-bloecke-deaktivieren",
    "--erweiterungs-schritt-sekunden",
    "12",
    "--crossfade-sekunden",
    "3",
    "--anschluss-conditioning-deaktivieren",
    "--anschluss-sekunden",
    "8",
    "--temperature",
    "0.72",
    "--top-k",
    "80",
    "--top-p",
    "0.0",
    "--cfg-coef",
    "4.0",
    "--vram-limit-fraction",
    "0.80",
    "--sekunden-pruefung-aktiv",
    "--min-sekunden-rms-db",
    "-52",
    "--genre-pruefung-aktiv",
    "--mp3-referenz-pruefung-aktiv",
    "--referenzen-pro-genre",
    "20",
    "--referenz-score-limit",
    "18",
    "--live-status-sekunden",
    "1",
    "--max-generierte-kandidaten",
    "0",
]


def slug(text: object) -> str:

    value = str(text or "").strip().lower()
    value = value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "lofi"


def lese_lora_genres() -> dict[str, dict[str, object]]:

    if not LORA_GENRES_CONFIG.exists():
        return {}
    try:
        payload = json.loads(LORA_GENRES_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    genres = payload.get("genres") if isinstance(payload, dict) else None
    return genres if isinstance(genres, dict) else {}


def schreibe_lora_genres(genres: dict[str, dict[str, object]]) -> None:

    LORA_GENRES_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    payload = {"genres": genres}
    LORA_GENRES_CONFIG.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def liste_genres_text() -> list[str]:

    genres = lese_lora_genres()
    return [f"{key}: {info.get('label', key)}" for key, info in genres.items()]


def liste_aus_text(value: str) -> list[str]:

    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def genre_key_aus_label(label: str) -> str:

    text = str(label or "").strip()
    if "lofi" not in text.lower().replace("-", ""):
        text = f"{text} Lofi"
    key = slug(text)
    return key if key.endswith("_lofi") else f"{key}_lofi"


def starte_genre_hinzufuegen(argumente: list[str]) -> int:

    genres = lese_lora_genres()
    name = (
        lese_argument_wert(argumente, "--name")
        or lese_argument_wert(argumente, "--genre")
        or frage_wert("Neues Genre", "Rainy Chill Lofi")
    )
    label = name if "lofi" in name.lower().replace("-", "") else f"{name} Lofi"
    key = lese_argument_wert(argumente, "--key") or genre_key_aus_label(label)
    caption = lese_argument_wert(argumente, "--caption") or frage_wert(
        "Prompt/Caption",
        f"{label.lower()} instrumental, calm controlled lofi mood, soft drums, light low end, no vocals",
    )
    prefixes = liste_aus_text(lese_argument_wert(argumente, "--prefixes")) or [label.lower()]
    label_basis = label.lower().replace(" lofi", "").strip()
    keywords = liste_aus_text(lese_argument_wert(argumente, "--keywords")) or [label.lower(), label_basis]
    required = liste_aus_text(lese_argument_wert(argumente, "--required")) or keywords

    genres[key] = {
        "label": label,
        "caption": caption,
        "prefixes": prefixes,
        "keywords": keywords,
        "required_terms": required,
    }
    schreibe_lora_genres(genres)

    print("Genre gespeichert", flush=True)
    print("================", flush=True)
    print(f"Name: {label}", flush=True)
    print(f"Key:  {key}", flush=True)
    print("", flush=True)
    print("Naechste Befehle:", flush=True)
    print(f'.venv/bin/python code/start.py --top10 --genre "{label}"', flush=True)
    print(f".venv/bin/python code/start.py --clips-5000 --mit-downloads --genres {key} --ziel-clips 1000", flush=True)
    return 0


def starte_genres_anzeigen() -> int:

    print("LoRA-Genres", flush=True)
    print("===========", flush=True)
    for line in liste_genres_text():
        print(line, flush=True)
    return 0


def clips_lora_bereit() -> bool:
    if not CLIPS_LORA_SUMMARY.exists():
        return False
    try:
        summary = json.loads(CLIPS_LORA_SUMMARY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    methode = str(summary.get("method") or "")
    target = int(summary.get("target_total") or 5000)
    return (
        bool(summary.get("training_ready"))
        and bool(summary.get("source_disjoint"))
        and bool(summary.get("minimum_independent_sources_met"))
        and methode.startswith("mp3_import_strict_lofi_source_disjoint")
        and int(summary.get("selected_total") or 0) >= target
    )


def clips_lora_stand() -> str:
    if not CLIPS_LORA_SUMMARY.exists():
        return "0/5000"
    try:
        summary = json.loads(CLIPS_LORA_SUMMARY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unbekannt"
    selected = int(summary.get("selected_total") or 0)
    target = int(summary.get("target_total") or 5000)
    missing = int(summary.get("missing_total") or max(0, target - selected))
    if not summary.get("source_disjoint"):
        return f"{selected}/{target} Clips, aber alte Split-Aufteilung (Neubau erforderlich)"
    return f"{selected}/{target} Clips, fehlen {missing}"


def qualitaetsdataset_bereit() -> bool:
    if not QUALITAETS_SUMMARY.exists():
        return False
    try:
        summary = json.loads(QUALITAETS_SUMMARY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        bool(summary.get("training_ready"))
        and bool(summary.get("source_disjoint"))
        and bool(summary.get("minimum_independent_sources_met"))
        and int(summary.get("selected_total") or 0) > 0
    )


def lora_dataset_root() -> Path:
    return CLIPS_LORA_SUMMARY.parent


def lora_dataset_stand() -> str:
    summary_path = lora_dataset_root() / "dataset_summary.json"
    if not summary_path.exists():
        return clips_lora_stand()
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unbekannt"
    selected = int(summary.get("selected_total") or 0)
    rejected = int(summary.get("rejected_total") or 0)
    target = int(summary.get("target_total") or selected)
    missing = int(summary.get("missing_total") or max(0, target - selected))
    return f"{selected}/{target} Clips, fehlen {missing}, {rejected} ausgeschlossen"


def lora_ziel_clips() -> int:

    summary_path = lora_dataset_root() / "dataset_summary.json"
    if not summary_path.exists():
        return 5000
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 5000
    return int(summary.get("target_total") or summary.get("selected_total") or 5000)


def lora_ziel_steps(train_clips: int) -> int:

    return max(1, int(math.ceil(max(1, train_clips) / 8.0)))


def lade_lora_stand() -> dict[str, object]:
    if not LORA_STAND.exists():
        return {}
    try:
        payload = json.loads(LORA_STAND.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def projektpfad(value: object) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def aktiver_lora_adapter() -> Path:
    stand = lade_lora_stand()
    keys = ("adapter", "best_adapter", "checkpoint") if stand.get("status") == "freigegeben" else ("best_adapter",)
    for key in keys:
        value = stand.get(key)
        if not value:
            continue
        path = projektpfad(value)
        if path.exists():
            return path
    return LORA_ADAPTER


def aktiver_lora_run_name() -> str:
    stand = lade_lora_stand()
    run_name = str(stand.get("aktiver_run") or stand.get("run_name") or "").strip()
    return run_name or "lora"


def qualitaets_stand() -> str:
    if not QUALITAETS_SUMMARY.exists():
        return "fehlt"
    try:
        summary = json.loads(QUALITAETS_SUMMARY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unbekannt"
    selected = int(summary.get("selected_total") or 0)
    target = int(summary.get("target_total") or 5000)
    missing = int(summary.get("missing_total") or max(0, target - selected))
    rejected = int(summary.get("rejected_total") or 0)
    ready = "bereit" if summary.get("training_ready") else "nicht bereit"
    return f"{selected}/{target} Clips, fehlen {missing}, {rejected} ausgeschlossen ({ready})"


def lora_adapter_bereit() -> bool:
    adapter = aktiver_lora_adapter()
    if not adapter.is_file():
        return False
    stand = lade_lora_stand()
    if stand and stand.get("status") != "freigegeben":
        best = stand.get("best_adapter")
        if best:
            try:
                return adapter.resolve() == projektpfad(best).resolve()
            except Exception:
                return False
        return False
    if stand.get("status") == "freigegeben":
        return True
    if not LORA_PLAN.is_file():
        return False
    try:
        plan = json.loads(LORA_PLAN.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return plan.get("status") == "finished" and int(plan.get("returncode") or 0) == 0


def lora_status_text() -> str:
    if prozess_laeuft("musicgen_steuerung.py train"):
        step_text = ""
        if LORA_LOG.is_file():
            try:
                matches = re.findall(
                    r"LoRA Step\s+(\d+)/(\d+)",
                    LORA_LOG.read_text(encoding="utf-8", errors="replace"),
                )
            except OSError:
                matches = []
            if matches:
                step, target = matches[-1]
                step_text = f" (step {step}/{target})"
        return f"Training laeuft{step_text}"
    stand = lade_lora_stand()
    if stand.get("status") == "freigegeben" and aktiver_lora_adapter().is_file():
        step = stand.get("checkpoint_step") or "unbekannt"
        run_name = stand.get("aktiver_run") or stand.get("run_name") or "lora"
        return f"bereit (Step {step}, {run_name})"
    if stand.get("status") == "bewertung_offen":
        step = stand.get("checkpoint_step") or "unbekannt"
        return f"Bewertung offen (Kandidat Step {step})"
    if lora_adapter_bereit():
        return "bereit"
    if aktiver_lora_adapter().is_file():
        return "Zwischencheckpoint vorhanden, noch nicht freigegeben"
    return "fehlt"


def melde_fehlenden_lora_adapter() -> int:
    print("Status:    Audioerzeugung nicht gestartet.", flush=True)
    print("Grund:     LoRA mit dem aktiven Clip-Dataset ist noch nicht abgeschlossen.", flush=True)
    print(f"Dataset:   {clips_lora_stand()}", flush=True)
    print(f"Erwartet:  {aktiver_lora_adapter().relative_to(ROOT)}", flush=True)
    print("", flush=True)
    print("Das LoRA-Dataset ist vorbereitet, LoRA muss", flush=True)
    print("aber erst den geplanten Lauf vollstaendig abschliessen:", flush=True)
    print(".venv/bin/python code/start.py --lora-fortsetzen", flush=True)
    return 5


def zeige_status_und_befehle() -> None:
    print("MusicGen Projekt")
    print("===============")
    print(f"LoRA Dataset: {clips_lora_stand()}")
    print(f"LoRA Adapter: {lora_status_text()}")
    print("")
    print("Lokale Starts:")
    print(".venv/bin/python code/start.py --clips-5000")
    print(".venv/bin/python code/start.py --clips-5000 --mit-downloads")
    print(".venv/bin/python code/start.py --genres-anzeigen")
    print('.venv/bin/python code/start.py --genre-hinzufuegen --name "Rainy Chill Lofi"')
    print(".venv/bin/python code/start.py --trainingsdaten")
    print(".venv/bin/python code/start.py --merkmale")
    print(".venv/bin/python code/start.py --referenzvergleich")
    print(".venv/bin/python code/start.py --lora-training")
    print(".venv/bin/python code/start.py --lora-fortsetzen")
    print(".venv/bin/python code/start.py --lora-weitere-500")
    print(".venv/bin/python code/start.py --testaudios")
    print(".venv/bin/python code/start.py --trainingsclip-test")
    print(".venv/bin/python code/start.py --lora-freigeben --checkpoint PFAD")
    print('.venv/bin/python code/start.py --top10 --genre "Jazz Lofi"')
    print(".venv/bin/python code/start.py")
    print(".venv/bin/python code/start.py --audio-loopen --quelle DATEI --dauer 5m")
    print("")
    print("Hinweis: Ohne Argumente werden neue MusicGen-Clips mit LoRA erzeugt.")


def prozess_laeuft(marker: str) -> bool:
    try:
        output = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except Exception:
        return False
    eigener_pid = str(__import__("os").getpid())
    for line in output.splitlines():
        if eigener_pid in line:
            continue
        if marker in line and "rg -i" not in line:
            return True
    return False


def lese_argument_wert(argumente: list[str], name: str) -> str:
    if name not in argumente:
        return ""
    index = argumente.index(name)
    if index + 1 >= len(argumente):
        return ""
    return argumente[index + 1]


def aktualisiere_lora_zielreport(argumente: list[str]) -> None:

    if not ZIELDATENSATZ.exists():
        return
    if "--ziel-clips" not in argumente:
        return
    ziel_clips = lese_argument_wert(argumente, "--ziel-clips") or "5000"
    genres = lese_argument_wert(argumente, "--genres")
    seed = lese_argument_wert(argumente, "--seed") or "4027"
    max_pro_quelle = lese_argument_wert(argumente, "--max-pro-quelle-pro-genre") or "250"
    command = [
        sys.executable,
        str(ZIELDATENSATZ),
        "--overwrite",
        "--ziel-clips",
        ziel_clips,
        "--seed",
        seed,
        "--max-pro-quelle-pro-genre",
        max_pro_quelle,
    ]
    if genres:
        command.extend(["--genres", genres])
    log_dir = ROOT / "daten" / "metadata" / "crawler" / "logs" / "zielstand"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"zielstand_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)


def suche_fehlende_quellen(argumente: list[str]) -> int:
    if not QUELLEN_LORA_SUCHE.exists():
        print(f"Quellensuche nicht gefunden: {QUELLEN_LORA_SUCHE}", file=sys.stderr)
        return 2
    aktualisiere_lora_zielreport(argumente)
    command = [sys.executable, str(QUELLEN_LORA_SUCHE)]
    genres = lese_argument_wert(argumente, "--genres")
    if genres:
        command.extend(["--genres", genres])
    if "--bekannte-erlauben" in argumente:
        command.append("--bekannte-erlauben")
    if "--plan" in argumente:
        command.append("--nur-plan")
    print("Schritt 1/2: Suche fehlende Quellen", flush=True)
    return subprocess.call(command, cwd=ROOT)


def starte_clips_lora(argumente: list[str]) -> int:
    if not CLIPS_LORA_PIPELINE.exists():
        print(f"Clip-Pipeline nicht gefunden: {CLIPS_LORA_PIPELINE}", file=sys.stderr)
        return 2
    mit_downloads = "--mit-downloads" in argumente
    argumente = [arg for arg in argumente if arg != "--mit-downloads"]
    standard_clip_argumente = [
        "--clip-hop-sec",
        "30",
        "--max-clips-pro-quelle",
        "250",
        "--max-pro-quelle-pro-genre",
        "250",
    ]
    if not mit_downloads:
        standard_clip_argumente.insert(0, "--lokale-mp3s")
    else:
        standard_clip_argumente.insert(0, "--mit-downloads")
        if "--max-links-pro-genre" not in argumente:
            standard_clip_argumente.extend(["--max-links-pro-genre", "10"])


    clip_argumente = [*standard_clip_argumente, *argumente]
    print("Startmodus", flush=True)
    print("==========", flush=True)
    print("Art:       LoRA-Clip-Vorbereitung", flush=True)
    print(f"Quelle:    {'Top-10-Downloads' if mit_downloads else 'lokale MP3s'}", flush=True)
    print("Ziel:      genrebalancierte LoRA-Clips", flush=True)
    print("Training:  nein", flush=True)
    if mit_downloads:
        print("Hinweis:   sucht offene Genres automatisch und laedt danach neue MP3s.", flush=True)
    print("", flush=True)
    if prozess_laeuft("clips_vorbereiten.py"):
        print("Status:    Clip-Vorbereitung laeuft bereits in einem Terminal.", flush=True)
        print("Hinweis:   Bitte erst diesen Lauf beenden lassen, dann LoRA starten.", flush=True)
        return 6
    if mit_downloads:
        such_code = suche_fehlende_quellen(argumente)
        if such_code != 0:
            print("", flush=True)
            print("Hinweis:   Neue Suche ohne direkt nutzbare Treffer.", flush=True)
            print("           Versuche vorhandene Top-10-Reports fuer offene Genres.", flush=True)
        if "--plan" not in argumente:
            print("", flush=True)
            print("Schritt 2/2: Lade MP3s und erstelle Clips", flush=True)
    return subprocess.call([sys.executable, str(CLIPS_LORA_PIPELINE), *clip_argumente], cwd=ROOT)


def starte_qualitaetsdataset(argumente: list[str]) -> int:
    if not QUALITAETS_PIPELINE.exists():
        print(f"Trainingsdaten-Datei nicht gefunden: {QUALITAETS_PIPELINE}", file=sys.stderr)
        return 2
    argumente = ["--nur-plan" if arg == "--dry-run" else arg for arg in argumente]
    print("Startmodus", flush=True)
    print("==========", flush=True)
    print("Art:       gepruefte Trainingsdaten", flush=True)
    print("Basis:     LoRA-Clips", flush=True)
    print("Filter:    schlechte Review-Quellen raus", flush=True)
    print("Training:  nein", flush=True)
    print("", flush=True)
    return subprocess.call([sys.executable, str(QUALITAETS_PIPELINE), *argumente], cwd=ROOT)


def starte_features(argumente: list[str]) -> int:
    if not FEATURE_EXTRACT.exists():
        print(f"Merkmal-Datei nicht gefunden: {FEATURE_EXTRACT}", file=sys.stderr)
        return 2
    if "--dataset-root" not in argumente:
        argumente = ["--dataset-root", str(lora_dataset_root()), *argumente]
    return subprocess.call([sys.executable, str(FEATURE_EXTRACT), *argumente], cwd=ROOT)


def starte_referenzvergleich(argumente: list[str]) -> int:

    if not REFERENZ_VERGLEICH.exists():
        print(f"Referenzvergleich nicht gefunden: {REFERENZ_VERGLEICH}", file=sys.stderr)
        return 2
    print("Referenzvergleich", flush=True)
    print("=================", flush=True)
    print("Training:  nein", flush=True)
    print("Generiert: nein", flush=True)
    print("Ziel:      MP3-Qualitaet vs. MusicGen-Ausgabe messen", flush=True)
    print("", flush=True)
    return subprocess.call([sys.executable, str(REFERENZ_VERGLEICH), *argumente], cwd=ROOT)


def starte_lora(argumente: list[str], *, weitere_runde: bool, neustart: bool = False) -> int:
    if not LORA_TRAINING.exists():
        print(f"LoRA-Datei nicht gefunden: {LORA_TRAINING}", file=sys.stderr)
        return 2
    if prozess_laeuft("code/src/Training/lora.py") or prozess_laeuft("musicgen_steuerung.py train"):
        print("Status: LoRA-Training laeuft bereits.", flush=True)
        return 6

    train_clips = lora_ziel_clips()
    ziel_steps = lora_ziel_steps(train_clips)

    print("Startmodus", flush=True)
    print("==========", flush=True)
    print("Art:       LoRA", flush=True)
    if neustart:
        print("Modus:     sauberer Neustart ab Basismodell", flush=True)
        print(f"Ziel:      {train_clips} Trainingsclips, gleichmaessig ueber Genres verteilt", flush=True)
    else:
        print("Modus:     weitere Trainingsrunde" if weitere_runde else "Modus:     fortsetzen", flush=True)
        print(f"Ziel:      aktiven LoRA-Lauf bis zum {train_clips}-Clip-Ziel fortsetzen", flush=True)
    print(f"Dataset:   {lora_dataset_stand()}", flush=True)
    print("Ausgabe:   Trainingsbeispiele + Restprozent", flush=True)
    print("", flush=True)

    if not clips_lora_bereit():
        if prozess_laeuft("clips_vorbereiten.py"):
            print("Status: Clip-Vorbereitung laeuft noch. Training startet erst danach.", flush=True)
            return 6
        print("Status: Training nicht gestartet, weil das Dataset noch nicht vollstaendig ist.", flush=True)
        print(f"Dataset: {clips_lora_stand()}", flush=True)
        print("Zuerst lokal Clips vorbereiten mit:", flush=True)
        print(".venv/bin/python code/start.py --clips-5000", flush=True)
        return 5

    basis_argumente = [
        "--dataset-root",
        str(lora_dataset_root()),
        "--run-root",
        str(ROOT / "training" / "musicgen"),
        "--run-name",
        "lora_training",
        "--resume-from",
        "auto",
        "--ziel-step",
        str(ziel_steps),
        "--steps-weiter",
        str(ziel_steps),
        "--train-clips",
        str(train_clips),
        "--learning-rate",
        "5e-6" if neustart else "1e-5",
        "--vram-limit-fraction",
        "0.80",
        "--save-steps",
        "25",
        "--eval-steps",
        "50",
        "--ausgabe",
        "kompakt",
    ]
    if neustart:
        basis_argumente.append("--ohne-resume")
    if weitere_runde:
        basis_argumente.append("--weitere-runde")
    return subprocess.call([sys.executable, str(LORA_TRAINING), *basis_argumente, *argumente], cwd=ROOT)


def starte_lora_freigabe(argumente: list[str]) -> int:
    checkpoint = lese_argument_wert(argumente, "--checkpoint") or lese_argument_wert(argumente, "--freigeben-checkpoint")
    if not checkpoint:
        stand = lade_lora_stand()
        checkpoint = str(stand.get("checkpoint") or "")
    if not checkpoint:
        print("Status: keine Freigabe moeglich, Checkpoint fehlt.", flush=True)
        print("Beispiel: .venv/bin/python code/start.py --lora-freigeben --checkpoint training/musicgen/lora_training/checkpoints/step_000625/lora_adapter.pt")
        return 4
    command = [sys.executable, str(LORA_TRAINING), "--freigeben-checkpoint", checkpoint]
    notiz = lese_argument_wert(argumente, "--notiz") or lese_argument_wert(argumente, "--freigabe-notiz")
    if notiz:
        command.extend(["--freigabe-notiz", notiz])
    return subprocess.call(command, cwd=ROOT)


def starte_testaudios(argumente: list[str]) -> int:

    if not TESTAUDIO_BEWERTUNG.exists():
        print(f"Testaudio-Datei nicht gefunden: {TESTAUDIO_BEWERTUNG}", file=sys.stderr)
        return 2
    if "--github-push" not in argumente and "--kein-github-push" not in argumente:
        argumente = [*argumente, "--github-push"]
    print("Testaudios", flush=True)
    print("==========", flush=True)
    print("Quelle:    MusicGen + aktueller LoRA-Adapter", flush=True)
    print("Ausgabe:   ein Bewertungsordner mit audio/", flush=True)
    print("GitHub:    audio-Ordner nach main", flush=True)
    print("", flush=True)
    return subprocess.call([sys.executable, str(TESTAUDIO_BEWERTUNG), *argumente], cwd=ROOT)


def starte_trainingsclip_test(argumente: list[str]) -> int:

    if not TRAININGSCLIP_BEWERTUNG.exists():
        print(f"Trainingsclip-Datei nicht gefunden: {TRAININGSCLIP_BEWERTUNG}", file=sys.stderr)
        return 2
    if "--github-push" not in argumente and "--kein-github-push" not in argumente:
        argumente = [*argumente, "--github-push"]
    print("Trainingsclip-Test", flush=True)
    print("==================", flush=True)
    print("Quelle:    echte 30s-Clips aus dem LoRA-Dataset", flush=True)
    print("Ausgabe:   ein Bewertungsordner mit audio/", flush=True)
    print("GitHub:    audio-Ordner nach main", flush=True)
    print("", flush=True)
    return subprocess.call([sys.executable, str(TRAININGSCLIP_BEWERTUNG), *argumente], cwd=ROOT)


def starte_top10_suche(argumente: list[str]) -> int:
    crawler = ROOT / "code" / "src" / "Crawler" / "quellen_suche.py"
    if not crawler.exists():
        print(f"Quellensuche nicht gefunden: {crawler}", file=sys.stderr)
        return 2
    if "--help" in argumente or "-h" in argumente:
        return subprocess.call([sys.executable, str(crawler), "top10", *argumente], cwd=ROOT)
    bekannte_erlauben = "--bekannte-erlauben" in argumente
    if bekannte_erlauben:
        argumente = [arg for arg in argumente if arg != "--bekannte-erlauben"]
    if "--genre" not in argumente:
        suchinput = frage_suchinput()
        argumente = ["--genre", suchinput, *argumente]
    if "--stimmung" not in argumente:
        argumente.extend(["--stimmung", "calm relaxed lofi mood"])
    if "--lizenz" not in argumente:
        argumente.extend(["--lizenz", "no copyright"])
    if "--ausschliessen-bekannte" not in argumente and not bekannte_erlauben:
        argumente.append("--ausschliessen-bekannte")
    print("Top-10-Suche", flush=True)
    print("============", flush=True)
    print(f"Genre: {lese_argument_wert(argumente, '--genre')}", flush=True)
    print(f"Bekannte Videos: {'erlaubt' if bekannte_erlauben else 'ausgeschlossen'}", flush=True)
    print("", flush=True)
    return subprocess.call([sys.executable, str(crawler), "top10", *argumente], cwd=ROOT)


def starte_audio_loop(argumente: list[str]) -> int:

    if not AUDIO_LOOPING.exists():
        print(f"Looping-Datei nicht gefunden: {AUDIO_LOOPING}", file=sys.stderr)
        return 2
    if prozess_laeuft("clips_loopen.py"):
        print("Status: Audio-Looping laeuft bereits.", flush=True)
        return 6
    print("Clip-Pool-Fallback", flush=True)
    print("==================", flush=True)
    print("MusicGen:   wird nicht gestartet", flush=True)
    print("LoRA:       wird nicht trainiert", flush=True)
    print("Quelle:     vorhandene Clips", flush=True)
    print("Methode:    taktnahe Loop-Grenzen", flush=True)
    print("", flush=True)
    if not argumente:
        argumente = interaktive_loop_argumente()
    return subprocess.call([sys.executable, str(AUDIO_LOOPING), *argumente], cwd=ROOT)


def frage_wert(text: str, standard: str) -> str:

    try:
        value = input(f"{text} [{standard}]: ").strip()
    except EOFError:
        value = ""
    return value or standard


def frage_suchinput() -> str:

    value = frage_wert("Suchinput / Genre", "Jazz Lofi")
    if "lofi" not in value.lower().replace("-", ""):
        value = f"{value} Lofi"
    return value


def frage_dauer() -> str:

    while True:
        value = frage_wert("Gewuenschte Laenge, z.B. 20m oder 2h", "20m")
        normalized = value.lower().replace(",", ".").replace(" ", "")
        number_text = normalized[:-1] if normalized[-1:] in {"s", "m", "h"} else normalized
        try:
            number = float(number_text)
        except ValueError:
            number = 0.0
        if number > 0.0:
            return normalized if normalized[-1:] in {"s", "m", "h"} else f"{normalized}m"
        print("Bitte eine gueltige positive Laenge eingeben, z.B. 30m oder 2h.", flush=True)


def dauer_in_sekunden(value: str) -> float:

    text = str(value).strip().lower().replace(",", ".")
    unit = text[-1]
    number = float(text[:-1] if unit in {"s", "m", "h"} else text)
    if unit == "h":
        return number * 3600.0
    if unit == "m":
        return number * 60.0
    return number


def rhythmus_block_sekunden(duration_sec: float) -> float:

    target = min(
        duration_sec,
        max(MIN_RHYTHMUS_SEKUNDEN, duration_sec * RHYTHMUS_ANTEIL_PROZENT / 100.0),
    )
    minimum = min(MIN_RHYTHMUS_SEKUNDEN, duration_sec)
    maximum_count = max(1, int(duration_sec // max(1.0, minimum)))
    ideal_count = max(1, int(round(duration_sec / max(1.0, target))))
    block_count = min(maximum_count, ideal_count)
    return duration_sec / block_count


def formatiere_zeit(seconds: float) -> str:

    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes:
        return f"{minutes}:{secs:02d} Minuten"
    return f"{secs} Sekunden"


def frage_genre() -> str:
    gespeicherte_genres = lese_lora_genres()
    labels = [str(info.get("label") or key.replace("_", " ").title()) for key, info in gespeicherte_genres.items()]
    if not labels:
        labels = ["Jazz Lofi", "Chillhop Lofi", "Dreamy Lofi", "Study Lofi", "Guitar Lofi"]
    genres = {str(index): label for index, label in enumerate(labels, start=1)}
    genres[str(len(genres) + 1)] = "Alle Lofi Genres"
    print("Genre:", flush=True)
    for number, name in genres.items():
        print(f"  {number}: {name}", flush=True)
    genre_input = frage_wert("Nummer oder Genre", "2")
    if genre_input.isdigit() and genre_input not in genres:
        print("Unbekannte Nummer; Chillhop Lofi wird verwendet.", flush=True)
        genre_input = "2"
    return genres.get(genre_input, genre_input)


def interaktive_musicgen_argumente() -> list[str]:

    print("Neue MusicGen-Audio", flush=True)
    print("====================", flush=True)
    dauer = frage_dauer()
    rhythmus_sekunden = rhythmus_block_sekunden(dauer_in_sekunden(dauer))
    genre = frage_genre()
    name_slug = "_".join(genre.lower().replace("-", " ").split())
    zeitstempel = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"pipeline_{zeitstempel}"
    final_name = f"{name_slug}_{dauer}_{zeitstempel}"

    print("", flush=True)
    print("Plan", flush=True)
    print("----", flush=True)
    print(f"Laenge:       {dauer}", flush=True)
    print(f"Genre:        {genre}", flush=True)
    print("Quelle:       neue MusicGen-Clips mit LoRA", flush=True)
    print("Alte Clips:   werden nicht verwendet", flush=True)
    print(
        f"Rhythmus:     etwa alle {formatiere_zeit(rhythmus_sekunden)} "
        f"({RHYTHMUS_ANTEIL_PROZENT:.0f}% der Gesamtdauer, mindestens "
        f"{MIN_RHYTHMUS_SEKUNDEN:.0f}s)",
        flush=True,
    )
    print("Fade-in/out:  5 Sekunden", flush=True)
    print("Uebergang:    3 Sekunden Crossfade", flush=True)
    print("Stiltreue:    lokale CLAP-Pruefung gegen bewertete Referenzen", flush=True)
    print("Qualitaet:    3 vollstaendige Kandidaten pro Rhythmusblock", flush=True)
    print("MP3-Vergleich: gute Trainingsclips als technischer Referenzstandard", flush=True)
    print("Looping:      30s MusicGen-Clips werden zu Rhythmusbloecken verlaengert", flush=True)
    print("", flush=True)

    return [
        "--stufen",
        "audio_generieren",
        "--dauer",
        dauer,
        "--genre",
        genre,
        "--run-name",
        run_name,
        "--final-name",
        final_name,
        "--best-adapter",
        str(aktiver_lora_adapter()),
        "--abschnitt-sekunden",
        "30",
        "--kandidaten-pro-abschnitt",
        "3",
        "--block-sekunden",
        "0",
        "--rhythmus-anteil-prozent",
        str(RHYTHMUS_ANTEIL_PROZENT),
        "--min-rhythmus-sekunden",
        str(MIN_RHYTHMUS_SEKUNDEN),
        "--block-variation-sekunden",
        "0",
        "--block-looping-aktiv",
        "--kontinuierliche-bloecke-deaktivieren",
        "--erweiterungs-schritt-sekunden",
        "12",
        "--crossfade-sekunden",
        "3",
        "--loop-crossfade-sekunden",
        "0",
        "--fade-in-sekunden",
        "5",
        "--fade-out-sekunden",
        "5",
        "--tempo-variation-prozent",
        "1.0",
        "--tempo-phase-sekunden",
        "300",
        "--anschluss-conditioning-deaktivieren",
        "--anschluss-sekunden",
        "8",
        "--temperature",
        "0.72",
        "--top-k",
        "80",
        "--top-p",
        "0.0",
        "--cfg-coef",
        "4.0",
        "--ziel-bpm",
        "78",
        "--bpm-toleranz",
        "4",
        "--uebergang-bpm-toleranz",
        "2.5",
        "--sekunden-pruefung-aktiv",
        "--min-sekunden-rms-db",
        "-52",
        "--genre-pruefung-aktiv",
        "--mp3-referenz-pruefung-aktiv",
        "--referenzen-pro-genre",
        "20",
        "--referenz-score-limit",
        "18",
        "--vram-limit-fraction",
        "0.80",
        "--max-generierte-kandidaten",
        "0",
        "--freigabe-langer-lauf",
        "--github-push",
        "--seed",
        "0",
    ]


def interaktive_loop_argumente() -> list[str]:

    print("Eingabe", flush=True)
    print("-------", flush=True)
    dauer = frage_dauer()
    rhythmus_sekunden = rhythmus_block_sekunden(dauer_in_sekunden(dauer))
    genre = frage_genre()
    name_slug = "_".join(genre.lower().replace("-", " ").split())
    run_name = f"{name_slug}_{dauer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return [
        "--dauer",
        dauer,
        "--genre",
        genre,
        "--name",
        run_name,
        "--block-sekunden",
        str(rhythmus_sekunden),
        "--block-variation-sekunden",
        "0",
        "--wiederholung-erlauben",
        "--crossfade-sekunden",
        "5",
        "--fade-in-sekunden",
        "5",
        "--fade-out-sekunden",
        "5",
        "--tempo-variation-prozent",
        "1.0",
        "--tempo-phase-sekunden",
        "300",
        "--seed",
        "0",
    ]


def main() -> int:
    if not PIPELINE.exists():
        print(f"Pipeline nicht gefunden: {PIPELINE}", file=sys.stderr)
        return 2
    argumente = sys.argv[1:]

    if argumente == ["--status"]:
        zeige_status_und_befehle()
        return 0

    if argumente and argumente[0] == "--genres-anzeigen":
        return starte_genres_anzeigen()

    if argumente and argumente[0] == "--genre-hinzufuegen":
        return starte_genre_hinzufuegen(argumente[1:])

    if not argumente:
        if not lora_adapter_bereit():
            return melde_fehlenden_lora_adapter()
        return subprocess.call(
            [sys.executable, str(PIPELINE), *interaktive_musicgen_argumente()],
            cwd=ROOT,
        )


    if argumente and argumente[0] == "--clips-5000":
        return starte_clips_lora(argumente[1:])

    if argumente and argumente[0] == "--trainingsdaten":
        return starte_qualitaetsdataset(argumente[1:])

    if argumente and argumente[0] == "--merkmale":
        return starte_features(argumente[1:])

    if argumente and argumente[0] == "--referenzvergleich":
        return starte_referenzvergleich(argumente[1:])

    if argumente and argumente[0] == "--lora-fortsetzen":
        return starte_lora(argumente[1:], weitere_runde=False)

    if argumente and argumente[0] == "--lora-weitere-500":
        return starte_lora(argumente[1:], weitere_runde=True)

    if argumente and argumente[0] in {"--lora-training", "--lora-neustart"}:
        return starte_lora(argumente[1:], weitere_runde=False, neustart=True)

    if argumente and argumente[0] == "--lora-freigeben":
        return starte_lora_freigabe(argumente[1:])

    if argumente and argumente[0] == "--testaudios":
        return starte_testaudios(argumente[1:])

    if argumente and argumente[0] == "--trainingsclip-test":
        return starte_trainingsclip_test(argumente[1:])

    if argumente and argumente[0] == "--top10":
        return starte_top10_suche(argumente[1:])

    if argumente and argumente[0] == "--audio-loopen":
        return starte_audio_loop(argumente[1:])


    if FREIGABE_ARGUMENT in argumente:
        argumente = [arg for arg in argumente if arg != FREIGABE_ARGUMENT]

    if argumente == ["--nur-plan"]:
        argumente = [*STANDARD_BASIS_ARGUMENTE, "--nur-plan"]


    return subprocess.call([sys.executable, str(PIPELINE), *argumente], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
