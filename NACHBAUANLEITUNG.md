# Anleitung: Projekt von Grund auf nachbauen

Schritt-für-Schritt-Anleitung, um Code, Modelle und Umgebung dieses Projekts (Audio-Pipeline
MusicGen+LoRA, Video-Pipeline LTX-Video, Studio-Website) auf einer neuen Maschine
nachzubauen. Alle Angaben sind live aus der tatsächlich laufenden Entwicklungsumgebung
ausgelesen (Stand 2026-08-23), keine Schätzungen.

**Aktueller Referenzstand:** Das aktive Audio-Modell ist LoRA-Adapter Step 625 (trainiert auf
5000 genrebalancierten Clips). Damit wurden am 22./23.08.2026 für alle 5 Genres je eine
20-Minuten-Kontrollaudio erzeugt (`Kontrolle_Jazz_20min`, `Kontrolle_UebergangFix_20min`
(Chillhop), `Kontrolle_Dreamy_20min`, `Kontrolle_Study_20min`, `Kontrolle_Guitar_20min`) —
das sind aktuell auch die einzigen auf der Website sichtbaren Audios (`/api/audio` filtert
bewusst auf den `Kontrolle_`-Namensprefix, alle anderen ~140 Test-/Entwicklungsläufe bleiben
auf der Platte, sind aber ausgeblendet).

**Bewusst nicht Teil dieser Anleitung:** die heruntergeladenen MP3-Rohquellen
(`daten/processed/musicgen_youtube_import_30s/`) und die daraus gebauten Trainings-Datensätze.
Das sind fremde, urheberrechtlich geschützte YouTube-Inhalte, die nicht mit ausgeliefert
werden — Schritt 7 beschreibt, wie man sich eine eigene, gleichwertige Quellenbasis
beschafft, ohne die konkreten Dateien zu benötigen.

## Schritt 1 — Code

Dieses Repository enthält bereits alle drei Bestandteile — Audio-Pipeline, Video-Pipeline
und Frontend. Ein einzelner Klon genügt:

```bash
git clone https://github.com/AdilBerke/Bachelor-AI-Pipeline.git
cd Bachelor-AI-Pipeline
```

Aufbau:

| Verzeichnis | Inhalt |
|---|---|
| `audio-pipeline/` | MusicGen + LoRA, Crawler, Bewertung, Backend (`code/src/…`) |
| `video-pipeline/` | LTX-Video, szenariobasiertes Training, NIQE-Bewertung, GIF-Erzeugung |
| `frontend/` | Studio-Weboberfläche (React/TanStack), inkl. `package.json` |
| `requirements.txt` | Python-Pakete beider Pipelines |
| `requirements-niqe.txt` | Pakete der getrennten NIQE-Umgebung (siehe Schritt 3b) |

Alle Pfadangaben in dieser Anleitung beziehen sich auf die Wurzel dieses Repositories.

## Schritt 2 — System-Voraussetzungen

| Voraussetzung | Getestete Version | Wofür |
|---|---|---|
| Betriebssystem | Ubuntu 22.04.5 LTS, Kernel 6.8 | — |
| NVIDIA-GPU + Treiber | RTX A6000 (48 GB VRAM), Treiber 580.173.02 | MusicGen-/LTX-Video-Training und -Generierung |
| Python | 3.11 (3.11.0rc1) | Backend, beide Pipelines |
| Node.js | 22.x (v22.22.3) | Frontend (Vite/React) |
| ffmpeg | 4.4.2 (`apt install ffmpeg`) | Audio-/Video-Dekodierung, Zusammenschnitt |

Ein separates CUDA-Toolkit ist **nicht** nötig — die CUDA-Runtime kommt über die
`nvidia-cu*`-Pakete in `requirements.txt` mit. Nur der GPU-Treiber muss auf dem System
installiert sein.

Mindestens ~16 GB freier VRAM werden für MusicGen-Melody-Large-LoRA-Training empfohlen
(siehe `vram_profile_for_hardware()` in `audio-pipeline/code/src/Training/musicgen_steuerung.py` — unter
24 GB gilt explizit als Notfallprofil, 24–39 GB als Fallback, ab 40 GB als empfohlenes Profil).

## Schritt 3 — Python-Umgebung

**3a. Haupt-Umgebung (Audio- und Videopipeline):**

```bash
cd Bachelor-AI-Pipeline
python3.11 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

`requirements.txt` (Projektwurzel) enthält alle 231 Pakete beider Pipelines, exakt gepinnt
per `pip freeze` aus der laufenden Umgebung — u. a. `audiocraft`, `transformers`, `xformers`,
`librosa`, `demucs`, `yt-dlp` (Audio) sowie `diffusers`, `ltxv_trainer` (per Git-Commit
gepinnt), `decord`, `opencv-python`, `gradio` (Video). Das `--extra-index-url` ist nötig,
weil `torch`/`torchaudio` als `+cu124`-Build referenziert sind, der nur im PyTorch-eigenen
Wheel-Index liegt, nicht auf PyPI selbst.

**3b. Zweite Umgebung für die NIQE-Bewertung (Pflicht für die Videoevaluation):**

```bash
python3.11 -m venv .venv-niqe
.venv-niqe/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-niqe.txt
```

> **Diese Pakete dürfen nicht in `.venv` installiert werden.** `pyiqa` erzwingt
> `transformers>=5`, während die Pipeline `transformers==4.52.2` benötigt. Landet `pyiqa`
> im Haupt-venv, starten `diffusers` und `ltxv_trainer` nicht mehr — die Videogenerierung
> bricht dann bereits beim Import ab.

NIQE (Naturalness Image Quality Evaluator, No-Reference-Metrik nach Mittal et al., 2013) ist
die quantitative Kennzahl für die Einzelbildqualität der erzeugten Rohvideos; niedrigere
Werte bedeuten eine geringere Abweichung vom Referenzmodell natürlicher Bildstatistiken.
Ausgewertet werden zwölf gleichmäßig über die Sequenz verteilte Frames je Video in
Originalauflösung, anschließend gemittelt.

Die Trennung ist im Code bereits umgesetzt: `evaluate_video.py` ruft die Bewertung über
`video-pipeline/lofi_pipeline/scripts/_niqe_worker.py` als Subprozess im zweiten venv auf.
Ohne Schritt 3b liefert die Videobewertung technische Kennzahlen (SSIM, Schärfe, Flackern,
Bewegung, Farbe), aber **keinen NIQE-Wert**.

Weitere NIQE-Werkzeuge:

```bash
# alle vorhandenen Videos nachträglich bewerten (MP4 + GIF)
.venv/bin/python video-pipeline/lofi_pipeline/scripts/backfill_metrics.py

# reiner Schärfe-Batchvergleich als Rangliste
.venv/bin/python video-pipeline/lofi_pipeline/scripts/schaerfe_bewertung.py
```

## Schritt 4 — Basismodelle herunterladen

Die Pipeline lädt Modelle **nicht automatisch beim ersten Gebrauch** (Ausnahme: LTX-Video,
siehe unten) — sie müssen vorher explizit lokal bereitgestellt werden.

**4a. MusicGen Melody Large (Pflicht für die Audio-Pipeline, ca. 15 GB):**

```bash
.venv/bin/python audio-pipeline/code/src/Training/setup_musicgen_melody_large.py --download
```

Lädt `facebook/musicgen-melody-large` von Hugging Face nach
`daten/modelle/musicgen/facebook_musicgen_melody_large/`. Ohne `--download` zeigt das
Skript nur den Status und den nötigen Befehl an, lädt aber nichts.

**4b. CLAP (für die semantische Genre-Prüfung, ca. 590 MB, optional aber empfohlen):**

```bash
.venv/bin/python audio-pipeline/code/src/Training/audio_modelle_einrichten.py --download clap
```

Lädt `laion/clap-htsat-unfused` nach `daten/modelle/audio_analyse/clap_htsat_unfused/`.
(Hinweis zur Aussagekraft dieser Prüfung: siehe
[`MUSIKMODELL_VERSUCHSDOKUMENTATION.md`](MUSIKMODELL_VERSUCHSDOKUMENTATION.md),
Abschnitt 10.1 — nur 40 % Trefferquote im Test, deshalb produktiv nicht aktiv.)

**4c. Demucs (optional, nur für manuelle Stem-Analyse):**

```bash
.venv/bin/python audio-pipeline/code/src/Training/audio_modelle_einrichten.py --download demucs
```

**4d. LTX-Video 13B (Video-Pipeline):** wird von der `ltxv_trainer`-Bibliothek beim ersten
Aufruf von `video-pipeline/pipeline/v003_manual/generate.py` automatisch von Hugging Face
geladen (`LtxvModelVersion.LTXV_13B_097_DEV`) und im Standard-HF-Cache abgelegt — kein
gesonderter Setup-Schritt nötig, dafür aber ein einmaliger, langsamerer erster Lauf.

**4e. RealESRGAN-Gewichte (Video-Nachbearbeitung/Upscaling):** Die Modellgewichte sind aus
Größengründen **nicht** Teil dieses Repositories. Für das KI-Upscaling der erzeugten Videos
muss `RealESRGAN_x4plus_anime_6B.pth` von der
[offiziellen Real-ESRGAN-Release-Seite](https://github.com/xinntao/Real-ESRGAN/releases)
heruntergeladen und unter `video-pipeline/lofi_pipeline/models/` abgelegt werden (Ordner ggf.
anlegen). Ohne diese Datei laufen Generierung und Bewertung normal, die Nachbearbeitung
liefert dann aber kein hochskaliertes 1920×1088-Video.

**4f. Trainiertes Lo-Fi-LoRA (das eigentliche Ergebnis, nicht nur die Pipeline):** Die Schritte
1–4e bauen nur die **Pipeline** nach — ein frisch heruntergeladenes MusicGen-Basismodell ohne
LoRA generiert noch keine Lo-Fi-Musik im trainierten Stil. Dieses Repository enthält aus
Größengründen **keinen** trainierten Adapter im Git-Verlauf, nur den Code zum Trainieren
(Schritt 8). Der aktuell freigegebene, echt genutzte Adapter (Step 625, 109 MB, trainiert auf
5000 genrebalancierten Clips — genau der Stand, mit dem die 5 Kontroll-Audios auf der Website
entstanden sind) liegt als GitHub-Release-Anhang bereit (öffentlich, kein Login nötig):

[github.com/AdilBerke/Bachelor-AI-Pipeline/releases/tag/lora-adapter-v1](https://github.com/AdilBerke/Bachelor-AI-Pipeline/releases/tag/lora-adapter-v1)

> **Live geprüft, Stand 2026-08-23:** Download funktioniert öffentlich (HTTP 200), kein
> Hugging-Face-Konto nötig — der frühere Hugging-Face-Verweis war fehlerhaft dokumentiert
> (das Konto existierte nie).

```bash
mkdir -p training/musicgen/lora_training/checkpoints/step_000625/
curl -L -o training/musicgen/lora_training/checkpoints/step_000625/lora_adapter.pt \
  https://github.com/AdilBerke/Bachelor-AI-Pipeline/releases/download/lora-adapter-v1/lora_adapter.pt

.venv/bin/python code/start.py --lora-freigeben \
  --checkpoint training/musicgen/lora_training/checkpoints/step_000625/lora_adapter.pt
```

Der zweite Befehl setzt den Symlink `training/musicgen/lora_training/adapter.pt`, den die
Generierung tatsächlich verwendet. Mit `.venv/bin/python code/start.py --status` prüfen — sollte
"LoRA Adapter: bereit (Step 625, lora)" zeigen.

**Alternative ohne Download:** Die Datei
`training/musicgen/lora_training/checkpoints/step_000625/lora_adapter.pt` (109 MB) einfach
direkt kopieren (USB-Stick, Cloud-Speicher o.ä.) und an derselben Stelle ablegen, dann nur den
zweiten Befehl oben ausführen. Ohne einen der beiden Wege liefert die Pipeline erst nach einem
vollständigen, selbst durchgeführten LoRA-Training (Schritt 8) vergleichbare Ergebnisse — und
selbst dann nicht bit-identisch, da MusicGen-Training/-Generierung stochastisch ist.

## Schritt 5 — Frontend

```bash
export PATH="$HOME/.cache/lo-fi-dreamer-node/node-v22.22.3-linux-x64/bin:$PATH"   # falls Node nicht separat installiert ist
cd frontend
npm install
npm run dev -- --port 8080
```

`package.json`/`package-lock.json` unter `frontend/` übernehmen für Node bereits die Rolle
von `requirements.txt` — kein separater Schritt nötig.

## Schritt 6 — Backend starten

```bash
cd Bachelor-AI-Pipeline
.venv/bin/python audio-pipeline/code/src/Pipeline/web_api.py
```

Reiner `http.server` ohne Framework, Port 8000. Website danach unter `http://localhost:8080`
erreichbar (das Frontend spricht die feste Backend-Adresse aus
`frontend/src/lib/settings.ts` an).

## Schritt 7 — Eigene Datengrundlage statt der Original-MP3s

Die ursprünglichen Trainingsquellen sind absichtlich nicht Teil des Repos. Um den Datensatz
nachzubauen:

```bash
.venv/bin/python code/start.py --top10          # Top10-Quellensuche pro Genre
# danach in der Website unter "Quellen" die gefundenen Videos importieren,
# oder gesammelt per Backend-Aktion "import_pending"
```

Das baut eine eigene, gleichwertige Quellenbasis über `audio-pipeline/code/src/Crawler/quellen_suche.py`
auf (YouTube-Suche + Lizenzfilter „no copyright"), ohne dass die ursprünglichen Dateien
benötigt werden. Aus den daraus erzeugten Clips lässt sich der Datensatz anschließend über
die Steuerung-Seite (Dataset bauen → LoRA-Training) genauso aufbauen wie im dokumentierten
Verlauf in `MUSIKMODELL_VERSUCHSDOKUMENTATION.md`.

## Schritt 8 — Funktionsprüfung

Kurzer Smoke-Test ohne vollständigen Trainingslauf:

```bash
.venv/bin/python audio-pipeline/code/src/Training/lora.py --nur-pruefen
```

Prüft Datensatz, Modellpfade und schreibt den geplanten Trainingsbefehl, ohne tatsächlich
zu trainieren — guter erster Nachweis, dass Python-Umgebung, Modelle und Datensatz-Pfade
korrekt zusammenspielen.

## Bekannte Einschränkung: xFormers-Warnung

Beim Trainingsstart erscheint durchgehend eine `xFormers`-Warnung (inkompatibler Build
gegen die installierte PyTorch/CUDA-Version). Bekannt und harmlos — der einzig kompatible
`xformers`-Build für die von `audiocraft` vorgeschriebene Versionsobergrenze (`<0.0.23`) ist
bereits installiert; ein Upgrade ist nicht möglich, ohne `audiocraft` selbst zu brechen.
Details siehe
[`MUSIKMODELL_VERSUCHSDOKUMENTATION.md`](MUSIKMODELL_VERSUCHSDOKUMENTATION.md),
Abschnitt 10.4.
