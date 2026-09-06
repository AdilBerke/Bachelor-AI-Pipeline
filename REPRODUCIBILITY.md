# Reproducibility Guide

Schritt-für-Schritt-Anleitung, um Code, Umgebung, Modelle und Datengrundlage dieses Projekts auf
einer neuen Maschine vollständig nachzubauen: Audio-Pipeline (MusicGen + LoRA), Video-Pipeline
(LTX-Video + LoRA) und die gemeinsame Studio-Website.

Alle Versions- und Hardwareangaben sind aus der tatsächlich genutzten Entwicklungsumgebung
übernommen (Stand 2026-08-23, siehe [`requirements.txt`](requirements.txt)),
keine Schätzungen.

**Aktueller Referenzstand:** Das aktive Audio-Modell ist LoRA-Adapter Step 625 (trainiert auf
5000 genrebalancierten Clips, siehe Schritt 4f). Damit wurden für alle 5 Genres je eine
20-Minuten-Kontrollaudio erzeugt — das sind aktuell auch die einzigen auf der Studio-Website
sichtbaren Audios (`/api/audio` filtert bewusst auf den `Kontrolle_`-Namensprefix, ältere
Test-/Entwicklungsläufe bleiben auf der Platte, sind aber ausgeblendet).

**Bewusst nicht Teil dieser Anleitung:** die konkreten YouTube-MP3/Video-Rohquellen und die daraus
gebauten Trainingsdatensätze. Das sind fremde, urheberrechtlich geschützte Inhalte, die nicht
mitgeliefert werden. Schritt 7 beschreibt, wie man sich eine eigene, gleichwertige Quellenbasis
beschafft, ohne die konkreten Originaldateien zu benötigen.

## Inhalt

1. [Code](#1-code)
2. [System-Voraussetzungen](#2-system-voraussetzungen)
3. [Python-Umgebung (gemeinsam für Audio + Video)](#3-python-umgebung-gemeinsam-für-audio--video)
4. [Basismodelle herunterladen](#4-basismodelle-herunterladen)
5. [Frontend](#5-frontend)
6. [Backend starten](#6-backend-starten)
7. [Eigene Datengrundlage statt der Original-Quellen](#7-eigene-datengrundlage-statt-der-original-quellen)
8. [Audio-Pipeline: Funktionsprüfung und typische Abläufe](#8-audio-pipeline-funktionsprüfung-und-typische-abläufe)
9. [Video-Pipeline: typische Abläufe](#9-video-pipeline-typische-abläufe)
10. [Bekannte Einschränkungen der Reproduzierbarkeit](#10-bekannte-einschränkungen-der-reproduzierbarkeit)

## 1. Code

Ein einzelnes Repository mit drei Teilprojekten:

```text
audio-pipeline/    Python — Datenerfassung, Audioverarbeitung, MusicGen+LoRA, Backend
video-pipeline/    Python — LTX-Video-LoRA-Training, Sample-/GIF-Generierung
frontend/          React/TypeScript — Studio-Weboberfläche
```

```bash
git clone https://github.com/AdilBerke/Bachelor-AI-Pipeline.git
cd Bachelor-AI-Pipeline
```

`audio-pipeline/` und `video-pipeline/` teilen sich **eine** Python-Umgebung (siehe Schritt 3);
`frontend/` hat eine eigene Node-Umgebung.

## 2. System-Voraussetzungen

| Voraussetzung | Getestete Version | Wofür |
|---|---|---|
| Betriebssystem | Ubuntu 22.04.5 LTS, Kernel 6.8 | — |
| NVIDIA-GPU + Treiber | RTX A6000 (48 GB VRAM), Treiber 580.173.02 | MusicGen-/LTX-Video-Training und -Generierung |
| Python | 3.11 (3.11.0rc1) | Backend, beide Pipelines |
| Node.js | 22.x (v22.22.3) | Frontend (Vite/React) |
| ffmpeg | 4.4.2 (`apt install ffmpeg`) | Audio-/Video-Dekodierung, Zusammenschnitt |

Ein separates CUDA-Toolkit ist **nicht** nötig — die CUDA-Runtime kommt über die `nvidia-cu*`-Pakete
in `requirements.txt` mit. Nur der GPU-Treiber muss installiert sein.

Mindestens ~16 GB freier VRAM werden für MusicGen-Melody-Large-LoRA-Training empfohlen (siehe
`vram_profile_for_hardware()` in `audio-pipeline/code/src/Training/musicgen_steuerung.py`: unter
24 GB gilt als Notfallprofil, 24–39 GB als Fallback, ab 40 GB als empfohlenes Profil). Für
LTX-Video-Training wurde durchgehend mit der vollen 48-GB-Karte gearbeitet; wiederholte
OOM-Vorfälle bei knapperem VRAM sind im Projektverlauf dokumentiert (siehe
`video-pipeline/dokumentation/02_verlauf.md`).

## 3. Python-Umgebung (gemeinsam für Audio + Video)

```bash
cd Bachelor-AI-Pipeline
python3.11 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

`requirements.txt` enthält alle Pakete beider Pipelines, exakt gepinnt per `pip freeze` aus der
laufenden Entwicklungsumgebung — u. a. `audiocraft`, `transformers`, `xformers`, `librosa`, `demucs`,
`yt-dlp` (Audio) sowie `diffusers` und `ltxv_trainer` (per Git-Commit gepinnt, aus
`github.com/Lightricks/LTX-Video-Trainer`), `decord`, `opencv-python`, `gradio` (Video). Das
`--extra-index-url` ist nötig, weil `torch`/`torchaudio` als `+cu124`-Build referenziert sind, der nur
im PyTorch-eigenen Wheel-Index liegt, nicht auf PyPI selbst.

Die `.venv` liegt in der Repo-Wurzel — alle Befehle für **beide** Pipelines nutzen diesen einen
Interpreter, sämtliche Beispiele unten sind entsprechend von der Wurzel aus formuliert.

**Zweite Umgebung für die NIQE-Bewertung:**

```bash
.venv/bin/python -m venv --system-site-packages .venv-niqe
.venv-niqe/bin/pip install -r requirements-niqe.txt
```

Diese Pakete dürfen **nicht** in `.venv` installiert werden: `pyiqa` erzwingt `transformers>=5`,
während die Pipeline `transformers==4.52.2` benötigt — andernfalls starten `diffusers` und
`ltxv_trainer` nicht mehr. `evaluate_video.py` ruft die NIQE-Berechnung deshalb über
`video-pipeline/lofi_pipeline/scripts/_niqe_worker.py` als Subprozess im zweiten venv auf.
Ohne diesen Schritt liefert die Videobewertung SSIM, Schärfe, Flackern, Bewegung und Farbe,
aber keinen NIQE-Wert — und damit nicht die in der Arbeit dokumentierten Vergleichswerte.

## 4. Basismodelle herunterladen

Die beiden Basismodelle (MusicGen und LTX-Video) werden beim ersten Gebrauch automatisch von
Hugging Face geladen. Die optionalen Zusatzmodelle müssen explizit bereitgestellt werden.

**4a. MusicGen Melody Large (Pflicht für die Audio-Pipeline, ca. 15 GB):**

Kein gesonderter Setup-Schritt nötig. Die Audiopipeline lädt das Modell beim ersten Lauf selbst
über die Modell-ID `facebook/musicgen-melody-large` (siehe `--model-id` in
`audio-pipeline/code/src/Training/audio_erstellen.py`) und legt es im Hugging-Face-Cache ab.
Der erste Generierungslauf dauert dadurch deutlich länger als die folgenden.

Optional lässt sich der Download vorziehen:

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('facebook/musicgen-melody-large')"
```

**4b. CLAP (semantische Genre-Prüfung, ca. 590 MB, optional):**

```bash
.venv/bin/python audio-pipeline/code/src/Training/audio_modelle_einrichten.py --download clap
```

Lädt `laion/clap-htsat-unfused` nach `audio-pipeline/daten/modelle/audio_analyse/clap_htsat_unfused/`.
Aussagekraft eingeschränkt (nur ~40 % Trefferquote im Test, siehe
`audio-pipeline/MUSIKMODELL_VERSUCHSDOKUMENTATION.md`, Abschnitt 10.1) — deshalb produktiv nicht aktiv.

**4c. Demucs (optional, nur für manuelle Stem-Analyse):**

```bash
.venv/bin/python audio-pipeline/code/src/Training/audio_modelle_einrichten.py --download demucs
```

**4d. LTX-Video 13B (Video-Pipeline):** wird von `ltxv_trainer` beim ersten Aufruf von
`video-pipeline/pipeline/v003_manual/generate.py` automatisch von Hugging Face geladen
(`LtxvModelVersion.LTXV_13B_097_DEV`) und im Standard-HF-Cache abgelegt — kein gesonderter
Setup-Schritt, dafür ein einmaliger, langsamerer erster Lauf.

**4e. RealESRGAN-Gewichte (Video-Nachbearbeitung/Upscaling, optional):** müssen manuell von der
[offiziellen Real-ESRGAN-Release-Seite](https://github.com/xinntao/Real-ESRGAN/releases) geladen und
unter `video-pipeline/lofi_pipeline/models/RealESRGAN_x4plus_anime_6B.pth` abgelegt werden.

**4f. Trainiertes Lo-Fi-LoRA (optional, um ohne eigenes Training sofort generieren zu können):**

Dieses Repository enthält aus Größengründen **keinen** trainierten LoRA-Adapter im Git-Verlauf, nur den
Code zum Trainieren (Schritt 8). Der aktuell freigegebene, menschlich bewertete Adapter (Step 625,
109 MB, trainiert auf 5.000 genrebalancierten Clips, SHA256
`2e7327592d91bc18bb4ebc02e69c7c0b2373f1ba3658d0ac64aea0f54efc26c6`) liegt als GitHub-Release-Anhang
bereit:

[github.com/AdilBerke/Bachelor-AI-Pipeline/releases/tag/lora-adapter-v1](https://github.com/AdilBerke/Bachelor-AI-Pipeline/releases/tag/lora-adapter-v1)

> **Live geprüft, Stand 2026-08-23:** Download funktioniert öffentlich (HTTP 200), kein Login nötig —
> der frühere Hugging-Face-Verweis in dieser Anleitung war fehlerhaft dokumentiert (das Hugging-Face-
> Konto existierte nie, HTTP 401 kam von einem nicht existenten User).

```bash
mkdir -p audio-pipeline/training/musicgen/lora_training/checkpoints/step_000625/
curl -L -o audio-pipeline/training/musicgen/lora_training/checkpoints/step_000625/lora_adapter.pt \
  https://github.com/AdilBerke/Bachelor-AI-Pipeline/releases/download/lora-adapter-v1/lora_adapter.pt

.venv/bin/python audio-pipeline/code/start.py --lora-freigeben \
  --checkpoint training/musicgen/lora_training/checkpoints/step_000625/lora_adapter.pt
```

Danach ist der heruntergeladene Adapter als aktiver, freigegebener Stand registriert und direkt für
Longform-Generierung (Schritt 8) nutzbar — ohne eigenen Trainingslauf. Ohne diesen Schritt liefert die
Pipeline erst nach einem vollständigen, selbst durchgeführten LoRA-Training (Schritt 8) Ergebnisse.
Alternative ohne Download: die 109-MB-Datei manuell kopieren (USB-Stick/Cloud) an denselben Zielpfad,
dann nur den zweiten Befehl oben ausführen.

## 5. Frontend

```bash
cd frontend
npm install
npm run dev -- --port 8080
```

`package.json`/`package-lock.json` übernehmen für Node bereits die Rolle von `requirements.txt` —
kein separater Schritt nötig. Das Frontend benötigt **keine** Konfigurationsdatei und keine
Zugangsdaten: Es spricht ausschließlich das lokale Backend auf Port 8000 an (feste Adresse in
`frontend/src/lib/settings.ts`). Externe Dienste werden nicht verwendet.

## 6. Backend starten

```bash
.venv/bin/python audio-pipeline/code/src/Pipeline/web_api.py
```

Reiner `http.server` ohne Framework, Port 8000. Website danach unter `http://localhost:8080`
erreichbar (Frontend spricht die feste Backend-Adresse aus `frontend/src/lib/settings.ts` an).

Die Video-Pipeline hat eine eigene, unabhängige lokale Oberfläche für Bewertung/Feedback:

```bash
.venv/bin/python video-pipeline/lofi_pipeline/scripts/feedback_ui.py
```

Standardmäßig auf Port 7860.

## 7. Eigene Datengrundlage statt der Original-Quellen

### Audio

```bash
.venv/bin/python audio-pipeline/code/start.py --top10          # Top10-Quellensuche pro Genre
```

Danach in der Website unter "Quellen" die gefundenen Videos importieren, oder gesammelt per
Backend-Aktion `import_pending`. Das baut über `audio-pipeline/code/src/Crawler/quellen_suche.py` (YouTube-Suche +
Lizenzfilter „no copyright") eine eigene, gleichwertige Quellenbasis auf, ohne dass die
Original-Dateien benötigt werden. Aus den daraus erzeugten Clips lässt sich der Datensatz
anschließend genauso aufbauen wie im dokumentierten Verlauf in
`audio-pipeline/MUSIKMODELL_VERSUCHSDOKUMENTATION.md`.

Bekannte Einschränkung: Massenimporte ohne Cookies schlagen häufig mit „Sign in to confirm you're not
a bot" fehl (YouTube-Bot-Sperre). Ein Workaround (`--cookies-from-browser`) ist zum Stand dieses
Dokuments noch nicht produktiv eingebaut.

### Video

Referenzclips für neue Szenarien werden manuell recherchiert und unter
`video-pipeline/lofi_pipeline/references/source_videos/` abgelegt; Nutzungsrechte sind vorab gegen
`video-pipeline/lofi_pipeline/references/legal/source_rights_checklist.md` zu prüfen. Ein neues
Szenario danach wie in Schritt 9 beschrieben anlegen.

## 8. Audio-Pipeline: Funktionsprüfung und typische Abläufe

Kurzer Smoke-Test ohne vollständigen Trainingslauf:

```bash
.venv/bin/python audio-pipeline/code/src/Training/lora.py --nur-pruefen
```

Prüft Datensatz, Modellpfade und schreibt den geplanten Trainingsbefehl, ohne zu trainieren — guter
erster Nachweis, dass Umgebung, Modelle und Datensatz-Pfade korrekt zusammenspielen.

Weitere Befehle (vollständige Liste inkl. Longform-Generierung und Referenzvergleich in
[`audio-pipeline/README.md`](audio-pipeline/README.md#lokale-befehle)):

```bash
.venv/bin/python audio-pipeline/code/start.py --status
.venv/bin/python audio-pipeline/code/start.py --clips-5000
.venv/bin/python audio-pipeline/code/start.py --lora-training
.venv/bin/python audio-pipeline/code/start.py --testaudios
.venv/bin/python audio-pipeline/code/start.py --referenzvergleich
```

## 9. Video-Pipeline: typische Abläufe

Vollständige Anleitung: [`video-pipeline/dokumentation/05_einrichtung.md`](video-pipeline/dokumentation/05_einrichtung.md).
Umgebungsvariablen für stabiles Training (wiederholte OOM-Vorfälle ohne diese Einstellungen):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LTXV_VRAM_LIMIT_FRACTION=0.80
```

```bash
# neues Szenario: erst scenario.yaml + Referenzclips anlegen, dann
.venv/bin/python video-pipeline/lofi_pipeline/scripts/preprocess_scenario.py --scenario <name>
.venv/bin/python video-pipeline/lofi_pipeline/scripts/train_lora.py --scenario <name> --round 1
.venv/bin/python video-pipeline/lofi_pipeline/scripts/generate_samples.py --scenario <name> --round 1 --gif
.venv/bin/python video-pipeline/lofi_pipeline/scripts/feedback.py --latest
```

## 10. Bekannte Einschränkungen der Reproduzierbarkeit

- **xFormers-Warnung:** Beim Trainingsstart erscheint durchgehend eine `xFormers`-Warnung
  (inkompatibler Build gegen die installierte PyTorch/CUDA-Version). Bekannt und harmlos — der
  einzig kompatible `xformers`-Build für die von `audiocraft` vorgeschriebene Versionsobergrenze
  (`<0.0.23`) ist bereits installiert; ein Upgrade ist nicht möglich, ohne `audiocraft` selbst zu
  brechen. Details: `audio-pipeline/MUSIKMODELL_VERSUCHSDOKUMENTATION.md`, Abschnitt 10.4.
- **Keine bit-identischen Ausgaben:** MusicGen- und LTX-Video-Generierung sind stochastisch
  (Sampling); dieselbe Konfiguration erzeugt vergleichbare, aber keine identischen Audios/Videos.
  Reproduzierbar sind Pipeline, Konfiguration und Bewertungsmethodik — nicht einzelne Sample-Bytes.
- **Quellenverfügbarkeit:** Da Trainingsquellen live von YouTube bezogen werden, kann sich die
  konkret verfügbare Quellenbasis zwischen zwei Nachbau-Zeitpunkten unterscheiden (gelöschte Videos,
  neue Top-Treffer).
- **CLAP-Genreprüfung** hat in Tests nur ~40 % Trefferquote erreicht und läuft deshalb nicht als
  produktiver Bewertungsschritt (nur optional verfügbar, siehe Schritt 4b).
- **Kein Szenario der Video-Pipeline gilt als vollständig trainiert**, siehe
  `video-pipeline/dokumentation/04_stand_szenarien.md` für den Stand je Szenario und bekannte
  Bild-/Objektfehler (z. B. `rabbit_lake`).
