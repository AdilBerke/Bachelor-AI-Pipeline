# Gesamt-Projektstruktur: Analyse jedes Ordners (Stand 07.08.2026)

Vollständige Bestandsaufnahme aller Ordner im Projekt (Projektwurzel + externe Orte).
Für Detailanalysen von Audio- bzw. Video-Code siehe `dokumentation/aktuell/` und
`dokumentation/video_gif/`. Dieses Dokument ist die Landkarte darüber.

**Repetitive Ordner** (viele nahezu identische Unterordner, z. B. `durchgang_042`,
`pipeline_20260629_154826`) werden hier als **Muster einmal beschrieben**, nicht
einzeln — das Muster gilt für alle Instanzen.

---

## 1. `Bachelorarbeit/` (16 GB) — Video-/GIF-Pipeline

| Ordner | Zweck | Status |
|---|---|---|
| `daten/processed/ltx_lora_manual/` | Trainingsdaten (VAE-Latents, T5-Embeddings, Vorschau-GIFs) der frühen "lofi_girl"-Phase (13.06.) | Historisch, weiter genutzt als Datenbasis für `lofi_girl_desk` |
| `logs/` | Rohe Trainings-/Preprocessing-Logs, 13.–21.06.2026 | Historisch |
| `pipeline/v003_manual/` | `generate.py` — Kern-Generierungsskript, ruft `LTXConditionPipeline` direkt auf | Aktiv, wird von mehreren neueren Skripten wiederverwendet (u. a. `metric_watcher.py`) |
| `pipeline/realistic_rabbit/` | **Neu (06.08.2026).** Eigenständiges Experiment: fotorealistisches Hase-am-See-Video statt Lo-Fi-Anime-Stil. Testmatrix A–G (Text2Video, Image2Video, Real-ESRGAN-Upscale, RIFE-Interpolation, Kombination). `results.json` mit automatischen Metrik-Feldern. | **Frühes Stadium** — nur Test A gelaufen (Wiederverwendung des Lo-Fi-R8-Checkpoints als Baseline-Check), Tests B–G noch `"pending"`. Paralleles Forschungsgleis zu `rabbit_lake`, nicht Teil des bisherigen Szenario-Systems |
| `training/video/ltx_lora_manual`, `ltx_lora_round2`–`round11` | Frühe iterative "lofi_girl"-Trainingsphase (21.06.–03.07.) | Historisch, siehe `dokumentation/video_gif/02_verlauf.md` Phase 2–3 |
| `training_runs/training_01`–`09` | `feedback.json`+`notes.json` pro früher Trainingsrunde, plus `overview.csv` | Historisch, dokumentiert Phase 2–3 |
| `lofi_pipeline/base_checkpoints/` | Geteilte Start-Checkpoints (`lofi_lora_r11_s100.safetensors`, `lofi_lora_best.safetensors`) | Aktiv genutzt als Trainingsbasis neuer Szenarien |
| `lofi_pipeline/configs/` | `model_paths.yaml`, `base_lora.yaml`, **neu:** `base_lora_v2.yaml` (06.08.) | Aktiv |
| `lofi_pipeline/models/` | Redundante Kopie von `RealESRGAN_x4plus_anime_6B.pth` (liegt auch in `tools/realesrgan-models/`) | Aktiv, aber doppelt gespeichert |
| `lofi_pipeline/references/` | Referenzvideos, Motion-Profile, Rechte-Checkliste (`legal/`), Analyse-Notizen | Referenzmaterial, teils genutzt |
| `lofi_pipeline/reports/` | Vorgesehen für `build_report.py`/`compare_rounds.py`-Ausgaben | **Leer** — nie produktiv ausgeführt |
| `lofi_pipeline/scenarios/cafe_scene`, `library_study`, `train_window` | Nur `scenario.yaml` + leere Unterordner | **0 Trainingsrunden**, nie begonnen |
| `lofi_pipeline/scenarios/lofi_girl_desk` | Formale Fortführung der frühen Phase 2/3 | **0 Runden** unter dem neuen System |
| `lofi_pipeline/scenarios/rainy_window` | 14 Trainingsrunden (03.–21.07.) | Am weitesten fortgeschritten, letzte Runde unbewertet |
| `lofi_pipeline/scenarios/rabbit_lake` | 8 Runden (Stand jetzt: Round 8 lief bis 06.08. weiter, inkl. neuem `.metrics.json` pro Sample) | Weiterhin ungelöstes Kernproblem (Figurenzahl), aber **jetzt mit automatischer Metrik pro Sample** |
| `lofi_pipeline/scenarios/rabbit_lake_v2` | **Neu (06.08.2026).** Frischer Neustart des rabbit_lake-Motivs, nutzt `rabbit_lake`-Precomputed-Daten weiter, aber mit `metric_targets` (overall 0.82, motion 0.50, sharpness 0.22, flicker 0.05) direkt in `scenario.yaml` verankert und `max_rounds: 8` / `max_total_steps: 800` als hartes Limit | **Konfiguriert, Trainingsstart nicht bestätigt** |
| `lofi_pipeline/scripts/` | Pipeline-Werkzeuge, siehe `dokumentation/video_gif/03_code_erklaerung.md`. **Neu:** `metric_watcher.py` (06.08.) — automatische Qualitätsmessung parallel zum Training, mit Auto-Stop-Funktion (`STOP_TRAINING`-Datei) bei Zielwert-Erreichung | Aktiv, **schließt einen zentralen Kritikpunkt aus der Machbarkeitsstudie (fehlende quantitative Metriken)** teilweise |

**Wichtigster Fund in diesem Ordner:** Zwischen 04.08. und 06.08.2026 wurde ein automatisches Qualitätsmetrik-System eingeführt (`evaluate_video.py`, `metric_watcher.py`): `temporal_ssim`, `sharpness` (Laplacian), `flicker`, `motion` (optischer Fluss), `color_consistency`, `brightness_consistency`, zusammengefasst zu `overall_score`. Erster Messwert (Rabbit-Lake Round 8, R8/Step150, Seed 777): `overall_score 0.784`, `motion 0.164` (unter dem für "dezent" definierten Bereich 0.5–2.0 → Video wirkt nach eigener Skala noch zu statisch), `sharpness 0.1116` (knapp über der "unscharf"-Schwelle 0.1). Das ist der erste objektive, nicht-subjektive Datenpunkt im gesamten Video-Projekt.

---

## 2. `bewertungen/` (120 KB) — menschliche Audio-Bewertungen

| Ordner | Zweck |
|---|---|
| `analyse/` | Auswertungsskripte/-ergebnisse über Bewertungsdaten |
| `musicgen/durchgang_002`–`013` | **Muster:** ein Ordner pro Bewertungsdurchgang, je `bewertung.csv` (menschliche Scores). Nur Text/CSV, klein |

---

## 3. `code/` (719 MB) — aktive Audio-Pipeline

| Ordner | Zweck | Status |
|---|---|---|
| `configs/` | `konfiguration.yaml` (Zentralkonfig), `configs/env/.env.local` (nur `ANTHROPIC_API_KEY`/`CLAUDE_MODEL`), `lora_genres.json` | Aktiv |
| `src/Crawler/` | `quellen_suche.py` (monolithischer YouTube-Crawler, 1364 Zeilen), `quellen_finden.py` | Aktiv, funktionsfähig |
| `src/Dataset/` | Datensatz-Erstellung/-Prüfung (`zieldatensatz.py`, `trainingsdaten_pruefen.py` u. a.) | Aktiv |
| `src/Merkmale/` | `audio_merkmale.py` — Feature-Extraktion (RMS, Peak, ZCR; kein librosa hier) | Aktiv |
| `src/Pipeline/` | `projekt_ablauf.py` (Orchestrierung), `clips_vorbereiten.py`, **`web_api.py`** (1539 Zeilen — lokaler HTTP-Backend-Server für die `lo-fi-dreamer`-Website, u. a. `/api/musicgen/...`-Endpunkte) | Aktiv, umfangreich |
| `src/Training/` | 12 Skripte: LoRA-Training, Generierung (`audio_erstellen.py`, 3834 Zeilen), Bewertung, Longform, Nachbearbeitung, Veröffentlichung (nur zu GitHub, nicht YouTube) | Aktiv, Kernstück der Audio-Pipeline |
| `website/lo-fi-dreamer/` | React/Vite-Frontend für die Audio-Pipeline, inkl. `dist/` (Build), `.lovable/` (via Lovable-AI-Website-Builder erstellt) | **Aktiv in Entwicklung** — Dateien bis 04.08. 19:34 Uhr bearbeitet (`library.tsx`, `model.tsx`, `reviews.tsx`, `api.ts`, `jobs.tsx`, `generate.tsx`, `settings.tsx`) |
| `werkzeuge/audiocraft_v1.3.0/`, `werkzeuge/ffmpeg/` | Vendorte externe Tools (AudioCraft-Konfiguration, ffmpeg-Binary) | Infrastruktur |

---

## 4. `daten/` (28 GB) — Audio-Rohdaten & verarbeitete Daten

| Ordner | Zweck |
|---|---|
| `features/audio_merkmale.jsonl` | Eine Datei — Ausgabe der Feature-Extraktion |
| `metadata/downloads/`, `crawler/`, `genrestandards/`, `trainingsdaten_pruefung/`, `cleanup/` | JSON/JSONL-Metadaten zu Crawling, Downloads, Genre-Referenzwerten, Datenprüfung |
| `modelle/musicgen/facebook_musicgen_melody_large/` | Lokales MusicGen-Basismodell (größter Speicherposten) |
| `modelle/audio_analyse/clap_htsat_unfused/`, `demucs/` | Optionale Analyse-/Nachbearbeitungsmodelle |
| `processed/lora_training/{train,valid,test,reports}` | Aufgeteiltes LoRA-Trainingsdatenset |
| `processed/musicgen_youtube_import_30s/` | Importierte 30s-Clips aus YouTube-MP3s |
| `raw/audio/youtube_imports/`, `raw/youtube/{candidates.jsonl,runs/}` | Rohdaten des Crawlers |

---

## 5. `dokumentation/` (80 KB)

| Ordner | Zweck |
|---|---|
| `aktuell/` | 4 Dateien, Audio-Pipeline-Doku (Struktur, Ablauf, Code, Einrichtung) — aktiv gepflegt, behandelt explizit **nur** Audio |
| `video_gif/` | 5 Dateien, in dieser Konversation neu erstellte Video-/GIF-Doku (Struktur, Verlauf, Code, Szenario-Stand, Einrichtung) |

---

## 6. `tools/` (995 MB) — externe Frameworks

| Ordner | Zweck |
|---|---|
| `LTX-Video-Trainer/` | Externes Trainings-Framework (eigenes Git-Repo, `src/ltxv_trainer/`), Grundlage aller LoRA-Trainingsläufe im Video-Teil |
| `realesrgan-models/` | Real-ESRGAN-Gewichte (Anime-Upscaling) |
| `rife-ncnn-vulkan/` | RIFE-Binary (Frameninterpolation), Modelldaten von 2022 (vorinstalliertes Drittanbieter-Tool) |

---

## 7. `training/` (3,9 GB) — Audio-Trainings-/Generierungsausgaben

| Ordner | Zweck |
|---|---|
| `ausgaben/musicgen_generiert/<lauf>` | **Muster:** ein Ordner pro fertig generierter Longform-Audio (WAV/MP3 + Score-Dateien). ~20 Instanzen |
| `ausgaben/musicgen_loops/<lauf>` | **Muster:** Loop-basierte Longform-Audios aus vorhandenen Clips. ~11 Instanzen |
| `bewertungen/musicgen/durchgang_002`–`017`, `longform_001`–`021` | **Muster:** ein Ordner pro Bewertungs-/Testaudio-Batch. ~40 Instanzen |
| `bewertungen/status/bewertungen_200/` | Aggregierter Stand über 200 Bewertungen |
| `musicgen/lora_training_archiv_20260806_164728/` | **Neu (06.08.).** Archivierter LoRA-Trainingsstand inkl. Checkpoints |
| `musicgen/merkmale/merkmale_<datum>/` | Feature-Extraktions-Läufe (2 Instanzen, u. a. vom 06.08.) |
| `musicgen/pipeline_runs/<lauf>` | **Muster:** technischer Log/Zustand pro Pipeline-Aufruf. **~130 Instanzen** — durchgehend von 11.06. bis 28.07., zeigt sehr hohe Iterationsfrequenz auf der Audio-Seite |
| `musicgen/referenzvergleich/vergleich_<datum>` | Referenzvergleichs-Ausgaben (2 Instanzen) |
| `musicgen/web_jobs/` | Job-Zustände der Web-API (`web_api.py`) |

---

## 8. `website/lofi-pipeline-ui/` (348 MB) — Video-Website, Variante 1

React/TS-Frontend (shadcn, Bun), Pages: `DataCollectionTab`, `VideoGenerationTab`, `AudioGenerationTab`, `YouTubeUploadTab`. **Alle Dateien mit identischem Zeitstempel (04.08., 18:01 Uhr) → Bulk-Extraktion, keine organische Entwicklung hier.** `YouTubeUploadTab.tsx` enthält nur Mock-Handler (`console.log` + Toast, explizit `(placeholder)` markiert) für Video+Audio-Merge und YouTube-Upload — keine echte Funktion.

---

## 9. Externe Orte (außerhalb der Projektwurzel)

| Ort | Zweck | Status |
|---|---|---|
| `/tmp/lo-fi-harmony-forge-newweb/` | **Video-Website, Variante 2 — die tatsächlich aktiv genutzte.** Läuft aktuell live (Vite Dev-Server, Port 8080) + Python-Backend (Port 8000). Route `studio.video.tsx`: Referenzbild-Upload, 5-Sterne-Bewertung, Problem-Tags, "Verbesserung anfordern" | UX-Entwurf vollständig, **aber im Code selbst als "Design-Schritt" markiert — keine Backend-Anbindung, kein `fetch`-Aufruf im gesamten File** |
| `~/Downloads/video_pipeline/` (+ 3 Zip-Sicherungen) | Allererster Versuch (06.–10.05.2026): vollautomatische YouTube→Clip→GIF-Trainingsdatenpipeline, 101 vordefinierte Kategorien | Code nachweislich funktionsfähig (Test mit synthetischem Clip erfolgreich), aber nie mit echten Daten durchgelaufen; `current_clip_count: 0` bei allen 101 Kategorien |

---

## Zusammenfassung: Was sich seit der letzten Analyse (04.08.) geändert hat

1. **Neues automatisches Metrik-System** (`metric_watcher.py`, `evaluate_video.py`) — adressiert direkt den in der Machbarkeitsstudie kritisierten Punkt "keine quantitativen Metriken, nur Selbsteinschätzung". Erste reale Messwerte liegen vor.
2. **`rabbit_lake_v2`**: Neustart des blockierten Szenarios, jetzt mit hart einprogrammierten Metrik-Zielwerten und Rundenlimit (`max_rounds: 8`).
3. **Neues Paralleluniversum `realistic_rabbit`**: fotorealistischer statt Lo-Fi-Anime-Ansatz für dasselbe Motiv — das ist eine **Scope-Erweiterung**, keine Fokussierung, und steht im Spannungsverhältnis zur Empfehlung aus der Machbarkeitsstudie, sich auf ein Szenario zu konzentrieren.
4. `train_lora.py` wurde am 06.08. verändert (vermutlich für die Metrik-Integration).
5. Audio-Website (`lo-fi-dreamer`) wird weiterhin aktiv entwickelt (bis 04.08. abends).
