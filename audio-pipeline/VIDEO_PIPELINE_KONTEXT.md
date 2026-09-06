# Lo-Fi Video Pipeline — Technische Referenz

> Bachelorarbeit · Inan Deniz Arduc · Maschinenbau M.Sc.
> Standalone-Kontextdatei für Claude Code Sessions.

---

## Entwicklungsgeschichte & Entscheidungen

Chronologie aller Modell-, Paket- und Konfigurations-Änderungen mit Begründungen.
Kein Entwicklungsverlauf — nur Was, Wann und Warum.

---

### April–Mai 2026 — Setup & erste Experimente

**Pakete installiert:**
| Datum | Paket | Version | Warum |
|-------|-------|---------|-------|
| ~Apr 20 | torchtext | 0.16.0 | frühe Audio-NLP-Experimente |
| Mai 6 | accelerate | 1.13.0 | Mixed-Precision Training Infrastruktur |
| Mai 7 | opencv-python | 4.13.0.92 | Videoframe-Verarbeitung |
| Mai 7 | gradio | 5.33.0 | erste Web-UI für Tests |
| Mai 8 | PyYAML | 6.0.3 | Config-Dateien lesen/schreiben |
| Mai 9 | diffusers | 0.33.1 | Hugging Face Diffusion Pipeline |
| Mai 10 | bitsandbytes | 0.45.2 | 8-Bit Quantisierung für VRAM-Einsparung |
| Mai 10 | torch | 2.6.0+cu124 | GPU-Kern mit CUDA 12.4 |
| Mai 10 | torchvision | 0.21.0 | Bild-Transformationen |

**Modell-Downloads:**
| Datum | Modell | Warum |
|-------|--------|-------|
| Mai 2 | `ltxv-2b-0.9.6-dev` (2B) | Erste Tests mit kleinerem Modell |
| Mai 21 | `LTX-Video-0.9.5` (diffusers-Format) | Alternative ältere Version testen |

**Status:** Beide Modelle nie für Training genutzt. Ergebnisse offenbar nicht überzeugend — keine Training-Aufzeichnungen aus dieser Phase.

---

### Juni 2026 — Umstieg auf 13B, erstes echtes Training

**13. Juni — LTX-Video-Trainer installiert**
- `tools/LTX-Video-Trainer` geklont und als editable package installiert (`ltxv-trainer 0.1.0`)
- Gleichzeitig: `Pillow 11.3.0` installiert (für Bild-I/O)
- **Warum:** ltxv-trainer liefert `LTXConditionPipeline` und `load_ltxv_components()` — nötig um LTXV LoRA-Training überhaupt zu starten

**21. Juni — Umstieg auf 13B 0.9.7-dev**
- `ltxv-13b-0.9.7-dev.safetensors` heruntergeladen (~59 GB gesamt mit Text-Encoder + VAE + Transformer)
- **Warum der Wechsel von 2B auf 13B:**  Das 2B-Modell zeigte nach Tests keine ausreichende Stilkonsistenz und Detailtiefe für Lo-Fi Anime. 13B hat 6.5× mehr Parameter → bessere Fähigkeit für feine Texturen, Cel-Shading, kohärente Szenen. Gleicher VRAM-Aufwand bei bf16 + 8-Bit-Encoder ist machbar (~28–32 GB).

**Basis-Konfiguration ab training_01 (21. Juni, 17:42):**
```
Modell:     LTXV_13B_097_DEV
Rank:       16, Alpha: 16
LR:         0.0002 (2e-4)
Steps:      50
Batch:      1
Optimizer:  adamw8bit
Scheduler:  cosine
Precision:  bf16
Validation: alle 25 Steps, 30 Inference-Steps
Auflösung:  832×480, 97 Frames
```

**Training-Vorgehen ab hier: manuelle Iteration**
Jede Runde wurde nach Sichtprüfung des generierten Videos bewertet — 4 Dimensionen:
- **Qualität** (Gesamteindruck 1–10)
- **Animation** (Bewegungsfluss 1–10)
- **Detail** (Schärfe, Textur 1–10)
- **Realismus** (Lo-Fi Anime Stil 1–10)

Basis: kein automatisches Metrik-Tool — rein visuell.

**Verlauf der lofi_girl_desk Trainings (alle 21.–22. Juni):**

| Run | Datum | Steps | Inference-Steps | Q | A | Entscheidung |
|-----|-------|-------|----------------|---|---|--------------|
| training_01 | 21.06 17:42 | 50 | 30 | 2 | 2 | Kaltstart, stark photorealistisch, kein Anime-Stil |
| training_02 | 21.06 23:05 | 50 | 30 | 3 | 2 | Anime-Tendenzen leicht erkennbar |
| training_03 | 21.06 23:31 | 50 | 30 | 3 | 3 | Cel-Shading erstmals sichtbar → Inference-Steps erhöhen |
| training_04 | 22.06 03:55 | 50 | **50** | 5 | 4 | **Großer Sprung**: klare Anime-Ästhetik durchgehend. Änderung: Inference 30→50 |
| training_05 | 22.06 04:36 | 50 | 50 | **7** | **6** | **PEAK**: beste Komposition, Stimmung optimal. Dieser Checkpoint = Referenz |
| training_06 | 22.06 05:13 | 50 | 50 | 4 | 4 | Szenendrift, Overfitting. Zu viele Steps nach Peak. → Zurück zu R5-Checkpoint |
| training_07 | 22.06 05:51 | **100** | 50 | 4 | 3 | Steps 50→100, aber Image-Conditioning-Bug: Letterboxing (schwarze Balken) |
| training_08 | 01.07 07:15 | 100 | 50 | 7 | 7 | Fix: `images: null`. 15 neue Clips im Trainingsset. Flüssigere Animation |
| training_09 | 03.07 02:18 | 100 | 30 | 7 | 7 | OOM-Crashes → Validation dauerhaft auf 9999 gestellt. Video separat generiert |

**Wichtige Einzel-Entscheidungen aus dieser Phase:**

- **Inference-Steps 30→50 (training_04):** Bessere Qualitätseinschätzung während Validation + bessere Outputs. Bleibt so.
- **Validation-Interval 25→9999 (training_09):** 2× OOM-Crash durch Inline-Validation beim 13B. Seitdem: Training ohne Validation, danach `generate_samples.py` separat.
- **`images: null` (training_08):** `images: reference_frame` erzeugte einen Zoom/Crop-Effekt (Letterboxing). Dauerhaft deaktiviert.
- **lofi_girl_desk R5-Checkpoint als Basis:** Dieser Checkpoint wurde als `lofi_lora_best.safetensors` gespeichert und ist der globale Startpunkt für alle folgenden Szenarien.

---

### Juli 2026 — lofi_pipeline Struktur, neue Szenarien, Post-Processing

**3. Juli — lofi_pipeline Struktur aufgebaut**
- `compare_rounds.py`, `build_report.py` erstellt
- `rainy_window` als erstes Szenario in der neuen Struktur angelegt (Runde 1: 3. Juli)
- **Warum:** die alten `training_runs/` waren unstrukturiert, kein Szenario-Konzept. lofi_pipeline trennt Szenarios, Runden, Checkpoints sauber.

**20. Juli — Post-Processing Pakete installiert**
| Paket | Version | Warum |
|-------|---------|-------|
| realesrgan | 0.3.0 | Frame-Upscaling für GIF/MP4 Output |
| basicsr | 1.4.2 | RRDBNet-Architektur (Backbone von ESRGAN) |
| scikit-image | 0.26.0 | SSIM-Metrik für automatische Bewertung |
| numpy | 2.3.5 | Array-Operationen (Metrik-Berechnungen) |

**20. Juli — rife-ncnn-vulkan heruntergeladen**
- Vulkan-Binary für GPU-basierte Frame-Interpolation
- **Warum RIFE statt minterpolate:** minterpolate (ffmpeg CPU) erzeugt bei Anime-Stil Ghosting-Artefakte. RIFE nutzt optischen Fluss auf GPU → deutlich flüssigere Übergänge.

**21. Juli — Real-ESRGAN Modell + enhance_video.py**
- `RealESRGAN_x4plus_anime_6B.pth` heruntergeladen
- `enhance_video.py` erstellt: erste RIFE + ESRGAN Pipeline
- **Warum x4plus_anime_6B:** Speziell für Anime-Stil trainiert, 6-Block-Variante (leichter als volle 23-Block-Variante), funktioniert gut auf 8fps Lo-Fi Videos.

**22. Juli — rabbit_lake als Hauptszenario gestartet**
- Runden 1–3: LR noch 0.0002

**rabbit_lake LR-Wechsel (ab Round 4):**
- R1–R3: LR 0.0002 (2e-4) — Standard aus training_runs
- **R4+: LR 0.0001 (1e-4)** — halbiert. Grund: bei R3 erste Anzeichen von Overfitting (Szene instabiler). Niedrigere LR = feinere Anpassung, weniger Drift.
- R6: Steps 100→50 (experimentell, zu kurz)
- R7: Steps wieder 100
- R8: Steps 150 (höchste Qualität, Q:7 A:7 in manueller Bewertung)
- R9: CUDA-OOM-Absturz → Training-Run unterbrochen

**rabbit_lake Checkpoint-Intervall-Änderung:**
- R1–R3: alle 25 Steps (aus training_runs übernommen)
- **R4+: alle 10 Steps** — mehr Granularität, besser für nachträgliche Auswahl des besten Checkpoints

**28. Juli — Feedback-UI (Flask, Port 7860)**
- `feedback_ui.py` + `feedback.py` erstellt
- `flask 3.1.3` installiert
- **Warum:** Schnelleres visuelles Feedback ohne Terminal — MP4s direkt im Browser bewerten, Noten vergeben, nächste Runde antriggern. Gradio-UI war zu unflexibel.

---

### August 2026 — Automatische Metriken, GIF-Pipeline, metric_watcher

**4. August — make_gif.py** (vollständige 3-Schritt-Pipeline)
- Ersetzt den einfachen ffmpeg-GIF-Befehl aus generate_samples.py
- **Warum:** minterpolate allein reicht nicht. Vollständige Kette: minterpolate → ESRGAN 4× → GIF (palettegen/paletteuse) + MP4 (H.264, bt709, CRF 16)

**6. August — evaluate_video.py** (automatische Metriken)
- 6 Metriken komplett lokal (OpenCV, scikit-image, NumPy)
- **Warum:** Manuelle Bewertung (1–10 Scores) ist subjektiv und reproduzierbar. evaluate_video.py liefert objektive, vergleichbare Zahlen pro Checkpoint — SSIM, Sharpness, Motion, Flicker, Color/Brightness Consistency.
- **Vorgehen ab hier:** nach jedem generate_samples-Lauf → evaluate_video.py → Metrik-JSON prüfen → Checkpoint-Auswahl basierend auf overall_score

**6. August — metric_watcher.py**
- Automatischer Filesystem-Watcher parallel zum Training
- **Warum:** Training ist ein Subprocess — kein Python-Level-Callback möglich. metric_watcher.py läuft parallel, triggert bei neuem Checkpoint automatisch generate + evaluate → schreibt metrics_log.json → kann Training via STOP_TRAINING-Datei stoppen.

**7. August — evaluate_video.py erweitert**
- motion_smoothness (Variationskoeffizient) als neue Sub-Metrik ergänzt
- Vergleichsmodus (`--compare`) für mehrere Videos/Ordner

**10. August — train_lora.py, generate_samples.py, preprocess_scenario.py refactored**
- Scripts aus den einzelnen Szenario-Ordnern in `scripts/` zentralisiert
- **Warum:** jedes Szenario hatte eigene Script-Kopien → Wartung aufwändig. Zentrale Scripts lesen scenario.yaml und sind für alle Szenarien wiederverwendbar.

---

### Bewertungs-Vorgehen im Überblick

**Phase 1 (Juni–Juli 2026): Manuell**
- 4 Dimensionen: Qualität, Animation, Detail, Realismus (je 1–10)
- Basis: Sichtprüfung des generierten MP4
- Gespeichert in `training_runs/training_0N/feedback.json` und `notes.json`
- Problem: subjektiv, nicht reproduzierbar, kein Checkpoint-Vergleich

**Phase 2 (ab August 2026): Automatisch + manuell**
- evaluate_video.py: 6 Metriken → overall_score (objektiv, reproduzierbar)
- Manuell: Sichtprüfung auf Stil, Prompt-Treue, Animations-Targets
- metric_watcher.py: automatische Auswertung während Training
- Ziel-Schwellwerte: overall_score > 0.82, motion > 0.50, sharpness > 0.22

**Warum kein automatisches Stop-Kriterium bisher:**
- metric_watcher.py neu (6. August) — noch nicht in vollem Einsatz
- Alle bisherigen Runden: manuelles Training-Ende nach Sichtprüfung

---

### Aktuelle Basis-Konfiguration (Stand August 2026)

```
Modell:          LTXV_13B_097_DEV  (seit 21. Juni, nie gewechselt)
Basis-LoRA:      lofi_lora_best.safetensors  (= lofi_girl_desk R5)
Rank:            16 (base_lora.yaml) / 32 (base_lora_v2.yaml, rabbit_lake_v2)
Alpha:           = Rank
LR:              1e-4 (seit rabbit_lake R4, vorher 2e-4)
Steps/Runde:     100 (Standard), 150 (rabbit_lake R8)
Checkpoint:      alle 10 Steps (vorher 25)
Validation:      deaktiviert (interval: 9999) — seit training_09 wegen OOM
Inference:       separat nach Training via generate_samples.py
Post-Processing: make_gif.py (minterpolate → ESRGAN → GIF + MP4)
Bewertung:       evaluate_video.py (6 Metriken, overall_score gewichtet)
Watcher:         metric_watcher.py (seit 6. August, parallel zum Training)
```

---

## Überblick

LTX-Video 13B + LoRA-Fine-Tuning für Lo-Fi Anime Loops.
- **Basismodell:** `LTXV_13B_097_DEV`
- **Pipeline-Klasse:** `LTXConditionPipeline` (ltxv_trainer)
- **Training:** LoRA via `accelerate` + `ltxv-trainer`
- **Python-Env:** `.venv/bin/python` (Repo-Wurzel, siehe NACHBAUANLEITUNG.md Schritt 3a)
- **Root:** `video-pipeline/lofi_pipeline/`

## Dateistruktur

```
lofi_pipeline/
  scripts/
    train_lora.py         # Training-CLI
    generate_samples.py   # Inference pro Runde
    make_gif.py           # Post-Processing: minterpolate → ESRGAN → GIF + MP4
    enhance_video.py      # RIFE oder minterpolate + ESRGAN (per Szenario/Runde)
    metric_watcher.py     # Checkpoint-Watcher, auto-eval, STOP_TRAINING
    feedback_ui.py        # Flask UI Port 7860
    compare_rounds.py     # Rundenvergleich
    preprocess_scenario.py
  configs/
    base_lora.yaml        # Standard: Rank 16, LR 1e-4
    base_lora_v2.yaml     # Verbessert: Rank 32, LR 5e-5, FF-Module
    model_paths.yaml      # Pfade zu Python, Trainer, Basis-Checkpoint
  scenarios/{id}/
    scenario.yaml         # Prompt, Auflösung, Seed, Steps, Post-Processing
    dataset.jsonl
    precomputed/          # latente Repräsentationen (einmalig, wiederverwendbar)
    rounds/round_NN/
      config.yaml
      training_config.yaml
      checkpoints/        # lora_weights_step_NNNNN.safetensors
      samples/            # step_NNNNN_0.mp4 + _final.gif + _final.mp4
      metrics_log.json    # metric_watcher Ergebnisse
  base_checkpoints/
    lofi_lora_best.safetensors  # globaler Basis-LoRA für alle Szenarien

pipeline/
  v003_manual/generate.py          # direkte Inference via LTXConditionPipeline
  realistic_rabbit/evaluate_video.py  # Metriken
```

## LoRA-Konfigurationen

### base_lora.yaml (Standard) — vollständig
```yaml
lora:
  rank: 16
  alpha: 16
  dropout: 0.0
  target_modules: ["to_k", "to_q", "to_v", "to_out.0"]

optimization:
  learning_rate: 1e-4
  steps: 100
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw8bit"
  scheduler_type: "cosine"
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: null
  load_text_encoder_in_8bit: true
  compile_with_inductor: false

conditioning:
  mode: "none"
  first_frame_conditioning_p: 0.1

validation:
  video_dims: [832, 480, 97]
  seed: 42
  inference_steps: 30
  interval: 9999          # deaktiviert (kein auto-validation)
  guidance_scale: 4.5

checkpoints:
  interval: 10
  keep_last_n: -1         # alle behalten

flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"

hub:
  push_to_hub: false

wandb:
  enabled: false
  project: "ltxv-lofi"
```

### base_lora_v2.yaml (rabbit_lake_v2) — Änderungen
| Parameter | base_lora | base_lora_v2 | Begründung |
|-----------|-----------|--------------|------------|
| rank / alpha | 16 / 16 | **32 / 32** | mehr Kapazität für Details + Bewegung |
| target_modules | to_k, to_q, to_v, to_out.0 | **+ ff.net.0.proj, ff.net.2** | FF-Schichten für Stil + Textur |
| learning_rate | 1e-4 | **5e-5** | feiner, weniger Overfitting bei Rank 32 |
| steps (default) | 100 | **200** | längeres Training |
| checkpoint interval | 10 | **5** | mehr Punkte für metric_watcher |
| wandb.project | ltxv-lofi | **ltxv-lofi-v2** | eigene W&B-Gruppe |

## Szenarien

| ID | Runden | Auflösung | Status |
|----|--------|-----------|--------|
| rainy_window | 14 | — | am weitesten, kein Checkpoint fertig |
| rabbit_lake | 9 | 832×480 | R9 = CUDA-OOM-Absturz |
| rabbit_lake_v2 | 1 | 832×480 | neu, base_lora_v2, mehr Bewegung |
| golden_hour_lake | 6 | — | — |
| lofi_girl_desk | — | 960×544 | ursprüngliches Testmodell |
| rabbit_2, hase_1/2, cafe_scene, train_window, library_study | — | — | frühe Experimente |

### scenario.yaml — vollständige Schlüsselfelder (rabbit_lake)
```yaml
scenario_id: rabbit_lake
description: "Two small rabbits by a dark atmospheric lake at night..."

# Generierung
frames: 97
fps: 8
resolution: [832, 480]
generate_inference_steps: 60
guidance_scale: 9.0
seed: 777
trigger_token: lofi_girl

# Post-Processing
post_processing:
  method: rife+realesrgan          # Primär (Vulkan GPU)
  fallback: minterpolate+lanczos   # CPU-Fallback
  target_fps: 60
  upscale: "Real-ESRGAN x4plus_anime 2x → 1664x960"

# Daten
precomputed_dir: .../scenarios/rabbit_lake/precomputed  # wiederverwendbar, ~1h gespart
dataset_jsonl: .../scenarios/rabbit_lake/dataset.jsonl

# Qualitätskriterien (für metric_watcher)
quality_criteria:
  max_rounds: 6
  max_total_steps: 600
  min_animation_score: 7
  min_quality_score: 7
  requires_lofi_style: true
  requires_stable_composition: true
  requires_visible_motion: true

# Animations-Targets (Teil des Prompt-Designs)
animation_targets:
  - "stars: 5-8 softly twinkling at offset timings, 2-4 sec cycles"
  - "tree leaves: slow sway left-right in gentle breeze, continuous"
  - "water surface: slow horizontal shimmer, moon+star reflections rippling"
  - "two rabbits: very subtle ear twitch every 4-5 seconds"
  - "two rabbits: slow soft blink every 5-6 seconds"
  - "grass foreground: very slight minimal sway"
  - "lantern: very subtle warm flicker"
```

## Generierung

```bash
# Samples für Runde generieren
python scripts/generate_samples.py --scenario rabbit_lake --round 8

# Bestimmter Checkpoint + Seed
python scripts/generate_samples.py --scenario rabbit_lake --round 8 \
  --checkpoint 80 --seed 1337

# Direkt via generate.py (höhere Auflösung möglich)
python pipeline/v003_manual/generate.py \
  --lora scenarios/rabbit_lake/rounds/round_08/checkpoints/lora_weights_step_00080.safetensors \
  --steps 60 --width 960 --height 544 --frames 97 \
  --guidance-scale 9.0 --seed 777 \
  --output ausgaben/rabbit_lake_r8.mp4
```

### generate.py Parameter
| Parameter | Standard | Beschreibung |
|-----------|----------|-------------|
| --lora | None | Pfad zu .safetensors |
| --steps | 50 | Denoising-Steps |
| --width / --height | 960 / 544 | Ausgabe-Auflösung |
| --frames | 97 | ~12s bei 8fps |
| --guidance-scale | 4.5 | Szenarien nutzen 9.0 |
| --seed | 42 | CUDA-Generator-Seed |
| --gpu-fraction | 1.0 | GPU-Throttle (0.8 = 80%) |

### VRAM-Schätzung
| Konfiguration | VRAM |
|--------------|------|
| 960×544×97 Frames | ~28–32 GB |
| 1280×720×97 Frames | ~36 GB (Limit: 39 GB) |
| Training 832×480, steps 100 | ~24–28 GB |

## Post-Processing Pipeline

```
Raw MP4 (8fps, 832–960px)
  → Schritt 1: Frame-Interpolation
               minterpolate (ffmpeg MCI/AOBMC/BIDIR) ODER rife-ncnn-vulkan v4.6
               8fps → 24fps (Ziel konfigurierbar)
  → Schritt 2: Real-ESRGAN x4plus_anime_6B
               tile=512, tile_pad=10, pre_pad=0, half=True (CUDA)
               4× → outscale=2 → 1920×1088, dann bt709 Re-Encode (CRF 16)
  → Schritt 3a: GIF
               palettegen: max_colors=256, stats_mode=diff
               paletteuse: dither=bayer, bayer_scale=5, diff_mode=rectangle
               15fps, 960px Breite, loop=0
  → Schritt 3b: MP4 final
               stream-copy + bt709 Metadaten, CRF 16, yuv420p
```

```bash
# Standard (mit ESRGAN)
python scripts/make_gif.py rounds/round_08/samples/step_00080_0.mp4

# Ohne Upscaling (schneller, schlechtere Qualität)
python scripts/make_gif.py rounds/round_08/samples/step_00080_0.mp4 --no-upscale

# Nur GIF
python scripts/make_gif.py rounds/round_08/samples/step_00080_0.mp4 --gif-only

# Kleineres GIF (< 15 MB Ziel)
python scripts/make_gif.py rounds/round_08/samples/step_00080_0.mp4 --gif-width 640

# Eigene FPS
python scripts/make_gif.py rounds/round_08/samples/step_00080_0.mp4 \
  --gif-fps 12 --interp-fps 30
```

### make_gif.py — alle Flags
| Flag | Standard | Effekt |
|------|----------|--------|
| `--no-upscale` | — | Real-ESRGAN überspringen |
| `--gif-only` | — | Nur GIF, kein finales MP4 |
| `--gif-fps` | 15 | GIF-Framerate |
| `--gif-width` | 960 | GIF-Breite in px (640 = kleiner) |
| `--interp-fps` | 24 | Ziel-FPS nach minterpolate |

### enhance_video.py — Methoden im Detail
| Aspekt | minterpolate | rife (empfohlen) |
|--------|-------------|-----------------|
| Tool | ffmpeg | rife-ncnn-vulkanr (Vulkan GPU) |
| Flags | `mi_mode=mci, mc_mode=aobmc, me_mode=bidir, vsbmc=1, me=umh, mb_size=8, scd=none` | rife-v4 Modell, `-n` für Frame-Anzahl |
| Upscale | lanczos (ffmpeg scale) | Real-ESRGAN x4plus_anime_6B |
| VRAM | CPU (kein GPU) | ~2–4 GB (ESRGAN) |
| Qualität | gut, kann Ghosting haben | viel flüssiger für Anime-Stil |
| Fallback | — | fällt automatisch auf minterpolate zurück wenn Binary fehlt |

## Bewertungsmetriken (evaluate_video.py)

Alle lokal berechnet (OpenCV, scikit-image). Kein externes Modell nötig.

### Gewichteter Gesamt-Score — Formel und Methoden
| Metrik | Gewicht | Berechnung | Gut | Schlecht |
|--------|---------|-----------|-----|---------|
| Temporal SSIM | **35%** | skimage `ssim(gray_a, gray_b, data_range=255)`, Mittelwert über alle Frame-Paare | > 0.85 | < 0.70 |
| Sharpness | **20%** | `cv2.Laplacian(gray, CV_64F).var()` pro Frame, Mittelwert / 3000 (Normierung), min(result, 1.0) | > 0.30 | < 0.10 |
| Flicker (inv.) | **20%** | `mean(|frame_a/255 - frame_b/255|)` aller Frame-Paare; Score = `max(0, 1 - flicker/0.1)` | < 0.02 | > 0.06 |
| Motion | **10%** | Farneback Opt. Fluss (`pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2`); Score = `1 - min(abs(motion - 1.5) / 5.0, 1.0)` (optimal ~1.5 px/Frame) | 0.5–2.0 px/Frame | < 0.1 oder > 5 |
| Color Consistency | **10%** | std der mittleren RGB-Kanalwerte über alle Frames; `max(0, 1 - avg_std/30)` | > 0.80 | < 0.50 |
| Brightness Consistency | **5%** | std der mittleren Grauwerte; `max(0, 1 - std/30)` | > 0.80 | < 0.50 |

### Zusätzliche Metriken (nicht im Gesamt-Score, aber in JSON)
| Metrik | Berechnung | Bedeutung |
|--------|-----------|-----------|
| `motion.smoothness` | `1 - min(CV, 2.0) / 2.0` wobei CV = std(flow_magnitudes) / mean — Variationskoeffizient | Gleichmäßigkeit der Bewegung; niedrig = ruckartig |
| `motion.cv` | Variationskoeffizient des opt. Flusses (raw) | < 0.5 = sehr gleichmäßig |
| `temporal_ssim.min` | Minimum-SSIM über alle Frame-Paare | Einzelne schlechte Übergänge sichtbar |
| `sharpness.raw_laplacian` | Mittelwert Laplacian-Varianz (vor Normierung) | Vergleich über Videos |

### rabbit_lake R8 Baseline (Referenz)
| Metrik | R8 Wert | Problem | Ziel v2 |
|--------|---------|---------|---------|
| Temporal SSIM | 0.9938 | — gut | > 0.85 |
| Sharpness | 0.1116 | **zu weich** | **> 0.22** |
| Flicker | 0.0039 | — gut | < 0.05 |
| Motion px/Frame | 0.164 | **fast eingefroren** | **> 0.50** |
| Gesamt-Score | 0.784 | — | **> 0.82** |

```bash
# Einzelnes Video auswerten
python pipeline/realistic_rabbit/evaluate_video.py path/to/video.mp4

# Mehrere Runden vergleichen
python pipeline/realistic_rabbit/evaluate_video.py --compare \
  rounds/round_06/samples/ rounds/round_07/samples/ rounds/round_08/samples/
```

## metric_watcher.py

Läuft parallel zum Training. Überwacht neue `.safetensors`-Dateien. Bei jedem neuen Checkpoint:
1. Kurzes Test-Video generieren (30 Frames, 30 Steps, ~3 min)
2. Mit evaluate_video.py auswerten
3. In `metrics_log.json` eintragen
4. Wenn Ziel-Score erreicht → `STOP_TRAINING` schreiben → train_lora.py beendet sauber

```bash
python scripts/metric_watcher.py \
  --scenario rabbit_lake_v2 --round 1 \
  --target-overall 0.82 --target-motion 0.50 --target-sharpness 0.22
```

## Training

```bash
# Neue Runde (Runde 1 = globaler Basis-Checkpoint)
python scripts/train_lora.py --scenario rabbit_lake --round 1 --steps 100

# Nächste Runde (auto-detect letzter Checkpoint der Vorrunde)
python scripts/train_lora.py --scenario rabbit_lake --round 2

# Mit Metric-Watcher (zwei Terminals)
python scripts/train_lora.py --scenario rabbit_lake_v2 --round 1 --steps 200
python scripts/metric_watcher.py --scenario rabbit_lake_v2 --round 1 --target-overall 0.82
```

**Checkpoint-Logik:** Runde 1 → `lofi_lora_best.safetensors`. Runde N → letzter Checkpoint von Runde N-1 (automatisch via `find_auto_checkpoint()`).

## Pakete & Versionen

| Paket | Version | Rolle |
|-------|---------|-------|
| torch | 2.6.0+cu124 | GPU-Kern |
| diffusers | 0.33.1 | Scheduler, Pipeline |
| transformers | 4.52.2 | Tokenizer, Text-Encoder |
| accelerate | 1.13.0 | Mixed-Precision Training |
| ltxv-trainer | 0.1.0 | LTXConditionPipeline |
| bitsandbytes | 0.45.2 | 8-Bit-Adam + Quantisierung |
| realesrgan | 0.3.0 | Upscaling |
| basicsr | 1.4.2 | RRDBNet-Architektur |
| opencv-python | 4.13.0.92 | Frameextraktion, Opt. Fluss, Laplacian |
| scikit-image | 0.26.0 | SSIM |
| Pillow | 11.3.0 | Bild-I/O |
| numpy | 2.3.5 | Array-Operationen |
| PyYAML | 6.0.3 | Config-Dateien |
| ffmpeg | 4.4.2 | minterpolate, GIF, Video-I/O |
| rife-ncnn-vulkan | 2022-10-29 | Frame-Interpolation (Vulkan) |
| RealESRGAN_x4plus_anime_6B.pth | — | Anime-Upscaler Gewichte |

## Modell-Konfiguration (generate.py)

```python
# Laden
components = load_ltxv_components(
    model_source=LtxvModelVersion.LTXV_13B_097_DEV,
    load_text_encoder_in_8bit=True,   # VRAM: Text-Encoder 8-Bit
    transformer_dtype=torch.bfloat16,
    vae_dtype=torch.bfloat16,
)
components.transformer.to("cuda")
components.vae.to("cuda")
# text_encoder bleibt wo bitsandbytes es platziert hat (bereits auf CUDA)

pipe = LTXConditionPipeline(
    scheduler=components.scheduler,
    tokenizer=components.tokenizer,
    text_encoder=components.text_encoder,
    vae=components.vae,
    transformer=components.transformer,
)
pipe.enable_vae_tiling()  # Peak-VRAM beim VAE-Decode reduzieren

# LoRA laden (falls vorhanden)
pipe.load_lora_weights("/pfad/zu/lora_weights_step_NNNNN.safetensors")

# Inference
generator = torch.Generator(device="cuda").manual_seed(seed)
torch.cuda.empty_cache()
frames = pipe(
    prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
    num_frames=97, width=960, height=544,
    num_inference_steps=60, guidance_scale=9.0,
    generator=generator,
).frames[0]
export_to_video(frames, "output.mp4", fps=8)
```

### Env-Variablen
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # verhindert CUDA-Fragmentierung
LTXV_VRAM_LIMIT_FRACTION=0.80                     # reserviert 20% VRAM als Puffer
```

### model_paths.yaml — vollständig
```yaml
python: ".venv/bin/python"   # Pfad zur Haupt-Umgebung, relativ zur Repo-Wurzel
trainer_script: ".../tools/LTX-Video-Trainer/scripts/train.py"
preprocess_script: ".../tools/LTX-Video-Trainer/scripts/preprocess_dataset.py"
generate_script: ".../pipeline/v003_manual/generate.py"
pipeline_root: ".../Bachelorarbeit/lofi_pipeline"
global_base_checkpoint: ".../lofi_pipeline/base_checkpoints/lofi_lora_best.safetensors"
model_source: "LTXV_13B_097_DEV"
env:
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
  LTXV_VRAM_LIMIT_FRACTION: "0.80"
```

## Studio-Integration (web_api.py)

Die Video-Pipeline ist über `code/src/Pipeline/web_api.py` (Port 8000) ans Studio angebunden:

| Endpoint | Funktion |
|----------|----------|
| POST /api/video/train | `video_make_train_job()` → startet train_lora.py |
| POST /api/video/generate | `video_make_generate_job()` → startet generate_samples.py |
| POST /api/video/make-gif | `video_make_gif()` → startet make_gif.py |
| POST /api/video/evaluate | `video_evaluate()` → evaluate_video.py, gibt flache Metriken zurück |
| GET /api/video/scenarios | listet alle Szenarien mit Rundeninfo |
| GET /api/video/media/{id}/{rel} | liefert MP4/GIF-Dateien |

Job-Status-Werte: `"running"`, `"completed"`, `"failed"` (NICHT "done"/"error").
Video-Dateipfade: `scenarios/{id}/rounds/round_NN/samples/datei.mp4`
