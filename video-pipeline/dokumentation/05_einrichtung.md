# Lokale Einrichtung: Video-/GIF-Pipeline

Basiert auf `lofi_pipeline/configs/model_paths.yaml` und `base_lora.yaml`
(Stand 04.08.2026). Kein separates `requirements.txt` für diesen Projektteil
gefunden — Abhängigkeiten laufen über das projektweite `.venv/`.

## Modell und Umgebung

- **Modell:** `LTXV_13B_097_DEV` (LTX-Video 13B, Version 0.9.7-dev)
- **Python:** projektweites `.venv/bin/python`
- **Trainings-/Preprocessing-Skripte:** aus `tools/LTX-Video-Trainer/scripts/`
  (`train.py`, `preprocess_dataset.py`) — externes Framework, nicht Teil des
  eigenen Codes
- **Umgebungsvariablen:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `LTXV_VRAM_LIMIT_FRACTION=0.80` (angesichts wiederholter OOM-Vorfälle wichtig,
  siehe `02_verlauf.md`)
- **GPU:** 1× NVIDIA RTX A6000, 49 GB VRAM (lokal, kein Cluster)

## Umgebung aktivieren

```bash
source .venv/bin/activate
```

## Ein neues Szenario anlegen

1. Szenario-Ordner unter `Bachelorarbeit/lofi_pipeline/scenarios/<name>/` mit
   `scenario.yaml` anlegen (Prompt, Negativ-Prompt, Auflösung, Qualitätskriterien —
   siehe vorhandene Szenarien als Vorlage, z. B. `rainy_window/scenario.yaml`).
2. Referenzclips in `assets/` ablegen.
3. Preprocessing ausführen:
   ```bash
   .venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/preprocess_scenario.py --scenario <name>
   ```

## Trainingsrunde starten

Von Grund auf (nutzt `global_base_checkpoint`, aktuell
`base_checkpoints/lofi_lora_best.safetensors`):
```bash
.venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/train_lora.py --scenario <name> --round 1
```

Fortsetzend vom letzten Checkpoint desselben Szenarios:
```bash
.venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/train_lora.py --scenario <name> --round N --resume
```

Von einem beliebigen Checkpoint:
```bash
.venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/train_lora.py --scenario <name> --round N --from-checkpoint /pfad/zu/checkpoint.safetensors
```

Trainingsparameter (LoRA-Rang 16, Lernrate 1e-4, Cosine-Scheduler, `adamw8bit`,
Checkpoint alle 25 Steps) liegen in `base_lora.yaml` bzw. werden je Runde in
`rounds/round_NN/config.yaml` festgehalten.

## Samples erzeugen

```bash
# nur MP4 (Standard — schnelleres Sichten während des Trainings)
.venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/generate_samples.py --scenario <name> --round N

# zusätzlich GIF
.venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/generate_samples.py --scenario <name> --round N --gif

# nur vorhandene MP4s zu GIF konvertieren
.venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/generate_samples.py --scenario <name> --round N --gif-only
```

## Sample bewerten

```bash
# Terminal
.venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/feedback.py --latest

# Browser-Oberfläche
.venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/feedback_ui.py
```

## Nachbearbeitung (optional)

```bash
.venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/enhance_video.py --scenario <name> --round N --checkpoint 100 --method rife --target-fps 60
```
Fällt automatisch auf `ffmpeg minterpolate` zurück, falls das RIFE-Binary unter
`tools/rife-ncnn-vulkan/` nicht gefunden wird.

## Runden vergleichen / Bericht bauen

```bash
.venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/compare_rounds.py --scenario <name> --rounds 4,9,14
.venv/bin/python Bachelorarbeit/lofi_pipeline/scripts/build_report.py --scenario <name>
```
Hinweis: Diese beiden Skripte wurden bislang nicht produktiv ausgeführt (siehe
`03_code_erklaerung.md`) — vor Nutzung im Rahmen der Arbeit einmal testweise
laufen lassen und Ausgabe prüfen.

## Wichtige Regeln (aus dem bisherigen Verlauf abgeleitet)

- Nicht blind bis Step 100 trainieren, wenn ein früherer Checkpoint (z. B. Step 25/50)
  in der Sichtprüfung bereits besser war — mehrfach beobachtetes Overfitting-/
  Drift-Muster (siehe `02_verlauf.md`, Round 5→6).
- Nach jeder Runde Checkpoints visuell vergleichen, bevor die nächste Runde darauf
  aufbaut.
- Bei wiederkehrenden inhaltlichen Fehlern (falsche Figurenzahl, falscher Stil) zuerst
  die Trainingsclips selbst per Frame-Extraktion prüfen, nicht nur den Prompt anpassen —
  im `rabbit_lake`-Fall lag die Ursache in fehlerhaften Trainingsclips, nicht im Prompt.
- GPU-Speicher im Blick behalten (`peak_gpu_memory_gb` in `notes.json`); bei Werten nahe
  45–47 GB ist ein OOM-Absturz wahrscheinlich.
