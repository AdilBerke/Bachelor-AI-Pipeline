# MusicGen Melody Large LoRA

Diese Anleitung beschreibt die neue getrennte MusicGen-Audio-LoRA-Strecke.
Die alte Full-Fine-Tuning-Pipeline bleibt unveraendert.

## Dataset

Aktives Dataset:

`daten/processed/musicgen_30s_aus_60s_clips`

Aufteilung:

- Train: 24964 Clips
- Validierung: 1360 Clips
- Test: 1372 Clips
- Dauer: 30 Sekunden
- Format: WAV, mono, 32 kHz

Das Dataset entsteht aus den vorhandenen 60s-Clips, die sauber in 30s-Clips
geteilt wurden.

## Lokales Modell

Basis-Modell:

`facebook/musicgen-melody-large`

Lokaler Zielordner:

`modelle/musicgen/facebook_musicgen_melody_large`

Modell pruefen:

```bash
.venv/bin/python code/src/Training/setup_musicgen_melody_large.py
```

Modell herunterladen:

```bash
.venv/bin/python code/src/Training/setup_musicgen_melody_large.py --download
```

Der Download wird nicht automatisch bei jedem Training wiederholt. Wenn der
lokale Ordner vollstaendig ist, wird die lokale Version verwendet.

## Smoke-Test

Der Smoke-Test prueft nur, ob Laden, LoRA-Injektion, wenige Trainingsschritte,
Checkpoint-Speicherung und Sample-Generierung technisch funktionieren.

```bash
.venv/bin/python code/src/Training/train_musicgen_melody_large_lora.py \
  --mode smoke \
  --run-name smoke_test_001 \
  --auto-vram
```

Ergebnisse:

- LoRA-Checkpoint: `training/musicgen/melody_large_lora/smoke_test_001/latest_lora.pt`
- Metriken: `training/musicgen/melody_large_lora/smoke_test_001/metrics.csv`
- Sample: `training/musicgen/melody_large_lora/smoke_test_001/samples/`

Fuer einen besonders schnellen technischen Test kann die Sample-Generierung
ausgelassen werden:

```bash
.venv/bin/python code/src/Training/train_musicgen_melody_large_lora.py \
  --mode smoke \
  --run-name smoke_test_001 \
  --auto-vram \
  --no-generate-review-sample
```

Standardmaessig trainiert die Pipeline Melody-Large textbasiert mit leerer
`self_wav`-Condition. Das passt zur spaeteren Prompt-Generierung ohne
eingespeiste Referenzmelodie. Wer explizit die Trainingsaudio als
Melody-Condition nutzen will, kann `--melody-conditioning audio` setzen.

## Echtes Training fuer ca. 44 GB VRAM

Empfohlener Start fuer ca. 44 GB VRAM:

```bash
.venv/bin/python code/src/Training/train_musicgen_melody_large_lora.py \
  --mode train \
  --run-name lora_30s_v001 \
  --auto-vram \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100
```

Mit `--auto-vram` erkennt das Skript die GPU und waehlt fuer 40 bis 48 GB VRAM
automatisch dieses Profil:

- `batch-size 1`
- `gradient-accumulation 8`
- `lora-rank 8`
- `lora-alpha 16`
- `lora-dropout 0.05`
- `learning-rate 8e-5`
- `precision bf16`, falls stabil unterstuetzt, sonst `fp16`
- `save-steps 100`
- `eval-steps 100`
- `max-steps 1000`

Training fortsetzen:

```bash
.venv/bin/python code/src/Training/train_musicgen_melody_large_lora.py \
  --mode train \
  --run-name lora_30s_v001_fortsetzung \
  --auto-vram \
  --resume-from training/musicgen/melody_large_lora/lora_30s_v001/latest_lora.pt
```

## Generierung mit LoRA

```bash
.venv/bin/python code/src/Training/generate_musicgen_lora.py \
  --adapter-path training/musicgen/melody_large_lora/lora_30s_v001/latest_lora.pt \
  --prompt "warm lofi hip hop instrumental, soft drums, mellow piano chords, vinyl texture, relaxed mood" \
  --duration-sec 30 \
  --count 4 \
  --mp3
```

Ausgaben:

`ausgaben/musicgen_lora/`

## VRAM-Profil

Hauptprofil, empfohlen fuer ca. 44 GB VRAM:

- `batch-size 1`
- `gradient-accumulation 8`
- `lora-rank 8`
- `lora-alpha 16`
- `learning-rate 8e-5`
- `precision auto`, intern bevorzugt `bf16`, falls die GPU es stabil unterstuetzt

Fallback fuer 24GB bis 39GB VRAM:

- `batch-size 1`
- `gradient-accumulation 16`
- `lora-rank 4`
- `learning-rate 5e-5`
- `precision fp16`

Ein 80GB-Profil wird hier nicht empfohlen, weil die aktuelle Maschine ca. 44 GB VRAM hat.

## CUDA Out Of Memory

Wenn CUDA Out Of Memory auftritt:

1. `batch-size` auf 1 lassen.
2. `gradient-accumulation` erhoehen.
3. `lora-rank` senken, zum Beispiel von 8 auf 4.
4. `--precision fp16` testen.
5. `--no-generate-review-sample` setzen und Samples separat erzeugen.

Optional kann `--auto-reduce-on-oom` gesetzt werden. Dann schreibt das Skript
bei OOM einen konservativeren Neustart-Befehl in den Run-Ordner. Es startet
nicht endlos neu.

## Wichtige Dateien

- Modell-Setup: `code/src/Training/setup_musicgen_melody_large.py`
- LoRA-Training: `code/src/Training/train_musicgen_melody_large_lora.py`
- LoRA-Generierung: `code/src/Training/generate_musicgen_lora.py`
- LoRA-Hilfsfunktionen: `code/src/Training/musicgen_lora_utils.py`
