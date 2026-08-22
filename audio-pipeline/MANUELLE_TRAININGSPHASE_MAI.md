# Rekonstruktion: Frühe Video-Trainingsphase (Fragestellung: Mai 2026)

Stand: 2026-08-17 — Evidenzbasierte Rekonstruktion, alle Aussagen mit Quellenangabe.
Erstellt aus: Dateizeitstempeln, training_config.yaml, Git-Log, dokumentation/video_gif/02_verlauf.md.

---

## 1. Gefundene Dateien (vollständige Pfadliste)

### 1a. Ordnerstruktur ltx_lora_manual / ltx_lora_round2–11

```
Bachelorarbeit/training/video/ltx_lora_manual/
  training_config.yaml                        (Zeitstempel: 21.06.2026 17:42)
  checkpoints/lora_weights_step_00005.safetensors
  checkpoints/lora_weights_step_00010.safetensors
  checkpoints/lora_weights_step_00015.safetensors
  checkpoints/lora_weights_step_00020.safetensors
  checkpoints/lora_weights_step_00050.safetensors
  checkpoints/comfy_lora_weights_step_00020.safetensors
  checkpoints/comfy_lora_weights_step_00050.safetensors
  samples/lofi_13b.gif
  samples/lofi_13b.mp4
  samples/lofi_13b_preview.gif
  samples/lofi_girl_step20.gif
  samples/lofi_improved.gif
  samples/lofi_improved.mp4
  samples/step_000000_0.mp4 … step_000050_0.mp4

Bachelorarbeit/training/video/ltx_lora_round2/training_config.yaml  (21.06.2026 23:05)
Bachelorarbeit/training/video/ltx_lora_round3/training_config.yaml  (21.06.2026 23:31)
Bachelorarbeit/training/video/ltx_lora_round4/training_config.yaml  (22.06.2026 03:55)
Bachelorarbeit/training/video/ltx_lora_round5/training_config.yaml  (22.06.2026 04:36)
Bachelorarbeit/training/video/ltx_lora_round6/training_config.yaml  (22.06.2026 05:13)
Bachelorarbeit/training/video/ltx_lora_round7/training_config.yaml  (22.06.2026 05:51)
Bachelorarbeit/training/video/ltx_lora_round8/training_config.yaml  (01.07.2026 07:15)
Bachelorarbeit/training/video/ltx_lora_round9/training_config.yaml  (03.07.2026 02:50)
Bachelorarbeit/training/video/ltx_lora_round10/training_config.yaml (03.07.2026 04:16)
Bachelorarbeit/training/video/ltx_lora_round11/training_config.yaml (03.07.2026 04:44)
```

Alle Checkpoints (`.safetensors`) in round8–11 bestätigen die Zeitstempel unabhängig
(round8-Checkpoints: 01.07.2026 07:28/07:36/07:37; round11-Checkpoints: 03.07.2026 04:47–04:55).

### 1b. Trainingsdaten-Quelle

```
Bachelorarbeit/daten/processed/ltx_lora_manual/
  ltx_lora_v003_manual.yaml      (Zeitstempel: 03.07.2026 04:43)
  dataset.jsonl                  (22.06.2026 00:03)
  precomputed/conditions/        (angelegt 13.06.2026 07:07)
  precomputed/latents/
  clips_raw/                     (22.06.2026 00:02)
  reference_frame_r5.jpg         (22.06.2026 05:50)
```

### 1c. Begleitende Dokumentation (enthält eigene Quellenrekonstruktion)

```
dokumentation/video_gif/01_struktur.md   — Strukturübersicht, Stand 04.08.2026
dokumentation/video_gif/02_verlauf.md   — Chronik, Stand 04.08.2026, aus Zeitstempeln rekonstruiert
dokumentation/video_gif/04_stand_szenarien.md
```

---

## 2. Trainingskonfiguration (Fakten, mit Quellenangabe je Parameter)

### Modell

| Parameter | Wert | Quelle |
|---|---|---|
| model_source | `LTXV_13B_097_DEV` | `ltx_lora_manual/training_config.yaml`, Zeile `model_source:` |
| Bestätigung in allen Rounds | `LTXV_13B_097_DEV` in allen 10 training_config.yaml-Dateien | Jede `training_config.yaml` in round2–11 |

**Befund: Das verwendete Modell ist NICHT LTX-Video 2B, sondern LTXV_13B_097_DEV.**
Kein einziges der training_config.yaml-Dateien in dieser Ordnerstruktur referenziert ein 2B-Modell.
Quelle für 2B: es existieren Template-Configs in `tools/LTX-Video-Trainer/configs/ltxv_2b_lora.yaml`
— das ist das external mitgelieferte Framework-Template, kein eigener Trainingslauf.

### LoRA-Konfiguration (konstant über alle Runden)

| Parameter | Wert | Quelle |
|---|---|---|
| rank | 16 | Alle training_config.yaml, Schlüssel `lora.rank` |
| alpha | 16 | Alle training_config.yaml, Schlüssel `lora.alpha` |
| dropout | 0.0 | Alle training_config.yaml |
| target_modules | to_k, to_q, to_v, to_out.0 | Alle training_config.yaml |

### Optimierung (konstant bis Round 7)

| Parameter | Wert | Quelle |
|---|---|---|
| learning_rate | 2e-4 (= 0.0002) | Alle training_config.yaml, Schlüssel `optimization.learning_rate` |
| batch_size | 1 | Alle training_config.yaml |
| optimizer_type | adamw8bit | Alle training_config.yaml |
| scheduler_type | cosine | Alle training_config.yaml |
| gradient_checkpointing | true | Alle training_config.yaml |
| mixed_precision | bf16 | Alle training_config.yaml |

### Steps pro Runde

| Runde | Steps | Quelle |
|---|---|---|
| ltx_lora_manual (=Round 1) | 50 | `ltx_lora_manual/training_config.yaml`, `optimization.steps: 50` |
| round2 | 50 | `ltx_lora_round2/training_config.yaml` |
| round3 | 50 | `ltx_lora_round3/training_config.yaml` |
| round4 | 50 | `ltx_lora_round4/training_config.yaml` |
| round5 | 50 | `ltx_lora_round5/training_config.yaml` |
| round6 | 50 | `ltx_lora_round6/training_config.yaml` |
| round7 | 50 | `ltx_lora_round7/training_config.yaml` |
| round8 | 100 | `ltx_lora_round8/training_config.yaml`, `optimization.steps: 100` |
| round9 | 100 | `ltx_lora_round9/training_config.yaml` |
| round10 | 100 | `ltx_lora_round10/training_config.yaml` |
| round11 | 100 | `ltx_lora_round11/training_config.yaml` |
| **Summe** | **750 Steps** | Rechnerisch aus obiger Tabelle |

### "Step 1000" — Befund

Die Zahl 1000 taucht in keiner der auffindbarena training_config.yaml-Dateien auf.
Quelle für "Step 1000": `dokumentation/video_gif/02_verlauf.md`, Phase 1:
> "v002: LTX-Video-LoRA, 512×288, 3 Sekunden, 29 Clips, 1000 Steps — als 'zu klein
>  für Thesis-Qualität' eingestuft."
Diese v002-Phase ist **im aktuellen Repository nicht auffindbar** (kein Code, kein Output,
kein Commit). Sie liegt entweder auf einem nicht versionierten Pfad oder wurde gelöscht.

### Validierungs-Konfiguration (Änderungen je Runde)

| Runde | guidance_scale | inference_steps | interval | Quelle |
|---|---|---|---|---|
| manual, round2, round3 | 3.5 | 30 | 25 | jeweilige training_config.yaml |
| round4, round5, round6 | 4.5 | 50 | 25 | jeweilige training_config.yaml |
| round7 | 4.5 | 50 | 25 + Image-Conditioning aktiv | `ltx_lora_round7/training_config.yaml`, `validation.images:` mit Pfad zu reference_frame_r5.jpg |
| round8 | 4.5 | 50 | 50 | `ltx_lora_round8/training_config.yaml` |
| round9, round10, round11 | 4.5 | 30 | 9999 (=deaktiviert) | jeweilige training_config.yaml, `validation.interval: 9999` |

### Prompts (chronologisch)

**Round 1 (ltx_lora_manual):**
> "lofi_girl, anime girl studying late at night, soft lamp glow, pencil on paper, rain sounds, peaceful, 16:9"
> Quelle: `ltx_lora_manual/training_config.yaml`, `validation.prompts[0]`

**Rounds 2–3 (erweitert):**
> "lofi_girl, 2D anime illustration, soft cel shading, anime girl sitting at wooden desk studying late at night, warm glowing desk lamp, pencil in hand writing in notebook, detailed bookshelves background, rain on window, cozy warm atmosphere, smooth subtle loop animation, 16:9"
> Quelle: `ltx_lora_round2/training_config.yaml` und `ltx_lora_round3/training_config.yaml`

**Round 4 (erstmals mit Negativprompt):**
Positiv: lofi_girl, 2D anime illustration, clean cel shading, sharp outlines, anime girl sitting at wooden desk writing…
Negativ: worst quality, inconsistent motion, blurry, jittery, distorted, grainy, noisy, low detail, washed out, photorealistic, 3D render, unclear face, missing features
Quelle: `ltx_lora_round4/training_config.yaml`

**Round 7 (Image-Conditioning-Versuch):**
Prompt enthält: "continue from Round 5 scene, same cozy lofi anime study room…"
`validation.images` zeigt auf: `.../daten/processed/ltx_lora_manual/reference_frame_r5.jpg`
Quelle: `ltx_lora_round7/training_config.yaml`

**Rounds 9–11 (Validation deaktiviert):**
`validation.prompts: []` — keine Validation-Generierung mehr.
Quelle: jeweilige training_config.yaml

---

## 3. Datenherkunft der Trainingsvideos

### Verwendeter Datenpfad (aus allen training_config.yaml):
```
data.preprocessed_data_root:
  /home/.../Bachelorarbeit/daten/processed/ltx_lora_manual/precomputed
```
Quelle: jede training_config.yaml, Schlüssel `data.preprocessed_data_root`

### Precomputed-Verzeichnis erstellt: 13.06.2026
Quelle: `ls -la Bachelorarbeit/daten/processed/ltx_lora_manual/precomputed/` → Verzeichnis-Zeitstempel 13.06.2026 07:07

### Dataset-Datei:
`daten/processed/ltx_lora_manual/dataset.jsonl` (22.06.2026 00:03)

### Rohclips:
`daten/processed/ltx_lora_manual/clips_raw/` (22.06.2026 00:02)

### Herkunft der Clips: NICHT REKONSTRUIERBAR

In den auffindbarena Dateien gibt es keinen yt-dlp-Aufruf, kein Download-Skript, keine
URL-Liste, die den Ursprung der in `clips_raw/` abgelegten Clips belegt. Mögliche Quellen
wären: manuell heruntergeladene CC-Lizenz-Videos (analog zur video_pipeline-Logik aus
Phase 0), oder bereits vorhandene lokale Clips. Ein Skript, das die Clips automatisch
beschafft hätte, ist in diesem Ordner und in `lofi_pipeline/` nicht vorhanden.

Für ein späteres Dataset (Round 8, laut `training_runs/overview.csv` "15 neue Clips,
S-4hwfyK-XQ und l98w9OSKVNA") sind die Quell-IDs dokumentiert, aber kein aktiver
Downloader im Code auffindbar.

---

## 4. Rolle der KI-Unterstützung (nur belegte Fakten)

### Kein Claude-Skript auffindbar

Es gibt kein Python-, Shell- oder sonstiges Skript in diesem Repository, das mit
einem Kommentar, Dateinamen oder Commit-Message auf eine Claude-Instanz als Autor
hinweist und zu diesen Ordnern gehört.

### Wer hat die training_config.yaml erzeugt?

Die Configs wurden mit hoher Wahrscheinlichkeit von `tools/LTX-Video-Trainer/` oder
einem manuell geführten Prozess erzeugt. Sie folgen exakt dem YAML-Schema des
LTX-Video-Trainers und enthalten die YAML-Tag-Syntax
`!!python/object/apply:builtins.getattr`, die der Trainer selbst beim Serialisieren erzeugt.
Dies ist ein technisches Indiz für maschinelle Erzeugung durch das Framework, kein
manuell getipptes Dokument. Eine Claude-Autorschaft dieser Dateien ist weder belegbar
noch widerlegbar.

### Was dokumentation/video_gif/02_verlauf.md zur KI-Unterstützung sagt:

Die Dokumentation (Stand 04.08.2026) beschreibt für Phase 0 (video_pipeline) eine
vollautomatische Pipeline mit `python main.py run-all`, enthält aber keinen expliziten
Hinweis auf Claude-Beteiligung beim Schreiben des Codes. Für Phase 1 (v002/v003) und
Phase 2 (v003_manual) gibt es keinen Hinweis auf eine Automatisierung durch Claude.

### Schlussfolgerung: NICHT REKONSTRUIERBAR

Ob und in welchem Umfang eine Claude-Instanz (a) Quellvideos gesucht, (b) Training-Configs
geschrieben, oder (c) den Training-Prozess unbeaufsichtigt bis "Step 1000" gesteuert hat,
lässt sich aus den vorhandenen Dateien nicht belegen. Es gibt keinen Code-Kommentar,
keine Commit-Message und keine README-Notiz, die das bestätigt oder widerlegt.

---

## 5. Zeitliche Einordnung — Chronologie-Bestätigung

### Kernbefund: Die ltx_lora_manual/round2–11-Phase ist NICHT aus Mai 2026

| Behauptung (Ausgangsfrage) | Befund | Nachweis |
|---|---|---|
| Zeitraum: ca. 6.–10. Mai 2026 | **FALSCH** — Zeitraum ist 21.06.–03.07.2026 | `ls -la training/video/ltx_lora_*/training_config.yaml` |
| Modell: LTX-Video 2B | **FALSCH** — Modell ist LTXV_13B_097_DEV | Alle `training_config.yaml`, Schlüssel `model.model_source` |
| GIF `gif_lofi_cozy_cafe_001_ltx.gif` am 10.05.2026 erzeugt | **NICHT AUFFINDBAR** | systemweite `find`-Suche ohne Treffer; auch in `02_verlauf.md` explizit vermerkt: "konkret gesucht und nicht gefunden" |

### Was in Mai 2026 tatsächlich stattfand

Das Projekt `~/Downloads/video_pipeline/` (außerhalb des Repositories) war im Mai 2026 aktiv.
Es handelt sich um eine Trainingsdaten-Beschaffungspipeline (YouTube CC → yt-dlp → ffmpeg-Clips → GIF),
**kein generatives Videomodell**. Die GIF-Erzeugung dort erfolgte via ffmpeg `palettegen/paletteuse`
aus heruntergeladenen Real-Videos, nicht durch LTX-Video-Inferenz.

Chronologie, vollständig belegbar:

| Datum | Ereignis | Nachweis |
|---|---|---|
| 06.05.2026 05:03 | Erste Zip-Sicherung `video_pipeline` (40 KB) | `02_verlauf.md`, Dateisystem-Zeitstempel |
| 06.05.2026 23:35 | Zweite Sicherung (gewachsen) | `02_verlauf.md` |
| 10.05.2026 08:01 | Dritte Sicherung (104 KB) | `02_verlauf.md` |
| 10.05.2026 16:05–19:46 | Git-Commits zu train_032–034 (MusicGen-Audio!) | `git log` |
| 13.06.2026 06:36 | ltxv-trainer 0.1.0 installiert | `.venv/lib/.../ltxv_trainer-0.1.0.dist-info`, mtime |
| 13.06.2026 07:07 | Precomputed-Daten angelegt | `daten/processed/ltx_lora_manual/precomputed/`, mtime |
| **21.06.2026 17:42** | **ltx_lora_manual — erster Trainingslauf** | `ltx_lora_manual/training_config.yaml`, mtime |
| 21.06.–22.06.2026 | round2–7 | training_config.yaml-Zeitstempel |
| 01.07.2026 | round8 | training_config.yaml-Zeitstempel |
| 03.07.2026 | round9–11 | training_config.yaml-Zeitstempel |

### Übergang 2B → 13B

Das 2B-Modell (`LTXV_2B_0.9.6_DEV`) wurde laut `02_verlauf.md` für eine geplante,
aber nie abgeschlossene v003-Overnight-Pipeline vorgesehen. Der erste nachweislich
tatsächlich durchgeführte Trainingslauf (ltx_lora_manual, 21.06.2026) verwendet bereits
das 13B-Modell. Ein Übergangs-Commit oder eine Notiz, die den Wechsel begründet, ist
nicht im Repository vorhanden. Laut `02_verlauf.md` (Phase 1):
> "v002: […] als 'zu klein für Thesis-Qualität' eingestuft."
Das ist die einzige dokumentierte Begründung für das Nicht-Verwenden des kleineren Modells.

---

## 6. Offene, nicht rekonstruierbare Punkte

1. **Woher stammt das GIF `gif_lofi_cozy_cafe_001_ltx.gif` (10.05.2026)?**
   Nicht auffindbar. Wahrscheinlich von `video_pipeline`'s ffmpeg-GIF-Maker aus einem
   heruntergeladenen CC-Video, nicht aus LTX-Video-Inferenz. Datei selbst nicht mehr vorhanden.

2. **Was genau war v002 (512×288, 1000 Steps)?**
   Kein Code, kein Output, kein Commit im Repository. Zeitraum und Infrastruktur unbekannt.
   Lage: vermutlich lokal ohne Versionierung, gelöscht oder überschrieben.

3. **Was war v003 (LTXV_2B_0.9.6_DEV, 2000 Steps, run_overnight.py)?**
   Ebenfalls nicht auffindbar. Ob dieser Lauf je gestartet wurde oder nur geplant war,
   lässt sich nicht belegen.

4. **Woher stammen die Trainingsclips in `clips_raw/`?**
   Kein Download-Skript, keine URL-Liste, kein yt-dlp-Aufruf auffindbar.
   Herkunft NICHT REKONSTRUIERBAR.

5. **Wurde Claude für Videosuche oder Training-Automatisierung eingesetzt?**
   Kein Code-Beleg, kein Commit-Hinweis. NICHT REKONSTRUIERBAR.

6. **Was passierte zwischen 10.05.2026 (letzte video_pipeline-Sicherung) und
   13.06.2026 (ltxv-trainer-Installation)?**
   Dieser Zeitraum von ~5 Wochen hat keine belegte Aktivität im Video-Bereich.
   v002 und v003 müssten in diesem Fenster gelegen haben — NICHT REKONSTRUIERBAR.
