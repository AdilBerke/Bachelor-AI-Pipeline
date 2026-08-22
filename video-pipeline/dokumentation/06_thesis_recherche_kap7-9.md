# Recherche-Ergebnis: Videogenerierung-Kapitel (Kap. 7–9)

Antwort auf das Recherche-Briefing. Stand der Untersuchung: 07.08.2026.

**Methodik-Hinweis (gilt für das gesamte Dokument):** Die Git-Historie deckt diesen
Projektteil **nicht** ab — der letzte Commit, der `Bachelorarbeit/` verändert hat,
stammt vom 08.06.2026 (`675e97a`); alle hier dokumentierten Trainingsrunden (ab
13.06.2026) liegen unversioniert nur lokal vor. `git log` kann daher **nicht** als
Quelle für Config-Diffs über die Zeit dienen. Die Chronologie wurde stattdessen aus
`notes.json`/`feedback.json`-Zeitstempeln (`start_time`/`end_time`) und
Dateisystem-mtimes rekonstruiert. Das ist selbst ein dokumentationswürdiger Befund
für die Arbeit (siehe Kap. 8/10: fehlende Versionskontrolle als methodisches Risiko).

Zwei Recherche-Durchgänge wurden mit Datei:Zeile-Belegpflicht durchgeführt; Lücken
sind durchgehend als "nicht dokumentiert" markiert statt geschätzt.

---

## Abschnitt 0: Chronologische Iterationshistorie

### Phase 1 — Vor-Szenario-System "ltx_lora" (13.06.–03.07.2026)

| Start | Runde | Basis-Checkpoint | Steps | Änderung | Ergebnis (Selbsteinschätzung 1–10) | Entscheidung | Quelle |
|---|---|---|---|---|---|---|---|
| 21.06. 17:42 | manual (R1) | Kaltstart | 50 | Erster 13B-LoRA-Lauf | 2 — "stark photorealistisch, kein Anime-Stil" | Fortsetzen | `training_runs/training_01/feedback.json:9-10` |
| 21.06. 23:05 | R2 | R1/s50 | 50 | Prompt erweitert (Cel-Shading, Bücherregal) | 3 | Fortsetzen | `overview.csv:3` |
| 21.06. 23:31 | R3 | R2/s50 | 50 | keine Config-Änderung | 3 — "Cel-Shading-Tendenzen erkennbar" | Fortsetzen | `overview.csv:4` |
| 22.06. 03:55 | R4 | R3/s50 | 50 | Validation-Inference-Steps 30→50 | 5 — "großer Qualitätssprung" | Fortsetzen | `overview.csv:5`; `ltx_lora_round4/training_config.yaml:53` |
| 22.06. 04:36 | R5 | R4/s50 | 50 | Prompt verfeinert | **7 — "beste Komposition, Referenz-Checkpoint"** | Als Referenz sichern | `training_runs/training_05/notes.json:20` |
| 22.06. 05:13 | R6 | R5/s50 | 50 | weiteres Training auf gesättigtem Checkpoint | 4 — **Szenendrift/Überanpassung** | "Nicht R6 verwenden! Zurück zu R5" | `training_runs/training_06/feedback.json:13` |
| 22.06. 05:51 | R7 | R5/s50 (Rollback) | 100 geplant | Image-Conditioning mit Referenzframe getestet | **Fehler:** Letterboxing durch Image-Conditioning | Image-Conditioning dauerhaft deaktiviert (`images: null`) | `training_runs/training_07/notes.json:20` |
| 01.07. 07:15 | R8 | R7/s50 | 100 | `images:null`-Fix, neues 15-Clip-Datenset | 7 — "flüssiger, kein Letterboxing" | Fortsetzen | `training_runs/training_08/notes.json:20` |
| 03.07. 02:18 | R9 | R8/s100 | 100 | Prompt auf "fluid motion" fokussiert | 7 — **2× CUDA-OOM-Crash** während Validation | Validation dauerhaft deaktiviert (`interval: 9999`) | `training_runs/training_09/feedback.json:11-12` |
| 03.07. ~04:16 | R10 | R9/s100 | 100 | — | keine Bewertung dokumentiert | — | `ltx_lora_round10/training_config.yaml` mtime |
| 03.07. ~04:44 | R11 | R10/s100 | 100 | — | keine Bewertung dokumentiert | Checkpoint als "best" verankert | `ltx_lora_round11/training_config.yaml` mtime |

**Wendepunkt 1 (03.07., ca. 05:29 Uhr):** `round11/lora_weights_step_00100.safetensors` wird zu `base_checkpoints/lofi_lora_best.safetensors` — Ende der linearen Round-Struktur, Beginn des Szenario-Systems.

> **Ergänzung zur vorherigen Analyse in dieser Konversation:** Der Wechsel vom kleineren LTX-Video-2B- zum 13B-Modell fand **vor** Phase 1 statt (nicht in den Runden-Daten sichtbar, da bereits alle hier erfassten Runden auf 13B laufen). Beleg dafür: `~/.cache/huggingface/hub/models--Lightricks--LTX-Video/` enthält sowohl `ltxv-2b-0.9.6-dev-04-25.safetensors` (Blob-Datum **02.05.2026**) als auch `ltxv-13b-0.9.7-dev.safetensors` (Blob-Datum **21.06.2026, 07:07 Uhr** — exakt zehn Stunden vor dem ersten dokumentierten 13B-Trainingslauf um 17:42 Uhr desselben Tages). Das 13B-Modell wurde also am Morgen des 21.06. heruntergeladen und noch am selben Abend erstmals trainiert.

### Phase 2 — Szenario "rainy_window" (03.07.–21.07.2026, 14 Runden)

| Start | Runde | Basis | Änderung | Feedback | Quelle |
|---|---|---|---|---|---|
| 03.07. 06:36 | R1 | `lofi_lora_best` | Erststart Szenario | "good"; Fenster/Regen zu wenig detailliert | `round_01/feedback.json:3,13-21` |
| 03.07. 07:17 | R2 | R1/s100 | Prompt: Regenfluss, schärfere Gebäude; inference_steps 30→40 | "good"; Regen noch nicht smooth | `round_02/feedback.json:3,13-19` |
| 03.07. 08:12 | R3 | R2/s100 | Schreibszene ergänzt; inference_steps 40→50; MP4 statt GIF primär | "good"; Auflösung fehlt noch | `round_03/feedback.json:3,20-21` |
| 10.07. 02:49 | R4 | R3/s100 | Kompositionsanker im Prompt; guidance_scale 4,5→5,0 | "good progress **but composition drifting**" | `round_04/feedback.json:3,21,33` |
| 10.07. ~03:47 | R5 | R4/s100 | Bugfix (guidance_scale kam vorher fälschlich hart-codiert statt aus Config); `enhance_video.py` neu erstellt | **"good — composition stable, clear improvement"** — letztes dokumentiertes Feedback | `round_05/feedback.json:3,29-31` |
| 10.07. ~04:58 | R6 | R5/s075 | — | Abbruch nach Step 25 (nur 1 statt 4 Checkpoints) | kein `notes.json`; nur `step_00025`-Checkpoint |
| 10.07. 06:46 | R7 | R6/s025 (Notlösung) | — | keine Bewertung dokumentiert | `round_07/notes.json:4,8` |
| **17.07. 16:36** (6 Tage Lücke) | R8 | R7/s100 | VRAM 37→39,4 GB, Speed 0,14→0,08 steps/s | keine Bewertung dokumentiert | `round_08/notes.json:15-16` |
| **21.07. 15:54** (4 Tage Lücke) | R9 | R8/s100 | `first_frame_conditioning_p` 0,0→0,1; erstmals negative_prompt gesetzt | keine Bewertung dokumentiert | `round_09/config.yaml:10,50` |
| 21.07. 17:27 | R10 | R9/s100 | — | keine Bewertung | `round_10/notes.json:4` |
| 21.07. 18:23 | R11 | R10/s100 | peak_gpu 45,1 GB (Höchstwert) | keine Bewertung | `round_11/notes.json:15` |
| 21.07. 20:07 | R12 | R11/s100 | — | keine Bewertung | `round_12/notes.json:4` |
| 21.07. 21:04 | R13 | R12/s100 | — | keine Bewertung | `round_13/notes.json:4` |
| 21.07. 21:56 | R14 | **R11/s075** (Rollback über R12/R13 hinweg!) | — | keine Bewertung; nachträglich (10.08.) `overall_score 0,7925` | `round_14/samples/*.metrics.json` |

**Wichtigste Einschränkung für Kapitel 9:** Für 9 von 14 Runden (R6–R14) existiert **kein** `feedback.json`. Der dokumentierte, textuell begründete Qualitätsverlauf endet bei Runde 5. Der Rollback in R14 auf R11 (statt auf R13 aufzubauen) legt einen unbewerteten Rückschritt in R12/R13 nahe, ist aber nicht schriftlich belegt.

### Phase 3 — Szenario "rabbit_lake" (22.07.–04.08.2026, 8 abgeschlossene + 1 gescheiterter Rundenordner)

| Start | Runde | Basis | Änderung | Ergebnis | Quelle |
|---|---|---|---|---|---|
| 22.07. 06:30 | R1 | Kaltstart (unabhängig von rainy_window) | Neuer Datensatz (bamboo+cabin, 13 Clips) | keine Bewertung (kein `feedback.json` existiert für rabbit_lake **überhaupt**) | `round_01/config.yaml` |
| 22.07. 16:28 | R2 | R1/s100 | — | keine Bewertung | `round_02/notes.json:8` |
| 22.07. 17:43 | R3 | R2/s075 | — | keine Bewertung | `round_03/notes.json:18-19` |
| 22.07. 21:21 | R4 | R3/s025 (Rollback) | Learning Rate 2e-4→1e-4 halbiert | keine Bewertung | Config-Diff R3→R4 |
| **23.07. 03:17** | R5 | Kaltstart (Komplett-Neustart) | `samples_generated: []` (leer) | **→ RESCUE_PLAN.md verfasst 05:21 Uhr: "Modell erzeugt konstant 3 Hasen statt exakt 2"** | `RESCUE_PLAN.md:5` |
| 23.07. 06:11 | R6 | R5/s100 | Steps 100→50; checkpoint-interval 25→10; **Seed 42→777** (Seed-Suche) | 5 Checkpoints, keine Bewertung dokumentiert | `RESCUE_PLAN.md:26`; Config-Diff |
| *(zw. R6/R7, 05:49–07:10)* | Daten-Update | — | 9 neue Clips (`rabbit_cafe_01-04`, `garden_01-05`), Datensatz 13→22 Clips, Latents neu berechnet | Umsetzung "Option B" aus RESCUE_PLAN | `dataset.jsonl` (22 Zeilen) |
| 23.07. 07:30 | R7 | R6/s050 | Neuer 22-Clip-Datensatz aktiv | keine Bewertung dokumentiert | `round_07/config.yaml:53` |
| **28.07. 16:16** (chronologisch **vor** R8) | "R9"-Versuch | R7/s100 | — | **CUDA-OOM-Absturz vor Step 1** — 0 Steps, 0 Checkpoints | `round_09/log.txt:149` |
| **04.08. 05:46** (12 Tage nach R7) | R8 | R7/s100 | Steps 100→**150**; VRAM fällt auf 37,3 GB, Speed verdoppelt (0,14 statt 0,07) | keine Bewertung; **ab 06./10.08. automatische Metriken** | `round_08/notes.json:26-27` |

**Anomalie:** Der Ordner `round_09` datiert chronologisch **vor** `round_08` (28.07. vs. 04.08.) — die Nummerierung folgt nicht der zeitlichen Reihenfolge.

**Rabbit-Lake-Kernproblem "3 statt 2 Figuren" — Lösungsversuche in Reihenfolge:**

| # | Ansatz | Umsetzung | Ergebnis |
|---|---|---|---|
| 1 | Seed-Suche | Seeds 1337/99/**777**/2024/512 (R6) | Nicht bestätigt (keine Bewertung) |
| 2 | Datensatz-Erweiterung ("Option B") | +9 Clips mit Hasen-Content (R7) | **Kritisch:** die neuen `rabbit_cafe_*`-Captions zeigen laut Datensatz jeweils **einen** Hasen, nicht zwei wie im Plan gefordert |
| 3 | Höhere Guidance Scale ("Option A") | `guidance_scale: 9,0` statt geplanter 7,0 (spätere manuelle Läufe, Dateiname `r8_s80_g9_seed777.mp4`) | Nicht bestätigt |
| 4 | Kompletter Neustart mit Metrik-Zielwerten | `rabbit_lake_v2` (06.08., s. Phase 4) | **Nie ausgeführt** |

**Bis heute nie schriftlich bestätigt**, ob das 2-statt-3-Problem gelöst wurde — die neuen automatischen Metriken (SSIM, Schärfe, Motion) messen die Bildqualität, aber **nicht** die Figurenanzahl.

### Phase 4 — Zwei parallele, unabhängige Neustart-Versuche (06.–07.08.2026)

Wichtige Klarstellung: Es gibt **zwei unterschiedliche** "v2"-Ordner, die **nicht** dieselbe Fortsetzung sind:

| Ordner | Erstellt | Charakter | Prompt-Inhalt | precomputed_dir | Ausgeführt? |
|---|---|---|---|---|---|
| `rabbit_lake_v2` | 06.08. 07:19 | Sorgfältig weitergeführt: identischer, ausführlicher Prompt wie rabbit_lake (inkl. "exactly two rabbits"-Absicherung im Negativ-Prompt), `metric_targets` (overall 0,82/motion 0,50/sharpness 0,22/flicker 0,05), `max_rounds: 8` | "zwei Hasen … Weiterentwicklung von rabbit_lake R8" | rabbit_lake-Daten wiederverwendet | **Nein** — kein `rounds/`-Ordner |
| `rabbit_lake_v2_0807_0356` | 07.08. 03:56 | **Andere Herkunft:** `_meta`-Block (`color_mood`, `scene_mood`, `visual_style` etc.) deutet auf automatisch/über ein Formular generierten Scenario-Eintrag hin (passt zum "Neues Szenario erstellen"-Formular der Website, `studio.video.tsx`). Kurzer, einfacher Prompt mit **Tippfehler** ("Ein Hase vor **ienem** See") und nur **einem** Hasen statt zwei; `trigger_token: lofi_scene` statt `lofi_girl` | "Ein Hase vor ienem See zur Nacht" | **leer** (`precomputed_dir: ''`) | **Nein** — kein `rounds/`-Ordner |

Für Kapitel 9 wichtig: `rabbit_lake_v2_0807_0356` sollte **nicht** als ernsthafte Fortsetzung der Problemlösung dargestellt werden — es wirkt wie ein Testartefakt einer (vermutlich UI-gestützten) Szenario-Erstellungsfunktion, nicht wie eine durchdachte Iteration.

### Phase 5 — Paralleles Experiment "realistic_rabbit" (06.–07.08.2026)

Fotorealistischer statt Lo-Fi-Anime-Ansatz für dasselbe Motiv. Testmatrix A–G (siehe Abschnitt B). Nur Test A tatsächlich ausgeführt (Wiederverwendung des bestehenden rabbit_lake-R8-Samples als Baseline), 7 von 8 Tests weiterhin `"pending"`.

### Sackgassen — Zusammenfassung

- **ltx_lora R6/R7 (22.06.):** Weitertraining auf gesättigtem Checkpoint → Szenendrift, komplett verworfen. Image-Conditioning-Versuch scheiterte an Letterboxing, dauerhaft deaktiviert.
- **rainy_window R6 (10.07.):** Trainingslauf bricht nach Step 25 von 100 ab, keine Fehleranalyse dokumentiert.
- **rainy_window R12/R13 (21.07.):** Unbewertet verworfen — R14 baut auf R11 statt auf R13 auf.
- **rabbit_lake "R9"-Versuch (28.07.):** Sofortiger CUDA-OOM-Absturz vor dem ersten Trainingsschritt.
- **rabbit_lake, 18 ungenutzte Clips:** `assets/clips/` enthält 40 Dateien (u. a. `rabbit_moon_01-04.mp4`), `dataset.jsonl` nutzt nur 22 davon — die übrigen 18 wurden stillschweigend nicht verwendet.
- **rabbit_lake_v2 (beide Varianten):** Konfiguriert, nie in einen tatsächlichen Trainingslauf überführt.
- **realistic_rabbit:** Testmatrix aufgesetzt, nach Test A nicht fortgeführt.
- **Kein bestätigter A/B-Vergleich Basismodell-ohne-LoRA vs. mit LoRA:** Die dafür vorgesehenen Tests B/B2 in `realistic_rabbit` sind weiterhin `"pending"`.

### Wendepunkte

1. **02.05.–21.06.2026:** Modellwechsel LTX-Video 2B (0.9.6) → 13B (0.9.7-dev) — vor der hier erfassten Rundenhistorie, belegt über Hugging-Face-Cache-Zeitstempel.
2. **03.07.2026, ~05:29 Uhr:** Übergang vom linearen Round-System zum Szenario-System (`lofi_pipeline/scenarios/`).
3. **23.07.2026, ~05:21–07:10 Uhr:** Nach RESCUE_PLAN-Diagnose erste inhaltliche Datensatz-Korrektur (13→22 Clips) bei rabbit_lake.
4. **06.08.2026:** Einführung automatischer Qualitätsmetriken (`evaluate_video.py`) — bislang nur retroaktiv auf 3 einzelne Samples angewendet, keine systematische Rundenbewertung.
5. **Durchgehendes Muster:** Nach jeder mehrtägigen Pause (rainy_window: R7→R8, 6 Tage; rabbit_lake: R7→R8, 12 Tage) ändern sich VRAM-Verbrauch und Trainingsgeschwindigkeit spürbar — Ursache nicht dokumentiert (vermutlich Systemzustand/andere GPU-Last, nicht verifiziert).

---

## Abschnitt A: Trainingsvorgehen

### LoRA-Konfiguration (Standard, `base_lora.yaml`)

| Parameter | Wert | Konstanz |
|---|---|---|
| Rang | 16 | Konstant über alle tatsächlich durchgeführten Runden |
| Alpha | 16 | Konstant |
| Dropout | 0,0 | Konstant |
| Ziel-Module | `to_k, to_q, to_v, to_out.0` | Konstant |
| Optimierer | adamw8bit | Konstant |
| Scheduler | cosine | Konstant |
| Precision | bf16 | Konstant |
| Basismodell | LTX-Video 13B 0.9.7-dev (`LTXV_13B_097_DEV`) | Konstant seit 21.06. |

**`base_lora_v2.yaml` (nur für `rabbit_lake_v2` vorgesehen, nie angewendet):**

| Parameter | v1 | v2 |
|---|---|---|
| Rang | 16 | **32** |
| Alpha | 16 | **32** |
| Ziel-Module | 4 Module | 4 Module + `ff.net.0.proj`, `ff.net.2` |
| Lernrate | 1e-4 | **5e-5** |
| Steps | 100 | **200** |
| Checkpoint-Intervall | 10 | 5 |

*(Hinweis: `train_lora.py` unterstützt `--config base_lora_v2.yaml` als Flag; ob damit je trainiert wurde, ist nicht belegt — kein Rundenordner unter `rabbit_lake_v2` existiert.)*

### Trainingsdaten je Szenario

| Szenario | Clips genutzt/verfügbar | Auflösung | Länge | Quelle |
|---|---|---|---|---|
| ltx_lora (Vorphase) | 15/15 | 832×480, 97 Frames | ~13s | 6 YouTube-Quellvideos |
| rainy_window | 9/9 | 832×480, 97 Frames | 13–30s | 3 Quellvideos × 3 Zeitstempel |
| rabbit_lake | 22/40 | 832×480, 97 Frames | ~13s | bamboo+cabin (13) + rabbit_cafe/garden (9) |
| rabbit_lake_v2 | 22 (wiederverwendet) | 832×480, 97 Frames | wie rabbit_lake | nie trainiert |

### GPU/VRAM/Dauer pro Runde

**rainy_window** (Minuten / Peak-VRAM GB / Steps pro Sekunde):
R1: 11,9/36,9/0,14 · R2: 11,7/36,9/0,14 · R3: 11,6/36,9/0,15 · R4: 11,8/37,0/0,14 · R5–R6: nicht dokumentiert · R7: 12,0/37,2/0,14 · R8: 21,2/39,4/0,08 · R9: 22,4/42,0/0,08 · R10: 22,2/42,0/0,08 · R11: 25,0/**45,1**/0,07 · R12: 22,3/42,0/0,08 · R13: 25,2/44,1/0,07 · R14: 22,2/42,1/0,08
Summe dokumentiert: **219,5 Minuten** über 12 von 14 Runden.

**rabbit_lake:** R1: 22,4/42,2/0,08 · R2: 22,3/42,8/0,08 · R3: 22,5/44,0/0,08 · R4: 22,5/42,7/0,08 · R5: 22,7/42,3/0,08 · R6: 12,0/42,5/0,07 · R7: 22,8/42,5/0,08 · "R9"-Versuch: Crash vor Step 1 · R8: 18,0/**37,3**/0,14
Summe dokumentiert: **165,2 Minuten** über 8 von 9 Rundenordnern.

**ltx_lora-Vorphase** (nur Dauer dokumentiert, kein VRAM-Feld in diesem frühen Format): 257+26+38+41+37+38+15+55+86 = **593 Minuten** über 9 Runden (`training_runs/overview.csv`).

Hardware konstant über das gesamte Projekt: 1× NVIDIA RTX A6000, 49 GB VRAM (siehe Hardware-Übersicht, bereits dokumentiert).

### Seeds

| Szenario | Seed | Änderung |
|---|---|---|
| ltx_lora, rainy_window | 42 | konstant |
| rabbit_lake R1–R5 | 42 | — |
| rabbit_lake R6–R8 | **777** | Wechsel im Rahmen der Seed-Suche (RESCUE_PLAN) |

### Trainingsrunden je Szenario — Übersicht

| Szenario | Rundenordner | Davon mit `feedback.json` |
|---|---|---|
| ltx_lora (Vorphase) | 11 | 9 (R10/R11 fehlen) |
| rainy_window | 14 | 5 (R1–R5) |
| rabbit_lake | 9 (inkl. 1 gescheiterter) | **0** |
| rabbit_lake_v2 (beide) | 0 (nur Configs) | — |
| **Gesamt tatsächlich trainiert** | **34** | **14** |

---

## Abschnitt B: Vergleich der Modellvarianten

**Basismodell ohne LoRA vs. LoRA — Ergebnis: Kein abgeschlossener Vergleich vorhanden.**
`realistic_rabbit/results.json` sieht genau dafür Tests B/B2 (Text-zu-Video **ohne** LoRA) vor, beide `"status": "pending"`, alle Scores `null`. Nur Test A (**mit** LoRA, Wiederverwendung von rabbit_lake R8/Step150) hat Status `"done"` mit automatischen Metriken (`auto_overall: 0,784`). Ein direkter A/B-Vergleich ist damit für die Arbeit **nicht belegbar**, nur einseitig (nur die LoRA-Seite) vermessen.

**Mehrere LoRA-Konfigurationen getestet?** Nur konzeptionell (siehe `base_lora_v2.yaml` in Abschnitt A) — ein tatsächlicher Trainingslauf mit Rang 32 ist in keiner der geprüften Dateien belegt.

**Review-Tool-Schema (tatsächlich erfasste Felder):**

| Werkzeug | Felder |
|---|---|
| `feedback.py` (Terminal) | `rating, note, corrections[], ts` |
| `feedback_ui.py` (Web) | `rating, note, ts` (kein `corrections`-Feld) |
| `feedback.json` (älteres, reichhaltigeres Schema, Runden 1–5 rainy_window) | `overall_direction, positives[], improvements_needed[], preserve[], recommendation, prompt_changes, gif_strategy` |
| `*.metrics.json` (neu, ab 06.08.) | `temporal_ssim{mean,min,std}, sharpness{normalized,raw_laplacian,std}, flicker{mean,max,std}, motion{mean,max,smoothness,cv}, color_consistency{...}, brightness_consistency{...}, overall_score` — Gewichtung: SSIM 35%, Schärfe 20%, Flicker 20%, Motion 10%, Farbe 10%, Helligkeit 5% |

---

## Abschnitt C: Nachbearbeitungspipeline

| Komponente | Angabe | Beleg |
|---|---|---|
| RIFE (Standard-Pipeline) | Modellordner `rife-v4` (unterstützt `-n` für feste Ziel-Framezahl) | `enhance_video.py:25` |
| RIFE (Rabbit-Lake-Tests) | **Abweichend:** `rife-v4.6`, mit `-n 4` (fester Multiplikator) | `test_config.yaml:11`, `run_tests.sh:127-131` |
| RIFE-Aufruf | `rife-ncnn-vulkan -i <frames_in> -o <frames_out> -m <modell> -n <ziel_n> -f %08d.png` | `enhance_video.py:84-91` |
| Ziel-FPS Standard | 24 fps (Default), Quelle nativ 8 fps | `enhance_video.py:169-170` |
| Real-ESRGAN | `RealESRGAN_x4plus_anime_6B.pth`, 17.938.799 Bytes, Basis-Scale 4, angewendet mit `outscale=2` (Default) | `enhance_video.py:26,105-115` |
| ffmpeg GIF-Export (tatsächlicher Pfad) | Zweipass: `palettegen=max_colors=256:stats_mode=diff` → `paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle`, `-loop 0` | `make_gif.py:134-147`, Default `fps=15, width=960` |
| ffmpeg-Fallback | `minterpolate=mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:me=umh:mb_size=8:search_param=64`, danach `scale=...:flags=lanczos+accurate_rnd`, Export `libx264 -crf 16 -pix_fmt yuv420p` | `enhance_video.py:33-42` |
| Pipeline-Reihenfolge (Standard) | MP4 (8fps) → `minterpolate` → Real-ESRGAN 4×(→2×) → GIF-Palette → finales MP4 mit Farbraum-Fix | `make_gif.py:4-8,198-218` |
| Pipeline-Reihenfolge (Rabbit-Lake-Test G) | Image-to-Video (960×544) → Real-ESRGAN (→1920×1088) → RIFE (→30fps) | `RABBIT_LAKE_PLAN.md:138-144` |
| Laufzeiten | Nur Schätzwerte aus der Testmatrix (A 15min, B 7min, B2 22min, C/D 11min, E 10min, F 5min, G ~30min gesamt) — **reale gemessene Laufzeiten `render_time_s` sind in `results.json` durchgehend `null`, also nicht dokumentiert** | `RABBIT_LAKE_PLAN.md:95-104`; `results.json` |

---

## Abschnitt D: Implementierung

### Aufrufkette (Standard-Trainingslauf)

```
train_lora.py
 └─ subprocess → tools/LTX-Video-Trainer/scripts/train.py <config.yaml>
      └─ ltxv_trainer.trainer.LtxvTrainer
 └─ (wenn nicht --no-generate) subprocess → generate_samples.py
      └─ subprocess → pipeline/v003_manual/generate.py (LTXConditionPipeline)
      └─ subprocess → scripts/make_gif.py
           └─ interpolate() → upscale_video() → make_gif() → finalize_mp4()
```

**Paralleler, unabhängiger Metrik-Prozess (neu):**
```
metric_watcher.py (manuell in 2. Terminal gestartet)
 └─ Polling-Loop (Default 20s) über Checkpoint-Ordner
 └─ generate_eval_video() → generate.py
 └─ evaluate_video() → pipeline/realistic_rabbit/evaluate_video.py
 └─ bei Zielwert erreicht → schreibt STOP_TRAINING-Datei
      → train_lora.py liest diese Datei in einem Watch-Thread und beendet den Trainingsprozess
```
Das ist die **einzige** echte Trainings-Automatisierung mit Rückkopplung im gesamten Video-Teil — direkt relevant für die vorher besprochene Automatisierungsfrage.

`enhance_video.py` liegt **außerhalb** dieser Kette — kein Skript ruft es automatisch auf, nur manueller CLI-Aufruf dokumentiert.

### Bibliotheken (aus `tools/LTX-Video-Trainer/pyproject.toml`)

```
accelerate>=1.2.1, av>=14.2.1, bitsandbytes>=0.45.2, decord>=0.6.0,
diffusers>=0.32.1, gradio==5.33.0, imageio>=2.37.0, imageio-ffmpeg>=0.6.0,
opencv-python>=4.11.0.86, optimum-quanto>=0.2.6, pandas>=2.2.3, peft>=0.14.0,
pillow-heif>=0.21.0, protobuf>=5.29.3, pydantic>=2.10.4, rich>=13.9.4,
safetensors>=0.5.0, scenedetect>=0.6.5.2, sentencepiece>=0.2.0,
setuptools>=75.6.0, torch>=2.6.0, torchvision>=0.21.0, typer>=0.15.1, wandb>=0.19.11
```
Tatsächlich installiert im `.venv` (exakte Versionen): `torch 2.6.0+cu124, diffusers 0.33.1, peft 0.14.0, transformers 4.52.2, accelerate 1.13.0, safetensors 0.7.0`.
`realesrgan 0.3.0` ist installiert, aber **nicht** in `pyproject.toml` gelistet (separat nachinstalliert).

### Automatisierungsgrad

| Läuft automatisch | Erfordert manuellen Start |
|---|---|
| `metric_watcher.py`: generiert+bewertet+stoppt Training selbstständig | `metric_watcher.py` selbst muss manuell in 2. Terminal gestartet werden |
| `train_lora.py` → ruft automatisch `generate_samples.py` auf | `train_lora.py` selbst: kein Scheduler/Cron im Repo |
| `generate_samples.py` → ruft automatisch `make_gif.py` auf | `enhance_video.py`: nirgends automatisch aufgerufen |
| `run_tests.sh` → ruft nach jedem Test automatisch `evaluate_video.py --update-results` auf | `run_tests.sh` selbst: manueller Aufruf mit Testbuchstabe |
| — | `feedback.py`/`feedback_ui.py`: Bewertung ist inhärent manuelle Nutzereingabe |

---

## Abschnitt E: Ergebnisse

### Finaler Status je Szenario

| Szenario | Status | Begründung |
|---|---|---|
| ltx_lora (Vorphase) | Abgeschlossen (abgelöst) | Checkpoint lebt als Basis aller Folgeszenarien weiter |
| rainy_window | Blockiert/pausiert | Letzte Aktivität 21.07.; 9 von 14 Runden unbewertet; kein "fertig"-Signal |
| rabbit_lake | Blockiert | Kernproblem nie schriftlich als gelöst bestätigt; kein `feedback.json` existiert überhaupt |
| rabbit_lake_v2 (beide) | Nicht begonnen | Nur Configs, kein `rounds/`-Ordner |
| realistic_rabbit | Nicht begonnen/abgebrochen | 7 von 8 Tests weiterhin "pending" |

### Qualitätsverlauf rainy_window (3 Belegzeitpunkte)

- **Früh (R1):** "good" — Stil getroffen, aber Fenster/Regen zu schwach (`round_01/feedback.json:13,16`)
- **Mittel (R4):** "good progress **but composition drifting**" — Komposition wandert zwischen Runden (`round_04/feedback.json:21`)
- **Spät, letztes dokumentiertes Feedback (R5):** "good — composition stable, **clear improvement**" (`round_05/feedback.json:5,10`)
- Danach: **9 Runden ohne jede dokumentierte Bewertung.**

### Automatische Metrikwerte (neu, zum Vergleich)

| Sample | temporal_ssim | sharpness | motion | overall_score |
|---|---|---|---|---|
| rabbit_lake R8 (`r8_s80_g9_seed777`) | 0,9938 | 0,1116 | 0,164 | 0,784 |
| rabbit_lake R8 (`step_00150_0`) | 0,9406 | 0,1321 | 0,533 | 0,7553 |
| rainy_window R14 (`step_00100_0`) | 0,9988 | 0,1201 | 0,017 | 0,7925 |

Nach den im Projekt selbst definierten Schwellwerten (`motion` 0,5–2,0 = "dezent"): rainy_window R14 mit 0,017 wirkt praktisch eingefroren; rabbit_lake liegt mit 0,164–0,533 näher am Zielbereich.

### Zeitaufwand pro Runde
Siehe Abschnitt A — Zusammenfassung: rainy_window ⌀ 18,3 Min/Runde, rabbit_lake ⌀ 20,7 Min/Runde, ltx_lora-Vorphase ⌀ 65,9 Min/Runde (durch den Kaltstart R1 mit 257 Minuten verzerrt; ohne R1: ⌀ 42,0 Min).

---

## Zusatzabschnitt F: Installierte Modelle/Pakete — welche, wann, vermutlich warum

Chronologie aus Installations-Zeitstempeln (`.venv`-`dist-info`-mtimes) und Hugging-Face-Cache-Blob-Daten. Kein Installations-Log im Projekt vorhanden — die Reihenfolge ist aus Dateisystem-Metadaten abgeleitet, die Kausalität ("warum") ist plausible Zuordnung zum jeweils zeitlich nächstliegenden Entwicklungsschritt, **nicht** direkt belegt.

| Datum | Paket/Modell | Typ | Vermuteter Zweck (zeitliche Nähe zu Entwicklungsschritt) |
|---|---|---|---|
| 20.04.2026 | Großteil der Python-Umgebung in einem Bulk-Install: `audiocraft 1.3.0`, `librosa 0.11.0`, `google-api-python-client`, `fastapi`, `scikit-learn` u. v. a. (>150 Pakete, alle identischer Zeitstempel) | Pakete | Initiales Projekt-Setup — Audio-Pipeline (MusicGen/AudioCraft) von Anfang an mitgeplant |
| 02.05.2026 | `LTX-Video 2B (0.9.6-dev)` | Modellgewicht (6,3 GB) | Erste Erkundung von LTX-Video, vor dem Bau der ersten Video-Pipeline |
| 06.05.2026 | `accelerate 1.13.0` | Paket | Fällt exakt auf den Tag des allerersten `video_pipeline`-Zip-Snapshots (Downloads-Ordner) |
| 09.–10.05.2026 | `diffusers 0.33.1`, `peft 0.14.0`, `transformers 4.52.2`, `torch 2.6.0+cu124` | Pakete | LoRA-/Diffusion-Fähigkeit — fällt auf denselben Zeitraum wie der dritte, größte `video_pipeline`-Zip-Snapshot (10.05.) |
| 21.05.2026 | `LTX-Video 0.9.5` | Modellgewicht (2,5 GB) | Weitere Modellversion evaluiert, zwischen 2B (Mai) und 13B (Juni) |
| 21.06.2026, 07:07 | `LTX-Video 13B (0.9.7-dev)` | Modellgewicht (28,6 GB) | Direkt vor dem ersten dokumentierten 13B-Trainingslauf (17:42 Uhr desselben Tages) |
| 08.08.2026 | `yt-dlp 2026.7.4` | Paket | Neuester Stand — vermutlich automatisches Self-Update des Tools, nicht zwingend projektbezogen |

**Weitere lokal vorgehaltene Modelle (Download-Datum nicht immer eindeutig ermittelbar):** `t5-base` (Text-Encoder für LTX-Video), `openai/clip-vit-base-patch32` (CLIP, für Bewertungs-/Analyse-Zwecke), `laion/clap-htsat-unfused` (Audio-Analyse), `google/gemma-3-12b-it-qat-q4_0-unquantized` (Zweck im Rahmen des Video-Teils nicht dokumentiert), `facebook/encodec_32khz` (MusicGen-Audio-Codec), `RealESRGAN_x4plus_anime_6B.pth` (Datei-Datum 08.12.2021 — vorgefertigtes Drittanbieter-Modell, kein Eigenversuch), RIFE-Binaries (Datum 29.10.2022 — ebenfalls vorgefertigt).

**Gesamtbild:** Die Installationsreihenfolge stützt die bereits dokumentierte Chronologie (Audio zuerst, dann schrittweise Aufbau der Video-Fähigkeiten im Mai, Umstieg auf das große 13B-Modell erst Ende Juni) und liefert dafür einen von den Projektdateien unabhängigen zweiten Beleg.

---

## Offene Punkte für Kap. 10 (Machbarkeitsdiskussion)

- Fehlende Versionskontrolle für den gesamten Video-Teil ist selbst ein Befund (Reproduzierbarkeit).
- 20 von 34 tatsächlich durchgeführten Trainingsrunden (59 %) haben keine dokumentierte Bewertung — das schränkt ein, wie viel über den Qualitätsverlauf wissenschaftlich belastbar ausgesagt werden kann.
- Die neuen automatischen Metriken existieren erst seit dem 06.08.2026 und wurden bislang nur auf 3 Einzel-Samples angewendet, nicht rückwirkend auf den gesamten Rundenverlauf.
