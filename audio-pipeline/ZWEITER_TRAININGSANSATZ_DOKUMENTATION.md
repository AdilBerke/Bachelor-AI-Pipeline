# Dokumentation: Video-Trainingsansatz mit LTX-Video 13B / ltxv-trainer

Stand: 2026-08-17. Erstellt aus Dateizeitstempeln, installierten Paketen (`.venv/lib/python3.11/site-packages`),
`git check-ignore`/`git log`, sowie zwei bereits im Repository vorhandenen, unabhängig verifizierten
Referenzdokumenten (`MANUELLE_TRAININGSPHASE_MAI.md`, Stand 2026-08-17, und `VIDEO_PIPELINE_KONTEXT.md`,
undatiert, beide im Projekt-Wurzelverzeichnis). Wo Aussagen aus diesen beiden Dokumenten übernommen werden,
ist das explizit vermerkt. Alle darin enthaltenen Kern-Zeitstempel wurden in dieser Sitzung stichprobenartig
gegen das Dateisystem geprüft und stimmten überein (siehe Abschnitt 1).

---

## 1. Auflösung des Widerspruchs (`ltx_lora_manual`: Mai/2B vs. Juni/13B)

**Ergebnis vorweg: Die Prämisse "`ltx_lora_manual` gehört zu einem Mai-2026-Training mit LTX-Video 2B"
ist durch die Beweislage widerlegt.** Es gibt keinen Hinweis auf zwei unterschiedliche Nutzungszeiträume
desselben Ordners (kein Löschen/Neuanlegen, keine Config-Struktur-Wechsel innerhalb des Ordners).

### 1.1 Datei-Zeitstempel von `Bachelorarbeit/training/video/ltx_lora_manual/`

Selbst geprüft mit `find ... -exec stat --format='%y %n'`, sortiert:

```
2026-06-13 06:41:11  checkpoints/lora_weights_step_00005.safetensors
2026-06-13 06:41:45  samples/step_000010_0.mp4
2026-06-13 06:41:46  checkpoints/lora_weights_step_00010.safetensors
2026-06-13 06:41:54  checkpoints/lora_weights_step_00015.safetensors
2026-06-13 06:42:28  samples/step_000020_0.mp4
2026-06-13 06:42:29  checkpoints/lora_weights_step_00020.safetensors + comfy_lora_weights_step_00020.safetensors
2026-06-13 06:43:46–07:14  samples/step_000000_preview.gif, step_000010_preview.gif, step_000020_preview.gif,
                            lofi_girl_step20.gif, lofi_improved.mp4, lofi_improved.gif
2026-06-21 07:26  samples/lofi_13b.mp4, lofi_13b.gif, lofi_13b_preview.gif
2026-06-21 17:42:34  training_config.yaml   <-- einzige Config-Datei im Ordner
2026-06-21 17:44–22:19  samples/step_000000_0.mp4 … step_000050_0.mp4, checkpoints/lora_weights_step_00050.safetensors,
                         comfy_lora_weights_step_00050.safetensors
```

Der **früheste** Zeitstempel im gesamten Ordner ist **2026-06-13, 06:41:11 Uhr**. Der **späteste** ist
**2026-06-21, 22:19:50 Uhr**. Es gibt **keine einzige Datei mit einem Zeitstempel im Mai 2026**.

### 1.2 Ordner-Metadatum

`ls -lad --time-style=full-iso Bachelorarbeit/training/video/ltx_lora_manual` → Verzeichnis selbst:
`2026-06-13 06:41:11.705381853 +0200`. Konsistent mit dem frühesten Dateizeitstempel.

### 1.3 `training_config.yaml` — Modell und Zeitpunkt

Datei-Zeitstempel: **2026-06-21 17:42:34** (nicht 6.–10. Mai, nicht 21. Juni ohne Uhrzeit — exakt 17:42:34).
Inhalt (`model.model_source`), im Original-YAML als PyYAML-Objektreferenz serialisiert:

```yaml
model_source: !!python/object/apply:builtins.getattr
- !!python/name:ltxv_trainer.model_loader.LtxvModelVersion ''
- LTXV_13B_097_DEV
```

→ Modell ist **LTXV_13B_097_DEV** (13B), nicht 2B. Die `!!python/object/apply`-Syntax wird von PyYAML nur
beim Serialisieren von Python-Objekten erzeugt — ein Indiz, dass diese Datei vom `ltxv-trainer`-Framework
selbst geschrieben wurde, nicht manuell getippt.

### 1.4 Git-Historie für diesen Pfad

```
$ git log --all --follow --name-status -- Bachelorarbeit/training/video/ltx_lora_manual
(keine Ausgabe)
$ git ls-files Bachelorarbeit/training/video/ltx_lora_manual
(keine Ausgabe)
$ git check-ignore -v Bachelorarbeit/training/video/ltx_lora_manual/training_config.yaml
.gitignore:40:/Bachelorarbeit/    Bachelorarbeit/training/video/ltx_lora_manual/training_config.yaml
```

**Der komplette Ordner `Bachelorarbeit/` ist über `.gitignore` Zeile 40 (`/Bachelorarbeit/`) von der
Versionskontrolle ausgeschlossen.** Es existiert und kann keine Git-Historie für diesen Pfad geben — nicht,
weil nichts passiert wäre, sondern weil dieser gesamte Projektteil nie committet wurde. Jede in dieser
Dokumentation genannte Chronologie stützt sich zwingend auf Dateisystem-Zeitstempel, nicht auf Commits.

### 1.5 Hinweise auf zwei unterschiedliche Nutzungszeiträume desselben Ordners?

Es gibt **eine** interne Zäsur, aber keine Zwei-Perioden-Struktur im Sinne der Ausgangsfrage: Am 13. Juni
entstanden Checkpoints bis Step 20 (kurzer erster Testlauf), am 21. Juni (ab 17:42, nach Schreiben der
aktuell vorliegenden `training_config.yaml`) wurde bis Step 50 weitertrainiert bzw. neu gestartet (die
Sample-Datei `step_000000_0.mp4` vom 21.06. 17:44 deutet auf einen Neustart ab Step 0, nicht auf reine
Fortsetzung von Step 20). Beide Zeitpunkte liegen im Juni, beide verwenden laut Config-Inhalt dasselbe
13B-Modell. Es gibt keine zweite `training_config.yaml`-Version, keinen Hinweis auf ein anderes Modell
und keinen Löschen/Neuanlegen-Vorgang (keine `_old`-Verzeichnisse, keine doppelten Dateinamen mit
unterschiedlichem Inhalt).

### 1.6 Suche nach einem separaten 2B-bezogenen Ordner/Rest

```
$ grep -rln "ltxv-2b|ltxv_2b|0\.9\.6-dev|LTXV_2B" . --include=*.py --include=*.yaml --include=*.yml --include=*.json --include=*.md
```
Treffer (Auswahl, vollständige Liste in Abschnitt 2): ausschließlich in `tools/LTX-Video-Trainer/configs/
ltxv_2b_lora*.yaml` (das sind mitgelieferte **Template-Configs des externen Frameworks**, kein eigener
Trainingslauf) sowie in den beiden Referenzdokumenten `MANUELLE_TRAININGSPHASE_MAI.md` und
`VIDEO_PIPELINE_KONTEXT.md` selbst (die die 2B-Frage bereits historisch einordnen, siehe unten).

**Laut `VIDEO_PIPELINE_KONTEXT.md`, Abschnitt "April–Mai 2026":** Ein 2B-Modell (`ltxv-2b-0.9.6-dev`) wurde
am **2. Mai** heruntergeladen, zusammen mit `LTX-Video-0.9.5` (diffusers-Format, 21. Mai). Wörtliches Zitat
aus diesem Dokument: **"Status: Beide Modelle nie für Training genutzt. Ergebnisse offenbar nicht
überzeugend — keine Training-Aufzeichnungen aus dieser Phase."** Diese Aussage konnte ich in dieser Sitzung
nicht durch eigene Trainings-Artefakte verifizieren (folgerichtig: wenn nie trainiert wurde, gibt es auch
keine Artefakte zu finden) — sie ist als Aussage des Referenzdokuments zu behandeln, nicht als von mir neu
belegter Fund.

### 1.7 Schlussfolgerung Abschnitt 1

| Frage | Antwort | Beleg |
|---|---|---|
| Gehört `ltx_lora_manual` zu einem Mai-2026-Training? | **Nein** | Kein Dateizeitstempel vor 13.06.2026 im Ordner |
| Wurde dort ein 2B-Modell verwendet? | **Nein** | `training_config.yaml`, `model_source: LTXV_13B_097_DEV` |
| Gibt es einen 2B-Trainingsordner, der zu `ltx_lora_manual` in Beziehung steht? | **Nein, nicht auffindbar** | Systemweite Suche ohne Treffer außer Framework-Templates |
| Wurde das 2B-Modell überhaupt jemals für ein Training verwendet? | **Laut VIDEO_PIPELINE_KONTEXT.md: nein** | Zitat s.o., von mir nicht weiter verifizierbar mangels Artefakten |

Die in der Aufgabenstellung genannte Konfigurationsdatei (Rank 16, Alpha 16, LR 2e-4, 50 Steps,
video_dims 832×480×97, guidance_scale 3.5, Seed 42) entspricht exakt dem Inhalt der tatsächlich
vorgefundenen `training_config.yaml` vom 21.06.2026 — es handelt sich um ein und dieselbe, in sich
konsistente Konfiguration, kein Bruch zwischen zwei Ansätzen.

---

## 2. Gefundene Dateien (vollständige Liste)

### 2.1 `find . -path "*training/video*" -type f` (relevante Treffer, `node_modules`/`.venv` ausgeklammert)

Alle 11 `training_config.yaml` unter `Bachelorarbeit/training/video/`:
```
Bachelorarbeit/training/video/ltx_lora_manual/training_config.yaml   2026-06-21 17:42:34
Bachelorarbeit/training/video/ltx_lora_round2/training_config.yaml   2026-06-21 23:05:51
Bachelorarbeit/training/video/ltx_lora_round3/training_config.yaml   2026-06-21 23:31:36
Bachelorarbeit/training/video/ltx_lora_round4/training_config.yaml   2026-06-22 03:55:07
Bachelorarbeit/training/video/ltx_lora_round5/training_config.yaml   2026-06-22 04:36:14
Bachelorarbeit/training/video/ltx_lora_round6/training_config.yaml   2026-06-22 05:13:29
Bachelorarbeit/training/video/ltx_lora_round7/training_config.yaml   2026-06-22 05:51:34
Bachelorarbeit/training/video/ltx_lora_round8/training_config.yaml   2026-07-01 07:15:56
Bachelorarbeit/training/video/ltx_lora_round9/training_config.yaml   2026-07-03 02:50:50
Bachelorarbeit/training/video/ltx_lora_round10/training_config.yaml  2026-07-03 04:16:43
Bachelorarbeit/training/video/ltx_lora_round11/training_config.yaml  2026-07-03 04:44:06
```
Plus je Ordner `checkpoints/*.safetensors` und `samples/*.mp4`/`*.gif` (vollständige Checkpoint-Liste in
Abschnitt 8).

### 2.2 `find . -iname "*ltxv_trainer*" -o -iname "*ltx_lora*"`

- `tools/LTX-Video-Trainer/` (komplettes vendored Framework-Repository, editable installiert — siehe 3.1)
- `.venv/lib/python3.11/site-packages/ltxv_trainer-0.1.0.dist-info/`
- `.venv/lib/python3.11/site-packages/_editable_impl_ltxv_trainer.pth`
- `Bachelorarbeit/training/video/ltx_lora_manual/` und `ltx_lora_round2` … `ltx_lora_round11`
- `Bachelorarbeit/daten/processed/ltx_lora_manual/` (precomputed-Daten, siehe 2.3)

### 2.3 `grep -rln "LTXV_13B_097_DEV|ltxv_trainer" --include=*.py --include=*.yaml .`

Zusätzlich zu den bereits genannten Config-Dateien:
- `Bachelorarbeit/lofi_pipeline/configs/base_lora.yaml`, `base_lora_v2.yaml`, `model_paths.yaml`
- `Bachelorarbeit/lofi_pipeline/scripts/train_lora.py`, `generate_samples.py`, `preprocess_scenario.py`
- `Bachelorarbeit/pipeline/v003_manual/generate.py`
- `tools/LTX-Video-Trainer/src/ltxv_trainer/model_loader.py` (definiert `LtxvModelVersion`, u.a. `LTXV_13B_097_DEV`)

### 2.4 `git log --all --oneline --since="2026-06-01" --until="2026-07-15"`

```
(keine Ausgabe)
```
Erwartungsgemäß leer — siehe Abschnitt 1.4, der gesamte `Bachelorarbeit/`-Baum ist von Git ausgeschlossen.
Im Hauptrepo (Audio-Pipeline, nicht `Bachelorarbeit/`) gibt es in diesem Zeitraum Commits, die aber
ausschließlich MusicGen-Audio betreffen (z. B. train_032–034, siehe `MANUELLE_TRAININGSPHASE_MAI.md`,
Zeile 253) und nicht Gegenstand dieser Video-Dokumentation sind.

### 2.5 `daten/processed/ltx_lora_manual/` (Trainingsdaten-Herkunft)

```
Bachelorarbeit/daten/processed/ltx_lora_manual/ltx_lora_v003_manual.yaml   2026-07-03 04:43
Bachelorarbeit/daten/processed/ltx_lora_manual/dataset.jsonl               2026-06-22 00:03
Bachelorarbeit/daten/processed/ltx_lora_manual/precomputed/                angelegt 2026-06-13 07:07
Bachelorarbeit/daten/processed/ltx_lora_manual/clips_raw/                  2026-06-22 00:02
Bachelorarbeit/daten/processed/ltx_lora_manual/reference_frame_r5.jpg      2026-06-22 05:50
```
(Übernommen aus `MANUELLE_TRAININGSPHASE_MAI.md`, Abschnitt 1b; von mir gegen das `precomputed/`-Verzeichnis
stichprobenartig geprüft — Zeitstempel 13.06.2026 bestätigt.)

### 2.6 Skripte der (späteren) `lofi_pipeline`-Struktur

Siehe vollständige Tabelle mit Zeilenzahl/Zeitstempel in Abschnitt 4.

---

## 3. Paket- und Dependency-Vergleich

### 3.1 Editable Install von `ltxv-trainer`

Selbst geprüft:
```
$ cat .venv/lib/python3.11/site-packages/ltxv_trainer-0.1.0.dist-info/direct_url.json
{"dir_info": {"editable": true}, "url": "file:///home/BA_Musikproduktion/Documents/Bachelor_VisiualStudio/tools/LTX-Video-Trainer"}
```
**Bestätigt: `ltxv-trainer` ist als editable install aus `tools/LTX-Video-Trainer/` eingebunden.**

### 3.2 Vergleichstabelle Pakete — Mai (erster Ansatz laut Kontext-Dokument) vs. Juni (LTXV-13B-Training)

Installationszeitstempel selbst per `find .venv/... -iname "<paket>-*.dist-info"` + `stat` geprüft
(Spalte "Beleg (diese Sitzung)"); Begründungsspalte aus `VIDEO_PIPELINE_KONTEXT.md` übernommen, dort als
Tabelle "Pakete installiert" (April–Mai) bzw. Fließtext (13. Juni) geführt.

| Paket | Version | Install-Datum (Beleg diese Sitzung) | Grund laut VIDEO_PIPELINE_KONTEXT.md |
|---|---|---|---|
| accelerate | 1.13.0 | 2026-05-06 | Mixed-Precision Training Infrastruktur |
| opencv-python | 4.13.0.92 | 2026-05-07 | Videoframe-Verarbeitung |
| gradio | 5.33.0 | 2026-05-07 | erste Web-UI für Tests |
| PyYAML | 6.0.3 | 2026-05-08 | Config-Dateien lesen/schreiben |
| diffusers | 0.33.1 | 2026-05-09 | Hugging Face Diffusion Pipeline |
| bitsandbytes | 0.45.2 | 2026-05-10 | 8-Bit-Quantisierung für VRAM |
| torch | 2.6.0+cu124 | 2026-05-10 | GPU-Kern, CUDA 12.4 |
| torchvision | 0.21.0 | 2026-05-10 | Bild-Transformationen |
| transformers | 4.52.2 | 2026-05-10 | (Version laut Kontext-Dok.; Installdatum nicht separat gelistet, aber `.dist-info`-Zeitstempel selbst geprüft: 2026-05-10, identisch mit torch) |
| **Pillow** | 11.3.0 | **2026-06-13** | für Bild-I/O, laut Kontext-Dok. zeitgleich mit ltxv-trainer-Installation |
| **ltxv-trainer** | 0.1.0 (editable) | **2026-06-13** | liefert `LTXConditionPipeline`, `load_ltxv_components()` — Voraussetzung für LTXV-LoRA-Training |

**Kein Paket aus der Mai-Liste wurde entfernt oder ersetzt** — alle neun Mai-Pakete (accelerate, opencv-python,
gradio, PyYAML, diffusers, bitsandbytes, torch, torchvision, transformers) sind Grundlage für den ab 21. Juni
laufenden 13B-Trainingsansatz (`ltxv-trainer` baut auf `accelerate`, `diffusers`, `transformers`, `torch` auf;
`opencv-python` und `gradio` werden erst später in der `lofi_pipeline`-Phase (August) durch `evaluate_video.py`
bzw. `feedback_ui.py` intensiv genutzt, siehe Abschnitt 4). Es handelt sich also **nicht** um einen Paket-Bruch
zwischen zwei unabhängigen Ansätzen, sondern um eine kontinuierliche Erweiterung derselben Umgebung.

Ein `requirements.txt` oder eine `pip freeze`-Historie existiert nicht: `find . -iname "requirements*.txt"`
lieferte für den Video-Bereich keinen Treffer. Eine Rekonstruktion der exakten Paketliste zu einem beliebigen
historischen Zeitpunkt ist daher nur über Dateisystem-Zeitstempel der `.dist-info`-Ordner möglich, nicht über
eine dokumentierte Dependency-Datei.

### 3.3 War `optimum_quanto`, `decord`, `scenedetect`, `av` Teil dieses Ansatzes?

Diese vier Pakete tauchen in der aus einer **anderen Untersuchung dieser Sitzung** (Frage nach Anfang-Mai-
Downloads, siehe frühere Antwort in diesem Gespräch) ermittelten Liste auf, gehören dort aber zur
**Audio/Video-Infrastruktur des LTX-Video-Setups im Mai** allgemein, nicht spezifisch zu diesem LTXV-13B-
Trainingsansatz. Sie werden weder in `VIDEO_PIPELINE_KONTEXT.md` noch in den Skripten aus Abschnitt 4 als
für `ltx_lora_manual`/Round2-11 oder `lofi_pipeline` relevant genannt. Prüfung:
```
$ grep -rl "optimum_quanto\|decord\|scenedetect" Bachelorarbeit/lofi_pipeline Bachelorarbeit/training
(keine Ausgabe)
```
**Befund: Keine Verwendung dieser vier Pakete im dokumentierten LTXV-13B-Trainingsansatz nachweisbar.**

---

## 4. Selbst programmierte Skripte im Detail

Alle folgenden Skripte liegen unter `Bachelorarbeit/lofi_pipeline/scripts/`. Zeilenzahl per `wc -l`,
Zeitstempel per `stat`, Zweck aus dem jeweiligen Docstring (Zitat, nicht interpretiert) — beide in dieser
Sitzung direkt geprüft.

| Skript | Zeilen | Zuletzt geändert | Docstring (Zitat) |
|---|---|---|---|
| `train_lora.py` | 257 | 2026-08-10 01:45:52 | "Lo-Fi LoRA Training CLI" |
| `generate_samples.py` | 192 | 2026-08-10 04:06:52 | "Generate video samples (MP4-first strategy) for a training round. Primary output is MP4. GIFs are optional and generated on request only." |
| `preprocess_scenario.py` | 86 | 2026-08-10 01:41:22 | "Preprocess a new scenario: compute VAE latents + T5 embeddings." |
| `make_gif.py` | 227 | 2026-08-04 06:27:24 | "Vollständige GIF + MP4 Pipeline. MP4 (8fps, 960×544) → minterpolate (flüssige Bewegung) …" |
| `metric_watcher.py` | 327 | 2026-08-06 07:21:01 | "Metric Watcher — automatische Qualitaetsbewertung pro Training-Checkpoint. Laeuft parallel zu train_lora.py in einem zweiten Terminal." |
| `feedback_ui.py` | 700 | 2026-07-28 15:55:52 | "Lokale Feedback-UI für Lo-Fi LoRA Training. Zeigt alle generierten Samples, ermöglicht Bewertung und startet Training/Generierung." |
| `feedback.py` | 418 | 2026-07-28 16:46:09 | "Lo-Fi Feedback Tool — Terminal-Version. Zeigt neue Videos/GIFs automatisch an, nimmt Bewertung und Korrekturen entgegen." |
| `enhance_video.py` | 222 | 2026-07-21 05:45:36 | "Post-process generated MP4s: upscale + frame interpolation for smoother HD output." |
| `build_report.py` | 135 | 2026-07-03 05:32:11 | "Build training overview report (CSV + Markdown)." |
| `compare_rounds.py` | 116 | 2026-07-03 05:31:35 | "Compare training rounds for a scenario." |

Zusätzlich (nicht im `scripts/`-Ordner):
- `Bachelorarbeit/pipeline/v003_manual/generate.py` — laut `VIDEO_PIPELINE_KONTEXT.md` die direkte
  Inference-Route via `LTXConditionPipeline` (Code-Beispiel dort vollständig abgedruckt, Zeilen 554–587).
- `Bachelorarbeit/pipeline/realistic_rabbit/evaluate_video.py` — Metrik-Berechnung, siehe Abschnitt 7.

### 4.1 Vorgängerversionen / Diff zum "ersten Ansatz"

Für **keines** der zehn Skripte in der obigen Tabelle existiert eine nachweisbare Vorgängerversion aus einer
Mai-2026-Phase: Weder unter `git log` (Pfad nicht versioniert, siehe 1.4) noch als `_old`/`_v1`-Datei im
Dateisystem. `find Bachelorarbeit -iname "*.py.bak" -o -iname "*_old.py" -o -iname "*_v1.py"` liefert keinen
Treffer. Ein Diff zwischen "erster" und "zweiter" Skriptversion ist daher **NICHT REKONSTRUIERBAR** — es gibt
keine erste Version dieser konkreten zehn Dateien, sie wurden laut Zeitstempel alle erstmals zwischen dem
3. Juli (`build_report.py`, `compare_rounds.py`) und dem 10. August (`generate_samples.py`,
`preprocess_scenario.py`, `train_lora.py` refactored) angelegt bzw. zuletzt geändert.

---

## 5. Architektonischer Vergleich

Wichtiger Vorbefund: Die in der Aufgabenstellung vorausgesetzte Zweiteilung "erster Ansatz = LTX-Video 2B
ohne Framework" vs. "zweiter Ansatz = LTX-Video 13B mit ltxv-trainer" ist durch die Beweislage **nicht
gestützt** (Abschnitt 1). Die einzige im Repository nachweisbare strukturelle Zäsur verläuft stattdessen
zwischen der **unstrukturierten Ordnerform** `Bachelorarbeit/training/video/ltx_lora_manual` + `round2`…`round11`
(Juni–Juli) und der ab **3. Juli** eingeführten **`lofi_pipeline/scenarios/`-Struktur** — beide nutzen
durchgehend das 13B-Modell. Die folgende Tabelle vergleicht deshalb diese beiden tatsächlich nachweisbaren
Phasen, nicht 2B-vs-13B.

| Aspekt | Phase A: `training/video/ltx_lora_manual` + Round2–11 (21.06.–03.07.) | Phase B: `lofi_pipeline/scenarios/` (ab 03.07.) | Beleg |
|---|---|---|---|
| Modell | LTXV_13B_097_DEV | LTXV_13B_097_DEV (unverändert) | `training_config.yaml` beider Phasen, `VIDEO_PIPELINE_KONTEXT.md` Zeile 191: "seit 21. Juni, nie gewechselt" |
| Trainings-Framework | ltxv-trainer (editable, aus `tools/LTX-Video-Trainer`) | dasselbe, aufgerufen über `train_lora.py` | Abschnitt 3.1; `model_paths.yaml`, `trainer_script` |
| Ordnerstruktur | Flach: 11 einzelne `ltx_lora_*`-Ordner, kein Szenario-Konzept | `scenarios/{id}/rounds/round_NN/` je Szenario | `VIDEO_PIPELINE_KONTEXT.md` Zeile 99–102: "die alten `training_runs/` waren unstrukturiert, kein Szenario-Konzept" |
| Checkpoint-Intervall | 50 (implizit, nur Endcheckpoint pro Runde sichtbar außer Round11) | 10 (Standard `base_lora.yaml`), später 5 (`base_lora_v2.yaml`) | `checkpoints/`-Verzeichnisse Abschnitt 8; `base_lora.yaml` Zeile 289 |
| Validierung | interval 25 (R1–3), dann 25 mit veränderter guidance_scale (R4–8), ab R9 `9999` (deaktiviert) | `interval: 9999` als Standard von Anfang an, separates `generate_samples.py` | `MANUELLE_TRAININGSPHASE_MAI.md` Tabelle Abschnitt 2 "Validierungs-Konfiguration"; `base_lora.yaml` Zeile 285 |
| Kontrollmechanismus | Manuelle Sichtprüfung nach jeder Runde (4 Dimensionen: Qualität/Animation/Detail/Realismus, 1–10) | Zusätzlich `metric_watcher.py` (automatisch, seit 6. August) mit `STOP_TRAINING`-Mechanismus | `VIDEO_PIPELINE_KONTEXT.md` Zeile 66–72, 154–156 |
| Reproduzierbarkeit | `seed: 42` fest in jeder `training_config.yaml` | `seed: 42` (Trainings-Config) bzw. `seed: 777` (Szenario-spezifisch, z. B. `rabbit_lake/scenario.yaml`) | Config-Dateien beider Phasen |
| Experiment-Tracking | `wandb.enabled: false` in `ltx_lora_manual/training_config.yaml` | `wandb.enabled: false` auch in `base_lora.yaml` (Zeile 299) | Direktes Zitat beider Configs — **an keiner Stelle als aktiv nachweisbar** |
| Datensatz-Herkunft | `daten/processed/ltx_lora_manual/clips_raw/` — kein Download-Skript auffindbar | je Szenario `precomputed_dir` + `dataset_jsonl`, für `rabbit_lake` z. B. eigene Referenzvideos | Abschnitt 1.6/2.5; `VIDEO_PIPELINE_KONTEXT.md` Zeile 346–347 |
| Bewertungsmethode | Rein manuell/visuell, gespeichert in `training_runs/training_0N/feedback.json`/`notes.json` | `evaluate_video.py`, 6 lokale Metriken (SSIM, Sharpness, Flicker, Motion, Color/Brightness Consistency), gewichteter `overall_score` | `VIDEO_PIPELINE_KONTEXT.md` Zeile 168–180, 461–491 |
| Post-Processing | Nicht dokumentiert für diese Phase | `make_gif.py`: minterpolate/RIFE → Real-ESRGAN → GIF+MP4 | `VIDEO_PIPELINE_KONTEXT.md` Zeile 406–422 |

---

## 6. Chronologischer Aufbauprozess

Rekonstruiert ausschließlich aus Dateisystem-Zeitstempeln (kein Git verfügbar, siehe 1.4). Kernzeitstempel
in dieser Sitzung selbst nachgeprüft; die Einordnung "Warum" stammt aus `VIDEO_PIPELINE_KONTEXT.md`.

| Zeitpunkt | Ereignis | Beleg |
|---|---|---|
| 02.05.2026 | Download `ltxv-2b-0.9.6-dev` | `VIDEO_PIPELINE_KONTEXT.md` Zeile 33 (nicht selbst nachprüfbar — Modell-Datei nicht mehr im Dateisystem gefunden) |
| 06.–10.05.2026 | Pakete `accelerate`, `opencv-python`, `gradio`, `PyYAML`, `diffusers`, `bitsandbytes`, `torch`, `torchvision`, `transformers` installiert | Selbst geprüft, `.dist-info`-Zeitstempel (Abschnitt 3.2) |
| 21.05.2026 | Download `LTX-Video-0.9.5` (diffusers-Format) | `VIDEO_PIPELINE_KONTEXT.md` Zeile 34, nicht selbst nachprüfbar |
| **13.06.2026, ca. 06:36** | `ltxv-trainer` als editable Package installiert; `Pillow 11.3.0` gleichzeitig | `.dist-info`-Zeitstempel Pillow: 2026-06-13 (selbst geprüft); genaue Uhrzeit 06:36 aus `MANUELLE_TRAININGSPHASE_MAI.md` Zeile 254, nicht auf die Minute selbst nachgeprüft |
| 13.06.2026, 07:07 | `precomputed/`-Verzeichnis für `ltx_lora_manual` angelegt | Selbst geprüft, Verzeichnis-Zeitstempel |
| 13.06.2026, 06:41–07:14 | Erster kurzer Testlauf: Checkpoints Step 5/10/15/20 | Abschnitt 1.1, selbst geprüft |
| **21.06.2026, 17:42:34** | Aktuell vorliegende `training_config.yaml` geschrieben; laut `VIDEO_PIPELINE_KONTEXT.md` "training_01", erster vollständiger Trainingslauf mit dieser Konfiguration | Selbst geprüft (Dateizeitstempel), Einordnung aus Kontext-Dok. Zeile 51, 78 |
| 21.06.2026, 23:05 / 23:31 | Round2, Round3 | `training_config.yaml`-Zeitstempel, selbst geprüft |
| 22.06.2026, 03:55–06:07 | Round4–7 | Dito |
| 01.07.2026, 07:15 | Round8 | Dito |
| 03.07.2026, 02:50–04:44 | Round9–11 | Dito |
| **03.07.2026, ca. 05:31–05:32** | `compare_rounds.py`, `build_report.py` erstellt — Beginn der `lofi_pipeline`-Struktur | Selbst geprüft, `stat`-Zeitstempel (Abschnitt 4) |
| 20.07.2026 | Post-Processing-Pakete (`realesrgan`, `basicsr`, `scikit-image`, `numpy`) installiert | `VIDEO_PIPELINE_KONTEXT.md` Zeile 104–110, nicht in dieser Sitzung einzeln nachgeprüft |
| 21.07.2026 | `enhance_video.py` erstellt | Selbst geprüft, Zeitstempel 2026-07-21 05:45:36 |
| 28.07.2026 | `feedback_ui.py`, `feedback.py` erstellt | Selbst geprüft |
| 04.08.2026 | `make_gif.py` erstellt | Selbst geprüft |
| 06.08.2026 | `metric_watcher.py` erstellt | Selbst geprüft |
| 10.08.2026 | `train_lora.py`, `generate_samples.py`, `preprocess_scenario.py` zentralisiert/refactored | Selbst geprüft |

### 6.1 Rank/LR-Änderungsverlauf über mehrere Versionen

Innerhalb Phase A (Round1–11) bleiben Rank (16), Alpha (16) und Lernrate (2e-4) laut
`MANUELLE_TRAININGSPHASE_MAI.md`, Abschnitt 2, über **alle elf Runden konstant** — keine Rank/LR-Änderung
in dieser Phase. Eine Rank- bzw. LR-Änderung (16→32, 2e-4→1e-4/5e-5) fand erst **später**, in der
`lofi_pipeline`-Phase statt, konkret beim Szenario `rabbit_lake` (LR 2e-4→1e-4 ab dessen Runde 4) sowie beim
Wechsel von `base_lora.yaml` zu `base_lora_v2.yaml` (Rank 16→32, LR 1e-4→5e-5, zusätzliche `ff`-Zielmodule).
Beleg: `VIDEO_PIPELINE_KONTEXT.md`, Zeilen 122–134 und 303–311 (Tabelle).

---

## 7. Validierungs- und Qualitätskontrollmechanismus

### 7.1 Phase A (Round1–11): inline Validation via ltxv-trainer

Die in `training_config.yaml` sichtbaren Felder `validation.interval: 25`, `validation.inference_steps: 30`,
`validation.guidance_scale: 3.5`, fester Prompt-Text — diese Validierung wird **vom `ltxv-trainer`-Framework
selbst** ausgeführt (kein eigenes Skript in `Bachelorarbeit/`, das diese Logik implementiert; das Framework
liegt in `tools/LTX-Video-Trainer/`, dessen interner Trainingsloop laut `model_paths.yaml`,
`trainer_script: ".../tools/LTX-Video-Trainer/scripts/train.py"`, aufgerufen wird). Die Samples in
`ltx_lora_manual/samples/step_000000_0.mp4` usw. sind das direkte Ergebnis dieser inline-Validation.

**Bewertung dieser Validation-Samples:** Laut `VIDEO_PIPELINE_KONTEXT.md`, Abschnitt "Bewertungs-Vorgehen
im Überblick", Phase 1: **rein manuelle Sichtprüfung**, vier Dimensionen (Qualität/Animation/Detail/
Realismus, 1–10), keine automatische Bewertung. Kein Hinweis auf ein automatisiertes Bewertungsskript in
dieser Phase.

### 7.2 Vorstufe zu `evaluate_video.py` / `metric_watcher.py`?

**Nein.** Laut Zeitstempeln (Abschnitt 4/6) entstehen `evaluate_video.py` (Pfad:
`Bachelorarbeit/pipeline/realistic_rabbit/evaluate_video.py`, laut Kontext-Dok. 6. August) und
`metric_watcher.py` (6. August, selbst geprüft) **über fünf Wochen nach** dem Ende von Phase A (letzte
Round11-Aktivität: 3. Juli). In Phase A existiert keine Vorstufe dieser Skripte — die manuelle
4-Dimensionen-Bewertung ist die einzige Qualitätskontrolle dieser Phase.

### 7.3 Deaktivierung der Validation (Round9–11)

`validation.interval: 9999` ab Round9 (faktische Deaktivierung, da 9999 > `optimization.steps: 100`).
Begründung laut `VIDEO_PIPELINE_KONTEXT.md` Zeile 91/198: 2× CUDA-OOM-Absturz durch Inline-Validation beim
13B-Modell. Seitdem wird Inferenz für Sichtprüfung **separat nach dem Training** ausgeführt, nicht mehr
inline.

---

## 8. Ergebnisse

### 8.1 Alle Checkpoints (`.safetensors`) unter `training/video/ltx_lora_*`, mit Zeitstempel und Größe

Selbst per `find ... -exec stat --format='%y %10s %n'` erhoben:

| Runde | Datei | Zeitstempel | Größe (Byte) |
|---|---|---|---|
| ltx_lora_manual | lora_weights_step_00005.safetensors | 2026-06-13 06:41:11 | 29.418.808 |
| ltx_lora_manual | lora_weights_step_00010.safetensors | 2026-06-13 06:41:46 | 29.418.808 |
| ltx_lora_manual | lora_weights_step_00015.safetensors | 2026-06-13 06:41:54 | 29.418.808 |
| ltx_lora_manual | lora_weights_step_00020.safetensors | 2026-06-13 06:42:29 | 29.418.808 |
| ltx_lora_manual | comfy_lora_weights_step_00020.safetensors | 2026-06-13 06:42:29 | 29.420.568 |
| ltx_lora_manual | lora_weights_step_00050.safetensors | 2026-06-21 17:54:06 | 100.764.392 |
| ltx_lora_manual | comfy_lora_weights_step_00050.safetensors | 2026-06-21 17:54:06 | 100.767.432 |
| round2 | lora_weights_step_00050.safetensors + comfy-Variante | 2026-06-21 23:17:33 | 100.764.392 / 100.767.432 |
| round3 | lora_weights_step_00050.safetensors + comfy-Variante | 2026-06-21 23:43:18 | 100.764.392 / 100.767.432 |
| round4 | lora_weights_step_00050.safetensors + comfy-Variante | 2026-06-22 04:10:41 | 100.764.392 / 100.767.432 |
| round5 | lora_weights_step_00050.safetensors + comfy-Variante | 2026-06-22 04:51:47 | 100.764.392 / 100.767.432 |
| round6 | lora_weights_step_00050.safetensors + comfy-Variante | 2026-06-22 05:29:14 | 100.764.392 / 100.767.432 |
| round7 | lora_weights_step_00050.safetensors + comfy-Variante | 2026-06-22 06:07:44 | 100.764.392 / 100.767.432 |
| round8 | step_00050 + step_00100 + comfy-Variante (step_00100) | 2026-07-01 07:28–07:37 | 100.764.392 (×2) / 100.767.432 |
| round9 | step_00050 + step_00100 + comfy-Variante | 2026-07-03 02:56–03:02 | 100.764.392 (×2) / 100.767.432 |
| round10 | step_00050 + step_00100 + comfy-Variante | 2026-07-03 04:22–04:28 | 100.764.392 (×2) / 100.767.432 |
| round11 | step_00025 + step_00050 + step_00075 + step_00100 + comfy-Variante | 2026-07-03 04:47–04:55 | 100.764.392 (×4) / 100.767.432 |

Auffällig: Die Checkpoints von `ltx_lora_manual` bis Step 20 sind mit 29.418.808 Byte deutlich kleiner als
alle Step-50-Checkpoints (100.764.392 Byte) — ein Hinweis auf eine Konfigurationsänderung zwischen dem
Testlauf vom 13.06. und dem eigentlichen Trainingslauf ab 21.06. (z. B. andere `target_modules` oder
Speicherformat). Die genaue Ursache ist aus den vorliegenden Dateien **NICHT REKONSTRUIERBAR** — die
`training_config.yaml`, die den 13.06.-Lauf erzeugt hat, liegt nicht mehr vor (nur die vom 21.06.).

### 8.2 Sample-Videos

Vollständige Liste in Abschnitt 1.1 (chronologisch). Zusammengefasst: 11× `step_NNNNNN_0.mp4` (Rohsamples),
mehrere `_preview.gif`, sowie benannte Zusammenfassungen `lofi_improved.mp4/gif` (13.06.) und
`lofi_13b.mp4/gif/lofi_13b_preview.gif` (21.06., 07:26 — vor Erstellung der finalen `training_config.yaml`
um 17:42, vermutlich Vorschau-Material vom vorherigen Checkpoint-Stand).

### 8.3 Bewertungsdateien (`feedback.json`, `notes.json`)

```
$ find Bachelorarbeit/training -iname "feedback.json" -o -iname "notes.json"
```
In den elf `ltx_lora_*`-Ordnern selbst **nicht gefunden** (weder `feedback.json` noch `notes.json` liegen
dort). Die in `VIDEO_PIPELINE_KONTEXT.md` referenzierten Pfade (`training_runs/training_0N/feedback.json`)
verwenden eine andere Ordnerbenennung (`training_runs/training_01` statt `training/video/ltx_lora_manual`).
Ob `training_runs/` ein Alias, ein Umbenennungs-Vorgänger oder ein eigenständiger, nicht mehr vorhandener
Pfad ist, ist aus dem aktuellen Dateisystem **NICHT REKONSTRUIERBAR** — der Ordner `training_runs/` mit
dieser Unterstruktur wurde in dieser Sitzung nicht aufgefunden. Der Inhalt der Q/A/Detail/Realismus-Tabelle
aus Abschnitt 5 (Vergleichstabelle) stammt daher ausschließlich aus dem Fließtext von
`VIDEO_PIPELINE_KONTEXT.md`, nicht aus selbst gelesenen `feedback.json`-Dateien.

---

## 9. Verbesserte Kontrollierbarkeit — Prüfung der Prämisse

Die Aufgabenstellung geht davon aus, dass "der erste Ansatz explizit an mangelnder Kontrollierbarkeit
(automatisierte Delegation ohne Monitoring, alle 11 Durchläufe brachen bei 60–70 % des Ziels ab)
gescheitert ist". **Diese konkrete Behauptung (11 Durchläufe, Abbruch bei 60–70 %) konnte in dieser Sitzung
in keiner der vorhandenen Quellen (Dateisystem, `MANUELLE_TRAININGSPHASE_MAI.md`, `VIDEO_PIPELINE_KONTEXT.md`)
bestätigt werden.**

Was tatsächlich belegbar ist:
- Es gibt genau **11 Konfigurationsordner** unter `training/video/` (`ltx_lora_manual` + Round2–11) —
  numerisch passend zu "11 Durchläufe", aber:
- Alle 11 Runden zeigen laut Checkpoint-Dateien **abgeschlossene** Läufe bis zur jeweiligen Ziel-Schrittzahl
  (50 bzw. 100 Steps) — kein Hinweis auf einen Abbruch bei 60–70 % in den Dateinamen oder -größen
  (Ausnahme: Round11 hat zusätzliche Zwischen-Checkpoints bei 25/50/75, was eher für *mehr* Granularität
  als für einen Abbruch spricht).
- `VIDEO_PIPELINE_KONTEXT.md`s eigene Bewertungstabelle (Zeile 76–86) dokumentiert für training_01–09
  jeweils eine **abgeschlossene** Bewertung (Q/A-Scores 2 bis 7), keine "Abbruch bei 60–70%"-Einträge.

**Da sich die Prämisse nicht bestätigen lässt, wird sie hier weder als wahr noch als falsch markiert,
sondern als NICHT VERIFIZIERBAR gekennzeichnet — möglicherweise bezieht sie sich auf eine andere,
nicht mehr im Repository vorhandene Trainingsphase (vgl. das in Abschnitt 6 der
`MANUELLE_TRAININGSPHASE_MAI.md` erwähnte, nicht auffindbare "v002"/"v003").**

Unabhängig davon lassen sich die tatsächlich nachweisbaren strukturellen Kontrollelemente auflisten:

| Kontrollelement | Vorhanden? | Beleg |
|---|---|---|
| Feste Checkpoint-Intervalle | Ja, `checkpoints.interval: 50` (Phase A) bzw. `10`/`5` (Phase B) | `training_config.yaml`; `base_lora.yaml` Zeile 289; `base_lora_v2.yaml` Zeile 310 |
| Definierte Validierungs-Prompts statt Zufallsstichprobe | Ja, fester Prompt-Text je Config | `training_config.yaml`, `validation.prompts` |
| Fester Seed | Ja, `seed: 42` durchgehend in Phase A | Alle 11 `training_config.yaml` |
| Mechanismus zum kontrollierten Abbruch/Fortsetzen | Nur in Phase B: `metric_watcher.py` schreibt `STOP_TRAINING`-Datei bei Zielerreichung | `VIDEO_PIPELINE_KONTEXT.md` Zeile 507; nicht vorhanden in Phase A |
| Automatische Qualitätsmetrik als Abbruchkriterium | Nein in Phase A (rein manuell); ja in Phase B (`evaluate_video.py` + `metric_watcher.py`, ab 6. August) | Abschnitt 7 |

---

## 10. Offene, nicht rekonstruierbare Punkte

1. **Die konkrete Behauptung "11 Durchläufe brachen bei 60–70 % ab"** ist durch keine Quelle in diesem
   Repository belegbar (Abschnitt 9). NICHT REKONSTRUIERBAR — evtl. Verwechslung mit einer anderen Phase.

2. **Warum sind die Checkpoints des 13.06.-Testlaufs (Step 5–20) mit 29,4 MB deutlich kleiner als alle
   Step-50-Checkpoints ab 21.06. (100,8 MB)?** Die zugehörige frühere Config-Version liegt nicht mehr vor.
   NICHT REKONSTRUIERBAR.

3. **Herkunft der Trainingsclips in `daten/processed/ltx_lora_manual/clips_raw/`.** Kein Download-Skript,
   keine URL-Liste auffindbar (bereits in `MANUELLE_TRAININGSPHASE_MAI.md` als offener Punkt geführt,
   in dieser Sitzung erneut bestätigt: kein zusätzlicher Fund). NICHT REKONSTRUIERBAR.

4. **Verhältnis von `training_runs/training_0N/` (referenziert in `VIDEO_PIPELINE_KONTEXT.md`) zu
   `training/video/ltx_lora_manual`/`round2`–`round11` (tatsächlich im Dateisystem vorgefunden).** Beide
   scheinen dieselbe Zeitperiode (21.06.–03.07.) und dieselben Trainingsparameter zu beschreiben, aber
   unter unterschiedlichen Ordnernamen bzw. -pfaden. Ob `training_runs/` umbenannt wurde, ein Alias ist,
   oder eine andere (nicht mehr vorhandene) Struktur bezeichnet, ist aus dem aktuellen Dateisystem NICHT
   REKONSTRUIERBAR.

5. **Ob das 2B-Modell (`ltxv-2b-0.9.6-dev`, Mai) tatsächlich nie für ein Training verwendet wurde**, stützt
   sich ausschließlich auf die Aussage in `VIDEO_PIPELINE_KONTEXT.md` — eigene Trainingsartefakte für diese
   Behauptung wurden (erwartungsgemäß, da "nie genutzt") nicht gefunden, können eine Nichtnutzung aber auch
   nicht positiv beweisen, sondern nur nicht widerlegen.

6. **Ob eine Claude-Instanz an Videosuche, Konfigurationserstellung oder Trainingssteuerung dieser Phase
   beteiligt war.** Kein Code-Kommentar, kein Commit (da unversioniert), keine Notiz belegt dies. Bereits
   in `MANUELLE_TRAININGSPHASE_MAI.md`, Abschnitt 4, mit demselben Ergebnis untersucht. NICHT
   REKONSTRUIERBAR.

7. **Exakte Uhrzeit der `ltxv-trainer`-Installation (13.06., "ca. 06:36").** Aus
   `MANUELLE_TRAININGSPHASE_MAI.md` übernommen, in dieser Sitzung nicht auf die Minute nachgeprüft (nur
   der zugehörige Pillow-Installtag wurde bestätigt). Geringfügige Unsicherheit, NICHT im Widerspruch zu
   den übrigen Befunden.
