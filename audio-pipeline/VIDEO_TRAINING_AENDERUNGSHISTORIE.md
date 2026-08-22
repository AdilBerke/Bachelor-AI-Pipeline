# Änderungshistorie des Video-Trainings unter `Bachelorarbeit/training/video/` — vollständige Chronik mit Begründungen

Stand: 2026-08-17. Diese Datei ergänzt `ZWEITER_TRAININGSANSATZ_DOKUMENTATION.md` (dort: Architektur- und
Widerspruchs-Klärung) um eine lückenlose, runde-für-runde bzw. änderung-für-änderung dokumentierte
Chronik: **was genau geändert wurde und warum**.

**Methodik:** Jede `training_config.yaml` der 11 Runden (`ltx_lora_manual`, `ltx_lora_round2`…`round11`)
wurde in dieser Sitzung per Python/PyYAML eingelesen, in Einzelfelder zerlegt und paarweise mit der
jeweiligen Vorgänger-Runde verglichen (`diff` auf Feldebene, nicht nur Sichtprüfung). Jede unten gelistete
Änderung ist das Ergebnis dieses maschinellen Vergleichs, nicht einer manuellen Durchsicht. Die
Begründungen ("Warum") stammen, wo nicht anders vermerkt, wörtlich oder sinngemäß aus
`VIDEO_PIPELINE_KONTEXT.md` (Referenzdokument im Projekt-Wurzelverzeichnis) — dort ist es die einzige
Quelle, die Begründungen überhaupt festhält, da unter `Bachelorarbeit/` keine Git-Historie existiert
(siehe `ZWEITER_TRAININGSANSATZ_DOKUMENTATION.md`, Abschnitt 1.4). Wo keine Begründung auffindbar ist,
steht das explizit da — es wird nichts erfunden.

---

## Teil A: Die 11 Runden unter `training/video/ltx_lora_*` (21.06.–03.07.2026)

### Übersichtstabelle: Was änderte sich in welcher Runde?

| Runde | Zeitstempel | Geänderte Felder (maschinell verglichen) |
|---|---|---|
| ltx_lora_manual | 21.06. 17:42 | — (erste Runde, kein Vorgänger) |
| round2 | 21.06. 23:05 | `model.load_checkpoint`, `output_dir`, `validation.prompts` |
| round3 | 21.06. 23:31 | `model.load_checkpoint`, `output_dir` (nur Checkpoint-Kette, keine Parameteränderung) |
| round4 | 22.06. 03:55 | `model.load_checkpoint`, `output_dir`, `validation.guidance_scale`, `validation.inference_steps`, `validation.negative_prompt`, `validation.prompts` |
| round5 | 22.06. 04:36 | `model.load_checkpoint`, `output_dir`, `validation.negative_prompt`, `validation.prompts` |
| round6 | 22.06. 05:13 | `model.load_checkpoint`, `output_dir`, `validation.negative_prompt`, `validation.prompts` |
| round7 | 22.06. 05:51 | `output_dir`, `validation.images`, `validation.negative_prompt`, `validation.prompts` (**kein** `model.load_checkpoint`-Wechsel — siehe 4.7) |
| round8 | 01.07. 07:15 | `model.load_checkpoint`, `optimization.steps`, `output_dir`, `validation.images`, `validation.interval` |
| round9 | 03.07. 02:50 | `model.load_checkpoint`, `output_dir`, `validation.inference_steps`, `validation.interval`, `validation.negative_prompt`, `validation.prompts` |
| round10 | 03.07. 04:16 | `model.load_checkpoint`, `output_dir` (nur Checkpoint-Kette) |
| round11 | 03.07. 04:44 | `checkpoints.interval`, `model.load_checkpoint`, `output_dir` |

Konstant über **alle elf Runden** (kein einziges Mal geändert): `model.model_source` (immer
`LTXV_13B_097_DEV`), `lora.rank` (16), `lora.alpha` (16), `lora.dropout` (0.0), `lora.target_modules`
(`to_k`, `to_q`, `to_v`, `to_out.0`), `optimization.learning_rate` (0.0002), `optimization.batch_size` (1),
`optimization.optimizer_type` (adamw8bit), `optimization.scheduler_type` (cosine),
`optimization.gradient_accumulation_steps` (1), `acceleration.mixed_precision_mode` (bf16),
`acceleration.load_text_encoder_in_8bit` (true), `data.preprocessed_data_root`, `seed` (42),
`validation.seed` (42), `validation.video_dims` ([832, 480, 97]), `validation.videos_per_prompt` (1),
`wandb.enabled` (false), `checkpoints.keep_last_n` (-1).

---

### 4.1 Round1 → Round2 (21.06. 17:42 → 23:05, ca. 5h 23min später)

**Änderung 1 — `model.load_checkpoint`:** `None` → Pfad auf `ltx_lora_manual/checkpoints/
lora_weights_step_00050.safetensors`.
*Beobachtung:* Round2 startet nicht neu, sondern lädt den Endcheckpoint (Step 50) von Round1 als Basis.
*Begründung:* Nicht explizit dokumentiert; aus der Struktur ("Checkpoint-Logik: Runde N → letzter
Checkpoint von Runde N-1", `VIDEO_PIPELINE_KONTEXT.md` Zeile 529, dort zwar für die spätere
`lofi_pipeline`-Phase formuliert, aber identisches Muster) ergibt sich: iteratives Weitertrainieren statt
Neustart pro Runde, um vorherige Lernfortschritte nicht zu verwerfen.

**Änderung 2 — `validation.prompts`:** von
> "lofi_girl, anime girl studying late at night, soft lamp glow, pencil on paper, rain sounds, peaceful, 16:9"

zu (deutlich länger, mit expliziten Stilbegriffen):
> "lofi_girl, 2D anime illustration, soft cel shading, anime girl sitting at wooden desk studying late at night, warm glowing desk lamp, pencil in hand writing in notebook, detailed bookshelves background, rain on window, cozy warm atmosphere, smooth subtle loop animation, 16:9"

*Beobachtung:* Der Prompt wird von einer kurzen Beschreibung zu einer detaillierten Stil-/Kompositionsvorgabe
erweitert (explizit "2D anime illustration", "cel shading" als Stilanker).
*Begründung laut `VIDEO_PIPELINE_KONTEXT.md`, Zeile 78–79 (Bewertungstabelle):* training_01 (=Round1) wurde
mit Q:2, A:2 bewertet: *"Kaltstart, stark photorealistisch, kein Anime-Stil"*. training_02 (=Round2) erhielt
Q:3, A:2: *"Anime-Tendenzen leicht erkennbar"*. Die Prompt-Erweiterung ist die direkte Reaktion auf das in
Round1 diagnostizierte Problem (fehlender Anime-Stil, zu photorealistisch) — durch explizitere Stilbegriffe
im Prompt sollte das Modell stärker Richtung Anime/Cel-Shading gelenkt werden.

**Nicht geändert:** Alle Trainingsparameter (Rank, Alpha, LR, Steps, Optimizer) bleiben identisch — die
Reaktion auf das Round1-Ergebnis erfolgte ausschließlich über den Prompt, nicht über die
Trainingskonfiguration selbst.

---

### 4.2 Round2 → Round3 (21.06. 23:05 → 23:31, ca. 26min später)

**Änderung:** Nur `model.load_checkpoint` (verweist jetzt auf Round2s Step-50-Checkpoint) und `output_dir`.
**Keine Parameter- oder Prompt-Änderung.**
*Beobachtung:* Round3 ist eine reine Fortsetzung von Round2 mit identischer Konfiguration — kein
Eingriff, keine Reaktion auf ein spezifisches Problem in der Config selbst erkennbar.
*Begründung laut `VIDEO_PIPELINE_KONTEXT.md` Zeile 80:* training_03 (=Round3) erhielt Q:3, A:3:
*"Cel-Shading erstmals sichtbar → Inference-Steps erhöhen"*. Die eigentliche Konsequenz aus diesem
Befund (Erhöhung der Inference-Steps) wird erst in der **nächsten** Runde (Round4) umgesetzt, nicht in
Round3 selbst — Round3 war der Lauf, dessen Ergebnis diese spätere Änderung motivierte.

---

### 4.3 Round3 → Round4 (21.06. 23:31 → 22.06. 03:55, ca. 4h 24min später — größte zeitliche Lücke bis dahin)

Dies ist die Runde mit den **meisten gleichzeitigen Änderungen** bis zu diesem Punkt:

**Änderung 1 — `validation.inference_steps`:** 30 → **50**.
*Begründung, wörtlich aus `VIDEO_PIPELINE_KONTEXT.md` Zeile 81 und 90:*
> "training_04 … **Großer Sprung**: klare Anime-Ästhetik durchgehend. Änderung: Inference 30→50"
> "Inference-Steps 30→50 (training_04): Bessere Qualitätseinschätzung während Validation + bessere
> Outputs. Bleibt so."
Dies ist die direkte Umsetzung der in Round3 erkannten Notwendigkeit (siehe 4.2).

**Änderung 2 — `validation.guidance_scale`:** 3.5 → **4.5**.
*Begründung:* Nicht separat von der Inference-Steps-Änderung im Kontext-Dokument diskutiert; beide
Änderungen fallen in dieselbe Runde und werden dort als gemeinsamer "großer Sprung" beschrieben. Eine
getrennte Begründung für die guidance_scale-Erhöhung allein ist NICHT REKONSTRUIERBAR.

**Änderung 3 — `validation.negative_prompt`:** von
> "worst quality, inconsistent motion, blurry, jittery, distorted"

zu (um sechs Begriffe erweitert)
> "…, grainy, noisy, low detail, washed out, photorealistic, 3D render, unclear face, missing features"

*Beobachtung:* Erstmalige Verwendung eines Negativ-Prompts in dieser Ausführlichkeit; insbesondere
"photorealistic" und "3D render" werden neu als Negativbegriffe ergänzt — passend zum in Round1/2
diagnostizierten Problem "stark photorealistisch, kein Anime-Stil" (4.1). Das Modell soll aktiv von
Fotorealismus weggelenkt werden.
*Begründung laut Kontext-Dokument:* Nicht als eigener Punkt benannt, aber inhaltlich konsistent mit der
in 4.1 zitierten Diagnose.

**Änderung 4 — `validation.prompts`:** deutlich detaillierterer Prompt mit expliziten Anweisungen zu
Gesichtsdetails ("detailed face with soft features nose mouth visible"), Handhaltung ("visible hand
holding pencil clearly") und Farbpalette ("warm amber color palette").
*Begründung:* Konsistent mit dem iterativen Verfeinerungsmuster — jede Runde reagiert im Prompt auf
zuvor beobachtete Schwächen (hier vermutlich: unklare Gesichts-/Handdarstellung), ohne dass diese
Einzelbegründung im Kontext-Dokument wörtlich für Round4 spezifisch benannt wird. **Teilweise
rekonstruierbar** (Muster ja, Einzelbegründung für jedes neue Prompt-Fragment nein.)

---

### 4.4 Round4 → Round5 (22.06. 03:55 → 04:36, ca. 41min später)

**Änderung 1 — `validation.negative_prompt`:** massive Erweiterung von 9 auf **59 Begriffe** (u.a. neu:
"halftone texture", "bad hands", "distorted fingers", "unstable animation", "warped body",
"oversaturated orange").

**Änderung 2 — `validation.prompts`:** deutlich ausführlicherer Prompt mit expliziten
Stabilitäts-Anweisungen ("controlled subtle head movement", "natural writing motion", "stable animation",
"visually clean and consistent across frames").

*Beobachtung:* Beide Änderungen zusammen zeigen eine massive Verschärfung der Prompt-Steuerung nach nur
einer Runde — deutlich mehr als der graduelle Zuwachs zwischen den vorherigen Runden.
*Begründung laut `VIDEO_PIPELINE_KONTEXT.md` Zeile 82:*
> "training_05 … **PEAK**: beste Komposition, Stimmung optimal. Dieser Checkpoint = Referenz"
Round5 ist laut Kontext-Dokument der **beste Checkpoint der gesamten Round1–11-Serie** ("Referenz",
später als `lofi_lora_best.safetensors` für alle folgenden Szenarien der `lofi_pipeline`-Phase
weiterverwendet, siehe Teil B). Die Prompt-Verschärfung in dieser Runde korreliert mit dem besten
Ergebnis — ob die Prompt-Änderung *ursächlich* für die Qualitätsverbesserung war oder andere Faktoren
(z. B. einfach der inkrementelle Trainingsfortschritt) verantwortlich waren, ist aus den Daten **nicht
kausal beweisbar**, nur zeitlich korreliert.

---

### 4.5 Round5 → Round6 (22.06. 04:36 → 05:13, ca. 37min später)

**Änderung:** `validation.negative_prompt` und `validation.prompts` erneut umformuliert (u. a. Wegfall
einiger sehr spezifischer Negativbegriffe wie "warped body", Hinzufügen von "cheap cartoon look",
"oversimplified anime style").
**Keine Parameter-Änderung** (Rank/LR/Steps unverändert, wie in allen 11 Runden).

*Begründung laut `VIDEO_PIPELINE_KONTEXT.md` Zeile 83:*
> "training_06 … Szenendrift, Overfitting. Zu viele Steps nach Peak. → Zurück zu R5-Checkpoint"

**Wichtiger Befund:** Round6 wird laut Kontext-Dokument als **Rückschritt gegenüber Round5** bewertet
(Q:4, A:4 vs. Round5s Q:7, A:6) — als Overfitting/Szenendrift eingeordnet. Die Konsequenz war laut Text
ein Rücksprung zum Round5-Checkpoint als Basis für die weitere Verwendung (nicht für Round7 selbst — die
Config-Kette in 4.6 zeigt, dass Round7 technisch von Round5s Checkpoint lädt, nicht von Round6s, siehe
dort). Round6 selbst blieb aber als eigener, dokumentierter Ordner mit eigenem Checkpoint erhalten
(archiviert, nicht gelöscht — konsistent mit dem im Hauptrepo etablierten Archivierungsprinzip).

---

### 4.6 Round6 → Round7 (22.06. 05:13 → 05:51, ca. 38min später)

**Wichtiger struktureller Befund:** Anders als bei allen anderen Runden ändert sich `model.load_checkpoint`
**nicht** zwischen Round6 und Round7 in der erwarteten Kette. Der maschinelle Diff zeigt für Round7 keinen
Eintrag unter `model.load_checkpoint` in der Liste der gegenüber Round6 geänderten Felder — das bedeutet,
Round7 lädt denselben Checkpoint-Pfad wie in Round6s Konfiguration eingetragen war
(`ltx_lora_round5/checkpoints/lora_weights_step_00050.safetensors`, **nicht** Round6s eigenen
Endcheckpoint). Das bestätigt exakt die in 4.5 zitierte Aussage *"Zurück zu R5-Checkpoint"* — Round7 baut
technisch nachweisbar auf Round5, nicht auf Round6 auf, obwohl Round6 zeitlich dazwischenliegt.

**Änderung 1 — `validation.images`:** `None` → `['.../daten/processed/ltx_lora_manual/reference_frame_r5.jpg']`.
*Beobachtung:* Erstmalige Verwendung von Image-Conditioning (ein Referenzbild wird der Validation
mitgegeben). Der Dateiname `reference_frame_r5.jpg` bestätigt den Bezug zu Round5 als Referenz-Frame-Quelle.
*Begründung laut `VIDEO_PIPELINE_KONTEXT.md` Zeile 84:*
> "training_07 … Steps 50→100, aber Image-Conditioning-Bug: Letterboxing (schwarze Balken)"
Ziel war offenbar, mittels Bildreferenz die Szene/Komposition von Round5 stabiler zu reproduzieren
("continue from Round 5 scene" im neuen Prompt, siehe unten) — das Ergebnis war jedoch ein **Bug**
(Letterboxing).

**Änderung 2 — `validation.prompts`:** neuer Prompt beginnt explizit mit "continue from Round 5 scene,
same cozy lofi anime study room … same camera framing, same character placement".
*Begründung:* Direkte Konsequenz aus dem Rücksprung zu Round5 als Referenzbasis (4.5) — der Prompt soll
das Modell anweisen, bei der Round5-Komposition zu bleiben statt weiter zu driften.

**Anmerkung zu `optimization.steps`:** Laut `VIDEO_PIPELINE_KONTEXT.md` sollte in training_07 bereits
"Steps 50→100" gelten. Der maschinelle Diff dieser Sitzung zeigt jedoch **keine Änderung von
`optimization.steps` zwischen Round6 und Round7** — der Wert bleibt in der `training_config.yaml` von
Round7 bei 50; die Änderung auf 100 erfolgt laut Diff erst **eine Runde später, in Round8** (siehe 4.7).
**Dies ist eine Diskrepanz zwischen dem Kontext-Dokument (das die Steps-Änderung Round7 zuschreibt) und
dem tatsächlichen Dateiinhalt (der die Änderung erst in Round8 zeigt).** Beide Quellen werden hier
transparent nebeneinandergestellt, ohne die eine zugunsten der anderen stillschweigend zu "korrigieren".

**Ergebnis dieser Runde laut Kontext-Dokument:** Q:4, A:3 — schlechter als Round5, mit explizit benanntem
Bug (Letterboxing durch Image-Conditioning).

---

### 4.7 Round7 → Round8 (22.06. 05:51 → 01.07. 07:15 — **größte zeitliche Lücke der gesamten Serie: ca. 9 Tage**)

**Änderung 1 — `optimization.steps`:** 50 → **100**.
*Begründung:* Siehe Diskrepanz-Hinweis in 4.6 — im Kontext-Dokument Round7 zugeschrieben, im tatsächlichen
Dateiinhalt erst hier in Round8 wirksam.

**Änderung 2 — `validation.images`:** Referenzbild-Pfad → `None` (Image-Conditioning wieder deaktiviert).
*Begründung, wörtlich aus `VIDEO_PIPELINE_KONTEXT.md` Zeile 92:*
> "`images: null` (training_08): `images: reference_frame` erzeugte einen Zoom/Crop-Effekt (Letterboxing).
> Dauerhaft deaktiviert."
Direkte Reaktion auf den in Round7 diagnostizierten Bug (4.6).

**Änderung 3 — `validation.interval`:** 25 → **50**.
*Begründung:* Nicht explizit einzeln benannt; könnte mit der Verdopplung von `optimization.steps` (50→100)
zusammenhängen (gleiche relative Validierungshäufigkeit bei doppelter Gesamtschrittzahl) — das ist jedoch
eine **Interpretation**, keine im Kontext-Dokument belegte Aussage. Als solche gekennzeichnet.

**Zusätzliche, außerhalb der `training_config.yaml` liegende Änderung laut Kontext-Dokument** (nicht
Teil des maschinellen YAML-Diffs, da den Trainingsdatensatz und nicht die Config betrifft):
> "Fix: `images: null`. 15 neue Clips im Trainingsset. Flüssigere Animation" (Zeile 85)
Diese Aussage über "15 neue Clips" konnte in dieser Sitzung nicht direkt anhand von Dateisystem-Belegen
nachvollzogen werden (kein Diff der `dataset.jsonl`-Zeilenzahl zwischen Round7 und Round8 durchgeführt,
da für frühere Versionen dieser Datei kein Snapshot vorliegt — die Datei existiert nur im aktuellen
Zustand). NICHT UNABHÄNGIG VERIFIZIERT, nur aus dem Kontext-Dokument übernommen.

**Die 9-Tage-Lücke selbst:** Aus keiner der vorliegenden Quellen geht hervor, was zwischen dem 22. Juni
(Round7) und dem 1. Juli (Round8) geschah. Möglich wären Pausen, andere Arbeit an der Bachelorarbeit
(z. B. am Audio-Teil, oder an der Datenaufbereitung für die "15 neuen Clips"), oder Auswertungszeit. NICHT
REKONSTRUIERBAR.

---

### 4.8 Round8 → Round9 (01.07. 07:15 → 03.07. 02:50 — zweitgrößte Lücke: ca. 1 Tag 19h 35min)

**Änderung 1 — `validation.inference_steps`:** 50 → **30** (Rückgang, nicht weitere Erhöhung).

**Änderung 2 — `validation.interval`:** 50 → **9999** (= faktisch deaktiviert, da > `optimization.steps`).

**Änderung 3 — `validation.negative_prompt`:** langer Text → **leerer String** `''`.

**Änderung 4 — `validation.prompts`:** Liste mit einem Prompt → **leere Liste** `[]`.

*Begründung, wörtlich aus `VIDEO_PIPELINE_KONTEXT.md` Zeile 86 und 91:*
> "training_09 … OOM-Crashes → Validation dauerhaft auf 9999 gestellt. Video separat generiert"
> "Validation-Interval 25→9999 (training_09): 2× OOM-Crash durch Inline-Validation beim 13B. Seitdem:
> Training ohne Validation, danach `generate_samples.py` separat."

Dies ist die am eindeutigsten dokumentierte Änderung der gesamten Serie: **Speicherprobleme (CUDA-Out-
of-Memory) beim gleichzeitigen Training + Inline-Validierung des 13B-Modells** zwangen dazu, die
Validierung während des Trainings komplett abzuschalten und stattdessen nach Abschluss des Trainings
separat Videos zu generieren. Die geleerten `prompts`/`negative_prompt`-Felder und der reduzierte
`inference_steps`-Wert sind Ausdruck derselben Entscheidung (Validation wird ohnehin nicht mehr inline
ausgeführt, die Felder sind funktionslos, aber technisch nicht entfernt).

---

### 4.9 Round9 → Round10 (03.07. 02:50 → 04:16, ca. 1h 26min später)

**Änderung:** Nur `model.load_checkpoint` (Kette zu Round9s Step-100-Checkpoint) und `output_dir`.
**Keine Parameter- oder Prompt-Änderung** — identisch zum Muster von Round2→Round3 (4.2): reine
Fortsetzung ohne Konfigurationseingriff.
*Begründung:* Nicht im Kontext-Dokument einzeln benannt. Da Validation seit Round9 deaktiviert ist,
liegt keine inline erzeugte Qualitätsrückmeldung vor, die eine Reaktion motivieren könnte — passt zur
Beobachtung "keine Änderung".

---

### 4.10 Round10 → Round11 (03.07. 04:16 → 04:44, ca. 28min später — letzte Runde dieser Serie)

**Änderung — `checkpoints.interval`:** 50 → **25**.
*Beobachtung:* Bestätigt durch die tatsächlichen Checkpoint-Dateien (Abschnitt 8.1 der
`ZWEITER_TRAININGSANSATZ_DOKUMENTATION.md`): Round11 besitzt als einzige der elf Runden **vier**
Zwischencheckpoints (Step 25/50/75/100) statt nur der finalen zwei (50/100 bzw. nur 50).
*Begründung:* Nicht explizit für Round11 im Kontext-Dokument benannt, aber konsistent mit dem später in
der `lofi_pipeline`-Phase für `rabbit_lake` dokumentierten Muster: *"Checkpoint-Intervall-Änderung: R4+:
alle 10 Steps — mehr Granularität, besser für nachträgliche Auswahl des besten Checkpoints"*
(`VIDEO_PIPELINE_KONTEXT.md` Zeile 132–134). Round11 ist die letzte Runde dieser Serie; ein feineres
Intervall für eine womöglich abschließende, genauer zu prüfende Runde ist plausibel, aber für Round11
selbst **nicht wörtlich belegt** — als Analogieschluss gekennzeichnet, nicht als Fakt.

---

## Teil B: Änderungen nach dem Strukturwechsel zur `lofi_pipeline` (ab 03.07.2026)

Ab dem 3. Juli (unmittelbar nach Round11, siehe `ZWEITER_TRAININGSANSATZ_DOKUMENTATION.md` Abschnitt 6)
wird die flache `training/video/ltx_lora_*`-Struktur durch `lofi_pipeline/scenarios/{id}/rounds/round_NN/`
abgelöst. Die folgenden Änderungen betreffen nicht mehr einzelne `training_config.yaml`-Diffs (dafür läge
in dieser Sitzung kein vollständiger Parallel-Datensatz vor wie in Teil A), sondern sind wörtlich/sinngemäß
aus `VIDEO_PIPELINE_KONTEXT.md` übernommen, wo sie als expliziter "Warum"-Verlauf dokumentiert sind.

### 4.11 Konfigurationsvarianten: `base_lora.yaml` vs. `base_lora_v2.yaml`

| Parameter | base_lora.yaml (Standard) | base_lora_v2.yaml | Begründung laut Kontext-Dokument |
|---|---|---|---|
| rank / alpha | 16 / 16 | **32 / 32** | "mehr Kapazität für Details + Bewegung" |
| target_modules | to_k, to_q, to_v, to_out.0 | **+ ff.net.0.proj, ff.net.2** | "FF-Schichten für Stil + Textur" |
| learning_rate | 1e-4 | **5e-5** | "feiner, weniger Overfitting bei Rank 32" |
| steps (Standard) | 100 | **200** | "längeres Training" |
| checkpoint interval | 10 | **5** | "mehr Punkte für metric_watcher" |
| wandb.project | ltxv-lofi | ltxv-lofi-v2 | "eigene W&B-Gruppe" |

Quelle: `VIDEO_PIPELINE_KONTEXT.md`, Tabelle "base_lora_v2.yaml (rabbit_lake_v2) — Änderungen", Zeile 303–311.
**Wichtig zur Einordnung:** `base_lora.yaml` selbst (Standard-Konfiguration der `lofi_pipeline`-Phase) hat
bereits **LR 1e-4** und **Steps 100**, nicht mehr LR 2e-4/Steps 50/100 wie in Teil A. Das heißt, der Übergang
von Teil A zu Teil B brachte bereits eine LR-Halbierung mit sich, bevor `base_lora_v2.yaml` die Werte für
`rabbit_lake_v2` nochmals auf 5e-5 senkte. Der genaue Zeitpunkt/Grund für die erste Halbierung (2e-4→1e-4)
zwischen Ende Teil A und Beginn `base_lora.yaml` ist im Kontext-Dokument nicht als expliziter Einzelschritt
benannt — dort wird die LR-Änderung nur für **rabbit_lake ab dessen Runde 4** narrativ festgehalten
(nächster Abschnitt), was nahelegt, dass `base_lora.yaml`s 1e-4 bereits das Ergebnis dieser
rabbit_lake-Erfahrung ist und rückwirkend als neuer Standard übernommen wurde. Diese Reihenfolge ist
**plausibel, aber nicht mit Zeitstempeln einzeln belegt** — als Interpretation gekennzeichnet.

### 4.12 Szenario `rabbit_lake` — Lernraten-Änderung während des laufenden Szenarios

| Runde | Lernrate | Begründung (wörtlich, `VIDEO_PIPELINE_KONTEXT.md` Zeile 124–130) |
|---|---|---|
| R1–R3 | 0.0002 (2e-4) | "Standard aus training_runs" |
| **R4+** | **0.0001 (1e-4)**, halbiert | "Grund: bei R3 erste Anzeichen von Overfitting (Szene instabiler). Niedrigere LR = feinere Anpassung, weniger Drift." |
| R6 | Steps 100→50 | "(experimentell, zu kurz)" |
| R7 | Steps wieder 100 | (Rückgängigmachung von R6) |
| R8 | Steps 150 | "(höchste Qualität, Q:7 A:7 in manueller Bewertung)" |
| R9 | — | "CUDA-OOM-Absturz → Training-Run unterbrochen" |

### 4.13 Szenario `rabbit_lake` — Checkpoint-Intervall

| Runde | Intervall | Begründung |
|---|---|---|
| R1–R3 | alle 25 Steps | "aus training_runs übernommen" |
| **R4+** | **alle 10 Steps** | "mehr Granularität, besser für nachträgliche Auswahl des besten Checkpoints" |

Quelle: `VIDEO_PIPELINE_KONTEXT.md` Zeile 132–134.

### 4.14 Post-Processing-Werkzeuge: warum RIFE statt minterpolate, warum ESRGAN x4plus_anime_6B

**RIFE (20. Juli) statt reinem `minterpolate`:**
> "Warum RIFE statt minterpolate: minterpolate (ffmpeg CPU) erzeugt bei Anime-Stil Ghosting-Artefakte.
> RIFE nutzt optischen Fluss auf GPU → deutlich flüssigere Übergänge."
Quelle: `VIDEO_PIPELINE_KONTEXT.md` Zeile 114. `enhance_video.py` behält `minterpolate` dennoch als
CPU-Fallback, falls die RIFE-Binary fehlt (Zeile 459 der Kontext-Datei, Tabelle
"enhance_video.py — Methoden im Detail").

**Real-ESRGAN `x4plus_anime_6B` (21. Juli), nicht die volle 23-Block-Variante:**
> "Warum x4plus_anime_6B: Speziell für Anime-Stil trainiert, 6-Block-Variante (leichter als volle
> 23-Block-Variante), funktioniert gut auf 8fps Lo-Fi Videos."
Quelle: `VIDEO_PIPELINE_KONTEXT.md` Zeile 119.

**`make_gif.py` (4. August) ersetzt den einfachen ffmpeg-GIF-Befehl aus `generate_samples.py`:**
> "Warum: minterpolate allein reicht nicht. Vollständige Kette: minterpolate → ESRGAN 4× → GIF
> (palettegen/paletteuse) + MP4 (H.264, bt709, CRF 16)"
Quelle: `VIDEO_PIPELINE_KONTEXT.md` Zeile 147.

### 4.15 Bewertungsmethodik: warum `evaluate_video.py` (6. August)

> "Warum: Manuelle Bewertung (1–10 Scores) ist subjektiv und reproduzierbar [sic, vermutlich
> 'nicht reproduzierbar' gemeint]. evaluate_video.py liefert objektive, vergleichbare Zahlen pro
> Checkpoint — SSIM, Sharpness, Motion, Flicker, Color/Brightness Consistency."
Quelle: `VIDEO_PIPELINE_KONTEXT.md` Zeile 151 (Tippfehler im Original wörtlich übernommen und
kenntlich gemacht, nicht stillschweigend korrigiert).

### 4.16 `metric_watcher.py` (6. August) — warum ein separater Watcher-Prozess

> "Warum: Training ist ein Subprocess — kein Python-Level-Callback möglich. metric_watcher.py läuft
> parallel, triggert bei neuem Checkpoint automatisch generate + evaluate → schreibt metrics_log.json
> → kann Training via STOP_TRAINING-Datei stoppen."
Quelle: `VIDEO_PIPELINE_KONTEXT.md` Zeile 156. Dies ist der erste im gesamten Verlauf dokumentierte
**automatische** Abbruchmechanismus — in Teil A (Round1–11) gab es keinen; jede Runde wurde manuell
gestartet und manuell nach Sichtprüfung beendet bzw. die nächste Runde manuell aufgesetzt.

### 4.17 Feedback-UI (28. Juli) — warum Wechsel von Gradio zu Flask

> "Warum: Schnelleres visuelles Feedback ohne Terminal — MP4s direkt im Browser bewerten, Noten vergeben,
> nächste Runde antriggern. Gradio-UI war zu unflexibel."
Quelle: `VIDEO_PIPELINE_KONTEXT.md` Zeile 139. Bemerkenswert: `gradio` selbst wurde bereits im Mai (7. Mai)
installiert (Teil des ursprünglichen Setups, siehe `ZWEITER_TRAININGSANSATZ_DOKUMENTATION.md` Abschnitt 3.2)
— erst am 28. Juli, rund 12 Wochen später, wird es durch Flask ersetzt, weil es sich für den tatsächlichen
Feedback-Workflow als zu unflexibel erwies.

### 4.18 Script-Zentralisierung (10. August)

> "Warum: jedes Szenario hatte eigene Script-Kopien → Wartung aufwändig. Zentrale Scripts lesen
> scenario.yaml und sind für alle Szenarien wiederverwendbar."
Quelle: `VIDEO_PIPELINE_KONTEXT.md` Zeile 164. Betrifft `train_lora.py`, `generate_samples.py`,
`preprocess_scenario.py` (Zeitstempel siehe Abschnitt 4 der `ZWEITER_TRAININGSANSATZ_DOKUMENTATION.md`).

---

## 5. Zusammenfassende Einordnung: Muster über den gesamten Verlauf

1. **Trainingsparameter (Rank/Alpha/LR/Optimizer) blieben in Teil A (11 Runden) komplett konstant** — jede
   Reaktion auf ein beobachtetes Problem erfolgte ausschließlich über Prompt/Negativ-Prompt oder
   Validierungs-Einstellungen, nie über die eigentlichen LoRA-Trainingsparameter. Erst in Teil B
   (`lofi_pipeline`, ab `rabbit_lake`) werden LR (2e-4→1e-4→5e-5) und Rank (16→32) tatsächlich verändert.

2. **Die einzige technisch hart erzwungene Änderung** (nicht optional/experimentell, sondern durch einen
   Absturz erzwungen) ist die Deaktivierung der Inline-Validierung ab Round9 wegen CUDA-OOM (4.8) — analog
   dazu später der CUDA-OOM-Abbruch bei `rabbit_lake` R9 (4.12). Beide Male war die 13B-Modellgröße im
   Zusammenspiel mit gleichzeitiger Validierung die Ursache.

3. **Prompt-Engineering war der mit Abstand am häufigsten genutzte Stellhebel** in Teil A: In 7 von 10
   Runden-Übergängen (alle außer Round2→3, Round9→10, Round10→11) wurde `validation.prompts` und/oder
   `validation.negative_prompt` verändert, gegenüber nur 2 Übergängen mit einer echten
   Trainings-/Validierungsparameter-Änderung (inference_steps/guidance_scale in Round4; steps/interval in
   Round8/9).

4. **Automatisierte Qualitätskontrolle existierte in der gesamten Teil-A-Phase (Juni/Juli) nicht.** Sie
   wurde erst ab 6. August (`evaluate_video.py`, `metric_watcher.py`) eingeführt — mehr als fünf Wochen
   nach Abschluss von Round11.

5. **Zwei Diskrepanzen zwischen dem Kontext-Dokument und dem tatsächlichen Dateiinhalt wurden in dieser
   Sitzung aufgedeckt** und bewusst nicht stillschweigend geglättet: die Steps-50→100-Änderung wird im
   Kontext-Dokument Round7 zugeschrieben, tritt im Dateidiff aber erst in Round8 auf (4.6/4.7); und die
   genaue Kausalkette der ersten LR-Halbierung (2e-4→1e-4) zwischen Teil A und `base_lora.yaml` ist nicht
   mit Zeitstempeln belegt, sondern nur plausibel erschlossen (4.11).

---

## 6. Offene Punkte dieser Chronik

1. Was zwischen Round7 (22.06.) und Round8 (01.07.) geschah (9-Tage-Lücke) — NICHT REKONSTRUIERBAR.
2. Ob die "15 neuen Clips" (laut Kontext-Dok. in Round8 ergänzt) tatsächlich zu einer nachweisbaren
   Änderung der Trainingsdaten-Menge führten — in dieser Sitzung nicht durch einen Datensatz-Snapshot-Diff
   verifiziert, nur aus dem Kontext-Dokument übernommen.
3. Die exakte Einzelbegründung für die `guidance_scale`-Erhöhung 3.5→4.5 in Round4 (separat von der
   `inference_steps`-Erhöhung) — NICHT REKONSTRUIERBAR, da im Kontext-Dokument nur gemeinsam mit den
   Inference-Steps als ein "großer Sprung" beschrieben.
4. Der exakte Zeitpunkt/Auslöser der ersten LR-Halbierung zwischen Teil A (2e-4) und `base_lora.yaml`
   (1e-4), vor der zusätzlichen Halbierung für `rabbit_lake` R4 — teilweise rekonstruierbar als
   Analogieschluss, nicht als belegter Einzelschritt (4.11).
