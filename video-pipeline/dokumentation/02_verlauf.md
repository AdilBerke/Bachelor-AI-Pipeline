# Verlauf: Von den Anfängen bis heute (Stand 04.08.2026)

Diese Chronik ist aus Dateizeitstempeln, Rohlogs, `training_runs/overview.csv` und den
JSON-Metadaten (`notes.json`/`feedback.json`) der einzelnen Trainingsrunden rekonstruiert.
Zeitangaben sind Dateisystem-Zeitstempel bzw. in den Metadaten protokollierte Zeiten.
Wo Angaben nicht aus dem Repository belegbar sind, ist das ausdrücklich vermerkt.

## Phase 0 — Allererster Versuch: automatisierte Trainingsdaten-Pipeline "video_pipeline" (06.–10.05.2026)

Der tatsächlich früheste Versuch liegt **außerhalb** des Git-Repositoriums, unter
`~/Downloads/video_pipeline/` (drei Zip-Sicherungen erhalten und ausgewertet: 06.05.2026
05:03 Uhr, 06.05.2026 23:35 Uhr, 10.05.2026 08:01 Uhr — jeweils gewachsen von 40 KB auf
104 KB).

**Architektur** (vollständig automatisiert startbar über `python main.py run-all`):
```
YouTube CC-Videos → [1] Suche/Filter (YouTube Data API v3, nur CC-Lizenz)
  → [2] Download (yt-dlp) / lokaler Import
  → [3] Automatischer Schnitt (ffmpeg, Schwarzbild-/Szenenerkennung, 15–20s-Clips)
  → [4] Frame-Extraktion (OpenCV/ffmpeg, 1–4 FPS)
  → [5] Analyse & Tagging (Keyword-Fallback oder CLIP/BLIP)
  → [6] Dataset-Erstellung (metadata.jsonl, HuggingFace-kompatibel)
  → [7] GIF-Erstellung (ffmpeg palettegen + paletteuse, gifsicle/Pillow-Optimierung)
```
Rechtliche Absicherung war von Anfang an eingebaut (Filter auf `videoLicense=creativeCommon`,
empfohlene lizenzfreie Quellen: Pexels, Pixabay, Internet Archive).

**Motivkatalog:** `data/metadata/category_presets.yaml` (57 KB, Stand 10.05.) definiert
**101 vorbereitete Lo-Fi-Kategorien**, u. a. `cozy_lofi_girl`, `lofi_girl_cafe_window`,
`rainy_window_cafe`, `cozy_winter_snow_run` — sichtbare Vorläufer der späteren
`lofi_pipeline`-Szenarien. Die allererste Version (06.05., 05:03 Uhr) suchte allerdings
noch nach einem **laufenden Charakter** ("running animation", "lofi running",
"anime running", "night city running"), nicht nach Fenster-/See-Szenarien.

**Automatisierungsgrad — was "vollautomatisch" hier konkret bedeutet:** Die Datenaufbereitung
(Schritte 1–6 oben) ist tatsächlich als ein einziger unbeaufsichtigter Befehl umgesetzt:
```
python main.py run-video-pipeline
```
(`main.py`, gebaut mit dem CLI-Framework `typer`) — kein manueller Zwischenschritt beim
Suchen, Herunterladen, Schneiden, Taggen oder GIF-Erzeugen. Das eigentliche **Modelltraining
war jedoch von Anfang an bewusst ausgelagert**, nicht Teil dieser Automatik. Wörtlich in
`training/README.md`, Phase 5 ("Training starten"):

> "Das eigentliche Training hängt vom gewählten Modell und Framework ab. […] Mit dem
> LTX-Video-Trainer (**separates Repository**): `python train_ltx_lora.py --config
> training/my_run_001.yaml --dataset dataset/metadata.jsonl`"

Diese Datei `train_ltx_lora.py` existiert in `video_pipeline` nirgends — die Suche danach
ergab keinen Treffer. Das hier bereits im Mai referenzierte "separate Repository" ist mit
hoher Wahrscheinlichkeit der Vorläufer-Gedanke zu dem, was ab Juni tatsächlich als
`tools/LTX-Video-Trainer/` (externes Lightricks-Framework) eingebunden wurde — siehe Phase 2.
**Wichtig für die Entwicklungslinie:** Der Übergang von "automatisch" (Mai, `video_pipeline`)
zu "manuell rundenbasiert" (`v003_manual` ab 13.06., siehe Phase 2) ist damit **keine
Abkehr von einem automatisierten Trainingsprozess**, sondern die erstmalige Umsetzung eines
Trainingsschritts, der in der Ursprungsplanung von Mai ohnehin nie automatisiert war —
nur die Datenbeschaffung davor war es.

**Was mit den erzeugten Videos passiert ist:** In allen drei erhaltenen Zip-Sicherungen
sind `data/raw/`, `data/clips/`, `data/frames/`, `outputs/gifs/` und `logs/` leer (nur
`.gitkeep`-Platzhalter, `current_clip_count: 0` bei allen 101 Kategorien). Das erklärt
sich durch die `.gitignore` der Pipeline, die genau diese Ordner ausdrücklich ausschließt
("Pipeline-Daten (können sehr groß werden)"). Nutzerangabe: Es gab damals tatsächlich
erzeugte Videos, die wahrscheinlich gelöscht wurden — das ist mit dem Befund vereinbar,
da gitignorte Ordner in keiner Zip-Sicherung erfasst werden, unabhängig davon, ob zum
Sicherungszeitpunkt Dateien darin lagen. Eine systemweite Suche nach übrig gebliebenen
Video-/GIF-Dateien aus diesem Zeitraum (Dateisystem, Papierkorb, verwaiste Git-Objekte)
ergab **keinen Treffer** — falls Videos existierten, sind sie nicht mehr wiederherstellbar.
Konkret gesucht und nicht gefunden: eine Datei namens `gif_lofi_cozy_cafe_001_ltx.gif`.

**Funktionsnachweis (04.08.2026):** Um zu klären, ob die Pipeline grundsätzlich
funktionierte oder von Anfang an fehlerhaft war, wurde `gif_maker.py` isoliert mit einem
synthetischen, rein lokal erzeugten Testclip ausgeführt (kein Download). Ergebnis: ein
valides, korrekt endlos loopendes GIF (640×360, GIF89a, 381 KB) über den originalen
Zwei-Schritt-ffmpeg-Weg (`palettegen` → `paletteuse`). **Die Pipeline-Logik war und ist
technisch funktionsfähig** — es fehlt nachweislich nur der tatsächlich vollständig
durchgeführte und dauerhaft gesicherte Datenbeschaffungsschritt, nicht die Umsetzung
selbst.

**Einordnung:** Diese erste Pipeline ist kein generatives KI-Modell, sondern ein
Trainingsdaten-Beschaffungs- und Clip-zu-GIF-Konvertierungswerkzeug (echte heruntergeladene
Clips → GIF per ffmpeg, keine KI-generierten Bildinhalte). Erst ab Phase 2 (`v003_manual`,
13.06.2026) kommt ein tatsächliches generatives Videomodell (LTX-Video) zum Einsatz.

## Phase 1 — Frühe LTX-Versionen v002/v003 (nicht mehr im Repository vorhanden)

Laut früheren Projektnotizen gab es zwischen Phase 0 und der ab hier dokumentierten
Phase mindestens zwei weitere Anläufe:

- **v002**: LTX-Video-LoRA, 512×288, 3 Sekunden, 29 Clips, 1000 Steps — als "zu klein
  für Thesis-Qualität" eingestuft.
- **v003 (automatisiert)**: geplant als Overnight-Pipeline (`run_overnight.py`) mit den
  Schritten Download → Prepare → Dataset → Preprocess → Train, Ziel 832×480, 8-Sekunden-Clips,
  ~80+ Videos, Modell `LTXV_2B_0.9.6_DEV`, 2000 Steps, LoRA Rang 16.

Weder Code noch Ergebnisse dieser Versionen sind im aktuellen Repository auffindbar. Der
Ansatz wurde offenbar zugunsten eines manuellen, kleinschrittigen Verfahrens aufgegeben,
bevor er nachweislich abgeschlossen wurde.

## Phase 2 — "v003_manual": Umstieg auf LTX-Video 13B, manuelles Training (13.06.–21.06.2026)

- **13.06.2026**: Preprocessing (VAE-Latents + T5-Embeddings) für ein neues, manuell
  kuratiertes Dataset (`daten/processed/ltx_lora_manual/`); erste Trainingsversuche
  (`logs/v003_manual_train.log`, `v003_manual_train_v2.log`).
- **21.06.2026, 17:42 Uhr**: Erster vollständiger Trainingslauf mit dem größeren
  **LTX-Video 13B 0.9.7-dev**-Modell (`training_01` / Ordner `ltx_lora_manual`, Log
  `v003_manual_13b_r1.log`). 50 Steps, kein Vorgänger-Checkpoint (Kaltstart).
  Selbsteinschätzung Qualität: 2/10. Motiv: "lofi_girl" — Anime-Figur am Schreibtisch,
  schreibend, warmes Lampenlicht, Bücherregal.
- Generierung erfolgte über `pipeline/v003_manual/generate.py`, das die
  `LTXConditionPipeline` aus `tools/LTX-Video-Trainer` direkt aufruft.

## Phase 3 — Iterative Fortsetzung "lofi_girl" (21.06.–03.07.2026, Round 2–11)

Zehn aufeinanderfolgende, jeweils auf dem Vorgänger-Checkpoint aufbauende Kurzrunden
(`training/video/ltx_lora_round2` … `ltx_lora_round11`), protokolliert in
`training_runs/overview.csv`:

| Runde | Datum | Steps | Qualität (1–10, Selbsteinschätzung) | Bemerkung |
|---|---|---|---|---|
| Round 2 | 21.–22.06. | 50 | 3 | Anime-Assoziation stärker |
| Round 3 | 22.06. | 50 | 3 | Cel-Shading-Tendenzen erkennbar |
| Round 4 | 22.06. | 50 | 5 | Inference-Steps 30→50: deutlicher Qualitätssprung |
| Round 5 | 22.06. | 50 | 7 | **Bester Kompositions-Checkpoint dieser Phase** |
| Round 6 | 22.06. | 50 | 4 | Szenendrift — Überanpassung nach dem Peak von Round 5 |
| Round 7 | 22.06. | 100 | 4 | Image-Conditioning-Fehler (Letterboxing) erkannt, zurückgesetzt auf Round-5-Basis |
| Round 8 | 01.07. | 100 | 7 | Neues 15-Clip-Dataset, Conditioning-Fix, flüssigere Animation |
| Round 9 | 03.07. | 100 | 7 | Validierung wegen OOM deaktiviert, 2× Absturz/Reload während des Laufs |
| Round 10 | 03.07. | – | – | Ordner vorhanden, nicht in `overview.csv` dokumentiert (nicht nachgewiesen, was hier trainiert wurde) |
| Round 11 | 03.07. | 100 | – | Checkpoint `step_00100` wird als geteilter Basis-Checkpoint für spätere Szenarien übernommen |

Diese Phase etabliert die grundlegende Trainingsmethodik, die danach beibehalten wird:
rundenbasiert, Human-in-the-Loop, Checkpoint alle 25 Steps, Entscheidung "weiter/zurück"
nach jeder Runde anhand von Sichtprüfung.

## Phase 4 — Umbau zum Szenario-System `lofi_pipeline/` (ab 03.07.2026)

- **03.07.2026**: Der Round-11-Checkpoint (Step 100) wird als
  `base_checkpoints/lofi_lora_r11_s100.safetensors` in die neue Pipeline übernommen —
  Startpunkt für mehrere thematisch eigenständige Szenarien statt eines einzigen
  fortlaufenden Trainingsstrangs. `lofi_pipeline/reports/` wird angelegt (bis heute
  ungenutzt, siehe `01_struktur.md`).
- Es werden **sechs Szenarien** definiert: `cafe_scene`, `library_study`,
  `lofi_girl_desk` (formale Fortführung von Phase 2/3, aber **nie** unter dem neuen
  System weitertrainiert), `rabbit_lake`, `rainy_window`, `train_window`.
- `rainy_window` startet als erstes neues Szenario (Round 1, 03.07.2026), aufbauend
  auf dem geteilten Basis-Checkpoint.

## Phase 5 — Parallele Weiterentwicklung zweier Szenarien (Juli 2026)

### `rainy_window` — 14 Runden, 03.07.–21.07.2026

Motiv: Mädchen schreibt am Fenster bei Regen, Stadtansicht im Hintergrund. Verlauf laut
`feedback.json`-Einträgen:

- **Round 3** (03.07.): Grundszene funktioniert, Fenster als Hauptfokus etabliert;
  offene Punkte: Auflösung, Regen-Sichtbarkeit, Schreibszene fehlt noch.
- **Round 4** (10.07.): Schreibszene ergänzt, Stadtansicht schärfer; erstmals als
  Problem benannt: **Kompositionsdrift** — jede Runde erzeugt eine leicht andere
  Anordnung statt dieselbe Szene zu verfeinern. Gegenmaßnahme: explizite
  "stable fixed composition unchanged"-Prompt-Anker, stärkere Regen-Betonung im Prompt,
  `guidance_scale` 4,5→5,0.
- **Round 9** (21.07., vormittags): weiteres Training, 100 Steps, Spitzenspeicher 42 GB.
- **Round 14** (21.07., abends, letzte dokumentierte Runde): 4 Checkpoints (25/50/75/100
  Steps) und 4 MP4-Samples erzeugt. **Keine `feedback.json` vorhanden** — diese Runde
  wurde bisher nicht menschlich bewertet. Es gibt damit **keinen offiziell
  abgeschlossenen/freigegebenen Stand** für dieses Szenario.

### `rabbit_lake` — 9 Runden, Startdatum nicht exakt protokolliert (vor 22.07., laut Feedback-Notizen "Round 5" bereits vor Erstellung des Rescue-Plans)

Motiv: zwei Hasen an einem See bei Nacht, Laterne, Spiegelungen, Hütte am
gegenüberliegenden Ufer.

- **Kernproblem seit Runde 1 ungelöst:** Das 13B-Basismodell erzeugt beharrlich **drei**
  statt zwei Figuren.
- **Ursachenanalyse** (dokumentiert in `scenarios/rabbit_lake/RESCUE_PLAN.md`, Stand
  Runde 5): Prior-Bias des Basismodells für Drei-Figuren-Kompositionen in dieser
  Szenenbeschreibung, der mit 100 LoRA-Steps auf den vorhandenen Trainingsclips nicht
  überschrieben werden kann.
- **Datenqualitätsproblem entdeckt:** Von 17 ursprünglichen Trainingsclips waren bei
  genauerer Prüfung **14 inhaltlich falsch** (falscher Stil, falsche Tageszeit, falsche
  Motive — u. a. Innenraumszenen, Cartoon-Stil, Wasserzeichen). Nur `bamboo_01-03` waren
  korrekt.
- **Datensatz-Neuaufbau ab Round 5:** ausschließlich noch geprüfte Clips
  (`bamboo_01-07`, `cabin_01-06`, 13 Clips gesamt).
- **Rescue-Plan-Phasen:** Phase 1 (exakt 2 Hasen im Standbild) → Phase 2 (Animation) →
  Phase 3 (MP4-Export). Nachweisbar wurde **nur Phase 1 bearbeitet, nie zuverlässig
  gelöst**; Phase 2 (Bewegung/Animation) wurde **nie erreicht**.
- **Round 9** (letzte dokumentierte Runde): Lauf endete mit einem **CUDA-Out-of-Memory-
  Absturz** (47,39 GB GPU-Kapazität, Anforderung von 32 MB scheiterte). Der Ordner
  enthält nur `config.yaml` und `log.txt` — keine Checkpoints/Samples dieser Runde.

### Die übrigen vier Szenarien

`cafe_scene`, `library_study`, `lofi_girl_desk`, `train_window` haben vollständige
Konfigurationsordner (`scenario.yaml`, leere `assets/`, `precomputed/`, `rounds/`), aber
**null Trainingsrunden**. Sie wurden angelegt, aber nie begonnen.

## Phase 6 — Werkzeugausbau (Juli 2026, parallel zu Phase 5)

Zeitgleich zu den Trainingsrunden entstehen unterstützende Skripte in
`lofi_pipeline/scripts/` (letzte Änderungen 28.07.2026):

- GIF-Export aus MP4 (`generate_samples.py`, ffmpeg-Zweipass-Palettenverfahren) —
  laut Code-Kommentar bewusst *sekundär*: "Primary output is MP4. GIFs are optional
  and generated on request only […] assess from MP4, convert best results to GIF later."
- Video-Nachbearbeitung (`enhance_video.py`): RIFE-Frameninterpolation +
  Real-ESRGAN-Anime-Hochskalierung, mit ffmpeg-`minterpolate`-Fallback falls RIFE
  nicht verfügbar ist.
- Lokale Review-Weboberfläche (`feedback_ui.py`) und Terminal-Variante (`feedback.py`)
  zur Sichtung/Bewertung neuer Samples.
- Auswertungs-Skripte `compare_rounds.py` (Runden vergleichen) und `build_report.py`
  (CSV-/Markdown-Bericht bauen) — **beide bislang nicht produktiv ausgeführt**
  (`reports/`-Ordner ist leer).

## Heute (Stand 04.08.2026)

- **Aktivste/reifste Spur:** `rainy_window`, mit einer unbewerteten Runde 14 als
  letztem Stand.
- **Blockierte Spur:** `rabbit_lake`, ungelöstes Inhaltsproblem seit Runde 1, letzter
  Lauf mit Absturz beendet, Animationsphase nie erreicht.
- **Keine Aktivität** an den übrigen vier Szenarien.
- **Phase 0 (`video_pipeline`) am 04.08.2026 nachträglich analysiert:** Code technisch
  funktionsfähig bestätigt (siehe oben), aber keine der ursprünglich erzeugten
  Videodaten mehr auffindbar; drei Zip-Sicherungen als Beleg gesichert in
  `~/Downloads/video_pipeline*.zip` (Original) bzw. entpackt zum Vergleich unter
  `video_pipeline_restored/v1_mai06_0503/`, `v2_mai06_2335/`, `v3_mai10/`.
- **Kein Git-Commit** hat den `Bachelorarbeit/`-Ordner seit dem **08.06.2026** verändert —
  die gesamte in Phase 2–6 dokumentierte Arbeit (13.06.–28.07.2026) liegt ausschließlich
  lokal und unversioniert vor. Phase 0 (`video_pipeline`, Mai 2026) lag ohnehin nie im
  Git-Repository, sondern separat unter `~/Downloads/`.
- Letzte lokale Dateiänderung im gesamten Video/GIF-Bereich der `lofi_pipeline`: **23.07.2026**
  (`references/`-Ordner). Seitdem (rund zwei Wochen bis heute) keine nachweisbare
  weitere Aktivität an diesem Projektteil.

## Entwicklungsbogen: von der Automatisierungs-Absicht zum tatsächlichen Ergebnis

Die Ausgangsidee im Mai (Phase 0) war eine vollautomatische Pipeline von der Quellensuche
bis zum fertigen GIF, mit dem eigentlichen Training als bewusst ausgelagertem, noch nicht
umgesetztem Schritt. Was daraus tatsächlich entstand, verlief anders als ursprünglich
geplant, aber nicht ergebnislos:

1. **Mai (Phase 0):** Automatisierte Datenbeschaffung konzipiert und lauffähig implementiert,
   nie mit echten Daten ausgeführt. Trainingsschritt bewusst offengelassen.
2. **Juni (Phase 2–3):** Der zuvor nur referenzierte externe Trainer wird real eingebunden
   (`tools/LTX-Video-Trainer/`), zunächst zwangsläufig manuell und rundenbasiert, weil ein
   automatisches Qualitätskriterium für "wann ist eine Runde gut genug" fehlte — diese Lücke
   war im Mai-Konzept nie geschlossen worden.
3. **Juli (Phase 4–5):** Ausweitung auf mehrere Szenarien, dabei Aufdeckung inhaltlicher
   Probleme (`rabbit_lake`), die rein technische Automatisierung ohnehin nicht gelöst hätte.
4. **06.08.2026 (Phase 7):** Erstmals wird genau die im Mai fehlende Zutat nachgereicht —
   ein automatisches Qualitätskriterium (`evaluate_video.py`/`metric_watcher.py`) mit
   Zielwerten, das Trainingsrunden ohne Mensch bewerten und bei Erreichen selbstständig
   stoppen kann. Das ist die erste tatsächlich funktionierende Automatisierung *innerhalb*
   des Trainings — etwas, das die Mai-Pipeline konzeptionell noch nicht vorsah.

**Fazit für die Arbeit:** Der Weg verlief nicht linear von "automatisch" zu "manuell" und
wieder zurück, sondern von einer *unvollständigen* Automatisierungsidee (nur Datenbeschaffung)
über eine notwendige manuelle Zwischenphase (mangels Bewertungskriterium) hin zu einer
*gezielteren* Automatisierung, die genau die ursprüngliche Lücke schließt — allerdings bislang
nur für einzelne Szenarien und ohne rückwirkende Anwendung auf die frühen, unbewerteten Runden.
