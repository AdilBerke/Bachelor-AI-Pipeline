# Code-Erklärung: Video-/GIF-Skripte

Beschreibt die aktiven Skripte des Video-/GIF-Teils. Für den Audio-Teil siehe `../../audio-pipeline/README.md`.

## Frühe Phase (bis 03.07.2026)

`Bachelorarbeit/pipeline/v003_manual/generate.py`
: Eigenständiges Generierungs-Skript. Lädt LTX-Video 13B 0.9.7-dev über
`ltxv_trainer.model_loader` und nutzt `LTXConditionPipeline`
(`tools/LTX-Video-Trainer`) direkt, mit fest im Skript hinterlegtem Prompt/
Negativ-Prompt für die "lofi_girl"-Szene. Vorläufer des heutigen
`generate_samples.py`.

## `lofi_pipeline/scripts/` — aktives Szenario-System

`preprocess_scenario.py`
: Berechnet für ein neues Szenario die VAE-Latents und T5-Text-Embeddings aus den
Rohclips. Muss einmalig pro Szenario laufen, bevor trainiert werden kann.
```
python preprocess_scenario.py --scenario rainy_window
```

`train_lora.py`
: Zentrales Trainings-CLI. Startet eine Trainingsrunde für ein Szenario, optional
fortsetzend von einem vorhandenen Checkpoint (`--resume`) oder von einem beliebigen
Checkpoint-Pfad (`--from-checkpoint`). Speichert Checkpoints alle 25 Steps und
protokolliert Kennzahlen (Dauer, GPU-Speicher, Trainingsgeschwindigkeit) in
`rounds/round_NN/notes.json`.
```
python train_lora.py --scenario rainy_window --round 15 --steps 100 --resume
```

`generate_samples.py`
: Erzeugt Beispiel-Videos aus den Checkpoints einer Runde. **MP4 ist das primäre
Ausgabeformat**; GIF ist laut Skript-Kommentar bewusst optional/sekundär, um beim
Sichten während des Trainings keine Zeit mit 256-Farb-Palettenqualität zu verlieren
— GIFs werden erst aus den besten MP4s im Nachhinein erzeugt. GIF-Konvertierung
läuft über ein Zweipass-ffmpeg-Palettenverfahren (`make_gif()`).
```
python generate_samples.py --scenario rainy_window --round 14          # nur MP4
python generate_samples.py --scenario rainy_window --round 14 --gif    # + GIF
python generate_samples.py --scenario rainy_window --round 14 --gif-only  # nur Konvertierung vorhandener MP4s
```

`enhance_video.py`
: Optionale Nachbearbeitung fertiger MP4s: Frameninterpolation (für höhere
Bildrate) und Hochskalierung. Zwei Methoden: `minterpolate` (ffmpeg,
motion-compensated, Standard) und `rife` (RIFE-ncnn-vulkan-Optical-Flow +
Real-ESRGAN-Anime-Upscale, fällt automatisch auf `minterpolate` zurück, falls das
RIFE-Binary nicht gefunden wird).
```
python enhance_video.py --scenario rainy_window --round 14 --checkpoint 100 --method rife --target-fps 60
```

`feedback.py`
: Terminal-basiertes Bewertungswerkzeug. Zeigt neue Samples an, nimmt Bewertung und
Korrekturwünsche entgegen und schreibt sie als `feedback.json` in den jeweiligen
Round-Ordner (Felder u. a. `overall_direction`, `positives`, `improvements_needed`,
`preserve`, `recommendation`, `prompt_changes`).
```
python scripts/feedback.py --latest
python scripts/feedback.py --round 7
```

`feedback_ui.py`
: Browser-basierte Variante desselben Zwecks — lokaler Server, zeigt Samples
(Video/GIF) in einer Weboberfläche zur Sichtung. Internes Review-Werkzeug für den
Studierenden, keine Endnutzer-Oberfläche.

`compare_rounds.py`
: Lädt die `notes.json`/`feedback.json` mehrerer Runden eines Szenarios und stellt
sie zum Vergleich gegenüber (Qualitätswerte, Prompt-Änderungen). **Bislang nicht
produktiv ausgeführt** — kein Aufruf-Ergebnis im Repository gefunden.
```
python compare_rounds.py --scenario rainy_window --rounds 4,9,14
```

`build_report.py`
: Aggregiert alle Runden eines oder aller Szenarien zu einem CSV-/Markdown-Bericht
in `lofi_pipeline/reports/`. **Bislang nicht produktiv ausgeführt** — der
`reports/`-Ordner ist leer.
```
python build_report.py --scenario rainy_window
```

## Wissenschaftliche Einordnung des Ist-Zustands

Die Video-/GIF-Pipeline ist — wie der Audio-Teil — konzeptionell ein
Human-in-the-Loop-System:

1. Ein Szenario wird als YAML-Konfiguration (Prompt, Auflösung, Qualitätskriterien)
   definiert.
2. Referenzclips werden manuell kuratiert und vorverarbeitet (`preprocess_scenario.py`).
3. LoRA wird in kleinen Schritten trainiert (`train_lora.py`), ausgehend vom
   Basismodell oder einem geteilten Basis-Checkpoint.
4. Nach jeder Runde werden Beispiel-Videos erzeugt (`generate_samples.py`).
5. Ein Mensch sichtet die Samples (`feedback.py`/`feedback_ui.py`) und entscheidet über
   Fortsetzung, Prompt-Anpassung oder Datensatz-Korrektur.
6. Optionale Nachbearbeitung (`enhance_video.py`) und GIF-Export erfolgen erst für als
   gut bewertete Ergebnisse.

**Unterschied zum Audio-Teil, der für die wissenschaftliche Bewertung relevant ist:**
Für den Video/GIF-Teil existieren — anders als für Audio (dort: automatischer
Referenzvergleich, technische Kandidatenprüfung, 5-Score-Bewertung) — **keine
automatisierten/quantitativen Qualitätsprüfungen**. Jede bisherige Bewertung eines
Video-Samples ist eine subjektive Einschätzung einer einzelnen Person, festgehalten in
`feedback.json`. Es gibt keine Entsprechung zu `audio_bewertung.py` oder
`referenz_vergleich.py` für Video.
