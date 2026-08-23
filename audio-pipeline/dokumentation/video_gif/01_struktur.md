# Struktur: Video-/GIF-Pipeline

Diese Dokumentation deckt den **Video-/GIF-Teil** des Projekts ab (LTX-Video-LoRA-Training,
Szenario-Generierung, GIF-Export, Nachbearbeitung). Sie ergänzt `dokumentation/aktuell/`,
das ausschließlich den MusicGen-Audio-Teil beschreibt.

Stand dieser Dokumentation: 04.08.2026, erstellt durch direkte Prüfung von Code,
Konfigurationsdateien, Logs und JSON-Metadaten im Repository.

## Hauptbereiche

`Bachelorarbeit/lofi_pipeline/`
: Aktives, szenario-basiertes System (seit ca. 03.07.2026). Enthält Konfiguration,
Skripte, Referenzmaterial, Szenario-Ordner mit Trainingsrunden, geteilte Basis-Checkpoints
und Berichte.

`Bachelorarbeit/training/video/`
: Älterer, nicht-szenario-basierter Trainingsverlauf (13.06.–03.07.2026):
`ltx_lora_manual` (Erststart) und `ltx_lora_round2` bis `ltx_lora_round11`
(10 Folgerunden auf einem einzigen Motiv, "lofi_girl"). Bildet die Vorstufe zu
`lofi_pipeline/`. Details siehe `02_verlauf.md`.

`Bachelorarbeit/pipeline/v003_manual/generate.py`
: Einzelnes Generierungs-Skript aus der frühen Phase, ruft die LTXV-13B-Pipeline
(`LTXConditionPipeline` aus `tools/LTX-Video-Trainer`) direkt auf.

`Bachelorarbeit/daten/processed/ltx_lora_manual/`
: Trainingsdaten (VAE-Latents, T5-Text-Embeddings, Vorschau-GIFs) für die frühe
"lofi_girl"-Szene.

`Bachelorarbeit/logs/`
: Rohe Trainings-/Preprocessing-Logs der frühen Phase (13.–21.06.2026).

`tools/LTX-Video-Trainer/`
: Externes Trainings-Framework, Grundlage aller LoRA-Trainingsläufe (Preprocessing,
Training, Validierungs-Inferenz).

`tools/rife-ncnn-vulkan/`, `tools/realesrgan-models/`
: Werkzeuge für optionale Nachbearbeitung (Frameninterpolation, Anime-Hochskalierung).

## `lofi_pipeline/` im Detail

`configs/`
: Globale Konfiguration (`model_paths.yaml`, `base_lora.yaml`).

`scenarios/<name>/`
: Ein Ordner pro Motiv/Szenario. Aktuell sechs definiert: `cafe_scene`, `library_study`,
`lofi_girl_desk`, `rabbit_lake`, `rainy_window`, `train_window`. Jeder Szenario-Ordner
enthält `scenario.yaml` (Prompt, Auflösung, Qualitätskriterien), `assets/`,
`precomputed/` und `rounds/`.

`scenarios/<name>/rounds/round_NN/`
: Eine Trainingsrunde. Enthält `config.yaml` (Trainingsparameter), `log.txt`
(Rohausgabe), `notes.json` (automatisch protokollierte Kennzahlen: Dauer, Steps,
GPU-Speicher, erzeugte Samples), optional `feedback.json` (menschliche Bewertung,
nur wenn die Runde tatsächlich gesichtet wurde), `checkpoints/` (`.safetensors` je
25/50/75/100 Steps) und `samples/` (MP4, optional GIF).

`base_checkpoints/`
: Geteilte Ausgangs-Checkpoints — `lofi_lora_r11_s100.safetensors` und
`lofi_lora_best.safetensors`. Das ist der aus der frühen "lofi_girl"-Phase
(Round 11, siehe `02_verlauf.md`) hervorgegangene LoRA-Stand, der als Startpunkt
für neue Szenarien dient.

`scripts/`
: Ausführbare Pipeline-Werkzeuge. Siehe `03_code_erklaerung.md`.

`reports/`
: Vorgesehen für automatisierte Auswertungsberichte aus `build_report.py`/
`compare_rounds.py`. **Leer** (Stand 04.08.2026) — diese Skripte wurden bisher nicht
produktiv ausgeführt.

`references/`
: Referenzvideos, Motion-Profile (`motion_profiles/`), Rechte-Checkliste
(`legal/source_rights_checklist.md`), Analyse-Notizen zur Prompt-/Stilentwicklung
(`notes/reference_analysis.md`).

## Außerhalb des Projektordners: `~/Downloads/video_pipeline/`

Der tatsächlich allererste Versuch (Mai 2026, siehe `02_verlauf.md`, Phase 0) liegt
**nicht** unter `Bachelor_VisiualStudio/`, sondern separat unter
`/home/BA_Musikproduktion/Downloads/video_pipeline/` — eine eigenständige, vollautomatische
Trainingsdaten- und Clip-zu-GIF-Pipeline (Suche → Download → Schnitt → Frame-Extraktion →
Tagging → Dataset → GIF). Drei chronologische Zip-Sicherungen davon liegen ebenfalls in
`~/Downloads/` und wurden zum Vergleich nach
`.../scratchpad/video_pipeline_restored/{v1_mai06_0503,v2_mai06_2335,v3_mai10}/` entpackt.
Kein eigenes Git-Repository, kein Bezug zu `Bachelor_VisiualStudio/.git`.

## Was im aktuellen Dateisystem nicht (mehr) vorhanden ist

Frühere Projektnotizen erwähnen eine Version **v002** (512×288, 3 s, 29 Clips) und eine
automatisierte Overnight-Pipeline **v003** (`run_overnight.py`, Ziel 832×480, 2000 Steps,
~80 Videos, Modell `LTXV_2B_0.9.6_DEV`). Im aktuellen Repository (Stand 04.08.2026) sind
**weder Code noch Ausgabedaten noch Git-Commits** zu diesen Versionen auffindbar. Der
Ansatz wurde offenbar zugunsten des manuellen, kleinschrittigen Verfahrens
(`v003_manual` → `lofi_pipeline`) aufgegeben, bevor er — soweit nachvollziehbar —
abgeschlossen wurde. Dies ist **nicht abschließend geklärt** (keine Löschung im
Git-Log sichtbar, da der Ordner offenbar nie committet wurde) und sollte bei Bedarf
mit dem Studierenden verifiziert werden.

## Git-Status dieses Bereichs

Der letzte Commit, der `Bachelorarbeit/` verändert hat, ist vom **08.06.2026**. Alle
in `02_verlauf.md` dokumentierten Trainingsrunden (13.06.–28.07.2026) liegen
**unversioniert** nur lokal vor. Die aktuell aktiv gepflegte Projektdokumentation
(`dokumentation/aktuell/`) und praktisch alle Commits der letzten Wochen behandeln
ausschließlich den Audio-Teil und bezeichnen `Bachelorarbeit/` als "Altbestand".
