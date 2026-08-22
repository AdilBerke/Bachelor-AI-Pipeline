# Video-/GIF-Pipeline

Erzeugt kurze, stimmungskonsistente Video-Loops (LTX-Video 13B + LoRA) passend zu Lo-Fi-Musikszenen
(z. B. "Regenfenster", "ruhiger See"). Fachlich und im Code vollständig getrennt von der Audio-Pipeline
(`../audio-pipeline`), teilt sich mit ihr aber dieselbe Python-Umgebung (`.venv`, siehe
[../REPRODUCIBILITY.md](../REPRODUCIBILITY.md)).

Bearbeitet von Inan Deniz Arduc (M. Sc. Maschinenbau) im Rahmen derselben Bachelorarbeit.

## Wichtiger Hinweis zu diesem Ordner

Dieser Ordner enthält nur **Code, Konfiguration und kleine Referenzdateien**. Nicht enthalten
(bewusst, siehe [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md)):

- Trainings-Checkpoints und LoRA-Gewichte (`base_checkpoints/`)
- Szenario-Trainingsdaten, Zwischenergebnisse und generierte Videos/GIFs (`scenarios/`)
- Referenz-Quellclips (`references/source_videos/*.mp4`)
- Das LTX-Video-13B-Basismodell selbst (wird beim ersten Lauf automatisch von Hugging Face geladen)

Diese Inhalte bleiben lokal auf dem Trainingsrechner bzw. werden gemäß Schritt 4 in
[../REPRODUCIBILITY.md](../REPRODUCIBILITY.md) neu erzeugt.

## Struktur

| Pfad | Inhalt |
|---|---|
| `lofi_pipeline/scripts/` | aktive Pipeline-Skripte: Preprocessing, LoRA-Training, Sample-Generierung, GIF-Export, Feedback-UI, Report-Bau |
| `lofi_pipeline/configs/` | Basis-LoRA-Konfigurationen (Rang, Lernrate, Scheduler) und Modellpfade |
| `lofi_pipeline/projects/` | Beispiel-Szenario-Definition |
| `lofi_pipeline/references/legal/` | Prüfliste zu Nutzungsrechten der Referenzquellen |
| `lofi_pipeline/references/motion_profiles/` | wiederverwendbare Bewegungsprofile (Wind, Wasser, Blinzeln, Sternschnuppen) |
| `lofi_pipeline/references/notes/` | Analysenotizen zu Referenzmaterial |
| `pipeline/v003_manual/generate.py` | aktuelle, manuell gesteuerte LTX-Video-Generierung |
| `pipeline/realistic_rabbit/` | Testskripte/Auswertung für ein einzelnes Szenario (Vorstufe) |
| `dokumentation/` | ausführliche Struktur-, Verlaufs- und Einrichtungsdokumentation (`05_einrichtung.md` = Schritt-für-Schritt-Anleitung für neue Szenarien, Training, Sample-Erzeugung, Bewertung) |

## Setup und Nutzung

Vollständige Befehle stehen in [`dokumentation/05_einrichtung.md`](dokumentation/05_einrichtung.md).
Kurzfassung:

```bash
# aus dem Projekt-Root, gemeinsames .venv (siehe ../REPRODUCIBILITY.md)
source ../audio-pipeline/../.venv/bin/activate   # bzw. Pfad zum eigenen .venv anpassen

python video-pipeline/lofi_pipeline/scripts/preprocess_scenario.py --scenario <name>
python video-pipeline/lofi_pipeline/scripts/train_lora.py --scenario <name> --round 1
python video-pipeline/lofi_pipeline/scripts/generate_samples.py --scenario <name> --round 1 --gif
python video-pipeline/lofi_pipeline/scripts/feedback_ui.py   # Browser-Bewertung, Port 7860
```

## Bekannter Stand

Kein Szenario gilt als vollständig trainiert; siehe `dokumentation/02_verlauf.md` und
`dokumentation/04_stand_szenarien.md` für den aktuellen Trainingsstand pro Szenario und bekannte
offene Probleme (z. B. `rabbit_lake`: teils falsche Objektanzahl im Ergebnis).
