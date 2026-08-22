# Entwicklung und Bewertung einer automatisierten Pipeline zur KI-gestützten Generierung von Lo-Fi-Musik

Bachelorarbeit, Technische Universität Berlin — Institut für Werkzeugmaschinen und Fabrikbetrieb,
Fachgebiet Industrielle Automatisierungstechnik.

Dieses Repository ist der konsolidierte Code- und Dokumentationsstand des Projekts: eine lokale
Pipeline, die Lo-Fi-Audiodaten erfasst, aufbereitet und daraus über MusicGen+LoRA neue Lo-Fi-Musik
generiert und automatisch bewertet, ergänzt um eine passende Video-/GIF-Pipeline (LTX-Video) und
eine gemeinsame Studio-Weboberfläche für beide Teile.

**Für den vollständigen Nachbau des Projekts (Umgebung, Modelle, Datensatz, alle drei Teile) siehe
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).**

## Team

| | Schwerpunkt |
|---|---|
| **Adil Berke Savcili** (M. Sc. Maschinenbau) | Datenerfassung, Audioverarbeitung, MusicGen+LoRA-Training, Musikgenerierung, Evaluation → [`audio-pipeline/`](audio-pipeline/) |
| **Inan Deniz Arduc** (M. Sc. Maschinenbau) | Video-/GIF-Generierung, LTX-Video-Training, visuelle Ausgabe → [`video-pipeline/`](video-pipeline/) |

Gemeinsam: Konzeption der Gesamtpipeline, Integration der Teilkomponenten, Studio-Website
([`frontend/`](frontend/)).

Betreuung: Prof. Dr.-Ing. Jörg Krüger. Ansprechpartner: Adam Michael Altenbuchner, M. Sc.

## Struktur

```text
.
├── audio-pipeline/   Python: Datenerfassung, Audioverarbeitung, MusicGen+LoRA, Longform-Generierung,
│                     Evaluation, lokales Backend (web_api.py)
├── video-pipeline/   Python: LTX-Video-LoRA-Training, Sample-/GIF-Generierung, Feedback-UI
├── frontend/         React/TypeScript: Studio-Weboberfläche für beide Pipelines
└── REPRODUCIBILITY.md
```

Jeder Teilordner hat eine eigene, ausführlichere README:
[`audio-pipeline/README.md`](audio-pipeline/README.md) ·
[`video-pipeline/README.md`](video-pipeline/README.md) ·
[`frontend/README.md`](frontend/README.md)

## Kurzüberblick Audio-Pipeline

1. Quellen suchen/importieren → 2. Rohaudio in standardisierte Clips überführen → 3. Audio-Merkmale
extrahieren & filtern → 4. genrebalanciertes Trainingsdataset bauen → 5. MusicGen lokal per LoRA
anpassen → 6. Testaudios erzeugen & bewerten → 7. aus kurzen Clips lange, geloopte Audios mit
Übergangsprüfung erzeugen → 8. automatische Bewertung gegen Genrestandard → 9. alles über die
Studio-Website steuerbar machen.

Details, Architektur und alle CLI-Befehle: [`audio-pipeline/README.md`](audio-pipeline/README.md).

## Kurzüberblick Video-/GIF-Pipeline

Erzeugt kurze Video-Loops (LTX-Video 13B + eigene LoRA-Adapter pro Szenario) passend zu Lo-Fi-Szenen,
mit Feedback-gestütztem, iterativem Training pro Szenario. Details:
[`video-pipeline/README.md`](video-pipeline/README.md).

## Wichtig: was in diesem Repository fehlt (und warum)

Dieses Repository enthält **Code, Konfiguration und Dokumentation**, bewusst **keine**:

- Rohdaten und Trainingsdatensätze (Audio-/Videoquellen sind überwiegend fremde, urheberrechtlich
  geschützte YouTube-Inhalte)
- vortrainierte Basismodelle (MusicGen, LTX-Video) und LoRA-Checkpoints
- generierte Ausgaben (Audios, Videos, GIFs) und Bewertungsergebnisse aus konkreten Testläufen
- virtuelle Python-Umgebungen, `node_modules`

Diese Inhalte sind entweder zu groß fürs Repository, nicht weitergabefähig, oder schlicht
Zwischenergebnisse, die sich reproduzierbar neu erzeugen lassen. Wie das geht, steht vollständig in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Wissenschaftliche Einordnung

Die Arbeit kombiniert generative Modellierung mit einem kontrollierten technischen
Bewertungssystem auf vier Ebenen: Datenebene, Modellebene, Generierungsebene, Evaluationsebene.
Diese Trennung macht Ergebnisse einem konkreten Problembereich zuordenbar (Daten-, Genre-,
Trainings-, Generierungs-, Übergangs- oder Bewertungsproblem). Ausführlich in
[`audio-pipeline/README.md`](audio-pipeline/README.md#16-wissenschaftliche-einordnung).
