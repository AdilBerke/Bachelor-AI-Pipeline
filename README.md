# Entwicklung und Bewertung einer automatisierten Pipeline zur KI-gestützten Generierung von Lo-Fi-Musik

Bachelorarbeit, Technische Universität Berlin — Institut für Werkzeugmaschinen und Fabrikbetrieb,
Fachgebiet Industrielle Automatisierungstechnik.

Dieses Repository ist der konsolidierte Code- und Dokumentationsstand des Projekts: eine lokale
Pipeline, die Lo-Fi-Audiodaten erfasst, aufbereitet und daraus über MusicGen+LoRA neue Lo-Fi-Musik
generiert und automatisch bewertet, ergänzt um eine passende Video-/GIF-Pipeline (LTX-Video) und
eine gemeinsame Studio-Weboberfläche für beide Teile.

**Für den vollständigen Nachbau des Projekts (Umgebung, Modelle, Datensatz, alle drei Teile):
[`NACHBAUANLEITUNG.md`](NACHBAUANLEITUNG.md) führt Schritt für Schritt durch die Einrichtung,
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) enthält dieselben Schritte ausführlicher mit
Hintergrund zu Versionen, Umgebungen und Reproduzierbarkeit.**

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
├── audio-pipeline/        Python: Datenerfassung, Audioverarbeitung, MusicGen+LoRA, Longform-
│                          Generierung, Evaluation, lokales Backend (web_api.py)
├── video-pipeline/        Python: LTX-Video-LoRA-Training, Sample-/GIF-Generierung, Feedback-UI
├── frontend/              React/TypeScript: Studio-Weboberfläche für beide Pipelines
├── models/                der in der Arbeit verwendete LoRA-Adapter der Audio-Pipeline (Git LFS)
├── requirements.txt       Hauptumgebung
├── requirements-niqe.txt  separate Umgebung für die NIQE-Bildqualitätsmetrik (siehe unten)
├── NACHBAUANLEITUNG.md
└── REPRODUCIBILITY.md
```

Die beiden Requirements-Dateien gehören zu **zwei getrennten virtuellen Umgebungen**: `pyiqa`
(NIQE) erzwingt eine Transformers-Version, die die Hauptumgebung unbrauchbar machen würde. Die
Trennung ist daher notwendig und in beiden Anleitungen beschrieben.

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

Aufbau und wichtigste Dateien: [`audio-pipeline/README.md`](audio-pipeline/README.md).
Die konkreten Befehle stehen in [`NACHBAUANLEITUNG.md`](NACHBAUANLEITUNG.md).

## Kurzüberblick Video-/GIF-Pipeline

Erzeugt kurze Video-Loops (LTX-Video 13B + eigene LoRA-Adapter pro Szenario) passend zu Lo-Fi-Szenen,
mit Feedback-gestütztem, iterativem Training pro Szenario. Details:
[`video-pipeline/README.md`](video-pipeline/README.md).

## Der trainierte Adapter

Der in der Arbeit tatsächlich verwendete LoRA-Adapter der Audio-Pipeline liegt unter
[`models/lora_adapter.pt`](models/lora_adapter.pt) — Rank 8, Alpha 16, Dropout 0,05, 625
Trainingsschritte, rund 9,44 Mio. trainierbare Parameter, trainiert auf 5.000 genrebalancierten
30-Sekunden-Clips. Die Ergebnisse der Arbeit lassen sich damit ohne eigenes Training nachvollziehen.

Die Datei wird über **Git LFS** verwaltet, da sie über GitHubs 100-MB-Grenze liegt. Zum Klonen wird
daher `git-lfs` benötigt; ohne installiertes `git-lfs` erhält man an dieser Stelle kommentarlos nur
eine kleine Zeigerdatei statt der Modellgewichte. Gleiches gilt für den Button *Download ZIP* auf
der Repository-Seite: Solche Archive lösen LFS-Dateien nicht auf. Der Download-Button auf der
Dateiseite selbst liefert die vollständige Datei. Einzelheiten und Prüfsumme:
[`models/README.md`](models/README.md).

## Wichtig: was in diesem Repository fehlt (und warum)

Darüber hinaus enthält das Repository **Code, Konfiguration und Dokumentation**, bewusst **keine**:

- Rohdaten und Trainingsdatensätze (Audio-/Videoquellen sind überwiegend fremde, urheberrechtlich
  geschützte YouTube-Inhalte)
- vortrainierte Basismodelle (MusicGen, LTX-Video) — diese werden beim ersten Lauf automatisch
  von Hugging Face geladen
- die LoRA-Checkpoints der Video-Pipeline (kein Szenario gilt als abgeschlossen trainiert)
- generierte Ausgaben (Audios, Videos, GIFs) und Bewertungsergebnisse aus konkreten Testläufen
- virtuelle Python-Umgebungen, `node_modules`

Diese Inhalte sind entweder zu groß fürs Repository, nicht weitergabefähig, oder schlicht
Zwischenergebnisse, die sich reproduzierbar neu erzeugen lassen. Wie das geht, steht vollständig in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Wissenschaftliche Einordnung

Die Arbeit kombiniert generative Modellierung mit einem kontrollierten technischen
Bewertungssystem auf vier Ebenen: Datenebene, Modellebene, Generierungsebene, Evaluationsebene.
Diese Trennung macht Ergebnisse einem konkreten Problembereich zuordenbar (Daten-, Genre-,
Trainings-, Generierungs-, Übergangs- oder Bewertungsproblem).

Die fertige Audioausgabe wird dabei in **acht Kategorien** mit Werten zwischen 0 und 100 bewertet,
die bewusst **nicht** zu einem Gesamtscore verrechnet werden — ein schwaches Ergebnis bleibt so
einer konkreten Ursache zuordenbar, statt in einem Mittelwert zu verschwinden. Welche Kategorien
das sind und wie sie berechnet werden: [`audio-pipeline/README.md`](audio-pipeline/README.md).
