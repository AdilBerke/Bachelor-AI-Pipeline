# Projektstruktur

Aktueller Fokus: MusicGen-Audio fuer die Bachelorarbeit. Die aktive Struktur
soll aus vier grossen Bereichen bestehen:

- `code/`
- `daten/`
- `training/`
- `dokumentation/`

Video- und GIF-Dateien werden nicht bereinigt oder umgebaut, weil daran separat
gearbeitet wird.

## Aktive Hauptbereiche

`code/`
: Alles, was gestartet oder importiert wird. Dazu gehoeren Crawler, Dataset-Bau,
LoRA-Training, Audio-Generierung, Looping, Bewertung und Konfiguration.

`daten/`
: Lokale Rohdaten, verarbeitete Datasets, Metadaten, Features und lokale
Modelle. Grosse MP3/WAV-Dateien und Modellgewichte bleiben lokal.

`training/`
: Alles, was durch Training, Bewertung oder Generierung entsteht:
Checkpoints, LoRA-Staende, Review-Audios, Longform-Audios und Reports.

`dokumentation/`
: Erklaertexte fuer Struktur, Ablauf, Code und lokales Setup.

## Wichtige aktive Audio-Pfade

`code/start.py`
: Zentraler Einstieg. Ohne Argumente fragt die Datei nach Laenge und Genre und
startet die MusicGen-Audioerzeugung mit aktivem LoRA-Adapter.

`code/src/Crawler/`
: Suche, Top-10-Quellen, YouTube-/MP3-Import. Der Crawler ist bewusst getrennt
von Training und Audioerzeugung.

`code/src/Dataset/`
: Clip-Erstellung, Genre-Zuordnung, LoRA Zieldaten und gepruefte
Trainingsdaten.

`code/src/Merkmale/`
: Eigener Bereich fuer technische Audio-Merkmale. Hier werden aus den
Dataset-Clips messbare Werte wie Dauer, RMS-Lautheit, Peak-Lautheit und
Zero-Crossing-Rate extrahiert.

`code/src/Training/`
: LoRA-Training, Testaudio-Erstellung, Longform-Generierung, Looping,
Audio-Pruefung und optionale Nachbearbeitung.

`daten/processed/trainingsdaten_lora/`
: Vollstaendiges genrebalanciertes Basisdataset mit 5000 Clips.

`daten/processed/trainingsdaten_geprueft/`
: Aktives LoRA-Trainingsdataset. Es enthaelt 5000 Clips und schliesst Quellen
aus, die in der menschlichen Review als schlecht markiert wurden.

`daten/processed/musicgen_youtube_import_30s/`
: Einzelne importierte MP3-Quellen als geschnittene 30s-Clip-Datasets.

`daten/modelle/musicgen/facebook_musicgen_melody_large/`
: Lokales MusicGen-Basismodell.

`daten/modelle/audio_analyse/`
: Lokale Analysemodelle, z.B. CLAP/Demucs, fuer Genre- und Qualitaetspruefung.

`training/musicgen/lora/`
: Aktiver freigegebener LoRA-Stand. `adapter.pt` zeigt auf den aktuell
freigegebenen Adapter.

`training/musicgen/lora_neustart/`
: Neuer sauberer LoRA-Lauf ab Basismodell. Dieser Lauf ist fuer den kontrollierten
Neustart in 500er-Schritten vorgesehen.

`training/bewertungen/musicgen/`
: Menschliche Bewertungen, Testaudios und Review-CSV-Dateien.

`training/ausgaben/musicgen_generiert/`
: Frisch mit MusicGen erzeugte lange Audios.

`training/ausgaben/musicgen_loops/`
: Loop-basierte Longform-Audios aus vorhandenen Clips.

## Altbestand

Im Projektroot gibt es noch Altbereiche wie `Bachelorarbeit/` und
`bewertungen/`. Sie gehoeren nicht mehr zur neuen kompakten Audio-Struktur.
Solange nicht ausdruecklich freigegeben, werden sie nicht geloescht, weil dort
noch alte Experimente, Bewertungen oder fremde Projektteile liegen koennen.

## Git-Regel

Code, kleine CSV-/JSON-Reports und Dokumentation koennen versioniert werden.
Grosse MP3/WAV-Dateien, Rohdaten, Modellgewichte und Checkpoints bleiben lokal,
ausser eine konkrete Audio soll bewusst nach GitHub gepusht werden.
