# Lokale Einrichtung

Aktueller Fokus: MusicGen-Audio, LoRA-Training und Longform-Erzeugung.

## Umgebung aktivieren

Vom Projektroot aus:

```bash
source .venv/bin/activate
```

Danach koennen alle Befehle auch direkt mit `.venv/bin/python` gestartet
werden.

## Status anzeigen

```bash
.venv/bin/python code/start.py --status
```

Dieser Befehl startet nichts.

## 5000 Clips vorbereiten

Nur lokale MP3s nutzen:

```bash
.venv/bin/python code/start.py --clips-5000
```

Fehlende Quellen suchen und danach Clips vorbereiten:

```bash
.venv/bin/python code/start.py --clips-5000 --mit-downloads
```

## Gepruefte Trainingsdaten bauen

```bash
.venv/bin/python code/start.py --trainingsdaten
```

Diese Trainingsdaten sind die bevorzugte Grundlage fuer LoRA.

## LoRA trainieren

Sauberer Neustart ab Basismodell:

```bash
.venv/bin/python code/start.py --lora-neustart
```

Aktiven LoRA-Lauf fortsetzen:

```bash
.venv/bin/python code/start.py --lora-fortsetzen
```

Bewusst weitere 500 Steps starten:

```bash
.venv/bin/python code/start.py --lora-weitere-500
```

## Testaudios erzeugen

```bash
.venv/bin/python code/src/Training/bewertung_audios_erstellen.py
```

Die finalen MP3s liegen danach im jeweiligen Ordner unter:

```text
training/bewertungen/musicgen/lora_bewertung_XXX/audio/
```

## Lange Audio erzeugen

Interaktiv mit Laenge und Genre:

```bash
.venv/bin/python code/start.py
```

Loop-Fallback mit vorhandener Datei:

```bash
.venv/bin/python code/start.py --audio-loopen --quelle DATEI --dauer 20m
```

## Wichtige Regeln

- Nicht blind weitertrainieren, wenn Testaudios schlechter werden.
- Nach jedem 500er-LoRA-Schritt Testaudios erzeugen und bewerten.
- Grosse Audios, Modelle und Checkpoints bleiben lokal.
- Crawler/Suche nur starten, wenn wirklich neue Quellen gebraucht werden.
- Video/GIF-Dateien nicht anfassen.
