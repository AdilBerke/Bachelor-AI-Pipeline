# Audio-Pipeline

Erzeugt längere Lo-Fi-Audioausgaben mit MusicGen und einem per LoRA angepassten Adapter.
Enthält außerdem die Datenbeschaffung, die Bewertung der Ergebnisse und das Backend, über
das die Weboberfläche beide Pipelines ansteuert.

Einrichtung und Inbetriebnahme sind in [`../NACHBAUANLEITUNG.md`](../NACHBAUANLEITUNG.md)
beschrieben, ausführlicher in [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).

## Ablauf

```
Nutzereingabe → MusicGen + LoRA → Kandidatenprüfung → Block-Looping
  → Nachbearbeitung → MP3-Ausgabe → Bewertung → Bereitstellung
```

MusicGen erzeugt pro Abschnitt mehrere 30-Sekunden-Kandidaten. Ein kombinierter Strafwert
aus technischen, übergangsbezogenen, blockinternen, semantischen und referenzbezogenen
Anteilen wählt den geeignetsten aus. Dieser wird über Block-Looping auf die Zieldauer
verlängert, nachbearbeitet und als WAV und MP3 abgelegt.

## Bewertung

Die fertige Longform-Ausgabe wird anschließend automatisch in **acht Kategorien** bewertet,
jeweils mit Werten zwischen 0 und 100:

Technische Audioqualität · Musikalische Kohärenz · Genre-Treue · Übergangsqualität ·
Referenzähnlichkeit · Stille-/Aktivitätsanteil · Rhythmische Stabilität · Klangfarbenbalance

Die Werte werden **nicht** zu einem Gesamtscore verdichtet. Für Genre-Treue und
Referenzähnlichkeit dient der am höchsten bewertete Referenzclip des jeweiligen Genres als
Vergleichsmaßstab. Ergebnisse landen als `score_bewertung.json` im Ausgabeordner des Laufs.

Diese abschließende Bewertung ist von der Kandidatenauswahl während der Generierung
getrennt: Erstere bewertet die fertige Datei, letztere entscheidet, welcher kurze Clip
überhaupt weiterverarbeitet wird.

## Wichtige Dateien

| Pfad | Aufgabe |
|---|---|
| `code/src/Training/audio_erstellen.py` | Generierung, Kandidatenauswahl, Longform-Montage |
| `code/src/Training/audio_bewertung.py` | Bewertung in acht Kategorien, Genrestandard |
| `code/src/Training/lora.py` | LoRA-Training und Checkpoint-Freigabe |
| `code/src/Crawler/quellen_suche.py` | Quellensuche und Import (nutzt `yt-dlp`, kein API-Schlüssel) |
| `code/src/Pipeline/web_api.py` | HTTP-Backend auf Port 8000, Jobsteuerung |
| `code/start.py` | Kommandozeilen-Einstiegspunkt |
| `code/configs/konfiguration.yaml` | Basismodell, Adapterpfad, Genres |

## Modellstand

Basismodell `facebook/musicgen-melody-large`, angepasst per LoRA mit Rank 8, Alpha 16 und
Dropout 0,05 — rund 9,44 Millionen trainierbare Parameter bei 625 Trainingsschritten.
Generiert wird mit Temperature 0,72, Top-k 80 und CFG 4,0 bei 32 kHz; das Zieltempo liegt
bei 78 ± 4 BPM.

Ein neu trainierter Checkpoint ersetzt den aktiven Stand nicht automatisch. Er gilt zunächst
als Kandidat und muss nach Erzeugung und Bewertung von Testaudios ausdrücklich freigegeben
werden.

## Hinweis zur semantischen Genre-Prüfung

Der optionale CLAP-Abgleich erreichte in einem gesonderten Test nur rund 40 % Trefferquote
bei der Genrezuordnung. Er wird deshalb im regulären Generierungsweg nicht als
Ausschlusskriterium verwendet.
