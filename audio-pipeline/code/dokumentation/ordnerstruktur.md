# Ordnerstruktur

Ziel: Neue Dateien sollen vorhersehbar landen, damit Training, Bewertung und Video-Erstellung nicht durcheinander geraten.

## Hauptprinzip

Der allgemeine Projektordner `Bachelor_VisiualStudio/` ist der aktive Projektordner. Die sichtbare Arbeit ist in sieben einfache Bereiche getrennt:

- `code/`
- `daten/`
- `training/`
- `bewertungen/`
- `ausgaben/`
- `modelle/`
- `archiv/`

Versteckte technische Ordner wie `.git`, `.venv` oder `.claude` bleiben daneben bestehen, werden aber nicht als Arbeitsbereiche genutzt.

## Code

- `code/src/Training/`: MusicGen-Training, Generierung, Review-Kampagnen, Checkpoint-Export.
- `code/src/Dataset/`: Audiodaten pruefen, schneiden, validieren, Audits.
- `code/src/Crawler/`: Download- und Crawling-Skripte.
- `code/src/Video/`: Video-, Visual- und LTX/ComfyUI-Pipelines.
- `code/src/Generation/`: Generierungslogik, sofern nicht direkt MusicGen-Training.
- `code/scripts/`: ausfuehrbare Hilfsskripte und Tests.
- `code/configs/`: lokale `.env`-Dateien und Konfigurationen.
- `code/packages/`: lokale Python-Pakete fuer Lo-Fi-Domain-Logik.
- `code/tools/`: externe Tools wie ComfyUI, Deno und LTX.
- `code/website/`: Frontend-Quellcode fuer die lokale Weboberflaeche.
- `code/werkzeuge/`: lokale Hilfsprogramme und FFmpeg/AudioCraft-Fallbacks.
- `code/dokumentation/`: kurze technische Notizen.

Nicht mehr aktive Einstiegspunkte liegen im Archiv unter `archiv/`.

## Daten

- `daten/raw/`: Rohdaten, z.B. MP3, YouTube-Metadaten, Rohvideos.
- `daten/processed/`: fertige Datensaetze fuer Training oder Review.
- `daten/metadata/`: Listen, Download-Reports, Cleanup-Reports.
- `daten/features/`: extrahierte Features.

Regel: Rohdaten werden nicht manuell in `training/` abgelegt. Verarbeitete Trainingsdaten gehoeren nach `daten/processed/`.

## Training und Reviews

- `training/musicgen/`: echte MusicGen-Trainingslaeufe und Checkpoints.
- `bewertungen/musicgen/`: kleine Audio-Pakete fuer menschliche Bewertung.
- `bewertungen/status/`: Status und Prompt-Banks fuer die 200-Bewertungen-Kampagne.
- `training/downloads/`: Download-Queues und Download-Logs.

Review-Regel: In neuen Review-Batches liegt pro Audio nur eine MP3 in `audio/`. `bewertung.csv` ist die Datei, die bewertet wird.

## Video

- `training/video/`: Video-Trainings- und Generierungslaeufe.
- `bewertungen/video/`: Video-Review-Pakete.
- `ausgaben/generated_video/`: finale oder manuell erzeugte Video-Ausgaben.
- `ausgaben/generated_visuals/`: Bilder, GIFs, Thumbnails und visuelle Datensaetze.

## Modelle und Fremdcode

- `modelle/`: Modell-Caches, exportierte Fine-Tunes, HuggingFace/Torch-Cache.
- `code/werkzeuge/`: lokale Hilfsprogramme und Fremdcode, z.B. FFmpeg und die eingefrorene AudioCraft-Version.
- `code/tools/`: grosse externe Tools im Workspace, z.B. ComfyUI oder LTX. Diese nicht in `code/src/` mischen.

## Altlasten

- `archiv/`: bewusst archivierte alte Projektteile.
- `archiv/old_search_entrypoint/`: alter `Suchen`-Einstieg.
- `archiv/video_training_pipeline_20260519/`: alte Video-Trainingspipeline plus Szenarien.
- `archiv/workspace_archives/`: alte Root-Arbeitsbereiche.
- Root-Ordner ausserhalb dieser Struktur sollen vermieden werden.

Wenn alte Dateien wirklich gebraucht werden, zuerst nach `archiv/` verschieben oder kontrolliert in die neue Struktur uebernehmen.

## Git-Regel

Nur kleine Review-Audios, Ratings, Code und Dokumentation pushen. Grosse Caches, WAV-Zwischenprodukte, Checkpoints und temporaere Outputs bleiben lokal, ausser sie werden explizit gebraucht.
