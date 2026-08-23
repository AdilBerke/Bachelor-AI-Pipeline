# MusicGen- und LTX-Lo-Fi-Projekt

Stand: 2026-08-22

Diese README ist der aktuelle Einstiegspunkt und das Arbeitsgedaechtnis fuer
das Projekt. Sie fasst die wichtigsten Entscheidungen aus Code, Dokumentation
und bisherigen Sitzungsnotizen zusammen, damit neue Arbeit nicht wieder bei
Null beginnt.

Wichtig: Einige Detaildokumente sind aelter und koennen in Einzelfragen
ueberholt sein. Wenn README, Code und alte Notizen voneinander abweichen, gilt
zuerst der aktuelle Code, danach diese README, danach die Detaildokumente.

**Projekt nachbauen:** Schritt-fuer-Schritt-Anleitung (Code, Basismodelle,
trainiertes LoRA, Frontend, Backend) in [`NACHBAUANLEITUNG.md`](NACHBAUANLEITUNG.md).

## Kurzueberblick

Das Repository enthaelt zwei fachlich getrennte, aber thematisch verbundene
Bachelor-/Masterarbeits-Pipelines:

| Bereich | Zweck | Aktueller Ort |
|---|---|---|
| Audio | Lo-Fi-Musik mit MusicGen Melody Large, LoRA, Longform-Generierung und Bewertung | `code/`, `daten/`, `training/`, `dokumentation/aktuell/` |
| Video/GIF | Lo-Fi-Anime-Loops mit LTX-Video 13B, LoRA, Szenarien, GIF/MP4-Postprocessing | `Bachelorarbeit/lofi_pipeline/`, `Bachelorarbeit/pipeline/`, `dokumentation/video_gif/` |

Die beiden Pipelines duerfen nicht vermischt werden. Die Audio-Pipeline ist der
aktive Hauptfokus im Root-Projekt. Die Video/GIF-Pipeline ist ein separater
Projektteil mit eigener Struktur, eigenen Skripten und eigenen Trainingslaeufen.

## Wichtigste Projektregeln

- Rechenintensive Trainings, Downloads und lange Generierungen werden nicht
  automatisch gestartet.
- Grosse Audiodateien, Rohdaten, Modellgewichte und Checkpoints bleiben lokal,
  ausser sie werden ausdruecklich freigegeben.
- Bestehende nicht selbst erzeugte Aenderungen im Git-Working-Tree nicht
  zuruecksetzen.
- Checkpoints sind Kandidaten, bis sie bewertet und bewusst freigegeben wurden.
- Alte Ordner wie `bewertungen/` oder Teile unter `Bachelorarbeit/` koennen
  Altdaten oder fremde Projektteile enthalten und werden nicht ungefragt
  geloescht.

## Aktuelle Struktur

| Pfad | Bedeutung |
|---|---|
| `code/start.py` | zentraler CLI-Einstieg fuer Audio-Pipeline, Status, Suche, Dataset, Training, Bewertung und Generierung |
| `code/src/Crawler/` | Quellen suchen, Top-10-Listen und Audio-/MP3-Import |
| `code/src/Dataset/` | Clip-Auswahl, Genre-Regeln, Zieldatensatz, Qualitaetspruefung |
| `code/src/Merkmale/` | technische Audio-Feature-Extraktion |
| `code/src/Training/` | MusicGen-Generierung, LoRA-Training, Looping, Bewertung, Referenzvergleich |
| `code/src/Pipeline/web_api.py` | lokale HTTP-API fuer Studio/Website-Jobs |
| `daten/processed/lora_training/` | aktives MusicGen-LoRA-Trainingsdataset |
| `training/musicgen/lora_training/` | aktiver LoRA-Lauf bzw. freigegebener Adapterstand |
| `training/ausgaben/musicgen_generiert/` | generierte Longform-Audios und Reports |
| `training/bewertungen/musicgen/` | Testaudio-, Longform- und Review-Bewertungen |
| `dokumentation/aktuell/` | aktuelle Audio-Dokumentation fuer Struktur, Ablauf, Code, Setup und Prozess |
| `Bachelorarbeit/lofi_pipeline/` | echte Video/GIF-Pipeline mit LTX-Video-Szenarien |
| `tools/LTX-Video-Trainer/` | eingebundener LTX-Video-Trainer |
| `tools/rife-ncnn-vulkan/` | Frame-Interpolation fuer Video/GIF-Postprocessing |

## Audio-Pipeline

Ziel der Audio-Pipeline ist eine reproduzierbare lokale Kette:

1. Quellen suchen oder importieren.
2. Audio in standardisierte 30-Sekunden-Clips ueberfuehren.
3. Clips technisch pruefen und genrebezogen organisieren.
4. Ein source-disjoint, genrebalanciertes LoRA-Dataset bauen.
5. MusicGen Melody Large ueber LoRA anpassen.
6. Testaudios erzeugen und bewerten.
7. Kurze MusicGen-Kandidaten zu Longform-Audios verbinden.
8. Uebergaenge, Loop-Stabilitaet und technische Qualitaet pruefen.
9. Finale Audios mit Referenzstandards vergleichen.
10. Reports fuer die wissenschaftliche Dokumentation sichern.

Aktuelle Zielgenres stehen in `code/configs/lora_genres.json`:

- Jazz Lofi
- Chillhop Lofi
- Dreamy Lofi
- Study Lofi
- Guitar Lofi

Das Basismodell ist `facebook/musicgen-melody-large`. Lokale Modellpfade und
Standardwerte sind in `code/configs/konfiguration.yaml` dokumentiert. Der
aktive Adapterpfad ist `training/musicgen/lora_training/adapter.pt`.

## Aktueller MusicGen-Stand

Die Versuchsdokumentation bis 2026-08-19 zeigt: Mehr Training oder aggressivere
Hyperparameter haben nicht automatisch bessere Musik ergeben. Mehrere Versuche
wurden als Regression archiviert.

Statuscheck am 2026-08-22:

- LoRA-Dataset: 5000/5000 Clips, keine fehlenden Clips.
- LoRA-Adapter: bereit, `step_000625`, Typ `lora`.

Wichtige Erkenntnisse:

- Der fruehe Checkpoint `step_000625` war lange der stabilste dokumentierte
  Adapterstand.
- Ein 3000-Schritte-Lauf verschlechterte mehrere Genres deutlich und wurde
  archiviert.
- Genre-Melody-Conditioning mit echten Referenzclips ist im Code als Option
  vorhanden, aber produktiv deaktiviert.
- Eine FMA-Datensatzerweiterung brachte mehr Quellenvielfalt, verschlechterte
  aber die Bewertung und wurde zurueckgesetzt bzw. archiviert.
- Die Bewertungsmethode hat Eigenrauschen. Einzelmessungen reichen nicht fuer
  belastbare Modellvergleiche; spaetere Vergleiche nutzen Wiederholungen.
- Die technische Audioqualitaet kann einen Deckeneffekt haben, weil grobe
  Defekte bereits waehrend der Generierung herausgefiltert werden.
- Neuere Experimente liegen unter anderem in
  `training/musicgen/lora_training_rank16_test/` und archivierten
  `training/musicgen/lora_training_archiv_*`-Ordnern.

Details: `MUSIKMODELL_VERSUCHSDOKUMENTATION.md`.

## Bewertungssystem

Der aktuelle Code in `code/src/Training/audio_bewertung.py` berechnet acht
Score-Kategorien:

1. Technische Audioqualitaet
2. Musikalische Kohaerenz
3. Genre-Treue
4. Uebergangsqualitaet
5. Referenzaehnlichkeit
6. Stille-/Aktivitaetsanteil
7. Rhythmische Stabilitaet
8. Klangfarbenbalance

Aeltere Notizen sprechen teilweise noch von fuenf Kategorien. Das ist fuer den
aktuellen Code veraltet. Falls Frontend oder Berichte noch fuenf Kategorien
erwarten, muss diese Schnittstelle vor Aenderungen geprueft werden.

Wichtige Bewertungsdateien pro Longform-Lauf:

| Datei | Inhalt |
|---|---|
| `score_bewertung.json` | Score-Daten fuer die generierte Audio |
| `score_bewertung.html` | visuelle Auswertung |
| `generation_report.json` | reproduzierbarer Generierungsreport |
| `finale_audio_info.json` | technischer Abschlussreport |
| `kandidaten_score.csv` | Bewertung einzelner Kandidaten |
| `uebergangs_pruefung.csv` | Uebergangsdiagnose |
| `bericht.md` | lauffaehiger Kurzbericht |

## Longform-Generierung

Longform-Audios entstehen nicht durch blindes Aneinanderhaengen. Pro Abschnitt
werden mehrere Kandidaten generiert, technisch bewertet, gegen den vorherigen
Block geprueft und erst danach geloopt bzw. per Crossfade verbunden.

Typische aktive Parameter aus `code/start.py`:

- Ziel-BPM: 78
- BPM-Toleranz: 4
- Kandidaten pro Abschnitt: 3
- MusicGen-Temperatur: 0.72
- Top-k: 80
- CFG: 4.0
- VRAM-Limit: 80 Prozent
- Live-Status: 1 Sekunde
- Genre- und MP3-Referenzpruefung aktiv

Die wichtigsten Skripte sind:

- `code/src/Training/audio_erstellen.py`
- `code/src/Training/audio_loopen.py`
- `code/src/Training/clips_loopen.py`
- `code/src/Training/referenz_vergleich.py`

## Lokale Befehle

Status pruefen:

```bash
.venv/bin/python code/start.py --status
```

Genres anzeigen:

```bash
.venv/bin/python code/start.py --genres-anzeigen
```

Top-10-Quellen fuer ein Genre suchen:

```bash
.venv/bin/python code/start.py --top10 --genre "Jazz Lofi"
```

Dataset vorbereiten:

```bash
.venv/bin/python code/start.py --clips-5000
```

LoRA trainieren oder fortsetzen:

```bash
.venv/bin/python code/start.py --lora-training
.venv/bin/python code/start.py --lora-fortsetzen
.venv/bin/python code/start.py --lora-weitere-500
```

Testaudios erzeugen:

```bash
.venv/bin/python code/start.py --testaudios
```

Kurzen Planlauf ohne Modellstart schreiben:

```bash
.venv/bin/python code/src/Pipeline/projekt_ablauf.py \
  --stufen audio_generieren \
  --dauer 5m \
  --genre "Chillhop Lofi" \
  --nur-plan
```

Echte kurze Longform-Generierung:

```bash
.venv/bin/python code/src/Pipeline/projekt_ablauf.py \
  --stufen audio_generieren \
  --dauer 5m \
  --genre "Chillhop Lofi" \
  --kein-github-push
```

Lokale API starten:

```bash
.venv/bin/python code/src/Pipeline/web_api.py
```

Standard-API-Adresse:

```text
http://127.0.0.1:8000
```

## Website / Studio

Die Studio-Website wurde in frueheren Sitzungen als getrenntes Frontend
beschrieben. In den aktuellen Root-Dateien liegt kein `code/website/`-Ordner
mehr. Die letzten Sitzungsnotizen nennen als Frontend-Ort:

```text
/tmp/lo-fi-harmony-forge-newweb
```

Das ist ein separates Git-Repo auf Branch `NewWeb` und kann nach einem Neustart
fehlen, weil es unter `/tmp` liegt. Vor Arbeiten an der Website muss deshalb
zuerst geprueft werden, ob dieses Repo noch existiert oder neu geklont werden
muss.

Node ist in frischen Shells nicht sicher im `PATH`. Der zuletzt bekannte
Node-Pfad war:

```text
/home/BA_Musikproduktion/.cache/lo-fi-dreamer-node/node-v22.22.3-linux-x64/bin
```

Die Website sollte nur echte lokale Daten anzeigen. Fruehere Mock-/Landingpage-
Sektionen mit erfundenen Werten wurden entfernt oder sollten entfernt bleiben.

## Video/GIF-Pipeline

Die echte Video/GIF-Pipeline liegt unter:

```text
Bachelorarbeit/lofi_pipeline/
```

Aktueller Stand aus `VIDEO_PIPELINE_KONTEXT.md`:

- Modell: `LTXV_13B_097_DEV`, nicht mehr 2B.
- Training: `Bachelorarbeit/lofi_pipeline/scripts/train_lora.py`.
- Generierung: `Bachelorarbeit/lofi_pipeline/scripts/generate_samples.py`.
- Feedback-UI: `Bachelorarbeit/lofi_pipeline/scripts/feedback_ui.py`, Port 7860.
- Postprocessing: `make_gif.py`, RIFE/minterpolate, ESRGAN, GIF und MP4.
- Bewertung: `evaluate_video.py` unter `Bachelorarbeit/pipeline/realistic_rabbit/`.
- Szenarien liegen unter `Bachelorarbeit/lofi_pipeline/scenarios/`.
- Kein Szenario gilt sicher als final fertig trainiert.
- `rabbit_lake` ist weit fortgeschritten, hatte aber CUDA-OOM und bekannte
  Prompt-/Objektprobleme.

Die Video-Seite im Studio war zuletzt eher Design/Mockup und nicht sauber an
diese echte Pipeline angebunden.

## Dokumentationslandkarte

| Dokument | Wofuer lesen? |
|---|---|
| `README.md` | aktueller Einstieg und Arbeitsgedaechtnis |
| `CLAUDE.md` | alte Sitzungsnotizen zur Studio-Website und API, nuetzlich aber teils veraltet |
| `MUSIKMODELL_VERSUCHSDOKUMENTATION.md` | chronologische MusicGen-LoRA-Versuche und Messmethodik |
| `VIDEO_PIPELINE_KONTEXT.md` | technische Referenz der LTX-Video-Pipeline |
| `dokumentation/aktuell/01_struktur.md` | Audio-Projektstruktur |
| `dokumentation/aktuell/02_ablauf.md` | Audio-Ablauf |
| `dokumentation/aktuell/03_code_erklaerung.md` | Code-Erklaerung |
| `dokumentation/aktuell/04_einrichtung.md` | lokales Setup |
| `dokumentation/aktuell/05_prozessdokumentation_musicgen.md` | wissenschaftliche Prozessbeschreibung |
| `dokumentation/video_gif/` | Video/GIF-Dokumentation |

## Offene Punkte

- Bewertungsdokumentation und Frontend ggf. von fuenf auf acht Score-Kategorien
  nachziehen.
- Bei neuen Experimenten erneut pruefen, ob `adapter.pt` weiterhin auf den
  bewusst freigegebenen Stand zeigt.
- Website-Repo unter `/tmp/lo-fi-harmony-forge-newweb` verifizieren oder neu
  herstellen.
- Cookie-basierte YouTube-Importe (`cookies-from-browser`) nur nach bewusster
  Entscheidung einbauen.
- Video/GIF-Studio-Anbindung erst nach genauer Pruefung der echten
  `Bachelorarbeit/lofi_pipeline/`-Schnittstellen bauen.
- Fuer Modellvergleiche keine Einzelmessungen mehr als harte Entscheidung
  verwenden; Wiederholungen und Standardabweichung dokumentieren.

## Naechster sinnvoller Arbeitsstart

Vor jeder neuen Arbeit:

1. `git status --short` ansehen.
2. Diese README lesen.
3. Bei Audioarbeit `MUSIKMODELL_VERSUCHSDOKUMENTATION.md` und
   `dokumentation/aktuell/05_prozessdokumentation_musicgen.md` querpruefen.
4. Bei Videoarbeit `VIDEO_PIPELINE_KONTEXT.md` lesen.
5. Danach erst Code oder lange Jobs anfassen.

Schneller technischer Check:

```bash
.venv/bin/python code/start.py --status
```
