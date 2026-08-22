# MusicGen-LoRA-Pipeline zur automatisierten Lo-Fi-Musikgenerierung

Stand: 2026-08-15

## 1. Zielsetzung

Dieses Projekt implementiert eine lokale Pipeline zur automatisierten Generierung,
Bewertung und Verwaltung von Lo-Fi-Musik auf Basis von MusicGen und LoRA. Der
Anwendungsfall ist eine Bachelorarbeit zur Frage, wie Lo-Fi-Musikdaten
systematisch erfasst, aufbereitet, modellseitig angepasst und anschließend als
lange, bewertbare Audiodateien erzeugt werden können.

Die Pipeline folgt dabei einem reproduzierbaren Ablauf:

1. Quellen suchen oder importieren.
2. Rohdaten in standardisierte Audioclips überführen.
3. Audio-Merkmale extrahieren und Qualitätskriterien prüfen.
4. Ein genrebalanciertes Trainingsdataset für LoRA aufbauen.
5. MusicGen mit LoRA lokal anpassen oder fortsetzen.
6. Testaudios erzeugen und bewerten.
7. Aus kurzen MusicGen-Clips lange Audiodateien erzeugen.
8. Die erzeugten Audios automatisch auswerten.
9. Alle relevanten Schritte über eine lokale Website steuerbar machen.
10. Prozess- und Ergebnisberichte für die wissenschaftliche Dokumentation sichern.

Lange Trainingsläufe und externe Downloads werden nicht automatisch gestartet.
Rechenintensive Schritte müssen bewusst über CLI oder Website ausgelöst werden.

## 2. Projektstruktur

Die aktive Projektstruktur ist kompakt gehalten:

| Ordner | Funktion |
|---|---|
| `code/` | Quellcode der lokalen Pipeline, API, Trainings- und Auswertungsmodule |
| `daten/` | Rohdaten, verarbeitete Clips, Features, lokale Modelle und Metadaten |
| `training/` | LoRA-Läufe, Ausgaben, Bewertungen, Reports und Checkpoints |
| `dokumentation/` | Begleitdokumentation, Struktur- und Verlaufsnotizen |
| `code/website/lo-fi-dreamer/` | Lokale Website zur Steuerung der Audio- und Video/GIF-Funktionen |

Der zentrale Einstiegspunkt ist:

```bash
.venv/bin/python code/start.py
```

Die lokale API wird gestartet mit:

```bash
.venv/bin/python code/src/Pipeline/web_api.py
```

Die Website wird im Frontend-Ordner gestartet:

```bash
cd code/website/lo-fi-dreamer
npm run dev
```

## 3. Genres und Zielprofile

Die LoRA-Genres sind zentral in `code/configs/lora_genres.json` definiert.
Derzeit werden folgende Zielklassen verwendet:

| Genre | Funktion im Projekt |
|---|---|
| Jazz Lofi | harmonisch geprägte Lo-Fi-Musik mit Rhodes, Piano oder Jazz-Anmutung |
| Chillhop Lofi | klassischer Lo-Fi-/Hip-Hop-Groove mit ruhigem Beat |
| Dreamy Lofi | weiche, atmosphärische Lo-Fi-Klangflächen |
| Study Lofi | stabile, unaufdringliche Musik für konzentriertes Arbeiten |
| Guitar Lofi | Lo-Fi-Musik mit Gitarrenanteilen und warmem Klangbild |

Diese Genreprofile werden für Quellensuche, Dataset-Aufbau, Prompt-Erzeugung,
Referenzvergleich und Website-Auswahl verwendet.

## 4. Datenerfassung

Die Datenerfassung ist vom eigentlichen Trainings- und Generierungsablauf
getrennt. Dadurch kann die Pipeline lokal mit bereits vorhandenen Daten
arbeiten, ohne bei jedem Lauf neue Downloads auszulösen.

Relevante Module:

| Datei | Aufgabe |
|---|---|
| `code/src/Crawler/quellen_suche.py` | Suche, Import und MP3-Verarbeitung einzelner Quellen |
| `code/src/Crawler/quellen_finden.py` | automatisierte Suche fehlender Quellen pro Genre |
| `code/src/Pipeline/clips_vorbereiten.py` | Nutzung von Top-Quellen zur Clip- und Dataset-Vorbereitung |

Die Quellensuche dokumentiert Suchbegriffe, gefundene Kandidaten, Top-Auswahl
und Ausschlussgründe. Manuell importierte Quellen werden separat markiert, damit
sie nicht unbeabsichtigt gelöscht oder mit automatischen Suchläufen verwechselt
werden.

## 5. Datenaufbereitung

Die Datenaufbereitung standardisiert Roh-Audiodateien in für MusicGen geeignete
Trainingsclips. Ziel sind kurze, vergleichbare Abschnitte mit stabiler Dauer,
einheitlicher Samplerate und verwertbarer Qualität.

Wichtige Schritte:

1. Importierte MP3/WAV/M4A-Dateien werden lokal erfasst.
2. Audios werden in Clips zerlegt.
3. Clips werden nach Genre sortiert.
4. Duplikate und problematische Quellen werden ausgeschlossen.
5. Das LoRA-Zieldataset wird genrebalanciert aufgebaut.
6. Split-Informationen werden in `train`, `valid` und `test` abgelegt.

Relevante Module:

| Datei | Aufgabe |
|---|---|
| `code/src/Dataset/zieldatensatz.py` | Aufbau des genrebalancierten Ziel-Datasets |
| `code/src/Dataset/trainingsdaten_pruefen.py` | Qualitätsfilterung und Ausschluss schlechter Clips |
| `code/src/Dataset/datensatz.py` | Validierung und Audit vorhandener Dataset-Strukturen |
| `code/src/Dataset/genre_regeln.py` | Regeln für Genre-Zuordnung und Quellenqualität |

Das aktive LoRA-Dataset liegt typischerweise unter:

```text
daten/processed/lora_training/
```

## 6. Feature Extraction und Qualitätsprüfung

Die Feature Extraction erzeugt technische Audio-Merkmale, die für Bewertung,
Filterung und wissenschaftliche Auswertung genutzt werden.

Geprüft werden unter anderem:

- Dauer und Samplerate
- Lautheit und RMS-Werte
- stille oder leise Sekunden
- Clipping-Risiko
- Bassanteil
- Höhen-/Shaker-Anteil
- Snare-/Präsenzbereich
- Tonalitäts- und Signaltonrisiko
- BPM-Schätzung
- Energieverteilung

Relevante Module:

| Datei | Aufgabe |
|---|---|
| `code/src/Merkmale/audio_merkmale.py` | Extraktion technischer Merkmale |
| `code/src/Training/audio_bewertung.py` | automatische Bewertung erzeugter Audios |
| `code/src/Training/referenz_vergleich.py` | Vergleich generierter Audios mit Genre-Referenzen |

Diese Werte dienen nicht als vollständiger Ersatz für menschliche Bewertung,
sondern als reproduzierbarer technischer Prüfrahmen.

## 7. LoRA-Modellanpassung

LoRA ist der zentrale Mechanismus zur Anpassung von MusicGen. Das Projekt nutzt
kein vollständiges Neutraining des Basismodells, sondern trainiert adapterartige
Gewichte, die später in die MusicGen-Generierung geladen werden.

Relevante Module:

| Datei | Aufgabe |
|---|---|
| `code/src/Training/lora.py` | aktives LoRA-Training und Fortsetzen von Läufen |
| `code/src/Training/musicgen_steuerung.py` | MusicGen-nahe Trainings- und Review-Steuerung |
| `code/start.py` | sichere CLI-Abstraktion für Training, Fortsetzen und Freigabe |

Wissenschaftlich wichtig ist die Trennung zwischen Kandidaten-Checkpoint und
freigegebenem Checkpoint:

- Zwischencheckpoints werden nicht automatisch für die finale Audioerzeugung
  verwendet.
- Ein Checkpoint muss bewertet und bewusst freigegeben werden.
- Trainingsabbrüche sollen vorhandene Checkpoints nicht zerstören.
- GPU-Nutzung wird standardmäßig auf maximal 80 Prozent begrenzt.

Typische Befehle:

```bash
.venv/bin/python code/start.py --lora-training
.venv/bin/python code/start.py --lora-fortsetzen
.venv/bin/python code/start.py --lora-freigeben --checkpoint PFAD
```

## 8. Testaudios

Nach LoRA-Trainingsläufen können Testaudios pro Genre erzeugt werden. Diese
dienen als kontrollierte Vergleichsbasis für die menschliche und automatische
Bewertung.

Zielgenres:

- Jazz Lofi
- Chillhop Lofi
- Dreamy Lofi
- Study Lofi
- Guitar Lofi

Relevante Module:

| Datei | Aufgabe |
|---|---|
| `code/src/Training/bewertung_audios_erstellen.py` | Testaudio-Generierung und Review-Struktur |
| `code/src/Training/testaudios_sammeln.py` | Zusammenführen von Testaudios |
| `code/src/Training/trainingsclips_bewertung.py` | Bewertung echter Trainingsclips |

Ausgaben werden unter `training/bewertungen/musicgen/` abgelegt.

## 9. Longform-Audioerzeugung

MusicGen erzeugt im Projekt kurze Kandidatenclips. Die Longform-Audio entsteht
nicht durch einfaches Aneinanderhängen dieser Clips, sondern durch einen
mehrstufigen Auswahl- und Loopingprozess.

Aktueller Ablauf:

1. Für einen geplanten Rhythmusblock wird ein 30-Sekunden-Kandidat erzeugt.
2. Pro Block werden mehrere Kandidaten erzeugt.
3. Jeder Kandidat wird technisch geprüft.
4. Der Übergang zum vorherigen Block wird bewertet.
5. Nur Kandidaten mit akzeptabler Technik und akzeptablem Übergang werden
   zugelassen.
6. Der beste Kandidat wird auf die Blocklänge geloopt.
7. Zwischen Blöcken wird ein kurzer Crossfade gesetzt.
8. Am Anfang der finalen Audio wird ein Fade-in gesetzt.
9. Am Ende der finalen Audio wird ein Fade-out gesetzt.
10. Die finale Audio wird automatisch bewertet und dokumentiert.

Relevante Module:

| Datei | Aufgabe |
|---|---|
| `code/src/Training/audio_erstellen.py` | aktive MusicGen-Longform-Generierung |
| `code/src/Training/audio_loopen.py` | robuste Loop-Analyse und Loop-Block-Erzeugung |
| `code/src/Training/clips_loopen.py` | Fallback aus vorhandenen bewerteten Clips |

Die Website unterstützt feste und freie Längen:

- 5 min
- 15 min
- 20 min
- 30 min
- 1 h
- 3 h
- freie Eingabe

Die Rhythmusblocklänge wird prozentual und über eine Mindestlänge bestimmt. Für
eine Stunde entspricht ein Wechsel etwa alle drei Minuten, sofern keine manuelle
Blockdauer gesetzt wird.

## 10. Übergangslogik

Die Übergangslogik ist entscheidend, weil ein technisch guter Clip musikalisch
ungeeignet sein kann, wenn er schlecht an den vorherigen Block anschließt.

Für jeden Kandidaten werden unter anderem geprüft:

- BPM-Ähnlichkeit
- Lautheit am Ende von Block A und Anfang von Block B
- Bass-Ähnlichkeit
- Snare-/Präsenz-Ähnlichkeit
- Höhen-/Shaker-Ähnlichkeit
- Energie-Sprung
- harmonische bzw. timbrale Nähe
- Rhythmus- und Energieverlauf
- abrupter Drop
- Stille am Anfang oder Ende
- abgeschnittener Ton
- Klickrisiko an der Clipkante

Ein harter Übergangsfehler führt dazu, dass der Kandidat nicht akzeptiert wird.
Damit gewinnt nicht automatisch der isoliert beste Clip, sondern der Clip mit
dem besten musikalischen Anschluss im Kontext der Longform-Audio.

## 11. Automatische Bewertung

Nach der Erzeugung wird die finale Audio automatisch bewertet. Die Bewertung
liefert fünf Hauptscores:

1. Technische Audioqualität
2. Musikalische Kohärenz
3. Genre-Treue
4. Übergangsqualität
5. Referenzähnlichkeit

Die generierte Audio wird gegen einen gespeicherten Genrestandard verglichen.
Dabei werden zwei getrennte Score-Darstellungen erzeugt:

- Genrestandard
- Generierte Audio

Beide Darstellungen verwenden dieselben Kategorien, dieselbe Reihenfolge und
eine Skala von 0 bis 100. Im Hauptbereich werden keine Rohkurven wie
Spektrogramme, RMS-Kurven oder BPM-Verläufe angezeigt.

Relevante Dateien pro Longform-Lauf:

| Datei | Inhalt |
|---|---|
| `finale_audio_info.json` | vollständiger technischer Abschlussreport |
| `generation_report.json` | reproduzierbarer Generierungsreport |
| `score_bewertung.json` | Fünf-Score-Bewertung |
| `score_bewertung.html` | visuelle Score-Auswertung |
| `kandidaten_score.csv` | Bewertung aller Kandidaten |
| `uebergangs_pruefung.csv` | finale Übergangsprüfung |
| `ablauf.csv` | zeitlicher Aufbau der Longform-Audio |
| `bericht.md` | wissenschaftlicher Kurzbericht des Laufs |

## 12. Website und lokale API

Die Website dient als lokale Bedienoberfläche für zentrale Pipelinefunktionen.
Sie ist kein extern gehosteter Dienst, sondern verbindet sich mit der lokalen
API.

API:

```bash
.venv/bin/python code/src/Pipeline/web_api.py
```

Standard-Adresse:

```text
http://127.0.0.1:8000
```

Wichtige API-Funktionsbereiche:

- Status
- Quellen
- Top-Quellen
- Clips
- LoRA-Status
- LoRA-Training und Fortsetzen
- Testaudios
- Longform-Generierung
- Audio-Bewertung
- Reports
- Audio-Download
- Video/GIF-Galerie

Die Website zeigt und startet lokale Jobs, ohne automatisch lange Trainings oder
Downloads auszulösen.

## 13. Video- und GIF-Integration

Die Video/GIF-Funktionen sind als separater Projektbereich eingebunden. Sie
werden über die Website angezeigt, aber die Audio-Pipeline ist davon fachlich
getrennt.

Relevante Bereiche:

| Ordner/Datei | Aufgabe |
|---|---|
| `Bachelorarbeit/lofi_pipeline/` | bestehende Video-/GIF-Pipeline |
| `dokumentation/video_gif/` | Dokumentation des Video-/GIF-Teils |
| `code/website/lo-fi-dreamer/src/routes/studio.video.tsx` | Website-Integration für Video/GIF |

Die Audio-Dokumentation verändert diesen Bereich nicht. Ziel ist die
Integration in der Oberfläche, nicht die Vermischung der Trainingspipelines.

## 14. Prozessdokumentation

Jeder wichtige Schritt erzeugt oder aktualisiert Reports. Diese Reports sind für
die Bachelorarbeit nutzbar, weil sie Eingaben, Entscheidungen und Ergebnisse
nachvollziehbar machen.

Dokumentiert werden unter anderem:

- verwendete Genres
- Suchbegriffe
- Quellen und Top-Auswahl
- ausgeschlossene Quellen und Clips
- Dataset-Zusammensetzung
- Trainingsstand und Checkpoints
- GPU-Limit
- erzeugte Testaudios
- verwendeter LoRA-Adapter
- Rhythmusblöcke
- Loop-Längen
- Übergänge und verworfene Übergänge
- finale Audio
- automatische Bewertung
- Empfehlung für den nächsten Schritt

Die Texte und Reports sind sachlich formuliert und dienen als Grundlage für die
wissenschaftliche Auswertung.

## 15. Typische Arbeitsabläufe

### 15.1 Status prüfen

```bash
.venv/bin/python code/start.py --status
```

### 15.2 Quellen suchen

```bash
.venv/bin/python code/start.py --top10 --genre "Jazz Lofi"
```

### 15.3 Clips vorbereiten

```bash
.venv/bin/python code/start.py --clips-5000
```

Mit Downloads:

```bash
.venv/bin/python code/start.py --clips-5000 --mit-downloads
```

### 15.4 LoRA trainieren oder fortsetzen

```bash
.venv/bin/python code/start.py --lora-training
.venv/bin/python code/start.py --lora-fortsetzen
```

### 15.5 Testaudios erzeugen

```bash
.venv/bin/python code/start.py --testaudios
```

### 15.6 Longform-Audio erzeugen

Kurzer lokaler Test ohne GitHub-Push:

```bash
.venv/bin/python code/src/Pipeline/projekt_ablauf.py \
  --stufen audio_generieren \
  --dauer 5m \
  --genre "Chillhop Lofi" \
  --kein-github-push
```

Nur planen, ohne Modellstart:

```bash
.venv/bin/python code/src/Pipeline/projekt_ablauf.py \
  --stufen audio_generieren \
  --dauer 5m \
  --genre "Chillhop Lofi" \
  --nur-plan
```

### 15.7 Referenzvergleich ausführen

```bash
.venv/bin/python code/start.py --referenzvergleich
```

## 16. Wissenschaftliche Einordnung

Das Projekt kombiniert generative Modellierung mit einem kontrollierten
technischen Bewertungssystem. Methodisch besteht die Arbeit aus vier Ebenen:

1. Datenebene: Erfassung, Filterung und genrebalancierte Strukturierung von
   Lo-Fi-Audiodaten.
2. Modellebene: Anpassung eines vortrainierten MusicGen-Modells über LoRA.
3. Generierungsebene: Erzeugung kurzer Kandidaten und Konstruktion langer
   Audios über Looping, Crossfades und Übergangsprüfung.
4. Evaluationsebene: automatische Score-Bewertung, Referenzvergleich und
   menschliche Bewertungsunterstützung.

Diese Trennung ist wichtig, weil sie die Ergebnisse interpretierbar macht. Ein
schlechtes Ergebnis kann dadurch einer konkreten Ebene zugeordnet werden:

- Datenproblem
- Genreproblem
- Trainingsproblem
- Generierungsproblem
- Übergangsproblem
- Bewertungs- oder Referenzproblem

## 17. Grenzen des Systems

Die automatische Bewertung ist ein technischer Näherungsansatz. Sie kann
auffällige Fehler erkennen, ersetzt aber keine vollständige musikalische
Beurteilung.

Bekannte Grenzen:

- BPM-Schätzung kann bei Lo-Fi-Material unsicher sein.
- Harmonische Nähe wird lokal über spektrale/timbrale Fingerprints angenähert.
- Genre-Treue hängt von Qualität und Umfang der Referenzen ab.
- Lange Audios müssen weiterhin stichprobenartig menschlich nachgehört werden.
- LoRA-Qualität hängt stark von Dataset-Balance und Clipqualität ab.

## 18. Aktueller nächster sinnvoller Schritt

Der nächste sinnvolle Schritt ist ein kurzer echter 5-Minuten-Testlauf mit einem
freigegebenen LoRA-Adapter:

```bash
.venv/bin/python code/src/Pipeline/projekt_ablauf.py \
  --stufen audio_generieren \
  --dauer 5m \
  --genre "Chillhop Lofi" \
  --kein-github-push
```

Danach sollten folgende Dateien geprüft werden:

- `lange_audio.mp3` oder `lange_audio.wav`
- `bericht.md`
- `score_bewertung.html`
- `uebergangs_pruefung.csv`
- `kandidaten_score.csv`

Diese Prüfung zeigt, ob die erzeugte Audio technisch stabil ist, ob die
Übergänge plausibel bewertet wurden und ob der Lauf als Grundlage für die
Bachelorarbeit verwendet werden kann.
