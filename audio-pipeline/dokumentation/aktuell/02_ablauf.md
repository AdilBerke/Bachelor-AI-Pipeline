# Audio-Ablauf

Die Audio-Pipeline besteht aus klar getrennten Schritten. Wichtig: Suche,
Download, Training und Audioerzeugung sind nicht dasselbe.

## 1. Quellen suchen

Ziel: passende YouTube-/MP3-Quellen pro Genre finden.

Start:

```bash
.venv/bin/python code/start.py --top10 --genre "Jazz Lofi"
```

Aktuell werden vor allem diese Genres genutzt:

- `Jazz Lofi`
- `Chillhop Lofi`
- `Dreamy Lofi`
- `Study Lofi`
- `Guitar Lofi`

Die Suche ist inputgesteuert. Bekannte oder sehr aehnliche Videos werden
standardmaessig ausgeschlossen, damit bei einem zweiten Suchlauf eine neue
Top-10 entsteht.

Direkter Mehrgenre-Plan:

```bash
.venv/bin/python code/src/Crawler/quellen_suche.py genres --nur-plan
```

## 2. MP3s importieren und Clips schneiden

Ziel: aus lokalen MP3s 30s-WAV-Clips erstellen und nach Genre sortieren.

Start fuer das LoRA Dataset:

```bash
.venv/bin/python code/start.py --clips-5000
```

Start mit automatischer Suche nach fehlenden Quellen:

```bash
.venv/bin/python code/start.py --clips-5000 --mit-downloads
```

Ergebnis:

```text
daten/processed/trainingsdaten_lora/
```

## 3. Gepruefte Trainingsdaten bauen

Ziel: schlechte Quellen aus der menschlichen Review ausschliessen und trotzdem
5000 Clips behalten.

Start:

```bash
.venv/bin/python code/start.py --trainingsdaten
```

Aktueller Stand:

```text
5000 Clips
800 Jazz Lofi
800 Chillhop Lofi
800 Dreamy Lofi
800 Study Lofi
800 Guitar Lofi
531 Clips aus schlechten Review-Quellen ausgeschlossen
```

Aktive Trainingsdaten:

```text
daten/processed/trainingsdaten_geprueft/
```

## 4. Audio-Merkmale

Ziel: technische Merkmale aus den vorbereiteten WAV-Clips extrahieren. Dieser
Schritt startet kein Training und erzeugt keine Musik.

Start:

```bash
.venv/bin/python code/start.py --merkmale
```

Ergebnis:

```text
training/musicgen/merkmale/<lauf>/werte/
```

Die wichtigsten Dateien sind:

- `audio_merkmale.jsonl`
- `merkmale_zusammenfassung.json`

## 5. LoRA trainieren

Ziel: MusicGen Melody Large mit den geprueften Trainingsdaten an den
gewuenschten Lofi-Stil anpassen.

Sauberer Neustart ab Basismodell:

```bash
.venv/bin/python code/start.py --lora-neustart
```

Fortsetzen des aktiven LoRA-Laufs:

```bash
.venv/bin/python code/start.py --lora-fortsetzen
```

Weitere 500 Steps bewusst starten:

```bash
.venv/bin/python code/start.py --lora-weitere-500
```

Regel: Nach jedem 500er-Schritt werden Testaudios erstellt und bewertet. Ein
Checkpoint wird nur behalten, wenn die Audios wirklich besser sind.

Schutzregel: Ein neu trainierter Checkpoint wird nach dem Training nur als
`bewertung_offen` markiert. Er wird nicht automatisch fuer Longform-Audios
genutzt. Erst nach menschlicher Bewertung wird er freigegeben:

```bash
.venv/bin/python code/start.py --lora-freigeben --checkpoint PFAD
```

Dadurch bleibt ein alter guter Checkpoint aktiv, falls ein neuer 500er-Schritt
schlechter klingt.

## 6. Testaudios erzeugen

Ziel: pro Review-Lauf 8 neue 30s-Audios erzeugen und bewerten.

Start:

```bash
.venv/bin/python code/src/Training/bewertung_audios_erstellen.py
```

Ergebnis:

```text
training/bewertungen/musicgen/lora_bewertung_XXX/
```

Darin liegen:

- `audio/`: finale MP3-Testaudios
- `bewertung.csv`: menschliche Bewertung
- `generation_manifest.json`: Generierungsdaten
- `kandidaten_pruefung.csv`: technische Kandidatenpruefung

## 7. Lange Audio erzeugen

Ziel: aus frisch generierten MusicGen-Abschnitten oder aus vorhandenen Clips
eine lange Audio bauen.

Normale MusicGen-Erzeugung:

```bash
.venv/bin/python code/start.py
```

Dabei fragt das Terminal nach Dauer und Genre.

Die Generierung nutzt automatisch:

- Sekundencheck: jede Sekunde muss hoerbar sein
- BPM-Kontrolle
- technische MP3-Referenzprofile aus `daten/processed/trainingsdaten_geprueft/`
- Kandidaten-Scoring fuer Bass, Hoehen/Shaker, Peaks, Dynamik und zweite Haelfte
- Uebergangs-Score zwischen benachbarten Abschnitten
- maximal 80 Prozent VRAM-Zielwert
- automatische 5-Score-Bewertung mit zwei getrennten Graphen
- automatischen Referenzvergleich nach der fertigen Audio

Nach jeder erfolgreichen Audio entstehen im Ausgabeordner:

```text
score_bewertung.html
score_bewertung.json
score_bewertung.csv
```

Die sichtbare HTML-Bewertung zeigt nur:

- Audioplayer
- Genre
- Graph `Genrestandard`
- Graph `Generierte Audio`
- fuenf Scores pro Graph
- kurze Auswertung der groessten Unterschiede

Die Website zeigt dieselbe reduzierte 5-Score-Ansicht in der Bibliothek an,
wenn die lokale Audio-API die `score_bewertung.json` eines Ausgabeordners
mitliefert. Es gibt dafuer keinen separaten Analyse-Button.

Loop-Fallback aus vorhandenen Clips:

```bash
.venv/bin/python code/start.py --audio-loopen --quelle DATEI --dauer 20m
```

Ergebnisse:

```text
training/ausgaben/musicgen_generiert/
training/ausgaben/musicgen_loops/
```

## 8. Referenzvergleich

Ziel: erzeugte MusicGen-/LoRA-Audios mit den guten MP3-Trainingsclips
vergleichen. Dieser Schritt startet kein Training und erzeugt keine neue Musik.

Bei normaler Audioerstellung wird dieser Vergleich automatisch am Ende als
Pipeline-Stufe geschrieben.

Start mit automatisch erkanntem letzten Audio-Ordner:

```bash
.venv/bin/python code/start.py --referenzvergleich
```

Start mit einem bestimmten Audio-Ordner:

```bash
.venv/bin/python code/start.py --referenzvergleich --generiert-root training/bewertungen/musicgen/trainingsclips_lora_review_001/audio
```

Ergebnis:

```text
training/musicgen/referenzvergleich/<lauf>/
```

Wichtigste Dateien:

- `vergleich.csv`: Abweichungen der generierten Audios zur MP3-Referenz
- `zusammenfassung.json`: kompakter Gesamtbericht
- `diagramm_daten.csv` und optional `diagramm.png`
- `bericht.txt`: kurze Lesefassung

## 9. Status pruefen

Start:

```bash
.venv/bin/python code/start.py --status
```

Dieser Befehl startet nichts. Er zeigt nur Dataset-, Qualitaets- und
LoRA-Status.
