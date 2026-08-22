# MusicGen MP3-First Lernstruktur

## Ziel

Das MusicGen-System soll nicht dauerhaft nur ueber kleine manuelle Review-Runden
gesteuert werden. Stattdessen soll es schrittweise in einen automatisierten
Lernkreislauf uebergehen:

1. lange MP3-Quellen sammeln
2. jede ganze Audio zuerst analysieren
3. daraus nur stabile 60s-Clips erzeugen
4. gute Clips identifizieren
5. damit weitertrainieren
6. spaeter einen Quality-Scorer nutzen, um neue Clips automatisch zu bewerten

Kurzfristig liegt der Fokus auf **60s-Lofi-Clips**.
Langfristig sollen weitere Genres wie **Jazz** dazukommen.

## Grundsatz

**MP3 ist die Standard-Quelle, WAV ist das Trainingsformat.**

Das bedeutet:

- Die langen Quelldateien duerfen und sollen weiter als `mp3` vorliegen.
- Fuer das eigentliche Fine-Tuning werden daraus normalisierte `wav`-Clips
  erzeugt, damit das Trainingsmaterial technisch stabil bleibt.
- Review-MP3s dienen primaer dem Anhoeren und Bewerten, nicht als Hauptquelle
  fuer das naechste grosse Training.

Merksatz:

- **Quell-MP3s werden trainiert.**
- **Generierte MusicGen-Audios werden bewertet.**
- **Quell-MP3-Clips werden nicht manuell einzeln benotet, wenn sie bereits das
  kuratierte Referenzmaterial darstellen.**

## Projektstruktur

```text
daten/
  raw/
    audio/                          <- lange MP3-Quellen (Hauptquelle)
  processed/
    musicgen_dataset_v001/          <- grosser bestehender Basisdatensatz
    musicgen_dataset_v0xx_feedback_60s/
                                     <- kleine manuell gewichtete Feedback-Datensaetze
  features/
    musicgen_clip_quality/          <- spaeter: automatisch berechnete Clip-Features

runs/
  musicgen/                         <- Fine-Tuning-Laeufe train_0xx
  musicgen_reviews/                 <- Review-Packs mit MP3s + bewertung.csv
  musicgen_scorer/                  <- spaeter: trainierte Quality-Scorer
  musicgen_autolearn/               <- spaeter: automatische Lernzyklen

src/
  Dataset/
    prepare_musicgen_dataset.py     <- schneidet MP3-Quellen in Trainingsclips
    run_musicgen_data_pipeline.py   <- Datensatzaufbau aus der Rohbibliothek
  Training/
    apply_musicgen_feedback.py      <- manuelles Feedback in Trainingszeilen umwandeln
    run_musicgen_feedback_cycle.py  <- Fine-Tuning aus vorhandenem Datensatz
    run_musicgen_smoke_train.py     <- eigentlicher Kurz-/Feedback-Trainingslauncher
    create_musicgen_review_pack.py  <- MP3-Review-Packs bauen
```

## Lernphasen

### Phase 1 - MP3-Bibliothek als Hauptquelle

Ziel:
- aus den langen MP3-Dateien einen starken 60s-Korpus bauen
- nicht dauernd nur 6-10 gute Review-Samples uebergewichten

Regeln:
- MP3-Quellen nicht blind in 60s-Clips schneiden
- zuerst die ganze Audio analysieren
- stabile Laengsabschnitte im Gesamtverlauf erkennen
- nur Clips verwenden, die innerhalb solcher stabilen Abschnitte liegen
- Randzonen um Abschnittswechsel aussparen, damit Uebergaenge nicht mitten im Trainingsclip landen
- pro Clip Metadaten erzeugen
- Clips mit offensichtlichen Problemen verwerfen:
  - starke Pegelspruenge
  - Audio-Dropouts
  - zu chaotische Groovewechsel
  - zu wenig Klarheit

Wichtige Folge:
- Ein einzelner 60s-Clip wird nicht mehr isoliert beurteilt.
- Die Quelle wird zuerst als **ganzer Track** bewertet.
- Erst danach werden Trainingsclips aus den ruhigen, stabilen Regionen entnommen.

Aktueller Vorteil:
- es liegen bereits 64 lange Audioquellen lokal vor
- daraus wurden schon tausende Clips gebaut

### Phase 2 - Manuelles Seed-Feedback

Ziel:
- dem System klar beibringen, was "gut" fuer den Stil bedeutet

Signalquellen:
- `rating_1_to_5`
- kurze Kommentare wie:
  - mehr Harmonie
  - smoothere Uebergaenge
  - Bass zu stark
  - mehr Klarheit
  - zu chaotisch

Bewertungsregel:
- `5`: stark positiv
- `4`: positiv
- `3`: neutral / nicht verstaerken
- `1-2`: negativ / verwerfen

### Phase 3 - 60s Quality-Scorer

Ziel:
- aus manuellen Bewertungen einen automatischen Clip-Bewerter trainieren

Input:
- 60s-Clip
- ggf. akustische Features
- spaeter optional Text-/Caption-Merkmale

Output:
- `wahrscheinlich gut`
- `unsicher`
- `wahrscheinlich schlecht`

Einsatz:
- gute Clips automatisch vorfiltern
- schlechte automatisch rauswerfen
- nur unsichere Grenzfaelle noch manuell pruefen

### Phase 4 - Automatischer Lernkreislauf

Ziel:
- aus neu erzeugten oder neu geschnittenen Clips automatisch weiterlernen

Kreislauf:
1. neue 60s-Clips erzeugen oder aus MP3-Bibliothek schneiden
2. Quality-Scorer bewerten lassen
3. nur High-Confidence-Gute uebernehmen
4. Fine-Tuning starten
5. neue Test-Audios erzeugen
6. periodisch Stichprobe manuell pruefen

## Automatisierungsregeln

### Was automatisiert werden darf

- MP3-Ingest
- Clip-Schnitt
- WAV-Normalisierung
- Datensatzaufbau
- Feature-Extraktion
- Vorbewertung durch Quality-Scorer
- Fine-Tuning
- Review-Pack-Generierung

### Was vorerst nicht vollautomatisch laufen sollte

- finale Geschmacksentscheidung bei Grenzfaellen
- Bewertung von 5-Minuten-Stuecken
- Genre-Umschaltung ohne getrennte Seed-Bewertungen

## Genre-Strategie

### Erst Lofi stabilisieren

Der erste Scorer soll nur fuer **60s-Lofi** robust werden.

### Danach Jazz einfuehren

Fuer Jazz braucht es:
- eigene MP3-Quellen
- eigene Clips
- eigene manuelle Seed-Bewertungen

Wichtig:
- "gut" fuer Lofi ist nicht automatisch "gut" fuer Jazz
- spaeter sollte der Scorer genre-aware werden

## Langform-Regel

Das aktuelle 60s-Fine-Tuning verbessert:
- Klangfarbe
- kurze Uebergaenge
- lokalen Stil

Es verbessert noch nicht automatisch:
- 5-Minuten-Groove-Stabilitaet
- minutenlang gleichbleibenden Takt

Deshalb gilt:
- **60s-Qualitaet zuerst automatisieren**
- **Longform spaeter separat behandeln**

Longform wird spaeter eine eigene Strategie brauchen:
- laengere Trainingssegmente
- konservativere Generierungsparameter
- stabilere Prompt-Formulierungen

## Entscheidungslogik fuer den naechsten Schritt

### Wenn eine Review-Runde stark ist

Dann darf sie den naechsten Feedback-Datensatz beeinflussen.

### Wenn eine Review-Runde schwach ist

Dann darf sie **nicht** uebermaessig hochgewichtet werden.
In diesem Fall soll wieder die grosse MP3-Bibliothek im Vordergrund stehen.

## Empfohlener naechster Schritt

1. **Nicht** blind mit einer schwachen Review-Runde weiterdrehen.
2. Zurueck auf die **64 langen MP3-Quellen**.
3. Einen neuen staerkeren 60s-Datensatz bauen mit Fokus auf:
   - Whole-Track-Analyse vor dem Clipping
   - stabiler Groove
   - weniger Rhythmuswechsel
   - smoothere Uebergaenge
   - weniger Chaos
   - mehr Klarheit
4. Danach erneutes Fine-Tuning.
5. Danach den ersten **60s-Quality-Scorer** bauen.

Der zentrale MP3-first Einstieg dafuer ist:

`src/Training/run_musicgen_mp3_master_cycle.py`

## Zielbild bis Ende Juni

- stabiles Audio-V1 fuer 60s
- MP3-first Datenpipeline
- teilautomatische 60s-Bewertung
- automatischer Trainingskreislauf mit manueller Stichprobe
- spaeter Genre-Erweiterung um Jazz
