# Prozessdokumentation der MusicGen-LoRA-Pipeline

Stand: 2026-08-15

## 1. Zweck des Dokuments

Dieses Dokument beschreibt den Entwicklungs- und Verbesserungsprozess der
MusicGen-LoRA-Pipeline fuer die Bachelorarbeit. Der Schwerpunkt liegt bewusst
nicht auf einzelnen Audioergebnissen, sondern auf den Prozessen, Entscheidungen,
Pruefmechanismen und technischen Iterationen, durch die das Projekt schrittweise
stabilisiert wurde.

Die Dokumentation ist als wissenschaftliche Prozessbeschreibung gedacht. Sie
kann fuer Methodik, Implementierung, Evaluation und Diskussion verwendet werden.
Sie erklaert, wie aus einem anfangs nur bedingt brauchbaren MusicGen-Ansatz eine
lokale, steuerbare und auswertbare Pipeline entstanden ist.

Zentrale Leitfrage:

```text
Wie kann eine lokale MusicGen-Pipeline so erweitert werden, dass Lo-Fi-Musik
genrebezogen vorbereitet, mit LoRA angepasst, ueber eine Website gesteuert,
technisch bewertet und fuer lange Audios nutzbar gemacht werden kann?
```

## 2. Ausgangslage

Am Anfang des Projekts stand ein vorhandenes MusicGen-basiertes System, das
zwar grundsaetzlich Audio erzeugen konnte, aber fuer den konkreten
Bachelorarbeitskontext noch nicht ausreichend kontrollierbar war. Das
Basismodell erzeugte teilweise Lo-Fi-nahe Musik, zeigte jedoch wiederkehrende
Probleme:

- zu dominante Bassanteile,
- zu starke Hoehen-, Shaker- oder Rauschanteile,
- musikalische Monotonie innerhalb kurzer 30-Sekunden-Clips,
- instabile Tonalitaet oder kippende Klangbilder,
- einzelne Abschnitte mit zu wenig oder gar keinem hoerbaren Ton,
- unklare Genre-Zuordnung,
- bei Guitar Lofi teilweise zu wenig erkennbare Gitarre,
- bei Jazz Lofi teilweise zu wenig erkennbare Piano-/Jazz-Harmonik,
- zu technische oder gestellte Wirkung bei Chillhop.

Diese Probleme wurden nicht als einzelner Fehler behandelt, sondern als Hinweis
darauf, dass Generierung, Datengrundlage, LoRA-Anpassung, Prompting,
Kandidatenauswahl und Nachbearbeitung gemeinsam betrachtet werden muessen.

Die Bewertungsanalyse dokumentiert diese fruehe Problemlage. In
`bewertungen/analyse/bewertungs_analyse_pro_durchgang.csv` wurden pro
Bewertungsdurchgang Problemklassen wie `bass_stark`, `shaker_stark`,
`ton_kippt`, `kein_ton`, `monoton`, `rauschen`, `gitarre_passt_nicht` und
`kreativitaet_fehlt` erfasst. Die Problem-Rate schwankte deutlich zwischen den
Durchgaengen. Das zeigt, dass einzelne Testlaeufe allein nicht ausreichen,
sondern ein wiederholbarer Bewertungsprozess notwendig ist.

## 3. Projektanforderungen

Die Pipeline wurde an folgenden Anforderungen ausgerichtet:

| Anforderung | Umsetzung im Projekt |
|---|---|
| Datenerfassung | Quellen werden gesucht, importiert und pro Genre dokumentiert. |
| Datenaufbereitung | MP3-/WAV-Quellen werden in standardisierte 30s-Clips zerlegt. |
| Feature Extraction | Technische Audiomerkmale werden fuer Clips und generierte Audios berechnet. |
| Modellanpassung | MusicGen wird mit LoRA adaptiert, nicht vollstaendig neu trainiert. |
| Musikgenerierung | MusicGen erzeugt kurze Kandidaten, daraus entstehen lange Audios. |
| Website | Frontend und lokales Backend steuern Suche, Clips, Training, Testaudios und Generierung. |
| Evaluation | Menschliche Bewertungen und automatische Scores werden kombiniert. |
| GIF/Video-Integration | Bestehende Video-/GIF-Teile wurden nicht entfernt oder zerstoert. |
| Lokale Ausfuehrbarkeit | Rechenintensive Schritte laufen lokal und werden bewusst gestartet. |
| Prozessdokumentation | Laeufe erzeugen Reports, Statusdateien, CSVs und JSON-Artefakte. |

## 4. Methodische Grundidee

Das Projekt folgt einem Human-in-the-Loop-Ansatz. Automatische Pruefungen
erkennen technische Risiken, waehrend menschliche Bewertungen entscheiden, ob
ein musikalisches Ergebnis tatsaechlich besser geworden ist.

Die Pipeline wurde daher nicht als einmaliger Trainingslauf aufgebaut, sondern
als iterativer Prozess:

1. Daten sammeln oder importieren.
2. Daten in ein kontrolliertes Clipformat ueberfuehren.
3. Clips technisch und menschlich pruefen.
4. LoRA mit genrebalancierten Clips trainieren.
5. Checkpoints erzeugen, aber nicht automatisch uebernehmen.
6. Testaudios pro Genre erstellen.
7. Menschliches Feedback auswerten.
8. Prompts, Scoring, Filter und Trainingsentscheidungen anpassen.
9. Lange Audios aus kurzen, geprueften MusicGen-Kandidaten erzeugen.
10. Ergebnisse technisch mit Genre-Referenzen vergleichen.

Wichtig ist die Trennung zwischen Automatisierung und Entscheidung. Das System
kann Kandidaten erzeugen und bewerten, aber ein neuer Checkpoint wird erst dann
genutzt, wenn er bewusst freigegeben wurde.

### 4.1 Chronologische Verlaufsuebersicht

Die folgende Uebersicht fasst den Entwicklungsverlauf in Prozessschritten
zusammen. Sie ist nicht als vollstaendige Git-Historie gemeint, sondern als
wissenschaftliche Rekonstruktion der wichtigsten Projektentscheidungen.

| Phase | Beobachtung oder Ausgangspunkt | Prozessfortschritt |
|---|---|---|
| Anfangszustand | MusicGen konnte Audio erzeugen, aber Stil, Klangbalance und Stabilitaet waren unzuverlaessig. | Problemklassen wurden durch Hoeren und Bewertung gesammelt. |
| Erste Bewertungen | Bassdominanz, Shaker/Hoehen, kippender Ton, stille Stellen und Genreunschaerfe traten wiederholt auf. | Bewertungskategorien wurden in CSV-/JSON-Auswertungen ueberfuehrt. |
| Dataset-Aufbau | Gute Quellen waren vorhanden, mussten aber systematisch als Trainingsclips nutzbar werden. | 30s-Clips, Genreordner, Duplikatpruefung und source-disjoint Splits wurden eingefuehrt. |
| Review-Datensatz | Einige Quellen verschlechterten die Trainingsbasis. | Schlechte Quellen wurden ausgeschlossen, 531 Clips wurden in der 4000er-Pruefung verworfen. |
| LoRA-Zieldataset | Das Training sollte ausgewogener werden. | Ziel wurde auf 5000 Clips, 1000 pro Genre, erweitert. |
| Erste LoRA-Staende | Nicht jeder Checkpoint klang besser als der vorherige. | Checkpoint-Freigabe wurde als menschliche Entscheidung eingefuehrt. |
| 625-Schritte-Stand | Ein frueher Stand erwies sich als vergleichsweise stabil. | `step_000625` wurde als bester dokumentierter Adapter beibehalten. |
| Weitere Trainingsversuche | Mehr Schritte fuehrten nicht automatisch zu besserer Musik. | Ein 3000-Schritte-Stand wurde als Regression behandelt und archiviert. |
| 1250-Schritte-Kandidat | Training konnte fortgesetzt werden, aber die Qualitaet musste erneut geprueft werden. | `step_001250` wurde als Kandidat mit Status `bewertung_offen` markiert. |
| Website-Integration | Terminalbefehle waren fuer wiederholte Bachelorarbeitslaeufe unpraktisch. | Website und lokale API wurden fuer Training, Freigabe, Clips, Generierung und Bewertung verbunden. |
| Testaudio-Korrektur | Unklarheit zwischen echten MusicGen-Clips und Dataset-Hoerproben. | Website trennt `MusicGen-Clips` und `Dataset-Clips`. |
| Longform-Logik | 30s-Clips allein reichen nicht fuer lange Audios. | Looping, Blocklaengen, Crossfades und Uebergangspruefung wurden ergaenzt. |
| Monotonieproblem | Einige technisch gueltige Clips wirkten zu leer. | Bewegungsmetriken und strengere Kandidatenauswahl wurden eingefuehrt. |
| Rausch-/Shakerproblem | Dreamy und andere Genres enthielten stoerende Hoehen-/Rauschanteile. | Prompt-Blocklist, `high_ratio`-Strafen und Lowpass-/EQ-Kontrolle wurden verschaerft. |
| Aktueller Stand | Pipeline ist steuerbar, aber die neueste Klangqualitaet muss noch gehoert werden. | Naechster Schritt ist ein neuer MusicGen-Clip-Lauf mit menschlicher Bewertung. |

## 5. Projektstruktur

Die aktive Projektstruktur wurde auf vier Hauptbereiche reduziert:

| Bereich | Funktion |
|---|---|
| `code/` | Quellcode der Pipeline, API, Website-Anbindung und Trainingslogik |
| `daten/` | Rohdaten, verarbeitete Clips, Features und lokale Modelle |
| `training/` | Trainingslaeufe, Checkpoints, Bewertungen, Testaudios und Reports |
| `dokumentation/` | Struktur-, Ablauf-, Code- und Prozessdokumentation |

Diese Struktur ist fuer die wissenschaftliche Nachvollziehbarkeit wichtig, weil
Code, Daten, Ergebnisse und Dokumentation nicht vermischt werden.

Der zentrale lokale Einstieg ist:

```bash
.venv/bin/python code/start.py
```

Die lokale API wird durch folgende Datei bereitgestellt:

```bash
.venv/bin/python code/src/Pipeline/web_api.py
```

Die Website liegt unter:

```text
code/website/lo-fi-dreamer/
```

## 6. Prozessphase 1: Quellen und Genres

### Ziel

Ziel dieser Phase war der Aufbau einer geeigneten Datengrundlage fuer mehrere
Lo-Fi-Untergenres. Statt ein unspezifisches Lo-Fi-Dataset zu nutzen, wurden die
Zielstile explizit getrennt.

Verwendete Zielgenres:

- Jazz Lofi
- Chillhop Lofi
- Dreamy Lofi
- Study Lofi
- Guitar Lofi

### Vorgehen

Die Quellensuche und der Import wurden von Training und Generierung getrennt.
Dadurch kann die Pipeline mit bereits vorhandenen Daten arbeiten, ohne bei
jedem Lauf neue Downloads auszufuehren.

Relevante Dateien:

| Datei | Aufgabe |
|---|---|
| `code/src/Crawler/quellen_suche.py` | Suche, Import und Verarbeitung von Audioquellen |
| `code/src/Crawler/quellen_finden.py` | genrebezogene Suche nach fehlenden Quellen |
| `code/src/Pipeline/clips_vorbereiten.py` | Vorbereitung von Quellen fuer das Clip-Dataset |
| `code/configs/lora_genres.json` | zentrale Genredefinitionen |

### Wissenschaftliche Bedeutung

Die Trennung nach Genres ist methodisch relevant, weil Lo-Fi kein einheitlicher
Klangraum ist. Guitar Lofi, Jazz Lofi und Chillhop besitzen unterschiedliche
Instrumentierungs- und Strukturmerkmale. Ein gemeinsames Modell ohne
genrebezogene Steuerung wuerde diese Unterschiede leichter verwischen.

## 7. Prozessphase 2: Clip-Vorbereitung

### Ziel

MusicGen arbeitet in diesem Projekt mit kurzen Audiokontexten. Deshalb wurden
laengere MP3-/WAV-Dateien in standardisierte 30-Sekunden-Clips ueberfuehrt.
Diese Clips bilden die Grundlage fuer Training, Referenzvergleich und
Testaudio-Bewertung.

### Vorgehen

Die Aufbereitung umfasst:

1. lokale Audioquellen erfassen,
2. Audios in 30s-Clips schneiden,
3. Clips nach Genre sortieren,
4. unbrauchbare Clips ausschliessen,
5. Duplikate entfernen,
6. Train/Valid/Test-Splits erzeugen,
7. Metadaten und Reports schreiben.

Relevante Dateien:

| Datei | Aufgabe |
|---|---|
| `code/src/Dataset/zieldatensatz.py` | Aufbau eines genrebalancierten LoRA-Datasets |
| `code/src/Dataset/trainingsdaten_pruefen.py` | Ausschluss schlechter Quellen und Clips |
| `code/src/Dataset/genre_regeln.py` | Genre- und Split-Regeln |
| `training/musicgen/dataset_pruefung/` | Reports zur Dataset-Pruefung |

### Entwicklung des Datasets

Ein frueher gepruefter Datensatz wurde mit menschlicher Review bereinigt.
Der Plan `training/musicgen/dataset_pruefung/trainingsdaten_plan.json` zeigt:

- Ziel: 4000 Clips,
- 800 Clips pro Genre,
- 6479 Kandidatenclips,
- 5948 akzeptierte Kandidaten,
- 531 verworfene Clips aus schlechten Quellen,
- source-disjoint Train/Valid/Test-Aufteilung,
- ausgeschlossene Problemquellen: `CzzBAIr_dOA`, `ED8rEI9DAc0`, `ZBzZLnmbXMk`.

Anschliessend wurde der Zielrahmen fuer LoRA auf 5000 Clips erweitert. Der Plan
`training/musicgen/dataset_pruefung/zieldatensatz_plan.json` dokumentiert:

- Ziel: 5000 Clips,
- Ziel pro Genre: 1000 Clips,
- 6497 verfuegbare Clips,
- 4961 initial ausgewaehlte Clips,
- 39 fehlende Jazz-Lofi-Clips in dieser Planstufe,
- source-disjoint Aufteilung,
- mindestens 5 unabhaengige Quellen pro Genre.

Der spaetere aktive LoRA-Trainingsstand verwendet laut
`training/musicgen/lora_training/stand.json` schliesslich:

- 5000 Clips gesamt,
- 1000 Clips pro Genre,
- 5 Genres,
- Dataset: `daten/processed/lora_training`.

### Entscheidung

Die Pipeline wurde so gestaltet, dass nicht nur die Anzahl der Clips zaehlt,
sondern auch Quellenqualitaet, Genreverteilung, Duplikatfreiheit und
source-disjoint Splits. Das ist wissenschaftlich wichtig, weil sonst die
Evaluation durch Quellenueberlappung verfaelscht werden koennte.

### 7.1 Quellengetrennter Auswahlalgorithmus (Round-Robin)

Die eigentliche Clip-Auswahl fuer das LoRA-Zieldataset erfolgt in der Funktion
`waehle_quellengetrennt()` (`code/src/Dataset/genre_regeln.py`). Sie loest
zwei Probleme gleichzeitig: Datenleck zwischen den Splits (derselbe Song
duerfte nicht gleichzeitig in Train und Test vorkommen) und Dominanz einzelner
Quellen innerhalb eines Splits.

Der Algorithmus laeuft in drei Schritten:

1. **Split-Zuordnung pro Quelle statt pro Clip.** Fuer jedes Genre werden alle
   Clips nach Ursprungsquelle gruppiert. Anschliessend wird ueber alle
   moeglichen Quellenpaare (eine Quelle fuer Valid, eine fuer Test) dasjenige
   Paar gewaehlt, das die Zielgroessen der drei Splits am vollstaendigsten
   erreicht (Bewertungsfunktion: Grad der Zielerreichung, Gesamtzahl gewaehlter
   Clips, Anzahl verbleibender Train-Quellen, minimierte Verschwendung
   ungenutzter Valid-/Test-Kapazitaet). Dadurch stammt keine Datei gleichzeitig
   aus zwei Splits.
2. **Round-Robin-Ziehung innerhalb eines Splits.** Fuer den Train-Split werden
   die Clips nicht quellenweise nacheinander entnommen (was die groessten
   Quellen bevorzugen wuerde), sondern reihum: Jede Quelle liefert einen Clip,
   dann die naechste, bis entweder das Ziel erreicht oder eine Quelle
   erschoepft ist. Das begrenzt den Einfluss einzelner sehr grosser Quellen
   (bis zu `max_pro_quelle_pro_genre`, Standard 250 Clips) und stellt sicher,
   dass auch kleine Quellen mit wenigen Clips ueberhaupt in die Auswahl
   gelangen.
3. **Nachfuellen (Backfill).** Wird die Zielanzahl durch Round-Robin allein
   knapp verfehlt (z. B. weil mehrere kleine Quellen fruehzeitig erschoepft
   sind), duerfen bereits genutzte Quellen im selben Split zusaetzliche Clips
   liefern, bis das Ziel erreicht ist oder kein Fortschritt mehr moeglich ist.

Diese Konstruktion war die technische Voraussetzung fuer das im Projekt
durchgefuehrte Experiment zur Quellenvielfalt (Abschnitt 24.3): Erst durch das
Round-Robin-Prinzip konnten neu hinzugefuegte, kleine Quellen (2 bis 40 Clips
je Datei) ueberhaupt gegen etablierte, grosse Quellen (bis zu 250 Clips)
bestehen, statt vom Auswahlverfahren systematisch verdraengt zu werden.

## 8. Prozessphase 3: Feature Extraction und technische Pruefung

### Ziel

Die automatische Feature Extraction wurde eingefuehrt, um Audioqualitaet nicht
nur subjektiv zu beschreiben. Sie dient als reproduzierbare technische
Pruefschicht vor menschlicher Bewertung.

### Gepruefte Merkmale

Die Pipeline berechnet unter anderem:

- Dauer,
- Samplerate,
- Peak und RMS,
- stille oder zu leise Sekunden,
- Clipping-Risiko,
- Bassanteil,
- Hoehen-/Shaker-Anteil,
- Tonalitaets- oder Signaltonrisiko,
- Dynamik,
- BPM-nahe Struktur,
- Energieverteilung,
- Veraenderung innerhalb eines 30s-Clips.

Relevante Dateien:

| Datei | Aufgabe |
|---|---|
| `code/src/Merkmale/audio_merkmale.py` | technische Audiomerkmale fuer Dataset-Clips |
| `code/src/Training/audio_bewertung.py` | automatische Bewertung erzeugter Audios |
| `code/src/Training/referenz_vergleich.py` | Vergleich zu guten Genre-Referenzen |
| `code/src/Training/audio_erstellen.py` | Kandidatenpruefung waehrend der Generierung |

### Entscheidung

Die technischen Merkmale wurden nicht als Ersatz fuer menschliches Hoeren
verwendet. Sie dienen dazu, offensichtliche Fehler frueh zu erkennen:
Stille, harte Pegelspruenge, uebermaessiger Bass, uebermaessige Hoehenanteile
oder monotone Kandidaten.

### 8.1 Exakte Berechnungsformeln der fuenf Bewertungskategorien

Fuer die wissenschaftliche Nachvollziehbarkeit ist relevant, dass die fuenf
Bewertungskategorien (Abschnitt 17) nicht als Blackbox berechnet werden,
sondern aus expliziten, in `code/src/Training/audio_bewertung.py` definierten
Formeln bestehen. Grundlage ist eine generische Distanzfunktion, die einen
gemessenen Wert gegen einen Referenzwert normiert:

```text
distanz_score(wert, referenz, skala) =
    100 - min(100, |wert - referenz| / skala * 100)
```

`skala` ist dabei kein statistisch geschaetzter Parameter, sondern ein fest
gewaehlter Toleranzbereich pro Merkmal (z. B. 10 BPM fuer Tempo oder 0.18 fuer
den Bassanteil). Ein Messwert, der genau `skala` vom Referenzwert entfernt
liegt, erhaelt den Score 0; bei exakter Uebereinstimmung ergibt sich 100.

Darauf aufbauend werden die fuenf Kategorien wie folgt berechnet
(`clamp_score` begrenzt das Ergebnis jeweils auf den Bereich 0 bis 100):

**Technische Audioqualitaet** (`_score_technisch`) startet bei 100 und zieht
Strafpunkte ab fuer: zu geringen Aktivanteil (`active_ratio < 0.98`, Faktor
140, max. 35 Punkte), Clipping-Anteil (Faktor 2500, max. 25 Punkte),
Pegelspitzen ueber -0.4 dB (Faktor 8), starke Halblautstaerke-Einbrueche
(`half_drop_db > 5`, Faktor 2), plötzliche Lautstaerkespruenge
(`explosion_db > 8`, Faktor 2.2), Signalton-/Tonalitaetsanteil ueber 0.20
(Faktor 80) und Hoehenanteil ueber 0.28 (Faktor 35).

**Musikalische Kohaerenz** (`_score_kohaerenz`) bestraft Tempo-Instabilitaet
(`bpm_std`, Faktor 4, max. 30 Punkte), abrupte Lautstaerkespruenge und
-einbrueche (aehnlich wie oben), zu geringe Dynamikspanne
(`rms_range_db < 2`, Faktor 4), zu grosse Dynamikspanne (`rms_range_db > 18`,
Faktor 1.5) und Signaltonanteil ueber 0.18 (Faktor 70).

**Genre-Treue** (`_score_genre`) ist der Mittelwert aus sechs
Distanz-Scores gegen das gespeicherte Genre-Referenzprofil: BPM (Skala 10),
Bassanteil (Skala 0.18), Snareanteil (Skala 0.16), Hoehenanteil (Skala 0.12),
spektraler Schwerpunkt (Skala 1400 Hz) und Dynamikspanne (Skala 8 dB). Diese
Kategorie ist im Projektverlauf durchgehend die volatilste und
schwaechste Kategorie gewesen (siehe Abschnitt 22).

**Uebergangsqualitaet** (`_score_uebergang`) bewertet ausschliesslich die
Naht zwischen Audioende und -anfang (fuer Loop-Faehigkeit): Lautstaerkesprung
an der Naht (`seam_loudness_jump_db > 3`, Faktor 5), spektraler Sprung
(Faktor 120), harmonischer Sprung (Faktor 60) und hoerbare Klicks
(`seam_click > 0.08`, Faktor 180).

**Referenzaehnlichkeit** (`_score_referenz`) ist der Mittelwert aus sieben
Distanz-Scores gegen den Median der technischen Merkmale echter
MP3-Referenzclips desselben Genres: Lautheit (RMS, Skala 8 dB), Pegelspitze
(Skala 5 dB), Aktivanteil (Skala 0.12), Bassanteil (Skala 0.20), Hoehenanteil
(Skala 0.14), Signaltonanteil (Skala 0.12) und BPM (Skala 12).

Wichtig fuer die methodische Einordnung: Alle fuenf Kategorien sind
merkmalsbasierte Abstandsmasse, keine gelernten oder wahrnehmungsbasierten
Modelle. Sie erkennen technische Abweichungen zuverlaessig, aber nicht
zwangslaeufig, ob eine Audio subjektiv gut oder als das richtige Genre
erkennbar klingt. Diese Einschraenkung wird in Abschnitt 21 (Semantische
Genre-Pruefung) und Abschnitt 22 (Messvaliditaet) empirisch belegt.

## 9. Prozessphase 4: LoRA-Modellanpassung

### Ziel

MusicGen sollte nicht vollstaendig neu trainiert werden. Stattdessen wurde LoRA
als ressourcenschonende Anpassung genutzt. Dadurch werden nur adapterartige
Gewichte trainiert, waehrend das Basismodell weitgehend unveraendert bleibt.

### Trainingskonfiguration

Der aktive Trainingslauf ist in `training/musicgen/lora_training/` dokumentiert.
Wichtige Parameter aus `training_config.json`:

| Parameter | Wert |
|---|---|
| Basismodell | `facebook/musicgen-melody-large` |
| lokales Modell | `daten/modelle/musicgen/facebook_musicgen_melody_large` |
| Hardware | NVIDIA RTX A6000 |
| VRAM-Limit | 80 Prozent, ca. 37.9 GB |
| Praezision | bfloat16 |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| Target Modules | `out_proj`, `linear1`, `linear2` |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| Learning Rate | 5e-06 |
| Max Steps | 1250 |
| Save Steps | 25 |
| Eval Steps | 50 |
| Resume | von `step_000625` |

Die Parameterzusammenfassung zeigt:

- Gesamtparameter: 2,460,917,760,
- trainierbare Parameter: 9,437,184,
- eingefrorene Parameter: 2,451,480,576.

Damit wird nur ein kleiner Teil der Modellparameter angepasst. Das passt zur
Bachelorarbeitsanforderung, lokal und ressourcenschonend zu arbeiten.

### Checkpoint-Strategie

Neue Checkpoints werden nicht automatisch als endgueltig uebernommen. Der Stand
`training/musicgen/lora_training/stand.json` dokumentiert:

- Kandidat: `step_001250/lora_adapter.pt`,
- bester bisheriger Adapter: `step_000625/lora_adapter.pt`,
- Status: `bewertung_offen`,
- `review_required`: true,
- `use_for_audio_generation`: false.

Diese Trennung ist zentral. Sie verhindert, dass ein neuer Trainingsstand
automatisch schlechte Longform-Audios erzeugt.

### Umgang mit Regression

Im Projektverlauf zeigte sich, dass mehr Trainingsschritte nicht automatisch
bessere Ergebnisse liefern. Ein spaeterer 3000-Schritte-Stand wurde als
Regression behandelt und archiviert. Daraus ergab sich die Regel:

```text
Ein neuer Checkpoint ist nur dann ein Fortschritt, wenn die erzeugten
Testaudios hoerbar und messbar besser sind.
```

## 10. Prozessphase 5: Testaudios

### Ziel

Testaudios dienen als kontrollierter Zwischenstand nach Trainings- oder
Promptaenderungen. Sie ersetzen keine finale Evaluation, liefern aber eine
vergleichbare Basis fuer menschliches Feedback.

### Umsetzung

Die Website-Aktion `MusicGen-Clips` startet pro Projektgenre einen Testlauf.
Der Testaudio-Prozess erzeugt fuer jedes der fuenf Genres ein Audio und sammelt
die Ergebnisse in einem gemeinsamen Ordner.

Wichtige Umsetzungspunkte:

- pro Genre ein Testaudio,
- 30s MusicGen-Clip als Basiseinheit,
- Looping auf 1 Minute fuer bessere Hoerbarkeit,
- gemeinsamer Sammelordner,
- GitHub-faehige Ausgabe,
- Bewertungsvorlage und technische Pruefung.

Relevante Dateien:

| Datei | Aufgabe |
|---|---|
| `code/src/Training/bewertung_audios_erstellen.py` | Testaudio-Erzeugung |
| `code/src/Training/testaudios_sammeln.py` | Zusammenfuehren der Testaudios |
| `code/src/Pipeline/web_api.py` | Website-Aktion `testaudios` |
| `training/bewertungen/musicgen/` | Review- und Bewertungsdaten |

### Wichtige Korrektur im Prozess

Zu Beginn war nicht immer klar, ob Testaudios echte MusicGen-Generierungen oder
nur Trainingsclip-Proben sind. Deshalb wurde die Website sprachlich und
funktional getrennt:

| Website-Aktion | Bedeutung |
|---|---|
| `MusicGen-Clips` | erzeugt neue Musik mit MusicGen und aktivem/freigegebenem LoRA-Adapter |
| `Dataset-Clips` | erstellt Hoerproben aus vorhandenen Trainingsclips, ohne neue Musik zu generieren |

Diese Trennung ist wichtig, weil nur `MusicGen-Clips` die Modellleistung
bewerten. `Dataset-Clips` pruefen dagegen die Qualitaet der Datengrundlage.

## 11. Prozessphase 6: Longform-Audio

### Ziel

MusicGen erzeugt in diesem Projekt kurze Kandidatenclips. Die Bachelorarbeit
benoetigt aber laengere Audios, zum Beispiel 5 Minuten, 20 Minuten oder 1
Stunde. Deshalb wurde eine Longform-Logik entwickelt.

### Grundprinzip

Lange Audios entstehen nicht durch simples Aneinanderhaengen. Stattdessen
werden kurze Kandidaten erzeugt, geprueft, geloopt und mit passenden
Uebergaengen verbunden.

Aktueller Ablauf:

1. MusicGen erzeugt einen 30s-Kandidaten.
2. Pro Abschnitt werden mehrere Kandidaten erzeugt.
3. Jeder Kandidat wird technisch geprueft.
4. Kandidaten mit Stille, zu starkem Bass, zu starken Hoehen oder zu wenig
   Bewegung werden abgewertet oder verworfen.
5. Ein stabiler 30s-Clip wird auf einen laengeren Block geloopt.
6. Der naechste Block wird nicht nur isoliert, sondern im Uebergang bewertet.
7. Crossfades reduzieren harte Schnitte.
8. Die finale Audio wird automatisch bewertet.

### Gepruefte Uebergangskriterien

Die Uebergangslogik beruecksichtigt:

- Pegel Ende von Block A vs. Anfang von Block B,
- Bassverhaeltnis,
- Hoehen-/Shaker-Verhaeltnis,
- Snare-/Praesenzbereich,
- Energie-Sprung,
- Peak-Sprung,
- Stille oder abgeschnittener Ton,
- grobe BPM-Nahe,
- klangliche Aehnlichkeit der Blockenden.

### Entscheidung

Die Pipeline waehlt nicht den isoliert besten Clip, sondern den Clip, der in
den geplanten Kontext passt. Das ist fuer Lo-Fi-Longform wichtig, weil kleine
30s-Clips einzeln akzeptabel klingen koennen, aber in der Wiederholung oder im
Uebergang stoerend wirken.

### 11.1 Zwei technisch unterschiedliche Generierungsmodi

Im Projektverlauf wurden zwei grundsaetzlich verschiedene Verfahren erprobt,
um aus 30s-MusicGen-Kandidaten laengere Bloecke zu erzeugen. Beide sind im
Code (`code/src/Training/audio_erstellen.py`) ueber die Flags
`--block-looping-aktiv` und `--kontinuierliche-bloecke-aktiv` steuerbar und
schliessen sich gegenseitig aus:

**Block-Looping (produktiv genutzt).** MusicGen erzeugt genau ein natives
30s-Fenster. Dieses wird anschliessend im Audio-Bereich (nicht erneut durch
das Modell) auf die Zieldauer verlaengert. Die Funktion `baue_loop_block()`
sucht dazu automatisch eine geeignete Taktgrenze innerhalb eines
Suchfensters von 7 Sekunden, begrenzt die Loop-Laenge auf 18 bis 29 Sekunden,
ueberblendet den Ruecksprung mit einem parametrisierbaren Crossfade und kann
optional eine pitch-erhaltende Tempovariation (`tempo_variation_percent`)
ueber die Zeit anwenden, damit der Loop nicht mechanisch identisch wirkt.

**Kontinuierliche Bloecke (fruehere Variante, inzwischen nicht mehr
Standard).** MusicGen setzt jeden Block als echte modellinterne Fortsetzung
fort (`extend_stride`-Mechanik von Audiocraft), erzeugt also durchgehend neue
Tokens statt zu loopen. Das vermeidet Wiederholung, fuehrt aber alle
`erweiterungs_schritt_sekunden` (Standard 12 s) zu einer neuen internen Naht,
an der das Modell erneut ansetzen muss. Im Projektverlauf zeigte sich, dass
genau diese Naehte fuer wahrgenommene "Uebergaenge innerhalb der 30 Sekunden"
und lokal duenn wirkende Abschnitte (z. B. eine auffaellig ruhige Stelle bei
Chillhop zwischen Sekunde 13 und 28 in einer fruehen Version) mitverantwortlich
waren. Die Umstellung auf Block-Looping als Standard war eine direkte,
messbar wirksame Reaktion auf dieses Problem.

### 11.2 Melody-Conditioning: zwei Anwendungsfaelle, ein Mechanismus

`facebook/musicgen-melody-large` unterstuetzt neben reiner Textsteuerung eine
zusaetzliche Chroma-Konditionierung ueber `generate_with_chroma()`: Aus einer
mitgegebenen Referenz-Audiospur wird intern ein Chromagramm (harmonische
Kontur) extrahiert, das die Generierung zusaetzlich zum Text-Prompt lenkt.
Im Projekt wurden zwei unterschiedliche Anwendungen dieses Mechanismus
untersucht:

1. **Anschluss-Conditioning** (Funktion `melody_reference_from_previous()`):
   Fuer den Uebergang zwischen zwei aufeinanderfolgenden Bloecken wird das
   Ende des vorherigen, bereits akzeptierten Blocks als kurze Referenz an den
   Anfang eines Referenzarrays gelegt (Rest mit Stille aufgefuellt, Ein-/
   Ausblendung ueber die letzte Sekunde, Pegelbegrenzung auf 0.65), damit der
   naechste Block harmonisch dort ansetzt, wo der vorherige endete.
2. **Genre-Melody-Conditioning** (Funktion `melody_reference_from_genre()`,
   experimentell): Statt des vorherigen Blocks wird ein realer, bereits
   vorhandener Trainingsclip des Zielgenres ueber die gesamte Blockdauer als
   Referenz verwendet, um die Generierung staerker an echtes Referenzmaterial
   zu binden. Dieser Ansatz wurde kontrolliert getestet und mit einer
   Genre-Treue-Verschlechterung von durchschnittlich 9.6 Punkten gegenueber
   dem Ausgangsstand wieder verworfen (Details in Abschnitt 24.2). Der Code
   bleibt ueber das Flag `--genre-melodie-conditioning-aktiv` erreichbar,
   ist aber in keinem produktiven Ablauf standardmaessig aktiviert.

## 12. Prozessphase 7: Website und Backend

### Ziel

Die Pipeline sollte nicht nur im Terminal nutzbar sein, sondern ueber eine
lokale Website bedient werden koennen. Das war notwendig, weil viele Prozesse
fuer die Bachelorarbeit wiederholt gestartet, beobachtet und bewertet werden
muessen.

### Website-Funktionen

Die Website deckt die wichtigsten lokalen Funktionen ab:

- Statusuebersicht,
- Quellen- und Importprozesse,
- Clip-Vorbereitung,
- LoRA-Training,
- LoRA-Fortsetzen,
- Checkpoint-Freigabe,
- MusicGen-Testaudios,
- Dataset-Clip-Pruefung,
- Longform-Generierung,
- Audio-Bibliothek,
- Reports und Bewertungen.

### Backend/API

Das lokale Backend in `code/src/Pipeline/web_api.py` uebersetzt Website-Aktionen
in konkrete lokale CLI-Befehle. Beispiel:

- `lora_training` startet ein neues LoRA-Training,
- `lora_training_weiter` setzt den bestehenden Lauf fort,
- `lora_freigeben` gibt einen Checkpoint frei,
- `testaudios` erzeugt MusicGen-Clips fuer die fuenf Genres,
- `generate` startet eine Longform-Audio.

### UI-Entscheidung

Der Bereich `LoRA & Bewertung` wurde so ueberarbeitet, dass Training,
Checkpoint-Freigabe und Clip-Pruefung visuell getrennt sind. Dadurch ist klarer,
welcher Button was ausloest. Diese Klarheit ist auch methodisch relevant, weil
ein versehentlich gestartetes Training oder eine falsche Bewertung den
Projektfortschritt verfaelschen koennte.

## 13. Iterative Problembehandlung

Die folgende Tabelle fasst zentrale Probleme und die jeweiligen technischen
Reaktionen zusammen.

| Beobachtung | Prozessentscheidung | Technische Umsetzung |
|---|---|---|
| Bass war in manchen Audios zu dominant | Bass darf nicht nur subjektiv gehoert, sondern technisch begrenzt werden | Bass Ratio, Referenzvergleich, Bass-Penalty, EQ-Kontrolle |
| Einzelne Audios hatten zu wenig Ton oder Aussetzer | Jeder Abschnitt muss ueber die Dauer hoerbar bleiben | Sekundenweiser RMS-Check, Mindestschwelle `-50 dB` |
| Shaker/Rauschen/Hoehen wirkten zu stark | Hochfrequente Artefakte muessen frueh bestraft werden | strengere `high_ratio`-Pruefung, Lowpass und Prompt-Blocklist |
| 30s-Clips waren teilweise zu monoton | Bewegung innerhalb des Clips muss bewertet werden | `musical_movement_score`, `spectral_movement`, RMS- und Centroid-Bewegung |
| Guitar Lofi hatte teils zu wenig Gitarre | Genreprompt muss Hauptinstrument explizit priorisieren | Guitar-Prompts mit Gitarre im Vordergrund |
| Jazz Lofi brauchte mehr Piano-/Jazzharmonik | Promptprofil wurde genrebezogen verfeinert | Jazz-Prompts mit Piano, Rhodes, Seventh Chords, Swing/Brushes |
| Chillhop klang teils gestellt | Prompts wurden weniger dramatisch und samplebasierter formuliert | Chillhop-Profil mit sample-based groove und natuerlicher Variation |
| Neuer Checkpoint war nicht automatisch besser | Trainingserfolg braucht Review statt blinder Uebernahme | Status `bewertung_offen`, beste Version bleibt erhalten |
| Testaudio-Lauf scheiterte wegen Freigabezustand | Review muss auch Kandidatenadapter pruefen koennen | `--review-adapter-erlauben` fuer Testaudios |

## 14. Prompt-Entwicklung

### Ausgangsproblem

Anfangs wurden Prompts teilweise zu allgemein oder mit problematischen
Klangbegriffen formuliert. Begriffe wie `vinyl`, `noise`, `texture`, `crackle`,
`hiss`, `shaker` oder auch das im Feedback auftauchende `shader` konnten
unerwuenschte hochfrequente Artefakte beguenstigen.

### Entscheidung

Die Prompt-Logik wurde genrebezogen und restriktiver aufgebaut. Ziel ist nicht,
MusicGen maximal viel Freiheit zu geben, sondern innerhalb eines stabilen
Lo-Fi-Rahmens musikalische Variation zu erzeugen.

### Umsetzung

In `code/src/Training/audio_erstellen.py` wurden folgende Konzepte ergaenzt:

- genrebezogene Promptprofile,
- Pflichtmerkmale pro Genre,
- bereinigte Instrumentenlisten,
- Blocklist fuer problematische Begriffe,
- De-Duplizierung von Promptbestandteilen,
- klare Vorgaben fuer 30s-Phrasen,
- Vorgaben gegen Stille und gegen zu starken Subbass.

Die Blocklist entfernt unter anderem Begriffe aus der Generierung, die
Rauschen, Shaker, Tape-/Vinyl-Texturen oder uebertriebene Hoehen beguenstigen
koennen.

### Wissenschaftliche Bedeutung

Prompting wird hier nicht als rein kreativer Vorgang behandelt, sondern als
steuerbarer Teil der experimentellen Konfiguration. Jede Promptaenderung kann
sich auf die Modellantwort auswirken und muss deshalb als Prozessentscheidung
dokumentiert werden.

## 15. Bewegung statt Monotonie

### Ausgangsproblem

Einige 30s-Clips klangen technisch korrekt, aber zu statisch. Fuer eine
einminuetige oder laengere Lo-Fi-Audio ist das problematisch, weil eine
Wiederholung statischer 30s-Clips schnell kuenstlich und ermuedend wirkt.

### Entscheidung

Die Pipeline bewertet nun nicht nur Lautheit und Frequenzanteile, sondern auch
musikalische Veraenderung innerhalb eines Clips.

### Umsetzung

In `audio_erstellen.py` wurden Bewegungsmetriken eingefuehrt:

- `musical_movement_score`,
- `spectral_movement`,
- `rms_movement_db`,
- `centroid_movement_hz`,
- Anzahl der Bewegungsfenster.

Diese Werte werden in die Kandidatenauswahl integriert. Zu statische Clips
werden dadurch abgewertet, auch wenn sie keine harten technischen Fehler haben.

### Bedeutung fuer die Bachelorarbeit

Dieser Schritt ist wichtig, weil er zeigt, dass die Zielqualitaet nicht nur aus
"kein Fehler" besteht. Eine Lo-Fi-Audio muss ruhig sein, aber nicht leer oder
bewegungslos.

## 16. Umgang mit Rauschen, Shaker und Hoehenartefakten

### Ausgangsproblem

In mehreren Bewertungen wurden Shaker, Rauschen oder ein hochfrequentes
Klangmuster als stoerend wahrgenommen. Teilweise wurde dieses Muster im
Feedback als "shader" bezeichnet. Gemeint war im Audiokontext ein
rausch-/shakerartiger Hoehenanteil, nicht ein visueller Shader.

### Entscheidung

Das Problem wurde an drei Stellen behandelt:

1. Prompt-Ebene: problematische Begriffe werden entfernt.
2. Kandidaten-Ebene: hoher Hoehenanteil wird staerker bestraft.
3. Nachbearbeitung: Lowpass und EQ reduzieren den kritischen Bereich.

### Umsetzung

Relevante Dateien:

| Datei | Aenderung |
|---|---|
| `code/src/Training/audio_erstellen.py` | strengere `high_ratio`, Prompt-Blocklist, Candidate-Penalty |
| `code/src/Training/audio_nachbearbeitung.py` | strengere Lowpass-/High-Control-Profile |
| `code/website/lo-fi-dreamer/src/routes/studio.generate.tsx` | UI-Defaults ohne Noise-/Texture-Begriffe |
| `code/website/lo-fi-dreamer/src/lib/mockData.ts` | Mock-Prompts ohne Shaker-/Texture-Sprache |

### Entscheidung

Die Pipeline soll keine Rauschtextur aktiv in die Prompts schreiben. Wenn
Lo-Fi-Waerme gewuenscht ist, soll sie ueber weiche Instrumentierung,
Arrangement, Mix und LoRA-Stil entstehen, nicht ueber explizite Noise- oder
Shaker-Begriffe.

## 17. Bewertungssystem

### Ziel

Nach jeder generierten Audio soll automatisch ein zusammenfassender
Qualitaetsbericht entstehen. Die Bewertung soll wissenschaftlich interpretierbar
sein, ohne die Website mit Rohkurven zu ueberladen.

### Fuenf Hauptkategorien

Die automatische Bewertung nutzt fuenf Kategorien:

1. Technische Audioqualitaet
2. Musikalische Kohaerenz
3. Genre-Treue
4. Uebergangsqualitaet
5. Referenzaehnlichkeit

Die Website zeigt zwei getrennte Graphen:

- Genrestandard,
- Generierte Audio.

Beide Graphen nutzen dieselben Kategorien und eine Skala von 0 bis 100.

### Bedeutung

Die Bewertung ist als Interpretationshilfe zu verstehen. Sie ersetzt keine
menschliche Hoerentscheidung, macht aber sichtbar, ob technische und
genrebezogene Zielwerte naeher an den Referenzen liegen.

## 18. Menschliche Bewertung und Feedbackschleife

### Bewertete Problemklassen

Aus den Bewertungsdurchgaengen wurden wiederkehrende Problemklassen abgeleitet:

- starker Bass,
- starker Shaker,
- Ton kippt,
- kein Ton,
- Monotonie,
- Rauschen,
- Gitarre passt nicht,
- Kreativitaet fehlt,
- Uebersteuerung.

Diese Klassen wurden nicht nur beschrieben, sondern in technische
Pruefkriterien uebersetzt.

### Beispielhafte Prozesslogik

Wenn der Nutzer meldet, dass Guitar Lofi zu wenig Gitarre enthaelt, wird das
nicht nur als Einzelurteil gespeichert. Daraus folgt:

- Promptprofil fuer Guitar Lofi wird angepasst,
- Gitarrenbegriffe werden priorisiert,
- Piano-/Rhodes-gepraegte Alternativen werden reduziert,
- neue Testaudios werden erzeugt,
- menschliches Feedback entscheidet, ob die Aenderung erfolgreich war.

Dasselbe Prinzip wurde fuer Jazz-Piano, Dreamy-Rauschen, Chillhop-Natuerlichkeit
und Monotonie angewendet.

## 19. Aktueller technischer Stand

### LoRA

Der aktive LoRA-Lauf ist abgeschlossen, aber der neueste Stand ist noch als
Review-Kandidat markiert:

| Merkmal | Stand |
|---|---|
| Run | `training/musicgen/lora_training` |
| finaler Checkpoint | `step_001250/lora_adapter.pt` |
| bester dokumentierter Adapter | `step_000625/lora_adapter.pt` |
| Status | `bewertung_offen` |
| Dataset | `daten/processed/lora_training` |
| Clips gesamt | 5000 |
| Clips pro Genre | 1000 |
| naechster Schritt | Testaudios erzeugen, bewerten, dann ggf. freigeben |

### Website/API

Die Website ist mit den lokalen Backend-Funktionen verbunden. Insbesondere
koennen folgende Prozesse ueber die Website gestartet werden:

- neues LoRA-Training,
- LoRA-Fortsetzung,
- Checkpoint-Freigabe,
- MusicGen-Clips fuer alle Genres,
- Dataset-Clips,
- Longform-Generierung,
- Audio-Bewertung,
- Status- und Bibliotheksansicht.

### Generierung

Die aktuelle Generierung verwendet:

- 30s MusicGen-Kandidaten,
- mehrere Kandidaten pro Abschnitt,
- Looping auf laengere Bloecke,
- Crossfade,
- BPM-Zielwert,
- genrebezogene Promptprofile,
- Referenzvergleich,
- Bewegungsmetriken gegen Monotonie,
- strengere Hoehen-/Rauschkontrolle,
- Sekundencheck gegen stille Abschnitte.

## 20. Prozessartefakte

Die folgenden Artefakte sind fuer die wissenschaftliche Ausarbeitung besonders
wichtig:

| Artefakt | Bedeutung |
|---|---|
| `README.md` | Gesamtuebersicht der Pipeline |
| `dokumentation/aktuell/01_struktur.md` | aktive Projektstruktur |
| `dokumentation/aktuell/02_ablauf.md` | Audio-Ablauf und Startpunkte |
| `dokumentation/aktuell/03_code_erklaerung.md` | Erklaerung der aktiven Code-Dateien |
| `training/musicgen/dataset_pruefung/trainingsdaten_plan.json` | gepruefter 4000er Datensatz |
| `training/musicgen/dataset_pruefung/zieldatensatz_plan.json` | 5000er Ziel-Dataset |
| `training/musicgen/dataset_pruefung/reports/genre_balance.csv` | Genre- und Splitverteilung |
| `training/musicgen/lora_training/pruefung.json` | Dataset- und Trainingspruefung |
| `training/musicgen/lora_training/training_config.json` | Trainingsparameter |
| `training/musicgen/lora_training/stand.json` | aktueller LoRA-Status |
| `training/musicgen/lora_training/abschluss.json` | Abschluss des Trainingslaufs |
| `bewertungen/analyse/bewertungs_analyse_pro_durchgang.csv` | Entwicklung der menschlichen Problemklassen |

### 20.1 Versionsverwaltung und GitHub

GitHub wurde im Projekt nicht als Speicherort fuer grosse Trainingsdaten oder
Modellgewichte verstanden, sondern als Nachweis fuer Code, kleine Reports,
Dokumentation und bewusst ausgewaehlte Audio-Artefakte. Dadurch bleibt das
Repository handhabbar, waehrend wichtige Entwicklungsstaende nachvollziehbar
bleiben.

Im Verlauf wurden unter anderem Testaudio-Sammlungen und einzelne
Longform-Audios bewusst versioniert. Gleichzeitig bleiben grosse Rohdaten,
Checkpoints und umfangreiche MP3-/WAV-Bestaende lokal. Diese Trennung ist fuer
die Reproduzierbarkeit und Projektpflege wichtig:

- Code und Dokumentation zeigen, wie Prozesse funktionieren.
- JSON-/CSV-Reports zeigen, welche Entscheidungen getroffen wurden.
- Ausgewaehlte Audios zeigen pruefbare Zwischenstaende.
- Grosse Daten und Modellgewichte bleiben lokal, um das Repository nicht zu
  ueberladen.

### 20.2 Kommentierung und Dokumentationsstrategie

Die Codebasis wurde nicht mit allgemeinen Kommentaren ueberladen. Kommentare
wurden dort ergaenzt, wo sie eine Prozessentscheidung oder eine Sicherheitsregel
erklaeren. Beispiele sind:

- keine automatische Nutzung ungepruefter Checkpoints,
- Fortsetzen eines LoRA-Laufs statt versehentlichem Neustart,
- bewusste Deaktivierung problematischer Conditioning-Varianten,
- Trennung zwischen MusicGen-Testaudio und Dataset-Hoerprobe,
- Begrenzung von GPU-/VRAM-Nutzung,
- Review-Pflicht vor Freigabe.

Fuer die wissenschaftliche Ausarbeitung ist diese Strategie sinnvoller als
kommentierter Selbstzweck: Entscheidend sind nicht Kommentare zu jeder Zeile,
sondern nachvollziehbare Begruendungen an den Stellen, an denen der Prozess
veraendert oder abgesichert wurde.

## 21. Semantische Genre-Pruefung mit CLAP

### Ausgangslage

Die fuenf Bewertungskategorien aus Abschnitt 8.1 sind merkmalsbasiert: Sie
vergleichen Zahlenwerte wie BPM oder Bassanteil, koennen aber nicht direkt
beurteilen, ob eine Audio *klingt wie* das gewuenschte Genre. Deshalb existiert
im Code bereits eine zweite, semantische Pruefschicht:
`code/src/Training/genre_pruefung.py` mit der Klasse `GenrePruefer`. Sie nutzt
ein lokal gespeichertes CLAP-Modell (`laion/clap-htsat-unfused`, Contrastive
Language-Audio Pretraining), bildet aus menschlich als "gut" bewerteten
Referenzclips pro Genre ein Zentrum im CLAP-Einbettungsraum und vergleicht
neue Kandidaten per Kosinus-Aehnlichkeit gegen diese Zentren.

### Befund: Referenzdatenbasis leer

Bei Ueberpruefung zeigte sich, dass diese Pruefung im produktiven Ablauf ueber
das Flag `--genre-pruefung-deaktivieren` durchgehend abgeschaltet war. Ursache:
Die hinterlegte Referenzliste (`training/bewertungen/musicgen/lora_review_001/
bewertung.csv`, 15 Zeilen, drei pro Genre) stammt aus einer inzwischen
abgeschafften manuellen Bewertungsfunktion und wurde nie ausgefuellt (Status-
Spalte durchgehend leer). Ohne mindestens zwei Genres mit positiv bewerteten
Referenzen kann `GenrePruefer` keine Zentren bilden und verweigert den Dienst.

### Reparatur und Evaluation

Als Ersatzdatenbasis wurde ein neues Skript (`code/src/Training/
clap_referenzen_bauen.py`) erstellt, das automatisch 20 Referenzclips pro
Genre aus dem bereits vorhandenen, genre-sortierten LoRA-Trainingsdatensatz
zieht, verteilt ueber moeglichst viele unabhaengige Quellen. Damit liess sich
`GenrePruefer` erstmals funktionsfaehig betreiben.

Um die Aussagekraft zu pruefen, wurde ein Held-out-Test durchgefuehrt: 10
echte Trainingsclips (2 pro Genre), die *nicht* Teil der neuen Referenzliste
waren, wurden gegen ihr jeweils korrektes Zielgenre geprueft.

| Genre | Ergebnis Clip 1 | Ergebnis Clip 2 |
|---|---|---|
| Chillhop Lofi | falsch (-> Dreamy) | falsch (-> Jazz) |
| Dreamy Lofi | richtig | falsch |
| Guitar Lofi | richtig | richtig |
| Jazz Lofi | falsch (-> Dreamy) | falsch (gemischt) |
| Study Lofi | richtig | falsch |

Ergebnis: **4 von 10 Zuordnungen korrekt (40 Prozent)**, gegenueber einer
Zufallsbasis von 20 Prozent bei fuenf Klassen. Auffaellig war eine
systematische Tendenz: Fast alle Fehlzuordnungen fielen auf `dreamy_lofi`,
unabhaengig vom tatsaechlichen Zielgenre. Das deutet darauf hin, dass das
CLAP-Modell die fuenf eng verwandten Lo-Fi-Subgenres akustisch nicht trennscharf
genug unterscheidet und einen Grossteil der Varianz auf ein generisches
"Lo-Fi allgemein"-Zentrum projiziert.

### Entscheidung

Die semantische Pruefung wurde auf Basis dieses Befunds **nicht** produktiv
aktiviert. Eine Trefferquote knapp ueber dem Zufallsniveau wuerde als
automatischer Filter mehr korrekte Kandidaten verwerfen als tatsaechliche
Fehlgenerierungen erkennen. Referenzdaten und Skript bleiben im Projekt
erhalten, falls ein anderes CLAP-Modell oder eine feinere Schwellenwert-
Kalibrierung in einer spaeteren Arbeit erneut geprueft werden soll.

## 22. Messvaliditaet und Testreliabilitaet

### Beobachtung

Im Verlauf mehrerer kontrollierter Verbesserungsversuche (Abschnitt 24) fiel
auf, dass in jedem Versuch ein anderes Genre den staerksten Ruecksetzer zeigte
(erst Jazz Lofi, dann Chillhop Lofi), obwohl jeweils ein anderer Parameter
veraendert wurde. Das warf die Frage auf, wie stabil die Genre-Treue-Messung
selbst ist, wenn am Modell nichts veraendert wird.

### Testaufbau

Der produktive Testaudio-Prozess erzeugt pro Genre genau eine finale Audio
(ausgewaehlt aus mehreren MusicGen-Kandidaten). Um die Wiederholgenauigkeit
dieser Einzelmessung zu pruefen, wurde derselbe, unveraenderte LoRA-Checkpoint
(`step_000625`, SHA-256-Praefix `a941b297...`, Status `freigegeben`) zweimal
hintereinander durch den vollstaendigen Testaudio-Prozess geschickt, ohne
jede Aenderung an Modell, Datensatz oder Konfiguration.

### Ergebnis

| Genre | Durchlauf 1 | Durchlauf 2 (identisches Modell) | Differenz |
|---|---|---|---|
| Chillhop Lofi | 47.6 | 30.6 | -17.0 |
| Dreamy Lofi | 55.9 | 54.4 | -1.5 |
| Guitar Lofi | 38.3 | 37.1 | -1.2 |
| Jazz Lofi | 59.4 | 16.9 | **-42.5** |
| Study Lofi | 30.2 | 23.5 | -6.7 |
| **Mittelwert** | **46.3** | **32.5** | **-13.8** |

### Interpretation

Bei unveraendertem Modell schwankt die Genre-Treue eines einzelnen Genres
(Jazz Lofi) um 42.5 Punkte, der Mittelwert ueber alle Genres um 13.8 Punkte.
Diese Schwankungsbreite ist vergleichbar mit oder groesser als die
Effektgroessen, die den vier in Abschnitt 24 dokumentierten
Verbesserungsversuchen zugeschrieben wurden (-9.6 bis -13.0 Punkte im
Mittelwert). Daraus folgt eine methodisch wichtige Einschraenkung: **Eine
einzelne generierte Audio pro Genre ist keine ausreichend stabile Messgroesse,
um zwei Modellstaende zuverlaessig zu vergleichen.** Ob die vier
Verbesserungsversuche real schlechtere Modelle erzeugten oder ob teilweise
Messrauschen faelschlich als Regression interpretiert wurde, laesst sich mit
der bisherigen Testmethode nicht abschliessend trennen.

Als plausible Fehlerquellen kommen in Frage: die Auswahl unter mehreren
MusicGen-Kandidaten pro Abschnitt (unterschiedliche Kandidaten koennen je nach
Zufallsentscheidung im Akzeptanzprozess gewaehlt werden), Sampling-Varianz im
Generierungsprozess sowie moegliche GPU-seitige Nichtdeterminismen einzelner
Rechenoperationen trotz gesetztem Zufalls-Seed.

### Konsequenz fuer die Methodik

Fuer belastbare Vorher-Nachher-Vergleiche sollte die Testaudio-Erzeugung
kuenftig mehrere Audios pro Genre erzeugen und den Durchschnitt (oder Median)
vergleichen, statt sich auf eine Einzelmessung zu verlassen. Dieser Befund ist
selbst ein wissenschaftlich relevantes Ergebnis der Arbeit: Er zeigt, dass die
Entwicklung einer verlaesslichen automatischen Evaluationsmethodik fuer
kurze, generative Lo-Fi-Audios eine eigene, nicht triviale Teilaufgabe ist.

## 23. Qualitaetssicherung: Grenze der automatischen Stille-Erkennung

### Beobachtung

Bei der manuellen Anhoerkontrolle eines produktiv erzeugten und veroeffentlichten
Testaudios (Chillhop Lofi) wurde ein Abschnitt zwischen Sekunde 5 und 18 als
klanglich leer wahrgenommen. Die automatische Sekunden-Pruefung
(`sekunden_pruefung_aktiv`, Schwelle `min_sekunden_rms_db = -50/-52 dB`) hatte
diesen Abschnitt jedoch als gueltig akzeptiert.

### Technische Analyse

Eine sekundenweise RMS-Messung des betroffenen Abschnitts ergab Werte um
-26 bis -28 dB — deutlich ueber der Stille-Schwelle und im Rahmen fuer eine
"laute" Passage. Eine zusaetzliche Frequenzbandanalyse (FFT desselben
Abschnitts, Energieanteil pro Frequenzband) zeigte jedoch:

| Frequenzband | Energieanteil |
|---|---|
| Sub-Bass (0-40 Hz) | 35.4 % |
| Bass (40-150 Hz) | 64.5 % |
| Untere Mitten (150-500 Hz) | 0.1 % |
| Mitten (500-2000 Hz) | 0.0 % |
| Hoehen (2000-8000 Hz) | 0.0 % |
| Sehr hoch (8000-16000 Hz) | 0.0 % |

**99.9 Prozent der Signalenergie lagen unterhalb von 150 Hz.** Der Abschnitt
enthielt praktisch keine Melodie, keine Percussion und keine wahrnehmbare
musikalische Struktur, sondern ausschliesslich ein tieffrequentes Brummen.
Eine gesicherte Ursachenanalyse auf Modellebene wurde im Rahmen dieser Arbeit
nicht durchgefuehrt; plausibel ist ein Kollaps in eine tonal verarmte, fast
monofrequente Ausgabe fuer mehrere Sekunden, wie er bei autoregressiven
Sequenzmodellen grundsaetzlich auftreten kann. Diese Einordnung ist eine
Interpretation der Beobachtung, kein durch eine Fremdquelle belegter Befund.

### Wissenschaftliche Einordnung

Dieser Befund legt eine konkrete Luecke im bestehenden Pruefsystem offen: Die
Sekunden-Pruefung misst ausschliesslich Lautstaerke (RMS), nicht ob im
hoerbar relevanten Frequenzbereich (etwa oberhalb 150 Hz) ueberhaupt
musikalischer Inhalt vorhanden ist. Ein Kandidat, der in reines
Bass-/Subbass-Rauschen kollabiert, ist damit "laut genug", um die Pruefung zu
bestehen, obwohl er fuer menschliche Hoerer wie eine stille Passage wirkt.
Da die Genre-Treue-Berechnung (Abschnitt 8.1) unter anderem `bass_ratio` und
`spectral_centroid` nutzt, kann ein solcher Kandidat zusaetzlich die
gemessene Genre-Treue in nicht reprasentativer Weise verzerren — ein
moeglicher Teilbeitrag zu der in Abschnitt 22 dokumentierten Messschwankung.

Ein sinnvoller technischer Gegenmassnahme waere eine Erweiterung der
Sekunden-Pruefung um ein Mindestmass an Energie oberhalb einer
Frequenzschwelle (z. B. 150 Hz), zusaetzlich zur bestehenden reinen
RMS-Schwelle. Diese Erweiterung war zum Zeitpunkt der Dokumentation
identifiziert, aber noch nicht umgesetzt.

## 24. Vier kontrollierte Verbesserungsversuche

Nachdem der Stand `step_000625` als vergleichsweise stabile Referenz etabliert
war (Genre-Treue-Mittelwert 46.3, siehe Abschnitt 22), wurden vier
unterschiedliche, jeweils einzeln isolierte Hypothesen getestet, mit denen die
Musikqualitaet weiter verbessert werden sollte. Alle vier wurden nach demselben
Muster behandelt: Hypothese formulieren, isoliert veraendern, ueber den
Testaudio-Prozess pruefen, mit der Referenz vergleichen, bei Verschlechterung
mit dem bestehenden Archivierungsmechanismus (Umbenennung statt Loeschen)
zurueckrollen. Angesichts des in Abschnitt 22 dokumentierten Messrauschens
sind die folgenden Differenzen als vorlaeufige Befunde und nicht als
endgueltig gesicherte Effekte zu verstehen.

| Versuch | Veraenderung gegenueber Referenz | Genre-Treue-Mittelwert | Differenz |
|---|---|---|---|
| Referenz (`step_000625`) | — | 46.3 | — |
| 24.1 Aggressive Hyperparameter | Rank 16, Alpha 32, Lernrate 2e-5 | 36.8 | -9.5 |
| 24.2 Genre-Melody-Conditioning | echter Referenzclip als Chroma-Vorgabe | 36.7 | -9.6 |
| 24.3 Datensatz-Diversifizierung | 19 zusaetzliche lizenzfreie Quellen | 34.0 | -12.3 |
| 24.4 Mehr Trainingsschritte | 3000 statt 625 Schritte | 33.2 | -13.0 |

### 24.1 Aggressive Hyperparameter

LoRA Rank 16 (statt 8), Alpha 32 (statt 16), Lernrate 2e-5 (statt 5e-6). Der
Trainingsverlust blieb ueber den gesamten Lauf nahezu unveraendert (Start
3.2599, Ende 3.2597), was fuer sich genommen weder fuer noch gegen einen
Qualitaetsgewinn spricht — Trainingsverlust bei MusicGen-LoRA korreliert im
Projektverlauf durchgehend schwach mit der tatsaechlichen Hoerqualitaet. Die
Testaudio-Auswertung zeigte ein gemischtes Bild: Chillhop Lofi verbesserte
sich um 13.0 Punkte, die uebrigen vier Genres verschlechterten sich, Dreamy
Lofi am staerksten (-25.8 Punkte). Entscheidung: zurueckgesetzt.

### 24.2 Genre-Melody-Conditioning

Siehe Abschnitt 11.2 fuer die technische Umsetzung. Fuer jeden Kandidaten
wurde ein realer, genre-passender Trainingsclip als Chroma-Referenz ueber die
gesamte Blockdauer vorgegeben, in der Annahme, dass echtes Referenzmaterial
die Genre-Treue direkter verbessert als reine Textsteuerung. Ergebnis je
Genre: Chillhop 42.4 (-5.2), Dreamy 41.9 (-14.0), Guitar 47.9 (+9.6), Jazz
16.4 (-43.0), Study 35.0 (+4.8). Auffaellig ist erneut der starke Einbruch bei
Jazz Lofi, dem Genre mit der historisch groessten Konzentration auf wenige
Trainingsquellen (Abschnitt 24.3). Entscheidung: zurueckgesetzt, Funktion
bleibt im Code als deaktivierbare Option erhalten.

### 24.3 Datensatz-Diversifizierung

Ausgangsdiagnose: Obwohl jedes Genre 1000 Trainingsclips umfasst, stammten
diese im Ausgangsdatensatz aus nur 5 bis 7 unabhaengigen Quellvideos. Bei
Jazz Lofi entfielen beispielsweise 492 von 1000 Clips (49 Prozent) auf nur
zwei Quellvideos. Diese Konzentration wurde als moegliche Ursache fuer
Overfitting-aehnliches Verhalten bei staerkerem Training eingeordnet.

Zur Gegenmassnahme wurden 19 zusaetzliche, lizenzfreie Lo-Fi-Aufnahmen von
Free Music Archive recherchiert und eingebunden (ueberwiegend CC0- und
CC-BY-lizenziert, sechs unabhaengige Kuenstler: HoliznaCC0, Ketsa, 1000 Handz,
legacyAlli, Alex-Productions, Scott Holmes Music). Zwei alternative Plattformen
wurden geprueft und verworfen: Jamendo, weil die exakte Lizenzvariante fuer die
meisten Treffer nicht programmatisch verifizierbar war und mehrere
Stichproben als CC-BY-NC-ND (keine Bearbeitung erlaubt) identifiziert wurden;
Pixabay, weil dessen Nutzungsbedingungen automatisiertes Sammeln von Inhalten
"fuer maschinelles Lernen" explizit untersagen.

Technisch erforderte die Einbindung eine Anpassung des bestehenden
Auswahlverfahrens (Abschnitt 7.1): Der produktive Pfad betrachtete ein Genre
als "voll", sobald 1000 Clips aus dem bestehenden Quellenpool verfuegbar
waren, und uebersprang dadurch neue, kleine Quellen vollstaendig. Erst ein
gezielter Neuaufbau des Zieldatensatzes ueber `zieldatensatz.py` mit dem
Round-Robin-Verfahren stellte sicher, dass die neuen Quellen tatsaechlich in
die Trainingsauswahl gelangten. Ergebnis: die Zahl unabhaengiger Quellen pro
Genre stieg von 5-7 auf 9-11 (siehe Tabelle).

| Genre | Quellen vorher | Quellen nachher |
|---|---|---|
| Chillhop Lofi | 6 | 11 |
| Dreamy Lofi | 5 | 10 |
| Guitar Lofi | 6 | 9 |
| Jazz Lofi | 7 | 10 |
| Study Lofi | 6 | 9 |

Vor dem Training wurden die 20 neuen Quellen zusaetzlich technisch (Clipping,
Aktivanteil, Pegel) und stilistisch (Distanz zum bestehenden Genre-
Klangprofil, Methodik wie Abschnitt 8.1) geprueft. Technisch waren alle 20
Quellen einwandfrei (Median-Score 98.5 von 100, kein Clipping). Stilistisch
war das Bild gemischt (Median-Distanz-Score 31.7, 10 von 20 Quellen mit
score unter 30) — ein Hinweis darauf, dass "lizenzrechtlich verfuegbares
Lo-Fi" nicht automatisch klanglich nah am bisherigen Genreprofil liegt.

Ein frisches LoRA-Training (identische Hyperparameter wie die Referenz, 625
Schritte, Neustart ab Basismodell statt Fortsetzung, um den Effekt der
Datenaenderung nicht mit Fortsetzungseffekten zu vermischen) auf dem
erweiterten Datensatz ergab: Chillhop 11.1 (-36.5), Dreamy 55.1 (-0.8),
Guitar 34.2 (-4.1), Jazz 42.7 (-16.7), Study 26.7 (-3.5). Der staerkste
Einbruch bei Chillhop Lofi faellt zeitlich mit der schwaechsten Einzelquelle
der neuen Auswahl zusammen (Stilistik-Score 4.4 von 100). Entscheidung:
zurueckgesetzt. Die erweiterten Rohquellen und der Aufbereitungscode
(`code/src/Training/clap_referenzen_bauen.py` als Nebenprodukt, Datensatz
unter `daten/processed/musicgen_youtube_import_30s/lokal_hop30_*`) bleiben
erhalten fuer eine moeglich gezieltere Nachkuratierung in einer Folgearbeit.

### 24.4 Mehr Trainingsschritte

Fortsetzung des Trainings von 625 auf 3000 Schritte (Faktor 4.8, mit
identischer Lernrate und identischem Datensatz). Der Lauf mit dieser
Zielsetzung wurde zweimal durch Infrastrukturausfaelle unterbrochen (ein
nativer Segmentation Fault bei Schritt 610, vermutlich CUDA-seitig, sowie ein
spaeterer echter Deadlock bei Schritt 2181, erkennbar an durchgehend
schlafenden Threads im Zustand `futex_wait_queue` ueber mehr als sechs
Minuten) und jeweils vom letzten gespeicherten Checkpoint fortgesetzt. Nach
Abschluss: Chillhop 45.5 (-2.1), Dreamy 35.1 (-20.8), Guitar 47.9 (+9.6), Jazz
12.7 (-46.7), Study 25.0 (-5.2). Die Trainings-Zielschrittzahl folgt im
Projekt der Faustregel `Schritte = aufgerundet(Trainingsclips / 8)`, wobei 8
sich aus Batchgroesse 1 mal Gradientenakkumulation 8 ergibt; 625 Schritte
entsprechen damit rechnerisch exakt einer Epoche ueber den 5000-Clip-
Datensatz, 3000 Schritten entsprechen rund 4.8 Epochen. Die Verschlechterung,
insbesondere bei Jazz Lofi (dem Genre mit der groessten Quellenkonzentration,
Abschnitt 24.3), ist konsistent mit einer Ueberanpassung an einzelne,
haeufig wiederholte Trainingsclips. Entscheidung: zurueckgesetzt.

### 24.5 Erwogene, nicht umgesetzte Alternativen

Zusaetzlich zu den vier durchgefuehrten Versuchen wurden zwei
Modellalternativen recherchiert, aber nicht implementiert. Die Recherche
stuetzt sich auf folgende Primaerquellen:

- **JASCO** (Meta AI/FAIR, Teil derselben `audiocraft`-Bibliothek wie das im
  Projekt eingesetzte MusicGen): unterstuetzt zusaetzlich zur Text- auch eine
  explizite Akkord- und Schlagzeug-Konditionierung und ist ueber offizielle
  `dora run solver=jasco/chords_drums ... continue_from=//pretrained/facebook/
  jasco-chords-drums-400M`-Befehle nachtrainierbar (Quelle:
  `facebookresearch/audiocraft`, `docs/JASCO.md` und
  `model_cards/JASCO_MODEL_CARD.md`, GitHub). Das Modell wurde urspruenglich
  beschrieben in Copet et al., *"Joint Audio and Symbolic Conditioning for
  Temporally Controlled Text-to-Music Generation"*, arXiv:2406.10970 (Meta
  AI, 2024). Eine unabhaengige Vergleichsstudie (*"Benchmarking Music
  Generation Models and Metrics via Human Preference Studies"*,
  arXiv:2506.19085) berichtet fuer die reine Audioqualitaet ein gemischtes
  Bild gegenueber MusicGen-Large: MusicGen-Large erreicht den niedrigeren
  (besseren) MAD-Wert (Mauve Audio Divergence) gegenueber MusicCaps-Referenzen
  (2.45 vs. 3.89 fuer JASCO), waehrend JASCO beim Kernel Audio Distance (KAD)
  niedriger und damit besser liegt (5.51 vs. 7.65 fuer MusicGen-Large). Ein
  eindeutiger Qualitaetsvorteil eines der beiden Modelle laesst sich aus
  diesen Werten nicht ableiten; der dokumentierte Vorteil von JASCO liegt in
  der praeziseren Steuerbarkeit, nicht in einer hoeheren Grundqualitaet. Eine
  Umsetzung im Projekt haette zudem eine neue Vorverarbeitungsstufe
  (automatische Akkord-/Melodieextraktion aus den Referenzclips) erfordert
  und wurde angesichts des unklaren Qualitaetsgewinns zurueckgestellt.
- **Stable Audio 3** (Stability AI, angekuendigt Mai 2026): neueres, offen
  gewichtetes Modell (Small- und Medium-Varianten) mit vollstaendig
  lizenzierten Trainingsdaten und offizieller LoRA-Trainingsdokumentation
  (Quelle: Stability AI, Produktankuendigung "Stable Audio 3", Mai 2026;
  `Stability-AI/stable-audio-3`, `docs/workflows/lora.md`, GitHub). Eine
  Umsetzung haette einen vollstaendigen Wechsel der Trainings- und
  Generierungsinfrastruktur (weg von `audiocraft`, dessen Werkzeuge im
  Projekt fuer Datenaufbereitung, LoRA-Training und Generierung tief
  integriert sind) bedeutet und wurde aus Aufwandsgruenden nicht verfolgt.

Fuer beide Alternativen gilt: Es handelt sich um eine Sekundaerrecherche zum
Zeitpunkt der Dokumentation, keine eigene Nachimplementierung oder eigene
Messung. Die berichteten FAD-/KAD-/MAD-Werte stammen aus der jeweils
zitierten Fremdstudie und wurden im Rahmen dieser Arbeit nicht reproduziert.

## 25. Wissenschaftliche Einordnung

Das Projekt kann als angewandtes, iteratives Entwicklungs- und
Evaluationssystem beschrieben werden. Die Modellverbesserung entstand nicht nur
durch Training, sondern durch das Zusammenspiel mehrerer Ebenen:

- bessere Datengrundlage,
- genrebalancierte Clip-Auswahl,
- Ausschluss schlechter Quellen,
- LoRA statt Volltraining,
- Checkpoint-Review statt automatischer Uebernahme,
- Promptsteuerung,
- technische Kandidatenfilterung,
- Loop- und Uebergangslogik,
- automatische Bewertung,
- menschliche Feedbackschleife,
- Website als reproduzierbare Bedienoberflaeche.

Damit ist die Pipeline nicht nur ein Generator, sondern ein kontrollierter
Forschungsworkflow. Gerade die negativen Zwischenergebnisse sind fuer die
Bachelorarbeit relevant, weil sie zeigen, welche Designentscheidungen aus
welchen Problemen abgeleitet wurden.

## 26. Grenzen des aktuellen Stands

Trotz deutlicher Stabilisierung bleiben mehrere Grenzen:

- Automatische Metriken erkennen nicht alle musikalischen Qualitaeten.
- Ein guter technischer Score garantiert keine subjektiv gute Musik.
- MusicGen kann trotz guter Quellen unerwuenschte Artefakte erzeugen,
  einschliesslich seltener Kollaps-Faelle in tieffrequentes Rauschen ohne
  musikalischen Inhalt, die von der bestehenden Lautstaerkepruefung nicht
  zuverlaessig erkannt werden (Abschnitt 23).
- LoRA-Training kann bei zu vielen Schritten regressieren (Abschnitt 24.4).
- Die automatische Genre-Treue-Messung selbst besitzt eine erhebliche
  Wiederholstreuung (bis zu 42.5 Punkte bei unveraendertem Modell, Abschnitt
  22). Einzelmessungen aus einer Audio pro Genre sind daher nur begrenzt
  belastbar; alle in dieser Dokumentation berichteten Differenzwerte sind vor
  diesem Hintergrund als vorlaeufige, nicht als endgueltig gesicherte Befunde
  zu lesen.
- Eine ergaenzende semantische Genre-Pruefung per CLAP-Einbettungen wurde
  evaluiert, erreichte im Held-out-Test aber nur 40 Prozent Trefferquote bei
  fuenf eng verwandten Lo-Fi-Subgenres und wurde deshalb nicht produktiv
  eingesetzt (Abschnitt 21).
- Promptaenderungen koennen einzelne Genres verbessern und andere
  verschlechtern.
- 30s-Clips muessen fuer Longform besonders sorgfaeltig geloopt werden.
- Mehr Trainingsdatenvielfalt fuehrt nicht automatisch zu besserer Qualitaet,
  wenn neue Quellen klanglich stark vom bisherigen Genreprofil abweichen
  (Abschnitt 24.3).
- Menschliche Bewertung bleibt fuer finale Freigaben notwendig.

Diese Grenzen sind keine reinen Implementierungsfehler, sondern Teil der
Forschungsfrage: Generative Musikmodelle muessen fuer laengere, stilistisch
konstante Audios kontrolliert und iterativ evaluiert werden — und die dafuer
verwendete Evaluationsmethodik muss selbst kritisch geprueft werden.

## 27. Naechster sinnvoller Schritt

Der naechste sinnvolle Prozessschritt ist:

1. neue MusicGen-Clips fuer alle fuenf Genres ueber die Website erzeugen,
2. die Clips menschlich bewerten,
3. besonders auf Rauschen/Shaker, Bass, Monotonie und Genretreue achten,
4. entscheiden, ob `step_001250` freigegeben oder `step_000625` beibehalten wird,
5. falls die Ergebnisse leicht besser sind, die Pipeline als funktionsfaehigen
   Bachelorarbeitsstand abschliessen.

Damit waere die Pipeline nicht perfekt, aber wissenschaftlich ausreichend
dokumentiert: Es gibt Datenerfassung, Aufbereitung, Feature Extraction,
LoRA-Anpassung, Generierung, Website-Steuerung, Evaluation und dokumentierte
Iterationen.

## 28. Kurzfazit

Der Projektfortschritt laesst sich wie folgt zusammenfassen:

```text
Aus einem anfangs instabilen MusicGen-Ansatz entstand eine lokale,
genrebezogene MusicGen-LoRA-Pipeline mit kontrollierter Datengrundlage,
fortsetzbarem Training, Checkpoint-Freigabe, Website-Steuerung,
Testaudio-Prozess, Longform-Logik und automatischer Bewertung.
```

Der wichtigste wissenschaftliche Beitrag liegt nicht darin, dass jedes
generierte Audio perfekt ist. Entscheidend ist, dass die Pipeline zeigt, wie
generative Musikproduktion systematisch verbessert, kontrolliert, bewertet und
dokumentiert werden kann.
