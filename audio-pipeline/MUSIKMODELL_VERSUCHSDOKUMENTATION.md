# Dokumentation aller Trainingsversuche des MusicGen-LoRA-Modells

Stand: 2026-08-19. Diese Datei dokumentiert lückenlos alle Verbesserungsversuche am
Lo-Fi-MusicGen-LoRA-Modell: Hypothese, technische Umsetzung, gemessenes Ergebnis, Entscheidung.
Ergänzt `dokumentation/aktuell/05_prozessdokumentation_musicgen.md` (dortige Abschnitte 21–24 decken
einen Teil dieser Versuche bereits ab) um eine vollständige, chronologische Einzeldokumentation mit allen
Zahlen, Dateipfaden und Zeitstempeln.

**Zentrale methodische Einordnung, die für die gesamte Dokumentation gilt:** Im Verlauf wurde entdeckt,
dass die Bewertungsmethode (Genre-Treue-Score aus einer einzelnen generierten Testaudio pro Genre)
erhebliches Eigenrauschen besitzt — bei komplett unverändertem Modell schwankte eine Wiederholungsmessung
um bis zu 42.5 Punkte bei einem einzelnen Genre, im Schnitt um 13.8 Punkte (siehe Abschnitt 2). Alle vor
dieser Entdeckung durchgeführten Vergleiche (Versuch 1–4) beruhen auf Einzelmessungen und sind entsprechend
vorsichtig zu lesen. Ab Versuch 5 wurde auf eine Methode mit drei Wiederholungen pro Konfiguration
umgestellt.

---

## 1. Ausgangslage

**Aktives Basismodell:** `facebook/musicgen-melody-large`, angepasst über LoRA (Rank 8, Alpha 16, Dropout
0.05, Zielmodule `out_proj`, `linear1`, `linear2`).

**Ursprünglicher Datensatz:** 5000 Trainingsclips (30 Sekunden), 1000 pro Genre, aus jeweils 5–7
unabhängigen YouTube-Quellen pro Genre (`daten/processed/lora_training`).

**Bewertungsgrundlage — die fünf Score-Kategorien** (berechnet in `code/src/Training/audio_bewertung.py`,
exakte Formeln in `dokumentation/aktuell/05_prozessdokumentation_musicgen.md`, Abschnitt 8.1):
Technische Audioqualität, Musikalische Kohärenz, **Genre-Treue** (die in dieser Dokumentation durchgehend
als Hauptvergleichsgröße verwendete Kategorie, da sie im Projektverlauf konsistent die schwächste und
volatilste war), Übergangsqualität, Referenzähnlichkeit.

**Referenzstand ("Baseline"), Checkpoint `step_000625`:**

| Genre | Messung 1 | Messung 2 (Wiederholung, identisches Modell) |
|---|---|---|
| Chillhop Lofi | 47.6 | 30.6 |
| Dreamy Lofi | 55.9 | 54.4 |
| Guitar Lofi | 38.3 | 37.1 |
| Jazz Lofi | 59.4 | 16.9 |
| Study Lofi | 30.2 | 23.5 |
| **Schnitt** | **46.3** | **32.5** |

Diese zwei Messungen sind identisch in Modell, Datensatz und Konfiguration — der einzige Unterschied ist
der Zeitpunkt der Generierung. Die Differenz (13.8 Punkte im Schnitt, 42.5 Punkte bei Jazz) ist Thema von
Abschnitt 2.

---

## 2. Zwischenbefund: Messvalidität (Voraussetzung für die Einordnung aller Versuche)

**Durchgeführt:** Nach vier bereits absolvierten Verbesserungsversuchen (Abschnitt 3–6), die alle eine
Verschlechterung zeigten, aber jeweils ein anderes Genre am stärksten betrafen, wurde der Verdacht auf
Messrauschen getestet: derselbe, unveränderte freigegebene Checkpoint (`step_000625`, SHA-256-Präfix
`a941b297...`) wurde zweimal hintereinander durch den vollständigen Testaudio-Prozess geschickt, ohne
jede Änderung.

**Ergebnis:** siehe Tabelle in Abschnitt 1. Mittelwert-Differenz 13.8 Punkte, Einzelgenre-Differenz bis
42.5 Punkte (Jazz Lofi).

**Konsequenz für die Methodik:** Ab diesem Punkt wurden alle weiteren Vergleiche (Versuch 5 und 6) mit
**drei Wiederholungsläufen pro Konfiguration** (15 Audios statt 5) durchgeführt, Durchschnitt und
Standardabweichung pro Genre berechnet. Auch das reduziert das Rauschen nicht vollständig — selbst bei
drei Wiederholungen lagen die Standardabweichungen einzelner Genres noch bei 5.2 bis 16.6 Punkten
(siehe Versuch 5 und 6).

---

## 3. Versuch 1: Aggressive Hyperparameter

**Zeitpunkt:** vor dem in dieser Dokumentation sonst abgedeckten Zeitraum (frühere Sitzung).

**Hypothese:** Höhere LoRA-Kapazität und stärkere Lernrate führen zu besserer Anpassung an den
Zieldatensatz.

**Änderung:** Rank 8→16, Alpha 16→32, Lernrate 5e-6→2e-5 (4× höher) — **drei Parameter gleichzeitig
geändert**, siehe methodische Anmerkung unten.

**Trainingsverlust:** blieb über den gesamten Lauf nahezu unverändert (Start 3.2599, Ende 3.2597) —
kein Hinweis auf ein Trainingsproblem, aber auch kein Hinweis auf einen Qualitätsgewinn.

**Ergebnis (Einzelmessung, vor der Mehrfachmessungs-Methodik):**

| Genre | Differenz zur Baseline |
|---|---|
| Chillhop Lofi | **+13.0** |
| Dreamy Lofi | −25.8 (stärkste Verschlechterung) |
| übrige drei Genres | jeweils verschlechtert |
| **Schnitt** | **−9.5** (46.3 → 36.8) |

**Entscheidung:** Zurückgesetzt (archiviert unter `training/musicgen/lora_training_archiv_aggressiv_versuch_20260813_141418`).

**Methodischer Mangel dieses Versuchs, im Nachhinein erkannt:** Rank, Alpha und Lernrate wurden gleichzeitig
verändert — ein sogenanntes konfundiertes Experiment. Es lässt sich nicht sagen, welcher der drei Parameter
(oder welche Kombination) für die Verschlechterung verantwortlich war. Recherche zu gängiger Praxis
(Abschnitt 9) legt nahe, dass Rank 16 allein eher der empfohlene Normalwert wäre, nicht "aggressiv" — der
eigentliche Übeltäter könnte die stark erhöhte Lernrate gewesen sein. Ein sauberer Einzeltest (nur Rank 16,
Lernrate unverändert bei 5e-6) wurde nie durchgeführt.

---

## 4. Versuch 2: Mehr Trainingsschritte (3000 statt 625)

**Hypothese:** Mehr Trainingsschritte auf demselben Datensatz verbessern die Anpassung weiter, da
625 Schritte rechnerisch nur einer Epoche über den 5000-Clip-Datensatz entsprechen
(`Schritte = aufgerundet(Trainingsclips / 8)`, wobei 8 = Batchgröße 1 × Gradientenakkumulation 8).

**Änderung:** Fortsetzung des bestehenden Trainings von Schritt 625 auf Ziel 3000 (Faktor 4.8, entspricht
rund 4.8 Epochen), Lernrate unverändert.

**Infrastrukturausfälle während dieses Versuchs (zwei getrennte Vorfälle, beide durch Fortsetzen vom
letzten Checkpoint behoben):**
- Nativer Absturz (`Fatal Python error: Segmentation fault`) bei Schritt 610, Traceback endend in
  `torch/nn/modules/rnn.py`. Fortgesetzt ab Checkpoint `step_000600`.
- Echter Deadlock (kein Absturz, aber kein Fortschritt mehr) bei Schritt 2181, erkannt an durchgehend
  schlafenden Threads im Zustand `futex_wait_queue` über mehr als 19 Minuten sowie eingefrorener
  GPU-Auslastung. Prozess manuell beendet, fortgesetzt ab Checkpoint `step_002175`.

**Ergebnis (Einzelmessung):**

| Genre | Baseline | 3000 Schritte | Differenz |
|---|---|---|---|
| Chillhop Lofi | 47.6 | 45.5 | −2.1 |
| Dreamy Lofi | 55.9 | 35.1 | −20.8 |
| Guitar Lofi | 38.3 | 47.9 | +9.6 |
| Jazz Lofi | 59.4 | 12.7 | **−46.7** |
| Study Lofi | 30.2 | 25.0 | −5.2 |
| **Schnitt** | **46.3** | **33.2** | **−13.0** |

**Entscheidung:** Zurückgesetzt (archiviert unter `training/musicgen/lora_training_archiv_3000schritte_regression_20260814_151601`).

**Einordnung:** Der massive Einbruch bei Jazz Lofi (dem Genre mit der historisch stärksten
Quellenkonzentration, siehe Versuch 5) ist konsistent mit Überanpassung an wenige, häufig wiederholte
Trainingsclips.

---

## 5. Versuch 3: Genre-Melody-Conditioning mit echten Referenzclips

**Hypothese:** MusicGen-Melody unterstützt eine zusätzliche Chroma-Konditionierung
(`generate_with_chroma()`) über eine mitgegebene Referenz-Audiospur. Ein echter Trainingsclip des
Zielgenres als Referenz sollte die Generierung direkter an echtes Referenzmaterial binden als reine
Textsteuerung.

**Technische Umsetzung:** Neue Funktion `melody_reference_from_genre()` in
`code/src/Training/audio_erstellen.py` — wählt für jeden Kandidaten einen realen, genre-passenden
Trainingsclip (deterministisch nach Seed) und gibt ihn über die gesamte Blockdauer als Chroma-Referenz
vor. Neues CLI-Flag `--genre-melodie-conditioning-aktiv`.

**Ergebnis (Einzelmessung):**

| Genre | Baseline | Mit Melody-Conditioning | Differenz |
|---|---|---|---|
| Chillhop Lofi | 47.6 | 42.4 | −5.2 |
| Dreamy Lofi | 55.9 | 41.9 | −14.0 |
| Guitar Lofi | 38.3 | 47.9 | +9.6 |
| Jazz Lofi | 59.4 | 16.4 | **−43.0** |
| Study Lofi | 30.2 | 35.0 | +4.8 |
| **Schnitt** | **46.3** | **36.7** | **−9.6** |

**Entscheidung:** Deaktiviert (Flag bleibt im Code verfügbar, in keinem produktiven Ablauf aktiv).

**Einordnung:** Erneut der stärkste Einbruch bei Jazz Lofi — zweites Mal in Folge dasselbe Genre am
stärksten betroffen, obwohl zwei völlig unterschiedliche Interventionen (mehr Schritte vs.
Melody-Conditioning) getestet wurden. Naheliegende Erklärung: Jazz Lofis besonders enge Quellenbasis
macht es für praktisch jede Intervention, die das Modell "härter" an vorhandenes Material bindet
(mehr Wiederholung oder engere Konditionierung), besonders anfällig.

---

## 6. Versuch 4: Datensatz-Diversifizierung mit lizenzfreien Zusatzquellen

**Ausgangsdiagnose:** Trotz 1000 Clips pro Genre stammten diese aus nur 5–7 unabhängigen Quellvideos.
Bei Jazz Lofi entfielen 492 von 1000 Clips (49 %) auf nur zwei Quellvideos.

**Recherche vor der Umsetzung:** Drei Plattformen geprüft.
- **Free Music Archive**: gewählt. Verifizierte Direct-Download-Links, überwiegend CC0/CC-BY.
- **Jamendo**: verworfen — Lizenzvariante für die meisten Treffer nicht programmatisch verifizierbar,
  mehrere Stichproben als CC-BY-NC-ND (keine Bearbeitung erlaubt) identifiziert.
- **Pixabay**: verworfen — Nutzungsbedingungen untersagen explizit automatisiertes Sammeln von Inhalten
  "für maschinelles Lernen".

**Umsetzung:** 19 Tracks von Free Music Archive heruntergeladen (sechs unabhängige Künstler: HoliznaCC0,
Ketsa, 1000 Handz, legacyAlli, Alex-Productions, Scott Holmes Music), importiert über
`POST /api/sources/upload`, geclippt, in den Datensatz integriert.

**Technisches Hindernis dabei entdeckt und behoben:** Der produktive Auswahlpfad betrachtete ein Genre als
"voll", sobald 1000 Clips aus dem bestehenden Pool verfügbar waren, und übersprang neue, kleine Quellen
komplett. Erst ein gezielter Neuaufbau über `zieldatensatz.py` mit dessen Round-Robin-Auswahlverfahren
stellte sicher, dass die neuen Quellen tatsächlich einflossen.

**Ergebnis — Quellenanzahl:**

| Genre | Vorher | Nachher |
|---|---|---|
| Chillhop Lofi | 6 | 11 |
| Dreamy Lofi | 5 | 10 |
| Guitar Lofi | 6 | 9 |
| Jazz Lofi | 7 | 10 |
| Study Lofi | 6 | 9 |

**Qualitätsprüfung der neuen Quellen vor dem Training:** Technisch einwandfrei (Median-Score 98.5/100,
kein Clipping). Stilistisch gemischt (Median-Distanz-Score zum bisherigen Genreprofil: 31.7 von 100,
10 von 20 Quellen unter 30).

**Ergebnis nach frischem Training (Einzelmessung):**

| Genre | Baseline | Mit FMA-Erweiterung | Differenz |
|---|---|---|---|
| Chillhop Lofi | 47.6 | 11.1 | **−36.5** |
| Dreamy Lofi | 55.9 | 55.1 | −0.8 |
| Guitar Lofi | 38.3 | 34.2 | −4.1 |
| Jazz Lofi | 59.4 | 42.7 | −16.7 |
| Study Lofi | 30.2 | 26.7 | −3.5 |
| **Schnitt** | **46.3** | **34.0** | **−12.3** |

**Entscheidung:** Zurückgesetzt. Datensatz archiviert unter
`daten/processed/lora_training_archiv_fma_mischung_20260817_200224` (spätere Version) bzw.
`daten/processed/lora_training_archiv_vor_fma_erweiterung_20260814_162411` (Originalzustand vor der
Erweiterung). Die 19 FMA-Rohquellen und ihre Clip-Ordner bleiben erhalten unter
`daten/processed/musicgen_youtube_import_30s_fma_archiv/` für eine mögliche gezieltere Nachkuratierung.

**Einordnung:** Diesmal Chillhop Lofi am stärksten betroffen, exakt zeitgleich mit der schwächsten
Einzelquelle der neuen Auswahl (Stilistik-Score 4.4 von 100 bei einer der neuen Chillhop-Quellen). Dritter
Versuch in Folge mit deutlicher Verschlechterung, drittes Mal ein jeweils anderes am stärksten betroffenes
Genre.

---

## 7. Versuch 5: Gleichmäßige Ausnutzung der vorhandenen Quellen (Extraktions-Reparatur)

**Ausgangsdiagnose:** Die Clip-Extraktion (`clippe_mp3()` in `code/src/Pipeline/clips_vorbereiten.py`)
nahm bei Quellen mit mehr verfügbaren 30s-Fenstern als dem Deckel (250 pro Quelle) immer die **ersten**
250 Fenster in chronologischer Reihenfolge. Bei mehrstündigen Quellen (mehrere 8-Stunden-Mixe unter den
Trainingsquellen bestätigt, z. B. "Sweet Dreams Lofi 💤 8 hours") bedeutete das: nur die ersten ca.
2 Stunden wurden je genutzt, der Rest (bis zu 75 % der Datei) blieb komplett ungenutzt, obwohl bereits
heruntergeladen.

**Technische Reparatur:** `clippe_mp3()` geändert — wenn mehr Kandidaten-Startpunkte verfügbar sind als
der Deckel erlaubt, werden sie jetzt gleichmäßig über die gesamte verfügbare Länge verteilt
(Index-Berechnung `round(i * (len(starts)-1) / (clip_limit-1))`) statt nur die ersten N zu nehmen.

**Umsetzung:** Alle 30 ursprünglichen Quellen (5–7 pro Genre) mit der reparierten Logik neu extrahiert
(vorherige Extraktion je Quelle archiviert, nicht gelöscht), Datensatz aus dem aktualisierten Pool neu
gebaut. FMA-Erweiterung aus Versuch 4 bewusst **nicht** wieder mit eingeschlossen, um den Effekt isoliert
zu testen.

**Stichprobe zur technischen Qualität der neu extrahierten Clips:** 30 zufällige Clips geprüft,
Median-Technik-Score 100.0, null Clips unter 70, kein Clipping — auch Material aus vorher ungenutzten,
tieferen Abschnitten der langen Quellen ist technisch einwandfrei.

**Ergebnis (Mehrfachmessung, 3 Läufe je Genre, erste Anwendung der neuen Methodik):**

| Genre | 3 Einzelwerte | Schnitt | StdAbw |
|---|---|---|---|
| Chillhop Lofi | 47.1 / 67.4 / 55.6 | 56.7 | 10.2 |
| Dreamy Lofi | 67.4 / 60.6 / 42.4 | 56.8 | 12.9 |
| Guitar Lofi | 33.4 / 39.3 / 55.3 | 42.7 | 11.3 |
| Jazz Lofi | 20.2 / 49.6 / 40.7 | 36.8 | 15.1 |
| Study Lofi | 29.8 / 18.4 / 21.2 | 23.1 | 5.9 |
| **Schnitt** | | **43.2** | |

**Entscheidung:** Beibehalten (kein Rücksetzen), da kein Rückschritt gegenüber der Baseline (~39.4,
Mittel der beiden Baseline-Einzelmessungen) erkennbar ist. Explizit **kein bewiesener Fortschritt** —
die Differenz (+3.8) liegt innerhalb der selbst beobachteten Streuung.

**Infrastrukturausfälle während dieses Versuchs:** Ein Trainingslauf hing bei Schritt 2096 (identisches
Muster wie Versuch 2: GPU-Auslastung über 8 Sekunden komplett eingefroren, Log-Stillstand). Fortgesetzt
ab Checkpoint `step_000250`.

---

## 8. Versuch 6: Individualisierte Trainings-Captions pro Clip

**Ausgangsdiagnose:** Alle 1000 Clips eines Genres erhielten beim Training exakt dieselbe, statische
Textbeschreibung (`GENRE_CAPTIONS` in `code/src/Dataset/genre_regeln.py`), unabhängig davon, wie
unterschiedlich die Clips tatsächlich klangen. Bestätigt durch direkten Code- und Datencheck.

**Recherche zu gängiger Praxis (Replicate-Blogpost zum MusicGen-Fine-Tuning):** Kommerzielle
Fine-Tuning-Dienste generieren automatisch pro Track individuelle Beschreibungen (Stimmung,
Instrumentierung, Tonart, BPM) statt einer einzigen Sammel-Beschreibung.

**Technische Umsetzung:** Neues Skript `code/src/Dataset/captions_anreichern.py`. Berechnet für jeden
Clip echte Audiomerkmale (dieselbe Funktion `score_kennwerte()` wie für die Bewertung: BPM, Bassanteil,
Snareanteil, Höhenanteil, Dynamikspanne) und hängt daraus abgeleitete, regelbasierte Textbausteine an
die bestehende Genre-Caption an (z. B. "slow, spacious tempo" bei BPM < 65, "bass-forward low end" bei
Bassanteil > 0.4). Keine neue Bibliothek, keine Erfindung — jeder Textbaustein entspricht einem
tatsächlich gemessenen Schwellenwert.

**Umsetzung auf dem vollständigen Datensatz:** 5000 Clips verarbeitet (4272 train, 456 valid, 272 test),
0 Fehler. Vorheriger Datensatzstand gesichert unter
`daten/processed/lora_training_archiv_vor_caption_anreicherung_20260818_220046`.

**Ergebnis (Mehrfachmessung, 3 Läufe je Genre):**

| Genre | 3 Einzelwerte | Schnitt | StdAbw | Vorher (Versuch 5) |
|---|---|---|---|---|
| Chillhop Lofi | 56.5 / 23.9 / 45.5 | 42.0 | 16.6 | 56.7 |
| Dreamy Lofi | 33.9 / 40.4 / 59.7 | 44.7 | 13.4 | 56.8 |
| Guitar Lofi | 44.6 / 47.6 / 35.7 | 42.6 | 6.2 | 42.7 |
| Jazz Lofi | 55.2 / 42.1 / 47.9 | 48.4 | 6.6 | 36.8 |
| Study Lofi | 30.8 / 34.4 / 41.1 | 35.4 | 5.2 | 23.1 |
| **Schnitt** | | **42.6** | | **43.2** |

**Entscheidung:** Beibehalten (kein Rücksetzen), da kein Rückschritt erkennbar. Ebenfalls **kein
bewiesener Fortschritt** gegenüber Versuch 5 — die Differenz (−0.6) liegt weit innerhalb der beobachteten
Streuung, und das Vorzeichen wechselt je Genre uneinheitlich (Jazz und Study besser, Chillhop und Dreamy
schlechter, Guitar praktisch gleich) — ein Muster, das eher zu reinem Rauschen als zu einem systematischen
Effekt passt.

**Infrastrukturausfälle während dieses Versuchs (zwei getrennte Vorfälle):**
- Deadlock bei Schritt 2096 während des ersten Trainingsversuchs (GPU-Auslastung über 8 Sekunden
  komplett eingefroren, Log-Stillstand über 7 Minuten). Fortgesetzt ab Checkpoint `step_000250`.
- Nativer Absturz (`Fatal Python error: Bus error`, Signal SIGBUS) bei Schritt 322 während der Fortsetzung,
  Traceback identisch zur Absturzstelle in Versuch 2: `torch/nn/modules/rnn.py`, aufgerufen aus
  EnCodecs Audiokodierung (`transformers/models/encodec/modeling_encodec.py` → `audiocraft/models/encodec.py`).
  Fortgesetzt ab Checkpoint `step_000300`.

---

## 9. Versuch 7: Isolierter Rank-16-Test

**Hypothese:** Versuch 1 (Abschnitt 3) hatte gleichzeitig Rank, Alpha und Lernrate verändert — ein
konfundiertes Experiment, bei dem unklar blieb, ob die Verschlechterung an der höheren Kapazität oder an
der 4-fach höheren Lernrate lag. Recherche zu gängiger Praxis (Abschnitt 10.3) legt Rank 16 als üblichen
Standardwert nahe, nicht als "aggressiv". Dieser Versuch isoliert die Rank/Alpha-Erhöhung sauber von der
Lernrate.

**Änderung:** Nur Rank 8→16, Alpha im selben Verhältnis mitskaliert (16→32, wie auch in Versuch 1 und wie
in der Praxis üblich), Lernrate **unverändert** bei 5e-6 (in Versuch 1 war sie auf 2e-5 angehoben). Gleicher
aktueller Datensatz (Stand nach Versuch 5 + Versuch 6, keine Datenänderung), frisches Training ab
Basismodell (ein höherer Rank kann nicht von einem Rank-8-Checkpoint fortgesetzt werden), 625 Zielschritte
— identisch zur sonstigen Trainingslänge, um nur die Rank/Alpha-Wirkung zu isolieren. Eigener, getrennter
Trainingsordner (`training/musicgen/lora_training_rank16_test`), der aktive Produktions-Checkpoint blieb
währenddessen unberührt.

**Training:** 625 Schritte, sauber durchgelaufen (`returncode=0`), kein Absturz und kein Hänger — im
Gegensatz zu vier der bisherigen sechs Versuche störungsfrei.

**Ergebnis (Mehrfachmessung, 3 Läufe je Genre, gemessen über `score_bewertung.json` /
`bewerte_audio_mit_genrestandard` — dieselbe Quelle wie bei Versuch 5 und 6):**

| Genre | 3 Einzelwerte | Schnitt | StdAbw | Versuch 5 | Versuch 6 (aktiv) |
|---|---|---|---|---|---|
| Jazz Lofi | 42.0 / 51.7 / 48.9 | 47.5 | 5.0 | 36.8 | 48.4 |
| Chillhop Lofi | 35.3 / 42.8 / 46.5 | 41.5 | 5.7 | 56.7 | 42.0 |
| Dreamy Lofi | 53.1 / 37.1 / 61.3 | 50.5 | 12.3 | 56.8 | 44.7 |
| Study Lofi | 42.5 / 28.2 / 40.0 | 36.9 | 7.6 | 23.1 | 35.4 |
| Guitar Lofi | 41.5 / 39.4 / 32.1 | 37.7 | 4.9 | 42.7 | 42.6 |
| **Schnitt** | | **42.8** | | **43.2** | **42.6** |

**Entscheidung:** Nicht übernommen — nicht weil es schlechter wäre, sondern weil kein belastbarer
Unterschied zum aktiven Modell besteht. Getesteter Checkpoint bleibt als reiner Vergleichslauf unter
`training/musicgen/lora_training_rank16_test` erhalten, ohne freigegeben zu werden.

**Einordnung:** Der Gesamtschnitt (42.8) liegt praktisch mittig zwischen Versuch 5 (43.2) und Versuch 6
(42.6) — eine Differenz von 0.4 bis 0.6 Punkten, weit innerhalb jeder hier beobachteten Streuung (die
Einzelgenre-Standardabweichungen liegen zwischen 4.9 und 12.3). Pro Genre zeigt sich dasselbe Bild wie
schon bei Versuch 6 gegenüber Versuch 5: Jazz und Study liegen näher am (besseren) Versuch-6-Wert,
Chillhop und Guitar fallen ab, Dreamy liegt dazwischen. Damit ist dies der **dritte neutrale Versuch in
Folge** (nach Versuch 5 und 6) — der isolierte Rank-16-Test bestätigt im Kern das Ergebnis von Versuch 1:
eine höhere LoRA-Kapazität allein verbessert die Genre-Treue bei diesem Datensatz nicht messbar, unabhängig
davon, ob die Lernrate mit angehoben wird oder nicht.

---

## 10. Untersuchte, aber nicht als eigener Trainingsversuch umgesetzte Alternativen

### 10.1 Semantische Genre-Prüfung mit CLAP

Bereits im Code vorhandene, aber inaktive Komponente (`code/src/Training/genre_pruefung.py`,
Klasse `GenrePruefer`) — nutzt ein lokal gespeichertes CLAP-Modell, um generierte Kandidaten per
Kosinus-Ähnlichkeit gegen Genre-Referenzzentren zu prüfen. War über `--genre-pruefung-deaktivieren`
in allen produktiven Abläufen abgeschaltet, weil die hinterlegte Referenzliste
(`training/bewertungen/musicgen/lora_review_001/bewertung.csv`) aus einer abgeschafften manuellen
Bewertungsfunktion stammt und nie ausgefüllt wurde (Status-Spalte durchgehend leer).

**Reparatur:** Neues Skript `code/src/Training/clap_referenzen_bauen.py` — baut automatisch 20
Referenzclips pro Genre aus dem bestehenden Trainingsdatensatz.

**Held-out-Test:** 10 echte, nicht in der Referenzliste enthaltene Clips (2 pro Genre) gegen ihr
korrektes Zielgenre geprüft. **4 von 10 korrekt (40 %)**, gegenüber 20 % Zufallsbasis bei fünf Klassen.
Auffällige Tendenz: fast alle Fehlzuordnungen fielen auf "Dreamy Lofi", unabhängig vom tatsächlichen
Genre — das CLAP-Modell trennt die fünf eng verwandten Lo-Fi-Subgenres offenbar nicht trennscharf genug.

**Entscheidung:** Nicht produktiv aktiviert. Referenzdaten und Skript bleiben für eine mögliche spätere
Neubewertung mit anderem Modell/anderen Schwellenwerten erhalten.

### 10.2 Qualitätsprüfungs-Lücke: Bass-Rumpel-Fund

Bei manueller Anhörkontrolle eines veröffentlichten Testaudios (Chillhop Lofi) wurde ein als klanglich
leer wahrgenommener Abschnitt (Sekunde 5–18) gemeldet, den die automatische Sekunden-Prüfung fälschlich
als gültig akzeptiert hatte (RMS-Wert lag mit −26 bis −28 dB deutlich über der Stille-Schwelle).

**Technische Analyse:** Frequenzbandanalyse desselben Abschnitts ergab **99.9 % der Signalenergie
unterhalb von 150 Hz** (35.4 % Sub-Bass 0–40 Hz + 64.5 % Bass 40–150 Hz), praktisch 0 % in Mitten,
Höhen oder sehr hohen Frequenzen — ein tieffrequentes Brummen ohne musikalischen Inhalt, das die rein
lautstärkebasierte Prüfung nicht von einer "lauten, gültigen" Passage unterscheiden kann.

**Status:** Lücke identifiziert und dokumentiert (Vorschlag: zusätzliche Mindestenergie-Schwelle oberhalb
150 Hz), **technisch noch nicht behoben**.

### 10.3 Recherche zu gängiger Industriepraxis

Vor Versuch 6 durchgeführt, mit folgenden verifizierten Kernbefunden:
- **musicgen-dreamboothing** (GitHub, Hugging-Face-Projekt): erfolgreiches Fine-Tuning von
  MusicGen-Melody bereits mit 27 Minuten einzelgenre-reiner Musik dokumentiert.
- **Replicate-Blogpost "Fine-tune MusicGen to generate music in any style"**: "Just a few tracks (9-10)
  are enough to fine-tune MusicGen on a musical style." Automatische Pro-Clip-Beschriftung via Essentia
  (Genre, Stimmung, Instrumentierung, Tonart, BPM) als Standardvorgehen — direkte Inspiration für
  Versuch 6. Vocals-Entfernung via Demucs als Standardschritt — als für unser Projekt nicht relevant
  eingestuft, da unsere Quellen bereits durchgehend als instrumental kuratiert sind (Titelprüfung ergab
  keinen Hinweis auf Gesangsanteile).
- **Unsloth LoRA-Hyperparameter-Guide** (primär für Sprachmodelle, nur bedingt übertragbar): Rank 16 als
  aktueller Standardwert (wir nutzen 8), Zielmodule "alle linearen Schichten" als Empfehlung (wir nutzen
  nur drei spezifische), "Datenqualität schlägt Datenmenge" als wiederkehrendes Prinzip.

**Daraus abgeleiteter Vorschlag, inzwischen umgesetzt:** ein isolierter Einzeltest von Rank 16 bei
unveränderter Lernrate (5e-6) — sauberer als der konfundierte Versuch 1. Durchgeführt als Versuch 7
(Abschnitt 9): neutrales Ergebnis, kein belastbarer Unterschied zum aktiven Modell.

### 10.4 xFormers-Versionskonflikt (geprüft und verworfen)

**Ursprüngliche Vermutung:** Die bei jedem Trainingsstart erscheinende Warnung ("xFormers can't load
C++/CUDA extensions... built for PyTorch 2.1.0+cu121... you have 2.6.0+cu124") könnte Ursache der
wiederholten nativen Abstürze (Versuch 2, Versuch 6) sein.

**Prüfung:** `audiocraft` (unsere Trainingsbibliothek) schreibt zwingend `xformers<0.0.23` vor; die
installierte Version (0.0.22.post7) ist bereits die höchste, die diese Vorgabe erfüllt — es existiert
keine neuere, kompatible Version zum Upgrade. Wichtiger: Die tatsächlichen Absturz-Tracebacks (Versuch 2
und 6) zeigen den Fehler durchgehend in `torch/nn/modules/rnn.py`, aufgerufen aus EnCodecs
Audio-Encoder — einem LSTM-basierten Codepfad, der mit xFormers' Aufmerksamkeitsmechanismus (der nur im
Transformer-Teil des Modells greift) nichts zu tun hat.

**Entscheidung:** Vermutung verworfen, keine Änderung vorgenommen. Die tatsächliche Ursache der
wiederkehrenden nativen Abstürze bleibt ungeklärt (vermutlich ein tieferliegendes CUDA-/Treiber-Problem).

---

## 11. Zusammenfassende Tabelle aller sieben Versuche

| # | Versuch | Methodik | Ergebnis (Schnitt) | Differenz | Entscheidung |
|---|---|---|---|---|---|
| — | Baseline | — | 46.3 / 32.5 (zwei Messungen) | — | Referenz |
| 1 | Aggressive Hyperparameter | Einzelmessung | 36.8 | −9.5 | zurückgesetzt |
| 2 | Mehr Trainingsschritte (3000) | Einzelmessung | 33.2 | −13.0 | zurückgesetzt |
| 3 | Melody-Conditioning | Einzelmessung | 36.7 | −9.6 | deaktiviert |
| 4 | Datensatz-Diversifizierung (FMA) | Einzelmessung | 34.0 | −12.3 | zurückgesetzt |
| 5 | Gleichmäßige Quellennutzung | 3-fach-Messung | 43.2 | +3.8 ggü. Baseline-Mittel | beibehalten (neutral) |
| 6 | Individualisierte Captions | 3-fach-Messung | 42.6 | −0.6 ggü. Versuch 5 | beibehalten (neutral) |
| 7 | Isolierter Rank-16-Test | 3-fach-Messung | 42.8 | −0.4 ggü. Versuch 5 / +0.2 ggü. Versuch 6 | nicht übernommen (neutral) |

**Bilanz:** Vier von sieben Versuchen zeigten eine klare, deutliche Verschlechterung. Die letzten drei
(alle nach Einführung der Mehrfachmessung) zeigten weder klaren Fortschritt noch klaren Rückschritt — ihre
scheinbaren Effekte liegen innerhalb der selbst gemessenen Rauschgrenze. Insbesondere Versuch 7 zeigt: auch
der sauber isolierte Rank-16-Test (ohne die Lernrate-Konfundierung von Versuch 1) bringt keinen messbaren
Vorteil — die Vermutung, Versuch 1 sei allein wegen der zu hohen Lernrate gescheitert, bestätigt sich damit
nicht.

---

## 12. Aktueller Stand (2026-08-19)

- **Aktiver Checkpoint:** `step_000625`, Status `freigegeben`, trainiert auf dem Datensatz aus Versuch 5
  (gleichmäßige Quellennutzung) und Versuch 6 (individualisierte Captions) kombiniert.
- **Datensatz:** 5000 Clips, 5 Genres, 30 Original-Quellen (keine FMA-Zusatzquellen aktiv).
- **Versuch 7 (Rank-16-Test)** liegt zusätzlich als abgeschlossener, aber nicht freigegebener
  Vergleichs-Checkpoint unter `training/musicgen/lora_training_rank16_test` vor (Rank 16/Alpha 32,
  sonst identische Einstellungen zum aktiven Modell).
- **Kein Job aktiv**, keine offenen Trainingsläufe.
- **Nicht abschließend geklärt:** Ursache der wiederkehrenden nativen Trainingsabstürze (insgesamt vier
  dokumentierte Vorfälle über alle Versuche: zwei Segmentation Faults/Hänger in Versuch 2, ein Hänger und
  ein Bus-Error in Versuch 6) — jedes Mal zuverlässig über Checkpoint-Wiederaufnahme behoben, aber die
  Ursache selbst bleibt unbekannt. Versuch 7 verlief dagegen störungsfrei.
- **Nicht umgesetzt:** Bass-Rumpel-Qualitätsprüfungs-Lücke (Abschnitt 10.2), breitere Zielmodule und
  vertiefte automatische Captions (Abschnitt 10.3), CLAP-Prüfung (Abschnitt 10.1, bewusst nicht aktiviert).

## 13. Für die Bachelorarbeit relevante Kernaussage

Der wissenschaftliche Wert dieser Versuchsreihe liegt nicht in einem nachgewiesenen Qualitätssprung,
sondern in zwei methodischen Erkenntnissen: erstens, dass naheliegende Standard-Stellschrauben (mehr
Trainingsschritte, aggressivere Hyperparameter, reichhaltigere Konditionierung, mehr Trainingsdaten, höhere
LoRA-Kapazität) bei diesem Modell und dieser Datenmenge nicht zuverlässig zu Verbesserungen führen und
teils deutlich schaden; zweitens, dass die Bewertungsmethode selbst — vor ihrer Verbesserung durch
Mehrfachmessung — nicht präzise genug war, um kleinere, aber reale Effekte überhaupt von Zufall zu
unterscheiden. Versuch 7 untermauert den ersten Punkt zusätzlich: selbst nach sauberer Trennung von Rank
und Lernrate (die in Versuch 1 noch vermischt waren) bleibt der Effekt aus. Beide Befunde sind für die
Diskussion generativer Musikmodelle mit kleinen, thesis-typischen Datenmengen relevant und wurden
durchgehend mit echten Messwerten belegt, nicht nur behauptet.
