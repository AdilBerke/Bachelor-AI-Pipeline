# Résumé: Taktik und Weiterentwicklung im Video-/GIF-Teil

Zusammenfassende Erzählung der strategischen Entscheidungen von Mai bis August 2026 —
nicht nur *was* passiert ist (siehe `02_verlauf.md` für die Detail-Chronologie), sondern
*welche Taktik* jeweils dahinterstand und *warum* sie sich änderte. Direkt verwendbar für
Kapitel 5.1 ("Iterative Weiterentwicklung des Konzepts") der Bachelorarbeit.

---

## Grundtaktik am Anfang: Infrastruktur vor Inhalt

Der Ausgangspunkt (Mai 2026, `video_pipeline`) war keine kleine Testübung, sondern der
Versuch, **zuerst die vollständige Maschinerie zu bauen**, bevor überhaupt ein einziges
GIF entstand: YouTube-Suche mit Lizenzfilter, automatischer Videoschnitt, Frame-Extraktion,
Tagging, ein Datensatzformat mit 101 vordefinierten Motivkategorien — und das alles hinter
einem einzigen Befehl (`main.py run-video-pipeline`). Das ist eine erkennbare Taktik: **erst
die Breite planen, dann in die Tiefe gehen**. Sie zeigt sich auch daran, dass rechtliche
Fragen (Lizenzfilter, `license_info`-Feld) von Anfang an mitgedacht wurden, nicht nachträglich.

**Schwachstelle dieser Taktik:** Die Infrastruktur wurde vollständig gebaut, bevor sie an
echten Daten erprobt wurde. Das Ergebnis: Kein einziger der 101 Kategorien wurde je mit
einem Clip befüllt. Der Trainingsschritt selbst wurde bewusst ausgelagert ("separates
Repository") und blieb damit die eigentlich unbearbeitete Kernfrage — die Pipeline löste
das leichtere Problem (Datenorganisation) sehr gründlich und ließ das schwerere Problem
(funktionierendes Training) offen.

---

## Bruch 1: Von der großen Vision zur direkten Handarbeit

Zwischen Mai und dem 13.06. (Beginn `v003_manual`) findet ein stiller, aber fundamentaler
Taktikwechsel statt: weg vom "erst alles vorbereiten", hin zu "sofort mit dem eigentlichen
Modell trainieren, auch ohne die geplante Automatisierung drumherum". Der externe LTX-Video-
Trainer wird direkt eingebunden, ein erster LoRA-Lauf startet noch am selben Tag, an dem das
13B-Modell heruntergeladen wurde. Das ist ein Wechsel von **Systemdesign zuerst** zu
**Ergebnis zuerst, System später** — eine typische und meist gesunde Korrektur, wenn die
ursprüngliche Planung zu ambitioniert war, um sie vor dem ersten Realitätscheck fertigzustellen.

---

## Bruch 2: Human-in-the-Loop als Ersatz für fehlende Metriken

Weil kein automatisches Qualitätskriterium existierte (das hatte auch die Mai-Pipeline nie
vorgesehen), wurde die einzig verfügbare Bewertungsinstanz genutzt: der Mensch, nach jeder
Trainingsrunde. Die Taktik dahinter war klug strukturiert, nicht beliebig — `feedback.json`
verlangt in den ersten fünf Runden von `rainy_window` konsistent dieselben Felder
(`overall_direction`, `positives`, `improvements_needed`, `preserve`, `prompt_changes`),
und Fehlschläge wurden dokumentiert, nicht verschwiegen (z. B. das Letterboxing durch
Image-Conditioning in Runde 7 der Vorphase, sofort mit Ursache und Gegenmaßnahme notiert).

**Schwachstelle:** Diese Disziplin ließ mit der Zeit nach. Ab Runde 6 von `rainy_window`
existiert für 9 von 14 Runden kein `feedback.json` mehr; bei `rabbit_lake` fehlt es
durchgehend für alle 9 Rundenordner. Die Taktik "Mensch bewertet jede Runde" wurde also
begonnen, aber nicht durchgehalten — vermutlich, weil sie bei steigender Rundenzahl und
längeren Trainingszeiten (bis zu 25 Minuten pro Runde) zu aufwändig wurde, ohne dass eine
leichtere Alternative bereitstand.

---

## Bruch 3: Parallelisierung über mehrere Szenarien

Ab dem 03.07. (Umbau zum Szenario-System) ändert sich die Taktik erneut: statt eines
einzigen, immer weiter verfeinerten Motivs ("lofi_girl") werden mehrere unabhängige
Szenarien nebeneinander aufgesetzt, mit einem geteilten Basis-Checkpoint als gemeinsamem
Startpunkt. Das ist eine **Explorationstaktik** — mehrere Wetten gleichzeitig platzieren,
um zu sehen, welches Motiv am ehesten funktioniert, statt sich früh auf eines festzulegen.

**Ergebnis der Taktik, ehrlich betrachtet:** Von sechs geplanten Szenarien erhielten nur
zwei überhaupt Trainingsrunden (`rainy_window`: 14, `rabbit_lake`: 9). Die Parallelisierung
fand also nur in der *Planung* statt (sechs `scenario.yaml`-Dateien existieren), nicht in
der *Durchführung*. Das ist ein wiederkehrendes Muster im ganzen Projekt: große Absichten
werden strukturell angelegt (Configs, Ordner, Kategorien), aber nur ein Bruchteil davon
wird tatsächlich mit Rechenzeit und Bewertung gefüllt.

---

## Wenn's hakt: strukturierte Root-Cause-Analyse statt Trial-and-Error

Das "3-statt-2-Hasen"-Problem bei `rabbit_lake` ist der Punkt im Projekt, an dem sich die
disziplinierteste Taktik zeigt: Statt einfach weiterzutrainieren, wurde am 23.07. ein
eigenes Analysedokument (`RESCUE_PLAN.md`) verfasst — mit Ursachenhypothese (Prior-Bias
des 13B-Basismodells), einer geordneten Testmatrix aus mehreren unabhängigen Lösungsansätzen
(Seed-Suche, Datensatz-Korrektur, höhere Guidance Scale, Re-Framing der Szene) und
expliziten Abbruchkriterien ("wenn 3 Optionen scheitern → Datensatz-Neuaufbau"). Das ist
systematisches Debugging, keine Verzweiflungstaktik.

**Aber:** Keiner der dokumentierten Lösungsversuche wurde je mit einem `feedback.json`
als erfolgreich bestätigt. Die sorgfältige Diagnose führte nicht zu einer ebenso
sorgfältig dokumentierten Verifikation — der Kreis wurde nicht geschlossen.

---

## Bruch 4: Nachträgliche Objektivierung durch automatische Metriken (06.08.)

Der bisher jüngste und methodisch bedeutsamste Taktikwechsel: Erstmals wird die im Mai-
Konzept übersprungene Zutat nachgereicht — ein automatisches, quantitatives
Bewertungssystem (`evaluate_video.py`, sechs Teilmetriken, gewichteter Gesamtscore) plus
`metric_watcher.py`, das Training bei Erreichen definierter Zielwerte selbstständig stoppen
kann. Das ist ein Wechsel von **subjektiver Einzelbewertung** zu **reproduzierbarer,
automatischer Bewertung** — und schließt exakt die Lücke, die die gesamte bisherige
Entwicklung methodisch am meisten geschwächt hat.

**Einschränkung, die für die Arbeit wichtig ist:** Die neuen Metriken wurden bislang nur
auf drei einzelne, bereits existierende Samples angewendet, nicht rückwirkend auf den
gesamten bisherigen Rundenverlauf. Der Werkzeugwechsel ist vollzogen, die systematische
Anwendung auf das gesamte bisherige Material steht noch aus.

---

## Aktuelle Phase (06.–07.08.): mehrere gleichzeitige Vorstöße

Zeitgleich mit der Metrik-Einführung entstehen drei neue, parallele Ansätze:

- **`rabbit_lake_v2`** — sorgfältig weitergeführter Neustart mit fest einprogrammierten
  Metrik-Zielwerten und Rundenlimit (`max_rounds: 8`). Konsequente Fortsetzung der
  bisherigen Taktik "Root-Cause-Analyse + geordneter Wiederanlauf".
- **`rabbit_lake_v2_0807_0356`** — wirkt dagegen wie ein automatisch/über ein Formular
  erzeugter Testeintrag (einfacherer Prompt, Tippfehler, nur ein statt zwei Hasen) und
  passt eher zur Kategorie "schnelles Ausprobieren einer neuen Werkzeugfunktion" als zu
  einer durchdachten Fortsetzung.
- **`realistic_rabbit`** — ein komplett neuer, fotorealistischer Ansatz für dasselbe Motiv,
  mit eigener Testmatrix A–G. Das ist wieder die Explorationstaktik aus Bruch 3, diesmal
  auf Stilebene statt auf Szenarienebene.

**Strategische Einordnung für das Mentorengespräch:** Diese drei parallelen Vorstöße sind
einerseits ein gutes Zeichen — die Taktik "bei Blockade strukturiert neue Wege suchen"
hat sich über das ganze Projekt bewährt. Andererseits ist es, gemessen an der verbleibenden
Zeit (siehe Machbarkeitsstudie), eine **Scope-Erweiterung genau in dem Moment, in dem
Fokussierung nötig wäre**. Keiner der drei Vorstöße hat bislang einen einzigen abgeschlossenen
und bewerteten Trainingslauf vorzuweisen.

---

## Wiederkehrende Muster über das gesamte Projekt

**Positiv, konsistent gute Taktik:**
- Bei technischen Fehlern (Letterboxing, OOM-Absturz, Szenendrift) wurde die Ursache
  benannt und eine konkrete Gegenmaßnahme ergriffen, nicht nur "nochmal versucht".
- Rollbacks auf ältere, nachweislich bessere Checkpoints wurden mehrfach bewusst
  vorgenommen (Round 5 statt 6, Round 11 statt 13) — Zeichen dafür, dass "neuer Checkpoint"
  nicht automatisch mit "besserer Checkpoint" gleichgesetzt wurde.
- Rechtliche/methodische Sorgfalt war von Anfang an vorhanden (Lizenzfilter, später die
  Rechte-Checkliste unter `references/legal/`).

**Wiederkehrende Schwachstelle:**
- Planungsartefakte (Configs, Ordnerstrukturen, Kategorienlisten) entstehen deutlich
  schneller als die dazugehörige Durchführung und Bewertung. Der Abstand zwischen "angelegt"
  und "tatsächlich beendet und bewertet" zieht sich durch alle Phasen: 101 Kategorien → 0
  gefüllt (Mai), 6 Szenarien → 2 mit Trainingsrunden (Juli), 3 aktuelle Vorstöße → 0
  abgeschlossen (August).
- Dokumentationsdisziplin (`feedback.json`) nimmt mit steigender Rundenzahl ab, statt
  gleichbleibend zu sein — vermutlich, weil kein leichtgewichtiges Zwischenformat zwischen
  "vollständiges Freitext-Feedback" und "gar keine Bewertung" existierte, bis die
  automatischen Metriken kamen.
- Keine Versionskontrolle für den gesamten Video-Teil — im Gegensatz zum Audio-Teil, der
  durchgehend über Git läuft (deutlich mehr, kleinteiligere Commits). Das legt nahe, dass
  der Video-Teil durchgehend als *experimenteller* Bereich behandelt wurde, der Audio-Teil
  dagegen als *produktiver*, versionierter Bereich — eine unausgesprochene, aber konsistente
  Priorisierung.

---

## Einordnung für die Bachelorarbeit

Die ehrliche Kernaussage dieses Résumés ist nicht "planlos", sondern **"die Taktik war
gut, das Durchhalten der Taktik nicht"**: Jeder einzelne Strategiewechsel (Infrastruktur→
Handarbeit, Einzelperson-Feedback→Automatisierung, ein Szenario→mehrere, Diagnose bei
Blockaden) war für sich genommen eine nachvollziehbare, methodisch begründbare Reaktion
auf eine erkannte Schwäche des vorherigen Ansatzes. Was in jedem Zyklus fehlte, war die
konsequente Anwendung der neuen Taktik auf das *gesamte* bisherige Material statt nur auf
die *nächsten* Schritte — die neuen Metriken laufen nicht rückwirkend, die Szenario-
Parallelisierung wurde nicht durchgehend bespielt, die Feedback-Disziplin der ersten
Runden wurde nicht in spätere Runden fortgeschrieben.

**Für Kapitel 5.1** liefert das einen ehrlichen, nicht beschönigten, aber auch nicht
entmutigenden Erzählbogen: eine Reihe methodisch sinnvoller Kurskorrekturen, deren größte
gemeinsame Schwäche die unvollständige Konsolidierung ist — nicht fehlendes Verständnis
oder fehlende Sorgfalt im Einzelfall. Das deckt sich mit der Empfehlung aus der
Machbarkeitsstudie (`machbarkeitsstudie_gif.md`, Abschnitt 9): Der nächste sinnvolle
Schritt ist nicht ein weiterer neuer Ansatz, sondern die konsequente Anwendung des bereits
entwickelten Werkzeugs (Metriken) auf das bereits vorhandene Material (`rainy_window`).
