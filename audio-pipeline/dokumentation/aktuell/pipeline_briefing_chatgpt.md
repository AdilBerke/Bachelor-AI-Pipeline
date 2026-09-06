# Briefing: Die 7 Pipeline-Bereiche der Lo-Fi Studio-Website

Kontext für dich (ChatGPT): Dies ist die technische Grundlage für ein Flowchart einer
Bachelorarbeit-Website. Die Website hat sieben große Funktionsbereiche ("Studio-Seiten"),
die zusammen zwei Pipelines steuern: eine Audio-Pipeline (MusicGen + LoRA, produktiv und
vollständig fertig) und eine Video-Pipeline (LTX-Video, aktuell nur als Oberfläche
vorhanden, technisch noch nicht angebunden). Unten stehen alle sieben Bereiche im Detail:
Zweck, was der Nutzer sieht/tut, was im Hintergrund technisch passiert, welche Daten rein-
und rausfließen, und wie die Bereiche miteinander verbunden sind. Nutze das, um präzisere
Flowchart-Beschriftungen, Kästchen-Texte oder eine ausführlichere Diagrammstruktur zu
formulieren.

## Gesamtarchitektur (Kurzfassung)

- **Frontend:** React + TanStack Router, eigenes Git-Repository, Vite-Dev-Server auf
  Port 8080. Die sieben Bereiche sind sieben Seiten im selben Menü.
- **Backend:** ein reiner Python-`http.server` (kein Framework), Port 8000, in derselben
  Codebasis wie die Pipelines. Startet alle Trainings-/Generierungs-/Suchaufgaben als
  eigene Betriebssystem-Prozesse (`subprocess.Popen`), nicht als Funktionsaufrufe im
  eigenen Prozess.
- **Jobs/Logs:** jeder gestartete Hintergrundprozess bekommt eine Job-ID und schreibt sein
  Live-Log in eine eigene Datei (`training/musicgen/web_jobs/<job_id>.log`). Das Frontend
  pollt den Job-Status und zeigt den Log-Inhalt live an.
- **Die Website ist nur die Oberfläche für die Audio-Pipeline** (Punkt 1–5, 7 unten) und
  zeigt zusätzlich einen Mockup-Bereich für die separate Video-Pipeline (Punkt 6), die
  komplett eigenen Code in einem anderen Ordner hat und (noch) nicht wirklich angebunden ist.

---

## 1. Quellen — Rohmaterial sammeln

**Zweck:** Ausgangspunkt der gesamten Audio-Pipeline. Hier kommt das Roh-Audiomaterial rein,
aus dem später der Trainingsdatensatz gebaut wird.

**Was der Nutzer sieht/tut:**
- Eine "Top10-Suche" pro Genre: die Website durchsucht YouTube automatisch nach den zehn am
  besten passenden, lizenzfrei markierten Videos für ein gewähltes Genre (z. B. "Jazz Lofi"),
  zeigt sie als Kandidatenliste mit Status "offen"/"importiert"/"abgelehnt".
- Manueller Link-Import: Nutzer kann einen einzelnen YouTube-Link direkt eintragen.
- Drag & Drop-Upload: eigene MP3-Dateien können direkt hochgeladen werden.
- Ein Sammel-Button "Alle offenen Videos importieren": lädt alle noch offenen Top10-Kandidaten
  über alle Genres gesammelt herunter, statt einzeln klicken zu müssen.

**Technisch dahinter:**
- `code/src/Crawler/quellen_suche.py` macht die eigentliche YouTube-Suche und den Download
  (über `yt-dlp`), mit Lizenzfilter ("no copyright") und Stimmungs-/Genre-Stichworten.
- Jeder Video-Download läuft in einem eigenen Kindprozess mit hartem Timeout (150 Sekunden),
  damit ein einzelnes hängendes Video (z. B. ein Livestream) nicht die ganze Warteschlange
  blockiert.
- Bekannte, aktuell ungelöste Einschränkung: YouTube verlangt bei Massenimporten oft eine
  Bot-Bestätigung ("Sign in to confirm you're not a bot"), was Downloads ohne Browser-Cookies
  zuverlässig scheitern lässt.

**Eingabe:** nichts (Startpunkt) — nur Sucheingaben des Nutzers (Genre, Stimmung).
**Ausgabe:** MP3-Dateien auf der Festplatte, organisiert pro Genre und Quelle.
**Fließt weiter nach:** Steuerung (dort werden aus diesen MP3s die Trainingsclips geschnitten).

---

## 2. Steuerung — das Modell trainieren

**Zweck:** Der "Maschinenraum". Hier wird aus dem Rohmaterial ein trainiertes,
generierungsfähiges Modell gebaut und kontrolliert. Das ist der technisch komplexeste
Bereich der ganzen Website.

**Was der Nutzer sieht/tut, in drei Teilschritten:**
1. **Dataset bauen:** aus den MP3-Quellen werden feste 30-Sekunden-Clips geschnitten
   (nicht überlappend, erste/letzte 20 Sekunden jeder Quelle werden übersprungen, maximal
   250 Clips pro Einzelquelle, gleichmäßig über die gesamte Quelllänge verteilt statt nur
   den Anfang zu nehmen). Ein Auswahlalgorithmus verteilt die Clips so über die Quellen,
   dass keine einzelne Quelle den Datensatz dominiert und Trainings-/Test-Teilmengen nie
   dieselbe Quelle teilen.
2. **LoRA-Training starten/fortsetzen:** ein "Neues Training starten" oder "Fortsetzen"-Knopf
   löst das eigentliche Fine-Tuning aus (Details siehe technischer Abschnitt unten).
3. **Testaudios erzeugen:** kurze Beispielgenerierungen pro Genre, um den aktuellen
   Trainingsstand zu prüfen, bevor er live geschaltet wird.

**Technisch dahinter:**
- Basismodell: `facebook/musicgen-melody-large` (ca. 15 GB, Meta AI), angepasst per **LoRA**
  (Low-Rank Adaptation) — es wird nicht das ganze Modell neu trainiert, sondern nur kleine,
  zusätzliche Gewichtsmatrizen in bestimmten linearen Schichten (`out_proj`, `linear1`,
  `linear2`), Rang 8, Alpha 16. Das Basismodell bleibt eingefroren.
- Training läuft über `code/src/Training/lora.py`, batch_size 1 mit Gradientenakkumulation 8
  (effektive Batchgröße 8), Lernrate typischerweise 5e-6.
- Checkpoints werden alle 25 Schritte gespeichert, sodass ein Absturz jederzeit vom letzten
  Checkpoint fortgesetzt werden kann (siehe unten, häufige native Abstürze der zugrunde
  liegenden EnCodec-Audiokodierung).
- Ein Freigabe-Schritt ("Checkpoint freigeben") setzt einen bestimmten Trainingsstand als
  den offiziell aktiven, für Generierung nutzbaren Adapter.

**Eingabe:** MP3-Quellen aus "Quellen".
**Ausgabe:** ein trainierter, freigegebener LoRA-Adapter (eine kleine Gewichtsdatei, die auf
das große Basismodell aufgesetzt wird) + Testaudios zur Kontrolle.
**Fließt weiter nach:** Audio (die eigentliche Longform-Generierung nutzt den hier
freigegebenen Adapter).
**Bekommt Rückmeldung von:** Bewertung (siehe dort — Trainingsentscheidungen basieren auf den
dort gemessenen Qualitätswerten).

---

## 3. Audio — finale Longform-Audios erzeugen

**Zweck:** Der eigentliche "Produktions"-Schritt — aus dem freigegebenen Modell werden lange,
fertige Musikstücke erzeugt (z. B. 20 Minuten), nicht nur kurze Testschnipsel.

**Was der Nutzer sieht/tut:**
- Genre auswählen, Zieldauer wählen (z. B. 20 Minuten), Segmentlänge frei wählbar (mit
  Standardwerten oder eigener Eingabe).
- Startet die Generierung als Hintergrundjob, sieht Live-Fortschritt.

**Technisch dahinter — die wichtigste Einschränkung der ganzen Pipeline:**
- MusicGen kann pro Durchlauf technisch nur **maximal 30 Sekunden** am Stück erzeugen
  (harte Architekturgrenze durch sinusförmige Positions-Embeddings, 1503 Tokens). Für
  20-minütige Stücke gibt es zwei mögliche Strategien:
  - **Block-Looping** (die tatsächlich verwendete Methode): mehrere native 30-Sekunden-
    Blöcke werden einzeln erzeugt und im Audio-Bereich aneinandergehängt/übergeblendet
    (Crossfade), ähnlich wie ein DJ mehrere Tracks mischt.
  - Kontinuierliche Blöcke (Modell-interne Fortsetzung/`extend_stride`) wurden getestet,
    verursachen aber hörbare interne Nahtstellen — deshalb verworfen.
- Für jeden Abschnitt werden mehrere Kandidaten erzeugt und automatisch geprüft (Lautstärke,
  BPM-Toleranz, Genre-Ähnlichkeit), verworfene Kandidaten werden protokolliert.
- Bekannte, noch nicht behobene Qualitätslücke: die automatische "Ist dieser Abschnitt
  stumm?"-Prüfung misst nur Lautstärke (RMS), nicht Frequenzinhalt — ein Abschnitt, der zu
  99 % aus Bassbrummen ohne Melodie besteht, kann fälschlich als "laut genug" durchgehen.

**Eingabe:** freigegebener LoRA-Adapter aus "Steuerung".
**Ausgabe:** eine fertige lange Audiodatei (MP3 + WAV).
**Fließt weiter nach:** Bewertung (automatische Qualitätsmessung direkt im Anschluss) und
Ausgaben (die fertige Datei landet in der Bibliothek).

---

## 4. Bewertung — automatische Qualitätsmessung

**Zweck:** Objektiv messen, wie gut eine generierte Audio tatsächlich ist — nicht nur "hört
sich gut an", sondern mit echten, berechneten Zahlen.

**Was der Nutzer sieht/tut:**
- Genre und Audio auswählen.
- Ein Radar-Diagramm (Netzdiagramm) zeigt die gemessenen Werte in acht Kategorien.
- Ein "Jetzt bewerten"-Knopf löst die Messung aus, falls noch keine vorliegt.

**Technisch dahinter — die fünf echten Kategorien** (es gibt keine weiteren; alles andere
wären erfundene/nicht berechnete Werte):
1. **Technische Audioqualität** — u. a. Clipping, Rausch-/Stille-Anteile.
2. **Musikalische Kohärenz** — Konsistenz innerhalb des Stücks.
3. **Genre-Treue** — Distanz zu einem "Genre-Standard-Profil" (aus echten Referenzclips
   berechnete typische Werte für BPM, Bassanteil, Snareanteil, Höhenanteil,
   Lautstärke-Spannweite, spektraler Schwerpunkt). Historisch die instabilste und
   niedrigste Kategorie.
4. **Übergangsqualität** — wie sauber die Block-Looping-Übergänge (Crossfades) klingen,
   gemessen an Lautstärke-Sprüngen, spektralen Sprüngen, Klicks an den Nahtstellen.
5. **Referenzähnlichkeit** — direkter Abstand zu echten Referenz-MP3-Clips desselben Genres.
- Berechnet in `code/src/Training/audio_bewertung.py`, Funktion `bewerte_audio_mit_genrestandard`,
  läuft normalerweise automatisch direkt nach jeder Generierung.
- Wichtiger methodischer Befund aus der Praxis: eine einzelne Messung ist stark verrauscht
  (bei exakt demselben, unveränderten Modell schwankte "Genre-Treue" zwischen zwei Messungen
  um bis zu 42 Punkte) — deshalb werden Vergleiche inzwischen über drei Wiederholungsläufe
  gemittelt.

**Eingabe:** generierte Audiodatei aus "Audio".
**Ausgabe:** Score-Werte (JSON + Radar-Diagramm-Daten).
**Fließt weiter nach:** Ausgaben (Scores werden dort mit angezeigt) UND — das ist die
wichtigste Rückkopplungsschleife der ganzen Pipeline — zurück nach Steuerung: die
Ergebnisse hier bestimmen, ob ein neuer Trainingsversuch behalten, verworfen oder weiter
angepasst wird.

---

## 5. Ausgaben — die Bibliothek

**Zweck:** Zentrale Übersicht über alles, was je erzeugt wurde.

**Was der Nutzer sieht/tut:**
- Liste aller generierten Longform-Audios mit eingebautem Player.
- Download-Möglichkeit.
- Link zu einem ausführlicheren Report pro Audio.
- Score-Vergleich zwischen mehreren Audios (eigene SVG-Liniendiagramme, kein externes
  Chart-Framework).

**Technisch dahinter:** liest die Dateisystemstruktur der generierten Audio-Ordner aus, holt
pro Ordner die zugehörige `score_bewertung.json` (aus Bereich 4) und stellt sie zusammen.

**Eingabe:** fertige Audios aus "Audio" + Scores aus "Bewertung".
**Ausgabe:** nichts weiter fließt von hier aus — es ist der Endpunkt der Kette für den
Endnutzer (zum Anhören/Herunterladen).

---

## 6. Video — Design/Mockup (separate Pipeline, nicht angebunden)

**Zweck:** Zeigt, wie die künftige Video-Integration aussehen soll — aktuell **rein visuelles
Mockup ohne echte Funktion**, bewusst so gekennzeichnet.

**Was der Nutzer sieht/tut (alles ohne echte Backend-Anbindung):**
- Drei Plätze für fertige GIF-Loops (aktuell Platzhalter).
- Ein "Eigenes GIF erstellen"-Flow: Name, Beschreibung, Referenzbilder und MP4-Clips per
  Drag & Drop.
- Ein Bewertungs-/Wünsche-Feedback-Loop mit Sternebewertung und Problem-Auswahl-Chips.

**Die echte, komplett getrennte Video-Pipeline existiert und funktioniert bereits**, hat
aber keine Verbindung zu dieser Website-Seite:
- Eigener Ordner `Bachelorarbeit/lofi_pipeline/`, eigenes Modell **LTX-Video 13B**
  (`LTXV_13B_097_DEV`), erzeugt kurze Videoloops zu vordefinierten "Szenarien" (z. B.
  "rainy_window", "rabbit_lake").
- Eigenes LoRA-Training pro Szenario, ähnliches Konzept wie bei Audio, aber komplett
  getrennter Code.
- Eigene, bereits funktionierende Weboberfläche auf einem anderen Port (Flask, Port 7860),
  unabhängig von der Studio-Website.
- Kein Szenario gilt bisher als fertig trainiert (auch das am weitesten fortgeschrittene
  nicht); ein Szenario hat einen bekannten Darstellungsfehler (rendert manchmal drei statt
  zwei Objekte).

**Eingabe/Ausgabe zur restlichen Website:** keine — bewusste Design-Entscheidung, um die
funktionierende Audio-Pipeline nicht durch unfertige Video-Integration zu gefährden.

---

## 7. Läufe — Job-Übersicht (quer zu allen anderen Bereichen)

**Zweck:** Kein eigener Pipeline-Schritt im Sinne von "Daten fließen hindurch", sondern eine
Beobachtungs-/Kontrollebene über allen anderen Bereichen gleichzeitig.

**Was der Nutzer sieht/tut:**
- Liste aller aktuell laufenden und vergangenen Hintergrundjobs (aus Quellen, Steuerung,
  Audio) mit Status (laufend/fertig/fehlgeschlagen).
- Live-Log-Ansicht pro Job.
- Abbrechen-Knopf für laufende Jobs.

**Technisch dahinter:** jede der anderen Aktionen (Top10-Suche, Dataset bauen, Training,
Generierung) wird vom Backend als eigener Prozess gestartet und bekommt eine Job-ID; dieser
Bereich liest nur den gesammelten Status aller Job-IDs aus, startet selbst nichts.

**Wichtige Einschränkung:** Wenn Prozesse außerhalb der Website direkt per Kommandozeile
gestartet werden (z. B. für Experimente, die die Website-Oberfläche nicht abbildet), laufen
sie zwar real weiter, tauchen hier aber **nicht** auf — die Job-Verfolgung ist an den
Start-über-die-Website gekoppelt, nicht an den Prozess selbst.

---

## Der Gesamt-Mechanismus in einem Satz

Quellen sammelt Rohmaterial → Steuerung baut daraus ein Modell → Audio erzeugt damit fertige
Musik → Bewertung misst objektiv, wie gut sie ist, und diese Messung fließt direkt zurück in
Steuerung, um die nächste Trainingsentscheidung zu treffen (behalten / verwerfen / weiter
anpassen) → Ausgaben sammelt alles zum Anhören. Läufe beobachtet diesen gesamten Ablauf quer
über alle Schritte. Video teilt sich zwar dieselbe Website, hat aber keinerlei Verbindung in
diese Kette — es ist ein eigenständiges, noch unverbundenes zweites System.
