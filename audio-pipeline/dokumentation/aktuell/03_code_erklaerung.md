# Code-Erklaerung

Dieses Dokument beschreibt die aktiven Audio-Dateien. Video/GIF-Dateien sind
nicht Teil dieser Erklaerung.

## Zentraler Einstieg

`code/start.py`
: Kurzer Starter fuer die lokalen Hauptaufgaben. Er zeigt Status, startet die
LoRA-Clip-Vorbereitung, baut gepruefte Trainingsdaten, startet LoRA oder
erzeugt lange Audios. Zusaetzlich gibt es Kurzbefehle fuer Top-10-Suche,
Referenzvergleich und bewusste LoRA-Freigabe.

## Crawler

`code/src/Crawler/quellen_suche.py`
: Allgemeine Suche, Download und Import externer Audioquellen.

`code/src/Crawler/quellen_finden.py`
: Spezialisierte Top-10-Suche fuer fehlende Genres im LoRA LoRA-Dataset.

## Dataset

`code/src/Dataset/genre_regeln.py`
: Gemeinsame Genre-Erkennung, Caption-Vorgaben und quellengetrennte
Train/Valid/Test-Auswahl.

`code/src/Dataset/zieldatensatz.py`
: Erstellt das genrebalancierte LoRA Dataset.

`code/src/Dataset/trainingsdaten_pruefen.py`
: Erstellt die aktiven geprueften Trainingsdaten, schliesst schlechte
Review-Quellen aus und laesst fehlende WAV-Dateien weg.

`code/src/Dataset/datensatz.py`
: Gebuendelter Dataset-Einstieg fuer Validierung, Audit und Clip-Erstellung.

`code/src/Dataset/genres_unterteilen.py`
: Ordnet vorhandene Clips nach Genres ein.

`code/src/Dataset/kuratiertes_dataset_erstellen.py`
: Erstellt ein kuratiertes Dataset fuer Uebergaenge und Melodie, wenn ein
gezielter Trainingsversuch gebraucht wird.

## Pipeline

`code/src/Pipeline/clips_vorbereiten.py`
: Automatisiert lokale MP3s, Download-Ergebnisse und Clip-Erstellung fuer das
LoRA Ziel.

`code/src/Pipeline/projekt_ablauf.py`
: Orchestriert die Audioerzeugung, Longform-Planung, Reports und optionale
Nachbearbeitung. Crawler/Suche bleiben getrennt. Nach erfolgreicher
Audioerstellung startet sie automatisch den Referenzvergleich gegen gute
MP3-Trainingsclips.

## Audio-Merkmale

`code/src/Merkmale/audio_merkmale.py`
: Eigenstaendige Extraktion technischer Audio-Merkmale. Die Datei
liest die Dataset-Manifeste, analysiert die WAV-Clips und schreibt Reports in
`training/musicgen/merkmale/`.

## Training und Generierung

`code/src/Training/lora.py`
: Sicherer LoRA-Starter. Prueft Dataset, Duplikate, fehlende Dateien, Resume
und Checkpoints. Neue Checkpoints werden zuerst als `bewertung_offen`
registriert. Sie werden erst nach menschlicher Bewertung mit
`--freigeben-checkpoint` als aktiver Adapter genutzt.

`code/src/Training/musicgen_steuerung.py`
: Zentrale MusicGen-Steuerung fuer Training, Review, Generierung, Longform und
Statusbefehle.

`code/src/Training/bewertung_audios_erstellen.py`
: Erzeugt 8 neue 30s-Testaudios mit dem aktuellen LoRA-Adapter und legt eine
Bewertungs-CSV an.

`code/src/Training/trainingsclips_bewertung.py`
: Erstellt Hoerproben aus echten Trainingsclips. Diese Datei generiert keine
neue Musik.

`code/src/Training/audio_erstellen.py`
: Erzeugt neue MusicGen-Abschnitte, prueft Kandidaten technisch und baut daraus
lange Audios. Die Kandidatenauswahl nutzt Sekundencheck, BPM-Kontrolle,
Uebergangs-Score, lokale Genre-Pruefung und technische MP3-Referenzprofile aus
den geprueften Trainingsdaten.

`code/src/Training/audio_loopen.py`
und `code/src/Training/clips_loopen.py`
: Loop-Fallback fuer vorhandene Clips. Das ist nicht die Hauptlogik, hilft aber
fuer stabile lange Audios.

`code/src/Training/audio_bewertung.py`
: Analysiert Audios und menschliche Bewertungen. Nach einer neuen
Audioerstellung erzeugt diese Datei zusaetzlich die 5-Score-Bewertung mit den
Kategorien `Technische Audioqualität`, `Musikalische Kohärenz`, `Genre-Treue`,
`Übergangsqualität` und `Referenzähnlichkeit`. Die sichtbare HTML-Datei zeigt
genau zwei getrennte Graphen: Genrestandard und generierte Audio.

`code/src/Training/referenz_vergleich.py`
: Vergleicht erzeugte MusicGen-/LoRA-Audios mit guten MP3-Referenzclips aus
den geprueften Trainingsdaten. Die Datei misst unter anderem Lautheit, Peaks,
stille Sekunden, Bass, Hoehen, Signalton-Risiko, Dynamik und grobes BPM. Sie
startet kein Training und erzeugt keine neue Musik.

`code/src/Training/genre_pruefung.py`
: Vergleicht generierte Audios lokal mit guten Referenzclips je Genre.

`code/src/Training/audio_nachbearbeitung.py`
: Optionale Nachbearbeitung fuer fertige Audios. Veraendert keine LoRA-Adapter
und keine Trainingsdaten.

`code/src/Training/audio_modelle_einrichten.py`
: Richtet optionale lokale Analyse-/Nachbearbeitungsmodelle wie CLAP oder
Demucs ein.

`code/src/Training/audio_veroeffentlichen.py`
: Pusht bewusst ausgewaehlte Audios nach GitHub.

## Wissenschaftliche Einordnung

Die Pipeline ist ein Human-in-the-Loop-System:

1. Quellen werden gesammelt.
2. Clips werden technisch vorbereitet.
3. Schlechte Quellen werden durch menschliche Bewertung ausgeschlossen.
4. LoRA wird in kleinen Schritten trainiert.
5. Neue Testaudios werden technisch gefiltert.
6. Der Mensch entscheidet, ob ein Checkpoint musikalisch besser ist.
7. Nur freigegebene Checkpoints werden fuer Longform-Audios genutzt.
8. Lange Audios entstehen aus MusicGen-Abschnitten oder aus einem Loop-Fallback.
9. Am Ende misst der Referenzvergleich die Abweichung zu guten MP3-Clips.
