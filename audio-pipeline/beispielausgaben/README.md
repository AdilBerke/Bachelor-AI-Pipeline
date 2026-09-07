# Beispielausgaben

Fünf vollständige 20-minütige Longform-Audios, eine je trainiertem Genre, samt der
automatisch erzeugten Bewertung (`score_bewertung.json`, acht Kategorien). Das sind
genau die fünf Kontrollausgaben, deren Bewertungswerte in Kapitel 8 der Arbeit
tabellarisch ausgewertet werden — als Beleg zum Nachhören und zur Überprüfung der
dort berichteten Zahlen.

| Ordner | Genre |
|---|---|
| `chillhop_lofi/` | Chillhop Lo-Fi |
| `jazz_lofi/` | Jazz Lo-Fi |
| `dreamy_lofi/` | Dreamy Lo-Fi |
| `study_lofi/` | Study Lo-Fi |
| `guitar_lofi/` | Guitar Lo-Fi |

Jeder Ordner enthält:
- `lange_audio.mp3` — die fertige Longform-Ausgabe (192 kbit/s, aus MusicGen +
  LoRA generiert, per Block-Looping auf 20 Minuten zusammengesetzt)
- `score_bewertung.json` — die automatische Bewertung in acht Kategorien
  (`code/src/Training/audio_bewertung.py`), jeweils Generiert-Wert, Genrestandard
  und Differenz

Die unkomprimierte WAV-Zwischendatei ist nicht enthalten (identischer Inhalt,
nur größer); zur Nachvollziehbarkeit reicht die MP3.
