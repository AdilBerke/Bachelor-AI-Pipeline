# Studio-Frontend

Weboberfläche des Projekts: React mit TanStack Start, Tailwind CSS und Recharts.
Sie steuert Audio- und Videopipeline, zeigt Bewertungen an und verwaltet die
erzeugten Medien.

Die eigentliche Verarbeitung findet nicht im Browser statt. Das Frontend spricht
ausschließlich das lokale Python-Backend auf Port 8000 an (feste Adresse in
`src/lib/settings.ts`).

## Starten

```bash
npm install
npm run dev -- --port 8080
```

Erreichbar unter `http://localhost:8080`. Das Backend muss parallel laufen, siehe
[`../NACHBAUANLEITUNG.md`](../NACHBAUANLEITUNG.md) Schritt 6.

Es wird **keine** Konfigurationsdatei und kein Zugangsschlüssel benötigt. Die Anwendung
verwendet keine externen Dienste; auch die Schriften liegen lokal unter `public/fonts/`.

## Aufbau

| Pfad | Inhalt |
|---|---|
| `src/routes/` | Seiten, dateibasiertes Routing (siehe `src/routes/README.md`) |
| `src/components/` | Wiederverwendbare Bausteine, u. a. `RadarComparison`, `ScoreComparison` |
| `src/lib/api.ts` | Backend-Anbindung und Typen, inkl. der acht Bewertungskategorien |
| `public/fonts/` | Lokal eingebundene Schriften (Inter, JetBrains Mono) |

## Seiten

Quellen · Audio · Video · Endprodukt · Bibliothek · Bewertung · Läufe · Steuerung

Die Bewertungsansicht stellt die acht vom Backend berechneten Kategorien sowohl als
Radar-Diagramm als auch tabellarisch dar, jeweils im Vergleich zum Genrestandard.

## Entstehung

[`DESIGN_BRIEFING.md`](DESIGN_BRIEFING.md) enthält den ursprünglichen Gestaltungsauftrag,
aus dem die erste Fassung der Oberfläche erzeugt wurde.
