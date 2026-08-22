# Lo-Fi Studio — Frontend

Lokale Studio-Weboberfläche für die Audio-Pipeline (`../audio-pipeline`). Ermöglicht Quellensuche,
Dataset-/LoRA-Steuerung, Longform-Generierung, Job-Überwachung und Bewertung/Vergleich der erzeugten
Audios über den Browser, anstatt jeden Schritt per CLI auszulösen.

Kein eigenständiges Produkt — das Frontend ist nur nutzbar, wenn parallel das lokale Backend aus
`../audio-pipeline` läuft (siehe [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md)).

## Tech-Stack

- React 19 + TypeScript
- TanStack Router (datei-basiertes Routing unter `src/routes/`) + TanStack Start/Vite
- Tailwind CSS 4
- Radix UI / shadcn-artige Komponenten (`src/components/ui/`)
- Recharts (Score-/Radar-Vergleiche)

## Voraussetzungen

- Node.js 22.x
- Laufendes Backend aus `../audio-pipeline` unter `http://127.0.0.1:8000` (Standardadresse,
  konfigurierbar in `src/lib/settings.ts`)

## Setup

```bash
npm install
npm run dev
```

Die Website ist danach standardmäßig unter `http://localhost:8080` erreichbar. Ohne erreichbares
Backend fällt die Oberfläche auf Mock-Daten zurück (`src/lib/mockData.ts`), damit UI-Arbeit auch
ohne laufende Pipeline möglich ist.

Weitere Skripte:

```bash
npm run build     # Produktions-Build
npm run lint      # ESLint
npm run format    # Prettier
```

## Struktur

| Pfad | Inhalt |
|---|---|
| `src/routes/` | Seiten: `studio.import` (Quellen), `studio.generate` (Audio), `studio.video` (Video/GIF-Mockup), `studio.library` (Ausgaben), `studio.evaluation` (Bewertung), `studio.jobs` (Läufe), `studio.model` (Steuerung) |
| `src/lib/api.ts` | zentrale HTTP-Schicht zum lokalen Backend |
| `src/lib/settings.ts` | persistierte UI-Einstellungen (u. a. Backend-URL) |
| `src/components/` | wiederverwendbare Komponenten (Player, Score-/Radar-Vergleiche, Sidebar) |
| `src/integrations/supabase/` | optionale Supabase-Anbindung |

## Umgebungsvariablen

`.env` enthält nur öffentliche, clientseitig sichere Supabase-Werte (`publishable key`, kein
Secret) und kann unverändert übernommen werden.
