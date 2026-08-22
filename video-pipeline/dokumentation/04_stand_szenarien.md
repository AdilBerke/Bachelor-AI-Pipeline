# Aktueller Stand pro Szenario (Stand 04.08.2026)

Kurzreferenz. Details und Quellen siehe `02_verlauf.md`.

| Szenario | Runden | Letzte Runde | Bewertet? | Bewegung/Animation erreicht? | Status |
|---|---|---|---|---|---|
| `rainy_window` | 14 | 21.07.2026 | **Nein** — Round 14 hat kein `feedback.json` | Ja, laut früherem Feedback vorhanden, aber nicht quantifiziert | Am weitesten fortgeschritten, aber kein freigegebener Endstand |
| `rabbit_lake` | 9 | vor Absturz | Nur bis Round 5 dokumentiert (`RESCUE_PLAN.md`) | **Nein** — Rescue-Plan-Phase 2 nie erreicht | Blockiert: Grundproblem (3 statt 2 Figuren) ungelöst; Round 9 endete mit CUDA-OOM-Absturz |
| `lofi_girl_desk` | 0 *(im neuen System)* | – | – | – | Formal angelegt, aber nie unter dem Szenario-System trainiert. Die inhaltlich zugehörige Vorarbeit liegt in Phase 1/2 (`training/video/ltx_lora_manual` + `round2`–`round11`, siehe `02_verlauf.md`) und wurde nicht migriert |
| `cafe_scene` | 0 | – | – | – | Nur Konfigurationsordner, nie begonnen |
| `library_study` | 0 | – | – | – | Nur Konfigurationsordner, nie begonnen |
| `train_window` | 0 | – | – | – | Nur Konfigurationsordner, nie begonnen |

## Was das für die Bachelorarbeit bedeutet

- Es gibt **keinen** Szenario-Stand, der als "fertig trainiertes Modell" bezeichnet
  werden kann — auch `rainy_window` nicht, da die letzte Runde unbewertet ist.
- Falls ein Ergebnis für die Arbeit gebraucht wird, ist `rainy_window` der einzig
  sinnvolle Kandidat: höchste Rundenzahl, am längsten stabil weiterentwickelt, keine
  ungelösten Blocker.
- `rabbit_lake` eignet sich nicht als "fertiges" Beispiel, ist aber als **dokumentierter
  Negativbefund** wissenschaftlich verwertbar (Prior-Bias eines großen Basismodells lässt
  sich mit wenig LoRA-Daten nicht zuverlässig überschreiben — siehe `RESCUE_PLAN.md` und
  `02_verlauf.md`, Phase 4).
- Vier von sechs Szenarien existieren ausschließlich als Konfiguration ohne jede
  Trainingsaktivität.
