# LOFI-QI — Quantifiziertes Video-Feedback-System

Dokumentiert den vollständigen Weg von subjektivem Nutzerfeedback zum messbaren Score.
Implementiert in: `code/src/Pipeline/web_api.py` + `src/routes/studio.video.tsx` + `src/lib/video-feedback.ts`

---

## Überblick

Wenn du auf der Studio-Website (`/studio/video`) ein generiertes Video bewertest, passieren
gleichzeitig zwei Dinge:

1. Dein **subjektives Urteil** wird in normierte Zahlen übersetzt (BARS-Methode)
2. Das System läuft **automatische Metriken** auf dem Video (evaluate_video.py + CLIP)

Beide werden zu einem einzigen Score zusammengerechnet: dem **LOFI-QI** (0.0 – 1.0).

---

## Schritt 1 — Dimensionales Feedback (BARS)

Statt einer einzigen Gesamtnote bewertest du **5 unabhängige Dimensionen**, je 1–5 Sterne.
Jede Dimension hat verankerte Beschreibungen pro Stern (Behaviorally Anchored Rating Scales,
Smith & Kendall 1963) — das reduziert subjektive Variation:

| Dimension | Skala-Anker 1 | Skala-Anker 5 | Mapped auf Parameter |
|---|---|---|---|
| **Stil-Treue** | Kein Anime-Stil, wirkt wie 3D-Render | Perfekter Lo-Fi-Stil, konsistente Cel-Shading-Ästhetik | `guidance_scale` |
| **Bewegungsqualität** | Komplett statisch, kein Element bewegt sich | Lebhafte, realistische Bewegung — Wasser, Blätter, Ohren | `motion_score` |
| **Bildschärfe** | Stark verschwommen, keine klaren Kanten | Sehr scharf, alle Details poliert und klar | `sharpness_score` |
| **Atmosphäre & Stimmung** | Falsche oder keine Stimmung, wirkt generisch | Vollständig immersiv, Lo-Fi-Feeling überzeugend | `aesthetic_score` |
| **Motiv-Genauigkeit** | Falsche Tiere/Objekte, Szene nicht erkennbar | Alle Motive exakt wie beschrieben dargestellt | `prompt_strength` |

Quelle: `src/lib/video-feedback.ts`, `FEEDBACK_DIMS`

### BARS → Zahl (Normierung)

```
bars_score = (Mittelwert der 5 Dimensionen − 1) / 4
```
Beispiel: alle Dimensionen auf 3 → bars_score = (3 − 1) / 4 = **0.50**
Alle auf 5 → bars_score = **1.00**

---

## Schritt 2 — Problem-Chips

Zusätzlich kannst du konkrete Probleme anklicken (Mehrfachauswahl):

| Chip | Gruppe | Braucht Nachtraining |
|---|---|---|
| Hase passt nicht | Motiv | Nein |
| Zu photorealistisch | Stil | Ja |
| Kein Lo-Fi-Stil | Stil | Ja |
| Falsche Farben | Stil | Nein |
| Keine / zu wenig Animation | Bewegung | Ja |

Jeder Chip fügt automatisch Text zum **Positiv-Prompt** und **Negativ-Prompt** hinzu —
das `scenario.yaml` wird direkt angepasst, bevor die nächste Generierung startet.

Quelle: `src/lib/video-feedback.ts`, `ISSUE_CHIPS`; Backend: `web_api.py`, `CORRECTIONS`-Dict

---

## Schritt 3 — Automatische Metriken (evaluate_video.py)

Parallel zu deiner Eingabe läuft das Backend automatisch auf dem MP4:

| Metrik | Methode | Gewicht in LOFI-QI |
|---|---|---|
| Temporal SSIM | Frame-zu-Frame-Ähnlichkeit (skimage) | Teil von `objective_score` |
| Sharpness | Laplacian-Varianz pro Frame | Teil von `objective_score` |
| Flicker | Mittlere Pixeldifferenz | Teil von `objective_score` |
| Motion | Farneback Optical Flow (OpenCV) | Teil von `objective_score` |
| Color Consistency | Farbhistogramm-Stabilität | Teil von `objective_score` |
| Brightness Consistency | Helligkeitsschwankung | Teil von `objective_score` |

Die 6 Metriken werden zu einem `overall_score` (0.0–1.0) zusammengefasst und als
`scenario_id/rounds/round_NN/*.metrics.json` gecacht.

Quelle: `Bachelorarbeit/pipeline/realistic_rabbit/evaluate_video.py`

---

## Schritt 4 — CLIP-Score (semantische Stiltreue)

Das Backend berechnet optional den **CLIP-Score** (OpenAI ViT-B/32):
Wie gut stimmt das generierte Bild mit dem verwendeten Prompt überein?

- Eingabe: erster Frame des Videos + Prompt-Text
- Ausgabe: Kosinus-Ähnlichkeit, normiert auf 0.0–1.0
- Gespeichert als `*.clip.json` neben dem MP4

Quelle: `web_api.py`, Funktion `video_evaluate()`

---

## Schritt 5 — LOFI-QI Formel

```
LOFI-QI = (BARS × 0.30 + Objektiv × 0.30 + CLIP × 0.20 + Stern × 0.05)
           ÷ Summe der verfügbaren Gewichte
```

| Komponente | Gewicht | Quelle |
|---|---|---|
| BARS (5 Dimensionen) | 30 % | dein strukturiertes Urteil |
| Objektiv (evaluate_video.py) | 30 % | automatische Signalverarbeitung |
| CLIP-Score | 20 % | semantische Prompt-Bild-Übereinstimmung |
| Gesamtstern (1–5, Likert) | 5 % | holistische Gesamtnote |
| *(freier Text)* | 0 % | gespeichert, aktuell nicht gewichtet |

**Wichtig:** Wenn eine Komponente nicht verfügbar ist (z.B. kein CLIP-Score berechnet),
wird ihr Gewicht aus Zähler **und** Nenner entfernt — keine Verzerrung durch Nullwerte.

Quelle: `web_api.py`, Zeilen 2624–2638

---

## Was danach passiert

Nach dem Absenden wird automatisch:

1. `lofi_qi` in `scenarios/<id>/feedback_history.jsonl` gespeichert (Zeile pro Bewertung)
2. `scenario.yaml` aktualisiert (Prompt, Negativ-Prompt, guidance_scale)
3. Neue Generierung mit dem aktualisierten Checkpoint gestartet → Job-ID zurückgegeben
4. Frontend wechselt zurück in die Trainingsphase und zeigt den Fortschritt

Quelle: `web_api.py`, Funktionen `video_quantified_feedback()` + `_append_feedback_history()`

---

## Feedback-Verlauf abfragen

```
GET /api/video/feedback-history?scenario_id=rabbit_lake
```

Gibt alle LOFI-QI-Einträge chronologisch zurück:

```json
{
  "history": [
    {
      "timestamp": "2026-08-10T14:22:00",
      "bars": { "stil_treue": 4, "bewegung": 2, "schaerfe": 3, "stimmung": 4, "motiv_genauigkeit": 3 },
      "bars_score": 0.65,
      "overall_star": 3,
      "objective_score": 0.784,
      "clip_score": 0.81,
      "lofi_qi": 0.731,
      "adjustments_applied": ["bewegung", "keine_animation"]
    }
  ]
}
```

---

## Ablauf-Diagramm

```
Du schaust das Video
        │
        ▼
5 Dimensionen bewerten (1–5 Sterne je)
+ Problem-Chips anklicken
+ optionaler Gesamtstern
        │
        ▼ POST /api/video/quantified-feedback
        │
   ┌────┴────────────────────────┐
   │                             │
   ▼                             ▼
BARS-Score berechnen     evaluate_video.py laufen lassen
(deine 5 Dimensionen     (SSIM, Schärfe, Motion, ...)
→ 0.0–1.0)               + CLIP-Score
                                 │
   └────────────┬────────────────┘
                │
                ▼
        LOFI-QI = gewichtetes Mittel
        (0.0 – 1.0)
                │
        ┌───────┴──────────┐
        ▼                  ▼
feedback_history.jsonl   scenario.yaml
(Verlauf)                (Prompt angepasst)
                          │
                          ▼
                  Neue Generierung startet
```

---

## Dateipfade

| Datei | Inhalt |
|---|---|
| `src/lib/video-feedback.ts` | BARS-Dimensionen, Issue-Chips, Typ-Definitionen |
| `src/routes/studio.video.tsx` | Feedback-UI, Submit-Logik |
| `code/src/Pipeline/web_api.py` | LOFI-QI-Formel, Prompt-Anpassung, Feedback-Speicherung |
| `Bachelorarbeit/pipeline/realistic_rabbit/evaluate_video.py` | 6 objektive Videometriken |
| `Bachelorarbeit/lofi_pipeline/scenarios/<id>/feedback_history.jsonl` | Gespeicherter LOFI-QI-Verlauf |
| `Bachelorarbeit/lofi_pipeline/scenarios/<id>/scenario.yaml` | Aktueller Prompt (wird angepasst) |
