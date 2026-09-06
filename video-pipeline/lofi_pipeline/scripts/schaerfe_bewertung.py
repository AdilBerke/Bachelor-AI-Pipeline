#!/usr/bin/env python3
"""Schaerfe-Bewertung fuer Videos/GIFs -- mit vergleichendem Score.

Berechnet pro Frame etablierte No-Reference-Schaerfemasse, aggregiert ueber die
Zeit und vergleicht anschliessend ALLE uebergebenen Clips gegeneinander
(retrospektiv, batch-relativ). Ergebnis: ein Schaerfe-Score 0..100 pro Clip
plus Rangliste. Nur OpenCV + NumPy noetig.

Einzelmetriken (pro Frame, dann Mittel ueber Zeit):
  - lap_var   : Varianz des Laplacian (Pech-Pacheco 2000). Standard-Schaerfemass.
  - tenengrad : Mittlere Sobel-Gradientenenergie. Robuster gegen Rauschen.
  - brenner   : Brenner-Gradient (nur Info, geht nicht in den Score ein).
  - fft_ratio : Anteil hoher Frequenzen im Spektrum (0..1). Weitgehend
                aufloesungs-/helligkeitsunabhaengig.

Score (batch-relativ):
  Jede der 3 Score-Metriken (lap_var, tenengrad, fft_ratio) wird ueber den
  Batch per Min-Max auf 0..1 normiert, dann gewichtet gemittelt
  (0.40 / 0.30 / 0.30) und *100 gerechnet. Der schaerfste Clip im Batch
  bekommt ~100, der unschaerfste ~0. Der Score sagt also "relativ zu DIESER
  Auswahl", nicht absolut.

  konstanz : 0..100, wie wenig die Schaerfe ueber die Frames schwankt
             (100 = voellig konstant). Fuer Loop-GIFs idealerweise hoch.

Aufruf:
    python schaerfe_bewertung.py clip1.mp4 clip2.gif ordner/*.mp4
    python schaerfe_bewertung.py --json bericht.json --every 3 "scenarios/*/web_outputs/*.mp4"
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Gewichte der Score-Metriken (Summe = 1.0)
GEWICHTE = {"lap_var": 0.40, "tenengrad": 0.30, "fft_ratio": 0.30}


def lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def brenner(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    d = g[:, 2:] - g[:, :-2]
    return float(np.mean(d * d))


def fft_ratio(gray: np.ndarray, frac: float = 0.25) -> float:
    g = gray.astype(np.float64)
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    ry, rx = int(h * frac / 2), int(w * frac / 2)
    total = mag.sum() + 1e-12
    low = mag[cy - ry:cy + ry, cx - rx:cx + rx].sum()
    return float(1.0 - low / total)


METRICS = {
    "lap_var": lap_var,
    "tenengrad": tenengrad,
    "brenner": brenner,
    "fft_ratio": fft_ratio,
}


def frames(path: str, every: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"kann nicht oeffnen")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        i += 1
    cap.release()


def evaluate(path: str, every: int) -> dict:
    series = {k: [] for k in METRICS}
    n = 0
    for gray in frames(path, every):
        for k, fn in METRICS.items():
            series[k].append(fn(gray))
        n += 1
    if n == 0:
        raise RuntimeError("keine Frames dekodierbar (evtl. AV1 -> vorher nach H.264 wandeln)")

    p = Path(path)
    kurz = "/".join(p.parts[-3:]) if len(p.parts) >= 3 else p.name
    out = {"datei": path, "name": p.name, "kurz": kurz, "frames_bewertet": n}
    for k, vals in series.items():
        a = np.asarray(vals, dtype=np.float64)
        out[k] = {
            "mean": float(a.mean()),
            "min": float(a.min()),
            "std": float(a.std()),
            "p10": float(np.percentile(a, 10)),
        }
    # zeitliche Konstanz der Schaerfe (Variationskoeffizient von lap_var, klein = konstant)
    lv = np.asarray(series["lap_var"], dtype=np.float64)
    out["cv_lap"] = float(lv.std() / (lv.mean() + 1e-9))
    return out


def _minmax(werte: list[float]) -> list[float]:
    lo, hi = min(werte), max(werte)
    if hi - lo < 1e-12:
        return [0.5 for _ in werte]
    return [(w - lo) / (hi - lo) for w in werte]


def score_batch(ergebnisse: list[dict]) -> None:
    """Fuegt jedem Ergebnis-Dict 'score', 'konstanz' und 'rang' hinzu."""
    if len(ergebnisse) == 1:
        ergebnisse[0]["score"] = None
        ergebnisse[0]["konstanz"] = None
        ergebnisse[0]["rang"] = 1
        return

    norm = {m: _minmax([e[m]["mean"] for e in ergebnisse]) for m in GEWICHTE}
    konst_norm = _minmax([-e["cv_lap"] for e in ergebnisse])  # weniger Schwankung -> hoeher

    for i, e in enumerate(ergebnisse):
        s = sum(GEWICHTE[m] * norm[m][i] for m in GEWICHTE)
        e["score"] = round(100.0 * s, 1)
        e["konstanz"] = round(100.0 * konst_norm[i], 1)

    for rang, e in enumerate(sorted(ergebnisse, key=lambda x: x["score"], reverse=True), 1):
        e["rang"] = rang


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pfade", nargs="+", help="Video-/GIF-Dateien oder Globs")
    ap.add_argument("--every", type=int, default=2, help="nur jeden N-ten Frame (default 2)")
    ap.add_argument("--json", type=str, default=None, help="vollen Bericht als JSON speichern")
    args = ap.parse_args(argv)

    dateien: list[str] = []
    for p in args.pfade:
        treffer = sorted(glob.glob(p))
        dateien.extend(treffer or [p])
    # Duplikate raus, Reihenfolge halten
    dateien = list(dict.fromkeys(dateien))

    ergebnisse: list[dict] = []
    for d in dateien:
        if not Path(d).exists():
            print(f"! fehlt: {d}", file=sys.stderr)
            continue
        try:
            ergebnisse.append(evaluate(d, max(1, args.every)))
        except Exception as e:  # noqa: BLE001
            print(f"! Fehler {Path(d).name}: {e}", file=sys.stderr)

    if not ergebnisse:
        return 1

    score_batch(ergebnisse)

    print(f"\n  Schaerfe-Vergleich  ({len(ergebnisse)} Clips, jeder {args.every}. Frame)\n")
    kopf = f"  {'#':>3}  {'Score':>6}  {'Konst':>6}  {'lap_var':>10}  {'tenengrad':>11}  {'fft':>5}  Datei"
    print(kopf)
    print("  " + "-" * (len(kopf) - 2))
    for e in sorted(ergebnisse, key=lambda x: x["rang"]):
        sc = "  n/a " if e["score"] is None else f"{e['score']:6.1f}"
        ko = "  n/a " if e["konstanz"] is None else f"{e['konstanz']:6.1f}"
        print(
            f"  {e['rang']:>3}  {sc}  {ko}  "
            f"{e['lap_var']['mean']:10.1f}  {e['tenengrad']['mean']:11.1f}  "
            f"{e['fft_ratio']['mean']:5.3f}  {e['kurz']}"
        )
    if len(ergebnisse) == 1:
        print("\n  Hinweis: Score ist batch-relativ und braucht >= 2 Clips.")
    else:
        print("\n  Score 100 = schaerfster Clip dieser Auswahl, 0 = unschaerfster.")
        print("  Konst 100 = Schaerfe ueber die Frames am konstantesten (gut fuer Loops).")

    if args.json:
        Path(args.json).write_text(json.dumps(ergebnisse, indent=2, ensure_ascii=False))
        print(f"\n  JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
