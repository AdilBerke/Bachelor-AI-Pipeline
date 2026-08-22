"""
Automatische Videoqualitatsmessung fuer Hase-am-See Generierungen.

Metriken (alle lokal, kein Download noetig):
  1. Temporal Consistency (SSIM)  — Stabilitaet zwischen Frames (1.0 = perfekt)
  2. Sharpness (Laplacian)        — Schaerfe pro Frame, Mittelwert
  3. Flicker Score                — Frame-zu-Frame Pixelaenderung (0.0 = kein Flackern)
  4. Motion Score                 — Optischer Fluss (0.0 = eingefroren, ~1-5 = dezent)
  5. Color Consistency            — Farbstabilitaet ueber alle Frames
  6. Brightness Consistency       — Helligkeitsschwankung

Verwendung:
  python evaluate_video.py outputs/test_B/rabbit_t2v_768x432.mp4
  python evaluate_video.py outputs/test_B/rabbit_t2v_768x432.mp4 --save-report
  python evaluate_video.py --compare outputs/test_A/ outputs/test_B/ outputs/test_C/
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ── Farb-Hilfsfunktionen ────────────────────────────────────────────────────

def clr(c, t): return f"\033[{c}m{t}\033[0m"
def cyan(t):   return clr("36", t)
def green(t):  return clr("32", t)
def yellow(t): return clr("33", t)
def red(t):    return clr("31", t)
def dim(t):    return clr("2",  t)
def bold(t):   return clr("1",  t)


def score_color(val, good_high=True):
    """Faerbt Wert gruen/gelb/rot je nach Qualitaet."""
    if good_high:
        if val >= 0.8: return green(f"{val:.4f}")
        if val >= 0.6: return yellow(f"{val:.4f}")
        return red(f"{val:.4f}")
    else:  # niedrig ist gut (z.B. Flicker)
        if val <= 0.02: return green(f"{val:.4f}")
        if val <= 0.06: return yellow(f"{val:.4f}")
        return red(f"{val:.4f}")


# ── Frame-Extraktion ─────────────────────────────────────────────────────────

def extract_frames(video_path: Path, max_frames=None):
    """Gibt alle Frames als numpy-Array (H, W, 3) zurueck."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps, (w, h), total


# ── Metrik 1: Temporal SSIM ─────────────────────────────────────────────────

def temporal_ssim(frames):
    """
    Mittlerer SSIM zwischen jeweils zwei aufeinanderfolgenden Frames.
    1.0 = identische Frames (perfekt stabil)
    0.0 = komplett unterschiedliche Frames (starkes Flackern)
    Typische gute Werte: 0.85 - 0.98
    """
    scores = []
    for i in range(len(frames) - 1):
        a = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        s = ssim(a, b, data_range=255)
        scores.append(s)
    return float(np.mean(scores)), float(np.min(scores)), float(np.std(scores))


# ── Metrik 2: Sharpness (Laplacian Variance) ────────────────────────────────

def sharpness_score(frames):
    """
    Laplacian-Varianz pro Frame — hohe Varianz = scharfes Bild.
    Normiert auf 0-1 (gegen 3000 Referenzwert fuer HD-Foto).
    Typische gute Werte: > 0.3 (alles < 0.1 ist sehr unscharf)
    """
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores.append(lap)
    raw_mean = float(np.mean(scores))
    # Normierung: 3000 = sehr scharfes Referenzbild
    normalized = min(raw_mean / 3000.0, 1.0)
    return normalized, raw_mean, float(np.std(scores))


# ── Metrik 3: Flicker Score ──────────────────────────────────────────────────

def flicker_score(frames):
    """
    Mittlere absolute Pixelaenderung zwischen aufeinanderfolgenden Frames.
    Normiert 0-1 (niedrig = gut, wenig Flackern).
    0.00-0.02 = sehr stabil, 0.02-0.06 = leichtes Flackern, >0.06 = starkes Flackern
    """
    diffs = []
    for i in range(len(frames) - 1):
        a = frames[i].astype(np.float32) / 255.0
        b = frames[i + 1].astype(np.float32) / 255.0
        diff = np.mean(np.abs(a - b))
        diffs.append(diff)
    return float(np.mean(diffs)), float(np.max(diffs)), float(np.std(diffs))


# ── Metrik 4: Motion Score (Optischer Fluss) ────────────────────────────────

def _optical_flow_magnitudes(frames):
    """Optischer Fluss für alle Frame-Paare — intern wiederverwendet."""
    magnitudes = []
    for i in range(len(frames) - 1):
        a = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(mag)))
    return magnitudes


def motion_score(frames):
    """
    Farneback Optischer Fluss — mittlere Bewegungsmagnitude pro Frame-Paar.
    0.0 = komplett eingefroren
    0.5-2.0 = dezente natuerliche Bewegung (ideal)
    >5.0 = starke / unruhige Bewegung
    """
    magnitudes = _optical_flow_magnitudes(frames)
    return float(np.mean(magnitudes)), float(np.max(magnitudes))


def motion_smoothness(frames):
    """
    Gleichmaessigkeit der Bewegung via Variationskoeffizient (CV) des optischen Flusses.
    CV = std / mean — misst wie konsistent die Bewegungsstaerke ueber alle Frames ist.
    Niedrig = fluessige, gleichmaessige Bewegung (ideal fuer Loops).
    Hoch = ruckartige oder chaotische Bewegung.
    Rueckgabe: smoothness (0-1, hoeher = besser), cv (roher Koeffizient)
    """
    magnitudes = _optical_flow_magnitudes(frames)
    mean = float(np.mean(magnitudes))
    if mean < 0.01:
        return 1.0, 0.0  # statisch = per Definition glatt
    cv = float(np.std(magnitudes) / mean)
    smoothness = max(0.0, 1.0 - min(cv, 2.0) / 2.0)  # CV > 2.0 → 0.0
    return smoothness, round(cv, 4)


# ── Metrik 5: Color Consistency ─────────────────────────────────────────────

def color_consistency(frames):
    """
    Standardabweichung der mittleren Kanalfarben ueber alle Frames.
    Niedrig = Farbe stabil (gut), hoch = Farbdrift oder Flackern.
    """
    means = []
    for frame in frames:
        means.append([
            np.mean(frame[:, :, 0]),  # R
            np.mean(frame[:, :, 1]),  # G
            np.mean(frame[:, :, 2]),  # B
        ])
    means = np.array(means)
    std_r = float(np.std(means[:, 0]))
    std_g = float(np.std(means[:, 1]))
    std_b = float(np.std(means[:, 2]))
    avg_std = (std_r + std_g + std_b) / 3.0
    # Normiert: 0 = perfekt stabil, 1 = stark wechselnd (255 Einheiten)
    consistency = max(0.0, 1.0 - avg_std / 30.0)
    return consistency, std_r, std_g, std_b


# ── Metrik 6: Brightness Consistency ────────────────────────────────────────

def brightness_consistency(frames):
    """
    Schwankung der mittleren Helligkeit ueber alle Frames.
    Konsistenz nahe 1.0 = gleichmaessige Belichtung (kein Blinken).
    """
    brightness = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness.append(float(np.mean(gray)))
    std = float(np.std(brightness))
    mean = float(np.mean(brightness))
    consistency = max(0.0, 1.0 - std / 30.0)
    return consistency, mean, std


# ── Gesamt-Score ────────────────────────────────────────────────────────────

def compute_overall(metrics):
    """
    Gewichteter Gesamtscore aus allen Metriken (0.0 - 1.0).
    Hoeher = besser.
    """
    ssim_w       = 0.35  # Temporale Stabilitaet — wichtigste Metrik
    sharpness_w  = 0.20  # Schaerfe
    flicker_inv_w= 0.20  # Flicker (invertiert: niedrig = gut)
    motion_w     = 0.10  # Moderate Bewegung bevorzugt
    color_w      = 0.10  # Farbstabilitaet
    bright_w     = 0.05  # Helligkeitsstabilitaet

    ssim_score = metrics["temporal_ssim"]["mean"]
    sharp_score = min(metrics["sharpness"]["normalized"], 1.0)
    flicker_inv = max(0.0, 1.0 - metrics["flicker"]["mean"] / 0.1)
    motion = metrics["motion"]["mean"]
    motion_score = 1.0 - min(abs(motion - 1.5) / 5.0, 1.0)  # optimal ~1.5 px/frame
    color_score = metrics["color_consistency"]["consistency"]
    bright_score = metrics["brightness_consistency"]["consistency"]

    overall = (
        ssim_score       * ssim_w +
        sharp_score      * sharpness_w +
        flicker_inv      * flicker_inv_w +
        motion_score     * motion_w +
        color_score      * color_w +
        bright_score     * bright_w
    )
    return round(float(overall), 4)


# ── Analyse ─────────────────────────────────────────────────────────────────

def analyze_video(video_path: Path, verbose=True):
    frames, fps, (w, h), total = extract_frames(video_path)

    if len(frames) < 2:
        print(red("Zu wenige Frames fuer Analyse"))
        return None

    if verbose:
        print(dim(f"  {len(frames)} Frames, {w}×{h}, {fps:.1f}fps"))

    ssim_mean, ssim_min, ssim_std      = temporal_ssim(frames)
    sharp_norm, sharp_raw, sharp_std   = sharpness_score(frames)
    flick_mean, flick_max, flick_std   = flicker_score(frames)
    mot_mean, mot_max                  = motion_score(frames)
    mot_smooth, mot_cv                 = motion_smoothness(frames)
    col_cons, col_r, col_g, col_b      = color_consistency(frames)
    bri_cons, bri_mean, bri_std        = brightness_consistency(frames)

    metrics = {
        "file": str(video_path),
        "resolution": [w, h],
        "fps": fps,
        "frames": len(frames),
        "temporal_ssim": {
            "mean": round(ssim_mean, 4),
            "min": round(ssim_min, 4),
            "std": round(ssim_std, 4),
            "description": "Stabilitaet zwischen Frames (1.0 = perfekt, >0.85 = gut)"
        },
        "sharpness": {
            "normalized": round(sharp_norm, 4),
            "raw_laplacian": round(sharp_raw, 2),
            "std": round(sharp_std, 2),
            "description": "Schaerfe 0-1 (>0.3 = gut, <0.1 = unscharf)"
        },
        "flicker": {
            "mean": round(flick_mean, 5),
            "max": round(flick_max, 5),
            "std": round(flick_std, 5),
            "description": "Pixelaenderung zwischen Frames (niedrig = stabil, <0.02 = gut)"
        },
        "motion": {
            "mean": round(mot_mean, 3),
            "max": round(mot_max, 3),
            "smoothness": round(mot_smooth, 4),
            "cv": round(mot_cv, 4),
            "description": "Opt. Fluss px/Frame (0=eingefroren, 0.5-2.0=dezent, >5=unruhig)"
        },
        "color_consistency": {
            "consistency": round(col_cons, 4),
            "std_r": round(col_r, 2),
            "std_g": round(col_g, 2),
            "std_b": round(col_b, 2),
            "description": "Farbstabilitaet 0-1 (>0.8 = gut)"
        },
        "brightness_consistency": {
            "consistency": round(bri_cons, 4),
            "mean_brightness": round(bri_mean, 1),
            "std_brightness": round(bri_std, 2),
            "description": "Helligkeitsstabilitaet 0-1 (>0.8 = gut)"
        },
    }
    metrics["overall_score"] = compute_overall(metrics)
    return metrics


def print_report(metrics, label=None):
    label = label or Path(metrics["file"]).name
    print(f"\n  {bold(label)}")
    print(f"  {metrics['frames']} Frames | {metrics['resolution'][0]}×{metrics['resolution'][1]} | {metrics['fps']:.1f}fps")
    print()

    m = metrics
    print(f"  {'Metrik':<28} {'Wert':>8}   {'Bewertung'}")
    print(f"  {'-'*60}")
    print(f"  {'Temporal SSIM (Stabilitaet)':<28} {score_color(m['temporal_ssim']['mean']):>8}   min={m['temporal_ssim']['min']:.4f}")
    print(f"  {'Schaerfe (normiert)':<28} {score_color(m['sharpness']['normalized']):>8}   raw Laplacian={m['sharpness']['raw_laplacian']:.0f}")
    print(f"  {'Flicker (niedrig=gut)':<28} {score_color(m['flicker']['mean'], good_high=False):>8}   max={m['flicker']['max']:.4f}")
    print(f"  {'Motion px/Frame':<28} {'':>8}   {m['motion']['mean']:.3f}  (optimal: 0.5-2.0)")
    print(f"  {'Farbstabilitaet':<28} {score_color(m['color_consistency']['consistency']):>8}   std R/G/B={m['color_consistency']['std_r']:.1f}/{m['color_consistency']['std_g']:.1f}/{m['color_consistency']['std_b']:.1f}")
    print(f"  {'Helligkeitsstabilitaet':<28} {score_color(m['brightness_consistency']['consistency']):>8}   std={m['brightness_consistency']['std_brightness']:.1f}")
    print(f"  {'-'*60}")

    overall = m['overall_score']
    col = green if overall >= 0.75 else (yellow if overall >= 0.55 else red)
    print(f"  {'GESAMT-SCORE':<28} {col(f'{overall:.4f}'):>8}")
    print()


# ── Vergleichs-Modus ─────────────────────────────────────────────────────────

def find_mp4(path: Path):
    """Findet erstes MP4 in Verzeichnis oder gibt Pfad direkt zurueck."""
    p = Path(path)
    if p.is_file():
        return p
    mp4s = sorted(p.glob("*.mp4"))
    if mp4s:
        return mp4s[0]
    return None


def compare_mode(paths):
    results = []
    for p in paths:
        video = find_mp4(Path(p))
        if not video:
            print(yellow(f"  Kein MP4 in: {p}"))
            continue
        print(cyan(f"\n[Analysiere] {video}"))
        m = analyze_video(video)
        if m:
            m["label"] = Path(p).name
            results.append(m)

    if not results:
        return

    # Sortiert nach Gesamt-Score
    results.sort(key=lambda x: x["overall_score"], reverse=True)

    print(f"\n{'='*70}")
    print(bold("  VERGLEICH — sortiert nach Gesamt-Score"))
    print(f"{'='*70}")
    print(f"  {'Test':<20} {'SSIM':>8} {'Scharf':>8} {'Flicker':>8} {'Motion':>8} {'GESAMT':>8}")
    print(f"  {'-'*66}")
    for m in results:
        overall = m["overall_score"]
        col = green if overall >= 0.75 else (yellow if overall >= 0.55 else red)
        print(
            f"  {m['label']:<20}"
            f"  {m['temporal_ssim']['mean']:>6.4f}"
            f"  {m['sharpness']['normalized']:>6.4f}"
            f"  {m['flicker']['mean']:>7.4f}"
            f"  {m['motion']['mean']:>6.2f}"
            f"  {col(f'{overall:.4f}'):>8}"
        )
    print(f"\n  Bester: {bold(results[0]['label'])}  (Score: {results[0]['overall_score']:.4f})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Video-Qualitaetsmetriken")
    parser.add_argument("input", nargs="?", help="MP4-Datei oder Verzeichnis")
    parser.add_argument("--save-report", action="store_true",
                        help="Ergebnis als JSON neben dem Video speichern")
    parser.add_argument("--update-results", type=str, default=None,
                        help="results.json Pfad — Metriken automatisch eintragen")
    parser.add_argument("--test-id", type=str, default=None,
                        help="Test-ID fuer --update-results (z.B. B, C)")
    parser.add_argument("--compare", nargs="+", metavar="PATH",
                        help="Mehrere Videos/Verzeichnisse vergleichen")
    args = parser.parse_args()

    if args.compare:
        compare_mode(args.compare)
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    video = find_mp4(Path(args.input))
    if not video:
        print(red(f"Keine MP4 gefunden: {args.input}"))
        sys.exit(1)

    print(cyan(f"\n  Analysiere: {video.name}"))
    metrics = analyze_video(video)
    if not metrics:
        sys.exit(1)

    print_report(metrics, label=video.name)

    if args.save_report:
        report_path = video.with_suffix(".metrics.json")
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(dim(f"  Report gespeichert: {report_path}"))

    if args.update_results and args.test_id:
        rpath = Path(args.update_results)
        if rpath.exists():
            with open(rpath) as f:
                results = json.load(f)
            for test in results["tests"]:
                if test["test_id"] == args.test_id:
                    test["status"] = "done"
                    test["scores"]["auto_ssim"]    = metrics["temporal_ssim"]["mean"]
                    test["scores"]["auto_sharpness"] = metrics["sharpness"]["normalized"]
                    test["scores"]["auto_flicker"]  = metrics["flicker"]["mean"]
                    test["scores"]["auto_motion"]   = metrics["motion"]["mean"]
                    test["scores"]["auto_color"]    = metrics["color_consistency"]["consistency"]
                    test["scores"]["auto_overall"]  = metrics["overall_score"]
                    break
            with open(rpath, "w") as f:
                json.dump(results, f, indent=2)
            print(dim(f"  results.json aktualisiert (Test {args.test_id})"))


if __name__ == "__main__":
    main()
