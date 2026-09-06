#!/usr/bin/env python3
"""Nachtraegliche, automatische Qualitaets-/Schaerfebewertung ALLER vorhandenen
Videos und GIFs der LTX-Video-Pipeline.

Fuer jedes gefundene MP4/GIF wird (falls noch nicht vorhanden) per
evaluate_video.py eine <name>.metrics.json geschrieben. Am Ende entsteht eine
Sammel-CSV mit allen Kennzahlen, sortiert nach Schaerfe-Score.

Das ist der Batch-Gegenpart zum Auto-Hook in generate.py (der ab jetzt jedes
NEU generierte Video automatisch bewertet).

Aufruf:
  python backfill_metrics.py                        # alle Szenarien, MP4 + GIF
  python backfill_metrics.py --scenarios rainy_window rabbit_lake
  python backfill_metrics.py --force                # auch schon bewertete neu rechnen
  python backfill_metrics.py --no-gif --csv /pfad/report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

PIPELINE_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PIPELINE_ROOT / "configs"
SCENARIOS_DIR = PIPELINE_ROOT / "scenarios"
EVALUATE_SCRIPT = PIPELINE_ROOT.parent / "pipeline" / "realistic_rabbit" / "evaluate_video.py"

# Nur generierte Ausgaben — Rohmaterial / Trainingsclips ausklammern.
INCLUDE_DIRS = ("rounds", "web_outputs")
SKIP_SUFFIXES = ("_raw.mp4",)


def load_python_bin() -> str:
    cfg = yaml.safe_load((CONFIGS_DIR / "model_paths.yaml").read_text())
    return cfg["python"]


def find_media(scenarios, include_gif: bool) -> list[Path]:
    exts = ["*.mp4"] + (["*.gif"] if include_gif else [])
    out: list[Path] = []
    for s in scenarios:
        sdir = SCENARIOS_DIR / s
        if not sdir.is_dir():
            print(f"  ! Szenario fehlt: {s}", file=sys.stderr)
            continue
        for sub in INCLUDE_DIRS:
            base = sdir / sub
            if not base.is_dir():
                continue
            for ext in exts:
                for f in base.rglob(ext):
                    if f.name.endswith(SKIP_SUFFIXES):
                        continue
                    out.append(f)
    return sorted(set(out))


def all_scenarios() -> list[str]:
    return sorted(p.name for p in SCENARIOS_DIR.iterdir()
                  if p.is_dir() and not p.name.startswith("_"))


def score_one(media: Path, python_bin: str, max_frames: int, force: bool) -> dict | None:
    mj = media.with_suffix(".metrics.json")
    if mj.exists() and not force:
        try:
            return json.loads(mj.read_text())
        except Exception:  # noqa: BLE001
            pass  # kaputt -> neu rechnen
    try:
        subprocess.run(
            [python_bin, str(EVALUATE_SCRIPT), str(media),
             "--save-report", "--max-frames", str(max_frames)],
            capture_output=True, text=True, timeout=600,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  ! Fehler {media.name}: {e}", file=sys.stderr)
        return None
    if mj.exists():
        try:
            return json.loads(mj.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  ! JSON kaputt {mj.name}: {e}", file=sys.stderr)
    return None


def row_from_metrics(media: Path, m: dict) -> dict:
    try:
        rel = media.relative_to(SCENARIOS_DIR)
        scenario = rel.parts[0]
    except ValueError:
        scenario = ""
    rnd = next((p for p in media.parts if p.startswith("round_")), "")
    sh = m.get("sharpness", {})
    nq = m.get("niqe", {})
    return {
        "scenario": scenario,
        "round": rnd,
        "kind": media.suffix.lstrip("."),
        "file": str(media.relative_to(PIPELINE_ROOT.parent)) if str(media).startswith(str(PIPELINE_ROOT.parent)) else str(media),
        "name": media.name,
        "resolution": "x".join(map(str, m.get("resolution", []))),
        "frames": m.get("frames", ""),
        "schaerfe_score_100": sh.get("score_100", ""),
        "laplacian": sh.get("raw_laplacian", ""),
        "tenengrad": sh.get("tenengrad", ""),
        "fft_high_ratio": sh.get("fft_high_ratio", ""),
        "konstanz": sh.get("konstanz", ""),
        "sharpness_normalized": sh.get("normalized", ""),
        "niqe": nq.get("mean", "") if nq.get("available") else "",
        "niqe_std": nq.get("std", "") if nq.get("available") else "",
        "temporal_ssim": m.get("temporal_ssim", {}).get("mean", ""),
        "flicker": m.get("flicker", {}).get("mean", ""),
        "motion": m.get("motion", {}).get("mean", ""),
        "overall_score": m.get("overall_score", ""),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=None, help="Szenario-IDs (default: alle)")
    ap.add_argument("--no-gif", action="store_true", help="nur MP4, keine GIFs")
    ap.add_argument("--force", action="store_true", help="auch vorhandene .metrics.json neu berechnen")
    ap.add_argument("--max-frames", type=int, default=61, help="Frame-Cap pro Video (default 61)")
    ap.add_argument("--csv", type=str, default=None, help="Ziel-CSV (default: scenarios/_schaerfe_report.csv)")
    args = ap.parse_args(argv)

    if not EVALUATE_SCRIPT.exists():
        print(f"evaluate_video.py nicht gefunden: {EVALUATE_SCRIPT}", file=sys.stderr)
        return 1

    scenarios = args.scenarios or all_scenarios()
    python_bin = load_python_bin()
    media = find_media(scenarios, include_gif=not args.no_gif)
    if not media:
        print("Nichts gefunden.")
        return 1

    csv_path = Path(args.csv) if args.csv else (SCENARIOS_DIR / "_schaerfe_report.csv")

    print(f"  {len(media)} Dateien in {len(scenarios)} Szenarien  (force={args.force})\n")
    rows: list[dict] = []
    t0 = time.time()
    for i, f in enumerate(media, 1):
        m = score_one(f, python_bin, args.max_frames, args.force)
        tag = "neu" if not f.with_suffix(".metrics.json").exists() else "ok"
        if m is None:
            print(f"  [{i:>3}/{len(media)}]  FEHLER   {f.name}")
            continue
        r = row_from_metrics(f, m)
        rows.append(r)
        print(f"  [{i:>3}/{len(media)}]  {r['schaerfe_score_100']:>5}  {r['scenario']}/{r['round']}/{f.name}")

    if not rows:
        return 1

    rows.sort(key=lambda x: (x["schaerfe_score_100"] if isinstance(x["schaerfe_score_100"], (int, float)) else -1), reverse=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # kurze Zusammenfassung pro Szenario
    print(f"\n  {'Szenario':<22} {'n':>4} {'Score Ø':>8} {'min':>6} {'max':>6}")
    print("  " + "-" * 50)
    by: dict[str, list[float]] = {}
    for r in rows:
        v = r["schaerfe_score_100"]
        if isinstance(v, (int, float)):
            by.setdefault(r["scenario"], []).append(v)
    for s in sorted(by):
        vs = by[s]
        print(f"  {s:<22} {len(vs):>4} {sum(vs)/len(vs):>8.1f} {min(vs):>6.1f} {max(vs):>6.1f}")

    print(f"\n  {len(rows)} bewertet in {time.time()-t0:.0f}s")
    print(f"  CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
