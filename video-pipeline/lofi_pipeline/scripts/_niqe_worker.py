#!/usr/bin/env python3
"""NIQE-Worker — laeuft im ISOLIERTEN venv `Bachelorarbeit/.venv-niqe`.

Bewertet ein Video/GIF mit NIQE (Naturalness Image Quality Evaluator, Mittal
et al. 2013): voll-blind, kein Training auf menschlichen Meinungswerten.
Niedriger Wert = natuerlicher / bessere Qualitaet. Ausgabe als JSON auf stdout.

Wird von evaluate_video.py per Subprozess aufgerufen — NICHT aus der Haupt-venv,
weil pyiqa `transformers>=5` verlangt und das mit der LTX-Pipeline (diffusers
0.33) der Haupt-venv kollidiert. Das isolierte venv erbt torch/cv2 der Haupt-venv
per `_parent_venv.pth`, bringt pyiqa + transformers5 nur lokal mit.

    <.venv-niqe>/bin/python _niqe_worker.py VIDEO [--max-frames N] [--sample K]
"""
import argparse
import json

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=61,
                    help="hoechstens so viele Frames einlesen")
    ap.add_argument("--sample", type=int, default=12,
                    help="so viele gleichmaessig verteilte Frames bewerten")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(json.dumps({"available": False, "reason": "cv2 kann Datei nicht oeffnen"}))
        return
    frames = []
    while len(frames) < args.max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        print(json.dumps({"available": False, "reason": "keine Frames dekodierbar"}))
        return

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    import pyiqa

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    metric = pyiqa.create_metric("niqe", device=dev)

    idx = np.unique(np.linspace(0, len(frames) - 1,
                                min(args.sample, len(frames))).astype(int))
    vals = []
    for i in idx:
        t = (torch.from_numpy(frames[int(i)]).permute(2, 0, 1)
             .unsqueeze(0).float().div(255.0).to(dev))
        with torch.no_grad():
            vals.append(float(metric(t)))
    vals = np.asarray(vals, dtype=float)

    print(json.dumps({
        "available": True,
        "metric": "niqe",
        "backend": "pyiqa",
        "lower_is_better": True,
        "mean": round(float(vals.mean()), 3),
        "std": round(float(vals.std()), 3),
        "min": round(float(vals.min()), 3),
        "max": round(float(vals.max()), 3),
        "frames_scored": int(len(vals)),
    }))


if __name__ == "__main__":
    main()
