"""
Lo-Fi Feedback Tool — Terminal-Version.
Zeigt neue Videos/GIFs automatisch an, nimmt Bewertung und Korrekturen entgegen.

Starten:
  python scripts/feedback.py                    # wartet auf neue Dateien
  python scripts/feedback.py --latest           # bewertet neuestes Sample sofort
  python scripts/feedback.py --round 7          # alle Samples von Round 7
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PIPELINE_ROOT = Path(__file__).parent.parent
SCENARIOS_DIR = PIPELINE_ROOT / "scenarios"
CONFIGS_DIR = PIPELINE_ROOT / "configs"

CORRECTIONS = {
    "1":  ("Ohren fehlen / nicht sichtbar",
           "both rabbits with long clearly visible upright ears, prominent tall ears",
           "missing ears, no ears, hidden ears, floppy ears", False),
    "2":  ("Drei Hasen statt zwei",
           "exactly two rabbits, only two rabbits, a pair of two rabbits",
           "three rabbits, four rabbits, trio, extra rabbit, third rabbit", False),
    "3":  ("Falsches Tier (Vogel/Küken)",
           "rabbit with round body and upright ears, clearly a rabbit not a bird",
           "bird, chick, beak, duck, wrong animal", True),
    "4":  ("Mund offen / überrascht",
           "calm closed mouths, serene peaceful expression",
           "open mouth, surprised expression, anxious", False),
    "5":  ("Stil zu 3D / photorealistisch",
           "flat anime illustration, hand-drawn 2D anime art, cel shading",
           "3D render, CGI, photorealistic, volumetric lighting, realistic fur", True),
    "6":  ("Zu wenig Animation",
           "animated loop, visible motion, gently swaying leaves, twinkling stars",
           "completely static, frozen image, no movement", True),
    "7":  ("Hasen zu weit auseinander",
           "two rabbits sitting very close together, shoulders touching, cuddling pose",
           "rabbits far apart", False),
    "8":  ("See fehlt",
           "calm dark lake clearly visible in middle ground with reflections",
           "no water, no lake", False),
    "9":  ("Wasser-Spiegelungen fehlen",
           "softly rippling lake water with shimmering cottage and moon reflections",
           "flat water, no reflections", False),
    "10": ("Szene zu hell",
           "deep dark indigo night sky, dark nocturnal atmosphere",
           "bright daylight, overexposed, washed out", False),
    "11": ("Gesichter nicht erkennbar",
           "both rabbits facing forward, clearly visible faces, visible round eyes",
           "no faces, empty faces, broken faces", True),
    "12": ("Ohren bewegen sich nicht",
           "rabbit ears gently swaying in soft breeze, ear tips moving like leaves",
           None, True),
    "13": ("Baum fehlt",
           "large framing tree on right side with gently swaying leaves",
           None, False),
    "14": ("Laterne fehlt",
           "warm glowing amber lantern between the two rabbits",
           "no lantern", False),
}



def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def clr(code, text):
    return f"\033[{code}m{text}\033[0m"


def yellow(t): return clr("33", t)
def green(t):  return clr("32", t)
def cyan(t):   return clr("36", t)
def dim(t):    return clr("2", t)
def bold(t):   return clr("1", t)


def open_viewer(path: Path):
    """Öffnet Video als lokale HTML-Datei im Browser — kein Server nötig."""
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{path.name}</title>
  <style>
    body {{ background: #0f0f1a; margin: 0; display: flex; flex-direction: column;
            align-items: center; justify-content: center; min-height: 100vh;
            font-family: sans-serif; color: #c0c0d0; }}
    video {{ max-width: 95vw; max-height: 85vh; border-radius: 10px;
             box-shadow: 0 0 40px rgba(120,80,255,0.3); }}
    h3 {{ color: #a78bfa; margin-bottom: 12px; font-size: 15px; letter-spacing: 1px; }}
    p {{ color: #555; font-size: 13px; margin-top: 10px; }}
  </style>
</head>
<body>
  <h3>&#128249; {path.name}</h3>
  <video autoplay loop controls src="{path.as_uri()}"></video>
  <p>Bewertung und Korrekturen im Terminal eingeben.</p>
</body>
</html>"""

    tmp = Path("/tmp") / "lofi_preview_current.html"
    tmp.write_text(html, encoding="utf-8")
    try:
        subprocess.Popen(["xdg-open", tmp.as_uri()],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(dim(f"  (Browser öffnet sich mit Video-Vorschau)"))
    except Exception:
        print(dim(f"  Pfad: {path}"))


def ratings_path(scenario_id):
    return SCENARIOS_DIR / scenario_id / "ratings.json"


def load_ratings(scenario_id):
    p = ratings_path(scenario_id)
    return json.loads(p.read_text()) if p.exists() else {}


def save_rating(scenario_id, key, rating, note="", corrections=None):
    data = load_ratings(scenario_id)
    data[key] = {"rating": rating, "note": note,
                 "corrections": corrections or [], "ts": time.time()}
    ratings_path(scenario_id).write_text(json.dumps(data, indent=2))


def next_round_number(scenario_id):
    rounds_dir = SCENARIOS_DIR / scenario_id / "rounds"
    if not rounds_dir.exists():
        return 1
    nums = [int(d.name.split("_")[1]) for d in rounds_dir.iterdir()
            if d.is_dir() and d.name.startswith("round_")]
    return max(nums) + 1 if nums else 1


def find_checkpoint(sample_path: Path):
    """Checkpoint aus Sample-Pfad ableiten."""
    import re
    round_dir = sample_path.parent.parent
    m = re.match(r"step_(\d+)", sample_path.stem)
    if m:
        ckpt = round_dir / "checkpoints" / f"lora_weights_step_{m.group(1)}.safetensors"
        if ckpt.exists():
            return ckpt
    ckpt_dir = round_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("lora_weights_step_*.safetensors"))
        if ckpts:
            return ckpts[-1]
    return None


def apply_corrections(scenario_id, correction_ids):
    """Wendet Korrekturen auf scenario.yaml an."""
    scenario_path = SCENARIOS_DIR / scenario_id / "scenario.yaml"
    cfg = load_yaml(scenario_path)
    prompt = cfg.get("prompt", "").strip().rstrip(",\n")
    negative = cfg.get("negative_prompt", "").strip().rstrip(",\n")
    needs_training = False

    for cid in correction_ids:
        c = CORRECTIONS.get(cid)
        if not c:
            continue
        label, prompt_add, neg_add, train = c
        if prompt_add and prompt_add not in prompt:
            prompt = prompt_add + ",\n  " + prompt
        if neg_add and neg_add not in negative:
            negative = neg_add + ",\n  " + negative
        if train:
            needs_training = True

    cfg["prompt"] = prompt
    cfg["negative_prompt"] = negative

    scenario_path.with_suffix(".yaml.bak").write_text(scenario_path.read_text())
    with open(scenario_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, width=120)

    return needs_training


def start_generate(scenario_id, checkpoint, seed=777, guidance=9.0, steps=60):
    paths_cfg = load_yaml(CONFIGS_DIR / "model_paths.yaml")
    scenario_cfg = load_yaml(SCENARIOS_DIR / scenario_id / "scenario.yaml")

    ckpt_path = Path(checkpoint)
    samples_dir = ckpt_path.parent.parent / "samples"
    samples_dir.mkdir(exist_ok=True)
    out = samples_dir / f"manual_s{seed}_g{guidance}_{int(time.time())}.mp4"

    cmd = [paths_cfg["python"], paths_cfg["generate_script"],
           "--lora", str(checkpoint), "--steps", str(steps),
           "--seed", str(seed), "--guidance-scale", str(guidance),
           "--output", str(out),
           "--prompt", scenario_cfg["prompt"].strip(),
           "--negative-prompt", scenario_cfg["negative_prompt"].strip()]

    env = os.environ.copy()
    env.update(paths_cfg.get("env", {}))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(cyan(f"\n  → Starte Generierung: {out.name}"))
    print(dim("    Läuft im Hintergrund (~10 min) ..."))
    subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def start_train(scenario_id, checkpoint=None, steps=50):
    paths_cfg = load_yaml(CONFIGS_DIR / "model_paths.yaml")
    next_round = next_round_number(scenario_id)

    cmd = [paths_cfg["python"],
           str(PIPELINE_ROOT / "scripts" / "train_lora.py"),
           "--scenario", scenario_id,
           "--round", str(next_round),
           "--steps", str(steps)]
    cmd += ["--from-checkpoint", str(checkpoint)] if checkpoint else ["--resume"]

    env = os.environ.copy()
    env.update(paths_cfg.get("env", {}))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log = PIPELINE_ROOT / "scripts" / f"_train_r{next_round}.log"
    print(cyan(f"\n  → Starte Training Round {next_round} ({steps} Steps)"))
    print(dim(f"    Log: {log}"))
    with open(log, "w") as lf:
        subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)



def review_sample(scenario_id, sample_path: Path):
    print("\n" + "═" * 56)
    print(bold(f"  📹  {sample_path.parent.parent.name} / {sample_path.name}"))
    print("═" * 56)

    open_viewer(sample_path)
    print(dim("  (Video/GIF öffnet sich automatisch im Bildbetrachter)"))

    while True:
        try:
            r = input(yellow("\nBewertung (1–10, Enter=überspringen): ")).strip()
            if r == "":
                print(dim("  Übersprungen."))
                return
            rating = int(r)
            if 1 <= rating <= 10:
                break
            print("  Bitte 1–10 eingeben.")
        except (ValueError, KeyboardInterrupt):
            print(dim("\n  Abgebrochen."))
            return

    print(yellow("\nProbleme auswählen (Nummern mit Komma, Enter=keine):"))
    for num, (label, _, _, needs_train) in CORRECTIONS.items():
        marker = yellow("⚡") if needs_train else " "
        print(f"  {dim(num.rjust(2))}  {marker} {label}")
    print(dim("   (⚡ = braucht Trainingsrunde)"))

    corr_input = input(yellow("Eingabe: ")).strip()
    correction_ids = [c.strip() for c in corr_input.split(",") if c.strip()] if corr_input else []
    valid_ids = [c for c in correction_ids if c in CORRECTIONS]

    note = input(yellow("Eigene Anmerkung (Enter=leer): ")).strip()

    key = f"{sample_path.parent.parent.name}/{sample_path.name}"
    save_rating(scenario_id, key, rating, note, valid_ids)
    print(green(f"\n  ✓ Bewertung {rating}/10 gespeichert"))

    if not valid_ids:
        print(dim("  Keine Korrekturen ausgewählt."))
        return

    print(cyan("\n  Korrekturen:"))
    for cid in valid_ids:
        print(f"    • {CORRECTIONS[cid][0]}")

    needs_training = apply_corrections(scenario_id, valid_ids)
    print(green("  ✓ Prompt in scenario.yaml aktualisiert"))

    checkpoint = find_checkpoint(sample_path)
    if needs_training:
        print(yellow("\n  ⚡ Einige Korrekturen brauchen eine Trainingsrunde."))

    print(yellow("\nWas soll als nächstes passieren?"))
    print("  [G] Neu generieren  (~10 min, kein Training)")
    if checkpoint:
        print("  [T] Training starten  (~40 min)")
    print("  [S] Nichts / Später")

    while True:
        action = input(yellow("Aktion: ")).strip().upper()
        if action in ("G", "T", "S", ""):
            break

    if action == "G" and checkpoint:
        seed_in = input(dim(f"  Seed [777]: ")).strip()
        seed = int(seed_in) if seed_in else 777
        start_generate(scenario_id, checkpoint, seed=seed)
    elif action == "T" and checkpoint:
        steps_in = input(dim(f"  Steps [50]: ")).strip()
        steps = int(steps_in) if steps_in else 50
        start_train(scenario_id, checkpoint, steps=steps)
    else:
        print(dim("  OK, nichts gestartet."))



def get_samples(scenario_id, round_filter=None):
    """Alle Samples eines Szenarios."""
    rounds_dir = SCENARIOS_DIR / scenario_id / "rounds"
    files = []
    for rd in sorted(rounds_dir.iterdir()):
        if not rd.is_dir():
            continue
        if round_filter and rd.name != f"round_{round_filter:02d}":
            continue
        sd = rd / "samples"
        if sd.exists():
            for f in sorted(sd.iterdir()):
                if f.suffix in (".mp4", ".gif"):
                    files.append(f)
    return files


def watch_mode(scenario_id):
    """Wartet auf neue Dateien und zeigt sie zur Bewertung."""
    print(bold(f"\n  🐰 Lo-Fi Feedback — Watch Mode"))
    print(dim(f"  Szenario: {scenario_id}"))
    print(dim("  Warte auf neue Videos/GIFs ... (Strg+C zum Beenden)\n"))

    seen = set(str(f) for f in get_samples(scenario_id))
    ratings = load_ratings(scenario_id)

    while True:
        try:
            current = set(str(f) for f in get_samples(scenario_id))
            new_files = current - seen
            for fp in sorted(new_files):
                seen.add(fp)
                key = f"{Path(fp).parent.parent.name}/{Path(fp).name}"
                if key not in ratings:
                    review_sample(scenario_id, Path(fp))
            time.sleep(5)
        except KeyboardInterrupt:
            print(dim("\n\nBeendet."))
            break


def latest_mode(scenario_id):
    """Neuestes Sample sofort bewerten."""
    files = get_samples(scenario_id)
    if not files:
        print("Keine Samples gefunden.")
        return
    newest = max(files, key=lambda f: f.stat().st_mtime)
    review_sample(scenario_id, newest)


def round_mode(scenario_id, round_num):
    """Alle Samples eines Rounds bewerten."""
    files = get_samples(scenario_id, round_num)
    if not files:
        print(f"Keine Samples in Round {round_num}.")
        return
    ratings = load_ratings(scenario_id)
    for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        key = f"{f.parent.parent.name}/{f.name}"
        if key in ratings:
            existing = ratings[key]["rating"]
            print(dim(f"\n  {f.name} — bereits bewertet ({existing}/10), überspringen? [Enter=ja]"))
            if input().strip() == "":
                continue
        review_sample(scenario_id, f)



def main():
    parser = argparse.ArgumentParser(description="Lo-Fi Feedback Tool")
    parser.add_argument("--scenario", default="rabbit_lake", help="Szenario ID")
    parser.add_argument("--latest", action="store_true", help="Neuestes Sample bewerten")
    parser.add_argument("--round", type=int, help="Alle Samples eines Rounds bewerten")
    args = parser.parse_args()

    if args.latest:
        latest_mode(args.scenario)
    elif args.round:
        round_mode(args.scenario, args.round)
    else:
        watch_mode(args.scenario)


if __name__ == "__main__":
    main()
