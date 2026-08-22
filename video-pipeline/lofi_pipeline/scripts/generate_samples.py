"""
Generate video samples (MP4-first strategy) for a training round.

Primary output is MP4. GIFs are optional and generated on request only.
This avoids wasting time on GIF quality (256-color palette) during training
evaluation — assess from MP4, convert best results to GIF later.

Usage:
  python generate_samples.py --scenario rainy_window --round 4
  python generate_samples.py --scenario rainy_window --round 4 --checkpoint 75
  python generate_samples.py --scenario rainy_window --round 4 --seed 1337
  python generate_samples.py --scenario rainy_window --round 4 --gif          # also make GIFs
  python generate_samples.py --scenario rainy_window --round 4 --gif-only     # GIFs from existing MP4s only
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
CONFIGS_DIR = PIPELINE_ROOT / "configs"
SCENARIOS_DIR = PIPELINE_ROOT / "scenarios"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_gif(mp4_path, gif_path, fps=8, scale=624):
    """Convert MP4 to GIF using high-quality two-pass palette method."""
    cmd = [
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-vf", (
            f"fps={fps},scale={scale}:-1:flags=lanczos,"
            f"split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];"
            f"[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
        ),
        "-loop", "0", str(gif_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


MAKE_GIF_SCRIPT = PIPELINE_ROOT / "scripts" / "make_gif.py"


def run_make_gif_pipeline(mp4_path: Path, paths_cfg: dict):
    """Full GIF+MP4 pipeline via make_gif.py (minterpolate + Real-ESRGAN 4x)."""
    env = os.environ.copy()
    env.update(paths_cfg.get("env", {}))
    cmd = [paths_cfg["python"], str(MAKE_GIF_SCRIPT), str(mp4_path)]
    print(f"  Post-processing: GIF + upscaled MP4...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  GIF/MP4 pipeline FAILED:\n{result.stderr[-300:]}")
    else:
        gif = mp4_path.with_suffix(".gif")
        upscaled = mp4_path.with_name(mp4_path.stem + "_upscaled.mp4")
        if gif.exists():
            print(f"  GIF:  {gif.name}")
        if upscaled.exists():
            print(f"  MP4 (upscaled): {upscaled.name}")


def generate_for_checkpoint(scenario_cfg, paths_cfg, ckpt_path, output_mp4, seed, make_gif_flag, gpu_fraction=1.0):
    env = os.environ.copy()
    env.update(paths_cfg.get("env", {}))

    guidance_scale = scenario_cfg.get("guidance_scale", 4.5)
    cmd = [
        paths_cfg["python"], paths_cfg["generate_script"],
        "--lora", str(ckpt_path),
        "--steps", str(scenario_cfg["generate_inference_steps"]),
        "--seed", str(seed),
        "--output", str(output_mp4),
        "--guidance-scale", str(guidance_scale),
        "--prompt", scenario_cfg["prompt"].strip(),
        "--negative-prompt", scenario_cfg["negative_prompt"].strip(),
    ]
    if gpu_fraction < 1.0:
        cmd += ["--gpu-fraction", str(gpu_fraction)]

    print(f"  Generating {output_mp4.name} (seed {seed})...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return False

    print(f"  MP4:  {output_mp4.name}")

    if output_mp4.exists():
        run_make_gif_pipeline(output_mp4, paths_cfg)

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate Lo-Fi samples (MP4-first)")
    parser.add_argument("--scenario", required=True, help="Scenario ID")
    parser.add_argument("--round", type=int, required=True, help="Round number")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Specific step to generate (default: all checkpoints in round)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed (default: from scenario.yaml)")
    parser.add_argument("--gif", action="store_true",
                        help="Also generate GIF from each MP4 (optional, secondary format)")
    parser.add_argument("--gif-only", action="store_true",
                        help="Only create GIFs from existing MP4s — no re-generation")
    parser.add_argument("--gpu-fraction", type=float, default=1.0,
                        help="GPU-Auslastung begrenzen, z.B. 0.8 für 80%% (wird an generate.py weitergegeben)")
    args = parser.parse_args()

    scenario_dir = SCENARIOS_DIR / args.scenario
    scenario_cfg = load_yaml(scenario_dir / "scenario.yaml")
    paths_cfg = load_yaml(CONFIGS_DIR / "model_paths.yaml")

    round_dir = scenario_dir / "rounds" / f"round_{args.round:02d}"
    ckpt_dir = round_dir / "checkpoints"
    samples_dir = round_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else scenario_cfg["seed"]

    if args.gif_only:
        mp4s = sorted(p for p in samples_dir.glob("step_*.mp4") if "upscaled" not in p.stem)
        if not mp4s:
            print(f"ERROR: No MP4s found in {samples_dir}")
            sys.exit(1)
        print(f"\n{'='*60}")
        print(f"  GIF-only mode — full pipeline on {len(mp4s)} MP4(s)")
        print(f"{'='*60}\n")
        for mp4 in mp4s:
            run_make_gif_pipeline(mp4, paths_cfg)
        return

    if not ckpt_dir.exists():
        print(f"ERROR: No checkpoints found at {ckpt_dir}")
        sys.exit(1)

    if args.checkpoint:
        ckpts = [ckpt_dir / f"lora_weights_step_{args.checkpoint:05d}.safetensors"]
        if not ckpts[0].exists():
            print(f"ERROR: Checkpoint step {args.checkpoint} not found")
            sys.exit(1)
    else:
        ckpts = sorted(ckpt_dir.glob("lora_weights_step_*.safetensors"))

    if not ckpts:
        print("ERROR: No checkpoints found")
        sys.exit(1)

    gif_note = " + GIF" if args.gif else " (MP4 only — use --gif to also create GIFs)"
    print(f"\n{'='*60}")
    print(f"  Generating samples{gif_note}")
    print(f"  Scenario: {args.scenario} | Round: {args.round} | Seed: {seed}")
    print(f"  Checkpoints: {len(ckpts)} | Inference steps: {scenario_cfg['generate_inference_steps']}")
    print(f"{'='*60}\n")

    generated_mp4s = []
    for i, ckpt in enumerate(ckpts):
        step_num = int(ckpt.stem.split("_")[-1])
        mp4_out = samples_dir / f"step_{step_num:05d}_0.mp4"
        if i > 0:
            time.sleep(8)
        ok = generate_for_checkpoint(scenario_cfg, paths_cfg, ckpt, mp4_out, seed, args.gif, gpu_fraction=args.gpu_fraction)
        if ok:
            generated_mp4s.append(str(mp4_out))

    notes_path = round_dir / "notes.json"
    if notes_path.exists():
        with open(notes_path) as f:
            notes = json.load(f)
        notes["samples_generated"] = generated_mp4s
        with open(notes_path, "w") as f:
            json.dump(notes, f, indent=2)

    print(f"\nDone. {len(generated_mp4s)} MP4(s) saved to {samples_dir}")
    for s in generated_mp4s:
        print(f"  {Path(s).name}")
    print(f"\n  Tip: run with --gif-only to re-run GIF+MP4 pipeline on existing samples.")


if __name__ == "__main__":
    main()
