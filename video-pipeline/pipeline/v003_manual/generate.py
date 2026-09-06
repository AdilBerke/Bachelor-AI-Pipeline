"""
Generate a lofi girl video using LTX-Video 13B 0.9.7-dev.
Uses ltxv_trainer's LTXConditionPipeline (same as validation inference).
"""
import argparse
import os
import sys
import time
import torch

sys.path.insert(0, "/home/BA_Musikproduktion/Documents/Bachelor_VisiualStudio/tools/LTX-Video-Trainer/src")

from diffusers.utils import export_to_video
from ltxv_trainer.model_loader import load_ltxv_components, LtxvModelVersion
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline

PROMPT = (
    "lofi_girl, cozy lofi anime study room, young woman at wooden desk writing in open notebook, "
    "fluid smooth animation, natural graceful hand movement, clearly visible pen writing in notebook, "
    "visible pen strokes and written lines on paper, eyes slightly open, subtle gentle blink, "
    "organic subtle gestures, smooth frame transitions, soft continuous motion, gentle hair movement, "
    "polished anime illustration, clean linework, refined face, warm desk lamp glow, amber colors, "
    "bookshelves plants desk objects, high quality, full frame 16:9, stable consistent scene"
)

NEGATIVE_PROMPT = (
    "jerky motion, abrupt movement, stiff animation, choppy frames, unclear hand motion, "
    "static pose, frozen movement, sudden jumps, teleporting limbs, new scenario, changed room, "
    "eyes fully closed, blank notebook page, empty paper, "
    "grayscale, black canvas, cropped video, low resolution, blurry, pixel noise, halftone, "
    "grain, compression artifacts, muddy details, flickering, warped hands, "
    "flat background, inconsistent character"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=str,
        default="/home/BA_Musikproduktion/Documents/Bachelor_VisiualStudio/Bachelorarbeit/training/video/ltx_lora_manual/samples/lofi_13b.mp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--gpu-fraction", type=float, default=1.0,
                        help="GPU-Auslastung begrenzen (0.8 = 80%%). Fügt Pausen zwischen Steps ein.")
    args = parser.parse_args()

    if args.prompt:
        global PROMPT
        PROMPT = args.prompt
    if args.negative_prompt:
        global NEGATIVE_PROMPT
        NEGATIVE_PROMPT = args.negative_prompt

    print("Loading LTX-Video 13B 0.9.7-dev...")
    components = load_ltxv_components(
        model_source=LtxvModelVersion.LTXV_13B_097_DEV,
        load_text_encoder_in_8bit=True,
        transformer_dtype=torch.bfloat16,
        vae_dtype=torch.bfloat16,
    )

    print("Moving components to CUDA...")
    components.transformer.to("cuda")
    components.vae.to("cuda")
    # text_encoder stays where bitsandbytes put it (already on cuda)

    pipe = LTXConditionPipeline(
        scheduler=components.scheduler,
        tokenizer=components.tokenizer,
        text_encoder=components.text_encoder,
        vae=components.vae,
        transformer=components.transformer,
    )

    # Reduce peak VRAM during VAE decode via tiling
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if args.lora:
        print(f"Loading LoRA from {args.lora}...")
        pipe.load_lora_weights(args.lora)

    print(f"Generating ({args.steps} steps, {args.width}x{args.height}x{args.frames}, gpu={args.gpu_fraction:.0%})...")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Free fragmented cached memory before generation
    torch.cuda.empty_cache()

    # GPU-Throttle: Pausen zwischen Steps um Auslastung zu begrenzen
    _step_times: list[float] = []
    _step_start: list[float] = [time.monotonic()]

    def _throttle_callback(pipe, step_idx, timestep, callback_kwargs):
        torch.cuda.synchronize()
        elapsed = time.monotonic() - _step_start[0]
        _step_times.append(elapsed)
        if args.gpu_fraction < 1.0 and elapsed > 0:
            sleep_sec = elapsed * (1.0 - args.gpu_fraction) / args.gpu_fraction
            time.sleep(sleep_sec)
        _step_start[0] = time.monotonic()
        return callback_kwargs

    throttle_kwargs = {}
    if args.gpu_fraction < 1.0:
        throttle_kwargs["callback_on_step_end"] = _throttle_callback
        throttle_kwargs["callback_on_step_end_tensor_inputs"] = []

    with torch.autocast("cuda", dtype=torch.bfloat16):
      result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        frame_rate=25,         # explicit temporal RoPE scale (8/25 = 0.32)
        decode_timestep=0.10,  # 0.10 allows more temporal variation (was 0.05, too aggressively smoothed motion)
        decode_noise_scale=0.025,
        output_reference_comparison=False,
        **throttle_kwargs,
      )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tmp_path = args.output.replace(".mp4", "_raw.mp4")
    export_to_video(result.frames[0], tmp_path, fps=8)

    # Re-encode with correct color space metadata (fixes green tint in browser/players)
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        args.output
    ], check=True, capture_output=True)
    os.remove(tmp_path)
    print(f"Saved: {args.output}")

    # ── Automatische Schaerfe-/Qualitaetsbewertung (best effort) ──────────────
    # Schreibt <output>.metrics.json neben das Video (Schaerfe-Score + SSIM,
    # Flicker, Motion, Farbe, Gesamt). Darf die Generierung NIE abbrechen.
    _eval_script = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "realistic_rabbit", "evaluate_video.py"))
    try:
        if os.path.exists(_eval_script):
            subprocess.run(
                [sys.executable, _eval_script, args.output,
                 "--save-report", "--max-frames", "61"],
                check=False, capture_output=True, text=True, timeout=300)
            _mj = args.output.replace(".mp4", ".metrics.json")
            if os.path.exists(_mj):
                import json as _json
                with open(_mj) as _f:
                    _m = _json.load(_f)
                _s = _m.get("sharpness", {})
                _n = _m.get("niqe", {})
                _nstr = (f"NIQE {_n.get('mean')}" if _n.get("available")
                         else "NIQE n/a")
                print(
                    f"Schaerfe-Score: {_s.get('score_100', '?')}/100  "
                    f"(Laplacian {_s.get('raw_laplacian', '?')}, "
                    f"Tenengrad {_s.get('tenengrad', '?')}, "
                    f"Konstanz {_s.get('konstanz', '?')})  |  "
                    f"{_nstr} (blind)  |  "
                    f"Gesamt-Score {_m.get('overall_score', '?')}")
        else:
            print(f"(Bewertung uebersprungen: evaluate_video.py nicht gefunden)")
    except Exception as _e:
        print(f"(Bewertung uebersprungen: {_e})")

if __name__ == "__main__":
    main()
