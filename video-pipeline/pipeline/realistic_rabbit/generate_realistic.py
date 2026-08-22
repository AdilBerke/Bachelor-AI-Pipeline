"""
Fotorealistischer Hase am See — Videogenerierung mit optionalem Image-Conditioning.

Verwendung:
  # Test B: Text-zu-Video, kein Referenzbild
  python generate_realistic.py --output outputs/test_B/rabbit_lake.mp4 --steps 50 --guidance 7.0 --width 768 --height 432

  # Test C: Bild-zu-Video mit Referenzbild
  python generate_realistic.py --image ref_rabbit.jpg --output outputs/test_C/rabbit_lake.mp4 --steps 50 --guidance 6.0

  # Test B2: Höhere Auflösung
  python generate_realistic.py --output outputs/test_B2/rabbit_lake.mp4 --steps 60 --guidance 7.0 --width 1280 --height 720
"""
import argparse
import os
import sys
import time
import torch

sys.path.insert(0, "/home/BA_Musikproduktion/Documents/Bachelor_VisiualStudio/tools/LTX-Video-Trainer/src")

from diffusers.utils import export_to_video, load_image
from ltxv_trainer.model_loader import load_ltxv_components, LtxvModelVersion
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline

POSITIVE_PROMPT = (
    "A photorealistic rabbit sits calmly at the edge of a natural lake shore, clearly visible in the foreground. "
    "Fine detailed fur texture, anatomically correct rabbit body, natural upright ears, expressive eyes. "
    "Calm lake with subtle water reflections, lush green plants and peaceful natural landscape in the background. "
    "Soft natural daylight, golden hour lighting, warm sunlight. "
    "Very slight ear movement, subtle breathing motion, subtle head tilt. "
    "Gentle ripple on water surface. Static camera, no camera movement. "
    "Same rabbit appearance in every frame, consistent fur color and body shape. "
    "High detail quality, minimal grain, natural motion, cinematic quality, 16:9 composition."
)

NEGATIVE_PROMPT = (
    "blurry fur, heavy noise, flickering, texture popping, changing fur color, changing body shape, "
    "extra ears, extra legs, deformed eyes, distorted anatomy, duplicate body parts, "
    "unstable background, strong camera movement, unnatural water movement, "
    "over-sharpening, compression artifacts, text, logo, watermark, "
    "cartoon, anime, illustration, painting, drawing, CGI render, 3D render, "
    "plastic look, low quality, worst quality, inconsistent motion, jittery, "
    "multiple rabbits, no rabbit, rabbit disappearing, morphing, transforming"
)


def main():
    parser = argparse.ArgumentParser(description="Realistischer Hase am See")
    parser.add_argument("--image", type=str, default=None,
                        help="Referenzbild für Image-to-Video Conditioning (optional)")
    parser.add_argument("--image-noise-scale", type=float, default=0.15,
                        help="Wie stark das Referenzbild beeinflusst wird (Standard: 0.15, weniger = statischer)")
    parser.add_argument("--strength", type=float, default=0.9,
                        help="Denoising-Stärke bei Image-Conditioning (Standard: 0.9)")
    parser.add_argument("--lora", type=str, default=None,
                        help="LoRA-Gewichte (Standard: keins — Base-Modell für Realismus)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="outputs/test_B/rabbit_lake.mp4")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative-prompt", type=str, default=None)
    args = parser.parse_args()

    prompt = args.prompt or POSITIVE_PROMPT
    neg_prompt = args.negative_prompt or NEGATIVE_PROMPT

    print("Lade LTX-Video 13B 0.9.7-dev...")
    t0 = time.time()
    components = load_ltxv_components(
        model_source=LtxvModelVersion.LTXV_13B_097_DEV,
        load_text_encoder_in_8bit=True,
        transformer_dtype=torch.bfloat16,
        vae_dtype=torch.bfloat16,
    )

    print("GPU-Transfer...")
    components.transformer.to("cuda")
    components.vae.to("cuda")

    pipe = LTXConditionPipeline(
        scheduler=components.scheduler,
        tokenizer=components.tokenizer,
        text_encoder=components.text_encoder,
        vae=components.vae,
        transformer=components.transformer,
    )

    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if args.lora:
        print(f"Lade LoRA: {args.lora}")
        pipe.load_lora_weights(args.lora)

    mode = "Image-to-Video" if args.image else "Text-to-Video"
    print(f"\nModus: {mode}")
    print(f"Auflösung: {args.width}×{args.height}×{args.frames} frames")
    print(f"Steps: {args.steps}, Guidance: {args.guidance}, Seed: {args.seed}")
    if args.image:
        print(f"Referenzbild: {args.image}")
        print(f"  image_cond_noise_scale: {args.image_noise_scale}")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.cuda.empty_cache()

    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=neg_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        frame_rate=25,
        decode_timestep=0.10,
        decode_noise_scale=0.025,
        output_reference_comparison=False,
    )

    if args.image:
        ref_image = load_image(args.image)
        call_kwargs["image"] = ref_image
        call_kwargs["image_cond_noise_scale"] = args.image_noise_scale

    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = pipe(**call_kwargs)

    load_time = time.time() - t0
    print(f"\nGenerierung abgeschlossen ({load_time:.0f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp_path = args.output.replace(".mp4", "_raw.mp4")
    export_to_video(result.frames[0], tmp_path, fps=8)

    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        args.output
    ], check=True, capture_output=True)
    os.remove(tmp_path)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Gespeichert: {args.output}  ({size_mb:.1f} MB, {args.width}×{args.height}, 8fps)")


if __name__ == "__main__":
    main()
