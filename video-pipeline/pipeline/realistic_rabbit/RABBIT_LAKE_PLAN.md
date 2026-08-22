# RABBIT LAKE — Fotorealistischer Hase am See
## Technische Analyse & Testmatrix

**Ziel:** Fotorealistisches Video eines Hasen an einem See — ≥1920×1080, ≥30fps, stabile Anatomie, wenig Rauschen/Flackern.

---

## 1. Verfügbare Ressourcen

| Ressource | Pfad | Status |
|---|---|---|
| LTX-Video 13B 0.9.7-dev | `~/.cache/huggingface/…/ltxv-13b-0.9.7-dev.safetensors` | 27 GB ✓ |
| LTXConditionPipeline (Image-to-Video) | `tools/LTX-Video-Trainer/src/ltxv_trainer/ltxv_pipeline.py` | Unterstützt `image=` ✓ |
| Real-ESRGAN Anime 6B | `tools/realesrgan-models/RealESRGAN_x4plus_anime_6B.pth` | 18 MB ✓ |
| RIFE v4.6 Binary | `tools/rife-ncnn-vulkan/…/rife-v4.6/` | Vorhanden ✓ |
| GPU | NVIDIA RTX A6000, 49 GB VRAM | ~39 GB nutzbar |

**Einschränkungen:**
- Kein Bild-Generator installiert (kein SD/FLUX/ComfyUI)
- Real-ESRGAN Anime-Modell nicht ideal für Foto-Realismus (Photo-Modell fehlt)
- Natives 1920×1080 bei LTX-Video → OOM-Risiko → Weg via 960×544 + 2× Upscale

---

## 2. Auflösungs-Machbarkeit

| Auflösung | Frames | VRAM | Zeit (est.) | Machbarkeit |
|---|---|---|---|---|
| 768×432 | 49 | ~28 GB | ~7 min | ✓ sicher |
| 960×544 | 49 | ~30 GB | ~11 min | ✓ sicher |
| 1280×720 | 49 | ~36 GB | ~22 min | ✓ wahrscheinlich |
| 1920×1080 | 49 | ~48+ GB | >50 min | ✗ OOM-Risiko |

**Strategie:** Generierung bei 960×544 oder 1280×720 → Real-ESRGAN 2× → 1920×1088

---

## 3. Technische Ansätze

### Ansatz 1 — Text-zu-Video (Tests B, B2)
Kein Referenzbild nötig. Base-Modell ohne LoRA generiert direkt aus Text-Prompt.
- **Vorteil:** Einfach, sofort startbar
- **Nachteil:** Hasen-Anatomie kann zwischen Frames inkonsistent sein

### Ansatz 2 — Bild-zu-Video (Tests C, D) ← Empfohlen
Referenzbild ankert Hasen-Aussehen. LTXConditionPipeline übergibt `image=` Parameter.
- **Vorteil:** Konstanteste Hasen-Identität über alle Frames
- **Nachteil:** Braucht Referenzbild (Foto oder Frame aus Test B)
- **Parameter:** `image_cond_noise_scale=0.15` (Standard) oder `0.05` (sehr statisch)

### Ansatz 3 — Upscaling (Test E)
Real-ESRGAN 4× → `outscale=2` = 2× Upscale auf 1920×1088.
- **Vorteil:** Kein neuer Generierungslauf, deutlich mehr Schärfe
- **Nachteil:** Anime-Modell — für Foto-Realismus nicht ideal

### Ansatz 4 — Frame-Interpolation (Test F)
RIFE v4.6 Binary: 8fps → 30fps via 4× Interpolation.
- **Vorteil:** Bessere Qualität als ffmpeg minterpolate
- **Nachteil:** Bei schnellen Bewegungen Geister-Artefakte möglich (dezente Szene: kein Problem)

### Ansatz 5 — Kombination (Test G) ← Finales Ziel
I2V (960×544) → Real-ESRGAN (1920×1088) → RIFE (30fps) → finales MP4

---

## 4. Prompts

**Positiv:**
```
A photorealistic rabbit sits calmly at the edge of a natural lake shore,
clearly visible in the foreground. Fine detailed fur texture, anatomically
correct body, natural upright ears, expressive eyes. Calm lake with subtle
water reflections, lush green plants and peaceful landscape in the background.
Soft natural daylight, golden hour lighting. Very slight ear movement, subtle
breathing motion. Gentle ripple on water surface. Static camera. Same rabbit
appearance in every frame. High detail quality, minimal noise, natural motion,
cinematic quality, 16:9 composition.
```

**Negativ:**
```
blurry fur, heavy noise, flickering, texture popping, changing fur color,
changing body shape, extra ears, extra legs, deformed eyes, distorted anatomy,
duplicate body parts, unstable background, strong camera movement, unnatural
water movement, over-sharpening, compression artifacts, text, logo, watermark,
cartoon, anime, illustration, painting, drawing, CGI render, 3D render,
plastic look, low quality, worst quality, inconsistent motion, jittery,
multiple rabbits, no rabbit, rabbit disappearing, morphing
```

---

## 5. Testmatrix

| Test | Ansatz | LoRA | Auflösung | Frames | Steps | Guidance | Est. Zeit |
|---|---|---|---|---|---|---|---|
| A | Baseline (Lo-Fi LoRA R8) | R8 step_150 | 960×544 | 97 | 80 | 9.0 | 15 min |
| B | Text-zu-Video | keins | 768×432 | 49 | 50 | 7.0 | 7 min |
| B2 | Text-zu-Video HD | keins | 1280×720 | 49 | 60 | 7.0 | 22 min |
| C | Bild-zu-Video | keins | 960×544 | 49 | 50 | 6.0 | 11 min |
| D | Bild-zu-Video stark | keins | 960×544 | 49 | 50 | 5.0 | 11 min |
| E | Real-ESRGAN 2× | — | →1920×1088 | — | — | — | 10 min |
| F | RIFE v4.6 30fps | — | gleich | — | — | — | 5 min |
| G | I2V + ESRGAN + RIFE | keins | 960×544→1920×1088 | 49 | 50 | 6.0 | ~30 min |

---

## 6. Scripts

```bash
# Verzeichnis:
cd Bachelorarbeit/pipeline/realistic_rabbit/

# Alle automatischen Tests (A, B, B2):
bash run_tests.sh

# Einzelner Test:
bash run_tests.sh B
bash run_tests.sh A

# Tests C/D (nach Referenzbild erstellen):
bash run_tests.sh C ref_rabbit.jpg
bash run_tests.sh D ref_rabbit.jpg

# Post-Processing auf bestes Ergebnis:
bash run_tests.sh E outputs/test_C/rabbit_i2v_standard.mp4
bash run_tests.sh F outputs/test_C/rabbit_i2v_standard.mp4
```

**Referenzbild-Optionen:**
1. Bestes Frame aus Test B: `cp ref_rabbit_from_B.jpg ref_rabbit.jpg` (automatisch nach Test B erstellt)
2. Eigenes Foto: `cp /pfad/zu/hase.jpg ref_rabbit.jpg`

---

## 7. Empfehlung

**Haupt-Pipeline: Test G**
1. Referenzbild bereitstellen (Foto oder aus Test B)
2. `bash run_tests.sh C ref_rabbit.jpg` (~11 min)
3. `bash run_tests.sh E outputs/test_C/rabbit_i2v_standard.mp4` (~10 min)
4. `bash run_tests.sh F outputs/test_E/rabbit_i2v_standard_esrgan.mp4` (~5 min)
5. Ergebnis: 1920×1088, 30fps, stabile Hasen-Anatomie

**Alternativ ohne Referenzbild: Test B2 + E + F**
1. `bash run_tests.sh B2` (~22 min)
2. Post-Processing darauf anwenden

---

## 8. Bewertungsschema

Scores 1–10 je Kriterium:

| Kriterium | Was bewertet wird |
|---|---|
| `rabbit_sharpness` | Fell-Textur, Ohren, Augen scharf? |
| `fur_quality` | Natürlich realistisches Fell? |
| `anatomy` | Korrekte Proportionen, keine Deformierungen? |
| `rabbit_stability` | Gleiche Hasen-Identität in allen Frames? |
| `background_stability` | See/Landschaft stabil, kein Flackern? |
| `noise_inv` | Wenig Rauschen (10=kein Rauschen, 1=stark verrauscht) |
| `flicker_inv` | Kein Flackern (10=stabil, 1=stark flackernd) |
| `motion_naturalness` | Bewegung natürlich und dezent? |
| `lake_quality` | See-Spiegelungen realistisch? |
| `overall` | Gesamteindruck |

`avg = Summe aller Scores / 10`

Ergebnisse in `results.json` eintragen.
