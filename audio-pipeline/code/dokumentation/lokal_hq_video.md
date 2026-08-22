# Lokal High-Quality Video (Uni-GPU)

Visuelles Pendant zu `lokal_hq_audio.md`. Der Generator liegt unter
`code/src/Video/local_video.py` und erzeugt aus einem `ScenePreset`
ein hochwertiges Lo-Fi-Loop-Video auf einer NVIDIA-GPU.

## Pipeline-Ueberblick

1. `prompt_builder.py` mappt das `ScenePreset` (Mood, Setting, TimeOfDay,
   Lighting, Atmosphere, RenderStyle, Character, Props) auf einen SDXL-Prompt
   inklusive Lo-Fi-Qualitaetssuffix und Negative-Prompt.
2. `image_generator.py` laedt SDXL-Base (+ optional Refiner) in fp16 und
   rendert ein PNG. Die Pipeline wird per Prozess-Cache nur einmal auf die
   GPU geladen.
3. `motion_filters.py` baut einen FFmpeg-Filtergraph: Ken-Burns-Zoom,
   TimeOfDay/Lighting-Farbkorrektur, atmosphaerische Overlays
   (Regen/Schnee/Feuer/Sterne/Dampf/Blaetter) und Filmkorn/Vignette.
4. `local_video.py` orchestriert Bild-Cache, FFmpeg-Render und optionalen
   Audio-Mux (Dauer via `ffprobe`).

## 1) Umgebung vorbereiten

```powershell
cd Bachelor_VisiualStudio
python -m venv code\backend\.venv
.\code\backend\.venv\Scripts\Activate.ps1
pip install -r code\backend\requirements.txt
```

Die Datei `code/backend/requirements.txt` enthaelt jetzt zusaetzlich
`diffusers`, `transformers`, `accelerate`, `safetensors`, `Pillow`.

## 2) GPU kurz pruefen

```powershell
python -c "import torch; print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('vram_gb=', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1) if torch.cuda.is_available() else 0)"
```

SDXL-Base braucht ~8 GB VRAM in fp16, Base+Refiner ~12 GB.

## 3) Kurzer Funktionstest (standard, 10 s)

```powershell
python src\Video\local_video.py `
  --quality standard `
  --duration 10 `
  --mood COZY `
  --setting STUDY_DESK `
  --atmosphere RAIN `
  --time-of-day NIGHT `
  --log-level INFO
```

Beim allerersten Aufruf laedt diffusers SDXL von huggingface.co herunter
(~7 GB).

## 4) Finale lokale HQ-Erzeugung (z.B. 60 min Track)

```powershell
python src\Video\local_video.py `
  --quality final_master `
  --audio outputs\generated_audio\musicgen_final_master_xxx.wav `
  --mood NIGHT `
  --setting ROOFTOP `
  --atmosphere NEON `
  --time-of-day LATE_NIGHT `
  --lighting NEON_GLOW `
  --render-style ANIME_STYLE `
  --seed 42 `
  --output-dir outputs\generated_video `
  --log-level INFO
```

Die Videodauer wird automatisch aus der Audiodatei bestimmt (`ffprobe`),
der Audio-Mux laeuft in einem FFmpeg-Aufruf mit `-shortest`.

Alternativ per Wrapper-Skript:

```powershell
.\run-local-video.ps1 -Quality final_master -Duration 3600 `
  -Audio "outputs\generated_audio\musicgen_final_master_xxx.wav" `
  -Mood NIGHT -Setting ROOFTOP -Atmosphere NEON -Seed 42
```

## Qualitaetsprofile

| Profil | Bildgroesse | Schritte | Refiner | Video | CRF |
| ------ | ----------- | -------- | ------- | ----- | --- |
| draft | 1024x576 | 18 | nein | 1280x720 | 22 |
| standard | 1280x720 | 30 | nein | 1280x720 | 20 |
| premium | 1536x864 | 40 | ja | 1920x1080 | 18 |
| final_master | 1920x1088 | 55 | ja | 1920x1080 | 16 |

Die Namen decken sich bewusst mit den Audio-Profilen aus
`foundation_prompting.py`, damit Audio und Video konsistent benannt sind.

## Backend-Integration

`backend/app.py` importiert `generate_local_video` beim Start. Der Endpoint
`/api/lofi/generate` akzeptiert jetzt folgende Zusatzfelder:

- `use_local_generator: true` schaltet die HQ-Pipeline ein.
- `hq_quality` (`draft` | `standard` | `premium` | `final_master`).
- `hq_duration_seconds` (hebt das normale 10-15 s Limit auf).
- `hq_audio_path` (optional, wird gemuxt; Dauer ueberschreibt `hq_duration_seconds`).
- `hq_seed` (optional, fuer reproduzierbare Bilder).

Die Antwort enthaelt zusaetzlich ein `hq`-Objekt mit Bildpfad, Params-JSON,
Render-Zeiten und Cache-Hit-Flag.

## Caching

- Pipelines werden pro Prozess gecached (`_PIPE_CACHE` in
  `image_generator.py`): SDXL wird nur einmal auf die GPU geladen.
- Generierte PNGs werden pro Kombination aus Prompt, Seed und Profil unter
  `<output-dir>/images/` abgelegt. Wiederholte Aufrufe fuer dieselbe Szene
  ueberspringen SDXL komplett und laufen in Sekunden durch.

## Hinweise

- Bei OOM auf der GPU hilft `--quality standard` (ohne Refiner) oder ein
  manueller `pipe.enable_model_cpu_offload()` im `image_generator.py`.
- Outputs liegen standardmaessig unter `ausgaben/generated_video/`.
- Bei Fehlern im FFmpeg-Schritt wird die letzte 1800 Zeichen aus stderr in
  die Exception gepackt.
