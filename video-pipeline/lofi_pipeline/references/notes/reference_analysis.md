# Reference Analysis Notes

## Status

Motion profile templates created (2026-07-17). Awaiting video URLs from user to fill in
`source_reference` fields and refine timing/amplitude values based on actual footage analysis.

## Motion Profiles Created

| File | Element | Status |
|---|---|---|
| `motion_profiles/tree_wind_motion.yaml` | Trees / leaves / branches | Template ready |
| `motion_profiles/rabbit_blink_ear_motion.yaml` | Rabbit faces, ears, blinking | Template ready |
| `motion_profiles/star_twinkle_shooting_star.yaml` | Night sky, stars, shooting star | Template ready |
| `motion_profiles/water_reflection_motion.yaml` | Lake surface, moon/star reflections | Template ready |

## What to do when video URLs arrive

1. Watch each video, note the timestamps with clearest motion examples
2. Fill in `source_reference` field per YAML with the YouTube URL
3. Refine `timing` fields: measure actual period/speed in seconds from video
4. Refine `amplitude`: estimate motion range in pixels relative to frame size
5. Add `observed_examples` list with specific timestamps (e.g., "at 0:34 — stars twinkle clearly")
6. Change `analysis_status: template_ready` → `analysis_status: analyzed`

## How these profiles feed into training

### Phase 1 (image fix — no animation yet)
- `rabbit_blink_ear_motion.yaml` → `anatomy_requirements` and `what_to_avoid` sections
  inform the prompt and negative_prompt in `scenario.yaml`
- Especially: count_constraint, negative_prompt_keywords

### Phase 2 (animation)
- All 4 profiles → used to select training clip content during next clip-sourcing session
- Videos with matching motion patterns become priority training clips
- Profiles also become the target description in clip captions

### Phase 3 (post-processing)
- Motion profiles → parameters for RIFE interpolation and potential compositing
- `timing` fields inform frame-count targets for seamless loops

## Key insight for rabbit_lake specifically

The 3-rabbit problem is a BASE MODEL prior (not a training data issue).
Motion profiles for stars, water, trees are CORRECT targets for training.
Rabbit animation must come from prompt engineering first, then training clips with exactly 2 rabbits.

---

## Notes on content that should NOT be in training data

The following categories would cause problems if added as training clips:
- Any Lo-Fi girl at desk clips → trains "person at desk" not "rabbits at lake"
- Cartoon-style bunny clips → trains wrong art style
- Clips with visible watermarks → embeds logos into model
- Clips with 3 or more animal figures → reinforces 3-figure composition prior
- Daytime scenes → trains wrong lighting and color palette
- Interior/bedroom scenes → trains wrong scene type
