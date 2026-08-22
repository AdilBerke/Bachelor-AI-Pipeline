# Lokal High-Quality Audio (Uni-GPU)

Dieses Projekt hat bereits einen lokalen High-Quality Generator:
`archiv/hybrid_ap4/ap4_prompting_musikgenerierung/foundation_prompting.py`.

## 1) Umgebung vorbereiten

```powershell
cd Bachelor_VisiualStudio
python -m venv code\backend\.venv
.\code\backend\.venv\Scripts\Activate.ps1
pip install -r code\backend\requirements.txt
```

## 2) GPU kurz pruefen

```powershell
python -c "import torch; print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('vram_gb=', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1) if torch.cuda.is_available() else 0)"
```

## 3) Kurzer Funktionstest (3 Minuten)

```powershell
python archiv\hybrid_ap4\ap4_prompting_musikgenerierung\foundation_prompting.py `
  --minutes 3 `
  --genre lofi `
  --mood random `
  --quality final_master `
  --model-pool facebook/musicgen-large `
  --render-budget-hours 1 `
  --log-level INFO
```

## 4) Finale lokale HQ-Erzeugung (z.B. 60 Minuten)

```powershell
python legacy\hybrid_ap4\ap4_prompting_musikgenerierung\foundation_prompting.py `
  --minutes 60 `
  --genre lofi `
  --mood random `
  --quality final_master `
  --model-pool facebook/musicgen-large `
  --render-budget-hours 8 `
  --log-level INFO
```

Alternativ per Wrapper-Skript:

```powershell
.\run-local-hq.ps1 -Minutes 60 -Quality final_master -RenderBudgetHours 8
```

Hinweise:
- `final_master` ist jetzt bewusst aufwendig konfiguriert (mehr Kandidaten, laengere Suche).
- Standardmaessig wird fuer `final_master` das grosse Modell `facebook/musicgen-large` priorisiert.
- Ohne `--seed` wird automatisch ein neuer Seed genutzt (mehr Zufall zwischen Runs).
- Outputs liegen standardmaessig in `outputs/generated_audio/`.
