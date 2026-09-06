# Trainierter LoRA-Adapter

`lora_adapter.pt` — der in der Bachelorarbeit verwendete Modellstand.

| Eigenschaft | Wert |
|---|---|
| Basismodell | `facebook/musicgen-melody-large` |
| Verfahren | LoRA, Rank 8, Alpha 16, Dropout 0,05 |
| Trainingsschritte | 625 |
| Trainierbare Parameter | ca. 9,44 Millionen |
| Trainingsdaten | 5.000 genrebalancierte 30-s-Clips |
| Größe | 113.585.054 Bytes |
| SHA256 | `2e7327592d91bc18bb4ebc02e69c7c0b2373f1ba3658d0ac64aea0f54efc26c6` |

## Verwendung

Datei an den in `audio-pipeline/code/configs/konfiguration.yaml` unter
`bester_lora_checkpoint` erwarteten Ort legen und über `lora.py --freigeben-checkpoint`
freigeben. Der Ablauf ist in `../NACHBAUANLEITUNG.md` Schritt 4f beschrieben.

Prüfsumme vor der Verwendung kontrollieren:

```bash
sha256sum models/lora_adapter.pt
```

Ohne diesen Adapter erzeugt ein frisch geladenes MusicGen-Basismodell keine Ausgaben im
trainierten Lo-Fi-Stil.
