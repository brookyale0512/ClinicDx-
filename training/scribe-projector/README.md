# Scribe AudioProjector Training

Trains the `Gemma3AudioProjector` — a lightweight bridge (~11.8M params) mapping
MedASR encoder outputs into MedGemma's embedding space.

## Architecture

```
MedASR Wav2Vec2 [frozen]  →  [B, T, 512]
  Frame stacking (k=4)    →  [B, T/4, 2048]
  Linear(2048 → 2560)
  RMSNorm + GELU
  Linear(2560 → 2560)
  Pad to 64 tokens        →  [B, 64, 2560]  (MedGemma embed dim)
```

## Training

```bash
# Stage 1: Train projector (base model, large speech dataset)
python train_projector.py --config configs/train_config.yaml

# Stage 2: Fine-tune on ClinicDx clinical phrases
python train_projector.py --config configs/train_config_clinicdx.yaml

# Validate
python validate_scribe.py --projector checkpoints/projector_final.pt
```

## Output

A single `projector_final.pt` file (~47MB) that is loaded by `serve_unified.py`
alongside the frozen MedGemma model.
