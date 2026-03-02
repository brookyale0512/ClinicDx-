# CDS LoRA Fine-tuning

Fine-tunes MedGemma 4B on clinical decision support reasoning using LoRA + GRPO.

## Training Stages

1. **SFT** (`train_cds_lora.py`) — supervised fine-tuning on CDS reasoning examples
2. **LoRA merge** (`merge_lora.py`) — bakes LoRA adapter into base weights
3. **Eval** (`eval_medqa.py`) — benchmarks on MedQA subset

## Usage

```bash
# Set up environment
bash scripts/setup_env.sh

# Prepare dataset
python prep_cds_training.py

# Start training
bash scripts/run_training.sh

# Merge LoRA
python merge_lora.py --base-model /path/to/medgemma --adapter ./checkpoints/best

# Evaluate
python eval_medqa.py --model /path/to/merged
```

## Config

See `config.yaml` — key parameters:
- `model_name_or_path`: base MedGemma path
- `lora_r`, `lora_alpha`: LoRA rank and scaling
- `per_device_train_batch_size`, `gradient_accumulation_steps`
- `num_train_epochs`
