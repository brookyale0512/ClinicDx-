# ClinicDx CDS SFT Dataset

Ante-hoc KB tool-use reasoning dataset for fine-tuning MedGemma on clinical decision support.

## Contents

```
CDS_dataset/
├── sft_train/
│   ├── train.jsonl     — 42,099 training records (1.3 GB)
│   ├── val.jsonl       —  4,677 validation records (145 MB)
│   └── manifest.json   — provenance, stats, quality gates
├── raw/
│   ├── cds_enriched/   — (symlink target: kbToolUseLora CDS shards)
│   └── mcq_enriched/   — (symlink target: kbToolUseLora MCQ shards)
├── build_sft_dataset.py — reproducible build script
└── README.md
```

## Dataset Statistics

| Split | Records | Size |
|-------|---------|------|
| Train | 42,099  | 1.3 GB |
| Val   |  4,677  | 145 MB |
| **Total** | **46,776** | **1.45 GB** |

**Composition:** 70.5% CDS clinical cases / 29.5% MCQ questions

**Sources:**
- CDS: 32,980 enriched OpenMRS clinical cases across 80 disease categories
- MCQ: 13,796 questions from MedQA (5,903), MedMCQA (5,948), AfriMedQA (2,131)

## Format

Each record has a `text` field containing a complete Gemma 3 chat-format training sequence:

```
<bos><start_of_turn>user
[System prompt + clinical case XML or MCQ question]
<end_of_turn>
<start_of_turn>model
<think>
[Multi-step clinical reasoning with KB queries and results]
</think>
[Structured CDS response or Final Answer: X]
<end_of_turn>
```

## Quality Gates Applied

- `text` field must contain `<think>` and `</think>` tags
- At least 1 KB query with score > 0
- Think blocks >= 100 tokens (actual avg: 8,802 for CDS, 1,447 for MCQ)
- MCQ: only correct answers retained
- Deduplication by record ID

**Filtered out:** 7 CDS + 6 MCQ JSON parse errors, 213 + 189 duplicates. Zero thin-think or zero-score records.

## Training Notes

- Use `medgemma_text_only` (3.88B) for SFT — no vision tower, avoids SigLIP issues
- Apply **loss masking**: mask user/case input and `<KB_RESULT>` tokens; gradient only on `<think>` + final answer
- LoRA r=16, alpha=32 (same as Voice Scribe Phase 1)
- The `text` field is already in Gemma 3 chat format — use `dataset_text_field="text"` in SFTTrainer
- Expected outcome: 72-75% MedQA accuracy after SFT

## Rebuild

```bash
cd /var/www/ClinicDx
python3 dataset/CDS_dataset/build_sft_dataset.py         # full build
python3 dataset/CDS_dataset/build_sft_dataset.py --dry-run  # stats only
```

The enrichment pipelines are still running (35K→83K CDS, 15K MCQ). Rebuild after enrichment completes for a larger dataset. The 70/30 ratio will be maintained automatically.
