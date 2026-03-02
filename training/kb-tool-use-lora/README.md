# KB Tool-Use LoRA (2-Query Format)

Trains MedGemma to use the knowledge base as a tool during reasoning.
The model learns to emit `<KB_QUERY>` tags and incorporate `<KB_RESULT>` injections.

## Training Data Format

Each example contains a clinical case, up to 2 KB queries, and a final answer:

```json
{
  "messages": [
    {"role": "user", "content": "<patient case>"},
    {"role": "assistant", "content": "<KB_QUERY>malaria diagnosis</KB_QUERY>"},
    {"role": "user", "content": "<KB_RESULT>...WHO content...</KB_RESULT>"},
    {"role": "assistant", "content": "## Clinical Assessment\n..."}
  ]
}
```

## Usage

```bash
pip install -r requirements.txt

# Validate that KB daemon is running
python validate_kb_live.py

# Train
python train.py --config config.yaml
```

## Notes

- Requires KB daemon running at `http://127.0.0.1:4276` during training data validation
- `validate_kb_live.py` runs model predictions against live KB to verify tool-use quality
