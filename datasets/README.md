# Datasets

Sample dataset files for `xpyd-bench`.

## Supported Formats

### JSONL (`.jsonl`)
One JSON object per line. Required field: `prompt` (or `text`, `input`, `question`).
Optional: `output_len` / `max_tokens`.

```jsonl
{"prompt": "What is the capital of France?", "output_len": 32}
{"prompt": "Explain relativity.", "output_len": 128}
```

### JSON (`.json`)
A JSON array of objects with the same fields as JSONL.

```json
[
  {"prompt": "Summarize open-source benefits.", "output_len": 100},
  {"prompt": "Compare Python and Rust.", "output_len": 150}
]
```

### CSV (`.csv`)
Header row with `prompt` column (or `text`, `input`, `question`).
Optional `output_len` / `max_tokens` column.

```csv
prompt,output_len
"Describe the water cycle.",128
"What is quantum computing?",64
```

## Synthetic Generation

Use `--dataset-name synthetic` with distribution options:

```bash
xpyd-bench --dataset-name synthetic \
  --num-prompts 500 \
  --input-len 256 \
  --output-len 128 \
  --synthetic-input-len-dist uniform \
  --synthetic-output-len-dist normal
```

Supported distributions: `fixed`, `uniform`, `normal`, `zipf`.
