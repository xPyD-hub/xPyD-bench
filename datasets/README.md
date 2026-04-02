# Datasets

Place benchmark datasets here.

## Supported formats

- **JSON**: Array of objects with `prompt` field
- **JSONL**: One object per line with `prompt` field

## Schema

### Completion endpoint

```json
[
  {"prompt": "Once upon a time"},
  {"prompt": "The quick brown fox"}
]
```

### Chat endpoint

```json
[
  {"messages": [{"role": "user", "content": "Hello, how are you?"}]},
  {"messages": [{"role": "user", "content": "Explain quantum computing"}]}
]
```
