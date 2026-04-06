# Contributing to xPyD-bench

## Development Setup

```bash
git clone https://github.com/xPyD-hub/xPyD-bench
cd xPyD-bench
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -q
```

## Code Style

- Python 3.10+
- Ruff for linting: `ruff check xpyd_bench/ tests/`
- All PRs must pass CI (lint + tests on 3.10/3.11/3.12 + integration trigger)

## PR Process

1. Create a branch from `main`
2. Make changes, add tests
3. Push and open PR
4. CI runs: unit tests + integration tests (via trigger)
5. Review and merge

## Bot Development

See [bot/](bot/) for automated development policies and iteration records.
