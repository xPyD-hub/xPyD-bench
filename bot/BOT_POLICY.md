<!-- CRITICAL: DO NOT SUMMARIZE OR COMPRESS THIS FILE -->
<!-- This file contains precise rules that must be read in full. -->

# Bot Policy — xPyD-bench

## Language
- **English only** — all code, docs, issues, PRs, comments on GitHub must be in English. No Chinese characters.

## Code Rules
- All changes go through PR. Never push directly to main.
- Every PR must have tests. No untested code.
- CI must be 100% green before merge. No skips allowed.
- No test may be skipped. If a test can't run, fix it or remove it.
- Rebase to latest main before pushing.

## Testing
- Unit tests in `tests/` — pure bench logic, no external dependencies.
- Integration tests in [xPyD-integration](https://github.com/xPyD-hub/xPyD-integration) — cross-component tests.

## Architecture
- Bench is a pure client tool. No server components.
- All inference backend interaction goes through xPyD-integration tests.
- Follow vLLM bench CLI compatibility (see [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md)).
