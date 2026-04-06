<!-- CRITICAL: DO NOT SUMMARIZE OR COMPRESS THIS FILE -->
<!-- This file contains precise rules that must be read in full. -->

# Review Policy — xPyD-bench

## Roles
| Role | GitHub Account | Action |
|------|---------------|--------|
| Implementer | `hlin99` | Write code, submit PRs |
| Reviewer 1 | `hlin99-Review-Bot` | Review PRs |
| Reviewer 2 | `hlin99-Review-BotX` | Review PRs |

Each reviewer uses its own dedicated token. Never use author's token for reviews.

## Timing Parameters
| Parameter | Value |
|-----------|-------|
| Iteration interval | 10 minutes |
| PR wait for review | max 15 minutes |
| Fix after request changes | max 10 minutes |
| Reviewer check frequency (has PRs) | every 5 minutes |
| Reviewer check frequency (no PRs) | every 15 minutes |
| Reviewer response deadline | 15 minutes after assign |
| Reviewer timeout action | close PR (iteration failed) |

## What to Review
1. Skip draft PRs.
2. Skip already-reviewed commits (only APPROVE counts as reviewed).
3. Re-requested reviews take priority — always perform fresh review.
4. One review per PR per commit SHA — never submit multiple reviews for same commit.

## Review Priority

Reviewers evaluate in this order:
1. **Design value** — Is this feature/change worth building? Does it solve a real problem? Is the approach sound? If the design itself is flawed, REQUEST_CHANGES before reviewing any code.
2. **Design conformance** — Does the implementation match the linked GitHub Issue spec? If it deviates, REQUEST_CHANGES — even if the code is technically correct.
3. **Code quality** — See checklist below.

A technically perfect implementation of a bad design is still wrong. Reject it.

## Review Checklist
For each non-draft PR with a new commit:

| Area | Check |
|---|---|
| CI | Must be fully green before APPROVE. May submit REQUEST_CHANGES or COMMENT while pending. |
| Merge conflicts | If mergeable == false, REQUEST_CHANGES. |
| Logic errors | Incorrect conditions, off-by-one, unhandled edge cases. |
| Type safety | Mismatched types, missing None checks. |
| Concurrency | Race conditions, missing locks, shared mutable state. |
| Exception handling | Bare except, swallowed exceptions, resource leaks. |
| Security | Injection risks, hardcoded secrets, unsanitized input. |
| Code style | Unused imports, shadowed variables, unclear naming. |
| Test coverage | New logic must have corresponding tests. |
| Design conformance | Implementation must match the linked GitHub Issue design. |

## Verdicts
- **APPROVE** — code correct, CI green, no issues.
- **REQUEST_CHANGES** — any issue found. Use inline comments.
- **COMMENT** — CI pending or noting something without blocking.

## Merge Policy (Loop Mode)
- Both reviewers approve + CI green → bot auto-merges.
- Reviewer timeout (15 min) → bot closes PR (iteration failed).
- Single approve is not enough to merge. Both must approve.
