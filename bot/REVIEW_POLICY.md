<!-- CRITICAL: DO NOT SUMMARIZE OR COMPRESS THIS FILE -->
<!-- This file contains precise rules that must be read in full. -->

# Review Policy — xPyD-bench

## Roles

| Role | GitHub Account | Action |
|------|---------------|--------|
| Implementer | `hlin99` | Write code, submit PRs, fix issues |
| Reviewer 1 | `hlin99-Review-Bot` | Review PRs: approve / request changes / close |
| Reviewer 2 | `hlin99-Review-BotX` | Review PRs: approve / request changes / close |

## Timing Parameters

These are the single source of truth for all timing values:

| Parameter | Value |
|-----------|-------|
| Iteration interval | 10 minutes |
| PR wait for review | max 15 minutes |
| Fix after request changes | max 10 minutes |
| Reviewer check frequency | every 5 minutes |
| Reviewer response deadline | 15 minutes after assign |
| Reviewer timeout action | close PR (iteration failed) |

## Review Standards

- At least 1 approval required to merge.
- Blockers (🔴) must be fixed. No exceptions.
- Yellow (🟡) issues should be fixed unless author provides good reason.
- All CI checks must pass.
