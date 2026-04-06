<!-- CRITICAL: DO NOT SUMMARIZE OR COMPRESS THIS FILE -->
<!-- This file contains operational steps that must be followed exactly. -->

# Development Loop — xPyD-bench

Autonomous iteration loop. References policies for rules — this file only describes the operational workflow.

## Rules

All rules are defined in policy files. Read them first:
- [BOT_POLICY.md](BOT_POLICY.md) — hard constraints
- [AUTHOR_POLICY.md](AUTHOR_POLICY.md) — code quality, PR process, doc updates
- [REVIEW_POLICY.md](REVIEW_POLICY.md) — timing, review standards

## Setup (every iteration)
```
git config user.email "tony.lin@intel.com"
git config user.name "hlin99"
```

## Each Iteration

1. Pull latest code
2. Read `ROADMAP.md` — find the next incomplete milestone
3. Read `DESIGN_PRINCIPLES.md` — follow the rules
4. Check open issues/PRs — handle unmerged PRs first
5. Create GitHub Issue: problem, solution, acceptance criteria, tests
6. Create branch, implement code + tests
7. Verify locally (lint + tests per AUTHOR_POLICY.md)
8. Push, create PR, request review
9. Wait for review (timing per REVIEW_POLICY.md)
10. Fix review comments, iterate until approved
11. Merge and update `bot/iterations/current.md`
12. Next iteration
