# Agentic automation for waypoint — design

## Context

Source material: a personal survey, "agentic-engineering-survey.md," on running AI
coding agents safely and productively. That survey is explicitly scoped to a
~10-person team on GitLab with financial-services compliance constraints
(vendor review, CODEOWNERS, per-role machine identities, OIDC credentials,
sandboxed CI runners). Most of that team/compliance machinery does not apply
to waypoint: a solo-maintained Python library on GitHub. What transfers is the
core thesis — **generation isn't the constraint; verification and review
attention are** — and the survey's staged rollout, scaled down:

- Phase 0 (substrate): one CI-authoritative verification command, agents
  can't weaken it, zero autonomy yet.
- Phase 2 (first autonomous queue, low stakes): scheduled maintenance only,
  full human review retained, opens PRs, never merges.

Phases 1, 3, 4 (supervised-delegation metrics, issue/alert-driven autonomy,
throughput optimization) are out of scope for this design — revisit only
after Phase 0+2 have been running for a while.

## Goal

Get waypoint to the point where:
1. Every change — human or agent-authored — passes one command, run
   identically locally and in CI, before it can merge to `main`.
2. A weekly scheduled agent handles low-risk maintenance unattended (dep
   version drift, lint/type drift, flaky test triage), always via a PR that a
   human reviews and merges.
3. Core analytics logic, the verification gate itself, and golden regression
   fixtures are off-limits to the scheduled agent — it cannot "fix" a
   failing check by weakening the check.

## 1. Verification loop

**`justfile`** at repo root (chose `just` over `make`: no tab/space
indentation footguns, no `.PHONY` boilerplate, self-documenting via
`just --list`; this is a command runner, not a file-dependency build, so
Make's actual strength — incremental rebuilds — buys nothing here):

```
check:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/
    uv run mypy
    uv run pytest
```

**`.github/workflows/ci.yml`** — triggers on PRs and pushes to `main`,
installs `just` and `uv`, runs `uv sync --extra dev` then `just check`.
This repo currently has no CI at all, so this workflow is what makes the
gate real rather than aspirational.

**Golden regression fixtures** — seeded synthetic portfolios with pinned
expected outputs, one fixture family per analytics surface:
`ExpectedReturn`, `Risk`, `Optimizer`, `WealthSimulation`. Fixtures live
under `tests/fixtures/golden/` (parquet or JSON, whichever is more natural
per fixture) and are loaded by dedicated `test_*_golden.py` files living
alongside the existing per-module test files. Comparisons use a tight
numeric tolerance, not exact equality (floating point). Updating a golden
fixture is a deliberate, reviewed act — never something a green test run
does on its own — and the scheduled agent (section 3) is explicitly
forbidden from touching these files.

**Property tests** — add `hypothesis` to the `test` extra. A handful of
tests on invariants that would be expensive to get silently wrong, not
exhaustive coverage:
- Portfolio weights always normalize to sum to 1.0 regardless of input
  weight vector.
- Risk/volatility outputs are never negative.
- Compounding math is consistent under re-aggregation (e.g., chaining two
  sub-period returns equals the equivalent single-period return).

## 2. `AGENTS.md` convention

Rename `CLAUDE.md` to `AGENTS.md` (the converged cross-tool filename the
survey names explicitly), and symlink `CLAUDE.md -> AGENTS.md` so Claude
Code keeps reading it under either name. Content is carried over as-is,
plus two new sections:

- **Automation & Verification** — documents `just check` as the sole
  authoritative gate (superseding the current three separately-listed
  commands), states that CI runs the identical command, and points at
  `tests/fixtures/golden/` as the regression safety net.
- **Scheduled Task Scope** — see section 3; this is where the scheduled
  agent's allowed/forbidden file scope is written down, since it's a rule
  that needs to be machine-readable context for that agent's own runs, not
  just a human note.

Not restructuring into directory-level sub-files (survey's "push specifics
to directory-level AGENTS.md files" recommendation) — at one package, 424
tests, single maintainer, the root file is dense but not yet a navigation
problem. Revisit if the file grows meaningfully past its current length.

## 3. Scheduled automation

One weekly Claude Code scheduled routine (via the `schedule` skill/cloud
routine mechanism), operating on a branch, never pushing to `main` directly.

Because waypoint is a library and deliberately does not commit `uv.lock`
("Never commit uv.lock — this is a library, not an application" is an
existing rule), "dependency maintenance" here is not a lockfile bump. The
routine instead:

1. Runs `just check` against currently-latest resolvable versions of the
   core dependencies (numpy, polars, scipy, cvxpy, plotly, pyarrow) to
   surface breakage or new deprecation warnings early.
2. Nudges `pyproject.toml` version floors upward when doing so is clearly
   safe (check passes clean, no new deprecation warnings introduced).
3. Opportunistically fixes ruff/mypy drift — formatting and lint-rule
   updates that are behavior-preserving by construction.
4. Retries and flags (but does not silently delete) flaky tests.

**Explicitly forbidden** from touching:
- `justfile`, `.github/workflows/ci.yml` — the gate itself.
- Anything under `src/waypoint/analysis/methods/` or other core analytics
  logic — behavior changes there require a human-authored change.
- `tests/fixtures/golden/` — it may run against these, never edit them.

Every run ends in a PR using a fixed template — what changed, what
verification ran (the `just check` output), what it was uncertain about —
or ends in no PR at all if there's nothing safe to change. It never merges.

## 4. Review/merge gate

GitHub branch protection on `main`:
- Require the CI status check to pass before merge.
- Require a PR (no direct pushes) — even for self-authored, self-reviewed
  changes.

This is what makes the scheduled agent's boundary real: it physically
cannot land a change without the gate passing and a review moment
happening, regardless of what it decides to attempt.

## 5. Testing additions summary

- `hypothesis` added to the `test` optional-dependency group.
- `tests/fixtures/golden/` — new directory, one fixture set per analytics
  family, loaded by new `test_*_golden.py` files.
- No changes to existing test structure or conventions beyond these
  additions.

## Out of scope (deferred, not decided against)

- Cost/token ceilings and run observability for the scheduled routine —
  add once it's running routinely and there's something to measure.
- Eval-task suite for model/prompt qualification — same reasoning.
- ADR log, structured provenance metadata — team-scale concerns from the
  survey; revisit only if waypoint gains other maintainers.
- Issue-driven or CI-failure-driven autonomous queues (survey Phase 3+) —
  not attempted until Phase 0+2 have run cleanly for a while.
