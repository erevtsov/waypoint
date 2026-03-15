# Waypoint

## Project Overview
- Python library for financial portfolio analysis focused on returns-based time series
- Installable package — not an application; no CLI, no server
- All internal data is decimal periodic returns (0.01 = 1%); never raw prices
- Analysis functions are stateless; accept domain objects (Asset, Portfolio) or bare Series

## Commands
- `uv sync --extra dev` — install all dev dependencies
- `uv run pytest` — run tests with coverage
- `uv run ruff check src/ tests/` — lint
- `uv run mypy` — type-check
- `uv build` — build sdist + wheel

## Architecture
- `src/waypoint/` — package root; `__init__.py` exposes `__version__` only
- `src/waypoint/analysis/` — stateless functions operating on `pl.Series` or domain objects
- `tests/` — mirrors src structure: `src/waypoint/foo.py` → `tests/test_foo.py`

## Conventions
- Always use decimal returns, never percentages or prices
- Always require a date index on any return series — enforce in `__post_init__`
- Always add analysis as free functions in `analysis/`; never as domain object methods
- Analysis functions accept domain objects (Asset, Portfolio) or `pl.Series` directly
- Use polars for all time series and tabular data; never pandas
- Use `importlib.metadata.version("waypoint")` for `__version__` — never hardcode it
- Use `np.random.default_rng(seed=42)` for reproducible test data
- Prefer explicit over implicit — no clever one-liners that obscure intent
- Never introduce a TODO without an inline explanation
- Never hardcode magic numbers — name constants explicitly
- No commented-out code in commits

## Testing Rules
- Always write or update tests when adding or modifying any feature
- Always write or update tests when fixing a bug — test must fail before the fix
- Never mark a task complete without running the full test suite
- Never mock what you can fake with real in-memory data
- Never delete or skip existing tests without an explicit comment explaining why

## Code Quality Rules
- Always use conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`)
- Run lint and typecheck before declaring any task done
- Never use bare `except` without logging and a comment
- Each function should do one thing; decompose anything over ~30 lines

## Agentic Behavior Rules
- Before deleting or overwriting any file, confirm with the user
- Before making changes across more than 3 files, present a plan and wait
- When uncertain between two approaches, present both with tradeoffs — don't pick
- Never install new dependencies without asking first
- Never modify this file during a task unless explicitly asked

## Never Do
- Never store prices internally — the library starts at returns
- Never use pandas — use polars instead
- Never add matplotlib, plotly, or data-fetching libs to core dependencies
- Never commit `uv.lock` — this is a library, not an application
