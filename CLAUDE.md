# Waypoint

## Project Overview
- Python library for financial portfolio analysis focused on returns-based time series
- Installable package — not an application; no CLI, no server
- All internal data is decimal periodic returns (0.01 = 1%); never raw prices
- Three main actors: **Assets** (data), **Portfolio** (collection + weights), **Analytics** (computation)

## Commands
- `uv sync --extra dev` — install all dev dependencies (includes vendor SDKs)
- `uv run pytest` — run tests with coverage
- `uv run ruff check src/ tests/` — lint
- `uv run mypy` — type-check
- `uv build` — build sdist + wheel

## Architecture
- `src/waypoint/` — package root; `__init__.py` exposes `__version__` only
- `src/waypoint/asset_def.py` — `AssetDef` frozen dataclass (name, symbol, vendor, frequency, metadata)
- `src/waypoint/assets.py` — `Asset` dataclass; `PERIODS_PER_YEAR` constant dict
- `src/waypoint/portfolio.py` — `Portfolio` (holds `AssetDef | Asset` per slot + weights; mutable data cache)
- `src/waypoint/factors.py` — `Factor` (lazy derived return series; inputs + polars transform callable)
- `src/waypoint/catalog.py` — built-in `AssetDef` constants (SPY, AGG, FRED series, etc.)
- `src/waypoint/cashflows.py` — `CashflowDefinition` protocol + `PeriodicCashflow`, `LumpSum`
- `src/waypoint/constraints.py` — user-friendly constraint objects + cvxpy translation layer
- `src/waypoint/data/` — fetch API + parquet cache + vendor providers
  - `data/__init__.py` — exposes `fetch(asset_def, start, end, force_refresh=False)`
  - `data/cache.py` — parquet read/write; cache key = `{vendor}/{symbol}.parquet`
  - `data/normalize.py` — raw prices → `pl.DataFrame[date, returns]` (date column always preserved)
  - `data/providers/base.py` — `Provider` Protocol
  - `data/providers/{yfinance,fred,eodhd}.py` — one file per vendor
- `src/waypoint/analysis/` — all analytics
  - `analysis/methods/returns.py` — pure return estimation functions (`HistoricalMean`, `EWMAMean`, …)
  - `analysis/methods/risk.py` — pure covariance functions (`SampleCovariance`, `LedoitWolf`, …)
  - `analysis/methods/simulation.py` — pure simulation engines (`MonteCarlo`, `Bootstrap`)
  - `analysis/expected_return.py` — `ExpectedReturn` analytic (orchestrates methods/)
  - `analysis/risk.py` — `Risk` analytic
  - `analysis/optimizer.py` — `Optimizer` → `EfficientFrontierResult` (uses cvxpy via constraints.py)
  - `analysis/simulation.py` — `WealthSimulation` → `SimulationResult`
  - `analysis/summary.py` — `SummaryStats` → `SummaryResult`
  - `analysis/viz.py` — all free-function visualizations (plotly)
- `tests/` — mirrors src structure: `src/waypoint/foo.py` → `tests/test_foo.py`

## Factor design
- `Factor(name, inputs, transform)` — lazy; stores the logic, not the data
- `inputs: dict[str, Asset | AssetDef | Factor]` — all three are valid; Factor is composable (factor-of-factors)
- `factor.get_returns(start, end)` — fetches/filters each input for the range, inner-joins into a wide `pl.DataFrame` with one column per input key, calls `transform(wide_df)`, validates output
- `transform` is a plain callable `(pl.DataFrame) -> pl.DataFrame[date, returns]` — user writes pure polars; the wide DataFrame has `"date"` + one column per input key
- Output must be `pl.DataFrame` with columns `["date"(pl.Date), "returns"(pl.Float64)]` — validated on first call
- `Asset.get_returns(start, end)` filters `self.returns` to the requested range — uniform interface across `Asset`, `AssetDef`, and `Factor`

## Portfolio design
- Holds a named slot → `AssetDef | Asset` mapping plus a weight vector
- Definitions-first is the common case: pass `AssetDef`s, analytics fetch data on demand per date range
- Asset-first: pass pre-loaded `Asset`s, analytics use the data as-is (no date params required)
- `portfolio.get_returns(start, end)` — fetches if slots are `AssetDef`, filters if slots are `Asset`; result cached in memory on the Portfolio instance (mutable internal cache, frozen definitions/weights)
- Weights are normalised to sum to 1.0 on construction

## Analytics design
- **Two layers**: pure math functions in `analysis/methods/` (no domain objects, just arrays/DataFrames) and analytic orchestrator classes in `analysis/` (extract inputs from Portfolio, call methods/, return Results)
- **Method protocol** (Strategy pattern): each analytic family has a `Method` Protocol; pass the method at construction time — `ExpectedReturn(method=HistoricalMean())`
- **Results are immutable frozen dataclasses** — re-run the analytic with different params to produce a new result
- **Visualization**: free functions in `analysis/viz.py` (stateless, fully testable); each Result has a `.plot()` method that delegates to the relevant free function as the default view
- **Optimizer**: uses cvxpy; `constraints.py` owns the translation from user-friendly objects to cvxpy constraints
- **Visualization library**: plotly (optional extra); matplotlib is not used

## Conventions
- Always use decimal returns, never percentages or prices
- Always add analysis as free functions in `analysis/`; never as domain object methods
- `Asset.returns` is a `pl.DataFrame` with columns `["date"(pl.Date), "returns"(pl.Float64)]` — never a bare Series
- Combine assets by joining on `"date"`: `asset1.returns.join(asset2.returns, on="date", how="inner")`
- Use polars for all time series and tabular data; never pandas
- Use `importlib.metadata.version("waypoint")` for `__version__` — never hardcode it
- Use `np.random.default_rng(seed=42)` for reproducible test data
- Vendor SDKs (yfinance, fredapi, eodhd) live in optional extras, never in `[project.dependencies]`
- plotly and cvxpy also live in optional extras
- For daily instruments, `fetch()` snaps date range to full calendar months — Asset.returns may cover a wider range than requested
- Cache key includes vendor: changing `AssetDef.vendor` starts a fresh cache
- `force_refresh=True` on `fetch()` bypasses cache and overwrites with fresh vendor data
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
- Never add vendor SDKs to `[project.dependencies]` — use optional extras
- Never commit `uv.lock` — this is a library, not an application
