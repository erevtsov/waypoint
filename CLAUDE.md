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
- `src/waypoint/` — package root; `__init__.py` exposes the full public API for `import waypoint as wp`
- `src/waypoint/enums.py` — `Frequency(StrEnum)`, `CashflowMode(StrEnum)`, `PERIODS_PER_YEAR` dict
- `src/waypoint/asset_def.py` — `AssetDef` frozen dataclass (definition of a return-series instrument)
- `src/waypoint/indicator_def.py` — `IndicatorDef` frozen dataclass (definition of a level/rate series)
- `src/waypoint/assets.py` — `Asset` dataclass (holds `pl.DataFrame[date, returns]`)
- `src/waypoint/indicators.py` — `Indicator` dataclass (holds `pl.DataFrame[date, value]` — raw levels, no pct_change)
- `src/waypoint/portfolio.py` — `Portfolio` (holds `AssetDef | Asset` per slot + weights; mutable data cache)
- `src/waypoint/factors.py` — `Factor` (lazy derived return series; inputs + polars transform callable)
- `src/waypoint/catalog.py` — built-in `AssetDef` and `IndicatorDef` constants (SPY, AGG, DGS10, etc.)
- `src/waypoint/cashflows.py` — `CashflowDefinition` protocol + `PeriodicCashflow`, `LumpSum`
- `src/waypoint/constraints.py` — user-friendly constraint objects + cvxpy translation layer
- `src/waypoint/data/` — fetch API + parquet cache + vendor providers
  - `data/__init__.py` — exposes `fetch(instrument, start, end, force_refresh=False)`; returns `Asset` for `AssetDef`, `Indicator` for `IndicatorDef`
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
- `portfolio.expected_return_method` and `portfolio.risk_method` — mutable properties (default `HistoricalMean()` / `SampleCovariance()`); also settable post-construction
- `native_frequency` — coarsest native frequency across all slots; `get_returns(frequency=X)` raises if `X` is finer than `native_frequency`

## Analytics design
- **Two layers**: pure math functions in `analysis/methods/` (no domain objects, just arrays/DataFrames) and analytic orchestrator classes in `analysis/` (extract inputs from Portfolio, call methods/, return Results)
- **Method protocol** (Strategy pattern): each analytic family has a `Method` Protocol; pass the method at construction time — `ExpectedReturn(method=HistoricalMean())`
- **Results are immutable frozen dataclasses** — re-run the analytic with different params to produce a new result
- **Visualization**: free functions in `analysis/viz.py` (stateless, fully testable); each Result has a `.plot()` method that delegates to the relevant free function as the default view
- **Optimizer**: uses cvxpy; `constraints.py` owns the translation from user-friendly objects to cvxpy constraints
- **Visualization library**: plotly (optional extra); matplotlib is not used
- **Portfolio method convention**: analytics that need return/risk estimates (e.g. `WealthSimulation`) must read `portfolio.expected_return_method` / `portfolio.risk_method` and pass them to `ExpectedReturn` / `Risk` — never compute `np.mean`/`np.var` inline. Exception: `ExpectedReturn` and `Risk` themselves always use their own explicitly-passed `method` and do not read portfolio properties.
- **Per-period conversion**: annualised results from `ExpectedReturn`/`Risk` must be converted to per-period before passing to simulation: `mu_per_period = er.portfolio / ppy`, `var_per_period = risk.portfolio_volatility**2 / ppy`

## Enums
- Use `StrEnum` for any field with a fixed set of string values (`Frequency`, `CashflowMode`)
- All public APIs that take a frequency or mode accept `TheEnum | str` — normalise to the enum via `TheEnum(value)` in `__post_init__` (frozen dataclasses use `object.__setattr__`)
- `StrEnum` members compare equal to their lowercase string counterparts, so existing `== "daily"` checks keep working
- `PERIODS_PER_YEAR` in `enums.py` is keyed by `Frequency`; plain-string lookup works due to `StrEnum` hash equality
- `Indicator` vs `Asset`: use `AssetDef`/`Asset` for return series (pct_change applied); use `IndicatorDef`/`Indicator` for level/rate series (raw values, no pct_change)

## Notebooks
- Always use `import waypoint as wp` as the sole top-level import; never import from submodules directly
- Access catalog constants as `wp.catalog.SPY`, domain objects as `wp.Portfolio`, analytics as `wp.analytics.WealthSimulation`, simulation methods as `wp.sim.MonteCarlo`, cashflows as `wp.cashflows.PeriodicCashflow`, etc.

## Conventions
- Always use decimal returns, never percentages or prices
- Always add analysis as free functions in `analysis/`; never as domain object methods
- `Asset.returns` is a `pl.DataFrame` with columns `["date"(pl.Date), "returns"(pl.Float64)]` — never a bare Series
- Combine assets by joining on `"date"`: `asset1.returns.join(asset2.returns, on="date", how="inner")`
- Use polars for all time series and tabular data; never pandas
- Use `importlib.metadata.version("waypoint")` for `__version__` — never hardcode it
- Use `np.random.default_rng(seed=42)` for reproducible test data
- Vendor SDKs (yfinance, fredapi, eodhd) live in optional extras, never in `[project.dependencies]`
- plotly and cvxpy are core dependencies (in `[project.dependencies]`)
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

## README sync
- Keep `README.md` up to date whenever public API surface changes: new analytics, new catalog entries, new cashflow types, new simulation methods, new Portfolio properties, or changes to the `wp.*` namespace
- README documents the public `import waypoint as wp` API only — not internals or test helpers

## Never Do
- Never store prices internally — the library starts at returns
- Never use pandas — use polars instead
- Never add vendor SDKs to `[project.dependencies]` — use optional extras
- Never commit `uv.lock` — this is a library, not an application
