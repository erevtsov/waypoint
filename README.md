# waypoint

A Python library for financial portfolio analysis, focused on returns-based time series.

All internal data is **decimal periodic returns** (0.01 = 1%) — never raw prices. The library is structured around three actors: **Assets** (data), **Portfolio** (collection + weights), and **Analytics** (computation).

## Installation

```bash
uv add waypoint
```

Optional extras include vendor data providers:

```bash
uv add "waypoint[dev]"   # all dev dependencies + vendor SDKs
```

## Quick start

```python
import waypoint as wp

# --- 1. Define a portfolio -------------------------------------------------
portfolio = wp.Portfolio(
    slots={
        "US Equities": wp.catalog.US_LARGE_CAP,
        "US Bonds":    wp.catalog.US_AGG_BONDS,
    },
    weights={"US Equities": 0.6, "US Bonds": 0.4},
    name="60/40",
)

# --- 2. Expected returns ---------------------------------------------------
er = wp.analytics.ExpectedReturn(method=wp.returns.HistoricalMean())
result = er.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
print(result.portfolio)          # annualised portfolio return

# --- 3. Risk / covariance -------------------------------------------------
risk = wp.analytics.Risk(method=wp.risk.SampleCovariance())
r = risk.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
print(r.portfolio_volatility)    # annualised portfolio volatility

# --- 4. Efficient frontier optimizer -------------------------------------
opt = wp.analytics.Optimizer(constraints=[wp.LongOnly(), wp.SumToOne()])
frontier = opt.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
frontier.plot()

# --- 5. Wealth simulation -------------------------------------------------
sim = wp.analytics.WealthSimulation(
    method=wp.sim.MonteCarlo(seed=42),
    horizon_years=30,
    initial_wealth=500_000,
    n_simulations=2000,
    cashflows=[
        wp.cashflows.PeriodicCashflow(amount=2_000, frequency="monthly", mode="dollar"),
    ],
)
result = sim.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
result.plot()
```

## Core concepts

### Assets

An `Asset` holds a `pl.DataFrame[date, returns]` of decimal periodic returns. Assets can be loaded from a vendor or constructed directly.

```python
# Fetch from a vendor (cached locally as parquet)
asset = wp.fetch(wp.catalog.US_LARGE_CAP, start="2020-01-01", end="2024-12-31")

# Construct directly
import polars as pl
from datetime import date
asset = wp.Asset(
    name="My Fund", ticker="XYZ",
    returns=pl.DataFrame({"date": [...], "returns": [...]}),
    frequency="daily",
)
```

A `LeveragedAsset` wraps an `Asset` and applies a constant-leverage return transformation — useful for modelling mortgaged real estate or margin accounts:

```python
leveraged = wp.LeveragedAsset(
    asset=underlying,
    leverage_ratio=1.5,    # 50% borrowed
    financing_cost=0.065,  # 6.5% annual financing rate
)
```

### Catalog

Pre-defined `AssetDef` constants for common instruments:

| Constant | Description |
|---|---|
| `wp.catalog.US_LARGE_CAP` | S&P 500 (`^SPX`, yfinance) |
| `wp.catalog.US_SMALL_CAP` | Russell 2000 (`^RUT`, yfinance) |
| `wp.catalog.INTL_DEVELOPED` | MSCI EAFE (`EFA`, yfinance) |
| `wp.catalog.EMERGING` | MSCI EM (`EEM`, yfinance) |
| `wp.catalog.US_AGG_BONDS` | Bloomberg Aggregate (`AGG`, yfinance) |
| `wp.catalog.US_TIPS` | US TIPS (`TIP`, yfinance) |
| `wp.catalog.MA_HPI` | Massachusetts HPI (`MASTHPI`, FRED, quarterly) |
| `wp.catalog.BOSTON_HPI` | Boston Metro HPI (`ATNHPIUS14454Q`, FRED, quarterly) |
| `wp.catalog.US_10Y_YIELD` | 10-Year Treasury yield (`DGS10`, FRED — indicator, not return) |

### Portfolio

A `Portfolio` holds named asset slots and a weight vector.

```python
portfolio = wp.Portfolio(
    slots={"Equities": asset_eq, "Bonds": asset_fi},
    weights={"Equities": 0.6, "Bonds": 0.4},
)

# Wide DataFrame of per-asset returns aligned on date
wide = portfolio.get_returns(start="2020-01-01", end="2024-12-31", frequency="monthly")

# Portfolio-level return series
port_returns = portfolio.portfolio_returns(frequency="monthly")
```

**Mixed-frequency portfolios** are supported — e.g. daily equities + quarterly real-estate indices. The `native_frequency` property reports the coarsest frequency across all slots; requesting a finer frequency raises a `ValueError`.

**Configurable estimation methods** — analytics that need return/risk estimates (e.g. `WealthSimulation`) use the portfolio's configured methods:

```python
portfolio.expected_return_method = wp.returns.HistoricalMean()
portfolio.risk_method = wp.risk.SampleCovariance()
```

### Analytics

All analytics live under `wp.analytics` and follow the same pattern: construct with a pluggable method, call `.compute(portfolio, start, end, frequency)`, get an immutable result dataclass back.

#### Expected returns — `wp.analytics.ExpectedReturn`

```python
er = wp.analytics.ExpectedReturn(method=wp.returns.HistoricalMean())
result = er.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
result.per_asset    # dict[str, float] — annualised per-asset returns
result.portfolio    # float — weighted portfolio return
```

Methods (`wp.returns`): `HistoricalMean`

#### Risk / covariance — `wp.analytics.Risk`

```python
risk = wp.analytics.Risk(method=wp.risk.SampleCovariance())
result = risk.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
result.covariance           # pl.DataFrame — annualised covariance matrix
result.volatilities         # dict[str, float] — per-asset annualised volatility
result.portfolio_volatility # float — sqrt(w^T Σ w)
```

Methods (`wp.risk`): `SampleCovariance`

#### Efficient frontier — `wp.analytics.Optimizer`

```python
from waypoint import LongOnly, SumToOne, WeightBounds

opt = wp.analytics.Optimizer(constraints=[LongOnly(), SumToOne()])
frontier = opt.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
frontier.weights_df   # pl.DataFrame of weights along the frontier
frontier.plot()
```

#### Wealth simulation — `wp.analytics.WealthSimulation`

Simulates long-horizon portfolio wealth paths. Uses the portfolio's `expected_return_method` and `risk_method` to estimate parameters from historical data, then draws `n_simulations` paths.

```python
sim = wp.analytics.WealthSimulation(
    method=wp.sim.MonteCarlo(seed=42),   # or Bootstrap
    horizon_years=30,
    initial_wealth=1_000_000,
    n_simulations=2000,
    inflation_rate=0.03,
    cashflows=[
        wp.cashflows.PeriodicCashflow(amount=-3_000, frequency="monthly", mode="dollar"),
        wp.cashflows.LumpSum(amount=100_000, period=60),
    ],
)
result = sim.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
result.summary()   # {"median_terminal": ..., "p5_terminal": ..., "p95_terminal": ...}
result.plot()      # fan chart
```

Simulation methods (`wp.sim`): `MonteCarlo`, `Bootstrap`

#### Scenario comparison — `wp.analytics.ComparisonResult`

Compare multiple simulation results side by side:

```python
comparison = wp.analytics.ComparisonResult.from_scenarios({
    "Keep": keep_result,
    "Sell": sell_result,
})
comparison.plot()
```

### Cash flows

```python
# Recurring cash flow
wp.cashflows.PeriodicCashflow(
    amount=2_000,           # positive = contribution, negative = withdrawal
    frequency="monthly",    # "monthly" or "annual"
    mode="dollar",          # "dollar", "pct_portfolio", "pct_portfolio_inflation_adjusted"
    inflation_rate=0.03,    # grows dollar amount at 3%/year
)

# One-time lump sum at period t
wp.cashflows.LumpSum(amount=50_000, period=120)  # period 120 = month 120 = year 10
```

### Data fetching

```python
# Fetch an asset (returns Asset with pl.DataFrame[date, returns])
asset = wp.fetch(wp.catalog.US_LARGE_CAP, start="2020-01-01", end="2024-12-31")

# Fetch an indicator (returns Indicator with pl.DataFrame[date, value] — raw levels)
indicator = wp.fetch(wp.catalog.US_10Y_YIELD, start="2024-01-01", end="2024-12-31")
risk_free_rate = float(indicator.values["value"].tail(1).item()) / 100

# Force refresh bypasses the local parquet cache
asset = wp.fetch(wp.catalog.US_LARGE_CAP, start="2020-01-01", end="2024-12-31", force_refresh=True)
```

Data is cached locally as parquet files keyed by `{vendor}/{symbol}.parquet`. For daily instruments, the date range is snapped to full calendar months.

## Development

```bash
git clone https://github.com/erevtsov/waypoint
cd waypoint
uv sync --extra dev
```

```bash
uv run pytest                        # tests + coverage
uv run ruff check src/ tests/        # lint
uv run mypy                          # type-check
uv build                             # build sdist + wheel
```

## License

MIT — see [LICENSE](LICENSE).
