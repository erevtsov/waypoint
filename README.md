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
        "US Equities": wp.catalog.equities.US_LARGE_CAP,
        "US Bonds":    wp.catalog.fixed_income.US_AGG_BONDS,
    },
    weights={"US Equities": 0.6, "US Bonds": 0.4},
    name="60/40",
)

# --- 2. Expected returns ---------------------------------------------------
er = wp.analytics.ExpectedReturn(method=wp.returns.ArithmeticMean())
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
        wp.cashflows.PeriodicCashflow(amount=2_000, frequency="monthly"),
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
asset = wp.fetch(wp.catalog.equities.US_LARGE_CAP, start="2020-01-01", end="2024-12-31")

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

Pre-defined `AssetDef` and `IndicatorDef` constants organised by asset class. Access via submodule (preferred) or flat on `wp.catalog` (backward compatible):

```python
equities = wp.fetch(wp.catalog.equities.US_LARGE_CAP, start="2020-01-01", end="2024-12-31")
bonds    = wp.fetch(wp.catalog.fixed_income.US_AGG_BONDS, start="2020-01-01", end="2024-12-31")
hpi      = wp.fetch(wp.catalog.real_estate.MA_HPI, start="2010-01-01", end="2024-12-31")
rf       = wp.fetch(wp.catalog.indicators.US_10Y_YIELD, start="2024-01-01", end="2024-12-31")
```

**`wp.catalog.equities`**

| Constant | Description |
|---|---|
| `US_LARGE_CAP` | S&P 500 (`^SPX`, yfinance) |
| `US_TOTAL_MARKET` | US Total Market (`VTI`, yfinance) |
| `US_LARGE_CAP_GROWTH` | US Large Cap Growth (`VUG`, yfinance) |
| `NASDAQ_100` | NASDAQ-100 (`^NDX`, yfinance) |
| `RUSSELL_1000` | Russell 1000 (`^RUI`, yfinance) |
| `US_SMALL_CAP` | Russell 2000 (`^RUT`, yfinance) |
| `US_FINANCIALS` | US Financials (`XLF`, yfinance) |
| `INTL_DEVELOPED` | MSCI EAFE (`EFA`, yfinance) |
| `EUROPE_DEVELOPED` | Europe Developed (`VGK`, yfinance) |
| `EMERGING` | MSCI EM (`EEM`, yfinance) |
| `CHINA_TECH` | China Technology (`CQQQ`, yfinance) |

**`wp.catalog.fixed_income`**

| Constant | Description |
|---|---|
| `US_AGG_BONDS` | Bloomberg Aggregate (`AGG`, yfinance) |
| `US_TIPS` | US TIPS (`TIP`, yfinance) |
| `RISK_FREE_RATE` | 3-Month T-Bill (`DTB3`, FRED, daily) |
| `CPI_YOY` | CPI YoY (`CPIAUCSL`, FRED, monthly) |

**`wp.catalog.real_estate`**

| Constant | Description |
|---|---|
| `MA_HPI` | Massachusetts HPI (`MASTHPI`, FRED, quarterly) |
| `BOSTON_HPI` | Boston Metro HPI (`ATNHPIUS14454Q`, FRED, quarterly) |

**`wp.catalog.indicators`** — `IndicatorDef` (raw levels, not returns)

| Constant | Description |
|---|---|
| `US_10Y_YIELD` | 10-Year Treasury yield (`DGS10`, FRED) |
| `REAL_RATE_10Y` | 10-Year Real Rate (`DFII10`, FRED) |

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
portfolio.expected_return_method = wp.returns.ShrinkageTowardGrandMean()
portfolio.risk_method = wp.risk.LedoitWolf()
```

### Aggregate

An `Aggregate` groups multiple `Portfolio` instances (e.g. brokerage, 401k, Roth IRA) into a single wealth picture. Each portfolio must have `initial_wealth` set.

```python
agg = wp.Aggregate([brokerage, portfolio_401k, roth_ira])

# Relative weight of each account by initial wealth
agg.wealth_weights()   # {"brokerage": 0.25, "401k": 0.65, "roth_ira": 0.10}

# Inspect combined data coverage across all accounts
agg.data_window(start="2015-01-01", end="2024-12-31", frequency="quarterly")

# Collapse all accounts into a single wealth-weighted portfolio
flat = agg.flatten()
```

### Analytics

All analytics live under `wp.analytics` and follow the same pattern: construct with a pluggable method, call `.compute(portfolio, start, end, frequency)`, get an immutable result dataclass back.

#### Expected returns — `wp.analytics.ExpectedReturn`

```python
er = wp.analytics.ExpectedReturn(method=wp.returns.ArithmeticMean())
result = er.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
result.per_asset    # dict[str, float] — annualised per-asset returns
result.portfolio    # float — weighted portfolio return
```

Methods (`wp.returns`):

| Method | Description |
|---|---|
| `ArithmeticMean` | Simple arithmetic mean of historical returns |
| `GeometricMean` | Compound annualised growth rate |
| `EWMAMean` | Exponentially weighted mean (recent observations upweighted) |
| `ShrinkageTowardGrandMean` | James-Stein shrinkage toward the cross-sectional mean |
| `CAPM` | CAPM-implied returns from beta to a market portfolio |
| `ViewReturn` | Forward-looking expected returns specified directly |

`ViewReturn` — forward-looking expected returns specified directly, ignoring history:

```python
# Direct construction (validated at compute time)
portfolio.expected_return_method = wp.returns.ViewReturn(
    expected_returns={"US Equities": 0.07, "US Bonds": 0.03}
)

# Validated at construction time against a specific portfolio
portfolio.expected_return_method = wp.returns.ViewReturn.for_portfolio(
    portfolio, expected_returns={"US Equities": 0.07, "US Bonds": 0.03}
)
```

#### Risk / covariance — `wp.analytics.Risk`

```python
risk = wp.analytics.Risk(method=wp.risk.SampleCovariance())
result = risk.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
result.covariance           # pl.DataFrame — annualised covariance matrix
result.volatilities         # dict[str, float] — per-asset annualised volatility
result.portfolio_volatility # float — sqrt(w^T Σ w)
```

Methods (`wp.risk`):

| Method | Description |
|---|---|
| `SampleCovariance` | Standard sample covariance matrix |
| `LedoitWolf` | Ledoit-Wolf shrinkage covariance (reduces estimation error) |
| `EWMACovariance` | Exponentially weighted covariance (recent observations upweighted) |
| `ViewRisk` | Forward-looking volatilities with historical or manual correlations |

`ViewRisk` — forward-looking risk view: user-specified volatilities combined with a correlation structure (historical or manual):

```python
import numpy as np

# Custom vols + historical correlations from SampleCovariance (default)
portfolio.risk_method = wp.risk.ViewRisk(
    volatilities={"US Equities": 0.15, "US Bonds": 0.06}
)

# Custom vols + fully specified correlation matrix
portfolio.risk_method = wp.risk.ViewRisk(
    volatilities={"US Equities": 0.15, "US Bonds": 0.06},
    correlation_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),  # zero correlation
)

# Validated at construction time against a specific portfolio
portfolio.risk_method = wp.risk.ViewRisk.for_portfolio(
    portfolio,
    volatilities={"US Equities": 0.15, "US Bonds": 0.06},
    correlation_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
)
```

#### Efficient frontier — `wp.analytics.Optimizer`

```python
from waypoint import LongOnly, SumToOne, WeightBounds

opt = wp.analytics.Optimizer(constraints=[LongOnly(), SumToOne()])
frontier = opt.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
frontier.weights_df   # pl.DataFrame of weights along the frontier
frontier.plot()
```

#### Wealth simulation — `wp.analytics.WealthSimulation`

Simulates long-horizon portfolio wealth paths for a single portfolio. Uses the portfolio's `expected_return_method` and `risk_method` to estimate parameters from historical data, then draws `n_simulations` paths.

```python
sim = wp.analytics.WealthSimulation(
    method=wp.sim.MonteCarlo(seed=42),   # or Bootstrap
    horizon_years=30,
    initial_wealth=1_000_000,
    n_simulations=2000,
    inflation_rate=0.03,
    cashflows=[
        wp.cashflows.PeriodicCashflow(amount=-3_000, frequency="monthly"),
        wp.cashflows.LumpSum(amount=100_000, at_year=10.0),
    ],
)
result = sim.compute(portfolio, start="2015-01-01", end="2024-12-31", frequency="monthly")
result.summary()   # {"median_terminal": ..., "p5_terminal": ..., "p95_terminal": ...}
result.plot()      # fan chart
result.plot_allocation()  # stacked area chart of per-asset dollar values
```

Simulation methods (`wp.sim`): `MonteCarlo`, `Bootstrap`

#### Multi-account simulation — `wp.analytics.MultiWealthSimulation`

Simulates wealth across all accounts in an `Aggregate` jointly, preserving cross-asset correlations. Each account has its own cashflow list; all unique assets are drawn from a single correlated return matrix.

```python
sim = wp.analytics.MultiWealthSimulation(
    method=wp.sim.MonteCarlo(seed=42),
    cashflows={
        "brokerage": brokerage_cashflows,
        "401k":      cashflows_401k,
    },
    horizon_years=30,
    n_simulations=2000,
    inflation_rate=0.03,
)
result = sim.compute(agg, start="2015-01-01", end="2024-12-31", frequency="quarterly", real=True)

result.total           # SimulationResult — combined wealth path
result.accounts        # dict[str, SimulationResult] — per-account paths
result.cashflow_schedule  # pl.DataFrame — net annual cashflows by account

result.plot()                  # total wealth fan chart
result.plot_accounts()         # per-account median trajectories
result.plot_cashflow_schedule()  # stacked bar chart of annual cashflows
```

The `cashflow_schedule` follows the `real` flag: real terms when `real=True`, nominal otherwise.

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
    amount=2_000,              # positive = contribution, negative = withdrawal
    frequency="monthly",       # "monthly" or "annual"
    mode="dollar",             # "dollar" or "pct_portfolio"
    real=True,                 # scale by cumulative inflation (dollar mode only)
    effective_tax_rate=0.30,   # gross-up withdrawals for tax (e.g. 401k distribution)
    start_year=22.0,           # first simulation year this cashflow is active
    end_year=30.0,             # last simulation year (None = forever)
    slots=("equities",),       # restrict to specific portfolio slots (None = all)
)

# One-time lump sum
wp.cashflows.LumpSum(
    amount=50_000,
    at_year=10.0,              # fires at simulation year 10
    real=True,                 # scale by cumulative inflation
)
```

For `mode="pct_portfolio"`, `amount` is a fraction of the current portfolio value (e.g. `-0.04` withdraws 4% per year). For `real=True` dollar cashflows, the `amount` is in today's dollars and is inflated each period using the simulation's `inflation_rate`.

### Social Security

```python
# Estimate AIME from a representative salary
aime = wp.social_security.estimate_aime(annual_salary=150_000, career_years=35)

# Full Retirement Age for a given birth year
fra = wp.social_security.full_retirement_age(birth_year=1965)

# Monthly benefit at a specific claiming age
benefit = wp.social_security.monthly_benefit(aime=5_000, birth_year=1965, claim_age=67.0)

# Convert directly to a simulation cashflow (COLA-adjusted real cashflow)
ss_cf = wp.social_security.as_cashflow(
    aime=aime,
    birth_year=1965,
    claim_age=67.0,
    real=True,          # COLA-adjusted
    start_year=31.0,    # simulation year when SS starts
)
```

### Data fetching

```python
# Fetch an asset (returns Asset with pl.DataFrame[date, returns])
asset = wp.fetch(wp.catalog.equities.US_LARGE_CAP, start="2020-01-01", end="2024-12-31")

# Fetch an indicator (returns Indicator with pl.DataFrame[date, value] — raw levels)
indicator = wp.fetch(wp.catalog.indicators.US_10Y_YIELD, start="2024-01-01", end="2024-12-31")
risk_free_rate = float(indicator.values["value"].tail(1).item()) / 100

# Force refresh bypasses the local parquet cache
asset = wp.fetch(wp.catalog.equities.US_LARGE_CAP, start="2020-01-01", end="2024-12-31", force_refresh=True)
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
