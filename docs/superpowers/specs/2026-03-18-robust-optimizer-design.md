# Robust Optimizer Design

**Date:** 2026-03-18
**Status:** Approved

## Problem

The current `Optimizer` uses classical mean-variance (MV): it minimises portfolio variance for a fixed return target using point estimates of μ (expected returns) and Σ (covariance). Two failure modes motivate this work:

- **Concentration** — the optimizer over-bets on assets with the highest sample mean, producing portfolios with 90–100% in one or two assets.
- **Instability** — small changes to the date window or estimation method cause large swings in frontier weights.

Both symptoms trace to sensitivity in μ estimates. Classical MV treats μ as known exactly; in practice it is estimated from limited history and is noisy.

## Approach: Ellipsoidal Uncertainty Robust MV

Replace the classical `μᵀw ≥ target` return constraint with a worst-case version over a covariance-scaled ellipsoidal uncertainty set around μ:

```
worst-case return = μᵀw − κ · ‖Σ^(1/2) w‖₂
```

The full problem remains: minimise `wᵀΣw` subject to the worst-case return constraint and all user constraints. This is a second-order cone program (SOCP), handled natively by CLARABEL (already the project's solver).

**Why this formulation:**
- κ = 0 recovers classical MV exactly — `ClassicalMV` becomes a degenerate case
- The uncertainty set is scaled by Σ, so assets with high estimation variance (high covariance) are penalised more
- No extra inputs required beyond a κ parameter; an `"auto"` mode derives κ from the data

## Architecture

### Strategy pattern on `Optimizer`

`Optimizer` gains one new optional field: `solve_method`, defaulting to `ClassicalMV()`. The field accepts any object satisfying the `OptimizationMethod` protocol. Constraints, `return_model`, and `risk_model` remain on `Optimizer` and flow through identically regardless of `solve_method`.

```python
opt = wp.analytics.Optimizer(
    return_model=wp.analytics.ExpectedReturn(method=wp.returns.ShrinkageTowardGrandMean()),
    risk_model=wp.analytics.Risk(method=wp.risk.LedoitWolf()),
    constraints=[wp.LongOnly(), wp.SumToOne()],
    solve_method=wp.opt.RobustMV(kappa=0.1),   # or ClassicalMV() (default)
)
frontier = opt.efficient_frontier(portfolio, start, end, frequency="quarterly")
```

### `OptimizationMethod` protocol

```python
class OptimizationMethod(Protocol):
    def solve_one_point(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        constraints: list[Constraint],
        asset_names: list[str],
        target_return: float,
    ) -> SolvePointResult:
        ...
```

`Optimizer.efficient_frontier` calls `solve_one_point` for each return target in the linspace loop. The extreme-return probing (`_solve_extreme_return`) is unchanged — it depends only on constraints, not on the solve method.

### `SolvePointResult`

```python
@dataclass(frozen=True)
class SolvePointResult:
    feasible: bool
    weights: np.ndarray | None   # None when infeasible
    ret: float | None
    risk: float | None
    reason: str | None           # human-readable explanation when infeasible
```

### `ClassicalMV`

Extracts the current inner-loop logic from `Optimizer.efficient_frontier` verbatim. Zero behaviour change. Exposed as `wp.opt.ClassicalMV`.

### `RobustMV`

```python
@dataclass(frozen=True)
class RobustMV:
    kappa: float | Literal["auto"] = 0.1
```

When `kappa="auto"`, κ is computed as `chi(p, 0.95) / sqrt(T)` — the 95%-confidence radius of the estimation error ellipsoid given T observations of p assets, where `chi(p, 0.95)` is the square root of the 95th percentile of the χ²(p) distribution.

The SOCP constraint added per frontier point:

```
μᵀw − κ · ‖Σ^(1/2) w‖₂  ≥  target_return
```

expressed in cvxpy as:

```python
cp.SOC(mu @ w - target_return, kappa * sigma_half @ w)
```

where `sigma_half` is the Cholesky factor of `sigma`. Exposed as `wp.opt.RobustMV`.

### Infeasibility feedback

When `solve_one_point` returns `feasible=False`:

1. `Optimizer.efficient_frontier` emits `warnings.warn(reason, stacklevel=2)` — visible in notebooks by default, suppressable.
2. The `(target_return, reason)` pair is appended to `infeasible_points` on `EfficientFrontierResult`.

`EfficientFrontierResult` additions:

```python
infeasible_points: list[tuple[float, str]] = field(default_factory=list)

def print_diagnostics(self) -> None:
    """Print a summary of skipped frontier points with their infeasibility reasons."""
```

Existing `EfficientFrontierResult` fields (`weights`, `expected_returns`, `risks`, `asset_names`) and methods (`optimal_sharpe`, `portfolio_at`, `min_volatility_portfolio`, `max_sharpe_portfolio`, `plot`) are unchanged.

## Public namespace

New file `src/waypoint/opt.py` following the pattern of `sim.py`, `returns.py`, `risk.py`:

```python
from waypoint.analysis.optimizer import ClassicalMV, RobustMV
__all__ = ["ClassicalMV", "RobustMV"]
```

`wp.opt` added to `src/waypoint/__init__.py` imports and `__all__`.

## Files changed

| File | Change |
|---|---|
| `src/waypoint/analysis/optimizer.py` | Add `SolvePointResult`, `OptimizationMethod` protocol, `ClassicalMV`, `RobustMV`; update `Optimizer` and `EfficientFrontierResult` |
| `src/waypoint/opt.py` | New re-export module |
| `src/waypoint/__init__.py` | Add `opt` to imports and `__all__` |
| `tests/test_robust_optimizer.py` | New test file |
| `README.md` | Document `wp.opt`, `ClassicalMV`, `RobustMV`, infeasibility fields |

## Testing

- `ClassicalMV` produces identical results to current `Optimizer` (no `solve_method`) — regression test on a known portfolio
- `RobustMV` produces lower max-weight concentration than `ClassicalMV` on the same portfolio
- `RobustMV` with `kappa=0` matches `ClassicalMV` output
- `RobustMV(kappa="auto")` runs without error and returns a valid frontier
- Infeasible points appear in `result.infeasible_points` and trigger `warnings.warn`
- `print_diagnostics()` prints to stdout without error

## Non-goals

- No changes to `Constraints`, `ExpectedReturn`, `Risk`, or any result type other than `EfficientFrontierResult`
- No changes to `WealthSimulation` or `MultiWealthSimulation`
- No Black-Litterman or Michaud resampling (separate future work if needed)
