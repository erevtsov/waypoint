# Robust Optimizer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strategy pattern to `Optimizer` with `ClassicalMV` (exact current behaviour) and `RobustMV` (SOCP ellipsoidal-uncertainty formulation) as pluggable solve methods.

**Architecture:** `OptimizationMethod` protocol with a single `solve_one_point` method is added to `optimizer.py`. `Optimizer` gains a `solve_method` field defaulting to `ClassicalMV()`. The current inner-loop logic moves verbatim into `ClassicalMV.solve_one_point`; `RobustMV.solve_one_point` replaces the linear return constraint with a CLARABEL SOCP cone constraint. `EfficientFrontierResult` gains an `infeasible_points` list and a `print_diagnostics()` method.

**Tech Stack:** Python 3.12, cvxpy (CLARABEL solver), numpy, scipy (chi2 quantile for `kappa="auto"`), polars

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/waypoint/analysis/optimizer.py` | Modify | Add `SolvePointResult`, `OptimizationMethod`, `ClassicalMV`, `RobustMV`; refactor `Optimizer.efficient_frontier`; extend `EfficientFrontierResult` |
| `src/waypoint/opt.py` | Create | Re-export module: `ClassicalMV`, `RobustMV`, `OptimizationMethod`, `SolvePointResult` |
| `src/waypoint/__init__.py` | Modify | Add `opt` sub-module to imports and `__all__` |
| `tests/test_robust_optimizer.py` | Create | All tests for the new functionality |
| `README.md` | Modify | Document `wp.opt`, `ClassicalMV`, `RobustMV`, `infeasible_points`, `print_diagnostics` |

---

## Task 1: Add `SolvePointResult` and `OptimizationMethod` protocol

**Files:**
- Modify: `src/waypoint/analysis/optimizer.py` (add near top, after imports)
- Test: `tests/test_robust_optimizer.py` (create)

Context: Open `src/waypoint/analysis/optimizer.py` and read it fully before editing. Also check `src/waypoint/constraints.py` to see how `@runtime_checkable` is used there.

- [ ] **Step 1: Write the failing test**

Create `tests/test_robust_optimizer.py` with:

```python
"""Tests for the robust optimizer strategy pattern."""

from __future__ import annotations

import numpy as np
import pytest

from waypoint.analysis.optimizer import (
    ClassicalMV,
    OptimizationMethod,
    RobustMV,
    SolvePointResult,
)
from waypoint.constraints import LongOnly, SumToOne


def test_solve_point_result_feasible():
    w = np.array([0.5, 0.5])
    r = SolvePointResult(feasible=True, weights=w, ret=0.08, risk=0.12, reason=None)
    assert r.feasible is True
    assert r.reason is None
    assert r.weights is not None
    np.testing.assert_array_equal(r.weights, w)


def test_solve_point_result_infeasible():
    r = SolvePointResult(feasible=False, weights=None, ret=None, risk=None, reason="no solution")
    assert r.feasible is False
    assert r.weights is None
    assert r.reason == "no solution"


def test_optimization_method_protocol():
    """ClassicalMV and RobustMV satisfy the OptimizationMethod protocol."""
    assert isinstance(ClassicalMV(), OptimizationMethod)
    assert isinstance(RobustMV(), OptimizationMethod)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest tests/test_robust_optimizer.py -v
```

Expected: ImportError — `ClassicalMV`, `OptimizationMethod`, `RobustMV`, `SolvePointResult` don't exist yet.

- [ ] **Step 3: Add `SolvePointResult` and `OptimizationMethod` to optimizer.py**

In `src/waypoint/analysis/optimizer.py`, add these imports and classes after the existing imports and before the `EfficientFrontierResult` class. Also add `field` to the `dataclasses` import and add `Literal` to the `typing` import.

New imports to add:
```python
import warnings
from typing import TYPE_CHECKING, Any, Literal
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
```

(Merge with existing imports — don't duplicate. The file already imports `dataclass` and `Any`; add `field`, `Literal`, `Protocol`, `runtime_checkable`, `warnings`.)

New classes to add (insert before `EfficientFrontierResult`):

```python
@dataclass(frozen=True)
class SolvePointResult:
    """Result of solving a single efficient frontier point."""

    feasible: bool
    weights: np.ndarray | None  # None when infeasible
    ret: float | None
    risk: float | None
    reason: str | None  # None when feasible; human-readable when infeasible


@runtime_checkable
class OptimizationMethod(Protocol):
    """Protocol for pluggable solve strategies used by Optimizer."""

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

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest tests/test_robust_optimizer.py::test_solve_point_result_feasible tests/test_robust_optimizer.py::test_solve_point_result_infeasible -v
```

Expected: 2 PASSED. (`test_optimization_method_protocol` is not run here — `ClassicalMV` and `RobustMV` don't exist yet, so that test would fail. It will be verified in Task 2.)

- [ ] **Step 5: Commit**

```bash
cd /Users/erevtsov/dev/waypoint && git add src/waypoint/analysis/optimizer.py tests/test_robust_optimizer.py && git commit -m "feat: add SolvePointResult and OptimizationMethod protocol"
```

---

## Task 2: Implement `ClassicalMV`

**Files:**
- Modify: `src/waypoint/analysis/optimizer.py` (add `ClassicalMV` class)
- Test: `tests/test_robust_optimizer.py` (add tests)

Context: `ClassicalMV` extracts the existing inner-loop logic from `Optimizer.efficient_frontier` verbatim. Look at lines 213–240 in the current `optimizer.py` — that's the code to move.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_robust_optimizer.py`:

```python
import datetime
from waypoint.assets import Asset
from waypoint.portfolio import Portfolio
from waypoint.asset_def import AssetDef
from waypoint.analysis.optimizer import Optimizer
from waypoint.analysis.expected_return import ExpectedReturn
from waypoint.analysis.risk import Risk
from waypoint.constraints import LongOnly, SumToOne
import polars as pl


def _make_test_portfolio() -> tuple[Portfolio, Asset, Asset]:
    """Two synthetic assets with known properties for deterministic tests."""
    rng = np.random.default_rng(seed=42)
    n = 60  # 60 quarterly observations = 15 years

    # Asset A: higher return, higher risk
    r_a = rng.normal(0.03, 0.08, n)
    # Asset B: lower return, lower risk, slight negative correlation with A
    r_b = rng.normal(0.015, 0.04, n) - 0.3 * r_a

    dates = pl.date_range(
        datetime.date(2010, 1, 1),
        datetime.date(2010, 1, 1) + datetime.timedelta(days=n * 90),
        interval="90d",
        eager=True,
    ).head(n)

    asset_a = Asset(
        definition=AssetDef(symbol="A", vendor="test"),
        returns=pl.DataFrame({"date": dates, "returns": r_a}),
    )
    asset_b = Asset(
        definition=AssetDef(symbol="B", vendor="test"),
        returns=pl.DataFrame({"date": dates, "returns": r_b}),
    )
    portfolio = Portfolio(
        slots={"A": asset_a, "B": asset_b},
        weights={"A": 0.5, "B": 0.5},
    )
    return portfolio, asset_a, asset_b


def test_classical_mv_matches_legacy_optimizer():
    """ClassicalMV produces identical results to Optimizer with no solve_method."""
    portfolio, _, _ = _make_test_portfolio()

    opt_legacy = Optimizer(
        return_model=ExpectedReturn(),
        risk_model=Risk(),
        constraints=[LongOnly(), SumToOne()],
    )
    opt_classical = Optimizer(
        return_model=ExpectedReturn(),
        risk_model=Risk(),
        constraints=[LongOnly(), SumToOne()],
        solve_method=ClassicalMV(),
    )

    frontier_legacy = opt_legacy.efficient_frontier(portfolio, start=None, end=None, frequency="quarterly")
    frontier_classical = opt_classical.efficient_frontier(portfolio, start=None, end=None, frequency="quarterly")

    np.testing.assert_allclose(
        frontier_legacy.risks.to_numpy(),
        frontier_classical.risks.to_numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        frontier_legacy.expected_returns.to_numpy(),
        frontier_classical.expected_returns.to_numpy(),
        atol=1e-6,
    )
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest tests/test_robust_optimizer.py::test_classical_mv_matches_legacy_optimizer -v
```

Expected: ImportError or AttributeError — `ClassicalMV` doesn't exist yet.

- [ ] **Step 3: Implement `ClassicalMV`**

Add after `OptimizationMethod` in `optimizer.py`:

```python
@dataclass(frozen=True)
class ClassicalMV:
    """Classical mean-variance solve: minimise wᵀΣw subject to μᵀw ≥ target."""

    def solve_one_point(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        constraints: list[Constraint],
        asset_names: list[str],
        target_return: float,
    ) -> SolvePointResult:
        n_assets = len(asset_names)
        w_var = cp.Variable(n_assets)
        point_constraints: list[Any] = [
            c for con in constraints for c in con.to_cvxpy(w_var, asset_names)
        ]
        point_constraints.append(mu @ w_var >= target_return)

        objective = cp.Minimize(cp.quad_form(w_var, sigma))  # type: ignore[attr-defined]
        problem = cp.Problem(objective, point_constraints)

        try:
            problem.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]
        except cp.SolverError as exc:
            return SolvePointResult(
                feasible=False, weights=None, ret=None, risk=None,
                reason=f"SolverError: {exc}",
            )

        if problem.status not in ("optimal", "optimal_inaccurate"):
            return SolvePointResult(
                feasible=False, weights=None, ret=None, risk=None,
                reason=f"Solver status: {problem.status}",
            )
        if w_var.value is None:
            return SolvePointResult(
                feasible=False, weights=None, ret=None, risk=None,
                reason="Solver returned no weights",
            )

        w_opt: np.ndarray = w_var.value
        port_variance = float(w_opt @ sigma @ w_opt)
        port_vol = math.sqrt(max(port_variance, 0.0))
        port_return = float(mu @ w_opt)
        return SolvePointResult(feasible=True, weights=w_opt, ret=port_return, risk=port_vol, reason=None)
```

- [ ] **Step 4: Refactor `Optimizer.efficient_frontier` to delegate to `solve_method`**

Update `Optimizer` to add the `solve_method` field and refactor `efficient_frontier`. The key changes:

1. Add `solve_method` field with `field(default_factory=ClassicalMV)`:

```python
@dataclass
class Optimizer:
    return_model: ExpectedReturn
    risk_model: Risk
    constraints: list[Constraint]
    solve_method: OptimizationMethod = field(default_factory=ClassicalMV)
```

2. Replace the inner loop in `efficient_frontier` with a call to `solve_method.solve_one_point`:

```python
for target in return_targets:
    result = self.solve_method.solve_one_point(
        mu=mu,
        sigma=sigma,
        constraints=self.constraints,
        asset_names=asset_names,
        target_return=float(target),
    )
    if not result.feasible:
        continue
    assert result.weights is not None
    assert result.ret is not None
    assert result.risk is not None
    rows_weights.append(result.weights.tolist())
    rows_returns.append(result.ret)
    rows_risks.append(result.risk)
```

Remove the old inner-loop variables (`w_var`, `point_constraints`, `objective`, `problem`) — they now live in `ClassicalMV.solve_one_point`.

- [ ] **Step 5: Run the tests**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest tests/test_robust_optimizer.py -v
```

Expected: all passing tests pass; `test_classical_mv_matches_legacy_optimizer` should PASS.

- [ ] **Step 6: Run full suite to confirm no regressions**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest -x -q
```

Expected: all existing tests pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/erevtsov/dev/waypoint && git add src/waypoint/analysis/optimizer.py tests/test_robust_optimizer.py && git commit -m "feat: extract ClassicalMV from Optimizer inner loop"
```

---

## Task 3: Implement `RobustMV`

**Files:**
- Modify: `src/waypoint/analysis/optimizer.py` (add `RobustMV` class, update `efficient_frontier` for `kappa="auto"`)
- Test: `tests/test_robust_optimizer.py` (add tests)

Context: `RobustMV` uses `cp.SOC` instead of a linear return constraint. `cp.SOC(t, x)` encodes `t >= ||x||_2`. The constraint `μᵀw − κ·‖Σ^(1/2)w‖₂ ≥ target` becomes `cp.SOC(mu @ w - target, kappa * sigma_half @ w)`. `sigma_half` is `np.linalg.cholesky(sigma)` (lower-triangular). For `kappa="auto"`, `Optimizer.efficient_frontier` resolves the float before the loop using `scipy.stats.chi2.ppf(0.95, df=p)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_robust_optimizer.py`:

```python
import warnings
from waypoint.analysis.optimizer import RobustMV


def test_robust_mv_lower_concentration_than_classical():
    """RobustMV should produce lower max-weight concentration than ClassicalMV."""
    portfolio, _, _ = _make_test_portfolio()

    opt_classical = Optimizer(
        return_model=ExpectedReturn(),
        risk_model=Risk(),
        constraints=[LongOnly(), SumToOne()],
        solve_method=ClassicalMV(),
    )
    opt_robust = Optimizer(
        return_model=ExpectedReturn(),
        risk_model=Risk(),
        constraints=[LongOnly(), SumToOne()],
        solve_method=RobustMV(kappa=0.5),
    )

    frontier_c = opt_classical.efficient_frontier(portfolio, start=None, end=None, frequency="quarterly")
    frontier_r = opt_robust.efficient_frontier(portfolio, start=None, end=None, frequency="quarterly")

    # Max weight across all frontier points and all assets
    asset_names = frontier_c.asset_names
    max_c = max(
        frontier_c.weights[name].max() for name in asset_names
    )
    max_r = max(
        frontier_r.weights[name].max() for name in asset_names
    )
    assert max_r <= max_c + 0.01, (
        f"RobustMV max weight {max_r:.3f} should be <= ClassicalMV {max_c:.3f}"
    )


def test_robust_mv_kappa_zero_matches_classical():
    """RobustMV(kappa=0) should produce the same frontier as ClassicalMV."""
    portfolio, _, _ = _make_test_portfolio()

    opt_classical = Optimizer(
        return_model=ExpectedReturn(),
        risk_model=Risk(),
        constraints=[LongOnly(), SumToOne()],
        solve_method=ClassicalMV(),
    )
    opt_robust = Optimizer(
        return_model=ExpectedReturn(),
        risk_model=Risk(),
        constraints=[LongOnly(), SumToOne()],
        solve_method=RobustMV(kappa=0.0),
    )

    frontier_c = opt_classical.efficient_frontier(portfolio, start=None, end=None, frequency="quarterly")
    frontier_r = opt_robust.efficient_frontier(portfolio, start=None, end=None, frequency="quarterly")

    np.testing.assert_allclose(
        frontier_c.risks.to_numpy(),
        frontier_r.risks.to_numpy(),
        atol=1e-4,
    )


def test_robust_mv_kappa_auto_returns_valid_frontier():
    """RobustMV(kappa='auto') runs without error and returns a valid frontier."""
    portfolio, _, _ = _make_test_portfolio()

    opt = Optimizer(
        return_model=ExpectedReturn(),
        risk_model=Risk(),
        constraints=[LongOnly(), SumToOne()],
        solve_method=RobustMV(kappa="auto"),
    )
    frontier = opt.efficient_frontier(portfolio, start=None, end=None, frequency="quarterly")
    assert len(frontier.risks) > 0
    assert all(r >= 0 for r in frontier.risks.to_list())


def test_robust_mv_non_pd_sigma_returns_infeasible():
    """RobustMV returns infeasible SolvePointResult when sigma is not positive definite."""
    # Build a singular (non-PD) sigma: rank-1 matrix
    v = np.array([1.0, 1.0])
    sigma_singular = np.outer(v, v)  # rank 1, not PD

    mu = np.array([0.1, 0.05])
    constraints = [LongOnly(), SumToOne()]
    asset_names = ["A", "B"]

    result = RobustMV(kappa=0.1).solve_one_point(
        mu=mu,
        sigma=sigma_singular,
        constraints=constraints,
        asset_names=asset_names,
        target_return=0.07,
    )
    assert result.feasible is False
    assert result.reason is not None
    assert "positive definite" in result.reason.lower() or "cholesky" in result.reason.lower()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest tests/test_robust_optimizer.py::test_robust_mv_lower_concentration_than_classical tests/test_robust_optimizer.py::test_robust_mv_kappa_zero_matches_classical tests/test_robust_optimizer.py::test_robust_mv_kappa_auto_returns_valid_frontier tests/test_robust_optimizer.py::test_robust_mv_non_pd_sigma_returns_infeasible -v
```

Expected: ImportError — `RobustMV` doesn't exist yet.

- [ ] **Step 3: Implement `RobustMV`**

Add after `ClassicalMV` in `optimizer.py`. Also add `import scipy.stats` at the top of the file (add to existing stdlib/third-party imports).

```python
@dataclass(frozen=True)
class RobustMV:
    """Robust mean-variance solve using ellipsoidal uncertainty on μ (SOCP).

    Replaces the classical return constraint μᵀw ≥ target with:
        μᵀw − κ · ‖Σ^(1/2) w‖₂ ≥ target

    Parameters
    ----------
    kappa:
        Uncertainty radius. ``"auto"`` derives it from data as the 95%-confidence
        radius of the estimation error ellipsoid: ``sqrt(chi2.ppf(0.95, p)) / sqrt(T)``.
    """

    kappa: float | Literal["auto"] = 0.1

    def solve_one_point(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        constraints: list[Constraint],
        asset_names: list[str],
        target_return: float,
    ) -> SolvePointResult:
        """Solve one frontier point.

        ``self.kappa`` must be a concrete float when this method is called —
        ``Optimizer.efficient_frontier`` replaces ``kappa="auto"`` with a
        pre-computed float value before the loop (see Task 3, Step 4).
        """
        assert isinstance(self.kappa, float), (
            "RobustMV.solve_one_point: kappa must be a float; "
            "kappa='auto' should have been resolved by Optimizer before calling this method."
        )
        kappa_val: float = self.kappa

        try:
            sigma_half = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            return SolvePointResult(
                feasible=False, weights=None, ret=None, risk=None,
                reason="sigma is not positive definite — Cholesky decomposition failed",
            )

        n_assets = len(asset_names)
        w_var = cp.Variable(n_assets)
        point_constraints: list[Any] = [
            c for con in constraints for c in con.to_cvxpy(w_var, asset_names)
        ]
        # SOC encodes: mu @ w - target_return >= kappa * ||sigma_half @ w||_2
        point_constraints.append(
            cp.SOC(mu @ w_var - target_return, kappa_val * sigma_half @ w_var)  # type: ignore[attr-defined]
        )

        objective = cp.Minimize(cp.quad_form(w_var, sigma))  # type: ignore[attr-defined]
        problem = cp.Problem(objective, point_constraints)

        try:
            problem.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]
        except cp.SolverError as exc:
            return SolvePointResult(
                feasible=False, weights=None, ret=None, risk=None,
                reason=f"SolverError: {exc}",
            )

        if problem.status not in ("optimal", "optimal_inaccurate"):
            return SolvePointResult(
                feasible=False, weights=None, ret=None, risk=None,
                reason=f"Solver status: {problem.status}",
            )
        if w_var.value is None:
            return SolvePointResult(
                feasible=False, weights=None, ret=None, risk=None,
                reason="Solver returned no weights",
            )

        w_opt: np.ndarray = w_var.value
        port_variance = float(w_opt @ sigma @ w_opt)
        port_vol = math.sqrt(max(port_variance, 0.0))
        port_return = float(mu @ w_opt)
        return SolvePointResult(feasible=True, weights=w_opt, ret=port_return, risk=port_vol, reason=None)
```

- [ ] **Step 4: Update `Optimizer.efficient_frontier` to resolve `kappa="auto"` before the loop**

Add `import scipy.stats` to the top of the file imports.

In `efficient_frontier`, after computing `er_result` and `risk_result`, add the following block that builds `effective_method` — the solve method that will be used in the loop. When `kappa="auto"`, it creates a new `RobustMV` with the concrete float; otherwise it uses `self.solve_method` unchanged. This keeps `solve_one_point` signatures clean and avoids a protocol mismatch.

```python
# Resolve kappa="auto" once before the loop; effective_method always has a concrete kappa
if isinstance(self.solve_method, RobustMV) and self.solve_method.kappa == "auto":
    # Get the returns matrix to derive T and p for the auto-kappa formula.
    # portfolio.get_returns is cached, so this does not re-fetch.
    wide_returns = portfolio.get_returns(start, end, frequency=freq)
    returns_matrix = wide_returns.select(asset_names).to_numpy()
    T, p = returns_matrix.shape
    kappa_val = math.sqrt(scipy.stats.chi2.ppf(0.95, df=p)) / math.sqrt(T)  # type: ignore[attr-defined]
    effective_method: OptimizationMethod = RobustMV(kappa=kappa_val)
else:
    effective_method = self.solve_method
```

Then the loop calls `effective_method` uniformly (no `isinstance` branching):

```python
for target in return_targets:
    result = effective_method.solve_one_point(
        mu=mu, sigma=sigma, constraints=self.constraints,
        asset_names=asset_names, target_return=float(target),
    )
    if not result.feasible:
        continue
    ...
```

- [ ] **Step 5: Run the new tests**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest tests/test_robust_optimizer.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run full suite**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest -x -q
```

Expected: all existing tests pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/erevtsov/dev/waypoint && git add src/waypoint/analysis/optimizer.py tests/test_robust_optimizer.py && git commit -m "feat: implement RobustMV SOCP solve method"
```

---

## Task 4: Add infeasibility feedback to `EfficientFrontierResult`

**Files:**
- Modify: `src/waypoint/analysis/optimizer.py` (`EfficientFrontierResult` and `Optimizer.efficient_frontier`)
- Test: `tests/test_robust_optimizer.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_robust_optimizer.py`:

```python
def test_infeasible_points_recorded_and_warned():
    """Infeasible frontier points are recorded in result.infeasible_points and trigger warnings."""
    portfolio, _, _ = _make_test_portfolio()

    # Use a very high kappa to force infeasibility at high return targets
    opt = Optimizer(
        return_model=ExpectedReturn(),
        risk_model=Risk(),
        constraints=[LongOnly(), SumToOne()],
        solve_method=RobustMV(kappa=10.0),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frontier = opt.efficient_frontier(portfolio, start=None, end=None, frequency="quarterly", n_points=10)

    # At least some points should be infeasible with kappa=10
    # (If all pass, kappa wasn't high enough — but 10.0 on a quarterly portfolio should force failures)
    # We only assert the structure is correct:
    assert isinstance(frontier.infeasible_points, list)
    for item in frontier.infeasible_points:
        assert len(item) == 2
        assert isinstance(item[0], float)
        assert isinstance(item[1], str)
    # Each infeasible point must have triggered a warning
    assert len(caught) == len(frontier.infeasible_points)


def test_print_diagnostics_outputs_to_stdout(capsys):
    """print_diagnostics() prints a summary without error."""
    from waypoint.analysis.optimizer import EfficientFrontierResult
    import polars as pl

    result = EfficientFrontierResult(
        weights=pl.DataFrame({"expected_return": [0.05], "A": [0.6], "B": [0.4]}),
        expected_returns=pl.Series("expected_return", [0.05]),
        risks=pl.Series("risk", [0.12]),
        asset_names=["A", "B"],
        infeasible_points=[(0.15, "Solver status: infeasible")],
    )
    result.print_diagnostics()
    captured = capsys.readouterr()
    assert "0.15" in captured.out or "infeasible" in captured.out.lower()


def test_efficient_frontier_result_backward_compat():
    """EfficientFrontierResult can be constructed without infeasible_points."""
    from waypoint.analysis.optimizer import EfficientFrontierResult
    import polars as pl

    # Old call sites don't pass infeasible_points — must not raise
    result = EfficientFrontierResult(
        weights=pl.DataFrame({"expected_return": [0.05], "A": [1.0]}),
        expected_returns=pl.Series("expected_return", [0.05]),
        risks=pl.Series("risk", [0.10]),
        asset_names=["A"],
    )
    assert result.infeasible_points == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest tests/test_robust_optimizer.py::test_infeasible_points_recorded_and_warned tests/test_robust_optimizer.py::test_print_diagnostics_outputs_to_stdout tests/test_robust_optimizer.py::test_efficient_frontier_result_backward_compat -v
```

Expected: AttributeError — `infeasible_points` and `print_diagnostics` don't exist yet.

- [ ] **Step 3: Update `EfficientFrontierResult`**

Add `field` to the dataclass imports (already done in Task 1). Add to `EfficientFrontierResult`:

```python
@dataclass
class EfficientFrontierResult:
    weights: pl.DataFrame
    expected_returns: pl.Series
    risks: pl.Series
    asset_names: list[str]
    infeasible_points: list[tuple[float, str]] = field(default_factory=list)  # NEW

    def print_diagnostics(self) -> None:
        """Print a summary of skipped frontier points with their infeasibility reasons."""
        if not self.infeasible_points:
            print("No infeasible frontier points.")
            return
        print(f"{len(self.infeasible_points)} infeasible frontier point(s):")
        for target, reason in self.infeasible_points:
            print(f"  target_return={target:.4f}: {reason}")

    # ... (all existing methods unchanged) ...
```

- [ ] **Step 4: Update `Optimizer.efficient_frontier` to collect infeasible points**

Replace the inner loop result handling:

```python
infeasible_pts: list[tuple[float, str]] = []

for target in return_targets:
    # ... (solve_one_point call as in Task 3) ...
    if not result.feasible:
        reason = result.reason or "unknown"
        warnings.warn(reason, stacklevel=2)
        infeasible_pts.append((float(target), reason))
        continue
    # ... (append to rows_weights, rows_returns, rows_risks) ...

# In the return statement:
return EfficientFrontierResult(
    weights=weights_df,
    expected_returns=pl.Series("expected_return", sorted_returns),
    risks=pl.Series("risk", sorted_risks),
    asset_names=asset_names,
    infeasible_points=infeasible_pts,
)
```

- [ ] **Step 5: Run new tests**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest tests/test_robust_optimizer.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run full suite**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest -x -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/erevtsov/dev/waypoint && git add src/waypoint/analysis/optimizer.py tests/test_robust_optimizer.py && git commit -m "feat: add infeasible_points and print_diagnostics to EfficientFrontierResult"
```

---

## Task 5: Create `src/waypoint/opt.py` and update `__init__.py`

**Files:**
- Create: `src/waypoint/opt.py`
- Modify: `src/waypoint/__init__.py`
- Test: `tests/test_robust_optimizer.py` (add import smoke test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_robust_optimizer.py`:

```python
def test_wp_opt_public_namespace():
    """wp.opt exposes ClassicalMV, RobustMV, OptimizationMethod, SolvePointResult."""
    import waypoint as wp

    assert hasattr(wp, "opt")
    assert hasattr(wp.opt, "ClassicalMV")
    assert hasattr(wp.opt, "RobustMV")
    assert hasattr(wp.opt, "OptimizationMethod")
    assert hasattr(wp.opt, "SolvePointResult")
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest tests/test_robust_optimizer.py::test_wp_opt_public_namespace -v
```

Expected: AttributeError — `wp.opt` doesn't exist yet.

- [ ] **Step 3: Create `src/waypoint/opt.py`**

```python
"""Optimisation method strategies for wp.opt.*."""

from waypoint.analysis.optimizer import ClassicalMV, OptimizationMethod, RobustMV, SolvePointResult

__all__ = ["ClassicalMV", "OptimizationMethod", "RobustMV", "SolvePointResult"]
```

- [ ] **Step 4: Update `src/waypoint/__init__.py`**

Add `opt` to the imports line:

```python
from waypoint import analytics, cashflows, catalog, opt, returns, risk, sim, social_security
```

Add `"opt"` to `__all__` under `# sub-modules`:

```python
"opt",
```

- [ ] **Step 5: Run test**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest tests/test_robust_optimizer.py::test_wp_opt_public_namespace -v
```

Expected: PASS.

- [ ] **Step 6: Run full suite**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest -x -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/erevtsov/dev/waypoint && git add src/waypoint/opt.py src/waypoint/__init__.py tests/test_robust_optimizer.py && git commit -m "feat: expose wp.opt namespace with ClassicalMV, RobustMV, OptimizationMethod, SolvePointResult"
```

---

## Task 6: Update README and run lint + typecheck

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README.md**

Find the optimizer section in README.md. Add a new subsection under the existing `Optimizer` documentation:

```markdown
### Optimisation methods (`wp.opt`)

By default `Optimizer` uses `ClassicalMV` (classical mean-variance). Pass `solve_method` to swap in a different strategy:

```python
import waypoint as wp

opt = wp.analytics.Optimizer(
    return_model=wp.analytics.ExpectedReturn(method=wp.returns.ShrinkageTowardGrandMean()),
    risk_model=wp.analytics.Risk(method=wp.risk.LedoitWolf()),
    constraints=[wp.LongOnly(), wp.SumToOne()],
    solve_method=wp.opt.RobustMV(kappa=0.1),   # default: wp.opt.ClassicalMV()
)
```

| Method | Description |
|---|---|
| `wp.opt.ClassicalMV()` | Classical mean-variance (default). Minimises variance subject to `μᵀw ≥ target`. |
| `wp.opt.RobustMV(kappa=0.1)` | Robust MV with ellipsoidal uncertainty on μ. Replaces the return constraint with the worst-case version `μᵀw − κ·‖Σ^(1/2)w‖₂ ≥ target`. Higher κ → more conservative, less concentrated portfolios. |
| `wp.opt.RobustMV(kappa="auto")` | Automatically derives κ from the data as the 95%-confidence estimation-error radius. |

**Infeasibility diagnostics:**

```python
frontier = opt.efficient_frontier(portfolio, start, end, frequency="quarterly")

# Inspect skipped points
print(frontier.infeasible_points)   # list of (target_return, reason) tuples
frontier.print_diagnostics()        # prints a formatted summary
```
```

- [ ] **Step 2: Run lint**

```bash
cd /Users/erevtsov/dev/waypoint && uv run ruff check src/ tests/
```

Expected: no errors. If lint errors appear, fix them before proceeding.

- [ ] **Step 3: Run typecheck**

```bash
cd /Users/erevtsov/dev/waypoint && uv run mypy
```

Expected: no errors. Common issues to watch for:
- `field(default_factory=ClassicalMV)` on `Optimizer.solve_method` — mypy may complain about the `OptimizationMethod` Protocol type; use `OptimizationMethod` as the field annotation.
- `cp.SOC` and `cp.quad_form` — these already have `# type: ignore[attr-defined]` in the existing code; use the same pattern.
- `scipy.stats` — if mypy doesn't find stubs, add `# type: ignore[import-untyped]` to that import.

- [ ] **Step 4: Run full test suite one last time**

```bash
cd /Users/erevtsov/dev/waypoint && uv run pytest -q
```

Expected: all tests pass with coverage.

- [ ] **Step 5: Commit**

```bash
cd /Users/erevtsov/dev/waypoint && git add README.md && git commit -m "docs: document wp.opt, RobustMV, infeasibility diagnostics in README"
```
