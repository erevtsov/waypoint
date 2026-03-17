"""Visualisation helpers for analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go

if TYPE_CHECKING:
    from waypoint.analysis.compare import ComparisonResult
    from waypoint.analysis.expected_return import ExpectedReturnResult
    from waypoint.analysis.optimizer import EfficientFrontierResult
    from waypoint.analysis.risk import RiskResult
    from waypoint.analysis.simulation import MultiWealthSimulationResult, SimulationResult


def _year_x(sim_result: Any) -> tuple[list, dict]:  # type: ignore[type-arg]
    """Return (x_values, xaxis_layout_dict) for simulation time-series plots.

    When the result has a ``"date"`` column, calendar dates are used and the
    axis is formatted to show 4-digit years.  Otherwise the period index is
    converted to fractional years from period 0.
    """
    df = sim_result.percentile_df
    if "date" in df.columns:
        return df["date"].to_list(), {"title": "Year", "tickformat": "%Y", "dtick": "M12"}
    n_periods = sim_result.paths.shape[1] - 1
    ppy = (n_periods / sim_result.horizon_years) if sim_result.horizon_years > 0 else 1.0
    return [round(t / ppy, 3) for t in df["period"].to_list()], {"title": "Year"}


def plot_account_trajectories(result: MultiWealthSimulationResult) -> go.Figure:
    """Line chart of per-account median wealth paths plus the total.

    Each account is a separate line.  The total is rendered as a thick dashed
    black line so it stands out from the per-account series.

    Parameters
    ----------
    result:
        A ``MultiWealthSimulationResult`` from ``MultiWealthSimulation.compute``.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    real = result.total.is_real
    _, xaxis_cfg = _year_x(result.total)

    for account_name, acct_result in result.accounts.items():
        x, _ = _year_x(acct_result)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=acct_result.percentile_df["p50"].to_list(),
                name=account_name,
                mode="lines",
            )
        )

    x, _ = _year_x(result.total)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=result.total.percentile_df["p50"].to_list(),
            name="TOTAL",
            mode="lines",
            line={"width": 3, "dash": "dash", "color": "black"},
        )
    )

    value_label = "Wealth (Real)" if real else "Wealth (Nominal)"
    fig.update_layout(
        title="Per-Account Median Wealth Paths" + (" (Real)" if real else " (Nominal)"),
        xaxis=xaxis_cfg,
        yaxis_title=value_label,
        hovermode="x unified",
        legend={"orientation": "h"},
    )
    return fig


def plot_expected_return(result: ExpectedReturnResult) -> go.Figure:
    """Horizontal bar chart of per-asset annualised expected returns, sorted descending.

    Parameters
    ----------
    result:
        An ``ExpectedReturnResult`` from ``ExpectedReturn.compute``.

    Returns
    -------
    go.Figure
    """
    items = sorted(result.per_asset.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": values, "colorscale": "RdYlGn", "showscale": False},
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Expected Returns by Asset ({result.method_name})",
        xaxis_title="Annualised Expected Return",
        xaxis={"tickformat": ".1%"},
        yaxis_title="",
        height=max(300, 40 * len(names)),
    )
    return fig


def plot_risk_return(er_result: ExpectedReturnResult, risk_result: RiskResult) -> go.Figure:
    """Scatter of annualised volatility (x) vs expected return (y), one point per asset.

    Parameters
    ----------
    er_result:
        An ``ExpectedReturnResult`` from ``ExpectedReturn.compute``.
    risk_result:
        A ``RiskResult`` from ``Risk.compute``, used for per-asset volatilities.

    Returns
    -------
    go.Figure
    """
    names = list(er_result.per_asset.keys())
    returns = [er_result.per_asset[n] for n in names]
    vols = [risk_result.volatilities[n] for n in names]

    fig = go.Figure(
        go.Scatter(
            x=vols,
            y=returns,
            mode="markers+text",
            text=names,
            textposition="top center",
            marker={
                "size": 10,
                "color": returns,
                "colorscale": "RdYlGn",
                "showscale": True,
                "colorbar": {"title": "Return"},
            },
            hovertemplate="%{text}<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Asset Risk vs. Expected Return",
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Expected Return",
        xaxis={"tickformat": ".1%"},
        yaxis={"tickformat": ".1%"},
        height=500,
    )
    return fig


def plot_correlation(risk_result: RiskResult) -> go.Figure:
    """Heatmap of the asset correlation matrix derived from a ``RiskResult``.

    Parameters
    ----------
    risk_result:
        A ``RiskResult`` from ``Risk.compute``.

    Returns
    -------
    go.Figure
    """
    import numpy as np
    import plotly.express as px

    names = list(risk_result.volatilities.keys())
    cov = risk_result.covariance.to_numpy()
    vols_arr = np.array([risk_result.volatilities[n] for n in names])
    safe_vols = np.where(vols_arr > 0, vols_arr, 1.0)
    corr = cov / np.outer(safe_vols, safe_vols)
    np.fill_diagonal(corr, 1.0)

    fig = px.imshow(
        corr,
        x=names,
        y=names,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        title=f"Asset Correlation Matrix ({risk_result.method_name})",
    )
    fig.update_layout(height=max(400, 60 * len(names)))
    return fig


def plot_efficient_frontier(result: EfficientFrontierResult) -> go.Figure:
    """Scatter plot of the efficient frontier (risk vs expected return).

    Each point shows the portfolio risk and expected return, with a hover
    tooltip listing the asset weights at that point.

    Parameters
    ----------
    result:
        An ``EfficientFrontierResult`` from ``Optimizer.efficient_frontier``.

    Returns
    -------
    go.Figure
    """
    risks = result.risks.to_list()
    returns = result.expected_returns.to_list()

    weight_cols = [c for c in result.weights.columns if c != "expected_return"]
    hover_texts: list[str] = []
    for row in result.weights.iter_rows(named=True):
        lines = [f"{name}: {row[name]:.1%}" for name in weight_cols]
        hover_texts.append("<br>".join(lines))

    fig = go.Figure(
        data=go.Scatter(
            x=risks,
            y=returns,
            mode="lines+markers",
            text=hover_texts,
            hovertemplate="Risk: %{x:.2%}<br>Return: %{y:.2%}<br>%{text}<extra></extra>",
            marker={"size": 6, "color": returns, "colorscale": "Viridis", "showscale": True},
        )
    )
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualised Risk (Volatility)",
        yaxis_title="Annualised Expected Return",
        xaxis={"tickformat": ".1%"},
        yaxis={"tickformat": ".1%"},
    )
    return fig


def plot_wealth_simulation(result: SimulationResult) -> go.Figure:
    """Fan chart showing percentile wealth paths from a simulation.

    Displays p5, p25, p50 (median), p75, and p95 wealth paths.  When the
    result was computed with a ``start_date`` the x-axis shows calendar dates;
    otherwise it shows integer periods.  The y-axis label reflects whether the
    values are nominal or real (inflation-adjusted).

    Parameters
    ----------
    result:
        A ``SimulationResult`` from ``WealthSimulation.compute``.

    Returns
    -------
    go.Figure
    """
    df = result.percentile_df
    x_values, xaxis_cfg = _year_x(result)

    value_label = "Portfolio Value (Real)" if result.is_real else "Portfolio Value (Nominal)"
    title = (
        "Wealth Simulation — Percentile Fan Chart"
        + (" (Real)" if result.is_real else " (Nominal)")
    )

    fig = go.Figure()

    # Shaded band: p5–p95
    fig.add_trace(
        go.Scatter(
            x=x_values + x_values[::-1],
            y=df["p95"].to_list() + df["p5"].to_list()[::-1],
            fill="toself",
            fillcolor="rgba(0, 100, 200, 0.1)",
            line={"color": "rgba(255,255,255,0)"},
            name="p5–p95",
            hoverinfo="skip",
        )
    )

    # Shaded band: p25–p75
    fig.add_trace(
        go.Scatter(
            x=x_values + x_values[::-1],
            y=df["p75"].to_list() + df["p25"].to_list()[::-1],
            fill="toself",
            fillcolor="rgba(0, 100, 200, 0.2)",
            line={"color": "rgba(255,255,255,0)"},
            name="p25–p75",
            hoverinfo="skip",
        )
    )

    # Median line
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=df["p50"].to_list(),
            mode="lines",
            line={"color": "rgb(0, 100, 200)", "width": 2},
            name="Median (p50)",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=xaxis_cfg,
        yaxis_title=value_label,
        legend={"orientation": "h"},
    )
    return fig


def plot_allocation_dollar(result: SimulationResult) -> go.Figure:
    """Stacked area chart of median per-asset dollar values over time.

    Each asset is rendered as a filled band stacked on the previous one,
    so the total height equals the sum of per-asset medians.  The x-axis
    shows calendar dates when the result was computed with a ``start_date``,
    otherwise integer periods.

    Parameters
    ----------
    result:
        A ``SimulationResult`` from ``WealthSimulation.compute``.

    Returns
    -------
    go.Figure
    """
    _, xaxis_cfg = _year_x(result)
    value_label = "Asset Value (Real)" if result.is_real else "Asset Value (Nominal)"
    title = "Asset Allocation — Median $ Values" + (" (Real)" if result.is_real else " (Nominal)")

    fig = go.Figure()
    for name, df in result.allocation_dollar.items():
        x_values, _ = _year_x(result)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["p50"].to_list(),
                mode="lines",
                stackgroup="one",
                name=name,
                hovertemplate="%{fullData.name}: %{y:$,.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis=xaxis_cfg,
        yaxis_title=value_label,
        legend={"orientation": "h"},
    )
    return fig


# Distinct colours for up to 6 scenarios; repeated thereafter via modulo.
_SCENARIO_COLORS = [
    (0, 100, 200),
    (200, 60, 0),
    (0, 160, 80),
    (140, 0, 200),
    (200, 160, 0),
    (0, 180, 180),
]


def plot_comparison(result: ComparisonResult) -> go.Figure:
    """Overlaid fan charts for all scenarios in a ``ComparisonResult``.

    Each scenario is rendered as a p5–p95 shaded band and a median line in
    its own colour.  When all scenarios share a ``start_date``, the x-axis
    shows calendar dates; otherwise it shows integer periods.

    Parameters
    ----------
    result:
        A ``ComparisonResult`` from ``ComparisonResult.from_scenarios``.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    first = next(iter(result.scenarios.values()))
    _, xaxis_cfg = _year_x(first)

    for idx, (label, sim_result) in enumerate(result.scenarios.items()):
        x_values, _ = _year_x(sim_result)
        df = sim_result.percentile_df
        r, g, b = _SCENARIO_COLORS[idx % len(_SCENARIO_COLORS)]

        # p5–p95 band
        fig.add_trace(
            go.Scatter(
                x=x_values + x_values[::-1],
                y=df["p95"].to_list() + df["p5"].to_list()[::-1],
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.10)",
                line={"color": "rgba(255,255,255,0)"},
                name=f"{label} p5–p95",
                legendgroup=label,
                hoverinfo="skip",
            )
        )

        # p25–p75 band
        fig.add_trace(
            go.Scatter(
                x=x_values + x_values[::-1],
                y=df["p75"].to_list() + df["p25"].to_list()[::-1],
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.20)",
                line={"color": "rgba(255,255,255,0)"},
                name=f"{label} p25–p75",
                legendgroup=label,
                hoverinfo="skip",
            )
        )

        # Median line
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["p50"].to_list(),
                mode="lines",
                line={"color": f"rgb({r},{g},{b})", "width": 2},
                name=f"{label} median",
                legendgroup=label,
            )
        )

    fig.update_layout(
        title="Scenario Comparison — Wealth Simulation",
        xaxis=xaxis_cfg,
        yaxis_title="Portfolio Value",
        legend={"orientation": "h"},
    )
    return fig
