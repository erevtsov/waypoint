"""Visualisation helpers for analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from waypoint.analysis.compare import ComparisonResult
    from waypoint.analysis.optimizer import EfficientFrontierResult
    from waypoint.analysis.simulation import SimulationResult


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
    use_dates = "date" in df.columns
    x_values = df["date"].to_list() if use_dates else df["period"].to_list()

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
        xaxis_title="Date" if use_dates else "Period",
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

    # Use dates on the x-axis only when every scenario was computed with one
    use_dates = all("date" in r.percentile_df.columns for r in result.scenarios.values())

    for idx, (label, sim_result) in enumerate(result.scenarios.items()):
        df = sim_result.percentile_df
        x_values = df["date"].to_list() if use_dates else df["period"].to_list()
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
        xaxis_title="Date" if use_dates else "Period",
        yaxis_title="Portfolio Value",
        legend={"orientation": "h"},
    )
    return fig
