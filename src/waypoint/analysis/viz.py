"""Visualisation helpers for analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
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

    Displays p5, p25, p50 (median), p75, and p95 wealth paths.

    Parameters
    ----------
    result:
        A ``SimulationResult`` from ``WealthSimulation.compute``.

    Returns
    -------
    go.Figure
    """
    df = result.percentile_df
    periods = df["period"].to_list()

    fig = go.Figure()

    # Shaded band: p5–p95
    fig.add_trace(
        go.Scatter(
            x=periods + periods[::-1],
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
            x=periods + periods[::-1],
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
            x=periods,
            y=df["p50"].to_list(),
            mode="lines",
            line={"color": "rgb(0, 100, 200)", "width": 2},
            name="Median (p50)",
        )
    )

    fig.update_layout(
        title="Wealth Simulation — Percentile Fan Chart",
        xaxis_title="Period",
        yaxis_title="Portfolio Value",
        legend={"orientation": "h"},
    )
    return fig
