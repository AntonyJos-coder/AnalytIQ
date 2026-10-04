"""Plotly-ready chart specifications (rendered by Plotly.js in the browser)."""
from __future__ import annotations

from typing import Any, Sequence

PALETTE = ["#2f5d8a", "#c7843a", "#4a8f7f", "#8a5a83", "#b35b4f", "#6d7b8d"]
_LAYOUT = {
    "font": {"family": "Segoe UI, system-ui, sans-serif", "size": 12, "color": "#2b3340"},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 56, "r": 16, "t": 44, "b": 56},
    "xaxis": {"gridcolor": "#e6e9ee", "automargin": True},
    "yaxis": {"gridcolor": "#e6e9ee", "automargin": True},
    "legend": {"orientation": "h", "y": -0.25},
}


def _layout(title: str, xlabel: str = "", ylabel: str = "") -> dict[str, Any]:
    layout = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _LAYOUT.items()}
    layout["title"] = {"text": title, "font": {"size": 14}}
    layout["xaxis"]["title"] = {"text": xlabel}
    layout["yaxis"]["title"] = {"text": ylabel}
    return layout


def _py(values: Sequence[Any]) -> list[Any]:
    out = []
    for v in values:
        if hasattr(v, "item"):
            v = v.item()
        out.append(v)
    return out


def bar_chart(x: Sequence[Any], y: Sequence[Any], title: str, xlabel: str = "", ylabel: str = "",
              signed_colors: bool = False) -> dict[str, Any]:
    ys = _py(y)
    colors = [("#b35b4f" if (v is not None and v < 0) else "#4a8f7f") for v in ys] if signed_colors else PALETTE[0]
    return {"data": [{"type": "bar", "x": _py(x), "y": ys, "marker": {"color": colors}}],
            "layout": _layout(title, xlabel, ylabel)}


def line_chart(series: dict[str, tuple[Sequence[Any], Sequence[Any]]], title: str, xlabel: str = "",
               ylabel: str = "") -> dict[str, Any]:
    data = []
    for i, (name, (x, y)) in enumerate(series.items()):
        data.append({"type": "scatter", "mode": "lines+markers", "name": name, "x": _py(x), "y": _py(y),
                     "line": {"color": PALETTE[i % len(PALETTE)], "width": 2.5}})
    layout = _layout(title, xlabel, ylabel)
    layout["showlegend"] = len(series) > 1
    return {"data": data, "layout": layout}


def pie_chart(labels: Sequence[Any], values: Sequence[Any], title: str, donut: bool = True) -> dict[str, Any]:
    return {"data": [{"type": "pie", "labels": _py(labels), "values": _py(values), "hole": 0.5 if donut else 0,
                      "marker": {"colors": PALETTE}}], "layout": _layout(title)}


def should_chart(intent: str, n_points: int) -> bool:
    """Only chart when it improves the answer: multiple points, and not a plain scalar."""
    return intent in {"trend", "rank", "breakdown", "change", "multihop"} and n_points >= 2
