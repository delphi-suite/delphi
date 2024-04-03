from typing import Union

import ipywidgets
import numpy as np
import plotly.graph_objects as go


def visualize_per_token_category(
    input: dict[Union[str, int], dict[str, tuple]],
    log_scale=False,
    line_metric="Means",
    checkpoint_mode=True,
    shade_color="rgba(68, 68, 68, 0.3)",
    line_color="rgb(31, 119, 180)",
    bar_color="purple",
    marker_color="SkyBlue",
    background_color="AliceBlue",
) -> go.FigureWidget:
    input_x = list(input.keys())
    categories = list(input[input_x[0]].keys())
    category = categories[0]

    def get_hovertexts(mid: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[str]:
        return [f"Loss: {m:.3f} ({l:.3f}, {h:.3f})" for m, l, h in zip(mid, lo, hi)]

    def get_plot_values(category: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.array([input[x][category] for x in input_x]).T
        means, err_lo, err_hi = x[0], x[1], x[2]
        return means, err_lo, err_hi

    means, err_lo, err_hi = get_plot_values(category)

    if checkpoint_mode:
        scatter_plot = go.Figure(
            [
                go.Scatter(
                    name="Upper Bound",
                    x=input_x,
                    y=means + err_hi,
                    mode="lines",
                    marker=dict(color=shade_color),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name="Lower Bound",
                    x=input_x,
                    y=means - err_lo,
                    marker=dict(color=shade_color),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor=shade_color,
                    fill="tonexty",
                    showlegend=False,
                ),
                go.Scatter(
                    name=line_metric,
                    x=input_x,
                    y=means,
                    mode="lines",
                    marker=dict(
                        color=line_color,
                        size=0,
                        line=dict(color=line_color, width=1),
                    ),
                ),
            ]
        )
    else:
        scatter_plot = go.Scatter(
            x=input_x,
            y=means,
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_hi,
                arrayminus=err_lo,
                color=bar_color,
            ),
            marker=dict(
                color=marker_color,
                size=15,
                line=dict(color=line_color, width=2),
            ),
            hovertext=get_hovertexts(means, err_lo, err_hi),
            hoverinfo="text+x",
        )
    g = go.FigureWidget(
        data=scatter_plot,
        layout=go.Layout(
            yaxis=dict(
                title="Loss",
                type="log" if log_scale else "linear",
            ),
            plot_bgcolor=background_color,
        ),
    )

    return g
