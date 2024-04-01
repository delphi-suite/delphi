from typing import Union

import ipywidgets
import numpy as np
import plotly.graph_objects as go


def visualize_per_token_category(
    input: dict[Union[str, int], tuple],
    # input: dict[Union[str, int], dict[str, tuple]],
    log_scale=False,
    **kwargs: Union[str, bool],
) -> go.FigureWidget:
    input_x = list(input.keys())
    # categories = list(input[input_x[0]].keys())
    # category = categories[0]

    def get_hovertexts(mid: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[str]:
        return [f"Loss: {m:.3f} ({l:.3f}, {h:.3f})" for m, l, h in zip(mid, lo, hi)]

    def get_plot_values() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.array([input[x] for x in input_x]).T
        means, err_lo, err_hi = x[0], x[1], x[2]
        return means, err_lo, err_hi

    means, err_lo, err_hi = get_plot_values()

    if kwargs.get("checkpoint_mode"):
        scatter_plot = go.Figure(
            [
                go.Scatter(
                    name="Upper Bound",
                    x=input_x,
                    y=means + err_hi,
                    mode="lines",
                    marker=dict(color=kwargs.get("shade_color", "#444")),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name="Lower Bound",
                    x=input_x,
                    y=means - err_lo,
                    marker=dict(color=kwargs.get("shade_color", "#444")),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor=kwargs.get("shade_color", "rgba(68, 68, 68, 0.3)"),
                    fill="tonexty",
                    showlegend=False,
                ),
                go.Scatter(
                    name=kwargs.get("line_metric", "Means"),
                    x=input_x,
                    y=means,
                    mode="lines",
                    marker=dict(
                        color=kwargs.get("line_color", "rgb(31, 119, 180)"),
                        size=0,
                        line=dict(
                            color=kwargs.get("line_color", "rgb(31, 119, 180)"), width=1
                        ),
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
                color=kwargs.get("bar_color", "purple"),
            ),
            marker=dict(
                color=kwargs.get("marker_color", "SkyBlue"),
                size=15,
                line=dict(color=kwargs.get("line_color", "MediumPurple"), width=2),
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
            plot_bgcolor=kwargs.get("bg_color", "AliceBlue"),
        ),
    )

    # selected_category = ipywidgets.Dropdown(
    #     options=categories,
    #     placeholder="",
    #     description="Token Category:",
    #     disabled=False,
    # )

    # def response(change):
    #     means, err_lo, err_hi = get_plot_values(selected_category.value)
    #     with g.batch_update():
    #         if kwargs.get("checkpoint_mode"):
    #             g.data[0].y = means
    #             g.data[1].y = means + err_hi
    #             g.data[2].y = means - err_lo
    #         else:
    #             g.data[0].y = means
    #             g.data[0].error_y["array"] = err_hi
    #             g.data[0].error_y["arrayminus"] = err_lo
    #             g.data[0].hovertext = get_hovertexts(means, err_lo, err_hi)

    # selected_category.observe(response, names="value")

    # return ipywidgets.VBox([selected_category, g])
    return g
