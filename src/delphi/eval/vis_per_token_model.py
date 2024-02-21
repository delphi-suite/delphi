import ipywidgets
import numpy as np
import plotly.graph_objects as go


def visualize_per_token_category(
    input: dict[str, dict[str, tuple]], log_scale=False, **kwargs: str
) -> ipywidgets.VBox:
    model_names = list(input.keys())
    categories = list(input[model_names[0]].keys())
    category = categories[0]

    def get_plot_values(category: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.array([input[name][category] for name in model_names]).T
        means, err_lo, err_hi = x[0], x[1], x[2]
        return means, err_lo, err_hi

    means, err_low, err_hi = get_plot_values(category)
    g = go.FigureWidget(
        data=go.Scatter(
            x=model_names,
            y=means,
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_hi,
                arrayminus=err_low,
                color=kwargs.get("bar_color", "purple"),
            ),
            marker=dict(
                color=kwargs.get("marker_color", "SkyBlue"),
                size=15,
                line=dict(color=kwargs.get("line_color", "MediumPurple"), width=2),
            ),
        ),
        layout=go.Layout(
            yaxis=dict(
                title="Loss",
                type="log" if log_scale else "linear",
            ),
            plot_bgcolor=kwargs.get("bg_color", "AliceBlue"),
        ),
    )

    selected_category = ipywidgets.Dropdown(
        options=categories,
        placeholder="",
        description="Token Category:",
        disabled=False,
    )

    def response(change):
        means, err_lo, err_hi = get_plot_values(selected_category.value)
        with g.batch_update():
            g.data[0].y = means
            g.data[0].error_y["array"] = err_hi
            g.data[0].error_y["arrayminus"] = err_lo

    selected_category.observe(response, names="value")

    return ipywidgets.VBox([selected_category, g])
