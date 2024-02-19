import ipywidgets
import numpy as np
import plotly.graph_objects as go
from beartype.typing import Dict


def visualize_per_token_category(input: Dict[str, Dict[str, tuple]]) -> ipywidgets.VBox:
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
                color="purple",
            ),
        ),
        layout=go.Layout(yaxis=dict(title="Loss")),
    )

    selected_category = ipywidgets.Dropdown(
        options=categories,
        placeholder="",
        description="Token Category:",
        disabled=False,
    )

    def response(change):
        if selected_category.value:
            means, err_lo, err_hi = get_plot_values(selected_category.value)
            with g.batch_update():
                g.data[0].y = means
                g.data[0].error_y["array"] = err_hi
                g.data[0].error_y["arrayminus"] = err_lo

    selected_category.observe(response, names="value")

    return ipywidgets.VBox([selected_category, g])
