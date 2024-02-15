import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact


def visualize_per_token_category(input):
    model_names = list(input.keys())
    categories = list(list(input.values())[0].keys())

    def _f(category):
        x = np.array([input[name][category] for name in model_names]).T
        means = np.mean(x, axis=0)
        median = np.median(x, axis=0)
        q1 = np.quantile(x, 0.25, axis=0)
        q3 = np.quantile(x, 0.75, axis=0)

        ax = plt.gca()
        ax.set_ylim([-5, 5])  # TODO

        plt.plot(model_names, means)
        plt.errorbar(model_names, median, yerr=[median - q1, q3 - median], fmt="o")

    interact(
        _f,
        category=widgets.Dropdown(
            options=categories,
            placeholder="",
            description="Token Category:",
            disabled=False,
        ),
    )


# Usage:
# from dataset.mock_per_token_performance import performance_datas
# visualize_per_token_category(performance_data)
