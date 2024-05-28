import math
import random
import uuid
from typing import Any, Optional, cast

import numpy as np
import panel as pn
import plotly.graph_objects as go
import torch
from datasets import Dataset
from IPython.core.display import HTML
from IPython.core.display_functions import display
from jaxtyping import Float, Int
from transformers import PreTrainedTokenizerBase


def single_loss_diff_to_color(loss_diff: float) -> str:
    # if loss_diff is negative, we want the color to be red
    # if loss_diff is positive, we want the color to be green
    # if loss_diff is 0, we want the color to be white
    # the color should be more intense the larger the absolute value of loss_diff

    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    scaled_loss_diff = sigmoid(loss_diff)  # scale to 0-1

    if scaled_loss_diff < 0.5:  # red
        red_val = 255
        green_blue_val = min(int(255 * 2 * scaled_loss_diff), 255)
        return f"rgb({red_val}, {green_blue_val}, {green_blue_val})"
    else:  # green
        green_val = 255
        red_blue_val = min(int(255 * 2 * (1 - scaled_loss_diff)), 255)
        return f"rgb({red_blue_val}, {green_val}, {red_blue_val})"


def token_to_html(
    token: int,
    tokenizer: PreTrainedTokenizerBase,
    bg_color: str,
    data: dict,
    class_name: str = "token",
) -> str:
    data = data or {}  # equivalent to if not data: data = {}
    # non-breakable space, w/o it leading spaces wouldn't be displayed
    str_token = tokenizer.decode(token).replace(" ", "&nbsp;")

    # background or user-select (for \n) goes here
    specific_styles = {}
    # for now just adds line break or doesn't
    br = ""

    if bg_color:
        specific_styles["background-color"] = bg_color
    if str_token == "\n":
        # replace new line character with two characters: \ and n
        str_token = r"\n"
        # add line break in html
        br += "<br>"
        # this is so we can copy the prompt without "\n"s
        specific_styles["user-select"] = "none"
    str_token = str_token.replace("<", "&lt;").replace(">", "&gt;")

    style_str = data_str = ""
    # converting style dict into the style attribute
    if specific_styles:
        inside_style_str = "; ".join(f"{k}: {v}" for k, v in specific_styles.items())
        style_str = f" style='{inside_style_str}'"
    if data:
        data_str = "".join(
            f" data-{k}='{v.replace(' ', '&nbsp;')}'" for k, v in data.items()
        )
    return f"<div class='{class_name}'{style_str}{data_str}>{str_token}</div>{br}"


_token_style = {
    "border": "1px solid #888",
    "display": "inline-block",
    # each character of the same width, so we can easily spot a space
    "font-family": "monospace",
    "font-size": "14px",
    "color": "black",
    "background-color": "white",
    "margin": "1px 0px 1px 1px",
    "padding": "0px 1px 1px 1px",
}
_token_emphasized_style = {
    "border": "3px solid #888",
    "display": "inline-block",
    "font-family": "monospace",
    "font-size": "14px",
    "color": "black",
    "background-color": "white",
    "margin": "1px 0px 1px 1px",
    "padding": "0px 1px 1px 1px",
}
_token_style_str = " ".join([f"{k}: {v};" for k, v in _token_style.items()])
_token_emphasized_style_str = " ".join(
    [f"{k}: {v};" for k, v in _token_emphasized_style.items()]
)


def vis_pos_map(
    pos_list: list[tuple[int, int]],
    selected_tokens: list[int],
    metrics: Float[torch.Tensor, "prompt pos"],
    token_ids: Int[torch.Tensor, "prompt pos"],
    tokenizer: PreTrainedTokenizerBase,
):
    """
    Randomly sample from pos_map and visualize the loss diff at the corresponding position.
    """

    token_htmls = []
    unique_id = str(uuid.uuid4())
    token_class = f"pretoken_{unique_id}"
    selected_token_class = f"token_{unique_id}"
    hover_div_id = f"hover_info_{unique_id}"

    # choose a random keys from pos_map
    key = random.choice(pos_list)

    prompt, pos = key
    all_toks = token_ids[prompt][: pos + 1]

    for i in range(all_toks.shape[0]):
        token_id = cast(int, all_toks[i].item())
        value = metrics[prompt][i].item()
        token_htmls.append(
            token_to_html(
                token_id,
                tokenizer,
                bg_color="white"
                if np.isnan(value)
                else single_loss_diff_to_color(value),
                data={"loss-diff": f"{value:.2f}"},
                class_name=token_class
                if token_id not in selected_tokens
                else selected_token_class,
            )
        )

    # add break line
    token_htmls.append("<br><br>")

    html_str = f"""
    <style>.{token_class} {{ {_token_style_str}}} .{selected_token_class} {{ {_token_emphasized_style_str} }} #{hover_div_id} {{ height: 100px; font-family: monospace; }}</style>
    {"".join(token_htmls)} <div id='{hover_div_id}'></div>
    <script>
        (function() {{
            var token_divs = document.querySelectorAll('.{token_class}');
            token_divs = Array.from(token_divs).concat(Array.from(document.querySelectorAll('.{selected_token_class}')));
            var hover_info = document.getElementById('{hover_div_id}');


            token_divs.forEach(function(token_div) {{
                token_div.addEventListener('mousemove', function(e) {{
                    hover_info.innerHTML = ""
                    for( var d in this.dataset) {{
                        hover_info.innerHTML += "<b>" + d + "</b> ";
                        hover_info.innerHTML += this.dataset[d] + "<br>";
                    }}
                }});

                token_div.addEventListener('mouseout', function(e) {{
                    hover_info.innerHTML = ""
                }});
            }});
        }})();
    </script>
    """
    display(HTML(html_str))


def token_selector(
    vocab_map: dict[str, int]
) -> tuple[pn.widgets.MultiChoice, list[int]]:
    tokens = list(vocab_map.keys())
    token_selector_ = pn.widgets.MultiChoice(name="Tokens", options=tokens)
    token_ids = [vocab_map[token] for token in cast(list[str], token_selector_.value)]

    def update_tokens(event):
        token_ids.clear()
        token_ids.extend([vocab_map[token] for token in event.new])

    token_selector_.param.watch(update_tokens, "value")
    return token_selector_, token_ids


def calc_model_group_stats(
    tokenized_corpus_dataset: Dataset,
    logprobs_by_dataset: dict[str, torch.Tensor],
    selected_tokens: list[int],
) -> dict[str, dict[str, float]]:
    """
    For each (model, token group) pair, calculate useful stats (for visualization)

    args:
    - tokenized_corpus_dataset: a list of the tokenized corpus datasets, e.g. load_dataset(constants.tokenized_corpus_dataset))["validation"]
    - logprob_datasets: a dict of lists of logprobs, e.g. {"llama2": load_dataset("transcendingvictor/llama2-validation-logprobs")["validation"]["logprobs"]}
    - selected_tokens: a list of selected token IDs, e.g. [46, 402, ...]

    returns: a dict of model names as keys and stats dict as values
        e.g. {"100k": {"mean": -0.5, "median": -0.4, "min": -0.1, "max": -0.9, "25th": -0.3, "75th": -0.7}, ...}

    Stats calculated: mean, median, min, max, 25th percentile, 75th percentile
    """
    model_group_stats = {}
    for model in logprobs_by_dataset:
        model_logprobs = []
        print(f"Processing model {model}")
        dataset = logprobs_by_dataset[model]
        for ix_doc_lp, document_lps in enumerate(dataset):
            tokens = tokenized_corpus_dataset[ix_doc_lp]["tokens"]
            for ix_token, token in enumerate(tokens):
                if ix_token == 0:  # skip the first token, which isn't predicted
                    continue
                logprob = document_lps[ix_token].item()
                if token in selected_tokens:
                    model_logprobs.append(logprob)

        if model_logprobs:
            model_group_stats[model] = {
                "mean": np.mean(model_logprobs),
                "median": np.median(model_logprobs),
                "min": np.min(model_logprobs),
                "max": np.max(model_logprobs),
                "25th": np.percentile(model_logprobs, 25),
                "75th": np.percentile(model_logprobs, 75),
            }
    return model_group_stats


def dict_filter_quantile(
    d: dict[Any, float], q_start: float, q_end: float
) -> dict[Any, float]:
    if not (0 <= q_start < q_end <= 1):
        raise ValueError("Invalid quantile range")
    q_start_val = np.nanquantile(list(d.values()), q_start)
    q_end_val = np.nanquantile(list(d.values()), q_end)
    return {
        k: v for k, v in d.items() if q_start_val <= v <= q_end_val and not np.isnan(v)
    }


def get_all_tok_metrics_in_label(
    token_ids: Int[torch.Tensor, "prompt pos"],
    selected_tokens: list[int],
    metrics: torch.Tensor,
    q_start: Optional[float] = None,
    q_end: Optional[float] = None,
) -> dict[tuple[int, int], float]:
    """
    From the token_map, get all the positions of the tokens that have a certain label.
    We don't use the token_map because for sampling purposes, iterating through token_ids is more efficient.
    Optionally, filter the tokens based on the quantile range of the metrics.

    Args:
    - token_ids (Dataset): token_ids dataset e.g. token_ids[0] = {"tokens": [[1, 2, ...], [2, 5, ...], ...]}
    - selected_tokens (list[int]): list of token IDs to search for e.g. [46, 402, ...]
    - metrics (torch.Tensor): tensor of metrics to search through e.g. torch.tensor([[0.1, 0.2, ...], [0.3, 0.4, ...], ...])
    - q_start (float): the start of the quantile range to filter the metrics e.g. 0.1
    - q_end (float): the end of the quantile range to filter the metrics e.g. 0.9

    Returns:
    - tok_positions (dict[tuple[int, int], Number]): dictionary of token positions and their corresponding metrics
    """

    # check if metrics have the same dimensions as token_ids
    if metrics.shape != token_ids.shape:
        raise ValueError(
            f"Expected metrics to have the same shape as token_ids, but got {metrics.shape} and {token_ids.shape} instead."
        )

    tok_positions = {}
    for prompt_pos, prompt in enumerate(token_ids.numpy()):
        for tok_pos, tok in enumerate(prompt):
            if tok in selected_tokens:
                tok_positions[(prompt_pos, tok_pos)] = metrics[
                    prompt_pos, tok_pos
                ].item()

    if q_start is not None and q_end is not None:
        tok_positions = dict_filter_quantile(tok_positions, q_start, q_end)

    return tok_positions


def visualize_selected_tokens(
    input: dict[str | int, tuple[float, float, float]],
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

    def get_hovertexts(mid: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[str]:
        return [f"Loss: {m:.3f} ({l:.3f}, {h:.3f})" for m, l, h in zip(mid, lo, hi)]

    def get_plot_values() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.array([input[x] for x in input_x]).T
        means, err_lo, err_hi = x[0], x[1], x[2]
        return means, err_lo, err_hi

    means, err_lo, err_hi = get_plot_values()

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
