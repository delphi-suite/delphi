import math
import random
import uuid
from typing import cast

import numpy as np
import panel as pn
import torch
from IPython.core.display import HTML
from IPython.core.display_functions import display
from jaxtyping import Float, Int
from transformers import PreTrainedTokenizerBase


def probs_to_colors(probs: Float[torch.Tensor, "next_pos"]) -> list[str]:
    # for the endoftext token
    # no prediction, no color
    colors = ["white"]
    for p in probs.tolist():
        red_gap = 150  # the higher it is, the less red the tokens will be
        green_blue_val = red_gap + int((255 - red_gap) * (1 - p))
        colors.append(f"rgb(255, {green_blue_val}, {green_blue_val})")
    return colors


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


def to_tok_prob_str(tok: int, prob: float, tokenizer: PreTrainedTokenizerBase) -> str:
    tok_str = tokenizer.decode(tok).replace(" ", "&nbsp;").replace("\n", r"\n")
    prob_str = f"{prob:.2%}"
    return f"{prob_str:>6} |{tok_str}|"


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


def vis_sample_prediction_probs(
    sample_tok: Int[torch.Tensor, "pos"],
    correct_probs: Float[torch.Tensor, "pos"],
    top_k_probs: torch.return_types.topk,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    colors = probs_to_colors(correct_probs)
    token_htmls = []

    # Generate a unique ID for this instance (so we can have multiple instances on the same page)
    unique_id = str(uuid.uuid4())

    token_class = f"token_{unique_id}"
    hover_div_id = f"hover_info_{unique_id}"

    for i in range(sample_tok.shape[0]):
        tok = cast(int, sample_tok[i].item())
        data = {}
        if i > 0:
            correct_prob = correct_probs[i - 1].item()
            data["next"] = to_tok_prob_str(tok, correct_prob, tokenizer)
            top_k_probs_tokens = top_k_probs.indices[i - 1]
            top_k_probs_values = top_k_probs.values[i - 1]
            for j in range(top_k_probs_tokens.shape[0]):
                top_tok = top_k_probs_tokens[j].item()
                top_tok = cast(int, top_tok)
                top_prob = top_k_probs_values[j].item()
                data[f"top{j}"] = to_tok_prob_str(top_tok, top_prob, tokenizer)

        token_htmls.append(
            token_to_html(
                tok, tokenizer, bg_color=colors[i], data=data, class_name=token_class
            )
        )

    html_str = f"""
    <style>.{token_class} {{ {_token_style_str} }} #{hover_div_id} {{ height: 100px; font-family: monospace; }}</style>
    {"".join(token_htmls)} <div id='{hover_div_id}'></div>
    <script>
        (function() {{
            var token_divs = document.querySelectorAll('.{token_class}');
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
    return html_str


def vis_pos_map(
    pos_list: list[tuple[int, int]],
    selected_tokens: list[int],
    metrics: Float[torch.Tensor, "prompt pos"],
    token_ids: Int[torch.Tensor, "prompt pos"],
    tokenizer: PreTrainedTokenizerBase,
    sample: int = 3,
):
    """
    Randomly sample from pos_map and visualize the loss diff at the corresponding position.
    """

    token_htmls = []
    unique_id = str(uuid.uuid4())
    token_class = f"pretoken_{unique_id}"
    selected_token_class = f"token_{unique_id}"
    hover_div_id = f"hover_info_{unique_id}"

    # choose n random keys from pos_map
    keys = random.sample(pos_list, k=sample)

    for key in keys:
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
    return html_str


def token_selector(
    vocab_map: dict[str, int]
) -> tuple[pn.widgets.MultiChoice, list[int]]:
    tokens = list(vocab_map.keys())
    token_selector = pn.widgets.MultiChoice(name="Tokens", options=tokens)
    token_ids = [vocab_map[token] for token in cast(list[str], token_selector.value)]

    def update_tokens(event):
        token_ids.clear()
        token_ids.extend([vocab_map[token] for token in event.new])

    token_selector.param.watch(update_tokens, "value")
    return token_selector, token_ids
