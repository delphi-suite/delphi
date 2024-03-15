import uuid
from typing import cast

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


def to_tok_prob_str(tok: int, prob: float, tokenizer: PreTrainedTokenizerBase) -> str:
    tok_str = tokenizer.decode(tok).replace(" ", "&nbsp;").replace("\n", r"\n")
    prob_str = f"{prob:.2%}"
    return f"{prob_str:>6} |{tok_str}|"


def token_to_html(
    token: int,
    tokenizer: PreTrainedTokenizerBase,
    bg_color: str,
    data: dict,
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

    style_str = data_str = ""
    # converting style dict into the style attribute
    if specific_styles:
        inside_style_str = "; ".join(f"{k}: {v}" for k, v in specific_styles.items())
        style_str = f" style='{inside_style_str}'"
    if data:
        data_str = "".join(
            f" data-{k}='{v.replace(' ', '&nbsp;')}'" for k, v in data.items()
        )
    return f"<div class='token'{style_str}{data_str}>{str_token}</div>{br}"


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
_token_style_str = " ".join([f"{k}: {v};" for k, v in _token_style.items()])


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
            token_to_html(tok, tokenizer, bg_color=colors[i], data=data).replace(
                "class='token'", f"class='{token_class}'"
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


def vis_sample_diff_probs(
    sample_tok: list[Int[torch.Tensor, "pos"]],
    diff_probs: list[Float[torch.Tensor, "pos"]],
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    # I have to use list instead of torch.Tensor because the lengths of the sequences are different
    # TODO: see if there's a way to do this with torch tensors
    token_htmls = []

    unique_id = str(uuid.uuid4())

    token_class = f"token_{unique_id}"
    hover_div_id = f"hover_info_{unique_id}"

    for i in range(len(sample_tok)):
        colors = probs_to_colors(diff_probs[i])
        for j in range(len(sample_tok[i])):
            tok = cast(int, sample_tok[i][j].item())
            data = {}
            if j == len(sample_tok[i]) - 1:
                # only annotate the last token
                diff_prob = diff_probs[i][j].item()
                data["diff_prob"] = to_tok_prob_str(tok, diff_prob, tokenizer)

            token_htmls.append(
                # color + 1 to compensate for the endoftext token
                token_to_html(
                    tok, tokenizer, bg_color=colors[j + 1], data=data
                ).replace("class='token'", f"class='{token_class}'")
            )

        # add a space between prompts and completions
        token_htmls.append("<br> <br>")

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
