import torch
from beartype.typing import Mapping
from jaxtyping import Float


def next_logprobs_dict_to_loss_dict(
    next_logprobs_dict: Mapping[str, Float[torch.Tensor, "prompt tok"]]
) -> dict[str, Float[torch.Tensor, "prompt tok"]]:
    """
    Given a dictionary of log probabilities, calculate the loss for each token.

    args:
    - next_logprobs_dict: a dictionary of log probabilities, e.g. {"model1": [[-0.1, -0.2, ...], [-0.3, -0.4, ...], ...], "model2": [[-0.1, -0.2, ...], [-0.3, -0.4, ...], ...], ...}

    returns: a dictionary of losses, e.g. {"model1": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...], "model2": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...], ...}
    """
    return {
        model: torch.neg(next_logprobs)
        for model, next_logprobs in next_logprobs_dict.items()
    }
