from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Int  # Add the missing import statement

from delphi.vis.utils import get_correct_and_all_probs


@dataclass
class ModelComparison:
    correct_prob_base_model: torch.Tensor
    correct_prob_lift_model: torch.Tensor
    top_k_tokens_lift_model: torch.Tensor
    top_k_probs_base_model: torch.Tensor
    top_k_probs_lift_model: torch.Tensor


def _pad_start(tensor: torch.Tensor) -> torch.Tensor:
    value_to_prepend = -1
    if len(tensor.shape) == 1:
        return torch.cat((torch.tensor([value_to_prepend]), tensor))
    else:
        # input: 2D tensor of shape [seq_len - 1, top_k]
        pre = torch.full((1, tensor.size()[-1]), value_to_prepend)
        return torch.cat((pre, tensor), dim=0)


def compare_models(
    model_a: nn.Module,
    model_b: nn.Module,
    sample_tok: Int[torch.Tensor, "pos"],
    top_k: int = 3,
) -> ModelComparison:
    """
    Compare the probabilities of the next token for two models and get the top k token predictions according to model B.
    Args:
    - model_a: The first model (assumed to be the base model)
    - model_b: The second model (assumed to be the improved model)
    - tokens: The tokenized prompt
    - top_k: The number of top token predictions to retrieve (default is 5)
    Returns:
    - A ModelComparison with tensors for:
        - The probabilities of the actual next token according to model A
        - The probabilities of the actual next token according to model B
        - The top k token predictions according to model B
        - The probabilities of these tokens according to model A
        - The probabilities of these tokens according to model B
    Tensors are aligned to the token they are predicting (by prepending a -1 to the start of the tensor)
    """
    assert (
        model_a.device == model_b.device
    ), "Both models must be on the same device for comparison."

    device = model_a.device
    sample_tok = sample_tok.to(device)

    next_probs_a, probs_a = get_correct_and_all_probs(model_a, sample_tok)
    next_probs_b, probs_b = get_correct_and_all_probs(model_b, sample_tok)

    top_k_b = torch.topk(probs_b, top_k, dim=-1)
    top_k_a_probs = torch.gather(probs_a, 1, top_k_b.indices)

    next_probs_a = _pad_start(next_probs_a)
    next_probs_b = _pad_start(next_probs_b)
    top_k_b_tokens = _pad_start(top_k_b.indices)
    top_k_a_probs = _pad_start(top_k_a_probs)
    top_k_b_probs = _pad_start(top_k_b.values)

    return ModelComparison(
        correct_prob_base_model=next_probs_a,
        correct_prob_lift_model=next_probs_b,
        top_k_tokens_lift_model=top_k_b_tokens,
        top_k_probs_base_model=top_k_a_probs,
        top_k_probs_lift_model=top_k_b_probs,
    )
