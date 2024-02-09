from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from jaxtyping import Int
from transformers import PreTrainedModel

from delphi.eval.utils import get_correct_and_all_probs


@dataclass
class ModelId:
    model_name: str


def identify_model(model: PreTrainedModel) -> ModelId:
    return ModelId(model_name=model.config.name_or_path)


@dataclass
class TokenPrediction:
    token: int
    base_model_prob: float
    lift_model_prob: float


@dataclass
class NextTokenStats:
    base_model: ModelId
    lift_model: ModelId
    next_prediction: TokenPrediction
    topk: list[TokenPrediction]


def _pad_start(tensor: torch.Tensor) -> torch.Tensor:
    value_to_prepend = -1
    if len(tensor.shape) == 1:
        return torch.cat((torch.tensor([value_to_prepend]), tensor))
    else:
        # input: 2D tensor of shape [seq_len - 1, top_k]
        pre = torch.full((1, tensor.size()[-1]), value_to_prepend)
        return torch.cat((pre, tensor), dim=0)


def compare_models(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    sample_tok: Int[torch.Tensor, "seq"],
    top_k: int = 3,
) -> list[NextTokenStats]:
    """
    Compare the probabilities of the next token for two models and get the top k token predictions according to model B.
    Args:
    - model_a: The first model (assumed to be the base model)
    - model_b: The second model (assumed to be the improved model)
    - sample_tok: The tokenized prompt
    - top_k: The number of top token predictions to retrieve (default is 5)
    Returns:
        A list of NextTokenStats objects, one for each token in the prompt.
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

    comparisons = []

    for next_p_a, next_p_b, top_toks_b, top_probs_a, top_probs_b in zip(
        next_probs_a, next_probs_b, top_k_b_tokens, top_k_a_probs, top_k_b_probs
    ):
        nts = NextTokenStats(
            base_model=identify_model(model_a),
            lift_model=identify_model(model_b),
            next_prediction=TokenPrediction(
                token=int(next_p_a.item()),
                base_model_prob=next_p_a.item(),
                lift_model_prob=next_p_b.item(),
            ),
            topk=[
                TokenPrediction(
                    token=int(top_toks_b[i].item()),
                    base_model_prob=top_probs_a[i].item(),
                    lift_model_prob=top_probs_b[i].item(),
                )
                for i in range(top_k)
            ],
        )
        comparisons.append(nts)

    return comparisons
