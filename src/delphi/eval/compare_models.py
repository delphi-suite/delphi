from dataclasses import dataclass

import torch
from jaxtyping import Int
from transformers import PreTrainedModel

from delphi.eval.utils import get_all_and_next_logprobs_single


def identify_model(model: PreTrainedModel) -> str:
    return model.config.name_or_path


@dataclass
class TokenPrediction:
    token: int
    base_model_prob: float
    lift_model_prob: float


@dataclass
class NextTokenStats:
    base_model: str
    lift_model: str
    next_prediction: TokenPrediction
    topk: list[TokenPrediction]


def compare_models(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    sample_tok: Int[torch.Tensor, "seq"],
    top_k: int = 3,
) -> list[NextTokenStats | None]:
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

    logprobs_a, next_probs_a = get_all_and_next_logprobs_single(model_a, sample_tok)
    logprobs_b, next_probs_b = get_all_and_next_logprobs_single(model_b, sample_tok)

    probs_a = torch.exp(logprobs_a)
    probs_b = torch.exp(logprobs_b)

    top_k_b = torch.topk(probs_b, top_k, dim=-1)
    top_k_a_probs = torch.gather(probs_a, 1, top_k_b.indices)

    top_k_b_tokens = top_k_b.indices
    top_k_b_probs = top_k_b.values

    comparisons = []
    # ignore first token when evaluating predictions
    comparisons.append(None)

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
