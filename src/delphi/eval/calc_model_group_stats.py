import numpy as np
from jaxtyping import Float


def calc_model_group_stats(
    tokenized_corpus_dataset: list,
    logprobs_by_dataset: dict[str, list[list[float]]],
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
                logprob = document_lps[ix_token]
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
