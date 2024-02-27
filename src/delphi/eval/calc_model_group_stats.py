import numpy as np


def calc_model_group_stats(
    tokenized_corpus_dataset: list,
    logprobs_by_dataset: dict[str, list[list[float]]],
    token_labels_by_token: dict[int, dict[str, bool]],
    token_labels: list[str],
) -> dict[tuple[str, str], dict[str, float]]:
    """
    For each (model, token group) pair, calculate useful stats (for visualization)

    args:
    - tokenized_corpus_dataset: the tokenized corpus dataset, e.g. load_dataset(constants.tokenized_corpus_dataset))["validation"]
    - logprob_datasets: a dict of lists of logprobs, e.g. {"llama2": load_dataset("transcendingvictor/llama2-validation-logprobs")["validation"]["logprobs"]}
    - token_groups: a dict of token groups, e.g. {0: {"Is Noun": True, "Is Verb": False, ...}, 1: {...}, ...}
    - models: a list of model names, e.g. constants.LLAMA2_MODELS
    - token_labels: a list of token group descriptions, e.g. ["Is Noun", "Is Verb", ...]

    returns: a dict of (model, token group) pairs to a dict of stats,
        e.g. {("llama2", "Is Noun"): {"mean": -0.5, "median": -0.4, "min": -0.1, "max": -0.9, "25th": -0.3, "75th": -0.7}, ...}

    Technically `models` and `token_labels` are redundant, as they are also keys in `logprob_datasets` and `token_groups`,
    but it's better to be explicit

    stats calculated: mean, median, min, max, 25th percentile, 75th percentile
    """
    model_group_stats = {}
    for model in logprobs_by_dataset:
        group_logprobs = {}
        print(f"Processing model {model}")
        dataset = logprobs_by_dataset[model]
        for ix_doc_lp, document_lps in enumerate(dataset):
            tokens = tokenized_corpus_dataset[ix_doc_lp]["tokens"]
            for ix_token, token in enumerate(tokens):
                if ix_token == 0:  # skip the first token, which isn't predicted
                    continue
                logprob = document_lps[ix_token]
                for token_group_desc in token_labels:
                    if token_labels_by_token[token][token_group_desc]:
                        if token_group_desc not in group_logprobs:
                            group_logprobs[token_group_desc] = []
                        group_logprobs[token_group_desc].append(logprob)
        for token_group_desc in token_labels:
            if token_group_desc in group_logprobs:
                model_group_stats[(model, token_group_desc)] = {
                    "mean": np.mean(group_logprobs[token_group_desc]),
                    "median": np.median(group_logprobs[token_group_desc]),
                    "min": np.min(group_logprobs[token_group_desc]),
                    "max": np.max(group_logprobs[token_group_desc]),
                    "25th": np.percentile(group_logprobs[token_group_desc], 25),
                    "75th": np.percentile(group_logprobs[token_group_desc], 75),
                }
    return model_group_stats
