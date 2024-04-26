#!/usr/bin/env python3
import argparse
from collections.abc import Iterable

import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import trange
from transformers import AutoModelForCausalLM

from delphi import utils
from delphi.eval.utils import get_all_and_next_logprobs

torch.set_grad_enabled(False)


def main(
    in_model_repo_id: str,
    revisions: Iterable[str],
    in_dataset_repo_id: str,
    split: str,
    feature: str,
    batch_size: int,
    out_repo_id: str,
):
    """
    Outputs the log probabilities of the next token for each token in the dataset.
    And uploads the resulting dataset to huggingface.
    """
    in_dataset_split = utils.load_dataset_split_sequence_int32_feature(
        in_dataset_repo_id, split, feature
    )
    in_dataset_split.set_format("torch")
    for revision in revisions:
        print(f"Loading model={in_model_repo_id}, {revision=}")
        model = AutoModelForCausalLM.from_pretrained(
            in_model_repo_id, revision=revision
        )
        logprobs_dataset = get_logprobs_single_model(
            model=model,
            dataset=in_dataset_split,
            feature=feature,
            batch_size=batch_size,
        )
        logprobs_dataset.push_to_hub(
            repo_id=out_repo_id,
            split=utils.hf_split_to_split_name(split),
            revision=revision,
        )


def get_logprobs_single_model(
    model: AutoModelForCausalLM,
    dataset: Dataset,
    feature: str,
    batch_size: int,
) -> Dataset:
    n_seq = len(dataset)
    seq_len = len(dataset[0][feature])
    logprobs = np.empty((n_seq, seq_len))
    logprobs[:, 0] = float("nan")
    print("Running inference...")
    for i in trange(0, n_seq, batch_size):
        batch_tokens = dataset[i : i + batch_size][feature]
        logprobs[i : i + batch_size, 1:] = (
            get_all_and_next_logprobs(model, batch_tokens)[1].cpu().numpy()  # type: ignore
        )
    return Dataset.from_dict({"logprobs": [row for row in logprobs]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference and generate log probabilities."
    )
    parser.add_argument(
        "--in-model-repo-id",
        "--im",
        type=str,
        required=True,
        help="The model",
    )
    parser.add_argument(
        "--revisions",
        "-r",
        help="comma separated revisions of the model to use or 'ALL_BRANCHES' to use all branches",
        type=str,
        default="main",
        required=False,
    )

    parser.add_argument(
        "--in-dataset-repo-id",
        "--id",
        type=str,
        required=True,
        help="The tokenized dataset",
    )
    parser.add_argument(
        "--feature",
        "-f",
        type=str,
        required=True,
        help="Name of the column containing token sequences in the input dataset",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        required=True,
        help="Split of the tokenized dataset, supports slicing like 'train[:10%%]'",
    )
    parser.add_argument(
        "--out-repo-id",
        "-o",
        type=str,
        required=True,
        help="Where to upload the next logprobs",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=80,
        help="How many sequences to evaluate at once",
    )
    # TODO
    # parser.add_argument(
    #     "--chunk-size",
    #     "-c",
    #     type=int,
    #     default=200_000,
    #     help="Size of the parquet chunks uploaded to HuggingFace",
    # )
    args = parser.parse_args()

    revisions = (
        args.revisions.split(",")
        if args.revisions != "ALL_BRANCHES"
        else utils.get_all_hf_branch_names(args.in_model_repo_id)
    )

    main(
        in_model_repo_id=args.in_model_repo_id,
        revisions=revisions,
        in_dataset_repo_id=args.in_dataset_repo_id,
        split=args.split,
        feature=args.feature,
        batch_size=args.batch_size,
        out_repo_id=args.out_repo_id,
    )
