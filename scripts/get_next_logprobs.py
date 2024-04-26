#!/usr/bin/env python3
import argparse

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
    model = AutoModelForCausalLM.from_pretrained(in_model_repo_id)
    in_dataset_split = utils.load_dataset_split_sequence_int32_feature(
        in_dataset_repo_id, split, feature
    )
    in_dataset_split.set_format("torch")
    n_seq = len(in_dataset_split)
    seq_len = len(in_dataset_split[0][feature])
    logprobs = np.empty((n_seq, seq_len))
    logprobs[:, 0] = float("nan")
    print("Running inference...")
    for i in trange(0, n_seq, batch_size):
        batch_tokens = in_dataset_split[i : i + batch_size][feature]
        logprobs[i : i + batch_size, 1:] = (
            get_all_and_next_logprobs(model, batch_tokens)[1].cpu().numpy()
        )

    hf_dataset = Dataset.from_dict({"logprobs": [row for row in logprobs]})

    hf_dataset.push_to_hub(
        repo_id=out_repo_id,
        split=utils.hf_split_to_split_name(split),
    )


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

    main(
        in_model_repo_id=args.in_model_repo_id,
        in_dataset_repo_id=args.in_dataset_repo_id,
        split=args.split,
        feature=args.feature,
        batch_size=args.batch_size,
        out_repo_id=args.out_repo_id,
    )
