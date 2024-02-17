import argparse
import os

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from jaxtyping import Int
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from delphi.eval.utils import get_all_and_next_logprobs, load_validation_dataset

torch.set_grad_enabled(False)


def main(
    model_name: str,
    batch_size: Int,
    dataset_name: str,
    username: str,
    token: str,
    funct_test: bool = False,
):
    """
    Outputs the log probabilities of the next token for each token in the validation dataset.
    And uploads the resulting dataset to huggingface.
    Args:
    - model_name: The name of the model to use for inference
    - batch_size: The batch size for processing. 80 worked well in CPU.
    - dataset_name: The name of the dataset from which validation set will be loaded
    - username: Hugging Face API username
    - token: Hugging Face API token
    """
    val_ds = load_validation_dataset(dataset_name)

    # model accepts 2D tensors (batch_size, seq_len)
    val_sequences = torch.tensor([s["tokens"] for s in val_ds])

    if funct_test:
        val_sequences = val_sequences[:320]

    model = AutoModelForCausalLM.from_pretrained(model_name)

    logprobs_list = []
    for i in tqdm(range(0, len(val_sequences), batch_size)):
        batch_sequences = val_sequences[i : i + batch_size]
        _, next_logprobs = get_all_and_next_logprobs(model, batch_sequences)
        logprobs_list.append(next_logprobs)
    accumulated_logprobs = torch.cat(logprobs_list, dim=0)

    nan_tensor = torch.full((accumulated_logprobs.size(0), 1), float("nan"))
    extended_next_logprobs = torch.cat(
        [nan_tensor, accumulated_logprobs], dim=1
    )  # 513 tokens

    df_dataset = pd.DataFrame({"logprobs": extended_next_logprobs.tolist()})
    hf_dataset = Dataset.from_pandas(df_dataset)

    # change the repo_id to your hf username in generate_logprobs.sh
    # change the yout hf token in generate_logprobs.sh

    repo_id = f"{username}/{model_name.rsplit('/', 1)[-1]}-validation-logprobs"
    if funct_test:
        repo_id += "-funct-test"
    hf_dataset.push_to_hub(
        repo_id=repo_id,
        split="validation",
        private=False,
        token=token,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference and generate log probabilities."
    )
    parser.add_argument(
        "model_name", type=str, help="Model name with or without delphi-suite/ prefix"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=80,
        help="Batch size for processing (default: 80)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name with or without delphi-suite/ prefix",
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Hugging Face API username",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--test-funct", action="store_true", help="Enable test function mode"
    )

    args = parser.parse_args()

    if "/" not in args.model_name:
        args.model_name = "delphi-suite/" + args.model_name

    main(
        args.model_name,
        args.batch_size,
        args.dataset_name,
        args.username,
        args.token,
        args.test_funct,
    )
