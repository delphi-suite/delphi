import argparse
import os

import numpy as np
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
    """
    val_ds = load_validation_dataset(dataset_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    total_sequences = (
        len(val_ds) if not funct_test else 320
    )  # Use only 320 sequences if funct_test is True

    logprobs = np.empty((total_sequences, 513))
    logprobs[:, 0] = float("nan")
    for i in tqdm(range(0, total_sequences, batch_size)):
        batch_end = min(i + batch_size, total_sequences)
        batch_sequences = [val_ds[j]["tokens"] for j in range(i, batch_end)]
        batch_sequences_tensor = torch.tensor(batch_sequences)

        logprobs_tensor = get_all_and_next_logprobs(model, batch_sequences_tensor)[1]
        logprobs[i:batch_end, 1:] = logprobs_tensor.cpu().numpy()

    df_dataset = pd.DataFrame({"logprobs": [row for row in logprobs]})
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
        args.test_funct,
    )
