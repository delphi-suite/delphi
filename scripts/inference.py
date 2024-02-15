import argparse
import os

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from delphi.eval.utils import get_all_and_next_logprobs, load_validation_dataset

torch.set_grad_enabled(False)


def main(model_name, batch_size, dataset_name, token):
    val_ds = load_validation_dataset(dataset_name)

    # model accepts 2D tensors (batch_size, seq_len)
    val_sequences = torch.tensor([s["tokens"] for s in val_ds])
    model = AutoModelForCausalLM.from_pretrained(model_name)
    logprobs, next_logprobs = get_all_and_next_logprobs(model, val_sequences)

    df_dataset = pd.DataFrame({"logprobs": next_logprobs.tolist()})
    hf_dataset = Dataset.from_pandas(df_dataset)

    # change the repo_id to your hf username
    # change the token in generate_logprobs.sh
    hf_dataset.push_to_hub(
        repo_id=f"transcendingvictor/{model_name.rsplit('/', 1)[-1]}-validation-logprobs",
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
        "--batch_size",
        type=int,
        default=80,
        help="Batch size for processing (default: 80)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name with or without delphi-suite/ prefix",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token",
    )
    args = parser.parse_args()

    if "/" not in args.model_name:
        args.model_name = "delphi-suite/" + args.model_name

    main(args.model_name, args.batch_size, args.dataset_name, args.token)
