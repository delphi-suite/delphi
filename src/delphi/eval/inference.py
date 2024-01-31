import argparse
import os

import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)


def get_correct_logprobs(model, samples_tok):
    # logits: seq, pos, d_vocab
    logits = model(samples_tok).logits
    # logprobs: [batch_size, seq_length, vocab_size]
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    # make probs a list of lists of correct token LOG probabilities.
    list_logprob = []
    for i, sample in enumerate(samples_tok):
        valid_length = len(sample) - 1  # Last token doesn't have a next token
        sample_logprobs = logprobs[i, :valid_length, :]  # [valid_length, vocab_size]

        # Extract the probabilities of the actual next tokens
        next_tokens = sample[
            1 : valid_length + 1
        ]  # Tokens that follow each token in the sequence
        correct_logprobs = sample_logprobs[torch.arange(valid_length), next_tokens]

        list_logprob.append(correct_logprobs)
    return list_logprob

    # outputs a list of lists of correct token LOG probabilities.
    # correct_logprobs = get_correct_logprobs(model, val_sequences[:10])


def main(model_name, dataset_split, batch_size):
    val_ds = load_dataset(
        "delphi-suite/tinystories-v2-clean-tokenized", split=dataset_split
    )
    # val_ds[0]["tokens"]    # access first sample

    # model accepts 2D tensors (batch_size, seq_len)
    val_sequences = torch.tensor([s["tokens"] for s in val_ds])
    val_sequences = val_sequences[:220]

    output_folder = "Correct_logprobs"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize an empty DataFrame to accumulate log probabilities
    accumulated_df = pd.DataFrame()

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Loop over the validation dataset in batches
    for i in tqdm(range(0, len(val_sequences), batch_size)):
        batch_sequences = val_sequences[i : i + batch_size]
        batch_logprobs = get_correct_logprobs(model, batch_sequences)
        # Convert batch log probabilities to a DataFrame
        batch_df = pd.DataFrame([logprob.tolist() for logprob in batch_logprobs])
        # Append the batch DataFrame to the accumulated DataFrame
        accumulated_df = pd.concat([accumulated_df, batch_df], ignore_index=True)

    # Save the accumulated DataFrame to a Parquet file
    output_file = os.path.join(output_folder, f'{model_name.replace("/", "-")}.parquet')
    accumulated_df.to_parquet(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference and generate log probabilities."
    )
    parser.add_argument(
        "model_name", type=str, help="Model name with or without delphi-suite/ prefix"
    )
    parser.add_argument(
        "dataset_split", type=str, help="Dataset split (e.g., train, validation, test)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=80,
        help="Batch size for processing (default: 80)",
    )

    args = parser.parse_args()

    # Default prefix handling
    if "/" not in args.model_name:
        args.model_name = "delphi-suite/" + args.model_name

    main(args.model_name, args.dataset_split, args.batch_size)
