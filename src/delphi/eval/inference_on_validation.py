# %% load validation dataset
from datasets import load_dataset
from tqdm.auto import tqdm

val_ds = load_dataset("delphi-suite/tinystories-v2-clean-tokenized", split="validation")

# %% Models are stored in "model" variable
import torch

torch.set_grad_enabled(False)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "delphi-suite/delphi-llama2-100k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# %% get their logits in parallel
def get_logits(model, samples_tok: list[str]) -> torch.Tensor:
    # Check that samples_tok is a list of sequences
    assert all(
        isinstance(seq, (list, torch.Tensor)) for seq in samples_tok
    ), "samples_tok must be a list of sequences"
    # Check that each tensor in samples_tok is 1D, and print debug info if not
    for seq in samples_tok:
        if isinstance(seq, torch.Tensor) and seq.ndim != 1:
            print(f"Found a non-1D tensor: {seq}")
            print(f"Shape: {seq.shape}")
            raise AssertionError("All tensors must be 1D")

    padded_matrix_samples = pad_sequences(samples_tok)
    logits = model(padded_matrix_samples).logits
    return logits  # (num_seqs, max_seq_len, vocab_size)


logits = get_logits(model, val_ds[:10])
print(logits.shape)


# %%


def get_correct_logprobs(model, samples_tok):
    # logits: seq, pos, d_vocab
    logits = get_logits(model, samples_tok)
    # probs: seq, pos, d_vocab
    probs = torch.softmax(logits, dim=-1)
    logprobs = torch.log(probs)

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
