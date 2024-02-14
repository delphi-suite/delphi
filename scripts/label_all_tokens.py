import pickle
import sys
from pathlib import Path

from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from delphi.eval import token_labelling


def tokenize(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, sample_txt: str
) -> int:
    # supposedly this can be different than prepending the bos token id
    return tokenizer.encode(tokenizer.bos_token + sample_txt, return_tensors="pt")[0]


# Decode a sentence
def decode(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, token_ids: list[int]
) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def main():
    print("\n", " LABEL ALL TOKENS ".center(50, "="), "\n")
    # Access command-line arguments
    args = sys.argv[1:]
    # Directory to save the results
    SAVE_DIR = Path("src/delphi/eval/")

    # Check if arguments are provided
    if len(args) == 0:
        print("No arguments provided.")
        return

    if len(args) > 1:
        print("Too many arguments provided.")
        return

    # Process arguments
    model_name = args[0]

    print(f"You chose the model: {model_name}\n")
    print(
        f"The language model will be loaded from Huggingface and its tokenizer used to do two things:\n\t1) Create a list of all tokens in the tokenizer's vocabulary.\n\t2) Label each token with its part of speech, dependency, and named entity recognition tags.\nThe respective results will be saved to files located at: '{SAVE_DIR}'\n"
    )

    # ================ (1) =================
    print("(1) Create a list of all tokens in the tokenizer's vocabulary ...")

    # Load the tokenizer from Huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    print("Loaded the tokenizer.\nThe vocab size is:", vocab_size)

    # Create a list of all tokens in the tokenizer's vocabulary
    tokens_str = ""  # will hold all tokens and their ids
    for i in range(tokenizer.vocab_size):
        tokens_str += f"{i},{decode(tokenizer, i)}\n"

    # Save the list of all tokens to a file
    filename = "all_tokens_list.txt"
    filepath = SAVE_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(tokens_str)

    print(f"Saved the list of all tokens to:\n\t{filepath}\n")

    # ================ (2) =================
    print("(2) Label each token ...")

    # let's label each token
    labelled_token_ids_dict: dict[int, dict[str, bool]] = {}  # token_id: labels
    max_token_id = tokenizer.vocab_size  # stop at which token id, vocab size
    batch_size = 500
    # we iterate (batchwise) over all token_ids, individually takes too much time
    for start in tqdm(range(0, max_token_id, batch_size), desc="Labelling tokens"):
        # create a batch of token_ids
        end = min(start + batch_size, max_token_id)
        token_ids = list(range(start, end))
        # decode the token_ids to get a list of tokens, a 'sentence'
        tokens = decode(tokenizer, token_ids)  # list of tokens == sentence
        # put the sentence into a list, to make it a batch of sentences
        sentences = [tokens]
        # label the batch of sentences
        labels = token_labelling.label_batch_sentences(
            sentences, tokenized=True, verbose=False
        )
        # create a dict with the token_ids and their labels
        labelled_sentence_dict = dict(zip(token_ids, labels[0]))
        # update the labelled_token_ids_dict with the new dict
        labelled_token_ids_dict.update(labelled_sentence_dict)

    # Save the labelled tokens to a file
    filename = "labelled_token_ids_dict_.pkl"
    filepath = SAVE_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(labelled_token_ids_dict, f)

    print(f"Saved the labelled tokens to:\n\t{filepath}\n")

    # sanity check that The pickled and the original dict are the same
    print("Sanity check ...", end="")
    # load pickle
    with open(filepath, "rb") as f:
        pickled = pickle.load(f)
    # compare
    assert labelled_token_ids_dict == pickled
    print(" completed.")

    print(" END ".center(50, "="))


if __name__ == "__main__":
    main()
