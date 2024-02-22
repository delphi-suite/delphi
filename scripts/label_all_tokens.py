import argparse
import pickle

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from delphi.constants import STATIC_ASSETS_DIR
from delphi.eval import token_labelling


def tokenize(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, sample_txt: str
) -> int:
    # supposedly this can be different than prepending the bos token id
    return tokenizer.encode(tokenizer.bos_token + sample_txt, return_tensors="pt")[0]


# Decode a sentence
def decode(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, token_ids: int | list[int]
) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def main():
    # Setup argparse
    parser = argparse.ArgumentParser(description="Tokenization and labeling utility.")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to use for tokenization and labeling.",
        default="delphi-suite/delphi-llama2-100k",
        required=False,
    )
    parser.add_argument(
        "--save_dir", type=str, help="Directory to save the results.", required=True
    )
    parser.add_argument(
        "--output_format",
        type=str,
        help="Format to save the results in. Options: csv, pkl. Default: csv.",
        default="csv",
        required=False,
    )
    args = parser.parse_args()

    # Access command-line arguments
    # Directory to save the results
    SAVE_DIR = Path(args.save_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)  # create directory if it does not exist
    model_name = args.model_name
    output_format = args.output_format
    assert output_format in [
        "csv",
        "pkl",
    ], f"Invalid output format. Allowed: csv, pkl. Got: {output_format}"

    print("\n", " LABEL ALL TOKENS ".center(50, "="), "\n")
    print(f"You chose the model: {model_name}\n")
    print(
        f"The language model will be loaded from Huggingface and its tokenizer used to do two things:\n\t1) Create a list of all tokens in the tokenizer's vocabulary.\n\t2) Label each token with its part of speech, dependency, and named entity recognition tags.\nThe respective results will be saved to files located at: '{STATIC_ASSETS_DIR}'\n"
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
    # filepath = SAVE_DIR / filename # TODO: use the static files of python module
    filepath = Path("src/delphi/eval/") / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(tokens_str)

    print(f"Saved the list of all tokens to:\n\t{filepath}\n")

    # ================ (2) =================
    print("(2) Label each token ...")

    # let's label each token
    labelled_token_ids_dict: dict[int, dict[str, bool]] = {}  # token_id: labels
    max_token_id = tokenizer.vocab_size  # stop at which token id, vocab size
    # we iterate over all token_ids individually
    for token_id in tqdm(range(0, max_token_id), desc="Labelling tokens"):
        # decode the token_ids to get a list of tokens, a 'sentence'
        tokens = decode(tokenizer, token_id)  # list of tokens == sentence
        # put the sentence into a list, to make it a batch of sentences
        sentences = [tokens]
        # label the batch of sentences
        labels = token_labelling.label_batch_sentences(
            sentences, tokenized=True, verbose=False
        )
        # create a dict with the token_ids and their labels
        # update the labelled_token_ids_dict with the new dict
        labelled_token_ids_dict[token_id] = labels[0][0]

    if output_format == "pkl":
        # Save the labelled tokens to a file
        filename = "labelled_token_ids_dict.pkl"
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

    # ----------- CSV ------------------------
    if output_format == "csv":
        print("\nCreating the CSV ...")
        # Create a pandas dataframe / CSV from the label dict
        df = pd.DataFrame(
            labelled_token_ids_dict.items(), columns=["token_id", "label"]
        )
        # split the label column into multiple columns
        df = df.join(pd.DataFrame(df.pop("label").tolist()))
        # Change datatype of columns to float
        df = df.astype(int)

        print("Sanity check pandas csv ...", end="")
        # Perform sanity check, that the table was created correctly
        for row_index, row_values in df.iterrows():
            token_id = row_values.iloc[0]
            label_pandas = list(
                row_values.iloc[1:]
            )  # we exclude the token_id from the colum
            label_dict = list(labelled_token_ids_dict[token_id].values())[:]
            assert (
                label_pandas == label_dict
            ), f"The dataframes are not equal for row {token_id}\n{label_pandas}\n{label_dict}"
        print(" completed.")

        # save the dataframe to a csv
        filename = "labelled_token_ids_df.csv"
        filepath = SAVE_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"Saved the labelled tokens as CSV to:\n\t{filepath}\n")

    print(" END ".center(50, "="))


if __name__ == "__main__":
    main()
