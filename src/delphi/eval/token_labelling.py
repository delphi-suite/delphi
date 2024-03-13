# %%
import pickle
from typing import Callable, Optional

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from delphi.constants import STATIC_ASSETS_DIR

# %%
# Load the token map from hf
token_map = load_dataset("delphi-suite/v0-token-map", split="validation")

tokenizer = AutoTokenizer.from_pretrained("delphi-suite/stories-tokenizer")


def decode(token_id: int, tokenizer: PreTrainedTokenizer = tokenizer) -> str:
    return tokenizer.decode([1, token_id])[3:]
    # decode with the <s> token, then remove it
    # to work around the tokenizer not outputing spaced at the beggining


def build_token_id_mapping(
    tokenizer: PreTrainedTokenizer = tokenizer,
) -> dict[str, int]:
    vocab_size = tokenizer.vocab_size
    # Preallocate the dictionary with the expected size for efficiency.
    token_id_mapping = {}
    for token_id in range(vocab_size):
        token_str = decode(token_id, tokenizer)
        token_id_mapping[token_str] = token_id
    return token_id_mapping


# NOTE: len(token_id_mapping) is 3896 (not vocab_size=4096)
# because several token_id share the same string representation

token_id_mapping = build_token_id_mapping()
# this dict aims to substitute the messy tokenizer


def encode(token: str, token_id_mapping: dict = token_id_mapping) -> int:
    return token_id_mapping.get(token)


# Example/problem:  "cut" = 3221       tokenizer.encode("cut") -> [1, 4023, 1242]
#                   "_cut" = 1242      tokenizer.encode(" cut") -> [1, 1242]
# NOTE: 1 is the <BOS> token (aka <s>), and 4023 is the space string " " token


def starts_with_space(token: str) -> bool:
    return token.startswith(" ") if token else False


def is_capitalized(token: str) -> bool:  # doesn't work for the empty string
    if token != "":
        capitalized = token[0].isupper()
    else:  # for the empty string
        capitalized = False

    if starts_with_space(token) and len(token) > 1:
        space_and_capitalized = token[1].isupper()
    else:
        space_and_capitalized = False
    return capitalized or space_and_capitalized
    # " Hello" -> True


TOKEN_LABELS = {"Starts with space": starts_with_space, "Capitalized": is_capitalized}
# dictionary of functions to label tokens, to be extended.


def label_single_token(token: str) -> dict[str, bool]:
    """
    Labels a single token. A token, that has been analyzed by the spaCy
    library.

    Parameters
    ----------
    token : Token | None
        The token to be labelled.

    Returns
    -------
    dict[str, bool]
        Returns a dictionary with the token's labels as keys and their
        corresponding boolean values.
    """
    labels = dict()  #  The dict holding labels of a single token
    # if token is None, then it is a '' empty string token or similar
    # if token is None or token == "":
    #     for label_name, category_check in TOKEN_LABELS.items():
    #         labels[label_name] = False
    #     labels["Is Other"] = True
    #     return labels
    # all other cases / normal tokens
    for label_name, category_check in TOKEN_LABELS.items():
        labels[label_name] = category_check(token)
    return labels


def label_tokens_from_tokenizer(
    tokenizer: PreTrainedTokenizer = tokenizer,
) -> list[dict[str, bool]]:
    """
    Labels all tokens in a tokenizer's vocabulary with the corresponding token categories.
    Returns two things: 1) `tokens_str`, a string where each token comprises 'token_id,token_str\n' and 2) `labelled_token_ids_dict` a dict that contains for each token_id (key) the corresponding token labels, which is in turn a dict, whith the label categories as keys and their boolean values as the dict's values.

    Parameters
    ----------
    tokenizer : The tokenizer with its tokens to be labelled, defaults to the global tokenizer.

    Returns
    -------
    tokens_str, labelled_token_ids_dict

    """
    vocab_size = tokenizer.vocab_size

    list_of_dicts = [None] * vocab_size  # token_id: labels
    # we iterate over each
    for token_id in tqdm(range(vocab_size), desc="Labelling tokens"):
        token = decode(token_id, tokenizer)
        labels = label_single_token(token)
        # create a dict with the token_ids and their labels
        # update the labelled_token_ids_dict with the new dict
        # first sentence of batch, label of first token
        list_of_dicts[token_id] = labels

    return list_of_dicts


token_repr = [
    repr(decode(token_id, tokenizer)).replace(" ", "_") for token_id in range(4096)
]  # this adds quotations to each token string (so it's 2 character longer)
# this avoids formating issues when exporting to csv
# the spaces of the tokens are replaced by underscores

token_instances = [
    len(x) if isinstance(x, list) else 0 for x in token_map["prompt_pos_idx"]
]  # list of a token's instances in the dataset (takes 20s to run)
# empty lists are considered as NoneType after .push_to_hub()


def export_csv(
    list_of_dicts: list[dict[str, bool]],
    token_repr: list[str] = token_repr,
    token_instances: list[int] = token_instances,
) -> pd.DataFrame:
    df = pd.DataFrame(list_of_dicts)
    # Reset the index so it becomes a column, and then rename the column to 'token_id'
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Token ID"}, inplace=True)
    df.insert(loc=1, column="String Representation", value=token_repr)
    df.insert(loc=2, column="Token Instances", value=token_instances)

    df.to_csv(STATIC_ASSETS_DIR / "token_labels.csv", index=False)


def import_csv_as_list_of_dicts(path: str) -> list[dict]:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(path)

    # Remove the columns that won't be in the dictionaries
    df.drop(
        columns=["String Representation", "Token Instances", "Token ID"], inplace=True
    )

    # Convert the DataFrame back to a list of dictionaries
    list_of_dicts = df.to_dict("records")

    return list_of_dicts


# To use the csv file
# run: export_csv(label_tokens_from_tokenizer())

# test whether the list_of_dicts is the same after imported/exported
# list_of_dicts = label_tokens_from_tokenizer()
# returned_list_of_dicts = import_csv_as_list_of_dicts(STATIC_ASSETS_DIR / "token_labels.csv")
