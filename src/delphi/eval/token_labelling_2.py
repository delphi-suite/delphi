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

# list of a token's instances in the dataset (takes 20s to run)
token_instances = [
    len(x) if isinstance(x, list) else 0 for x in token_map["prompt_pos_idx"]
]
# %%

model = "delphi-suite/delphi-llama2-100k"
tokenizer = AutoTokenizer.from_pretrained(model)

# %%


def tokenize(token: str, tokenizer: PreTrainedTokenizer = tokenizer) -> int:
    return int(tokenizer.encode(token, return_tensors="pt")[0][-1])


# Decode a sentence
def decode(token_ids: int, tokenizer: PreTrainedTokenizer = tokenizer) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# %%
def string_id(token: str) -> str:
    return token


def is_capitalized(token: str) -> bool:
    capitalized = token[0].isupper()
    space_and_capitalized = token[1].isupper() if len(token) > 1 else False
    return capitalized or space_and_capitalized
    # " Hello" -> True


def starts_with_space(token: str) -> bool:
    return token.startswith(" ") if token else False


def instances_in_dataset(token: str) -> int:
    return token_instances[tokenize(token)]


TOKEN_LABELS = {
    "String representation": string_id,
    "Instances in dataset": instances_in_dataset,
    "Starts with space": starts_with_space,
    "Capitalized": is_capitalized,
}


# %%
def label_single_token(token: str | None) -> dict[str, bool]:
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
    # if token is None, then it is a '' empty strong token or similar
    if token is None or token == "":
        for label_name, category_check in TOKEN_LABELS.items():
            labels[label_name] = False
        labels["String representation"] = string_id(token)
        labels["Instances in dataset"] = instances_in_dataset(token)
        labels["Is Other"] = True
        return labels
    # all other cases / normal tokens
    for label_name, category_check in TOKEN_LABELS.items():
        labels[label_name] = category_check(token)
    return labels


# %%
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

    # 1) Create a list of all tokens in the tokenizer's vocabulary
    tokens_str = [None] * vocab_size
    for i in range(vocab_size):
        tokens_str[i] = decode(i)

    # 2) let's label each token
    labelled_token_dict = [None] * vocab_size  # token_id: labels
    # we iterate over each
    for token_id, token in enumerate(tqdm(tokens_str, desc="Labelling tokens")):
        # label the batch of sentences
        labels = label_single_token(token)
        # create a dict with the token_ids and their labels
        # update the labelled_token_ids_dict with the new dict
        # first sentence of batch, label of first token
        labelled_token_dict[token_id] = labels

    return tokens_str, labelled_token_dict


# %%
# convert list of dictionaries into pandas dataframe
def convert_label_dict_to_df(
    labelled_token_ids_dict: list[dict[str, bool]]
) -> pd.DataFrame:
    df = pd.DataFrame(labelled_token_ids_dict)
    # Reset the index so it becomes a column, and then rename the column to 'token_id'
    df.reset_index(inplace=True)
    df.rename(columns={"index": "token_id"}, inplace=True)
    return df


# %%
def export_df_to_csv(df: pd.DataFrame, file_path: str, escapechar: str = "\\") -> None:
    """
    Exports a pandas DataFrame to a CSV file, with an option to set escape character.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be exported.
    file_path : str
        The path (including file name) where the CSV file will be saved.
    escapechar : str, optional
        The escape character to use in the CSV file for escaping, by default '\\'.
    """
    df.to_csv(file_path, index=False, escapechar=escapechar)
    # index=False to avoid exporting the index column


# %%


filename = "labelled_token_ids_dict.pkl"
filepath = STATIC_ASSETS_DIR.joinpath(filename)
with open(f"{filepath}", "rb") as f:
    label_dict = pickle.load(f)

df = pd.DataFrame.from_dict(label_dict, orient="index")

# Reset the index to get the ID column
df.reset_index(inplace=True)
df.rename(columns={"index": "ID"}, inplace=True)

# Save to CSV
df.to_csv("token_labels.csv", index=False)

# %%
