# %%
import pickle
from typing import Callable, Optional

import spacy
from datasets import load_dataset
from spacy.tokens import Doc, Token
from spacy.util import is_package
from transformers import AutoTokenizer

from delphi.constants import STATIC_ASSETS_DIR

SPACY_MODEL = "en_core_web_sm"  # small: "en_core_web_sm", large: "en_core_web_trf"
NLP = None  # global var to hold the language model

# Option A) Load the token map from hf
# token_map = load_dataset("delphi-suite/v0-token-map", split="validation")

# Option B) Use the downloaded token map (just has)
with open(STATIC_ASSETS_DIR.joinpath("token_map.pkl"), "rb") as f:
    token_map = pickle.load(f)

model = "delphi-suite/delphi-llama2-100k"
tokenizer = AutoTokenizer.from_pretrained(model)

# %%
TOKEN_LABELS: dict[str, Callable] = {
    # --- custom categories ---
    "Starts with space": (lambda token: token.text.startswith(" ")),  # bool
    "Capitalized": (
        lambda token: token.text[0].isupper()
        if len(token.text) == 1
        else (token.text[1].isupper() or token.text[0].isupper())
    ),  # bool (in case the first is a space)
    "Instances in dataset": (
        lambda token: len(
            token_map.get(tokenizer.encode(token.text, add_special_tokens=False)[0], [])
        )
    ),
}


# %%
def label_single_token(token: Token | None) -> dict[str, bool]:
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
    if token is None:
        for label_name, category_check in TOKEN_LABELS.items():
            labels[label_name] = False
        labels["Is Other"] = True
        return labels
    # all other cases / normal tokens
    for label_name, category_check in TOKEN_LABELS.items():
        labels[label_name] = category_check(token)
    return labels


def label_sentence(tokens: Doc | list[Token]) -> list[dict[str, bool]]:
    """
    Labels spaCy Tokens in a sentence. Takes the context of the token into account
    for dependency labels (e.g. subject, object, ...), IF dependency labels are turned on.

    Parameters
    ----------
    tokens : list[Token]
        A list of tokens.

    Returns
    -------
    list[dict[str, bool]]
        Returns a list of the tokens' labels.
    """
    labelled_tokens = list()  # list holding labels for all tokens of sentence
    # if the list is empty it is because token is '' empty string or similar
    if len(tokens) == 0:
        labels = label_single_token(None)
        labelled_tokens.append(labels)
        return labelled_tokens
    # in all other cases
    for token in tokens:
        labels = label_single_token(token)
        labelled_tokens.append(labels)
    return labelled_tokens


def label_batch_sentences(
    sentences: list[str] | list[list[str]],
    tokenized: bool = True,
    verbose: bool = False,
) -> list[list[dict[str, bool]]]:
    """
    Labels tokens in a sentence batchwise. Takes the context of the token into
    account for dependency labels (e.g. subject, object, ...).

    Parameters
    ----------
    sentences : list
        A batch/list of sentences, each being a list of tokens.
    tokenized : bool, optional
        Whether the sentences are already tokenized, by default True. If the sentences
        are full strings and not lists of tokens, then set to False. If true then `sentences` must be list[list[str]].
    verbose : bool, optional
        Whether to print the tokens and their labels to the console, by default False.

    Returns
    -------
    list[list[dict[str, bool]]
        Returns a list of sentences. Each sentence contains a list of its
        corresponding token length where each entry provides the labels/categories
        for the token. Sentence -> Token -> Labels
    """
    global NLP, SPACY_MODEL

    if NLP is None:
        # Load english language model
        NLP = spacy.load(SPACY_MODEL)
    # labelled tokens, list holding sentences holding tokens holding corresponding token labels
    labelled_sentences: list[list[dict[str, bool]]] = list()

    # go through each sentence in the batch
    for sentence in sentences:
        if tokenized:
            # sentence is a list of tokens
            doc = Doc(NLP.vocab, words=sentence)  # type: ignore
            # Apply the spaCy pipeline, except for the tokenizer
            for name, proc in NLP.pipeline:
                if name != "tokenizer":
                    doc = proc(doc)
        else:
            # sentence is a single string
            doc = NLP(sentence)  # type: ignore

        labelled_tokens = list()  # list holding labels for all tokens of sentence
        labelled_tokens = label_sentence(doc)

        # print the token and its labels to console
        if verbose is True:
            # go through each token in the sentence
            for token, labelled_token in zip(doc, labelled_tokens):
                print(f"Token: {token}")
                print(" | ".join(list(TOKEN_LABELS.keys())))
                printable = [
                    str(l).ljust(len(name)) for name, l in labelled_token.items()
                ]
                printable = " | ".join(printable)
                print(printable)
                print("---")
        # add current sentence's tokens' labels to the list
        labelled_sentences.append(labelled_tokens)

        if verbose is True:
            print("\n")

    return labelled_sentences
