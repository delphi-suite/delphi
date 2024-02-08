"""
This script creates labels for tokens in a sentence. 
It takes the context of the token into account. 
Additionally, it can visualize the sentences and their poart-of-speech (POS) tags.
"""

from typing import Callable, Optional, Union

import spacy  # pylint: disable=import-error
from spacy.tokens import Doc  # pylint: disable=import-error
from spacy.tokens import Token

# make sure the english language model capabilities are installed by the equivalent of:
# python -m spacy download en_core_web_sm
# Should be run once, initially. Download only starts if not already installed.
spacy.cli.download("en_core_web_sm", False, False, "-q")


TOKEN_LABELS: dict[str, Callable] = {
    # --- custom categories ---
    "Starts with space": (lambda token: token.text.startswith(" ")),  # bool
    "Capitalized": (lambda token: token.text[0].isupper()),  # bool
    # --- POS (part-of-speech) categories ---
    # -> "POS Tag": (lambda token: token.pos_),  # 'NOUN', 'VB', ..
    "Is Noun": (lambda token: token.pos_ == "NOUN"),  # redundant
    "Is Pronoun": (lambda token: token.pos_ == "PRON"),  # redundant
    "Is Adjective": (lambda token: token.pos_ == "ADJ"),  # redundant
    "Is Verb": (lambda token: "VB" in token.tag_),  # redundant
    "Is Adverb": (lambda token: token.pos_ == "ADV"),  # redundant
    "Is Preposition": (lambda token: token.pos_ == "ADP"),  # redundant
    "Is Conjunction": (lambda token: token.pos_ == "CONJ"),  # redundant
    "Is Interjunction": (lambda token: token.pos_ == "INTJ"),  # redundant
    #  --- dependency categories ---
    # -> "Dependency": (lambda token: token.dep_),  # 'nsubj', 'ROOT', 'dobj', ..
    # "Is Subject": (lambda token: token.dep_ == "nsubj"),
    # "Is Object": (lambda token: token.dep_ == "dobj"),
    # "Is Root": (
    #     lambda token: token.dep_ == "ROOT"
    # ),  # root of the sentence (often a verb)
    # "Is auxiliary": (lambda token: token.dep_ == "aux"),  # redundant
    # --- Named entity recognition (NER) categories ---
    # "Named Entity Type": (lambda token: token.ent_type_),  # '', 'PERSON', 'ORG', 'GPE', ..
    "Is Named Entity": (lambda token: token.ent_type_ != ""),
}


def explain_Token_labels(token: Optional[Token] = None) -> None:
    """
    Prints the explanation of a specific token's labels or of ALL
    possible labels (POS, dependency, NER, ...), if no token is provided.

    Parameters
    ----------
    token : Optional[Token], optional
        The token, whose labels should be explained. If None, all labels
        possible labels are explained, by default None.
    """
    if token is not None:
        # get token labels
        labels = label_single_Token(token)
        print(" Explanation of token labels ".center(45, "-"))
        print("Token text:".ljust(20), token.text)
        print("Token dependency:".ljust(20), spacy.glossary.explain(token.dep_))
        print("Token POS:".ljust(20), spacy.glossary.explain(token.pos_))
        print(" Token labels ".center(45, "-"))
        for i, (label_name, value) in enumerate(labels.items()):
            print(f" {i:2}  ", label_name.ljust(20), value)

    else:
        glossary = spacy.glossary.GLOSSARY
        print(
            f"Explanation of all {len(glossary.keys())} token labels (POS, dependency, NER, ...):"
        )
        for label, key in glossary.items():
            print("   ", label.ljust(10), key)


def label_single_Token(token: Token) -> dict[str, bool]:
    """
    Labels a single token. A token, that has been analyzed by the spaCy
    library.

    Parameters
    ----------
    token : Token
        The token to be labelled.

    Returns
    -------
    dict[str, bool]
        Returns a dictionary with the token's labels as keys and their
        corresponding boolean values.
    """
    assert isinstance(token, Token)
    labels = dict()  #  The list holding labels of a single token
    for label_name, category_check in TOKEN_LABELS.items():
        labels[label_name] = category_check(token)
    return labels


def label_sentence(tokens: Union[Doc, list[Token]]) -> list[dict[str, bool]]:
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
    for token in tokens:
        labels = label_single_Token(token)
        labelled_tokens.append(labels)
    return labelled_tokens


def label_batch_sentences(
    sentences: Union[list[str], list[list[str]]],
    tokenized: bool = True,
    verbose: bool = False,
) -> list[list]:
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
    assert isinstance(sentences, list)
    # Load english language model
    nlp = spacy.load("en_core_web_sm")
    # labelled tokens, list holding sentences holding tokens holding corresponding token labels
    labelled_sentences: list[list[dict[str, bool]]] = list()

    # go through each sentence in the batch
    for sentence in sentences:
        if tokenized:
            # sentence is a list of tokens
            doc = Doc(nlp.vocab, words=sentence)
            # Apply the spaCy pipeline, except for the tokenizer
            for name, proc in nlp.pipeline:
                if name != "tokenizer":
                    doc = proc(doc)
        else:
            # sentence is a single string
            doc = nlp(sentence)

        labelled_tokens = list()  # list holding labels for all tokens of sentence
        labelled_tokens = label_sentence(doc)

        # go through each token in the sentence
        for token, labelled_token in zip(doc, labelled_tokens):
            # labelled_token = label_single_Token(token)
            # labels = list()  #  The list holding labels of a single token
            # for _, category_check in TOKEN_LABELS.items():
            #     label = category_check(token)
            #     labels.append(label)
            # add current token's to the list
            # labelled_tokens.append(labelled_token)

            # print the token and its labels to console
            if verbose is True:
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


if __name__ == "__main__":
    # result = label_tokens(
    #     ["Hi, my name is Joshua.".split(" "), "The highway is full of car s, Peter.".split(" ")],
    #     tokenized=True,
    #     verbose=True,
    # )
    result = label_batch_token(
        ["Hi, my name is Joshua.", "The highway is full of car s, Peter."],
        tokenized=False,
        verbose=True,
    )
    print(result)
