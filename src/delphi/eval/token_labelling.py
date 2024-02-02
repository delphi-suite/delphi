"""
This script creates labels for tokens in a sentence. 
It takes the context of the token into account. 
Additionally, it can visualize the sentences and their poart-of-speech (POS) tags.
"""

from pprint import pprint
from typing import List, Optional

import spacy  # pylint: disable=import-error
from spacy.tokens import Doc  # pylint: disable=import-error
from spacy.tokens import Token

# make sure the english language model capabilities are installed by the equivalent of:
# python -m spacy download en_core_web_sm
# Should be run once, initially. Download only starts if not already installed.
spacy.cli.download("en_core_web_sm", False, False, "-q")


CATEGORIES = {
    # custom categories
    "Starts with space": (lambda token: token.text.startswith(" ")),  # bool
    "Capitalized": (lambda token: token.text[0].isupper()),  # bool
    # POS (part-of-speech) categories
    # "POS Tag": (lambda token: token.pos_),  # 'NOUN', 'VB', ..
    "Is Noun": (lambda token: token.pos_ == "NOUN"),  # redundant
    "Is Pronoun": (lambda token: token.pos_ == "PRON"),  # redundant
    "Is Adjective": (lambda token: token.pos_ == "ADJ"),  # redundant
    "Is Verb": (lambda token: "VB" in token.tag_),  # redundant
    "Is Adverb": (lambda token: token.pos_ == "ADV"),  # redundant
    "Is Preposition": (lambda token: token.pos_ == "ADP"),  # redundant
    "Is Conjunction": (lambda token: token.pos_ == "CONJ"),  # redundant
    "Is Interjunction": (lambda token: token.pos_ == "INTJ"),  # redundant
    # dependency categories
    # "Dependency": (lambda token: token.dep_),  # 'nsubj', 'ROOT', 'dobj', ..
    "Is Subject": (lambda token: token.dep_ == "nsubj"),
    "Is Object": (lambda token: token.dep_ == "dobj"),
    "Is Root": (
        lambda token: token.dep_ == "ROOT"
    ),  # root of the sentence (often a verb)
    "Is auxiliary": (lambda token: token.dep_ == "aux"),  # redundant
    # Named entity recognition (NER) categories
    # "Named Entity Type": (lambda token: token.ent_type_),  # '', 'PERSON', 'ORG', 'GPE', ..
    "Is Named Entity": (lambda token: token.ent_type_ != ""),
}


def explain_token_labels(token: Optional[Token] = None) -> None:
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
        labels = label_single_token(token)
        print(" Explanation of token labels ".center(45, "-"))
        print("Token text:".ljust(20), token.text)
        print("Token dependency:".ljust(20), spacy.glossary.explain(token.dep_))
        print("Token POS:".ljust(20), spacy.glossary.explain(token.pos_))
        print(" Token labels ".center(45, "-"))
        for i, (label, value) in enumerate(zip(CATEGORIES.keys(), labels)):
            print(f" {i:2}  ", label.ljust(20), value)

    else:
        glossary = spacy.glossary.GLOSSARY
        print(
            f"Explanation of all {len(glossary.keys())} token labels (POS, dependency, NER, ...):"
        )
        for label, key in glossary.items():
            print("   ", label.ljust(10), key)


def label_single_token(token: Token) -> List:
    """
    Labels a single token.

    Parameters
    ----------
    token : Token
        The token to be labelled.

    Returns
    -------
    List
        The labels of the token.
    """
    assert isinstance(token, Token)
    labels = list()  #  The list holding labels of a single token
    for _, category_check in CATEGORIES.items():
        label = category_check(token)
        labels.append(label)
    return labels


def label_batch_token(
    sentences: List, tokenized: bool = True, verbose: bool = False
) -> List[List]:
    """
    Labels tokens in a sentence batchwise. Takes the context of the token into
    account for dependency labels (e.g. subject, object, ...).

    Parameters
    ----------
    sentences : List
        A batch/list of sentences, each being a list of tokens.
    tokenized : bool, optional
        Whether the sentences are already tokenized, by default True. If the sentences
        are full strings and not lists of tokens, then set to False.
    verbose : bool, optional
        Whether to print the tokens and their labels to the console, by default False.

    Returns
    -------
    List[List]
        Returns a list of sentences. Each sentence contains a list of its
        corresponding token length where each entry provides the labels/categories
        for the token. Sentence -> Token -> Labels
    """
    assert isinstance(sentences, list)
    # Load english language model
    nlp = spacy.load("en_core_web_sm")
    # labelled tokens, List holding sentences holding tokens holding corresponding token labels
    labelled_sentences = list()

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

        labelled_tokens = list()  # List holding labels for all tokens of sentence

        for token in doc:
            labels = list()  #  The list holding labels of a single token
            for _, category_check in CATEGORIES.items():
                label = category_check(token)
                labels.append(label)
            # add current token's to the list
            labelled_tokens.append(labels)

            # print the token and its labels to console
            if verbose is True:
                print(f"Token: {token.text}")
                print(" | ".join(list(CATEGORIES.keys())))
                printable = [
                    str(l).ljust(len(cname))
                    for l, cname in zip(labels, CATEGORIES.keys())
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
