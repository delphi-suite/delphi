from typing import Callable, Optional

import spacy
from spacy.tokens import Doc, Token
from spacy.util import is_package
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# make sure the english language model capabilities are installed by the equivalent of:
# python -m spacy download en_core_web_sm
# Should be run once, initially. Download only starts if not already installed.
SPACY_MODEL = "en_core_web_sm"  # small: "en_core_web_sm", large: "en_core_web_trf"
NLP = None  # global var to hold the language model
if not is_package(SPACY_MODEL):
    spacy.cli.download(SPACY_MODEL, False, False)


TOKEN_LABELS: dict[str, Callable] = {
    # --- custom categories ---
    "Starts with space": (lambda token: token.text.startswith(" ")),  # bool
    "Capitalized": (lambda token: token.text[0].isupper()),  # bool
    # --- POS (part-of-speech) categories ---
    # They include the Universal POS tags (https://universaldependencies.org/u/pos/)
    # -> "POS Tag": (lambda token: token.pos_),  # 'NOUN', 'VB', ..
    "Is Adjective": (lambda token: token.pos_ == "ADJ"),
    "Is Adposition": (lambda token: token.pos_ == "ADP"),
    "Is Adverb": (lambda token: token.pos_ == "ADV"),
    "Is Auxiliary": (lambda token: token.pos_ == "AUX"),
    "Is Coordinating conjuction": (lambda token: token.pos_ == "CCONJ"),
    "Is Determiner": (lambda token: token.pos_ == "DET"),
    "Is Interjunction": (lambda token: token.pos_ == "INTJ"),
    "Is Noun": (lambda token: token.pos_ == "NOUN"),
    "Is Numeral": (lambda token: token.pos_ == "NUM"),
    "Is Particle": (lambda token: token.pos_ == "PART"),
    "Is Pronoun": (lambda token: token.pos_ == "PRON"),
    "Is Proper Noun": (lambda token: token.pos_ == "PROPN"),
    "Is Punctuation": (lambda token: token.pos_ == "PUNCT"),
    "Is Subordinating conjuction": (lambda token: token.pos_ == "SCONJ"),
    "Is Symbol": (lambda token: token.pos_ == "SYM"),
    "Is Verb": (lambda token: token.pos_ == "VERB"),
    "Is Other": (lambda token: token.pos_ == "X"),
    #  --- dependency categories ---
    # -> "Dependency": (lambda token: token.dep_),  # 'nsubj', 'ROOT', 'dobj', ..
    # "Is Subject": (lambda token: token.dep_ == "nsubj"),
    # "Is Object": (lambda token: token.dep_ == "dobj"),
    # "Is Root": (
    #     lambda token: token.dep_ == "ROOT"
    # ),  # root of the sentence (often a verb)
    # "Is auxiliary": (lambda token: token.dep_ == "aux"),
    # --- Named entity recognition (NER) categories ---
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
        for i, (label_name, value) in enumerate(labels.items()):
            print(f" {i:2}  ", label_name.ljust(20), value)

    else:
        glossary = spacy.glossary.GLOSSARY
        print(
            f"Explanation of all {len(glossary.keys())} token labels (POS, dependency, NER, ...):"
        )
        for label, key in glossary.items():
            print("   ", label.ljust(10), key)


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


def label_tokens_from_tokenizer(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> tuple[str, dict[int, dict[str, bool]]]:
    """
    Labels all tokens in a tokenizer's vocabulary with the corresponding token categories (POS, named entity, etc). Returns two things: 1) `tokens_str`, a string where each token comprises 'token_id,token_str\n' and 2) `labelled_token_ids_dict` a dict that contains for each token_id (key) the corresponding token labels, which is in turn a dict, whith the label categories as keys and their boolean values as the dict's values.

    Parameters
    ----------
    tokenizer : The tokenizer with its tokens to be labelled.

    Returns
    -------
    tokens_str, labelled_token_ids_dict

    """

    def decode(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        token_ids: int | list[int],
    ) -> str:
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    vocab_size = tokenizer.vocab_size

    # 1) Create a list of all tokens in the tokenizer's vocabulary
    tokens_str = ""  # will hold all tokens and their ids
    for i in range(vocab_size):
        tokens_str += f"{i},{decode(tokenizer, i)}\n"

    # 2) let's label each token
    labelled_token_ids_dict: dict[int, dict[str, bool]] = {}  # token_id: labels
    max_token_id = vocab_size  # stop at which token id, vocab size
    # we iterate over all token_ids individually
    for token_id in tqdm(range(0, max_token_id), desc="Labelling tokens"):
        # decode the token_ids to get a list of tokens, a 'sentence'
        token = decode(tokenizer, token_id)  # list of tokens == sentence
        # put the sentence into a list, to make it a batch of sentences
        sentences = [token]
        # label the batch of sentences
        labels = label_batch_sentences(sentences, tokenized=True, verbose=False)
        # create a dict with the token_ids and their labels
        # update the labelled_token_ids_dict with the new dict
        label = labels[0][0]  # first sentence of batch, label of first token
        labelled_token_ids_dict[token_id] = label

    return tokens_str, labelled_token_ids_dict
