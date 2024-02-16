# this is a quick hack to label tokens based on @joshuawe's work in https://github.com/delphi-suite/delphi/pull/21

import logging
from dataclasses import dataclass
from typing import Callable

from tqdm import tqdm

import spacy
from spacy.tokens import Doc, Token
from spacy.util import is_package

from transformers import AutoTokenizer

from delphi.eval.constants import LLAMA2_HF_MODELS
from delphi.eval.utils import GenericPreTrainedTokenizer

_NLP = None  # global to hold language model
_SPACY_MODEL = "en_core_web_trf"


@dataclass
class HackTokenLabel:
    description: str
    includes: Callable  # usage: hack_token_label.includes(token) -> bool


class HackTokenLabels:
    STARTS_WITH_SPACE = HackTokenLabel(
        "Starts with space", lambda token: token.text.startswith(" ")
    )
    CAPITALIZED = HackTokenLabel("Capitalized", lambda token: token.text[0].isupper())
    # --- POS (part-of-speech) categories ---
    # They include the Universal POS tags (https://universaldependencies.org/u/pos/)
    # -> "POS Tag": (lambda token: token.pos_),  # 'NOUN', 'VB', ..
    IS_ADJECTIVE = HackTokenLabel("Is Adjective", lambda token: token.pos_ == "ADJ")
    IS_ADPOSITION = HackTokenLabel("Is Adposition", lambda token: token.pos_ == "ADP")
    IS_ADVERB = HackTokenLabel("Is Adverb", lambda token: token.pos_ == "ADV")
    IS_AUXILIARY = HackTokenLabel("Is Auxiliary", lambda token: token.pos_ == "AUX")
    IS_COORDINATING_CONJUNCTION = HackTokenLabel(
        "Is Coordinating conjuction", lambda token: token.pos_ == "CCONJ"
    )
    IS_DETERMINER = HackTokenLabel("Is Determiner", lambda token: token.pos_ == "DET")
    IS_INTERJUNCTION = HackTokenLabel(
        "Is Interjunction", lambda token: token.pos_ == "INTJ"
    )
    IS_NOUN = HackTokenLabel("Is Noun", lambda token: token.pos_ == "NOUN")
    IS_NUMERAL = HackTokenLabel("Is Numeral", lambda token: token.pos_ == "NUM")
    IS_PARTICLE = HackTokenLabel("Is Particle", lambda token: token.pos_ == "PART")
    IS_PRONOUN = HackTokenLabel("Is Pronoun", lambda token: token.pos_ == "PRON")
    IS_PROPER_NOUN = HackTokenLabel(
        "Is Proper Noun", lambda token: token.pos_ == "PROPN"
    )
    IS_PUNCTUATION = HackTokenLabel(
        "Is Punctuation", lambda token: token.pos_ == "PUNCT"
    )
    IS_SUBORDINATING_CONJUNCTION = HackTokenLabel(
        "Is Subordinating conjuction", lambda token: token.pos_ == "SCONJ"
    )
    IS_SYMBOL = HackTokenLabel("Is Symbol", lambda token: token.pos_ == "SYM")
    IS_VERB = HackTokenLabel("Is Verb", lambda token: token.pos_ == "VERB")
    IS_OTHER = HackTokenLabel("Is Other", lambda token: token.pos_ == "X")
    IS_NAMED_ENTITY = HackTokenLabel(
        "Is Named Entity", lambda token: token.ent_type_ != ""
    )

    ALL_LABELS = [
        STARTS_WITH_SPACE,
        CAPITALIZED,
        IS_ADJECTIVE,
        IS_ADPOSITION,
        IS_ADVERB,
        IS_AUXILIARY,
        IS_COORDINATING_CONJUNCTION,
        IS_DETERMINER,
        IS_INTERJUNCTION,
        IS_NOUN,
        IS_NUMERAL,
        IS_PARTICLE,
        IS_PRONOUN,
        IS_PROPER_NOUN,
        IS_PUNCTUATION,
        IS_SUBORDINATING_CONJUNCTION,
        IS_SYMBOL,
        IS_VERB,
        IS_OTHER,
    ]


def convert_to_spacy_token(token_str: str) -> Token:
    global _NLP
    if _NLP is None:
        if not is_package(_SPACY_MODEL):
            logging.info(f"Downloading spaCy model '{_SPACY_MODEL}'")
            spacy.cli.download(_SPACY_MODEL, False, False)
        _NLP = spacy.load(_SPACY_MODEL)

    doc = Doc(_NLP.vocab, words=[token_str])
    # Apply the spaCy pipeline, except for the tokenizer
    for name, proc in _NLP.pipeline:
        if name != "tokenizer":
            doc = proc(doc)
    assert len(doc) == 1, "Expected only one token"
    return doc[0]


def label_string_token(token: str) -> list[HackTokenLabel]:
    spacy_token = convert_to_spacy_token(token)
    return [
        label for label in HackTokenLabels.ALL_LABELS if label.includes(spacy_token)
    ]


def label_vocalulary(tokenizer: GenericPreTrainedTokenizer) -> dict[str, list[int]]:
    vocab_size = tokenizer.vocab_size
    token_groups = {label.description: list() for label in HackTokenLabels.ALL_LABELS}
    token = ""
    for i in tqdm(range(vocab_size), desc=token):
        token = tokenizer.decode(i)
        if token == "":
            continue
        labels = label_string_token(token)
        for label in labels:
            token_groups[label.description].append(i)
    return token_groups


# get pretrained tokenizer
# tokenizer = AutoTokenizer.from_pretrained(LLAMA2_HF_MODELS[0])
# labeled_vocab = label_vocalulary(tokenizer)
