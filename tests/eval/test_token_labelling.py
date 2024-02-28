import pickle
from pathlib import Path

import pytest
from spacy.language import Language
from spacy.tokens import Doc
from transformers import AutoTokenizer

import delphi.eval.token_labelling as tl

labelled_token_ids_dict: dict[int, dict[str, bool]] = {}


@pytest.fixture
def dummy_doc() -> tuple[str, Doc, dict[str, bool]]:
    """
    Create a dummy Doc (list of Tokens) with specific attributes for testing purposes.
    """
    nlp_dummy = Language()

    # Assume we're creating a dummy token with specific attributes
    words = ["Peter", "is", "a", "person"]
    spaces = [True, True, True, True]  # No space after "dummy_token"
    pos_tags = ["PROPN", "AUX", "DET", "NOUN"]  # Part-of-speech tag
    dep_tags = ["nsubj", "ROOT", "det", "attr"]  # Dependency tag
    ner_tags = ["PERSON", "", "", ""]  # Named entity tag

    # Ensure the length of pos_tags and dep_tags matches the length of words
    assert len(words) == len(pos_tags) == len(dep_tags) == len(ner_tags)

    # Create a Doc with one dummy token
    doc = Doc(nlp_dummy.vocab, words=words, spaces=spaces)

    # Manually set POS, dependency and NER tags
    for token, pos, dep, ner_tag in zip(doc, pos_tags, dep_tags, ner_tags):
        token.pos_, token.dep_, token.ent_type_ = pos, dep, ner_tag

    # Token labels for "Peter" in the dummy doc
    PETER_TOKEN_LABEL = {
        "Starts with space": False,
        "Capitalized": True,
        "Is Adjective": False,
        "Is Adposition": False,
        "Is Adverb": False,
        "Is Auxiliary": False,
        "Is Coordinating conjuction": False,
        "Is Determiner": False,
        "Is Interjunction": False,
        "Is Noun": False,
        "Is Numeral": False,
        "Is Particle": False,
        "Is Pronoun": False,
        "Is Proper Noun": True,
        "Is Punctuation": False,
        "Is Subordinating conjuction": False,
        "Is Symbol": False,
        "Is Verb": False,
        "Is Other": False,
        "Is Named Entity": True,
    }
    text = " ".join(words)
    return text, doc, PETER_TOKEN_LABEL


def test_explain_token_labels(dummy_doc):
    """
    Test the explain_token_labels function.
    """
    # explain all labels
    tl.explain_token_labels()
    # print explanations for the first token in doc
    text, doc, PETER_TOKEN_LABEL = dummy_doc
    tl.explain_token_labels(doc[0])


def test_label_single_token(dummy_doc):
    """
    Test the label_single_token function.
    """
    # create a dummy token
    text, doc, PETER_TOKEN_LABEL = dummy_doc
    token = doc[0]
    # label the token
    labels = tl.label_single_token(token)
    # check if the labels are correct
    assert labels == PETER_TOKEN_LABEL


def test_label_sentence(dummy_doc):
    """
    Test the label_sentence function.
    """
    text, doc, PETER_TOKEN_LABEL = dummy_doc
    # label the sentence
    labels = tl.label_sentence(doc)
    # assert the first token is labeled correctly
    assert labels[0] == PETER_TOKEN_LABEL
    # iterate through tokens in doc
    for token, label in zip(doc, labels):
        assert label == tl.label_single_token(token)


def test_label_batch_sentences(dummy_doc):
    """
    Test the label_batch_sentences function.
    """
    # create a batch of sentences
    text, doc, PETER_TOKEN_LABEL = dummy_doc
    text = text.split(" ")
    batch = [text, text, text]
    # label the batch
    labels = tl.label_batch_sentences(batch, tokenized=True)
    # assert the first token is labeled correctly
    assert labels[0][0] == PETER_TOKEN_LABEL
    assert labels[1][0] == PETER_TOKEN_LABEL
    assert labels[2][0] == PETER_TOKEN_LABEL
    # iterate through tokens in doc
    for token, label in zip(doc, labels[0]):
        assert label == tl.label_single_token(token)


def is_valid_structure(obj: dict[int, dict[str, bool]]) -> bool:
    """
    Checks whether the obj fits the structure of `dict[int, dict[str, bool]]`. Returns True, if it fits, False otherwise.
    """
    if not isinstance(obj, dict):
        print(f"Main structure is not dict! Instead is type {type(obj)}")
        return False
    for key, value in obj.items():
        if not isinstance(key, int) or not isinstance(value, dict):
            print(
                f"Main structure is dict, but its keys are either not int or its values are not dicts. Instead key is type {type(key)} and value is type {type(value)}"
            )
            return False
        for sub_key, sub_value in value.items():
            if not isinstance(sub_key, str) or not isinstance(sub_value, bool):
                print(
                    f"The structure dict[int, dict[X, Y]] is True, but either X is not str or Y is not bool. Instead X is type {type(sub_key)} and Y is type {type(sub_value)}"
                )
                return False
    return True


@pytest.mark.dependency()
def test_label_tokens_from_tokenizer():
    """
    Simple test, checking if download of tokinzer and the labelling of all tokens in its vocabulary works.
    """
    global labelled_token_ids_dict
    # get a tokinzer
    model_name = "delphi-suite/delphi-llama2-100k"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size

    tokens_str, labelled_token_ids_dict = tl.label_tokens_from_tokenizer(tokenizer)
    # count the number of lines in the token_str
    assert tokens_str.count("\n") == (vocab_size + 1)  # + 1, because of token '\n'
    assert len(labelled_token_ids_dict.keys()) == vocab_size
    assert is_valid_structure(labelled_token_ids_dict) == True


@pytest.mark.dependency(depends=["test_label_tokens_from_tokenizer"])
@pytest.mark.parametrize(
    "path", [Path("temp/token_labels.csv"), Path("temp/token_labels.pkl")]
)
def test_import_token_labels(path: Path):
    global labelled_token_ids_dict
    assert (
        labelled_token_ids_dict is not None
    ), "It should be filled for the test to run. Check test-dependency."
    assert (
        labelled_token_ids_dict != {}
    ), "It should be filled for the test to run. Check test-dependency."
    # create the path
    path.parent.mkdir(parents=True, exist_ok=True)
    # save the file
    if path.suffix == ".pkl":
        with open(path, "wb") as file:
            pickle.dump(labelled_token_ids_dict, file)
    elif path.suffix == ".csv":
        df = tl.convert_label_dict_to_df(labelled_token_ids_dict)
        df.to_csv(path, index=False)
    else:
        raise ValueError("The file ending is incorrect.")

    # load the file with our function to be tested
    loaded_dict = tl.import_token_labels(path)

    # assure that the structure is correct
    assert loaded_dict == labelled_token_ids_dict
    assert is_valid_structure(loaded_dict) == True
