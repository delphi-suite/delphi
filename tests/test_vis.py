import pytest
import torch
from beartype.roar import BeartypeCallHintViolation
from IPython.display import HTML
from transformers import AutoModelForCausalLM, AutoTokenizer

from delphi.eval.compare_models import ModelComparison, compare_models
from delphi.eval.utils import load_text_from_dataset, load_validation_dataset, tokenize

torch.set_grad_enabled(False)


# define a pytest fixture for the model name
@pytest.fixture
def model_name():
    return "roneneldan/TinyStories-1M"


# define a pytest fixture for a default tokenizer using the model_name fixture
@pytest.fixture
def tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


# define a pytest fixture for a default model using the model_name fixture
@pytest.fixture
def model(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name)


# define a pytest fixture for the raw dataset
@pytest.fixture
def ds_txt():
    return load_text_from_dataset(load_validation_dataset("tinystories-v2-clean"))[:100]


# define a pytest fixture for the tokenized dataset
@pytest.fixture
def ds_tok(tokenizer, ds_txt):
    return [tokenize(tokenizer, txt) for txt in ds_txt]


# define a pytest fixture for a tokenized sample
@pytest.fixture
def sample_tok(ds_tok):
    return ds_tok[0]


def test_compare_models(model, sample_tok):
    model_instruct = AutoModelForCausalLM.from_pretrained(
        "roneneldan/TinyStories-Instruct-1M"
    )
    K = 3
    model_comparison = compare_models(model, model_instruct, sample_tok, top_k=K)
    assert isinstance(model_comparison, ModelComparison)

    assert model_comparison.correct_prob_base_model.shape == sample_tok.shape
    assert model_comparison.correct_prob_lift_model.shape == sample_tok.shape
    assert model_comparison.top_k_tokens_lift_model.shape == (sample_tok.shape[0], K)
    assert model_comparison.top_k_probs_base_model.shape == (sample_tok.shape[0], K)
    assert model_comparison.top_k_probs_lift_model.shape == (sample_tok.shape[0], K)
