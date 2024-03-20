import pytest
from dacite import from_dict
from transformers import BloomConfig, BloomForCausalLM, LlamaConfig, LlamaForCausalLM

from delphi.train.config.models import ModelConfig, TypedLlamaConfig
from delphi.train.config.models.model_config import ModelConfig, config_to_model


@pytest.fixture
def llama_config():
    return from_dict(
        ModelConfig,
        {
            "model_type": "llama2",
            "llama2": {"hidden_size": 49, "num_attention_heads": 7},
        },
    )


@pytest.fixture
def bloom_config():
    return from_dict(
        ModelConfig,
        {
            "model_type": "BloomForCausalLM",
            "transformers_config": {"layer_norm_epsilon": 0.0042},
        },
    )


def test_deserialziation(llama_config):
    direct_llama_config = ModelConfig(
        model_type="llama2",
        llama2=TypedLlamaConfig(hidden_size=49, num_attention_heads=7),
    )
    assert llama_config == direct_llama_config


def test_model_config_is_predefined_type(llama_config):
    assert llama_config.is_predefined_type()


def test_model_config_is_not_predefined_type(bloom_config):
    assert not bloom_config.is_predefined_type()


def test_config_to_model_predefined(llama_config):
    model = config_to_model(llama_config)

    assert isinstance(model, LlamaForCausalLM)
    assert isinstance(model.config, LlamaConfig)
    assert model.config.hidden_size == 49


def test_config_to_model_generic_type(bloom_config):
    model = config_to_model(bloom_config)

    assert isinstance(model, BloomForCausalLM)
    assert isinstance(model.config, BloomConfig)
    assert model.config.layer_norm_epsilon == 0.0042
