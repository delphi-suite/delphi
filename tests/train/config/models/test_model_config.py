import pytest
from dacite import from_dict
from transformers import BloomConfig, BloomForCausalLM

from delphi.train.config.model_config import ModelConfig


@pytest.fixture
def llama_config():
    return from_dict(
        ModelConfig,
        {
            "model_type": "LlamaForCausalLM",
            "model_params": {"hidden_size": 49, "num_attention_heads": 7},
        },
    )


@pytest.fixture
def bloom_config():
    return from_dict(
        ModelConfig,
        {
            "model_type": "BloomForCausalLM",
            "model_params": {"layer_norm_epsilon": 0.0042},
        },
    )


def test_deserialziation(bloom_config):
    direct_bloom_config = ModelConfig(
        model_type="BloomForCausalLM",
        model_params=dict(layer_norm_epsilon=0.0042),
    )
    assert bloom_config == direct_bloom_config


def test_config_to_model(bloom_config):
    model = bloom_config.get_model()

    assert isinstance(model, BloomForCausalLM)
    assert isinstance(model.config, BloomConfig)
    assert model.config.layer_norm_epsilon == 0.0042
