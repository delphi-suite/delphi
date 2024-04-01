import os
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
import torch
import transformers
from dacite import from_dict

from delphi.train.config import TrainingConfig
from delphi.train.utils import ModelTrainingState, initialize_model_training_state
from delphi.train.wandb_utils import init_wandb, log_to_wandb, silence_wandb


@pytest.fixture
def mock_training_config():
    config = from_dict(
        TrainingConfig,
        {
            "run_name": "test_run",
            "device": "cpu",
            "model_config": {
                "model_type": "LlamaForCausalLM",
                "model_params": {
                    "hidden_size": 48,
                    "intermediate_size": 48,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2,
                    "vocab_size": 4096,
                },
            },
            "wandb_config": {
                "log": True,
                "entity": "test_entity",
                "project": "test_project",
            },
        },
    )
    return config


@pytest.fixture
def mock_model_training_state(mock_training_config):
    device = torch.device(mock_training_config.device)
    # this is gross and horrible, sorry, I'm rushing
    mts = initialize_model_training_state(config=mock_training_config, device=device)
    mts.step = 1
    mts.epoch = 1
    mts.iter_num = 1
    mts.lr = 0.001
    return mts


@patch.dict("os.environ", {}, clear=True)
def test_silence_wandb():
    silence_wandb()
    assert os.environ["WANDB_SILENT"] == "true"


@patch("wandb.init")
def test_init_wandb(mock_wandb_init: MagicMock, mock_training_config):
    init_wandb(mock_training_config)
    mock_wandb_init.assert_called_once_with(
        entity="test_entity",
        project="test_project",
        name="test_run",
        config=asdict(mock_training_config),
    )


@patch("wandb.log")
def test_log_to_wandb(mock_wandb_log: MagicMock):
    model = MagicMock(spec=transformers.LlamaForCausalLM)
    optimizer = MagicMock(spec=torch.optim.AdamW)
    log_to_wandb(
        mts=ModelTrainingState(
            model=model,
            optimizer=optimizer,
            step=5,
            epoch=1,
            iter_num=55,
            lr=0.007,
            last_training_step_time=0.0,
        ),
        losses={"train": 0.5, "val": 0.4},
        tokens_so_far=4242,
    )
    assert mock_wandb_log.call_count == 1
    mock_wandb_log.assert_called_with(
        {
            "epoch": 1,
            "epoch_iter": 5,
            "global_iter": 55,
            "tokens": 4242,
            "loss/train": 0.5,
            "loss/val": 0.4,
            "lr": 0.007,
        },
        step=55,
    )
