import os
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
import torch
from dacite import from_dict

from delphi.train.config import GigaConfig
from delphi.train.config.models import TypedLlamaConfig
from delphi.train.utils import (
    EvalData,
    ModelTrainingState,
    config_to_model,
    initialize_model_training_state,
)
from delphi.train.wandb_utils import init_wandb, log_to_wandb, silence_wandb


@pytest.fixture
def mock_giga_config():
    config = from_dict(
        GigaConfig,
        {
            "run_name": "test_run",
            "device": "cpu",
            "model_config": {
                "model_type": "llama",
                "llama": asdict(TypedLlamaConfig()),
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
def mock_model_training_state(mock_giga_config):
    device = torch.device(mock_giga_config.device)
    # this is gross and horrible, sorry, I'm rushing
    mts = initialize_model_training_state(config=mock_giga_config, device=device)
    mts.step = 1
    mts.epoch = 1
    mts.iter_num = 1
    mts.lr = 0.001
    mts.running_mfu = 3.0
    return mts


@pytest.fixture
def mock_eval_data(mock_giga_config, mock_model_training_state):
    eval_data = EvalData(
        model_training_state=mock_model_training_state,
        tokens_per_iter=1000,
        losses={"train": 0.5, "val": 0.4},
        new_best_val_loss=False,
        config=mock_giga_config,
    )
    return eval_data


@patch.dict("os.environ", {}, clear=True)
def test_silence_wandb():
    silence_wandb()
    assert os.environ["WANDB_SILENT"] == "true"


@patch("wandb.init")
def test_init_wandb(mock_wandb_init: MagicMock, mock_giga_config):
    init_wandb(mock_giga_config)
    mock_wandb_init.assert_called_once_with(
        entity="test_entity",
        project="test_project",
        name="test_run",
        config=asdict(mock_giga_config),
    )


@patch("wandb.log")
def test_log_to_wandb(mock_wandb_log, mock_eval_data):
    log_to_wandb(mock_eval_data)
    mock_wandb_log.assert_called_once_with(
        {
            "iter": 1,
            "tokens": 1000,
            "loss/train": 0.5,
            "loss/val": 0.4,
            "lr": 0.001,
            "mfu": 300.0,
        },
        step=1,
    )
