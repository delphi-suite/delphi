# TODO: there are some ugly hacks here, and the test states are way too complicated
# clean this up as other parts of the codebase are refactored

from dataclasses import asdict

import pytest
import torch
from dacite import from_dict

from delphi.train.config import GigaConfig
from delphi.train.config.utils import load_preset
from delphi.train.train_step import accumulate_gradients, train_step
from delphi.train.utils import (
    ModelTrainingState,
    get_xy_batch,
    load_delphi_training_dataset,
)


def test_accumulate_gradients_accumulates():
    """
    check that gradient accumulation works as expected and doesn't reset on each microstep
    """
    # setup
    model = load_preset("debug").model_config.get_model()
    dataset = load_delphi_training_dataset("train", limit=64)
    indices_set_a = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    # different batch but idential last batch;
    # this should result in a different accumulated gradient
    indices_set_b = [
        [7, 8, 9],
    ]

    batches_a = [
        get_xy_batch(
            dataset=dataset,
            indices=indices_set_a[microstep],
            batch_size=1,
            step=0,
            microstep=microstep,
            gradient_accumulation_steps=3,
            device=torch.device("cpu"),
        )
        for microstep in range(len(indices_set_a))
    ]
    batches_b = [
        get_xy_batch(
            dataset=dataset,
            indices=indices_set_b[microstep],
            batch_size=1,
            step=0,
            microstep=microstep,
            gradient_accumulation_steps=3,
            device=torch.device("cpu"),
        )
        for microstep in range(len(indices_set_b))
    ]

    # accumulate
    total_loss = accumulate_gradients(model, batches_a, len(batches_a))

    grad_a = sum(
        [param.grad.norm() for param in model.parameters() if param.grad is not None]
    )

    # reset grad on model
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
            param.grad = None

    total_loss = accumulate_gradients(model, batches_b, len(batches_b))
    grad_b = sum(
        [param.grad.norm() for param in model.parameters() if param.grad is not None]
    )
    # test
    assert grad_a != grad_b


def test_accumulate_gradients_consistent():
    """
    Validate that the gradients are consistent when the same batch is passed to accumulate_gradients
    """
    # setup
    model = load_preset("debug").model_config.get_model()
    dataset = load_delphi_training_dataset("train", limit=64)
    indices_set = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    batches_a = [
        get_xy_batch(
            dataset=dataset,
            indices=indices_set[microstep],
            batch_size=1,
            step=0,
            microstep=microstep,
            gradient_accumulation_steps=2,
            device=torch.device("cpu"),
        )
        for microstep in range(3)
    ]
    # exact copy
    batches_aa = [
        get_xy_batch(
            dataset=dataset,
            indices=indices_set[microstep],
            batch_size=1,
            step=0,
            microstep=microstep,
            gradient_accumulation_steps=2,
            device=torch.device("cpu"),
        )
        for microstep in range(3)
    ]
    num_batches = len(batches_a)

    # accumulate
    total_loss = accumulate_gradients(model, batches_a, num_batches)

    grad_a = sum(
        [param.grad.norm() for param in model.parameters() if param.grad is not None]
    )

    # reset grad on model
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
            param.grad = None

    total_loss = accumulate_gradients(model, batches_aa, num_batches)
    grad_aa = sum(
        [param.grad.norm() for param in model.parameters() if param.grad is not None]
    )

    # test
    assert grad_a == grad_aa


def get_model_training_state(model, optimizer, step):
    return ModelTrainingState(
        model=model,
        optimizer=optimizer,
        iter_num=0,
        epoch=0,
        step=step,
        train_loss=0.0,
        lr=0.01,
        best_val_loss=float("inf"),
        last_training_step_time=0.0,
    )


def test_train_step_no_training():
    """
    Test train_step when no_training is set to True
    """
    # setup
    config_dict = asdict(load_preset("debug"))
    config_dict["debug_config"] = {"no_training": True}
    config = from_dict(GigaConfig, config_dict)
    model = config.model_config.get_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model_training_state = get_model_training_state(
        model=model, optimizer=optimizer, step=0
    )
    train_ds = load_delphi_training_dataset("train", limit=64)
    device = torch.device("cpu")
    indices = [0, 1, 2, 3]

    # (don't) train
    train_step(model_training_state, train_ds, config, device, indices)

    # test
    assert model_training_state.train_loss == 0.0


def test_train_step_with_training():
    """
    Test train_step when training is performed
    """
    # setup
    config_dict = asdict(load_preset("debug"))
    config_dict["debug_config"] = {"no_training": False}
    config_dict["batch_size"] = 16
    config_dict["optimizer"] = {"gradient_accumulation_steps": 4}
    config_dict["grad_clip"] = 1.0
    config = from_dict(GigaConfig, config_dict)
    model = config.model_config.get_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model_training_state = get_model_training_state(
        model=model, optimizer=optimizer, step=0
    )
    train_ds = load_delphi_training_dataset("train", limit=64)
    device = torch.device("cpu")
    indices = [0, 1, 2, 3]

    # train
    train_step(model_training_state, train_ds, config, device, indices)

    # test
    assert model_training_state.train_loss > 0.0
