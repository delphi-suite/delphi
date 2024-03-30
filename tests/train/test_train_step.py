# TODO: there are some ugly hacks here, and the test states are way too complicated
# clean this up as other parts of the codebase are refactored

from dataclasses import asdict

import dacite
import pytest
import torch
from datasets import Dataset

from delphi.train.config import TrainingConfig
from delphi.train.config.utils import load_preset
from delphi.train.train_step import accumulate_gradients, train_step
from delphi.train.utils import ModelTrainingState, get_xy_batch


@pytest.fixture
def dataset():
    ds = Dataset.from_dict(
        {
            "tokens": [list(range(i, i + 512)) for i in range(64)],
        },
    )
    ds.set_format(type="torch")
    return ds


@pytest.fixture
def model():
    # TODO: replace this with a model config dict after model_config update is in (next PR)
    return load_preset("debug").model_config.get_model()


def test_basic_reproducibility(dataset, model):
    """
    check that the same batch produces the same gradient
    """
    # setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model_training_state = ModelTrainingState(
        model=model,
        optimizer=optimizer,
        iter_num=0,
        epoch=0,
        step=0,
        train_loss=0.0,
        lr=0.01,
        best_val_loss=float("inf"),
        last_training_step_time=0.0,
    )
    device = torch.device("cpu")
    indices = list(range(64))

    # train
    train_step(model_training_state, dataset, load_preset("debug"), device, indices)

    params = torch.cat([p.flatten() for p in list(model.parameters())])

    assert torch.isclose(
        params[1000],
        torch.tensor([-0.0057]),
    )


def test_accumulate_gradients_accumulates(dataset, model):
    """
    check that gradient accumulation works as expected and doesn't reset on each microstep
    """
    # setup
    indices_set_a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # different batch but idential last batch (with batches of 3);
    # this should result in a different accumulated gradient
    indices_set_b = [7, 8, 9, 7, 8, 9, 7, 8, 9]
    batch_size = 3
    num_batches = len(indices_set_a) // batch_size

    batches_a = [
        get_xy_batch(
            dataset=dataset,
            indices=indices_set_a,
            batch_size=3,
            batch_num=microstep,
            feature_name="tokens",
            device=torch.device("cpu"),
        )
        for microstep in range(num_batches)
    ]
    batches_b = [
        get_xy_batch(
            dataset=dataset,
            indices=indices_set_b,
            batch_size=3,
            batch_num=microstep,
            feature_name="tokens",
            device=torch.device("cpu"),
        )
        for microstep in range(num_batches)
    ]

    # accumulate
    _total_loss = accumulate_gradients(model, batches_a, len(batches_a))

    grads_a = torch.cat(
        [
            param.grad.clone().detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
    )

    # reset grad on model
    model.zero_grad()

    _total_loss = accumulate_gradients(model, batches_b, len(batches_b))
    grads_b = torch.cat(
        [
            param.grad.clone().detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
    )

    # test
    assert not torch.isclose(grads_a, grads_b).all()


def test_accumulate_gradients_consistent(dataset, model):
    """
    Validate that the gradients are consistent when the same batch is passed to accumulate_gradients
    """
    # setup
    indices_set = list(range(1, 10))
    num_batches = 3
    batch_size = 3
    batches_a = [
        get_xy_batch(
            dataset=dataset,
            indices=indices_set,
            batch_size=batch_size,
            batch_num=microstep,
            feature_name="tokens",
            device=torch.device("cpu"),
        )
        for microstep in range(num_batches)
    ]
    batches_aa = [
        get_xy_batch(
            dataset=dataset,
            indices=indices_set,
            batch_size=batch_size,
            batch_num=microstep,
            feature_name="tokens",
            device=torch.device("cpu"),
        )
        for microstep in range(num_batches)
    ]

    # accumulate
    total_loss = accumulate_gradients(model, batches_a, num_batches)

    grads_a = torch.cat(
        [
            param.grad.clone().detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
    )

    # reset grad on model
    model.zero_grad()

    total_loss = accumulate_gradients(model, batches_aa, num_batches)
    grads_aa = torch.cat(
        [
            param.grad.clone().detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
    )

    # test
    assert torch.isclose(grads_a, grads_aa).all()


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


def test_train_step_no_training(dataset, model):
    """
    Test train_step when no_training is set to True
    """
    # setup
    config_dict = asdict(load_preset("debug"))
    config_dict["debug_config"] = {"no_training": True}
    config = dacite.from_dict(TrainingConfig, config_dict)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model_training_state = get_model_training_state(
        model=model, optimizer=optimizer, step=0
    )
    device = torch.device("cpu")
    indices = [0, 1, 2, 3]

    # (don't) train
    train_step(model_training_state, dataset, config, device, indices)

    # test
    assert model_training_state.train_loss == 0.0


def test_train_step_with_training(dataset, model):
    """
    Test train_step when training is performed
    """
    # setup
    config_dict = asdict(load_preset("debug"))
    config_dict["debug_config"] = {"no_training": False}
    config_dict["batch_size"] = 16
    config_dict["optimizer"] = {"gradient_accumulation_steps": 4}
    config_dict["grad_clip"] = 1.0
    config = dacite.from_dict(TrainingConfig, config_dict)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model_training_state = get_model_training_state(
        model=model, optimizer=optimizer, step=0
    )
    device = torch.device("cpu")
    indices = list(range(len(dataset)))

    # train
    train_step(model_training_state, dataset, config, device, indices)

    # test
    assert model_training_state.train_loss > 0.0
