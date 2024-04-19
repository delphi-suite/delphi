from dataclasses import asdict

import dacite
import pytest
import torch
from datasets import Dataset
from jaxtyping import Float

from delphi.constants import TEST_CONFIGS_DIR
from delphi.train.config import TrainingConfig
from delphi.train.config.utils import build_config_from_files_and_overrides
from delphi.train.train_step import accumulate_gradients, train_step
from delphi.train.utils import (
    ModelTrainingState,
    get_xy_batch,
    init_model,
    setup_determinism,
)


def load_test_config(preset_name: str) -> TrainingConfig:
    """Load a test config by name, e.g. `load_preset("debug")`."""
    preset_path = TEST_CONFIGS_DIR / f"{preset_name}.json"
    return build_config_from_files_and_overrides([preset_path], {})


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
    setup_determinism(42)
    return init_model(
        {
            "model_class": "LlamaForCausalLM",
            "hidden_size": 48,
            "intermediate_size": 48,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "vocab_size": 4096,
        },
        seed=42,
    )


def get_params(model: torch.nn.Module) -> Float[torch.Tensor, "params"]:
    params = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
    params.sort(key=lambda x: x[0])
    return torch.cat([p.flatten() for _, p in params])


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
        last_training_step_time=0.0,
    )
    device = torch.device("cpu")
    indices = list(range(len(dataset)))
    train_step(
        model_training_state, dataset, load_test_config("debug"), device, indices
    )

    params = get_params(model)

    assert torch.isclose(
        params[[1000, 2000, 3000]],
        torch.tensor([-0.01782517, -0.00771354, 0.03517739]),
    ).all()


def test_accumulate_gradients_accumulates(dataset, model):
    """
    check that gradient accumulation works as expected and doesn't reset on each microstep
    """
    # setup
    indices_set_a = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    # different batch but idential last batch;
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
    indices_set = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
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
        last_training_step_time=0.0,
    )


def test_train_step_no_training(dataset, model):
    """
    Test train_step when no_training is set to True
    """
    # setup
    config_dict = asdict(load_test_config("debug"))
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
    config_dict = asdict(load_test_config("debug"))
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
