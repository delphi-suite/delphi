from dataclasses import asdict

import dacite
import pytest
import torch
from datasets import Dataset
from jaxtyping import Float
from transformers import PreTrainedModel

from delphi.constants import TEST_CONFIGS_DIR
from delphi.eval.utils import get_all_and_next_logprobs
from delphi.train.config import TrainingConfig
from delphi.train.config.utils import build_config_from_files_and_overrides
from delphi.train.train_step import accumulate_gradients, train_step
from delphi.train.utils import (
    ModelTrainingState,
    gen_minibatches,
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
        torch.tensor([-0.01780166, -0.00762226, 0.03532362]),
    ).all()


def test_performance(dataset, model):
    """check that predictions improve with training"""
    # setup
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model_training_state = ModelTrainingState(
        model=model,
        optimizer=optimizer,
        iter_num=0,
        epoch=0,
        step=0,
        train_loss=0.0,
        lr=1e-3,
        last_training_step_time=0.0,
    )
    device = torch.device("cpu")
    indices = list(range(len(dataset)))

    next_logprobs_before = get_all_and_next_logprobs(model, dataset["tokens"])[1]

    train_step(
        model_training_state, dataset, load_test_config("debug"), device, indices
    )

    next_logprobs_after = get_all_and_next_logprobs(model, dataset["tokens"])[1]
    # should generally increse with training
    frac_increased = (next_logprobs_after > next_logprobs_before).float().mean().item()
    assert frac_increased > 0.95


def get_grads(model: PreTrainedModel) -> Float[torch.Tensor, "grads"]:
    grads = [
        param.grad.flatten() for param in model.parameters() if param.grad is not None
    ]
    return torch.cat(grads)


def test_accumulate_gradients_accumulates(dataset, model):
    """
    check that gradient accumulation works as expected and doesn't reset on each microstep
    """
    batch_size = 3
    num_batches = 3
    # first 2 mini-batches different, last mini-batch the same
    indices_set_a = [1, 2, 3] + [4, 5, 6] + [7, 8, 9]
    indices_set_b = [7, 8, 9] * 3

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_minibatches=num_batches,
        step=0,
        device=torch.device("cpu"),
        feature_name="tokens",
    )
    batches_a = gen_minibatches(indices=indices_set_a, **kwargs)  # type: ignore
    batches_b = gen_minibatches(indices=indices_set_b, **kwargs)  # type: ignore

    # accumulate
    _total_loss = accumulate_gradients(model, batches_a, num_batches)

    grads_a = get_grads(model)

    # reset grad on model
    model.zero_grad()

    _total_loss = accumulate_gradients(model, batches_b, num_batches)
    grads_b = get_grads(model)

    # test
    assert not torch.isclose(grads_a, grads_b).all()


def test_accumulate_gradients_consistent(dataset, model):
    """
    Validate that the gradients are consistent when the same batch is passed to accumulate_gradients
    """
    # setup
    num_batches = 3
    batch_size = 3
    kwargs = dict(
        indices=list(range(1, 10)),
        dataset=dataset,
        batch_size=batch_size,
        num_minibatches=num_batches,
        step=0,
        device=torch.device("cpu"),
        feature_name="tokens",
    )
    batches_a = gen_minibatches(**kwargs)  # type: ignore
    batches_aa = gen_minibatches(**kwargs)  # type: ignore

    # accumulate
    _total_loss = accumulate_gradients(model, batches_a, num_batches)

    grads_a = get_grads(model)

    # reset grad on model
    model.zero_grad()

    _total_loss = accumulate_gradients(model, batches_aa, num_batches)
    grads_aa = get_grads(model)

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
