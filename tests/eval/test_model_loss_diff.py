import pytest
import torch

from delphi.eval.model_loss_diff import next_logprobs_dict_to_loss_dict


def test_next_logprobs_dict_to_loss_dict():
    """
    Test the next_logprobs_dict_to_loss_dict function.
    """
    next_logprobs_dict = {
        "model1": torch.tensor([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]]),
        "model2": torch.tensor([[0.1, -0.2, 0.3], [0.4, -0.5, -0.6]]),
    }
    expected_loss_dict = {
        "model1": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "model2": torch.tensor([[-0.1, 0.2, -0.3], [-0.4, 0.5, 0.6]]),
    }
    for model, expected_loss in expected_loss_dict.items():
        assert torch.allclose(
            next_logprobs_dict_to_loss_dict(next_logprobs_dict)[model], expected_loss
        )
