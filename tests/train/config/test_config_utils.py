from pathlib import Path
from typing import cast

import pytest

from delphi.constants import CONFIG_PRESETS_DIR
from delphi.train.config.utils import (
    build_config_from_files_and_overrides,
    dot_notation_to_dict,
    merge_dicts,
    merge_two_dicts,
)


def test_merge_two_dicts():
    dict1 = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
    dict2 = {"a": 5, "c": {"d": 6}}
    merge_two_dicts(dict1, dict2)
    assert dict1 == {"a": 5, "b": 2, "c": {"d": 6, "e": 4}}


def test_merge_dicts():
    dict1 = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
    dict2 = {"a": 5, "c": {"d": 6}}
    dict3 = {"a": 7, "b": 8, "c": {"d": 9, "e": 10}}
    merged = merge_dicts(dict1, dict2, dict3)
    assert merged == {"a": 7, "b": 8, "c": {"d": 9, "e": 10}}


def test_dot_notation_to_dict():
    vars = {"a.b.c": 4, "foo": False}
    result = dot_notation_to_dict(vars)
    assert result == {"a": {"b": {"c": 4}}, "foo": False}


def test_build_config_from_files_and_overrides():
    config_files = [CONFIG_PRESETS_DIR / "debug.json"]
    overrides = {"model_config": {"hidden_size": 128}, "eval_iters": 5}
    config = build_config_from_files_and_overrides(config_files, overrides)
    # check overrides
    assert config.model_config["hidden_size"] == 128
    assert config.eval_iters == 5
    # check base values
    assert config.max_epochs == 2
    assert config.data_config.train_sample_limit == 256
