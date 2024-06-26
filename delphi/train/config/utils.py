import ast
import json
import logging
import os
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import _GenericAlias  # type: ignore
from typing import Any, Type, TypeVar, Union

import platformdirs
from dacite import Config as dacite_config
from dacite import from_dict

from .training_config import TrainingConfig

T = TypeVar("T")


def merge_two_dicts(merge_into: dict[str, Any], merge_from: dict[str, Any]):
    """recursively merge two dicts, with values in merge_from taking precedence"""
    for key, val in merge_from.items():
        if (
            key in merge_into
            and isinstance(merge_into[key], dict)
            and isinstance(val, dict)
        ):
            merge_two_dicts(merge_into[key], val)
        else:
            merge_into[key] = val


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge multiple dictionaries, with later dictionaries taking precedence.
    """
    merged = {}
    for d in dicts:
        merge_two_dicts(merged, d)
    return merged


def get_user_config_path() -> Path:
    """
    This enables a user-specific config to always be included in the training config.

    This is useful for things like wandb config, where you'll generally want to use your own account.
    """
    _user_config_dir = Path(platformdirs.user_config_dir(appname="delphi"))
    _user_config_dir.mkdir(parents=True, exist_ok=True)
    user_config_path = _user_config_dir / "config.json"
    return user_config_path


def build_config_dict_from_files(config_files: list[Path]) -> dict[str, Any]:
    """
    Given a list of config json paths, merge them into a combined config dict (with later files taking precedence).
    """
    config_dicts = []
    for config_file in config_files:
        logging.debug(f"Loading {config_file}")
        with open(config_file, "r") as f:
            config_dicts.append(json.load(f))
    combined_config = merge_dicts(*config_dicts)
    return combined_config


def cast_types(config: dict[str, Any], target_dataclass: Type):
    """
    user overrides are passed in as strings, so we need to cast them to the correct type
    """
    dc_fields = {f.name: f for f in fields(target_dataclass)}
    for k, v in config.items():
        if k in dc_fields:
            field = dc_fields[k]
            field_type = _unoptionalize(field.type)
            if is_dataclass(field_type):
                cast_types(v, field_type)
            elif isinstance(field_type, dict):
                #  for dictionaries, make best effort to cast values to the correct type
                for _k, _v in v.items():
                    v[_k] = ast.literal_eval(_v)
            else:
                config[k] = field_type(v)


def build_config_from_files_and_overrides(
    config_files: list[Path],
    overrides: dict[str, Any],
) -> TrainingConfig:
    """
    This is the main entrypoint for building a TrainingConfig object from a list of config files and overrides.

    1. Load config_files in order, merging them into one dict, with later taking precedence.
    2. Cast the strings from overrides to the correct types
        (we expect this to be passed as strings w/o type hints from a script argument:
        e.g. `--overrides model_config.hidden_size=42 run_name=foo`)
    3. Merge in overrides to config_dict, taking precedence over all config_files values.
    4. Build the TrainingConfig object from the final config dict and return it.
    """
    combined_config = build_config_dict_from_files(config_files)
    cast_types(overrides, TrainingConfig)
    merge_two_dicts(merge_into=combined_config, merge_from=overrides)
    return from_dict(TrainingConfig, combined_config, config=dacite_config(strict=True))


def dot_notation_to_dict(vars: dict[str, Any]) -> dict[str, Any]:
    """
    Convert {"a.b.c": 4, "foo": false} to {"a": {"b": {"c": 4}}, "foo": False}
    """
    nested_dict = dict()
    for k, v in vars.items():
        if v is None:
            continue
        cur = nested_dict
        subkeys = k.split(".")
        for subkey in subkeys[:-1]:
            if subkey not in cur:
                cur[subkey] = {}
            cur = cur[subkey]
        cur[subkeys[-1]] = v
    return nested_dict


def _unoptionalize(t: Type | _GenericAlias) -> Type:
    """unwrap `Optional[T]` to T.

    We need this to correctly interpret user-passed overrides, which are always strings
    without any type information attached. We need to look up what type they should be
    and cast accordingly. As part of this lookup we need to pierce Optional values -
    if the user is setting a value, it's clearly not Optional, and we need to get the underlying
    type to cast correctly.
    """
    # Under the hood, `Optional` is really `Union[T, None]`. So we
    # just check if this is a Union over two types including None, and
    # return the other
    if hasattr(t, "__origin__") and t.__origin__ is Union:
        args = t.__args__
        # Check if one of the Union arguments is type None
        if len(args) == 2 and type(None) in args:
            return args[0] if args[1] is type(None) else args[1]
    return t
