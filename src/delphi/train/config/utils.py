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
from beartype.typing import Any, Iterable
from dacite import from_dict

from delphi.constants import CONFIG_PRESETS_DIR

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


def get_preset_paths() -> Iterable[Path]:
    return Path(CONFIG_PRESETS_DIR).glob("*.json")  # type: ignore


def get_user_config_path() -> Path:
    _user_config_dir = Path(platformdirs.user_config_dir(appname="delphi"))
    _user_config_dir.mkdir(parents=True, exist_ok=True)
    user_config_path = _user_config_dir / "config.json"
    return user_config_path


def get_presets_by_name() -> dict[str, TrainingConfig]:
    return {
        preset.stem: build_config_from_files([preset]) for preset in get_preset_paths()
    }


def build_config_dict_from_files(config_files: list[Path]) -> dict[str, Any]:
    config_dicts = []
    for config_file in config_files:
        logging.debug(f"Loading {config_file}")
        with open(config_file, "r") as f:
            config_dicts.append(json.load(f))
    combined_config = merge_dicts(*config_dicts)
    return combined_config


def filter_config_to_actual_config_values(target_dataclass: Type, config: dict):
    """Remove non-config values from config dict.

    This can happen if e.g. being lazy and passing in all args from a script
    """
    datafields = fields(target_dataclass)
    name_to_field = {f.name: f for f in datafields}
    to_remove = []
    for k, v in config.items():
        if k not in name_to_field.keys():
            logging.debug(f"removing non-config-value {k}={v} from config dict")
            to_remove.append(k)
        elif isinstance(v, dict) and is_dataclass(name_to_field.get(k)):
            filter_config_to_actual_config_values(name_to_field[k].type, v)
    for k in to_remove:
        config.pop(k)


def set_backup_vals(config: dict[str, Any], config_files: list[Path]):
    if len(config_files) == 1:
        prefix = f"{config_files[0].stem}__"
    else:
        prefix = ""
    if "run_name" not in config:
        run_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        config["run_name"] = f"{prefix}{run_time}"
        logging.info(f"Setting run_name to {config['run_name']}")
    if "output_dir" not in config:
        config["output_dir"] = os.path.join(
            platformdirs.user_data_dir(appname="delphi"), config["run_name"]
        )
        logging.info(f"Setting output_dir to {config['output_dir']}")


def log_config_recursively(config: dict, logging_fn, indent="  ", prefix=""):
    for k, v in config.items():
        if isinstance(v, dict):
            logging_fn(f"{prefix}{k}")
            log_config_recursively(v, logging_fn, indent, prefix=indent + prefix)
        else:
            logging_fn(f"{prefix}{k}: {v}")


def cast_types(config: dict[str, Any], target_dataclass: Type):
    """
    user overrides are passed in as strings, so we need to cast them to the correct type
    """
    dc_fields = {f.name: f for f in fields(target_dataclass)}
    for k, v in config.items():
        if k in dc_fields:
            field = dc_fields[k]
            field_type = _unoptionalize(field.type)  # type: ignore
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
    combined_config = build_config_dict_from_files(config_files)
    cast_types(overrides, TrainingConfig)
    merge_two_dicts(merge_into=combined_config, merge_from=overrides)
    set_backup_vals(combined_config, config_files)
    filter_config_to_actual_config_values(TrainingConfig, combined_config)
    logging.debug("User-set config values:")
    log_config_recursively(
        combined_config, logging_fn=logging.debug, prefix="  ", indent="  "
    )
    return from_dict(TrainingConfig, combined_config)


def build_config_from_files(config_files: list[Path]) -> TrainingConfig:
    return build_config_from_files_and_overrides(config_files, {})


def load_preset(preset_name: str) -> TrainingConfig:
    preset_path = CONFIG_PRESETS_DIR / f"{preset_name}.json"
    return build_config_from_files([preset_path])


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
    """unwrap `Optional[T]` to T"""
    # Under the hood, `Optional` is really `Union[T, None]`. So we
    # just check if this is a Union over two types including None, and
    # return the other
    if hasattr(t, "__origin__") and t.__origin__ is Union:
        args = t.__args__
        # Check if one of the Union arguments is type None
        if len(args) == 2 and type(None) in args:
            return args[0] if args[1] is type(None) else args[1]
    return t
