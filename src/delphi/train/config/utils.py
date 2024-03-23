import json
import logging
import os
from datetime import datetime
from pathlib import Path

import platformdirs
from beartype.typing import Any, Iterable
from dacite import from_dict

from delphi.constants import CONFIG_PRESETS_DIR

from .gigaconfig import GigaConfig


def _merge_dicts(merge_into: dict[str, Any], merge_from: dict[str, Any]):
    """recursively merge two dicts, with values in merge_from taking precedence"""
    for key, val in merge_from.items():
        if (
            key in merge_into
            and isinstance(merge_into[key], dict)
            and isinstance(val, dict)
        ):
            _merge_dicts(merge_into[key], val)
        else:
            merge_into[key] = val


def get_preset_paths() -> Iterable[Path]:
    return Path(CONFIG_PRESETS_DIR).glob("*.json")  # type: ignore


def get_user_config_path() -> Path:
    _user_config_dir = Path(platformdirs.user_config_dir(appname="delphi"))
    _user_config_dir.mkdir(parents=True, exist_ok=True)
    user_config_path = _user_config_dir / "config.json"
    return user_config_path


def get_presets_by_name() -> dict[str, GigaConfig]:
    return {
        preset.stem: build_config_from_files([preset]) for preset in get_preset_paths()
    }


def get_config_dicts_from_files(config_files: list[Path]) -> list[dict[str, Any]]:
    """loads config files in ascending priority order"""
    config_dicts = []
    for config_file in config_files:
        logging.info(f"Loading {config_file}")
        with open(config_file, "r") as f:
            config_dicts.append(json.load(f))
    return config_dicts


def combine_configs(configs: list[dict[str, Any]]) -> dict[str, Any]:
    # combine configs dicts, with key "priority" setting precendence (higher priority overrides lower priority)
    sorted_configs = sorted(configs, key=lambda c: c.get("priority", -999))
    combined_config = dict()
    for config in sorted_configs:
        _merge_dicts(merge_into=combined_config, merge_from=config)
    return combined_config


def build_config_dict_from_files(config_files: list[Path]) -> dict[str, Any]:
    configs_in_order = get_config_dicts_from_files(config_files)
    combined_config = combine_configs(configs_in_order)
    return combined_config


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


def build_config_from_files_and_overrides(
    config_files: list[Path],
    overrides: dict[str, Any],
) -> GigaConfig:
    combined_config = build_config_dict_from_files(config_files)
    _merge_dicts(merge_into=combined_config, merge_from=overrides)
    set_backup_vals(combined_config, config_files)
    return from_dict(GigaConfig, combined_config)


def build_config_from_files(config_files: list[Path]) -> GigaConfig:
    return build_config_from_files_and_overrides(config_files, {})


def load_preset(preset_name: str) -> GigaConfig:
    preset_path = Path(CONFIG_PRESETS_DIR) / f"{preset_name}.json"  # type: ignore
    return build_config_from_files([preset_path])
