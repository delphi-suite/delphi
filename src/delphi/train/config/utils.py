import json
import logging
from pathlib import Path

from beartype.typing import Any, Iterable
from dacite import from_dict
from platformdirs import user_config_dir

from delphi.constants import CONFIG_PRESETS_DIR
from delphi.train.config.gigaconfig import GigaConfig


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
    _user_config_dir = Path(user_config_dir(appname="delphi"))
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


def build_config_from_files_and_overrides(
    config_files: list[Path], overrides: dict[str, Any]
) -> GigaConfig:
    combined_config = build_config_dict_from_files(config_files)
    _merge_dicts(merge_into=combined_config, merge_from=overrides)
    return from_dict(GigaConfig, combined_config)


def build_config_from_files(config_files: list[Path]) -> GigaConfig:
    return build_config_from_files_and_overrides(config_files, {})


def load_preset(preset_name: str) -> GigaConfig:
    preset_path = Path(CONFIG_PRESETS_DIR) / f"{preset_name}.json"  # type: ignore
    return build_config_from_files([preset_path])
