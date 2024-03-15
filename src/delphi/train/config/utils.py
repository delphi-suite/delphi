import json
import logging
from pathlib import Path

from beartype.typing import Any, Iterable
from platformdirs import user_config_dir

from delphi.constants import CONFIG_PRESETS_DIR
from delphi.train.config.gigaconfig import GigaConfig


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


def get_configs_in_priority_order(config_files: list[Path]) -> list[dict[str, Any]]:
    """loads config files in ascending priority order"""
    config_dicts = []
    for config_file in config_files:
        logging.info(f"Loading {config_file}")
        with open(config_file, "r") as f:
            config_dicts.append(json.load(f))
    config_dicts.sort(key=lambda cd: cd.get("priority", 0))
    return config_dicts


def update_config(config: GigaConfig, new_vals: dict[str, Any]):
    """update config in place. Supports dot notation (e.g. "x.y.z" = val) for nested attributes

    args:
        config: GigaConfig to be updated
        new_vals: dict of new values to update config with
    """
    for key, val in new_vals.items():
        if val is None:
            continue
        # support x.y.z = val
        keys = key.split(".")
        cur = config
        while len(keys) > 1:
            if hasattr(cur, keys[0]):
                cur = getattr(cur, keys.pop(0))
            else:
                break

        if hasattr(cur, keys[0]):
            setattr(cur, keys[0], val)
            print(f"Set {key} = {val}")
        else:
            print(f"Could not set {key} = {val}")


def combine_configs(configs: list[dict[str, Any]]) -> dict[str, Any]:
    # combine configs dicts, with key "priority" setting precendence (higher priority overrides lower priority)
    sorted_configs = sorted(configs, key=lambda c: c.get("priority", -999))
    combined_config = dict()
    for config in sorted_configs:
        combined_config.update(config)
    return combined_config


def build_config_from_files(config_files: list[Path]) -> GigaConfig:
    configs_in_order = get_configs_in_priority_order(config_files)
    config = GigaConfig()
    for _config in configs_in_order:
        update_config(config, _config)
    return config


def load_preset(preset_name: str) -> GigaConfig:
    preset_path = Path(CONFIG_PRESETS_DIR) / f"{preset_name}.json"  # type: ignore
    return build_config_from_files([preset_path])
