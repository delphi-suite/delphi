import json
import logging
from pathlib import Path

from beartype.typing import Any, Iterable

from delphi.constants import CONFIG_PRESETS_DIR
from delphi.train.gigaconfig import GigaConfig


def get_presets() -> Iterable[Path]:
    return Path(CONFIG_PRESETS_DIR).glob("*.json")  # type: ignore


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
