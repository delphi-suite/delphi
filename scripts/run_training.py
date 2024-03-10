import argparse
import copy
import json
import logging
import pathlib as path
from dataclasses import fields
from importlib.resources import files

# import stdlib function for flattening nested lists
from itertools import chain
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

from delphi.constants import CONFIG_PRESETS_DIR
from delphi.train.gigaconfig import GigaConfig, debug_config
from delphi.train.training import run_training


def get_presets():
    return Path(CONFIG_PRESETS_DIR).glob("*.json")  # type: ignore


def get_user_config_path() -> Path:
    _user_config_dir = Path(user_config_dir(appname="delphi"))
    _user_config_dir.mkdir(parents=True, exist_ok=True)
    user_config_path = _user_config_dir / "config.json"
    return user_config_path


def get_config_files(args: argparse.Namespace) -> list[Path]:
    user_config_path = get_user_config_path()
    cands = [user_config_path] if user_config_path.exists() else []
    # flatten args.config_file, which is a nested list
    config_files = list(chain(*args.config_file))
    print(config_files)
    cands += map(Path, config_files) if args.config_file else []
    configs = []
    for candpath in cands:
        if candpath.exists():
            configs.append(candpath)
        else:
            logging.error(f"Config file {candpath} does not exist, exiting.")
            exit(1)
    return configs


def update_config(config: GigaConfig, new_vals: dict[str, Any]):
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

    for field in fields(config):
        if new_vals.get(field.name) is not None:
            setattr(config, field.name, new_vals[field.name])


def main():
    # Setup argparse
    parser = argparse.ArgumentParser(description="Train a delphi model")
    config_arg_group = parser.add_argument_group("Config arguments")
    for field in fields(GigaConfig):
        config_arg_group.add_argument(
            f"--{field.name}",
            type=field.type,
            required=False,
            help=f"Default: {field.default}",
        )
    parser.add_argument(
        "--config_file",
        help=(
            "Path to a json file containing config values (see sample_config.json). "
            "Specific values can be overridden with --arguments."
        ),
        action="append",
        nargs="*",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--debug",
        help="Use debug config values. Overridden by config file values and --arguments.",
        required=False,
        action="store_true",
    )
    for preset in get_presets():
        parser.add_argument(
            f"--{preset.stem}",
            help=f"Use {preset.stem} preset config",
            action="store_true",
        )
    args = parser.parse_args()

    # setup config
    if args.debug:
        config = copy.copy(debug_config)
    else:
        config = GigaConfig()
    # config file overrides default values
    config_files = get_config_files(args)
    for config_file in config_files:
        logging.info(f"Loading {config_file}")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        update_config(config, config_dict)
    # specific arguments override everything else
    update_config(config, vars(args))

    # run training
    run_training(config)


if __name__ == "__main__":
    main()
