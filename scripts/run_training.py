import argparse
import json
import logging
from dataclasses import fields
from itertools import chain
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

from delphi.constants import CONFIG_PRESETS_DIR
from delphi.train.gigaconfig import GigaConfig
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
    # get all preset args
    for preset in get_presets():
        if hasattr(args, preset.stem) and getattr(args, preset.stem):
            cands.append(preset)
    # flatten args.config_file, which is a nested list
    config_files = list(chain(*args.config_file)) if args.config_file else []
    cands += map(Path, config_files)
    configs = []
    for candpath in cands:
        if candpath.exists():
            configs.append(candpath)
            logging.info(f"Found config file {candpath}...")
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
        else:
            logging.warning(f"Config field {key} does not exist?")


def setup_parser() -> argparse.ArgumentParser:
    # Setup argparse
    parser = argparse.ArgumentParser(description="Train a delphi model")
    parser.add_argument(
        "--config_file",
        help=(
            "Path to json file(s) containing config values. "
            "Specific values can be overridden with --arguments. "
            "e.g. '--config_file primary_config.json secondary_config.json --log_interval 42'. "
            f"See preset configs in {CONFIG_PRESETS_DIR}"
        ),
        action="append",
        nargs="*",
        required=False,
        type=str,
    )
    config_arg_group = parser.add_argument_group("Config arguments")
    for field in fields(GigaConfig):
        # test if field is a dataclass
        if hasattr(field.type, "__dataclass_fields__"):
            sub_arg_group = parser.add_argument_group(f"Config {field.name} arguments")
            for subfield in fields(field.type):
                sub_arg_group.add_argument(
                    f"--{field.name}.{subfield.name}",
                    type=subfield.type,
                    required=False,
                    help=f"Default: {subfield.default}",
                )
        else:
            config_arg_group.add_argument(
                f"--{field.name}",
                type=field.type,
                required=False,
                help=f"Default: {field.default}",
            )
    preset_arg_group = parser.add_argument_group("Presets configs")
    for preset in sorted(get_presets()):
        preset_arg_group.add_argument(
            f"--{preset.stem}",
            help=f"Use {preset.stem} preset config",
            action="store_true",
        )
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    # base config
    config = GigaConfig()

    # config files override default values
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
