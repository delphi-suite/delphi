import argparse
import logging
from dataclasses import fields
from itertools import chain
from pathlib import Path

from delphi.constants import CONFIG_PRESETS_DIR
from delphi.train.config_utils import (
    build_config_from_files,
    get_preset_paths,
    get_user_config_path,
    update_config,
)
from delphi.train.gigaconfig import GigaConfig
from delphi.train.training import run_training


def get_preset_args(args: argparse.Namespace) -> list[Path]:
    cands = []
    for preset in get_preset_paths():
        if hasattr(args, preset.stem) and getattr(args, preset.stem):
            cands.append(preset)
    return cands


def get_config_files(args: argparse.Namespace) -> list[Path]:
    user_config_path = get_user_config_path()
    cands = [user_config_path] if user_config_path.exists() else []
    cands += get_preset_args(args)
    config_files = list(chain(*args.config_file)) if args.config_file else []
    cands += map(Path, config_files)
    configs = []
    for candpath in cands:
        if candpath.exists():
            configs.append(candpath)
            logging.info(f"Found config file {candpath}...")
        else:
            raise FileNotFoundError(candpath, f"Config file {candpath} does not exist.")
    return configs


def setup_parser() -> argparse.ArgumentParser:
    # Setup argparse
    parser = argparse.ArgumentParser(description="Train a delphi model")
    parser.add_argument(
        "--config_file",
        help=(
            "Path to json file(s) containing config values. Specific values can be overridden with --arguments. "
            "e.g. `--config_file primary_config.json secondary_config.json --log_interval 42`. "
            'If passing multiple configs with overlapping args, use "priority" key to specify precedence, e.g. {"priority": 100} '
            f'overrides {{"priority": 99}} See preset configs in {CONFIG_PRESETS_DIR}'
        ),
        action="append",
        nargs="*",
        required=False,
        type=str,
    )
    config_arg_group = parser.add_argument_group("Config arguments")
    for field in fields(GigaConfig):
        # support for nested attributes
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
    preset_arg_group = parser.add_argument_group("Preset configs")
    for preset in sorted(get_preset_paths()):
        preset_arg_group.add_argument(
            f"--{preset.stem}",
            help=f"Use {preset.stem} preset config",
            action="store_true",
        )
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    config_files = get_config_files(args)
    config = build_config_from_files(config_files)
    # specific arguments override everything else
    update_config(config, vars(args))

    # run training
    run_training(config)


if __name__ == "__main__":
    main()
