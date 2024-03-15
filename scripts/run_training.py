import argparse
import logging
from dataclasses import fields
from itertools import chain
from pathlib import Path
from typing import Union

from delphi.constants import CONFIG_PRESETS_DIR
from delphi.train.config.default_model_configs import default_config_dicts
from delphi.train.config.gigaconfig import GigaConfig
from delphi.train.config.utils import (
    build_config_from_files,
    get_preset_paths,
    get_user_config_path,
    update_config,
)
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


def build_model_arguments(parser: argparse.ArgumentParser):
    model_config_arg_group = parser.add_argument_group(f"Model config arguments")
    # first, map field names to config types they appear in
    param_name_to_config_type_names = dict()
    for model_type_str, default_model_config in default_config_dicts.items():
        for param_name in default_model_config.keys():
            if param_name not in param_name_to_config_type_names:
                param_name_to_config_type_names[param_name] = []
            param_name_to_config_type_names[param_name].append(model_type_str)
    # then add each field, list the model types that use it, the default values in each
    # model type, and check that all instances of the field across different model configs
    # have the same type
    for param_name, config_type_names in param_name_to_config_type_names.items():
        # get the actual values used in each config type
        config_type_to_val = {
            config_type_name: default_config_dicts[config_type_name][param_name]
            for config_type_name in config_type_names
        }
        # get type(s)
        types = tuple(type(val) for val in config_type_to_val.values())
        # if there is more than one type, define a new Union type
        param_type = Union[types] if len(set(types)) > 1 else types[0]  # type: ignore
        # build strings specifying per-model-config-type defaults
        default_strings = [
            f"{val} ({model_config_type})"
            for model_config_type, val in config_type_to_val.items()
        ]
        default_arg_string = "; ".join(default_strings)
        # actually add the argument
        model_config_arg_group.add_argument(
            f"--model_args.{param_name}",
            type=param_type,
            required=False,
            help=f"Model types: {list(config_type_to_val.keys())}. Defaults: {default_arg_string}",
        )


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
        # model_args is a special case
        if field.name == "model_args":
            continue
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
    build_model_arguments(parser)
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
