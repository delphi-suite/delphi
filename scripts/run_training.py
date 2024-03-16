import argparse
import logging
import os
from dataclasses import fields, is_dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Optional

from delphi.constants import CONFIG_PRESETS_DIR
from delphi.train.config.gigaconfig import GigaConfig
from delphi.train.config.utils import (
    build_config_from_files_and_overrides,
    get_preset_paths,
    get_user_config_path,
)
from delphi.train.training import run_training
from delphi.train.utils import get_run_output_dir, save_results


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


def add_dataclass_args_recursively(
    parser: argparse.ArgumentParser,
    dc: type[object],
    default_group: argparse._ArgumentGroup,
    group: Optional[argparse._ArgumentGroup] = None,
    prefix: str = "",
):
    for field in fields(dc):  # type: ignore
        if is_dataclass(field.type):
            _group = group or parser.add_argument_group(
                f"{field.name.capitalize()} arguments"
            )
            add_dataclass_args_recursively(
                parser,
                field.type,
                default_group,
                _group,
                prefix=f"{prefix}{field.name}.",
            )
        else:
            _group = group or default_group
            _group.add_argument(
                f"--{prefix}{field.name}",
                type=field.type,
                required=False,
                help=f"Default: {field.default}",
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
    add_dataclass_args_recursively(parser, GigaConfig, config_arg_group)
    preset_arg_group = parser.add_argument_group("Preset configs")
    for preset in sorted(get_preset_paths()):
        preset_arg_group.add_argument(
            f"--{preset.stem}",
            help=f"Use {preset.stem} preset config",
            action="store_true",
        )
    return parser


def var_args_to_dict(config_vars: dict[str, Any]) -> dict[str, Any]:
    # {"a.b.c" = 4} to {"a": {"b": {"c": 4}}}
    d = {}
    for k, v in config_vars.items():
        cur = d
        subkeys = k.split(".")
        for subkey in subkeys[:-1]:
            if subkey not in cur:
                cur[subkey] = {}
            cur = cur[subkey]
        if v is not None:
            cur[subkeys[-1]] = v
    return d


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    # at the toplevel, filter for args corresponding to field names in GigaConfig
    field_names = set(field.name for field in fields(GigaConfig))
    config_vars = {
        k: v for k, v in vars(args).items() if k.split(".")[0] in field_names
    }
    return var_args_to_dict(config_vars)


def main():
    parser = setup_parser()
    args = parser.parse_args()

    config_files = get_config_files(args)
    args_dict = args_to_dict(args)
    config = build_config_from_files_and_overrides(config_files, args_dict)

    # run training
    results, run_context = run_training(config)
    final_out_dir = os.path.join(get_run_output_dir(config), "final")
    save_results(config, results, run_context, final_out_dir)
    print(f"Saved results to {final_out_dir}")


if __name__ == "__main__":
    main()
