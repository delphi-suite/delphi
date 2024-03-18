#!/usr/bin/env python3
import argparse
import logging
import os
from dataclasses import fields, is_dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Optional, Type, Union

from delphi.constants import CONFIG_PRESETS_DIR
from delphi.train.config import (
    GigaConfig,
    build_config_from_files_and_overrides,
    get_preset_paths,
    get_user_config_path,
)
from delphi.train.training import run_training
from delphi.train.utils import save_results


def _unoptionalize(t: Type) -> Type:
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


def get_preset_args(args: argparse.Namespace) -> list[Path]:
    cands = []
    for preset in get_preset_paths():
        if hasattr(args, preset.stem) and getattr(args, preset.stem):
            cands.append(preset)
    return cands


def get_config_files(args: argparse.Namespace) -> list[Path]:
    user_config_path = get_user_config_path()
    cands = [user_config_path] if user_config_path.exists() else []
    cands += get_preset_paths()
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


def add_preset_args(parser: argparse.ArgumentParser):
    preset_arg_group = parser.add_argument_group("Preset configs")
    for preset in sorted(get_preset_paths()):
        preset_arg_group.add_argument(
            f"--{preset.stem}",
            help=f"Use {preset.stem} preset config {'***and set log level to DEBUG***' if preset.stem == 'debug' else ''}",
            action="store_true",
        )


def add_dataclass_args_recursively(
    parser: argparse.ArgumentParser,
    dc: type[object],
    default_group: argparse._ArgumentGroup,
    group: Optional[argparse._ArgumentGroup] = None,
    prefix: str = "",
):
    for field in fields(dc):  # type: ignore
        # if field is an Optional type, strip it to the actual underlying type
        _type = _unoptionalize(field.type)
        if is_dataclass(_type):
            _group = group or parser.add_argument_group(f"{field.name}")
            add_dataclass_args_recursively(
                parser,
                _type,
                default_group,
                _group,
                prefix=f"{prefix}{field.name}.",
            )
        else:
            _group = group or default_group
            _group.add_argument(
                f"--{prefix}{field.name}",
                type=_type,
                required=False,
                help=f"Default: {field.default}"
                if field.default != field.default_factory
                else f"Must be specified as part of {_group.title}",
            )


def add_logging_args(parser: argparse.ArgumentParser):
    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=None,
        help="Increase verbosity level, repeatable (e.g. -vvv). Mutually exclusive with --silent, --loglevel",
    )
    logging_group.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Silence all logging. Mutually exclusive with --verbose, --loglevel",
        default=False,
    )
    logging_group.add_argument(
        "--loglevel",
        type=int,
        help="Logging level. 10=DEBUG, 50=CRITICAL. Mutually exclusive with --verbose, --silent",
        default=None,
    )


def set_logging(args: argparse.Namespace):
    logging.basicConfig(format="%(message)s")
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose is not None:
        if args.verbose == 1:
            loglevel = logging.INFO
        elif args.verbose == 2:
            loglevel = logging.DEBUG
        else:
            loglevel = logging.DEBUG - 10 * (args.verbose - 2)
        logging.getLogger().setLevel(loglevel)
    if args.loglevel is not None:
        logging.getLogger().setLevel(args.loglevel)
    if args.silent:
        logging.getLogger().setLevel(logging.CRITICAL)
    else:
        logging_level_str = logging.getLevelName(
            logging.getLogger().getEffectiveLevel()
        )
        print(f"set logging level to {logging_level_str}")


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
    add_preset_args(parser)
    add_logging_args(parser)
    return parser


def var_args_to_dict(config_vars: dict[str, Any]) -> dict[str, Any]:
    # {"a.b.c" = 4} to {"a": {"b": {"c": 4}}}
    d = {}
    for k, v in config_vars.items():
        if v is None:
            continue
        cur = d
        subkeys = k.split(".")
        for subkey in subkeys[:-1]:
            if subkey not in cur:
                cur[subkey] = {}
            cur = cur[subkey]
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
    set_logging(args)

    config_files = get_config_files(args)
    args_dict = args_to_dict(args)
    config = build_config_from_files_and_overrides(config_files, args_dict)
    # run training
    results, run_context = run_training(config)
    final_out_dir = os.path.join(config.output_dir, "final")
    save_results(config, results, run_context, final_out_dir)
    print(f"Saved results to {final_out_dir}")


if __name__ == "__main__":
    main()
