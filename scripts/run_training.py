#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, Type, Union

import platformdirs

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
    group: argparse._ArgumentGroup,
    help_parsers: dict[str, argparse.ArgumentParser],
    prefix: str = "",
    depth: int = 0,
    max_help_depth=1,
):
    """Recursively add arguments to an argparse parser from a dataclass


    To keep --help sane, once we reach max_help_depth we start hiding options
    from --help and instead add a --<name>_help option to see config options
    below that level (e.g. model_config.llama config)
    """
    for field in fields(dc):  # type: ignore
        # if field is an Optional type, strip it to the actual underlying type
        _type = _unoptionalize(field.type)
        name = f"{prefix}{field.name}"
        if is_dataclass(_type):
            # at max-depth,
            if depth == max_help_depth:
                help_name = f"{name}_help"
                group.add_argument(
                    f"--{help_name}",
                    help=f"***Print help for {name} options***",
                    default=False,
                    action="store_true",
                )
                help_parser = argparse.ArgumentParser(help_name)
                help_group = help_parser.add_argument_group(name)
                help_parsers[help_name] = help_parser
                add_dataclass_args_recursively(
                    help_parser,
                    _type,
                    help_group,
                    help_parsers,
                    prefix=f"{name}.",
                    depth=depth + 1,
                    max_help_depth=999,
                )
            _group = parser.add_argument_group(f"{name}")
            add_dataclass_args_recursively(
                parser,
                _type,
                _group,
                help_parsers,
                prefix=f"{name}.",
                depth=depth + 1,
            )
        else:
            help_str: str = (
                str(field.metadata.get("help")) + ". "
                if field.metadata and "help" in field.metadata
                else ""
            )
            if depth > max_help_depth:
                help_str = argparse.SUPPRESS
            elif field.default != field.default_factory:
                help_str += f"Default: {field.default}"
            else:
                help_str += f"Must be specified as part of {group.title}"
            group.add_argument(
                f"--{name}",
                type=_type,
                required=False,
                help=help_str,
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
    logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.verbose is not None:
        if args.verbose == 1:
            loglevel = logging.DEBUG
        elif args.verbose >= 2:
            loglevel = 0
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


def setup_parser() -> (
    tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]
):
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
    help_parsers = dict()
    add_dataclass_args_recursively(parser, GigaConfig, config_arg_group, help_parsers)
    add_preset_args(parser)
    add_logging_args(parser)
    return parser, help_parsers


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


def special_help_if_invoked(args: argparse.Namespace, help_parsers: dict[str, Any]):
    for name, parser in help_parsers.items():
        if hasattr(args, name) and getattr(args, name):
            parser.print_help()
            exit(0)


def set_name_from_config_file(args: argparse.Namespace, config_files: list[Path]):
    """if no run_name is specified + exactly one config file is, use the name of the config file"""
    if args.run_name is None:
        configs = [c for c in config_files if c != get_user_config_path()]
        if len(configs) == 1:
            run_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            args.run_name = f"{configs[0].stem}__{run_time}"


def set_output_dir(args: argparse.Namespace):
    """if output_dir not set, set based on run name"""
    if args.output_dir is None:
        args.output_dir = os.path.join(
            platformdirs.user_data_dir(appname="delphi"), args.run_name
        )


def main():
    parser, help_parsers = setup_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)
    special_help_if_invoked(args, help_parsers)
    set_logging(args)

    config_files = get_config_files(args)
    set_name_from_config_file(args, config_files)
    set_output_dir(args)
    args_dict = args_to_dict(args)
    config = build_config_from_files_and_overrides(config_files, args_dict)
    # run training
    results, run_context = run_training(config)
    final_out_dir = os.path.join(config.output_dir, "final")
    save_results(config, results, run_context, final_out_dir)
    print(f"Saved results to {final_out_dir}")


if __name__ == "__main__":
    main()
