#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from delphi.train.config import (
    build_config_from_files_and_overrides,
    dot_notation_to_dict,
)
from delphi.train.training import run_training
from delphi.train.utils import save_results


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


def set_logging(args: argparse.Namespace):
    logging.basicConfig(format="%(message)s")
    logging.getLogger().setLevel(logging.INFO)
    if args.verbose is not None:
        if args.verbose == 1:
            loglevel = logging.DEBUG
        elif args.verbose >= 2:
            loglevel = 0
        logging.getLogger().setLevel(loglevel)
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
        "--config_files",
        "--config_file",
        "-c",
        help=(
            "Path to json file(s) containing config values. Specific values can be overridden with --overrides. "
            "e.g. `--config_files primary_config.json secondary_config.json"
        ),
        type=str,
        required=False,
        nargs="*",
    )
    parser.add_argument(
        "--overrides",
        help=(
            "Override config values with comma-separated declarations. "
            "e.g. `--overrides model_config.hidden_size=42 run_name=foo`"
        ),
        type=str,
        required=False,
        nargs="*",
        default=[],
    )
    add_logging_args(parser)
    return parser


def overrides_to_dict(overrides: list[str]) -> dict[str, Any]:
    # ["a.b.c=4", "foo=false"] to {"a": {"b": {"c": 4}}, "foo": False}
    config_vars = {k: v for k, v in [x.split("=") for x in overrides if "=" in x]}
    return dot_notation_to_dict(config_vars)


def main():
    parser = setup_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)
    set_logging(args)

    args_dict = overrides_to_dict(args.overrides)
    config_files = [Path(f) for f in args.config_files]
    config = build_config_from_files_and_overrides(config_files, args_dict)
    # run training
    results, run_context = run_training(config)
    # to save & upload to iterX folder/branch
    save_results(config, results, run_context, final=False)
    # to save & upload to main folder/branch
    save_results(config, results, run_context, final=True)


if __name__ == "__main__":
    main()
