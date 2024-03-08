import argparse
import copy
from dataclasses import fields

from delphi.train.gigaconfig import GigaConfig, debug_config
from delphi.train.training import run_training


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
        "--debug",
        help="Use debug config values (can still override with other arguments)",
        required=False,
        action="store_true",
    )
    args = parser.parse_args()

    if args.debug:
        config = copy.copy(debug_config)
    else:
        config = GigaConfig()
    for field in fields(GigaConfig):
        if getattr(args, field.name) is not None:
            setattr(config, field.name, getattr(args, field.name))
    run_training(config)


if __name__ == "__main__":
    main()
