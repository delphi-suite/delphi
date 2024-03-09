import argparse
import copy
import json
from dataclasses import fields
from typing import Any

from delphi.train.gigaconfig import GigaConfig, debug_config
from delphi.train.training import run_training


def update_config(config: GigaConfig, new_vals: dict[str, Any]):
    for key, val in new_vals.items():
        if val is None:
            continue
        # support x.y.z = val
        keys = key.split(".")
        cur = config
        while len(keys) > 1:
            cur = getattr(cur, keys.pop(0))
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
        required=False,
        type=str,
    )
    parser.add_argument(
        "--debug",
        help="Use debug config values. Overridden by config file values and --arguments.",
        required=False,
        action="store_true",
    )
    # TODO: add presets as static json files and make accessible
    args = parser.parse_args()

    # setup config
    if args.debug:
        config = copy.copy(debug_config)
    else:
        config = GigaConfig()
    # config file overrides default values
    if args.config_file is not None:
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)
        update_config(config, config_dict)
    # specific arguments override everything else
    update_config(config, vars(args))

    # run training
    run_training(config)


if __name__ == "__main__":
    main()
