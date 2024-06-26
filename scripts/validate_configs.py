#!/usr/bin/env python3
import argparse
import pathlib

from delphi.train.config import build_config_from_files_and_overrides
from delphi.train.utils import init_model, overrides_to_dict


def get_config_path_with_base(config_path: pathlib.Path) -> list[pathlib.Path]:
    """If config path is in directory which includes base.json, include that as the first config."""
    if (config_path.parent / "base.json").exists():
        return [config_path.parent / "base.json", config_path]
    return [config_path]


def get_config_paths(config_path: str) -> list[list[pathlib.Path]]:
    """If config path is a directory, recursively glob all json files in it. Otherwise, just use the path and create a list of 1."""
    paths = (
        list(pathlib.Path(config_path).rglob("*.json"))
        if pathlib.Path(config_path).is_dir()
        else [pathlib.Path(config_path)]
    )
    # exclude base.json files
    paths = [path for path in paths if not path.name.startswith("base")]
    # supplement non-base configs with base.json if it exists in same dir
    return [get_config_path_with_base(path) for path in paths]


def main():
    parser = argparse.ArgumentParser()
    # we take one positional argument, a path to a directory or config
    parser.add_argument(
        "config_path",
        type=str,
        help="path to a training config json or directory of training config jsons",
    )
    parser.add_argument(
        "--overrides",
        help=(
            "Override config values with space-separated declarations. "
            "e.g. `--overrides model_config.hidden_size=42 run_name=foo`"
        ),
        type=str,
        required=False,
        nargs="*",
        default=[],
    )
    parser.add_argument("--init", help="initialize the model", action="store_true")
    args = parser.parse_args()
    config_paths = get_config_paths(args.config_path)
    print(
        f"validating configs: {' | '.join(str(config_path[-1]) for config_path in config_paths)}"
    )
    overrides = overrides_to_dict(args.overrides)
    errors = []
    sizes = []
    for config_path in config_paths:
        try:
            config = build_config_from_files_and_overrides(config_path, overrides)
            if args.init:
                model = init_model(config.model_config, seed=config.torch_seed)
                sizes.append((config_path, model.num_parameters()))
        except Exception as e:
            errors.append((config_path, e))
            continue
    if errors:
        print("errors:")
        for config_path, e in errors:
            print(f"  {config_path[-1]}: {e}")
    else:
        print("all configs loaded successfully")
        if sizes:
            print("model sizes:")
            for config_path, size in sizes:
                print(f"  {config_path[-1]}: {size}")


if __name__ == "__main__":
    main()
