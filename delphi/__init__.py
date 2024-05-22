from importlib.resources import files
from pathlib import Path
from typing import cast

from beartype.claw import beartype_this_package  # <-- hype comes

beartype_this_package()  # <-- hype goes

__version__ = "0.2"
TEST_CONFIGS_DIR = cast(Path, files("delphi.test_configs"))
