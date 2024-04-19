from importlib.resources import files
from pathlib import Path
from typing import cast

TEST_CONFIGS_DIR = cast(Path, files("delphi.test_configs"))

CORPUS_DATASET = "delphi-suite/stories"
TINYSTORIES_TOKENIZED_HF_DATASET = "delphi-suite/v0-tinystories-v2-clean-tokenized"
