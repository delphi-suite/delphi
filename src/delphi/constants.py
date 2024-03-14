from importlib.resources import files

STATIC_ASSETS_DIR = files("delphi.static")
CONFIG_PRESETS_DIR = STATIC_ASSETS_DIR / "configs"

CORPUS_DATASET = "delphi-suite/tinystories-v2-clean"
TOKENIZED_CORPUS_DATASET = "delphi-suite/v0-tinystories-v2-clean-tokenized"


class ModelTypes:
    LLAMA2C = "llama2c"
    LLAMA2HF = "llama2-huggingface"
    MAMBA = "mamba"
