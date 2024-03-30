from importlib.resources import files

STATIC_ASSETS_DIR = files("delphi.static")
CONFIG_PRESETS_DIR = STATIC_ASSETS_DIR / "configs"

CORPUS_DATASET = "delphi-suite/stories"
TINYSTORIES_TOKENIZED_HF_DATASET = "delphi-suite/v0-tinystories-v2-clean-tokenized"
