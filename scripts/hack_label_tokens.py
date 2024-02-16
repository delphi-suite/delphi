from transformers import AutoTokenizer

from delphi.eval.hack_token_label import HackTokenLabels, label_vocalulary
from delphi.eval.utils import GenericPreTrainedTokenizer
from delphi.eval.constants import LLAMA2_HF_MODELS

import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--output",
        help="Output file path",
        default="data/token_groups.json",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(LLAMA2_HF_MODELS[0])
    labeled_vocab = label_vocalulary(tokenizer)
    with open(args.output, "w") as f:
        json.dump(labeled_vocab, f, indent=2)
