"""
Using existing token position mappings (token_maps pickle) and logprobs (HuggingFace Datasets),
gather the sum of logprobs and count of each token in each model. This allows for fast aggregation.
"""
from delphi.eval import constants

import argparse
import pickle
from tqdm import tqdm
from dataclasses import dataclass
from typing import cast
from datasets import Dataset, load_dataset
import pandas as pd


@dataclass
class TokenModelStats:
    """Stats for a token in a model, to enable efficient aggregation"""

    token: int
    model: str
    logprob_sum: float  # use sum instead of average to reduce floating point error
    count: int


def get_model_name_from_logprob_dataset_name(dataset_id: str) -> str:
    return dataset_id.split("/")[-1].split("-validation-logprobs")[0]


def load_token_map(tokens_maps_path: str) -> pd.DataFrame:
    """
    token_mappings is dict of token to document indices and positons (int -> list(tuple(int, int)))
    converting it to a df speeds up processing approximately 20x
    (From 3 hours to 10 minutes per dataset on my machine) - Jai
    """
    with open(tokens_maps_path, "rb") as f:
        raw_token_mappings = pickle.load(f)
    return pd.DataFrame(
        [
            (token, doc, pos)
            for token, locs in raw_token_mappings.items()
            for doc, pos in locs
        ],
        columns=["token", "doc", "position"],
    )


def gather_token_model_stats(
    token_mappings_path: str,
) -> dict[tuple[int, str], TokenModelStats]:
    token_mapping_df = load_token_map(token_mappings_path)
    token_model_stats = {}
    for model_dataset in constants.LLAMA2_LOGPROB_DATASETS:
        model_name = get_model_name_from_logprob_dataset_name(model_dataset)
        print(f"Processing model {model_dataset}")
        logprob_ds = cast(Dataset, load_dataset(model_dataset))["validation"]
        for _, row in tqdm(
            token_mapping_df.iterrows(), total=token_mapping_df.shape[0]
        ):
            token = int(row["token"])
            doc = int(row["doc"])
            pos = int(row["position"])
            # if a token occurs at the start of a document, we have no logprob for it
            if pos == 0:
                continue
            # logprob predicts the next token, so we need to subtract 1 from the position
            logprob = logprob_ds[doc]["logprobs"][pos - 1]
            if (token, model_name) not in token_model_stats:
                token_model_stats[(token, model_name)] = TokenModelStats(
                    token, model_name, logprob, 1
                )
            else:
                token_model_stats[(token, model_name)].logprob_sum += logprob
                token_model_stats[(token, model_name)].count += 1
    return token_model_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--token_maps_path",
        help="Path to the token maps pickle file",
        default="data/token_map.pkl",
    )
    parser.add_argument(
        "--output",
        help="Output file path",
        default="data/token_model_stats.csv",
    )
    args = parser.parse_args()
    token_model_stats_dct = gather_token_model_stats(args.token_maps_path)
    token_model_stats_df = pd.DataFrame(
        [
            (stats.token, stats.model, stats.logprob_sum, stats.count)
            for stats in token_model_stats_dct.values()
        ],
        columns=["token", "model", "logprob_sum", "count"],
    )
    with open(args.output, "wb") as f:
        token_model_stats_df.to_csv(f, index=False)
