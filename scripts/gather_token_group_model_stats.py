# Aggregate logprobs for token groups for each model

from delphi.eval.constants import LLAMA2_MODELS

from delphi.eval.hack_token_label import HackTokenLabels

import json
import pandas as pd

with open("../../../data/token_groups.json", "r") as f:
    token_groups = json.load(f)

with open("../../../data/token_model_stats.csv", "rb") as f:
    token_model_stats = pd.read_csv(f)

model_token_group_stats = {}
for model in LLAMA2_MODELS:
    print(f"Processing model {model}")
    for token_group_desc in HackTokenLabels.ALL_LABELS:
        token_group_model_stats = token_model_stats[
            token_model_stats.token.isin(token_groups[token_group_desc.description])
            & (token_model_stats.model == model)
        ]
        sum_logprob = token_group_model_stats["logprob_sum"].sum()
        sum_count = token_group_model_stats["count"].sum()
        if sum_count == 0:
            continue
        mean_logprob = sum_logprob / sum_count
        model_token_group_stats[(token_group_desc.description, model)] = mean_logprob

# model_token_group_stats to df with columns token_group, model, logprob

model_token_group_stats_df = pd.DataFrame(
    [
        (token_group, model, logprob)
        for (token_group, model), logprob in model_token_group_stats.items()
    ],
    columns=["token_group", "model", "logprob"],
)
