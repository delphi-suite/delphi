# Static Data Files



## `token_map.pkl`
pickle file: All locations of all tokens. dict of token to list of (doc, pos) pairs.

## `model_group_stats.pkl`
useful statistics for data visualization of (model, tokengroup) pairs; dict of (model, tokengroup) to dict of (str, float):
e.g. {("llama2", "Is Noun"): {"mean": -0.5, "median": -0.4, "min": -0.1, "max": -0.9, "25th": -0.3, "75th": -0.7}, ...}