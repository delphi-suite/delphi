import json
import pandas as pd

from datasets import Dataset


splits = [
    ("../train/llama2c/data/TinyStoriesV2-GPT4-train-clean.json", "train"),
    ("../train/llama2c/data/TinyStoriesV2-GPT4-valid-clean.json", "validation")
]

def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)
    
    
for (filename, split) in splits:
    stories = load_dataset(filename)
    dataset = Dataset.from_pandas(pd.DataFrame(stories))
    dataset.push_to_hub(
        repo_id="",
        split=split,
        token=""
    )