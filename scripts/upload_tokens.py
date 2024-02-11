import pandas as pd
from datasets import Dataset
from functools import partial

from delphi import PretokDataset


batch_size = 1
max_seq_len = 512
vocab_size = 4096
vocab_source = "custom"
device = "cuda"

for split in ["train", "validation"]:

    ds = PretokDataset(
        split=split,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        vocab_source=vocab_source,
    )
    
    num_batches = len(PretokDataset)
    
    tokens = []
    for idx, (chunk) in enumerate(ds):
        if idx >= num_batches: 
            break
        tokens.append({'tokens': chunk.numpy()})
        
    dataset = Dataset.from_pandas(pd.DataFrame(tokens))
    dataset.push_to_hub(
        repo_id="",
        split=split,
        token=""
    )
