from datasets import load_dataset
from tqdm.auto import tqdm


def load_clean_dataset(split: str, tokenized: bool = False) -> list[str]:
    # checking just startswith, because you can include slice like "train[:1000]"
    assert split.startswith("train") or split.startswith("validation")
    hf_ds = load_dataset(
        f"jbrinkma/tinystories-v2-clean{'-tokenized' if tokenized else ''}",
        split=split,
    )
    dataset = []
    # hf_ds technically isn't guaranteed to be subscriptable, but it is in this case
    for sample in tqdm(hf_ds["tokens" if tokenized else "text"]):  # type: ignore
        dataset.append(sample)
    return dataset
