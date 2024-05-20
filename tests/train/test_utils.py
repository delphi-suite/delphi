import torch
from datasets import Dataset

from delphi.train.utils import gen_minibatches


def test_gen_minibatches():
    DS_SIZE = 6
    SEQ_LEN = 5
    NUM_MINIBATCHES = 3
    MINIBATCH_SIZE = DS_SIZE // NUM_MINIBATCHES
    FEATURE_NAME = "tokens"
    dataset = Dataset.from_dict(
        {
            FEATURE_NAME: [list(range(i, i + SEQ_LEN)) for i in range(DS_SIZE)],
        },
    )
    dataset.set_format(type="torch")
    indices = list(range(DS_SIZE - 1, -1, -1))
    minibatches = gen_minibatches(
        dataset=dataset,
        batch_size=DS_SIZE,
        num_minibatches=NUM_MINIBATCHES,
        step=0,
        indices=indices,
        device=torch.device("cpu"),
        feature_name=FEATURE_NAME,
    )
    minibatches = list(minibatches)
    assert len(minibatches) == NUM_MINIBATCHES
    shuffled_ds = dataset[FEATURE_NAME][indices]  # type: ignore
    for i, minibatch in enumerate(minibatches):
        assert minibatch.shape == (MINIBATCH_SIZE, SEQ_LEN)
        expected_mb = shuffled_ds[i * MINIBATCH_SIZE : (i + 1) * MINIBATCH_SIZE]
        assert torch.all(minibatch == expected_mb)
