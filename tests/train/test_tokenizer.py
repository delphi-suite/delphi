from transformers import AutoTokenizer

from delphi.train.dataset_tokenization import get_tokenized_batches


def test_get_tokenized_batches():
    CTX_SIZE = 10
    tokenizer = AutoTokenizer.from_pretrained("delphi-suite/v0-llama2-tokenizer")

    text_stories = [
        "Once upon a",
        "Mother woke up alert. She put on her coat",
        "Once upon a time, in a small town, there was a weird",
        "Once upon a time, there was a",
        "Sara and Tom are friends. They like to play in the park.",
    ]
    correct_batches = [
        [1, 432, 440, 261, 2, 367, 501, 1917, 372, 3398, 4037],
        [1, 4037, 341, 577, 359, 342, 1854, 2, 432, 440, 261],
        [1, 261, 403, 4045, 317, 261, 560, 1000, 4045, 406, 286],
        [1, 286, 261, 2567, 2, 432, 440, 261, 403, 4045, 406],
        [1, 406, 286, 261, 2, 787, 269, 396, 484, 415, 4037],
        [1, 4037, 311, 519, 268, 326, 317, 264, 525, 4037, 2],
    ]
    assert get_tokenized_batches(text_stories, tokenizer, CTX_SIZE) == correct_batches

    tokenized_stories = [
        [1618, 3520, 2223, 3961, 853, 3376, 1820, 1442, 1573],
        [46, 3515, 2941, 1637, 1377],
        [1439, 3378, 3897, 3807, 343, 1140, 3843, 3848, 1343, 3812, 947, 2871, 1973],
        [1163, 1358, 1930, 3590, 2216, 3659, 278],
        [604, 2920, 1330, 2240, 786, 4088, 1416, 2122, 1556, 3501, 3159, 3427],
    ]
    correct_batches = [
        [1, 1618, 3520, 2223, 3961, 853, 3376, 1820, 1442, 1573, 2],
        [1, 2, 46, 3515, 2941, 1637, 1377, 2, 1439, 3378, 3897],
        [1, 3897, 3807, 343, 1140, 3843, 3848, 1343, 3812, 947, 2871],
        [1, 2871, 1973, 2, 1163, 1358, 1930, 3590, 2216, 3659, 278],
        [1, 278, 2, 604, 2920, 1330, 2240, 786, 4088, 1416, 2122],
        [1, 2122, 1556, 3501, 3159, 3427, 2],
    ]
    assert (
        get_tokenized_batches(
            tokenized_stories, tokenizer, CTX_SIZE, input_tokenized=True
        )
        == correct_batches
    )
