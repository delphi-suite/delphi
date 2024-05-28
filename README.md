# delphi

delphi is a set of tools for standardized and (mostly) reproducible training of small language models. You can use delphi to train a custom tokenizer, tokenize your dataset, and train your model. We build on top of HuggingFace, supporting every `CausalLM` architecture. Datasets, tokenizers and models (including checkpoints!) can be downloaded from and uploaded to HuggingFace automatically, with no need to manage local files.


# Setup

1. Clone the repo
```shell
git clone https://github.com/delphi-suite/delphi.git
cd delphi  
```
2. Make & activate python >= 3.10 virtual env
```shell
python3.10 -m venv .venv
source .venv/bin/activate
```
3. Install the project in editable state  
`pip install -e .`  
See `[project.optional-dependencies]` section in `pyproject.toml` for additional dependencies, e.g. you may want to `pip install -e ."[dev,mamba_cuda]"`
4. get your HuggingFace and W&B tokens and put them in the environment variables
```shell
export HF_TOKEN=...
export WANDB_API_KEY=...
```


# Training a tokenizer

If you want to train a small and efficient model on a narrow dataset, then we recommend using a custom tokenizer with a small vocabulary. To train a reversible, GPT2-style, BPE tokenizer you can use `scripts/train_tokenizer.py`.

Script usage:

```
> scripts/train_tokenizer.py --help
usage: train_tokenizer.py [-h] --in-dataset IN_DATASET --feature FEATURE --split SPLIT
                          --vocab-size VOCAB_SIZE
                          [--out-dir OUT_DIR] [--out-repo OUT_REPO]

Train a custom, reversible, BPE tokenizer (GPT2-like). You need to provide --out-repo or --out-dir.

options:
  -h, --help            show this help message and exit
  --in-dataset IN_DATASET, -i IN_DATASET
                        Dataset you want to train the tokenizer on. Local path or HF repo id
  --feature FEATURE, -f FEATURE
                        Name of the feature (column) containing text documents in the input dataset
  --split SPLIT, -s SPLIT
                        Split of the dataset to be used for tokenizer training, supports slicing like 'train[:10%]'
  --vocab-size VOCAB_SIZE, -v VOCAB_SIZE
                        Vocabulary size of the tokenizer
  --out-dir OUT_DIR     Local directory to save the resulting tokenizer
  --out-repo OUT_REPO   HF repo id to upload the resulting tokenizer
```

Here's how we trained the tokenizer for our `stories-*` suite of models. Please note that you can use single letter abbreviations for most arguments.

```
> scripts/train_tokenizer.py \
    --in-dataset delphi-suite/stories \
    --feature story \
    --split train \
    --vocab-size 4096 \
    --out-repo delphi-suite/stories-tokenizer
```

We use the only feature named `story` in the `train` split of [delphi-suite/stories](https://huggingface.co/datasets/delphi-suite/stories). We train a tokenizer with a vocabulary of 4096 tokens, and upload it to HF model repo [delphi-suite/stories-tokenizer](https://huggingface.co/delphi-suite/stories-tokenizer).


# Tokenizing a dataset

To turn a collection of text documents into sequences of tokens required for model training, you can use `scripts/tokenize_dataset.py`. All documents are tokenized and concatenated, with the `<eos>` token as a separator, e.g.
```
doc1_tok1, doc1_tok2, ..., doc1_tokX, <eos>, doc2_tok1, doc2_tok2, ..., doc2_tokX, <eos>, doc3_tok1, ...
```
Then this is divided into chunks, and the `<bos>` token is inserted at the begining of each chunk, e.g.
```
<bos> doc1_tok1, doc1_tok2, ..., doc1_tokX, <eos>, doc2_tok1
<bos> doc2_tok2, ..., doc2_tok511
<bos> doc2_tok512, doc2_tok513, ..., doc2_tokX <eos>, doc3_tok1, ...
...
```
It will produce sequences of specified size, by discarding the last chunk if it's too short. We don't use padding.

Script usage:

```
> scripts/tokenize_dataset.py --help
usage: tokenize_dataset.py [-h] --in-dataset IN_DATASET --feature FEATURE --split SPLIT
                           --tokenizer TOKENIZER --seq-len SEQ_LEN
                           [--batch-size BATCH_SIZE] [--chunk-size CHUNK_SIZE]
                           [--out-dir OUT_DIR] [--out-repo OUT_REPO]

Tokenize a text dataset using a specific tokenizer

options:
  -h, --help            show this help message and exit
  --in-dataset IN_DATASET, -i IN_DATASET
                        Dataset you want to tokenize. Local path or HF repo id
  --feature FEATURE, -f FEATURE
                        Name of the feature (column) containing text documents in the input dataset
  --split SPLIT, -s SPLIT
                        Split of the dataset to be tokenized, supports slicing like 'train[:10%]'
  --tokenizer TOKENIZER, -t TOKENIZER
                        HF repo id or local directory containing the tokenizer
  --seq-len SEQ_LEN, -l SEQ_LEN
                        Length of the tokenized sequences
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        How many text documents to tokenize at once (default: 50)
  --chunk-size CHUNK_SIZE, -c CHUNK_SIZE
                        Maximum number of tokenized sequences in a single parquet file (default: 200_000)
  --out-dir OUT_DIR     Local directory to save the resulting dataset
  --out-repo OUT_REPO   HF repo id to upload the resulting dataset
```

Here's how we tokenized the dataset for our `stories-*` suite of models. Please note that you can use single letter abbreviations for most arguments.

For `train` split:
```
> scripts/tokenize_dataset.py \
    --in-dataset delphi-suite/stories \
    --feature story \
    --split train \
    --tokenizer delphi-suite/stories-tokenizer \
    --seq-len 512 \
    --out-repo delphi-suite/stories-tokenized
```
For `validation` split, repeated arguments omitted:
```
> scripts/tokenize_dataset.py \
    ...
    --split validation \
    ...
```

The input dataset is the same as in tokenizer training example above. We tokenize it with our custom [delphi-suite/stories-tokenizer](https://huggingface.co/delphi-suite/stories-tokenizer) into sequences of length 512. We upload it to HF dataset repo [delphi-suite/stories-tokenized](https://huggingface.co/datasets/delphi-suite/stories-tokenized).

Please note that you can use any HuggingFace tokenizer, you don't need to train a custom one.

# Training a model

To train a model, you'll need to create a config file. For examples see `configs/`, and for field descriptions see `delphi/train/config/training_config.py`. The training script is located in `scripts/train_model.py`.

Script usage:

```
> scripts/train_model.py --help
usage: train_model.py [-h] [--overrides [OVERRIDES ...]] [-v | -s] [config_files ...]

Train a delphi model

positional arguments:
  config_files          Path to json file(s) containing config values, e.g. 'primary_config.json secondary_config.json'.

options:
  -h, --help            show this help message and exit
  --overrides [OVERRIDES ...]
                        Override config values with space-separated declarations. e.g. `--overrides model_config.hidden_size=42 run_name=foo`
  -v, --verbose         Increase verbosity level, repeatable (e.g. -vvv). Mutually exclusive with --silent, --loglevel
  -s, --silent          Silence all logging. Mutually exclusive with --verbose, --loglevel
```

You can specify primary config and secondary config, which is useful if you're training a suite of models that only differ in a few parameters. Additionally, you can override specific fields using the `--overrides` flag. If you don't want to push the model and its checkpoints to HF, you need to explicitly set `out_repo=""`. If you don't want to log to W&B, you need to set `wandb=""`. Please note that by default we save the optimizer state (2x model size) with every checkpoint.

Here is how we trained our `stories-mamba-100k` model
```
> scripts/train_model.py \
    configs/stories/mamba/base.json \
    configs/stories/mamba/100k.json \
    --overrides \
      out_repo="delphi-suite/stories-mamba-100k" \
      wandb="delphi-suite/delphi"
```

# Development

1. Install the `dev` and `notebooks` dependencies `pip install -e ."[dev,notebooks]"`.
2. Run the tests `pytest`.
3. Install pre-commit `pre-commit install`.
4. Install the recommended vscode extensions.

When you save a file vscode should automatically format it. Otherwise, pre-commit will do that, but you will need to add the changes and commit again.

# Citation

If you use delphi in your research, please cite using the following

```bibtex
@software{delphi,
  title = {delphi: small language models training made easy},
  author = {Jett Janiak, Jai Dhyani, Jannik Brinkmann, Gonçalo Paulo, Joshua Wendland, Víctor Abia Alonso, Siwei Li, Phan Anh Duong, Alice Rigg},
  year = 2024,
  url = {https://github.com/delphi-suite/delphi},
  license = {apache-2.0}
}
```