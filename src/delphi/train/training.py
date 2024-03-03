"""

"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from typing import cast

import numpy as np
import torch
from datasets import Dataset
from llama2c import Task, model_export
from llama2c.model import ModelArgs as Llama2ModelArgs
from llama2c.model import Transformer as Llama2Model
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from delphi import constants
from delphi.eval.utils import load_delphi_dataset
from delphi.train.shuffle import shuffle_list


class TokenizedDocumentDataset(Dataset):
    def __init__(self, tokenized_docs, max_len):
        self.tokenized_docs = tokenized_docs
        self.doc_len = len(tokenized_docs[0]["tokens"])
        self.max_len = max_len
        self.indices = self._default_indices()
        self._total_tokens = self.doc_len * len(self.tokenized_docs)
        self.batched_tokens = (
            torch.Tensor()
        )  # will be initialized in initialize_samples

    def initialize_samples(self):
        # self.tokenized_docs is an (X, 1) tensor of dicts. Each entry is just {"tokens": [int]}
        # where [int] is doc_len long
        # we want to turn this into a (num_batches, max_len + 1) tensor of ints
        # the +1 is for the last Y token prediction, and implies an overlap of 1 token between batches
        # this is because each batch will be broken into X [:-1] and Y [1:]

        num_tokens = self.doc_len * len(self.tokenized_docs)
        num_batches = num_tokens // self.max_len
        tensor_tokens = torch.stack(
            [
                torch.tensor(doc["tokens"], dtype=torch.int16)
                for doc in self.tokenized_docs
            ]
        )
        self.batched_tokens = tensor_tokens.flatten().unfold(
            0, self.max_len + 1, self.max_len
        )

    def _default_indices(self):
        return list(range(len(self.tokenized_docs)))

    def shuffle(self, epoch: int):
        """this is inefficient, but tinyevals are tiny, so nbd probably"""
        # reset for idempotent determinism
        self.indices = self._default_indices()
        shuffle_list(self.indices, seed=epoch)

    def __len__(self):
        return self._total_tokens // self.max_len

    def __getitem__(self, idx):
        sample = self.batched_tokens[self.indices[idx], :]
        X = sample[:-1]
        Y = sample[1:]
        return torch.tensor(X), torch.tensor(Y)

    def __iter__(self):
        while True:
            # drop the last partial batch (last token doesn't have a Y token to predict)
            for idx in self.indices:
                X, Y = self[idx]
                yield X, Y


# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'
# wandb logging
wandb_log = True  # disabled by default
# wandb_entity = "jannik-brinkmann"
wandb_entity = "jaiwithani"
wandb_project = "delphi"
wandb_run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
batch_size = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256
vocab_source = (
    "llama2"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
)
vocab_size = 32000  # the Llama 2 tokenizer has 32K tokens
# model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
# adamw optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_epochs = 10  # total number of training epochs
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for


# Jai Overrides TODO: remove these
vocab_source = "custom"
vocab_size = 4096
max_seq_len = 512
dim = 48
n_layers = 2
n_heads = 2
n_kv_heads = 2
max_epochs = 2
eval_interval = 500
eval_iters = 10


# system
# TODO: when fixing all of this, also dynamically detect env
# and set the device accordingly (e.g. mps on macbooks, cuda on linux)
device = (
    # "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    "mps"  # TODO: remove, this is for debugging on macbooks
)
dtype = "float32"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(
    open("./llama2c/configurator.py").read()
)  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging


# -----------------------------------------------------------------------------

# fixing some hyperparams to sensible defaults
train_docs_ds = load_delphi_dataset(constants.TOKENIZED_CORPUS_DATASET, "train")
validation_docs_ds = load_delphi_dataset(
    constants.TOKENIZED_CORPUS_DATASET, "validation"
)

train_ds = TokenizedDocumentDataset(train_docs_ds, max_seq_len)
validation_ds = TokenizedDocumentDataset(validation_docs_ds, max_seq_len)

num_batches = len(train_ds) // batch_size
print(f"num_batches: {num_batches}")
num_steps = num_batches // gradient_accumulation_steps
eval_iters = min(12, len(validation_ds) // batch_size)

lr_decay_iters = max_epochs * num_batches  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# validating checks
assert vocab_source in ["llama2", "custom"]
assert (
    vocab_source == "custom" or vocab_size == 32000
), "The vocab from Meta has 32K tokens"

# various inits, derived attributes, I/O setup
seed = 1337
tokens_per_iter = gradient_accumulation_steps * batch_size * max_seq_len
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(
    f"breaks down as: {gradient_accumulation_steps} grad accum steps * {batch_size} batch size * {max_seq_len} max seq len"
)

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]


def batch_gen(
    ds: Dataset,
    batch_size: int,
    device: str,
    epoch: int,
    seed: int,
    num_workers: int = 0,
):
    # use epoch and seed in dataloader
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        yield x, y


# task-specific setup
iter_batches = partial(
    batch_gen,
    batch_size=batch_size,
    device=device,
    num_workers=0,
    seed=seed,
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = Llama2ModelArgs(**model_args)
    model = Llama2Model(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in [
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "vocab_size",
        "multiple_of",
        "max_seq_len",
    ]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = Llama2ModelArgs(**model_args)
    model = Llama2Model(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
model.to(device)


# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# wrap model into DDP container


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(split_to_batch_iter):
    out = {}
    model.eval()
    for split, batch_iter in split_to_batch_iter.items():
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            # forward pass, which will also compute the loss
            _logits = model(X, Y)
            loss = cast(Tensor, model.last_loss)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log:
    import wandb

    wandb.init(
        entity=wandb_entity, project=wandb_project, name=wandb_run_name, config=config
    )


# training loop
t0 = time.time()


local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
epoch = 0
for epoch in range(max_epochs):
    train_docs_ds.shuffle(epoch)
    validation_docs_ds.shuffle(epoch)
    train_batch_iter = iter(DataLoader(train_docs_ds, batch_size=batch_size))  # type: ignore
    val_batch_iter = iter(DataLoader(validation_docs_ds, batch_size=batch_size))  # type: ignore
    # get the first batch
    X, Y = next(train_batch_iter)

    for _ in tqdm(range(num_steps)):
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss({"train": train_batch_iter, "val": val_batch_iter})
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if wandb_log:
                try:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "tokens": iter_num * tokens_per_iter,
                            "loss/train": losses["train"],
                            "loss/val": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        },
                        step=iter_num,
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                    model_export(model, os.path.join(out_dir, "model.bin"), version=0)
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            logits = model(X, Y)
            loss = model.last_loss
            loss = loss / gradient_accumulation_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = next(train_batch_iter)
            # backward pass, with gradient scaling if training in fp16
            loss.backward()
        # clip the gradient
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        optimizer.set()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1
