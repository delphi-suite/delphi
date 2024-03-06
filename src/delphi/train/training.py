"""

"""

import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from llama2c import model_export
from torch.utils.data import DataLoader
from tqdm import tqdm

from delphi import constants
from delphi.eval.utils import load_delphi_dataset
from delphi.train.gigaconfig import jai_config as config
from delphi.train.tokenized_chunks_dataset import TokenizedChunksDataset
from delphi.train.utils import (
    estimate_loss,
    get_device,
    get_lr,
    get_optimizer,
    initialize_model,
    resume_model,
)

# system
device = get_device()
dtype = "float32"  # float32|bfloat16|float16

# -----------------------------------------------------------------------------

# load data
train_docs_ds = load_delphi_dataset(constants.TOKENIZED_CORPUS_DATASET, "train").select(
    range(256)  # TODO: remove when done debugging
)
validation_docs_ds = load_delphi_dataset(
    constants.TOKENIZED_CORPUS_DATASET, "validation"
)

train_ds = TokenizedChunksDataset(train_docs_ds, config.max_seq_len, device)
validation_ds = TokenizedChunksDataset(validation_docs_ds, config.max_seq_len, device)

train_ds.initialize_samples()
validation_ds.initialize_samples()

# fixing some hyperparams to sensible defaults

num_batches = len(train_ds) // config.batch_size
print(f"num_batches: {num_batches}")
num_steps = num_batches // config.gradient_accumulation_steps
config.eval_iters = min(12, len(validation_ds) // config.batch_size)

lr_decay_iters = (
    config.max_epochs * num_batches
)  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# validating checks
assert config.vocab_source in ["llama2", "custom"]
assert (
    config.vocab_source == "custom" or config.vocab_size == 32000
), "The vocab from Meta has 32K tokens"

# various inits, derived attributes, I/O setup
seed = 1337
tokens_per_iter = (
    config.gradient_accumulation_steps * config.batch_size * config.max_seq_len
)
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(
    f"breaks down as: {config.gradient_accumulation_steps} grad accum steps * {config.batch_size} batch size * {config.max_seq_len} max seq len"
)

os.makedirs(config.out_dir, exist_ok=True)
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

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=config.dim,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    n_kv_heads=config.n_kv_heads,
    vocab_size=config.vocab_size,
    multiple_of=config.multiple_of,
    max_seq_len=config.max_seq_len,
    dropout=config.dropout,
)  # start with model_args from command line
if config.init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = initialize_model(**model_args)
    checkpoint = None
elif config.init_from == "resume":
    print(f"Resuming training from {config.out_dir}")
    model_mid_train = resume_model(Path(config.out_dir), device, **model_args)
    model = model_mid_train.model
    iter_num = model_mid_train.iter_num
    best_val_loss = model_mid_train.best_val_loss
    checkpoint = model_mid_train.checkpoint
model.to(device)


# optimizer
optimizer = get_optimizer(
    model=model,
    weight_decay=config.weight_decay,
    learning_rate=config.learning_rate,
    beta_1=config.beta1,
    beta_2=config.beta2,
    device_type=device_type,
    checkpoint=checkpoint
    if checkpoint is not None and "optimizer" in checkpoint
    else None,
)
checkpoint = None  # free up memory

# wrap model into DDP container


# logging
if config.wandb_log:
    import wandb

    wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),
    )


# training loop
t0 = time.time()


local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
epoch = 0
for epoch in range(config.max_epochs):
    train_ds.shuffle(epoch)
    train_batch_iter = iter(DataLoader(train_ds, batch_size=config.batch_size))  # type: ignore
    # get the first batch
    X, Y = next(train_batch_iter)

    for _ in tqdm(range(num_steps)):
        # determine and set the learning rate for this iteration
        lr = (
            get_lr(
                iter_num,
                config.warmup_iters,
                config.learning_rate,
                lr_decay_iters,
                min_lr,
            )
            if config.decay_lr
            else config.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(
                model=model,
                eval_iters=config.eval_iters,
                batch_size=config.batch_size,
                split_to_ds={"train": train_ds, "val": validation_ds},
            )
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if config.wandb_log:
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
            if losses["val"] < best_val_loss or config.always_save_checkpoint:
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
                    print(f"saving checkpoint to {config.out_dir}")
                    torch.save(checkpoint, os.path.join(config.out_dir, "ckpt.pt"))
                    model_export(
                        model, os.path.join(config.out_dir, "model.bin"), version=0
                    )
        if iter_num == 0 and config.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(
            min(config.gradient_accumulation_steps, num_steps - iter_num)
        ):
            logits = model(X, Y)
            loss = model.last_loss / config.gradient_accumulation_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = next(train_batch_iter)
            loss.backward()
        # clip the gradient
        if config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)  # type: ignore
        optimizer.step()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config.log_interval == 0:
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            lossf = loss.item() * config.gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = model.estimate_mfu(
                    config.batch_size * config.gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1
