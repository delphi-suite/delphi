import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime

import torch


@dataclass
def TrainingConfig(config):
    # -----------------------------------------------------------------------------
    # I/O
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 100
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = (
        False  # if True, always save a checkpoint after each eval
    )
    init_from: bool = "scratch"  # 'scratch' or 'resume'
    # wandb logging
    wandb_log: bool = False  # disabled by default
    wandb_project: str = "llamac"
    wandb_run_name: str = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # data
    batch_size: int = (
        128  # if gradient_accumulation_steps > 1, this is the micro-batch size
    )
    max_seq_len: int = 256
    vocab_source: str = (
        "llama2"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
    )
    vocab_size: str = 32000  # the Llama 2 tokenizer has 32K tokens
    # model
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6
    multiple_of: int = 32
    dropout: int = 0.0
    # adamw optimizer
    gradient_accumulation_steps: int = 4  # used to simulate larger batch sizes
    learning_rate: float = 5e-4  # max learning rate
    max_iters: int = 100000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 1000  # how many steps to warm up for
    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16"  # float32|bfloat16|float16
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    exec(open("configurator.py").read())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging

    # -----------------------------------------------------------------------------

    # fixing some hyperparams to sensible defaults
    lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
    min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # validating checks
    assert vocab_source in ["llama2", "custom"]
    assert (
        vocab_source == "custom" or vocab_size == 32000
    ), "The vocab from Meta has 32K tokens"

    # various inits, derived attributes, I/O setup
    seed = 1337
    os.makedirs(out_dir, exist_ok=True)


def model_initialization(config):
    # model
    if config["model"] == "llama2":
        from delphi.models.llama2 import LLaMA2, LLaMA2Args

        model_args = dict(
            dim=config["dim"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            n_kv_heads=config["n_kv_heads"],
            vocab_size=config["vocab_size"],
            multiple_of=config["multiple_of"],
            max_seq_len=config["max_seq_len"],
            dropout=config["dropout"],
        )
        gptconf = LLaMA2Args(**model_args)
        model = LLaMA2(gptconf)
    elif config["model"] == "mamba":
        from delphi.models.mamba import Mamba, MambaArgs

        model_args = dict(
            dim=config["dim"],
            n_layers=config["n_layers"],
            vocab_size=config["vocab_size"],
        )
        mambaconf = MambaArgs(**model_args)
        model = Mamba(mambaconf)

    if config["init_from"] == "resume":
        print(f"Resuming training from {config['out_dir']}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(config["out_dir"], "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=config["device"])
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
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        config.iter_num = checkpoint["iter_num"]
        config.best_val_loss = checkpoint["best_val_loss"]

    model.to(config["device"])
    # compile the model
    if config["compile"]:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0
    return model, model_args


def train_loop(model, TrainConf):
    torch.manual_seed(TrainConf.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in TrainConf.device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[TrainConf.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(TrainConf.dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        TrainConf.weight_decay,
        TrainConf.learning_rate,
        (TrainConf.beta1, TrainConf.beta2),
        device_type,
    )
    if TrainConf.init_from == "resume" and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    if TrainConf.wandb_log:
        import wandb

        wandb.init(
            project=TrainConf.wandb_project,
            name=TrainConf.wandb_run_name,
            config=TrainConf.config,
        )

    train_batch_iter = TrainConf.iter_batches(split="train")
    X, Y = next(train_batch_iter)  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model  # unwrap DDP container if needed
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = (
            get_lr(iter_num, TrainConf)
            if TrainConf.decay_lr
            else TrainConf.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % TrainConf.eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if TrainConf.wandb_log:
                try:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "tokens": iter_num * TrainConf.tokens_per_iter,
                            "loss/train": losses["train"],
                            "loss/val": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        },
                        step=iter_num,
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")
            if losses["val"] < best_val_loss or TrainConf.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                    model_export(
                        raw_model, os.path.join(out_dir, "model.bin"), version=0
                    )
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
                loss = loss / gradient_accumulation_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = next(train_batch_iter)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
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
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, TrainConf):
    # 1) linear warmup for warmup_iters steps
    if it < TrainConf.warmup_iters:
        return TrainConf.learning_rate * it / TrainConf.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > TrainConf.lr_decay_iters:
        return TrainConf.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - TrainConf.warmup_iters) / (
        TrainConf.lr_decay_iters - TrainConf.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return TrainConf.min_lr + coeff * (TrainConf.learning_rate - TrainConf.min_lr)
