"""Reproducible token-stream pretraining for MM-E001."""

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import append_metric, load_checkpoint, save_checkpoint
from .model import MeshMindConfig, MeshMindLM
from .train import seed_everything


@dataclass
class PretrainConfig:
    steps: int = 100
    batch_size: int = 4
    sequence_length: int = 128
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 10
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337
    checkpoint_every: int = 50
    eval_every: int = 25


def learning_rate(step, cfg):
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)


def sample_batch(tokens, cfg, generator):
    if len(tokens) <= cfg.sequence_length:
        raise ValueError("token stream must be longer than sequence_length")
    max_start = len(tokens) - cfg.sequence_length - 1
    starts = torch.randint(0, max_start + 1, (cfg.batch_size,), generator=generator)
    rows = [tokens[i : i + cfg.sequence_length + 1] for i in starts.tolist()]
    batch = torch.tensor(rows, dtype=torch.long)
    return batch[:, :-1], batch[:, 1:]


def evaluate(model, tokens, cfg, generator, batches=4):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = sample_batch(tokens, cfg, generator)
            _, loss = model(inputs, targets)
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def pretrain(train_tokens, valid_tokens, model_cfg=None, cfg=None, output_dir="runs/mm-e001", resume=None):
    cfg = cfg or PretrainConfig()
    model_cfg = model_cfg or MeshMindConfig(context_length=cfg.sequence_length)
    seed_everything(cfg.seed)
    random.seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    model = MeshMindLM(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    start_step = 0
    if resume:
        payload = load_checkpoint(resume, model, optimizer)
        start_step = payload["step"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = output / "metrics.jsonl"
    started = time.perf_counter()

    for step in range(start_step, cfg.steps):
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = sample_batch(train_tokens, cfg, generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        completed = step + 1
        append_metric(metrics, step=completed, train_loss=float(loss), lr=lr, grad_norm=float(grad_norm))

        if cfg.eval_every and completed % cfg.eval_every == 0:
            val_loss = evaluate(model, valid_tokens, cfg, generator)
            append_metric(metrics, step=completed, valid_loss=val_loss, perplexity=math.exp(min(20, val_loss)))
        if cfg.checkpoint_every and completed % cfg.checkpoint_every == 0:
            save_checkpoint(output / f"step-{completed:06d}.pt", model, optimizer, completed, model_cfg, cfg)

    append_metric(metrics, step=cfg.steps, elapsed_seconds=time.perf_counter() - started)
    return model
