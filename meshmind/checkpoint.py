"""Checkpoint and metrics utilities for reproducible MeshMind experiments."""

import json
from dataclasses import asdict
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, step, model_config, train_config):
    payload = {
        "format_version": 1,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    return payload


def append_metric(path, **metric):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
