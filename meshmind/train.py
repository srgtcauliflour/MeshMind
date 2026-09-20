"""Minimal reproducible training loop for MeshMind experiments."""

from dataclasses import dataclass
import random

import torch

from .model import MeshMindConfig, MeshMindLM


@dataclass
class TrainConfig:
    steps: int = 20
    batch_size: int = 4
    sequence_length: int = 32
    learning_rate: float = 3e-4
    seed: int = 1337


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def smoke_train(cfg: TrainConfig | None = None) -> tuple[float, float]:
    """Overfit a deterministic synthetic pattern to prove the training path works."""
    cfg = cfg or TrainConfig()
    seed_everything(cfg.seed)
    model_cfg = MeshMindConfig(
        vocab_size=64,
        context_length=cfg.sequence_length,
        n_layers=2,
        d_model=64,
        n_heads=4,
        ffn_hidden=128,
    )
    model = MeshMindLM(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    first_loss = None
    last_loss = None
    for _ in range(cfg.steps):
        starts = torch.randint(0, model_cfg.vocab_size, (cfg.batch_size, 1))
        offsets = torch.arange(cfg.sequence_length + 1).unsqueeze(0)
        sequence = (starts + offsets) % model_cfg.vocab_size
        inputs, targets = sequence[:, :-1], sequence[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

        value = float(loss.detach())
        first_loss = value if first_loss is None else first_loss
        last_loss = value

    return first_loss, last_loss


if __name__ == "__main__":
    start, end = smoke_train()
    print(f"smoke training loss: {start:.4f} -> {end:.4f}")
