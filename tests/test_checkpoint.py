import torch

from meshmind.checkpoint import append_metric, load_checkpoint, save_checkpoint
from meshmind.model import MeshMindConfig, MeshMindLM
from meshmind.train import TrainConfig


def test_checkpoint_round_trip(tmp_path):
    cfg = MeshMindConfig(vocab_size=32, context_length=8, n_layers=1, d_model=16, n_heads=2, ffn_hidden=32)
    model = MeshMindLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 7, cfg, TrainConfig())
    original = {name: value.clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.data.zero_()
    payload = load_checkpoint(path, model, optimizer)
    assert payload["step"] == 7
    assert all(torch.equal(model.state_dict()[name], value) for name, value in original.items())


def test_metrics_are_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_metric(path, step=1, loss=3.5)
    assert path.read_text().strip() == '{"loss": 3.5, "step": 1}'
