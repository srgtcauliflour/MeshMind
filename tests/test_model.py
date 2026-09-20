import torch

from meshmind import MeshMindConfig, MeshMindLM


def tiny_config():
    return MeshMindConfig(vocab_size=256, context_length=32, n_layers=2, d_model=64, n_heads=4, ffn_hidden=128)


def test_forward_shape():
    model = MeshMindLM(tiny_config())
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and torch.isfinite(loss)


def test_backward():
    model = MeshMindLM(tiny_config())
    x = torch.randint(0, 256, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    assert model.embedding.weight.grad is not None


def test_weight_tying():
    model = MeshMindLM(tiny_config())
    assert model.embedding.weight is model.lm_head.weight


def test_context_guard():
    model = MeshMindLM(tiny_config())
    x = torch.randint(0, 256, (1, 33))
    try:
        model(x)
    except ValueError:
        return
    raise AssertionError("expected context length guard")
