from meshmind.model import MeshMindConfig
from meshmind.pretrain import PretrainConfig, learning_rate, pretrain


def test_learning_rate_warms_and_decays():
    cfg = PretrainConfig(steps=10, warmup_steps=2, learning_rate=1e-3, min_learning_rate=1e-4)
    assert learning_rate(1, cfg) == cfg.learning_rate
    assert learning_rate(9, cfg) < learning_rate(2, cfg)


def test_real_token_stream_pretraining_writes_metrics(tmp_path):
    train = [(i % 31) + 1 for i in range(800)]
    valid = [(i % 31) + 1 for i in range(300)]
    cfg = PretrainConfig(steps=4, batch_size=2, sequence_length=8, warmup_steps=1, eval_every=2, checkpoint_every=2)
    model_cfg = MeshMindConfig(vocab_size=64, context_length=8, n_layers=1, d_model=32, n_heads=4, ffn_hidden=64)
    pretrain(train, valid, model_cfg, cfg, tmp_path)
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "step-000004.pt").exists()
