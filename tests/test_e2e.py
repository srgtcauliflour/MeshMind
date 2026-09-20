from meshmind.bpe import train_bpe
from meshmind.model import MeshMindConfig
from meshmind.pretrain import PretrainConfig, pretrain
from meshmind.shards import read_token_shard, write_token_shard


def test_text_to_token_shards_to_checkpoint(tmp_path):
    train_text = "mesh nodes learn together. " * 40
    valid_text = "mesh nodes learn. " * 20
    tokenizer = train_bpe([train_text], vocab_size=280)
    train_path = tmp_path / "train.bin"
    valid_path = tmp_path / "valid.bin"
    write_token_shard(train_path, tokenizer.encode(train_text), tokenizer.vocab_size)
    write_token_shard(valid_path, tokenizer.encode(valid_text), tokenizer.vocab_size)
    train_ids = read_token_shard(train_path)
    valid_ids = read_token_shard(valid_path)

    cfg = PretrainConfig(steps=3, batch_size=2, sequence_length=8, warmup_steps=1, eval_every=1, checkpoint_every=3)
    model_cfg = MeshMindConfig(vocab_size=tokenizer.vocab_size, context_length=8, n_layers=1, d_model=32, n_heads=4, ffn_hidden=64)
    pretrain(train_ids, valid_ids, model_cfg, cfg, tmp_path / "run")
    assert (tmp_path / "run" / "step-000003.pt").exists()
