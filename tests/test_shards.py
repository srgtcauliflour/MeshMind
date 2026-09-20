from meshmind.shards import read_token_shard, write_token_shard


def test_token_shard_round_trip(tmp_path):
    path = tmp_path / "tokens.bin"
    tokens = [0, 1, 255, 256, 16_383]
    write_token_shard(path, tokens, 16_384)
    assert read_token_shard(path) == tokens
    assert '"tokens": 5' in (tmp_path / "tokens.bin.json").read_text()
