from pathlib import Path

from meshmind.fast_tokenizer import train_byte_bpe
from meshmind.shards import StreamingTokenShardWriter, read_token_shard
from scripts.prepare_mm_e001 import encode_lines


def test_streaming_encode_matches_linewise_tokenization(tmp_path):
    source = tmp_path / "corpus.txt"
    lines = ["Hello MeshMind!\n", "Unicode: 🧠 café\n", "code = x + 1\n"]
    source.write_text("".join(lines), encoding="utf-8")
    tokenizer = train_byte_bpe([source], vocab_size=300)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    shard = tmp_path / "tokens.bin"
    with StreamingTokenShardWriter(shard, tokenizer.get_vocab_size(), "test-hash") as writer:
        stats = encode_lines(tokenizer, source, writer)

    expected = [token for line in lines for token in tokenizer.encode(line).ids]
    assert read_token_shard(shard) == expected
    assert stats["tokens"] == len(expected)
    assert stats["bytes"] == len("".join(lines).encode("utf-8"))
    assert tokenizer.decode(expected) == "".join(lines)


def test_streaming_writer_rejects_out_of_vocab(tmp_path):
    shard = tmp_path / "bad.bin"
    try:
        with StreamingTokenShardWriter(shard, 10) as writer:
            writer.write([10])
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-vocabulary token accepted")
