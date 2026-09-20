"""Portable little-endian binary token shards for MeshMind training."""

import json
import struct
from pathlib import Path


class StreamingTokenShardWriter:
    def __init__(self, path, vocab_size, tokenizer_sha256=None):
        if vocab_size > 65_536:
            raise ValueError("uint16 shards support at most 65,536 token IDs")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size
        self.tokenizer_sha256 = tokenizer_sha256
        self.tokens = 0
        self._handle = self.path.open("wb")

    def write(self, token_ids):
        ids = list(token_ids)
        if any(token < 0 or token >= self.vocab_size for token in ids):
            raise ValueError("token ID outside vocabulary")
        if ids:
            self._handle.write(struct.pack(f"<{len(ids)}H", *ids))
            self.tokens += len(ids)

    def close(self):
        if self._handle.closed:
            return
        self._handle.close()
        meta = {
            "format_version": 2,
            "dtype": "uint16-le",
            "tokens": self.tokens,
            "vocab_size": self.vocab_size,
            "tokenizer_sha256": self.tokenizer_sha256,
        }
        self.path.with_suffix(self.path.suffix + ".json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._handle.close() if exc_type else self.close()


def write_token_shard(path, token_ids, vocab_size, tokenizer_sha256=None):
    with StreamingTokenShardWriter(path, vocab_size, tokenizer_sha256) as writer:
        writer.write(token_ids)


def read_token_shard(path):
    raw = Path(path).read_bytes()
    if len(raw) % 2:
        raise ValueError("invalid uint16 shard length")
    return list(struct.unpack(f"<{len(raw) // 2}H", raw))
