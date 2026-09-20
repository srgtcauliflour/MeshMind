"""Portable binary token shards for MeshMind training."""

import json
from array import array
from pathlib import Path


def write_token_shard(path, token_ids, vocab_size):
    if vocab_size > 65_536:
        raise ValueError("uint16 shards support at most 65,536 token IDs")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = array("H", token_ids)
    with path.open("wb") as handle:
        payload.tofile(handle)
    meta = {"format_version": 1, "dtype": "uint16", "tokens": len(payload), "vocab_size": vocab_size}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def read_token_shard(path):
    payload = array("H")
    with Path(path).open("rb") as handle:
        payload.fromfile(handle, Path(path).stat().st_size // payload.itemsize)
    return list(payload)
