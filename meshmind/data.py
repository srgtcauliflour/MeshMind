"""Dataset manifests and deterministic token-block utilities."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class DataSource:
    name: str
    uri: str
    license: str
    sha256: str
    notes: str = ""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: str | Path, sources: list[DataSource]) -> None:
    payload = {"version": 1, "sources": [source.__dict__ for source in sources]}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> list[DataSource]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("unsupported dataset manifest version")
    return [DataSource(**source) for source in payload["sources"]]


def make_causal_blocks(token_ids: list[int], context_length: int):
    """Yield input/target blocks for next-token prediction."""
    if context_length < 1:
        raise ValueError("context_length must be positive")
    width = context_length + 1
    for start in range(0, len(token_ids) - context_length, context_length):
        chunk = token_ids[start : start + width]
        if len(chunk) == width:
            yield chunk[:-1], chunk[1:]
