"""Prepare a verified MM-E001 corpus, tokenizer, shards, and experiment report."""

import argparse
import json
from pathlib import Path

from meshmind.bpe import train_bpe
from meshmind.provenance import sha256_file, verify_artifacts
from meshmind.shards import write_token_shard


def stats(text, ids):
    raw = text.encode("utf-8")
    return {
        "bytes": len(raw),
        "characters": len(text),
        "tokens": len(ids),
        "bytes_per_token": len(raw) / max(1, len(ids)),
        "characters_per_token": len(text) / max(1, len(ids)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="runs/mm-e001-prep")
    parser.add_argument("--vocab-size", type=int, default=16_384)
    args = parser.parse_args()

    verify_artifacts(args.manifest, args.root)
    root, output = Path(args.root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_text = (root / "train.txt").read_text(encoding="utf-8")
    valid_text = (root / "valid.txt").read_text(encoding="utf-8")

    tokenizer = train_bpe([train_text], args.vocab_size)
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    train_ids, valid_ids = tokenizer.encode(train_text), tokenizer.encode(valid_text)
    write_token_shard(output / "train.bin", train_ids, tokenizer.vocab_size)
    write_token_shard(output / "valid.bin", valid_ids, tokenizer.vocab_size)

    report = {
        "format_version": 1,
        "manifest_sha256": sha256_file(args.manifest),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "train": stats(train_text, train_ids),
        "valid": stats(valid_text, valid_ids),
    }
    (output / "prep-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
