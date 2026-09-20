"""Prepare verified MM-E001 data without materialising the full corpus in RAM."""

import argparse
import json
from pathlib import Path

from meshmind.fast_tokenizer import train_byte_bpe
from meshmind.provenance import sha256_file, verify_artifacts
from meshmind.shards import StreamingTokenShardWriter


def encode_lines(tokenizer, source, writer):
    byte_count = character_count = token_count = 0
    with Path(source).open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            ids = tokenizer.encode(line).ids
            writer.write(ids)
            byte_count += len(line.encode("utf-8"))
            character_count += len(line)
            token_count += len(ids)
    return {
        "bytes": byte_count,
        "characters": character_count,
        "tokens": token_count,
        "bytes_per_token": byte_count / max(1, token_count),
        "characters_per_token": character_count / max(1, token_count),
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
    train_path, valid_path = root / "train.txt", root / "valid.txt"

    tokenizer = train_byte_bpe([train_path], args.vocab_size)
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    tokenizer_sha = sha256_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    with StreamingTokenShardWriter(output / "train.bin", vocab_size, tokenizer_sha) as writer:
        train_stats = encode_lines(tokenizer, train_path, writer)
    with StreamingTokenShardWriter(output / "valid.bin", vocab_size, tokenizer_sha) as writer:
        valid_stats = encode_lines(tokenizer, valid_path, writer)

    special_tokens = {token: tokenizer.token_to_id(token) for token in ("<bos>", "<eos>", "<pad>")}
    report = {
        "format_version": 2,
        "tokenizer_engine": "tokenizers-bytelevel-bpe",
        "manifest_sha256": sha256_file(args.manifest),
        "tokenizer_sha256": tokenizer_sha,
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "train": train_stats,
        "valid": valid_stats,
    }
    (output / "prep-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
