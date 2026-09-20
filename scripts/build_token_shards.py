"""Train/load a MeshMind BPE tokenizer and build uint16 corpus shards."""

import argparse
from pathlib import Path

from meshmind.bpe import BPETokenizer, train_bpe
from meshmind.shards import write_token_shard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output-dir", default="data/processed/mm-e001")
    parser.add_argument("--tokenizer")
    parser.add_argument("--vocab-size", type=int, default=16_384)
    args = parser.parse_args()

    train_text = Path(args.train).read_text(encoding="utf-8")
    valid_text = Path(args.valid).read_text(encoding="utf-8")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.tokenizer:
        tokenizer = BPETokenizer.load(args.tokenizer)
    else:
        tokenizer = train_bpe([train_text], args.vocab_size)
        tokenizer.save(output / "tokenizer.json")

    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)
    write_token_shard(output / "train.bin", train_ids, tokenizer.vocab_size)
    write_token_shard(output / "valid.bin", valid_ids, tokenizer.vocab_size)
    print(f"train={len(train_ids):,} tokens valid={len(valid_ids):,} tokens vocab={tokenizer.vocab_size:,}")


if __name__ == "__main__":
    main()
