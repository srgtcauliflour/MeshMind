"""Run MM-E001 from prebuilt token shards."""

import argparse

from meshmind.model import MeshMindConfig
from meshmind.pretrain import PretrainConfig, pretrain
from meshmind.shards import read_token_shard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output-dir", default="runs/mm-e001")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    train_tokens = read_token_shard(args.train)
    valid_tokens = read_token_shard(args.valid)
    cfg = PretrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    model_cfg = MeshMindConfig(context_length=args.sequence_length)
    model = pretrain(train_tokens, valid_tokens, model_cfg, cfg, args.output_dir, args.resume)
    print(f"completed {args.steps:,} steps with {model.parameter_count():,} parameters")


if __name__ == "__main__":
    main()
