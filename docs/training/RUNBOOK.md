# MM-E001 Runbook

## 1. Acquire and pin data

Download the selected TinyStories splits with `scripts/acquire_tinystories.py`. Record each printed
SHA-256 in the provenance manifest before treating a run as an MM-E001 result.

## 2. Build the tokenizer and shards

Train the MeshMind byte-level BPE and create portable uint16 shards:

`python scripts/build_token_shards.py --train data/raw/tinystories/train.txt --valid data/raw/tinystories/valid.txt`

Archive `tokenizer.json` and the shard metadata beside every experiment.

## 3. Pretrain

`python scripts/run_pretrain.py --train data/processed/mm-e001/train.bin --valid data/processed/mm-e001/valid.bin --steps 1000 --device auto`

The runner records JSONL metrics and checkpoints. Resume interrupted work with `--resume`.

## 4. Promotion gate

Do not call a checkpoint MM-E001 successful until the exact dataset hashes, tokenizer artifact,
model configuration, seed, token count, validation loss, perplexity, elapsed time and peak memory
are recorded. The first real run is evidence collection, not a quality claim.
