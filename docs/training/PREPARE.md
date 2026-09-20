# Preparing the first MM-E001 run

The `MM-E001 Prepare` workflow is intentionally manual. It downloads the exact requested train and
validation artifacts, computes their SHA-256 digests, constructs a pinned manifest, verifies those
bytes, trains MeshMind's 16,384-token byte-level BPE, creates uint16 token shards, and uploads the
result as a GitHub Actions artifact.

The preparation artifact contains the pinned manifest, tokenizer, tokenizer hash, token shards and a
compression/statistics report. Corpus text itself is not uploaded as part of the preparation bundle.

A preparation artifact is an input to training, not a model result. Review its report before spending
GPU compute.
