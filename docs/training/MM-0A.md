# MM-0A Training Plan

MM-0A begins from random weights. The objective is next-token prediction and the first milestone
is reproducible learning, not useful assistant behaviour.

## Gates

1. CPU synthetic smoke training decreases loss.
2. Dataset inputs are provenance-tracked.
3. Tokenizer round-trips Unicode text.
4. A small real-text corpus can be tokenized into causal blocks.
5. Training checkpoints and metrics are reproducible from a fixed seed.
6. Only after these gates pass do we schedule an external GPU run.

## Tokenizer strategy

The byte tokenizer in the bootstrap code is an intentionally lossless control tokenizer. It
lets us validate the entire data/training path without making an irreversible vocabulary choice.

The research tokenizer target remains 16,384 tokens. It will be trained from the selected corpus
and benchmarked against the byte baseline for compression ratio, fertility, multilingual/code
coverage, memory footprint, and downstream model quality.

## Data policy

MeshMind records provenance before training. Corpus selection will favor sources with clear
licensing/usage terms and reproducible acquisition. Dataset quality, deduplication, filtering,
PII handling, contamination checks, and mixture weights are separate measurable stages.
