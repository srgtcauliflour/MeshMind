# MeshMind Architecture

MeshMind is a research project for mesh-native AI: models designed from the start for distributed inference across resource-constrained and unreliable devices.

## MM-0A baseline

MM-0A is deliberately conventional. It is the scientific control against which mesh-native changes will be measured.

- decoder-only causal language model
- vocabulary: 16,384
- context: 512 tokens
- layers: 8
- hidden width: 512
- attention heads: 8
- head width: 64
- SwiGLU feed-forward network
- RMSNorm
- rotary positional embeddings (RoPE)
- tied token embedding / language-model head weights

## Research sequence

1. MM-0A: dense control
2. MM-0B: explicit virtual communication boundaries
3. MM-0C: bottlenecked boundary representations
4. MM-0D: learned neural communication codec
5. MM-0E: communication-aware objective
6. MM-0F: topology dropout and redundant paths
7. MM-0G: low-bit / ternary experiments
8. physical MCU deployment experiments

Every step must be benchmarked against the previous step so regressions have an identifiable cause.
