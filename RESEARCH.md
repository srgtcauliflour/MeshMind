# Research Program

## Core hypothesis

A useful language model can be trained so that communication cost, topology constraints, unreliable nodes, and constrained local memory are properties of the learned system rather than deployment problems added after training.

## Research rules

- Distribution is native.
- Communication is expensive.
- Failure is expected.
- Local computation is preferred over network transfer when quality permits.
- Experiments change one major variable at a time.
- Failed experiments are retained and documented.
- Claims require measurements.
- Simulation precedes large physical deployments.

## Initial experiments

### MM-E001 — Baseline learning
Train MM-0A from random initialization and establish reproducible loss/perplexity baselines.

### MM-E002 — Communication accounting
Insert virtual node boundaries and measure bytes/messages crossing them.

### MM-E003 — Representation bottleneck
Sweep boundary dimensions and measure quality versus communication.

### MM-E004 — Learned codec
Train encoder/decoder boundary codecs jointly with the model.

### MM-E005 — Communication-aware loss
Add a tunable communication penalty and map the Pareto frontier.

### MM-E006 — Topology dropout
Train under simulated node/link failures and measure graceful degradation.
