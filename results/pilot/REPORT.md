# Pilot report, seed 7

This run validates the simulator, training loop, checkpoints, corruption evaluation, and information probes. It is not evidence for a paper claim.

## Result

| Model | Split | Observation MSE | Reward MSE | Rate (nats/step) | State R² | Goal-information lower bound (bits) |
|---|---|---:|---:|---:|---:|---:|
| Gaussian IB | clean | 0.0279 | 0.1331 | 23.73 | 0.590 | 1.803 |
| Modular ring IB | clean | 0.0298 | 0.1383 | 27.21 | 0.572 | 1.572 |
| Gaussian IB | corrupted | 0.0307 | 0.1350 | 23.73 | 0.532 | 1.296 |
| Modular ring IB | corrupted | 0.0327 | 0.1406 | 27.20 | 0.548 | 1.167 |

The imposed circular model does **not** dominate the Gaussian model. It has higher clean distortion, transmits more rate, and retains less language-goal information. Its state-decoding degradation under corruption is smaller, but this single-seed, unmatched-rate observation cannot support a robustness claim.

## Diagnosis

The current structured baseline normalizes four latent pairs after sampling. This creates circular coordinates but does not create continuous-attractor dynamics, multi-scale path integration, or an information-optimal code. It is therefore a useful negative control, not yet the intended algorithm.

## Next decision

Implement and compare:

1. a rate-matched Gaussian model selected from a beta sweep;
2. a structured model with phase-transition dynamics rather than post-hoc normalization;
3. a parameter-matched model with the same auxiliary penalty but randomly rotated subspaces;
4. a vector-quantized bottleneck;
5. frozen LLM embeddings with held-out paraphrases.

The next primary plot must be a rate–distortion frontier over at least five seeds. Do not optimize against this single test set.

