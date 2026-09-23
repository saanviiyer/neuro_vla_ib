# Neuro-VLA information bottlenecks

This is a research scaffold. It asks whether ring and grid codes, of the kind seen in head-direction and grid cells, give a better rate-distortion tradeoff in language-conditioned robot world models.

## Question

At a fixed information rate, do modular continuous-attractor latents keep more task-relevant physical information than unstructured variational latents? Do they also improve performance on corrupted rollouts?

The study has three parts:

1. **Information theory.** The latent channel is scored with a variational rate bound, predictive distortion, probes for semantic and physical information, and curves of performance under corruption.
2. **Language conditioning.** Language instructions condition the recurrent world model. The pilot uses a learned instruction table. The full study will use frozen embeddings from a large language model or a vision-language-action (VLA) backbone, and will test paraphrase and compositional generalization.
3. **Neuro-inspired structure.** The structured model gives pairs of latent dimensions to circular modules. Head-direction and grid-cell continuous attractors motivate this design.

## Pilot

The simulator makes noisy egocentric landmark observations for a differential-drive agent that follows paraphrased language goals. It compares two models:

- `gaussian_ib`: an ordinary variational recurrent world model.
- `modular_ring_ib`: a parameter-matched model with four normalized circular latent modules.

Both models predict the next observation and the reward. The analysis reports the rate upper bound in nats, predictive distortion, heading and state decodability, goal accuracy, effective latent dimension, and the loss in performance under joint observation and latent corruption.

### Pilot result (seed 7)

| Model | Split | Observation MSE | Reward MSE | Rate (nats/step) | State R² | Goal-information lower bound (bits) |
|---|---|---:|---:|---:|---:|---:|
| Gaussian IB | clean | 0.0279 | 0.1331 | 23.73 | 0.590 | 1.803 |
| Modular ring IB | clean | 0.0298 | 0.1383 | 27.21 | 0.572 | 1.572 |
| Gaussian IB | corrupted | 0.0307 | 0.1350 | 23.73 | 0.532 | 1.296 |
| Modular ring IB | corrupted | 0.0327 | 0.1406 | 27.20 | 0.548 | 1.167 |

The circular model does not beat the Gaussian model. It has higher clean distortion and sends more rate. It also keeps less language-goal information. Its state decoding degrades less under corruption. That observation comes from a single seed at unmatched rate, so it cannot support a robustness claim.

The pilot model normalizes four latent pairs after sampling. This gives circular coordinates. It does not create continuous-attractor dynamics or multi-scale path integration, and it is not an information-optimal code. It is a negative control, and the intended algorithm is not built yet. The pilot validates the engineering only and is not paper evidence. `results/pilot/REPORT.md` lists the next comparisons.

## Run

Requires Python 3.11 or later.

    pip install torch numpy scikit-learn pyyaml matplotlib pytest
    PYTHONPATH=src python3 -m neuro_vla_ib.train --config configs/pilot.yaml
    python3 -m pytest -q

The pilot writes to `results/pilot/`. Other scripts:

    PYTHONPATH=src python3 scripts/run_rate_sweep.py --config configs/pilot.yaml   # beta sweep over seeds
    PYTHONPATH=src python3 scripts/evaluate_checkpoints.py --checkpoint-dir results/pilot
    PYTHONPATH=src python3 scripts/summarize_pilot.py results/pilot/pilot_metrics.json
    PYTHONPATH=src python3 scripts/extract_language_embeddings.py --model <hf-model> --output <file.npy>

The last script needs the `transformers` package. It writes frozen instruction embeddings for the `language_embeddings` field of the config.

## Planned full study

| Axis | Values |
|---|---|
| Latent | Gaussian IB, vector-quantized, modular rings, learned attractor |
| Rate | beta sweep plus dimension-matched controls |
| Language | vocabulary embedding, frozen LLM, VLA backbone layer sweep |
| Tasks | navigation, object-goal navigation, language-conditioned manipulation |
| Shift | paraphrases, novel compositions, sensor corruption, longer horizon, new layouts |
| Causal test | phase lesion, module scrambling, matched random subspace lesion |

The primary endpoint is the area under the distortion-rate curve under corruption. Secondary endpoints are RL return and sample efficiency, topology, calibration, and the separation of semantic and physical information.

## Layout

    src/neuro_vla_ib/   simulator, models, training loop and analysis
    scripts/            rate sweep, evaluation, summaries, language embedding extraction
    configs/            pilot and smoke-test configs
    results/            pilot and smoke-sweep metrics (JSON) and the pilot report
    paper/              positioning, experiment registry and notes on related work
    tests/              smoke test

Model checkpoints (`*.pt`) are not in the repository. The training command regenerates them.
