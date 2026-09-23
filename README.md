# Neuro-VLA Information Bottlenecks

A research scaffold for studying whether neuro-inspired ring/grid codes provide a better rate–distortion tradeoff in language-conditioned robotic world models.

## Submission question

**At a fixed information rate, do modular continuous-attractor latents preserve task-relevant physical information and improve corrupted-rollout performance more than unstructured variational latents?**

This framing makes three elements indispensable rather than decorative:

1. **Information theory:** the latent channel is evaluated through a variational rate bound, predictive distortion, semantic/physical information probes, and robustness curves.
2. **LLM/VLA relevance:** language instructions condition the recurrent world model. The pilot uses a learned instruction table; the full study replaces it with frozen embeddings from an LLM/VLA backbone and tests paraphrase and compositional generalization.
3. **Neuro-inspired robotics:** the structured model allocates paired latent dimensions to circular modules, motivated by head-direction and grid-cell continuous attractors.

## Pilot

The included simulator generates noisy egocentric landmark observations for a differential-drive agent following paraphrased language goals. It compares:

- `gaussian_ib`: an ordinary variational recurrent world model;
- `modular_ring_ib`: a parameter-matched model with four normalized circular latent modules.

Both predict the next observation and reward. Analysis reports the rate upper bound in nats, predictive distortion, heading/state decodability, goal accuracy, effective latent dimension, and degradation under joint observation/latent corruption.

```bash
PYTHONPATH=src python3 -m neuro_vla_ib.train --config configs/pilot.yaml
python3 -m pytest -q
```

Pilot outputs are written to `results/pilot/`. They are engineering validation, not paper evidence. The paper requires multiple seeds, rate sweeps, real pretrained language embeddings, stronger world-model baselines, topology analysis, and robot benchmark experiments.

## Planned full study

| Axis | Values |
|---|---|
| Latent | Gaussian IB, vector-quantized, modular rings, learned attractor |
| Rate | beta sweep plus dimension-matched controls |
| Language | vocabulary embedding, frozen LLM, VLA backbone layer sweep |
| Tasks | navigation, object-goal navigation, language-conditioned manipulation |
| Shift | paraphrases, novel compositions, sensor corruption, longer horizon, new layouts |
| Causal test | phase lesion, module scrambling, matched random subspace lesion |

Primary endpoint: area under the distortion–rate curve under corruption. Secondary endpoints: RL return/sample efficiency, topology, calibration, and semantic–physical information separation.

