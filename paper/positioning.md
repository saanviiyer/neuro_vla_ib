# Positioning and related work

## One-sentence contribution

We test whether modular continuous-attractor channels improve the rate–distortion–robustness frontier of language-conditioned robot world models, and use conditional information and causal perturbations to determine which latent subspaces transmit semantic goals versus physical state.

## What is new

The contribution should not be “grid cells in an RL agent.” That claim is crowded and easy to overstate. The intended novelty is the combination of:

- an explicit sequential rate–distortion formulation for a VLA/world-model latent channel;
- modular neural-code priors evaluated at matched information rate;
- separation of semantic, physical, and goal-relative information;
- causal manifold perturbations tied to downstream RL utility;
- robustness and calibration rather than tuning-curve resemblance alone.

## Nearest work and separation

1. **Schaeffer et al., 2023, Self-Supervised Learning of Representations for Space Generates Multi-Modular Grid Cells.** Establishes conditions under which multi-modular grid representations can arise from self-supervision. Our work asks whether analogous structure improves the rate–distortion frontier of language-conditioned embodied prediction and RL.
2. **Schaeffer et al., 2023, Disentangling Fact from Grid Cell Fiction in Trained Deep Path Integrators.** Shows that successful path integration does not itself establish emergent grid cells. We adopt this warning by separating decoding, topology, dynamics, and causal utility.
3. **Khona and Fiete, 2022, Attractor and Integrator Networks in the Brain.** Supplies the dynamical-systems motivation. The final structured model must instantiate attractor dynamics, not merely normalize latent vectors.
4. **Huh et al., 2024, The Platonic Representation Hypothesis.** Motivates cross-model geometry, but representational convergence is secondary here. Any alignment analysis must use scale-matched null calibration.
5. **Khan et al., 2025, Controlling Vision–Language–Action Policies through Sparse Latent Directions.** Demonstrates that sparse latent directions can steer robot actions and also exposes feature entanglement. Our proposed interventions target world-model dynamics and information allocation rather than only action logits.
6. **VLA-MBPO, 2026, Towards Practical World Model-based Reinforcement Learning for Vision-Language-Action Models.** Establishes a contemporary world-model RL setting for VLA fine-tuning. Our work focuses on the information geometry and robustness of the learned state channel.
7. **Kotar et al., 2025, World Modeling with Probabilistic Structure Integration** (arXiv:2509.09737). Extracts structure by causal inference and mixes it back into the training diet as new token types. We turn that procedure into a measurable rate claim (H5) rather than treating it as engineering.
8. **Aw et al., 2026, Zero-shot World Models Are Developmentally Efficient Learners** (arXiv:2604.10333). Obtains structured, controllable representations with no variational bottleneck at all, using asymmetric masking in time. This is the sharpest falsifier of the bottleneck framing here and must be run as a baseline arm, not cited politely. See `zwm_structure_integration.md`.
9. **World Critic Model, 2026.** Joint future-latent and value prediction supports the premise that critics need temporally predictive state representations. Our paper asks how those representations should allocate limited information.

## Editorial fit test

The paper fits “Information Theory and Large Language Models” only if all three conditions hold:

1. a genuine pretrained LLM or VLA representation is experimentally necessary;
2. an information-theoretic quantity or bound is central to the algorithm and primary result;
3. the contribution generalizes beyond the toy simulator.

If any condition remains unmet by 20 September, do not force the paper into this special issue. Use the review/perspective fallback described in `manuscript_outline.md` or target a robotics/neuro-AI venue instead.

