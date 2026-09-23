# Structure integration as a rate statement

The Stanford NeuroAI world-model line published two results in 2025 and 2026
that bear directly on this manuscript, and they push it in opposite directions.
One of them is the sharpest available falsifier of the whole information-
bottleneck framing. The other is a new primary hypothesis that only this project
is equipped to test. Both should be in the paper before it goes to *Entropy*.

## References, with identifiers checked

- Kotar, Lee, Venkatesh, Chen, Bear, Watrous, Kim, Aw, Chen, Stojanov, Feigelis,
  Thobani, Durango, Jedoui, Kazemian, Yamins, *World Modeling with Probabilistic
  Structure Integration*, 2025, arXiv:2509.09737. Three-step cycle: probabilistic
  prediction, structure extraction by causal inference, then integration of the
  extracted structure back into the training diet as new token types. Trained on
  1.4T tokens of video; yields state-of-the-art flow, self-supervised depth and
  segmentation.
- Aw, Kotar, Lee, Kim, Jedoui, Venkatesh, Chen, Frank, Yamins, *Zero-shot World
  Models Are Developmentally Efficient Learners*, 2026, arXiv:2604.10333. A
  two-frame predictor sees frame one in full and roughly ten percent of frame
  two. Symmetric masking at matched budget is substantially worse. Competence on
  flow, depth, segmentation and intuitive physics with no task training, from
  132 hours of one child's egocentric video.
- Venkatesh et al., *Physical Object Understanding with a Physically Controllable
  World Model*, CVPR 2026 Highlight, arXiv:2606.00439.
- Bear, Feigelis et al., *Unifying (Machine) Vision via Counterfactual World
  Modeling*, 2023, arXiv:2306.01828.
- *Discovering and using Spelke segments*, 2025, arXiv:2507.16038.
- *Unifying Vision, Language, and World Modeling via Streams of Thought*,
  CCN 2026. Not verified; do not cite until the proceedings entry exists.

## The falsifier, and why it should be run first

ZWM obtains structured, controllable, physically meaningful representations with
no variational bottleneck, no KL term, and no explicit rate constraint. The
structure comes from *where the prediction problem is made hard*: an asymmetric
mask in time. If a masked predictor with no KL matches the modular-ring model on
the corruption-conditioned distortion curve at matched capacity, then the rate
term in this manuscript is decorative and the reviewers will say so.

**Model 6, required before submission: masked temporally-factored predictor.**
Same recurrent core, same instruction conditioning, no variational latent, no
KL. Observations withheld in contiguous blocks during training, with a small
retained fraction per masked step following ZWM's ten percent. Rate is measured
post hoc on its latent by the same estimator used for the others, so it can be
placed on the same axes even though nothing constrained it during training.

This is not a courtesy baseline. It is the arm that decides whether the paper
has a result. Two outcomes, both publishable:

- The masked predictor sits on the same frontier. Then the honest claim is that
  the bottleneck's *location* is what matters, on the observation stream rather
  than on the state, and the manuscript becomes a comparison of where to impose
  a rate constraint. This is a better paper than the current outline and a
  better fit to the special issue than another modular-code result.
- The modular model dominates at matched post hoc rate. Then H1 survives a test
  it was not built to survive, and the paper says so explicitly.

## H5, new primary hypothesis: structure integration buys bits

PSI extracts low-dimensional structure by causal inference and mixes it back in
as new tokens. That procedure is stated as engineering, but it is an information
claim in disguise: if the extracted structure is genuine, predicting the next
observation given the structure token should require fewer bits through the
latent channel than predicting it without.

Nobody has measured that, and this project already has the instrument.

**H5.** At matched predictive distortion, reintegrating a zero-shot-extracted
structure (heading phase, and optionally flow between consecutive landmark
observations) as an auxiliary input channel reduces the variational rate through
the latent bottleneck by a measurable margin.

    Delta R = R(no structure token) - R(structure token), at matched D.

**Falsified if** the rate difference is within the seed spread at matched
distortion, or if it disappears once the structure channel's own rate is
included in the accounting. That second condition is the trap and must be
pre-registered: a structure token is itself a channel, and moving bits from one
channel to another is not compression. The claim only survives if total
transmitted information falls.

**Controls, in the order a reviewer will ask for them.**

1. Shuffled structure token: same marginal distribution, wrong pairing with the
   observation. Kills the possibility that the model is just using extra
   capacity.
2. Rotated structure token: the phase advanced by a constant offset, so the
   token is as informative in aggregate but wrong at every step.
3. Random-projection token of the same dimension drawn from the encoder state.
4. The gain must survive at matched total rate, not only at matched latent rate.

**Why this project and not the other one.** The robowomo study has the extractor
and no rate instrument; this one has the rate instrument. The extraction step is
already implemented there as `src/zeroshot.py`, which finds the heading plane by
counterfactual action matching with no pose labels and recovers a gain of about
0.93 against a physically correct value of 1 on blackout-trained models. Port
the estimator, do not rebuild it.

## H6: zero-shot factorization replaces the probe for semantic and physical information

The current H2 tests semantic-physical factorization with trained probes and
subspace lesions. Trained probes measure decodability, which is a property of
the probe as much as of the channel, and the manuscript's own citation of
Schaeffer's grid-cell-fiction warning applies to itself here.

Counterfactual prompting gives a probe-free version. Perturb the language
embedding and measure which latent coordinates change the model's own
predictions; perturb the action and do the same. The overlap between the two
response sets is a label-free measure of factorization, computed entirely from
the model's outputs.

**H6.** The counterfactual response sets for language and for self-motion are
more disjoint in the modular model than in the Gaussian model at matched rate,
and the small overlap set carries goal-relative affordance information.

This replaces one of the existing probe analyses rather than adding to the
budget. Report both for one model to show they agree, then use the zero-shot
version everywhere.

## A data-budget axis, cheaply

ZWM's headline is efficiency: competence from one child's experience. The
manuscript sweeps beta and dimension but not data. In a toy simulator a data
sweep is nearly free, and a rate-distortion-data surface is a genuinely
information-theoretic object that the special issue's scope invites.

**H7, secondary.** The modular model's advantage, if any, is largest at small
data budgets and shrinks with data. Falsified if the gap is flat in data budget,
which would itself be worth reporting because it separates an inductive-bias
explanation from a capacity explanation.

## What this does to the editorial fit test

The three conditions in `positioning.md` were: a pretrained LLM or VLA
representation must be experimentally necessary; an information-theoretic
quantity must be central; and the contribution must generalize past the toy
simulator.

H5 moves the second condition from "central to the framing" to "central to the
primary result", because the result *is* a rate difference. It does nothing for
the first or third. Those still gate the submission, and the fallback in
`manuscript_outline.md` still applies. Do not let a stronger information result
disguise a missing language result.

## What must not enter the claim

- No claim that this reproduces PSI. PSI is 1.4T tokens of internet video and a
  full autoregressive stack; this is a toy simulator with a recurrent core.
  The claim is about the mechanism, tested where it can be measured cleanly.
- No brain or developmental claim. There is no neural data in this project.
- No use of the phrase "zero-shot" for anything that was fitted, including the
  structure extractor's own hyperparameters.
- Streams of Thought stays uncited until confirmed.
