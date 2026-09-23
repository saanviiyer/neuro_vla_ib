# Experiment registry

Use one row per completed run. Never overwrite failed runs.

| run ID | commit | seed | model | language encoder | beta/rate | task | shift | status | artifact |
|---|---|---:|---|---|---:|---|---|---|---|
| pilot-007-gib | pending | 7 | Gaussian IB | instruction table | 0.0005 | 2D navigation | clean + noise | running | `results/pilot` |
| pilot-007-ring | pending | 7 | Modular ring IB | instruction table | 0.0005 | 2D navigation | clean + noise | running | `results/pilot` |

## Primary pilot comparison

Compare relative degradation in observation/reward distortion at corruption 0.12. The pilot is a systems check; no inferential claim is allowed from one seed or one rate.

