# Original five-frame results, native latent display

2026-11-25. ANALYZED from returned reports; no retraining or feature-coordinate recomputation.

Evaluation c45df90165111f0f uses L1a, window5, evaluation stride4,20 held-out trajectories/1500 windows. All four run IDs and feature hashes match the original a70d291efb9ff88c evaluation. Evaluator source hash matches the current local code. Learned z is plotted directly; only the delay-PCA baseline uses PCA. All matched scorecard R2 differences from the original evaluation are at most4.55e-8. This is a new display of existing models, not improved model performance.

| Method | Feature | Angle linear R2 | Velocity linear R2 |
|---|---|---:|---:|
| bt_same | h | 0.982352 | 0.783701 |
| bt_same | z | 0.382112 | 0.061281 |
| bt_temporal | h | 0.973982 | 0.708702 |
| bt_temporal | z | 0.000044 | 0.000014 |
| infonce | h | 0.971534 | 0.574054 |
| infonce | z | 0.005048 | 0.000031 |
| nsv | h | 0.999604 | 0.992266 |
| nsv | z_d2 | 0.000037 | 0.000042 |
| delay_pca | pca | 0.998877 | 0.982117 |

The encoders retain observable state information, especially NSV h64. Five-frame velocity is recoverable, unlike the single-frame observation ambiguity. The displayed 2D heads/refiner outputs perform much worse on these probes. Nonlinear MLP readouts of BT-temporal and NSV z_d2 remain near R2=0; InfoNCE z improves to angle0.278/velocity0.025, BT-same to0.454/0.115. This supports a problem at the low-dimensional representation stage, without proving irreversible information loss or identifying a particular architectural cause. Visible curves/clouds are not sufficient evidence of successful state recovery or constant collapse.

NSV encoder h64 train-only ID medians across k10/20/40 are1.845/1.914/1.963, consistent with a roughly2D state reference at these sampling scales. This does not establish that the fitted z_d2 successfully parameterizes that state. NSV z_d8 nonlinear angle/velocity R2 is0.988/0.976; dimension8 working better does not imply physical ID8. PCA64 baseline linear R2 is0.999/0.982; these scores use all64 components, not the plotted two.

Limits: one training seed; no method superiority significance claim, no checkpoint-level causal diagnosis, and no raw features/checkpoints in the report ZIP. Prioritize auditing h-to-z mappings, refiner reconstruction/normalization and head dimensionality under controlled training, rather than interpreting the new orientation as a performance change. No learning code changed.
