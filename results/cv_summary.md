# CV Summary — ENSO-BSISO/MJO Self-Supervised Learning

## Self-Supervised Representation Learning for ENSO Modulation of Tropical Intraseasonal Oscillations
*Data-Driven Climate Science*

Built an end-to-end deep-learning pipeline (PyTorch, ERA5 reanalysis 1979–2023) to learn how ENSO modulates the BSISO and MJO beyond classical composite analysis.

- **Reproduced the Wheeler–Hendon RMM MJO index** from daily-mean ERA5 via combined EOF analysis and validated it against the official Bureau of Meteorology index (Pearson r ≈ 0.95; canonical correlation 0.97/0.95; uniform across ENSO states), using a rotation-aware (Procrustes + canonical-correlation) comparison for the near-degenerate EOF pair.
- **Estimated the intrinsic dimensionality** of the oscillation state manifolds with a Neural State Variables approach (lag-prediction encoder–decoder + intrinsic-dimension estimation): BSISO ≈ 4-D, MJO ≈ 7-D, recovering ENSO modulation with permutation z-scores up to 20.9 (vs 12.2 supervised).
- **Diagnosed and fixed a training collapse** in a 2-D supervised contrastive encoder (root cause: InfoNCE temperature; quantified via embedding effective-rank/eigenvalue metrics), and via a dimension sweep independently corroborated that BSISO phase is ~2-D and the full state ~4-D.
- **Designed and tested a novel temporally-graded Barlow Twins objective**, showing that invariance-based SSL recovers the slow ENSO/amplitude envelope (ENSO modulation effectively ~2-D, robust across representations) while contrastive/prediction methods capture the fast MJO phase cycle — establishing a complementarity between self-supervised learning families.

*Tools: PyTorch, scikit-learn, xarray, Copernicus CDS/ERA5, Google Colab.*

---

### One-line version
Applied self-supervised learning (contrastive, Neural State Variables, and a novel temporally-graded Barlow Twins) to ERA5 reanalysis to quantify ENSO's modulation of the MJO/BSISO; reproduced the operational RMM index (r ≈ 0.95) and showed the ENSO modulation is effectively 2-dimensional.
