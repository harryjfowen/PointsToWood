# Biome Distillation Report — Spain

Generated: 2026-03-17 00:02

## Summary

| | Value |
|---|---|
| Biome | Spain |
| Teacher model | mcc-eu.pth |
| Compression ratio | 82.1x |
| Teacher params | 23,552,176 |
| Student params | 286,978 |
| Best student HMCC | 0.6116 (PACED KD) |
| Teacher HMCC | 0.2561 |
| PACED retention of teacher HMCC | 238.8% |
| PACED − Scratch HMCC | +0.0010 |

---

## Primary Metric: Harmonic MCC

HMCC = harmonic mean of MCC with-reflectance and without-reflectance.
Higher is better. Penalises models that rely on reflectance to function.

| Model | HMCC | MCC (with refl) | MCC (no refl) |
|---|---|---|---|
| **Teacher** (NetFull, pan-EU) | 0.2561 | 0.2990 | 0.2240 |
| **Scratch** (NetLight, no KD) | 0.6106 | 0.6400 | 0.5873 |
| **PACED KD** (frontier-weighted) | 0.6116 | 0.6389 | 0.5900 |
| PACED − Scratch | +0.0010 | -0.0012 | +0.0027 |
| Teacher − PACED | -0.3554 | -0.3399 | -0.3659 |

---

## Probability Ranking Quality: Harmonic AUPRC

AUPRC is threshold-agnostic — it measures how well the model ranks points regardless of
where the decision boundary sits. This is a fairer comparison for the teacher, which is a
pan-EU generalist whose optimal threshold on a single biome is not necessarily 0.5.

| Model | H-AUPRC | AUPRC (with refl) | AUPRC (no refl) |
|---|---|---|---|
| **Teacher** (NetFull, pan-EU) | 0.7840 | 0.8846 | 0.7040 |
| **Scratch** (NetLight, no KD) | 0.9725 | 0.9843 | 0.9609 |
| **PACED KD** (frontier-weighted) | 0.9694 | 0.9797 | 0.9612 |
| PACED − Scratch | -0.0031 | -0.0046 | +0.0003 |
| Teacher − PACED | -0.1854 | -0.0951 | -0.2572 |

---

## Classification Quality: Fbeta (β=0.5, precision-weighted)

| Model | Fbeta (with refl) | Fbeta (no refl) |
|---|---|---|
| **Teacher** | 0.3616 | 0.3365 |
| **Scratch** | 0.7648 | 0.7664 |
| **PACED KD** | 0.7581 | 0.7469 |
| PACED − Scratch | -0.0067 | -0.0195 |

---

## Boundary Quality: Edge Fbeta

Evaluated only on transition points (wood/leaf boundaries).
Key metric for complex branching geometry (e.g. Mediterranean Spain).

| Model | Edge Fbeta (with refl) | Edge Fbeta (no refl) |
|---|---|---|
| **Scratch** | 0.8926 | 0.8143 |
| **PACED KD** | 0.8822 | 0.8162 |
| PACED − Scratch | -0.0104 | +0.0019 |

---

## Interpretation

**PACED KD matches scratch** (ΔHMCC = +0.0010 on Spain): distillation neither helps nor hurts at this compression. Local biome data is sufficient for this student capacity.

PACED student retains 238.8% of teacher HMCC — strong.

No-reflectance: PACED matches scratch (+0.0027 MCC) — geometry-only performance preserved at 82x compression.
