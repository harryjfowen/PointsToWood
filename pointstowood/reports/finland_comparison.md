# Biome Distillation Report — Finland

Generated: 2026-03-17 07:02

## Summary

| | Value |
|---|---|
| Biome | Finland |
| Teacher model | mcc-eu.pth |
| Compression ratio | 82.1x |
| Teacher params | 23,552,176 |
| Student params | 286,978 |
| Best student HMCC | 0.4319 (PACED KD) |
| Teacher HMCC | 0.2539 |
| PACED retention of teacher HMCC | 170.1% |
| PACED − Scratch HMCC | +0.0079 |

---

## Primary Metric: Harmonic MCC

HMCC = harmonic mean of MCC with-reflectance and without-reflectance.
Higher is better. Penalises models that rely on reflectance to function.

| Model | HMCC | MCC (with refl) | MCC (no refl) |
|---|---|---|---|
| **Teacher** (NetFull, pan-EU) | 0.2539 | 0.2883 | 0.2268 |
| **Scratch** (NetLight, no KD) | 0.4240 | 0.4460 | 0.4042 |
| **PACED KD** (frontier-weighted) | 0.4319 | 0.4590 | 0.4087 |
| PACED − Scratch | +0.0079 | +0.0130 | +0.0046 |
| Teacher − PACED | -0.1781 | -0.1707 | -0.1819 |

---

## Probability Ranking Quality: Harmonic AUPRC

AUPRC is threshold-agnostic — it measures how well the model ranks points regardless of
where the decision boundary sits. This is a fairer comparison for the teacher, which is a
pan-EU generalist whose optimal threshold on a single biome is not necessarily 0.5.

| Model | H-AUPRC | AUPRC (with refl) | AUPRC (no refl) |
|---|---|---|---|
| **Teacher** (NetFull, pan-EU) | 0.8058 | 0.9228 | 0.7151 |
| **Scratch** (NetLight, no KD) | 0.9465 | 0.9778 | 0.9175 |
| **PACED KD** (frontier-weighted) | 0.9341 | 0.9719 | 0.8993 |
| PACED − Scratch | -0.0123 | -0.0059 | -0.0183 |
| Teacher − PACED | -0.1284 | -0.0492 | -0.1841 |

---

## Classification Quality: Fbeta (β=0.5, precision-weighted)

| Model | Fbeta (with refl) | Fbeta (no refl) |
|---|---|---|
| **Teacher** | 0.3669 | 0.3583 |
| **Scratch** | 0.6265 | 0.6104 |
| **PACED KD** | 0.6042 | 0.5892 |
| PACED − Scratch | -0.0223 | -0.0213 |

---

## Boundary Quality: Edge Fbeta

Evaluated only on transition points (wood/leaf boundaries).
Key metric for complex branching geometry (e.g. Mediterranean Spain).

| Model | Edge Fbeta (with refl) | Edge Fbeta (no refl) |
|---|---|---|
| **Scratch** | 0.9000 | 0.7489 |
| **PACED KD** | 0.8757 | 0.7143 |
| PACED − Scratch | -0.0243 | -0.0346 |

---

## Interpretation

**PACED KD matches scratch** (ΔHMCC = +0.0079 on Finland): distillation neither helps nor hurts at this compression. Local biome data is sufficient for this student capacity.

PACED student retains 170.1% of teacher HMCC — strong.

No-reflectance: PACED matches scratch (+0.0046 MCC) — geometry-only performance preserved at 82x compression.
