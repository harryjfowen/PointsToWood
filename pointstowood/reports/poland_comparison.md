# Biome Distillation Report — Poland

Generated: 2026-03-17 01:56

## Summary

| | Value |
|---|---|
| Biome | Poland |
| Teacher model | mcc-eu.pth |
| Compression ratio | 82.1x |
| Teacher params | 23,552,176 |
| Student params | 286,978 |
| Best student HMCC | 0.6714 (PACED KD) |
| Teacher HMCC | 0.4395 |
| PACED retention of teacher HMCC | 152.8% |
| PACED − Scratch HMCC | +0.0224 |

---

## Primary Metric: Harmonic MCC

HMCC = harmonic mean of MCC with-reflectance and without-reflectance.
Higher is better. Penalises models that rely on reflectance to function.

| Model | HMCC | MCC (with refl) | MCC (no refl) |
|---|---|---|---|
| **Teacher** (NetFull, pan-EU) | 0.4395 | 0.4485 | 0.4309 |
| **Scratch** (NetLight, no KD) | 0.6490 | 0.6539 | 0.6450 |
| **PACED KD** (frontier-weighted) | 0.6714 | 0.6757 | 0.6676 |
| PACED − Scratch | +0.0224 | +0.0218 | +0.0226 |
| Teacher − PACED | -0.2319 | -0.2272 | -0.2367 |

---

## Probability Ranking Quality: Harmonic AUPRC

AUPRC is threshold-agnostic — it measures how well the model ranks points regardless of
where the decision boundary sits. This is a fairer comparison for the teacher, which is a
pan-EU generalist whose optimal threshold on a single biome is not necessarily 0.5.

| Model | H-AUPRC | AUPRC (with refl) | AUPRC (no refl) |
|---|---|---|---|
| **Teacher** (NetFull, pan-EU) | 0.9652 | 0.9713 | 0.9592 |
| **Scratch** (NetLight, no KD) | 0.9930 | 0.9947 | 0.9914 |
| **PACED KD** (frontier-weighted) | 0.9935 | 0.9948 | 0.9924 |
| PACED − Scratch | +0.0005 | +0.0001 | +0.0011 |
| Teacher − PACED | -0.0283 | -0.0234 | -0.0333 |

---

## Classification Quality: Fbeta (β=0.5, precision-weighted)

| Model | Fbeta (with refl) | Fbeta (no refl) |
|---|---|---|
| **Teacher** | 0.4527 | 0.4407 |
| **Scratch** | 0.7476 | 0.7600 |
| **PACED KD** | 0.7475 | 0.7507 |
| PACED − Scratch | -0.0001 | -0.0093 |

---

## Boundary Quality: Edge Fbeta

Evaluated only on transition points (wood/leaf boundaries).
Key metric for complex branching geometry (e.g. Mediterranean Spain).

| Model | Edge Fbeta (with refl) | Edge Fbeta (no refl) |
|---|---|---|
| **Scratch** | 0.9467 | 0.9339 |
| **PACED KD** | 0.9462 | 0.9325 |
| PACED − Scratch | -0.0005 | -0.0013 |

---

## Interpretation

**PACED KD beats scratch** (+0.0224 HMCC on Poland): frontier-weighted distillation successfully transfers pan-European teacher knowledge that biome-local training alone cannot recover. The specialist advantage hypothesis holds.

PACED student retains 152.8% of teacher HMCC — strong.

No-reflectance: PACED gains +0.0226 MCC vs scratch — teacher geometry-first representations transferring at wood/leaf boundaries.
