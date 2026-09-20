# A Closed-Form Three-Feature Law for Molecular Stretching Frequencies

Lightweight, O(N) prediction of molecular stretching vibrational frequencies,
together with a small Δ-learning correction layer. This repository accompanies the
manuscript *"A Closed-Form Three-Feature Law for Molecular Stretching Frequencies
with a Lightweight Δ-Learning Correction Layer"* (Y. Lin, submitted to JCTC).

## The closed-form law (R1)

For a bond between atoms of reduced mass μ, effective bond length R, and bond order BO,

```
ν̃ = 2962 · μ^(−0.527) · R^(−0.856) · BO^(0.680)   [cm⁻¹]
```

with `R = √(R_i · R_j)` and the locked spectral radial lengths

| element | H | C | N | O | F |
|---|---|---|---|---|---|
| R_i (Å) | 0.9166 | 1.119 | 1.073 | 1.034 | 1.002 |

The four parameters are fitted once from 30 representative bond types against
experimental frequencies and then frozen; no quantum-chemical calculation,
coordinate optimization, or training data is used at inference for the analytic
layer.

## Quick start (no data download, runs in < 1 second)

```bash
python demo_r1.py
```

This prints predicted stretching wavenumbers for representative C–H, O–H,
C–C (1/2/3), C–O and C=O bonds. Requires only Python + NumPy.

## Repository layout

```
demo_r1.py                     # minimal, dependency-light R1 demonstration
src/v7_qm9_lib.py              # shared constants, R1 formula, QM9 I/O
src/r14c2_mpnn.py              # hand-written edge-conditioned MPNN Δ-learning layer
src/r14b_linear_delta.py       # linear ridge Δ-learning baseline
weights/c2_mpnn_weights.pt     # locked MPNN weights (276 KB)
external_validation/          # independent VIBFREQ1295 (CCSD(T)-F12c) validation
examples/                      # example per-mode tables (SI excerpts)
```

## Requirements

```
numpy scipy networkx torch
```

Install: `pip install -r requirements.txt`

## Data

The large derived arrays (`records_aug.npz`, QM9 geometries) are **not** shipped
here because of their size; they are regenerated from the public sources below:

- **QM9** (B3LYP/6-31G(2df,p), 130k CHONF structures + harmonic frequencies):
  Ramakrishnan et al., *Sci. Data* **2014**, 1, 140022.
- **VIBFREQ1295** (CCSD(T)-F12c harmonic / experimental fundamentals):
  Harvard Dataverse, DOI 10.7910/DVN/VLVNU7.

Update `BASE` at the top of `src/v7_qm9_lib.py` to point to your local QM9
directory before running the full pipeline.

## Reproduced headline numbers (from the manuscript)

| model | X–H relMAE | heavy-atom multiple bonds |
|---|---|---|
| R1 raw | ~3.5% | ~5–10% |
| R1 + linear ridge | 1.75% | – |
| R1 + MPNN Δ-layer | **1.35%** | **1.90%** |
| External (VIBFREQ1295, CCSD(T)-F12c, no retraining) | C=O 1.75%, O–H 1.76% | – |

Inference is O(N): R1 evaluates ≈ 6.9×10⁷ modes per second on a single CPU core.

## Author

Yue Lin — School of Mathematics and Computer Science, Northwest Minzu University, Lanzhou, China.
Corresponding email: [insert email after acceptance].

## License

Code: MIT (to be confirmed). Datasets remain under their original licenses.
