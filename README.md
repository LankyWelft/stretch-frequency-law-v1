# A Closed-Form Three-Feature Law for Molecular Stretching Frequencies

Lightweight, **O(N)** prediction of molecular **stretching** vibrational frequencies,
together with a small Δ-learning correction layer. This repository accompanies the
manuscript

> *"A Closed-Form Three-Feature Law for Molecular Stretching Frequencies with a
> Lightweight Δ-Learning Correction Layer"* — Yue Lin, *J. Chem. Theory Comput.*
> (submitted).

The analytic layer needs no quantum-chemical calculation, coordinate optimization,
or training data at inference. The optional Δ-layer learns only the small, systematic
environmental residual.

**Scope.** This work models **stretching modes only**. Bending, torsion, mode
coupling, energies, forces, and gradients are explicitly out of scope; heavy-atom
single bonds carry no reliable per-bond labels and are reported as a boundary.

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
experimental frequencies and then frozen.

## Quick start (no data download, seconds)

```bash
pip install -r requirements.txt

# (0) analytic R1 on representative bonds — NumPy only
python demo_r1.py

# (1) external VIBFREQ1295 validation, R1 + bond-type table (B2)
python external_validation/V7-R14_ext_vibfreq.py

# (2) external VIBFREQ1295 validation, R1 + locked MPNN Δ-layer (no retraining)
python external_validation/run_mpnn_external.py
```

Command (2) builds molecular graphs from SMILES with RDKit (no 3-D coordinates, no
QM9-derived files), loads the locked weights, and self-checks against the shipped
per-mode table `examples/si_modes.csv`. Expected output: 74 molecules / 307 modes,
self-check mean |Δ| ≈ 0.02%, overall relMAE **11.82%** vs CCSD(T)-F12c, with
**C=O 1.75%**, **O–H 1.76%**, C–H 12.96%. The large overall number is honest, not a
failure: the external set contains heavy-atom single bonds the MPNN was never
supervised on, and symmetry-split C–H modes; the strongly-bonded, assignable modes
transfer tightly. Command (1) gives overall **12.99%**, C=O **1.53%**.

## Full QM9 reproduction (Tables 2–5, Fig. 7)

```bash
# 1. download + unpack QM9 (~82 MB) into qm9_data/
bash scripts/download_qm9.sh

# 2. build the per-bond augmented records  -> qm9_data/records_aug.npz
python scripts/build_records_aug.py

# 3. analyses / models (each is standalone)
python src/r14a_error_map.py     # Table 2: R1 raw error map, learnability (84.1%)
python src/r14b_linear_delta.py  # Table 3: constant / bond-type / ridge Δ-models
python src/r14c_mlp.py           # Table 3: edge-MLP Δ-model
python src/r14c2_mpnn.py         # Tables 4–5: 3-layer edge-conditioned MPNN
python src/r14d_active.py        # Fig. 7 : active-learning strategy simulation
```

To point the pipeline at an existing QM9 copy elsewhere, set the data root:

```bash
export V7_QM9_BASE=/path/to/dir_holding_dsgdb9nsd
```

The small shipped files `data/bad_qm9.txt` (unusable molecule ids) and
`data/match_prior.json` (bond-type matching priors) are always read from the repo.
The derived `records_aug.npz` (~94 MB) and raw geometries are regenerated, not
committed. A full MPNN retrain writes `c2_mpnn_weights_retrained.pt` to the data
root and never overwrites the locked `weights/c2_mpnn_weights.pt`.

### Reproduced numbers (B3LYP/6-31G(2df,p), 130,831 molecules, 717,839 assignable modes)

| stage | relMAE |
|---|---|
| R1 raw | 4.30% |
| B2 bond-type table | 1.98% |
| B3 ridge (full environment) | 1.75% |
| edge MLP | 1.55% |
| MPNN, X–H | **1.35%** |
| MPNN, heavy-atom multiple bonds | **≈1.9%** |
| learnable fraction of R1 residual variance | 84.1% |
| size extrapolation (train ≤6 heavy, test 7–9), ridge / MLP / MPNN | 1.84% / 5.49% / see Table 5 |

## Repository layout

```
demo_r1.py                              # minimal NumPy-only R1 demonstration
requirements.txt
src/v7_qm9_lib.py                       # constants, R1 law, portable paths, QM9 I/O
src/r14a_error_map.py                   # R1 error map + learnability criterion
src/r14b_linear_delta.py                # ridge Δ-learning baseline
src/r14c_mlp.py                         # edge-MLP Δ-model
src/r14c2_mpnn.py                       # hand-written edge-conditioned MPNN
src/r14d_active.py                      # active-learning strategy simulation
scripts/download_qm9.sh                 # fetch + unpack QM9
scripts/build_records_aug.py            # QM9 .xyz -> records_aug.npz
data/bad_qm9.txt, data/match_prior.json # shipped small support files
weights/c2_mpnn_weights.pt              # locked MPNN weights (~280 KB)
external_validation/V7-R14_ext_vibfreq.py   # R1 + B2 on VIBFREQ1295
external_validation/run_mpnn_external.py    # R1 + MPNN on VIBFREQ1295 (self-contained)
external_validation/b2_bondtype_factors.json# frozen B2 multiplicative factors
external_validation/VIBFREQ1295_Data.csv    # external benchmark subset (C/H/O/N/F)
examples/si_modes.csv, si_molecules.csv     # per-mode / per-molecule SI tables
```

## Requirements

`numpy, scipy, networkx, torch, rdkit, scikit-learn` (see `requirements.txt`).
`rdkit` is the PyPI package (`pip install rdkit`). The external MPNN inference runs
on CPU in a fraction of a second for 74 molecules; set `OMP_NUM_THREADS=1` if needed.

## Data sources

- **QM9**, B3LYP/6-31G(2df,p), ~130k CHONF structures with harmonic frequencies:
  Ramakrishnan et al., *Sci. Data* **1**, 140022 (2014); Figshare file 3195389.
- **VIBFREQ1295**, CCSD(T)-F12c harmonic / experimental fundamentals:
  Harvard Dataverse, DOI 10.7910/DVN/VLVNU7.

Datasets remain under their original licenses; the MIT license below covers the code.

## Author

Yue Lin (林岳) — School of Mathematics and Computer Science, Northwest Minzu
University, Lanzhou, China. Sole and corresponding author.
Email: woneng_1216@qq.com

## License

Code: MIT (see `LICENSE`). Bundled datasets remain under their original licenses.
