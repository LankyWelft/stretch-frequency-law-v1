# An Analytic Three-Feature Scaling Law for Molecular Stretching Vibrational Frequencies

[![Reproducibility](https://github.com/LankyWelft/stretch-frequency-law-v1/actions/workflows/ci.yml/badge.svg)](https://github.com/LankyWelft/stretch-frequency-law-v1/actions/workflows/ci.yml)

Lightweight, **O(N)** prediction of molecular **stretching** vibrational frequencies,
together with a small Δ-learning correction layer. This repository accompanies the
manuscript

> *"An Analytic Three-Feature Scaling Law with a Lightweight Δ-Learning Correction for
> Molecular Stretching Vibrational Frequencies: Validation against CCSD(T) and Gas-Phase
> Experiment"* — Yue Lin, *Spectrochimica Acta Part A* (submitted).

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

# (3) pre-registered external label audit (paper SI S10); read-only, no model
python external_validation/audit_external_labels.py

# (4) experimental gas-phase FUNDAMENTALS benchmark (paper Sec. 4.7 / SI S11)
python external_validation/experimental_gasphase_benchmark.py
```

Commands (1)–(2) build molecular graphs from SMILES with RDKit (no 3-D coordinates,
no QM9-derived files), load the locked factors/weights, and self-check against the
shipped per-mode table `examples/si_modes.csv`. Expected results vs CCSD(T)-F12c over
all 307 released modes (74 molecules):

| model | all 307 (as released) | audited 294 |
|---|---|---|
| R1 raw | 13.56% | 8.34% |
| R1 + B2 | **11.88%** (C=O 1.53%, O–H 1.75%, N–H 4.35%) | 6.43% |
| R1 + MPNN | **11.81%** (C=O 1.75%, O–H 1.76%) | **6.31%** |

Command (3) re-derives, from the released label and CCSD frequency alone (two
model-independent physical-range rules, fixed before inspecting any prediction), the
**13 mis-labeled modes** shipped inside VIBFREQ1295 itself (e.g. a 1518 cm⁻¹ CH₃
deformation labeled "C–H stretch"; high-frequency C–H stretches labeled "C=C
stretch"). It reproduces every SI S10 number: after excluding those 13 rows the
audited aggregate is 6.31%, C–H falls 12.96%→**2.38%** and C=C 9.74%→**3.89%**, while
every supervised single-mode type is unchanged. The removable log-ratio variance
rises from 10.2% to **46.0%** overall and from 42.9% to **82.1%** on the supervised
types, matching the QM9 value of 84.1%. All 307 rows are retained in
`examples/si_modes.csv` (flagged in the `audit_flag` column); no unfavorable mode is
deleted. The remaining audited error is structural (heavy-atom single bonds the MPNN
was never supervised on, plus a small XHₙ symmetry-splitting floor of ~1.7%), not a
failure of transfer.

### (4) Comparison with experimental gas-phase fundamentals

Command (4) scores the same frozen models against the measured gas-phase **fundamental**
wavenumbers (`nu_exp` in `examples/si_modes.csv`, compiled by VIBFREQ1295 from NIST
CCCBDB and the primary literature), reusing the exact bond-type normalization, audit
rules and relMAE of command (3). Aggregate relMAE on the 294 audited modes:

| reference | R1 raw | R1+B2 | R1+MPNN |
|---|---|---|---|
| experimental fundamentals | 7.54% | **7.10%** | 7.39% |
| CCSD(T)-F12c harmonics | 8.34% | 6.43% | **6.31%** |

The CCSD(T) harmonics themselves lie **4.04%** above the measured fundamentals on the
same modes (the harmonic-to-fundamental anharmonic gap), which is the floor for any
harmonic-trained model; on supervised types R1+MPNN reaches 5.14% against experiment
(4.61% for X–H; C=O 3.51%, C–H 3.87%). The complete per-bond-type matrix (all bond
types, as-released 307 and audited 294) is printed by the script and tabulated in SI S11.

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

To use a QM9 copy you already have instead of downloading, point the data root
at a directory containing either the extracted `*.xyz` files directly or a
`dsgdb9nsd/` folder of them (both layouts, and a nested `dsgdb9nsd/dsgdb9nsd/`,
are auto-detected):

```bash
export V7_QM9_BASE=/path/to/qm9     # contains *.xyz  or  dsgdb9nsd/*.xyz
python scripts/build_records_aug.py
```

The small shipped files `data/bad_qm9.txt` (unusable molecule ids) and
`data/match_prior.json` (bond-type matching priors) are always read from the repo.
The derived `records_aug.npz` (~94 MB) and raw geometries are regenerated, not
committed. A full MPNN retrain writes `c2_mpnn_weights_retrained.pt` to the data
root and never overwrites the locked `weights/c2_mpnn_weights.pt`.

### Reproduced numbers (B3LYP/6-31G(2df,p), 130,815 molecules, 1,435,692 assigned modes)

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
external_validation/audit_external_labels.py # pre-registered label audit (SI S10), read-only
external_validation/experimental_gasphase_benchmark.py # NIST/CCCBDB fundamentals benchmark (Sec. 4.7/SI S11)
external_validation/b2_bondtype_factors.json# frozen B2 multiplicative factors
external_validation/VIBFREQ1295_Data.csv    # external benchmark subset (C/H/O/N/F)
examples/si_modes.csv, si_molecules.csv     # per-mode (307, audit_flag column) / per-molecule SI tables
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
