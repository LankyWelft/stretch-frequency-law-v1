# -*- coding: utf-8 -*-
"""
Experimental gas-phase benchmark for the R1 / R1+B2 / R1+MPNN stretching-frequency
models (paper Section 4.x and SI table for the Spectrochimica Acta Part A version).

The 307 external modes in `examples/si_modes.csv` carry, alongside the CCSD(T)-F12c
harmonic wavenumbers, the experimental gas-phase FUNDAMENTAL wavenumbers (`nu_exp`)
compiled by VIBFREQ1295 from NIST CCCBDB and the primary literature. This script
scores the three models against BOTH references and quantifies the harmonic->fundamental
anharmonic gap, which is the irreducible floor for a model trained on B3LYP harmonics.

Bond-type normalisation, the pre-registered audit rules and relMAE are imported from
audit_external_labels.py so the numbers share one exact accounting path.

Run:  python external_validation/experimental_gasphase_benchmark.py
"""
import os
import numpy as np
import pandas as pd

from audit_external_labels import (
    btype_chem, flagged, relmae, SUPERVISED, UNSUPERVISED,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "examples", "si_modes.csv")

MODELS = [("nu_R1", "R1 raw"), ("nu_R1+B2", "R1+B2"), ("nu_MPNN", "R1+MPNN")]
# per-bond-type order used in the paper external table
PER_TYPE = ["C-H", "N-H", "O-H", "C=O", "C=C", "C#C", "C#N", "C=N",
            "C-C", "C-O", "C-N", "C-F", "N-F", "O-F", "N=O"]


def anharmonic_gap(df):
    """relMAE of CCSD(T)-F12c HARMONIC wavenumbers vs experimental FUNDAMENTALS:
    the theory-level floor that no harmonic-trained model can cross."""
    return relmae(df["nu_CCSD(T)"], df["nu_exp"])


def main():
    df = pd.read_csv(CSV)
    assert len(df) == 307
    df["bt_chem"] = df["Mode"].map(btype_chem)
    df["rule_flag"] = df.apply(flagged, axis=1)
    aud = df[~df["rule_flag"]]

    print("=" * 86)
    print("Aggregate relMAE (%)   vs experimental gas-phase fundamentals  |  vs CCSD(T)-F12c harmonic")
    print("=" * 86)
    for col, name in MODELS:
        e307 = relmae(df[col], df["nu_exp"]);   c307 = relmae(df[col], df["nu_CCSD(T)"])
        e294 = relmae(aud[col], aud["nu_exp"]); c294 = relmae(aud[col], aud["nu_CCSD(T)"])
        print(f"  {name:9s} as-released(307)  {e307:5.2f} exp   {c307:5.2f} CCSD")
        print(f"  {'':9s} audited   (294)  {e294:5.2f} exp   {c294:5.2f} CCSD")
    print(f"\n  harmonic->fundamental gap (CCSD(T) harmonic vs exp fundamental): "
          f"{anharmonic_gap(df):.2f}% (307) / {anharmonic_gap(aud):.2f}% (294)")

    print("\n" + "=" * 86)
    print("Per-bond-type relMAE (%, audited set)   n | R1+B2 exp | MPNN exp | MPNN CCSD")
    print("=" * 86)
    for bt in PER_TYPE:
        a = aud[aud.bt_chem == bt]
        if len(a) == 0:
            continue
        print(f"  {bt:5s} n={len(a):3d}   {relmae(a['nu_R1+B2'], a['nu_exp']):5.2f}   "
              f"{relmae(a['nu_MPNN'], a['nu_exp']):5.2f}   {relmae(a['nu_MPNN'], a['nu_CCSD(T)']):5.2f}")

    print("\n" + "=" * 86)
    print("Subset relMAE (%) on audited modes vs experimental fundamentals")
    print("=" * 86)
    # subset by the released-label token via btype_chem mapping on supervised list
    sup = aud[aud.bt_chem.isin(set(SUPERVISED) | {"C-H"})]
    unsup = aud[aud.bt_chem.isin(set(UNSUPERVISED))]
    xh = aud[aud.bt_chem.isin(["C-H", "N-H", "O-H"])]
    for nm, sub in [("supervised stretch types", sup),
                    ("X-H stretches", xh),
                    ("unsupervised single/halogen", unsup)]:
        line = f"  {nm:30s} n={len(sub):3d}"
        for col, _ in MODELS:
            line += f"   {relmae(sub[col], sub['nu_exp']):5.2f}"
        print(line)

    print("\nColumns above per model: R1 raw / R1+B2 / R1+MPNN (vs exp).")


if __name__ == "__main__":
    main()
