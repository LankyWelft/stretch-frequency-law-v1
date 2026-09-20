# -*- coding: utf-8 -*-
"""
Pre-registered external label audit for VIBFREQ1295 (paper SI S10).

This script is self-contained and read-only with respect to the model. It starts from
the released per-mode table `examples/si_modes.csv` (307 modes) and independently
re-derives the physically inconsistent labels using two model-independent rules that
were fixed before any model prediction was inspected:

  (R1) a mode LABELED as an X-H stretch (C-H / O-H / N-H) whose CCSD(T)-F12c
       wavenumber is below 2500 cm^-1;
  (R2) a mode LABELED as a C=C stretch whose CCSD(T)-F12c wavenumber is above
       2200 cm^-1.

The allene cumulative C=C=C asymmetric stretch (~2018 cm^-1) and the resonance
NO2 N=O stretches (~1680 cm^-1) are physically correct and do not trigger either rule.

It then reports, both as-released (307) and audited (294):
  * R1 / R1+B2 / R1+MPNN relMAE against CCSD(T)-F12c (all modes and per bond type),
  * the removable log-ratio variance (learnability) under bond-type binning.

Run:  python external_validation/audit_external_labels.py
"""
import os
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "examples", "si_modes.csv")

XH_TOKENS = ("C-H", "O-H", "N-H")


def _first_token(mode: str) -> str:
    return [t for t in mode.replace(" stretch", "").split()
            if t.lower() not in ("sym", "asym", "symmetric", "antisymmetric")][0]


def btype_label(mode: str) -> str:
    """Bond-type key equal to the benchmark's released label token (N#C and C=C=C
    kept as their own tokens). Used for the learnability/variance decomposition of
    paper Table 7 and SI S9/S10, so the bins reproduce the released labels exactly."""
    return _first_token(mode)


def btype_chem(mode: str) -> str:
    """Chemically normalized undirected bond type used for per-bond-type relMAE in
    paper Table 6: N#C -> C#N (atom-order independent), C=C=C -> C=C (cumulative)."""
    bt = _first_token(mode)
    return {"N#C": "C#N", "C=C=C": "C=C"}.get(bt, bt)


def flagged(row) -> bool:
    """Pre-registered physical-range rules; uses only the released label and CCSD freq."""
    m = row["Mode"]
    v = float(row["nu_CCSD(T)"])
    if any(t in m for t in XH_TOKENS) and v < 2500.0:
        return True
    if "C=C" in m and v > 2200.0:   # "C\u2261C" does not contain the substring "C=C"
        return True
    return False


def relmae(pred, ref):
    pred = np.asarray(pred, float)
    ref = np.asarray(ref, float)
    return 100.0 * np.mean(np.abs(pred - ref) / ref)


def removable_fraction(df, key="bt_label", minn=5):
    """1 - (within-bond-type residual std / total std)^2, bins with >= minn modes.
    Bins use the released-label token (`bt_label`), matching paper Table 7 / SI S9."""
    e = np.log(df["nu_R1"].to_numpy(float) / df["nu_CCSD(T)"].to_numpy(float))
    bucket = defaultdict(list)
    for k, ek in zip(df[key].to_numpy(), e):
        bucket[k].append(ek)
    resid = np.concatenate([np.array(v) - np.mean(v)
                            for v in bucket.values() if len(v) >= minn])
    return 1.0 - (resid.std() / e.std()) ** 2


# released-label tokens for the single-mode supervised and unsupervised subsets (Table 7)
SUPERVISED = ["O-H", "N-H", "C=O", "C=C", "C#C", "C#N", "C=N", "N=O", "N=N"]
UNSUPERVISED = ["C-C", "C-O", "C-N", "C-F", "N-F", "O-F", "N-N", "O-O", "N-O"]


def main():
    df = pd.read_csv(CSV)
    assert len(df) == 307, f"expected 307 released modes, got {len(df)}"
    df["bt_label"] = df["Mode"].map(btype_label)
    df["bt_chem"] = df["Mode"].map(btype_chem)
    df["rule_flag"] = df.apply(flagged, axis=1)

    # cross-check the independently re-derived flags against the shipped audit_flag column
    if "audit_flag" in df.columns:
        shipped = (df["audit_flag"].fillna("") == "FLAGGED").to_numpy()
        assert bool(np.array_equal(shipped, df["rule_flag"].to_numpy())), \
            "re-derived flags disagree with shipped audit_flag column"

    fl = df[df["rule_flag"]]
    print("=" * 78)
    print(f"Flagged modes: {len(fl)} (rules R1/R2)")
    print("=" * 78)
    for _, r in fl.iterrows():
        print(f"  #{int(r.name)+1:3d} {r['Molecule']:10s} {r['Mode']:22s} "
              f"CCSD={r['nu_CCSD(T)']:7.1f}  exp={r['nu_exp']:7.1f}")

    aud = df[~df["rule_flag"]]
    print("\n" + "=" * 78)
    print("Aggregate relMAE (%) vs CCSD(T)-F12c :  as-released (307) -> audited (294)")
    print("=" * 78)
    for col, name in [("nu_R1", "R1 raw"), ("nu_R1+B2", "R1+B2"),
                      ("nu_MPNN", "R1+MPNN")]:
        print(f"  {name:10s} {relmae(df[col], df['nu_CCSD(T)']):6.2f}  ->  "
              f"{relmae(aud[col], aud['nu_CCSD(T)']):6.2f}")

    # per-bond-type relMAE uses the chemically normalized undirected key (Table 6)
    print("\nPer-bond-type R1+MPNN relMAE (%, chemically normalized key)  [rel -> aud]")
    for bt in ["C=O", "O-H", "N-H", "C#N", "O-F", "C=C", "C-O", "C-H"]:
        a = df[df.bt_chem == bt]
        b = aud[aud.bt_chem == bt]
        extra = f"{relmae(a.nu_MPNN, a['nu_CCSD(T)']):5.2f} -> {relmae(b.nu_MPNN, b['nu_CCSD(T)']):5.2f}" \
                if len(a) != len(b) else f"{relmae(a.nu_MPNN, a['nu_CCSD(T)']):5.2f} (unchanged)"
        print(f"  {bt:5s} n={len(a):3d}/{len(b):3d}   {extra}")

    # learnability uses the released-label token key (Table 7 / SI S9-S10)
    print("\nRemovable log-ratio variance (released-label bins, >=5 modes)  [rel -> aud]")
    rows = [("all modes", df, aud),
            ("excluding C-H", df[df.bt_label != "C-H"], aud[aud.bt_label != "C-H"]),
            ("single-mode supervised", df[df.bt_label.isin(SUPERVISED)],
             aud[aud.bt_label.isin(SUPERVISED)]),
            ("unsupervised single/halogen", df[df.bt_label.isin(UNSUPERVISED)],
             aud[aud.bt_label.isin(UNSUPERVISED)])]
    for name, r, a in rows:
        print(f"  {name:28s} n={len(r):3d}/{len(a):3d}   "
              f"{removable_fraction(r)*100:5.1f}% -> {removable_fraction(a)*100:5.1f}%")
    print("\nExpected (paper Tables 6-7, SI S10): aggregate MPNN 11.81->6.31; "
          "C-H 12.96->2.38; C=C 9.74->3.89; learnability all 10.2->46.0, supervised 42.9->82.1.")


if __name__ == "__main__":
    main()
