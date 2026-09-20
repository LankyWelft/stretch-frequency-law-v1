#!/usr/bin/env python3
"""Minimal demonstration of the closed-form three-feature stretching law (R1).

Runs in <1 second, no dataset download required.

    nu = 2962 * mu^(-0.527) * R^(-0.856) * BO^(0.680)   [cm^-1]

R = sqrt(R_i * R_j); R_i are the locked spectral radial lengths.
"""
import math

MASS = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998}
R_V6 = {'H': 0.9166, 'C': 1.119, 'N': 1.073, 'O': 1.034, 'F': 1.002}

C_R1, P_MU, P_R, P_BO = 2962.0, -0.527, -0.856, 0.680


def stretch_freq(elem_i, elem_j, bond_order, r_ij=None):
    """Predicted stretching wavenumber in cm^-1 for the (i,j) bond."""
    ma, mb = MASS[elem_i], MASS[elem_j]
    mu = ma * mb / (ma + mb)
    R = r_ij if r_ij is not None else math.sqrt(R_V6[elem_i] * R_V6[elem_j])
    return C_R1 * mu ** P_MU * R ** P_R * bond_order ** P_BO


if __name__ == '__main__':
    print("R1 law: nu = 2962 * mu^-0.527 * R^-0.856 * BO^0.680  [cm^-1]\n")
    cases = [
        ('C', 'H', 1, 'sp3 C-H ~ 2900-3000'),
        ('N', 'H', 1, 'N-H ~ 3300'),
        ('O', 'H', 1, 'O-H ~ 3600'),
        ('C', 'C', 1, 'C-C single ~ 900-1100'),
        ('C', 'C', 2, 'C=C double ~ 1600'),
        ('C', 'C', 3, 'C#C triple ~ 2100'),
        ('C', 'O', 1, 'C-O single ~ 1000-1100'),
        ('C', 'O', 2, 'C=O carbonyl ~ 1700'),
    ]
    for a, b, bo, note in cases:
        nu = stretch_freq(a, b, bo)
        print(f"{a}-{b:<3} BO={bo}: {nu:7.1f} cm^-1   ({note})")
