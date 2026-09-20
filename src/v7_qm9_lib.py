"""
V7 QM9 shared library (Linux cloud, reproduced from the Windows R10 pipeline).
Constants are identical to the Windows run so numbers reproduce exactly.
"""
import os, math
import numpy as np

# ---- paths (Linux cloud) ----
BASE = '/home/user/Doubao/chats/38438738864945410/V7-R10'
QM9_DIR = os.path.join(BASE, 'dsgdb9nsd', 'dsgdb9nsd')  # tar extracts to dsgdb9nsd/dsgdb9nsd/*.xyz
BAD_IDS_FILE = os.path.join(BASE, 'bad_qm9.txt')

# if extraction layout differs, auto-detect: find the dir holding *.xyz
if not os.path.isdir(QM9_DIR):
    for cand in [os.path.join(BASE, 'dsgdb9nsd'), BASE]:
        if any(True for _ in __import__('glob').glob(os.path.join(cand, '*.xyz'))[:1]):
            QM9_DIR = cand
            break

SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
ZMAP = {v: k for k, v in SYM.items()}
MASS = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998}

# V6 radial lengths (Angstrom), locked values used by the Windows R10 run.
# H radial eigenvalue is a0*sqrt(3) = 0.9166 (NOT bare a0=0.529); pinned by
# C-H base median = 0.985 on QM9. Heavy values per the locked V6 table.
R_V6 = {1: 0.9166, 6: 1.119, 7: 1.073, 8: 1.034, 9: 1.002}

# R1 stretch law, locked exponents (b=-0.527 reproduces harmonic oscillator mu^-1/2)
C_R1 = 2962.0
B_MU, P_R, Q_BO = -0.527, -0.856, 0.680


def reduced_mass(za, zb):
    ma, mb = MASS[za], MASS[zb]
    return ma * mb / (ma + mb)


def r1_nu(mu, Rij, bo):
    return C_R1 * mu ** B_MU * Rij ** P_R * bo ** Q_BO


def _fl(t):
    return float(t.replace('*^', 'e'))


def parse_xyz(path):
    """Return (gid, atoms[list of element str], coords(ndarray), freqs(ndarray), smi)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    n = int(lines[0].strip())
    gid = int(lines[1].split()[1])
    atoms, xy = [], []
    for i in range(n):
        q = lines[2 + i].split()
        atoms.append(q[0])
        xy.append([_fl(q[1]), _fl(q[2]), _fl(q[3])])
    coords = np.asarray(xy, dtype=float)
    freq_line = lines[2 + n]
    freqs = np.array([_fl(t) for t in freq_line.split()], dtype=float)
    smi_parts = lines[3 + n].split()
    smi = smi_parts[-1] if smi_parts else ''
    return gid, atoms, coords, freqs, smi


def assign_bonds(atoms, coords):
    """Minimal connectivity used for X-H assignment: each H bonds to nearest heavy."""
    Zn = [ZMAP[a] for a in atoms]
    heavy = [i for i, z in enumerate(Zn) if z != 1]
    bonds = []
    hc = coords[heavy]
    for h, z in enumerate(Zn):
        if z == 1:
            j = heavy[int(np.argmin(np.linalg.norm(hc - coords[h], axis=1)))]
            bonds.append((h, j))
    return Zn, bonds
