#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent external validation of the *trained* MPNN delta-learning layer
on VIBFREQ1295 (CCSD(T)-F12c harmonic / experimental fundamentals).

This script is fully self-contained:
  * graphs are built from SMILES with RDKit (no QM9 derived arrays, no 3-D
    coordinates -- R uses the locked spectral radial lengths, bond orders come
    from the SMILES bond table, and rings from SSSR);
  * the locked MPNN weights are loaded from ../weights/c2_mpnn_weights.pt;
  * predictions are compared per mode to the shipped examples/si_modes.csv
    ground truth (the nu_MPNN column), then aggregated against CCSD(T)-F12c
    and experimental frequencies.

Run from any working directory:
    python run_mpnn_external.py
"""
import csv, re, math, json, os
import numpy as np
from rdkit import Chem
from rdkit.Chem import GetSymmSSSR
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---- locked physical constants (identical to src/v7_qm9_lib.py / demo) ----
R_V6  = {1: 0.9166, 6: 1.119, 7: 1.073, 8: 1.034, 9: 1.002}
MASS  = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998}
ZINNER = {1: 0, 6: 2, 7: 2, 8: 2, 9: 2}
ZIDX  = {6: 0, 7: 1, 8: 2, 9: 3}
RING_LEVELS = [0, 3, 4, 5, 6]

def r1_nu(mu, Rij, bo):  return 2962.0 * mu**(-0.527) * Rij**(-0.856) * bo**0.680
def redmass(a, b):      return MASS[a]*MASS[b]/(MASS[a]+MASS[b])
def chi(z):             return (z - ZINNER[z]/math.pi)/R_V6[z]
def bo_type(t):
    return {Chem.BondType.SINGLE:1, Chem.BondType.DOUBLE:2, Chem.BondType.TRIPLE:3}.get(t, 1)

# ---- MPNN architecture (verbatim copy of src/r14c2_mpnn.py) ----
FN, FE, H = 20, 13, 64
class MPNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.vin = nn.Linear(FN, H)
        s.msg = nn.ModuleList([nn.Sequential(nn.Linear(2*H+FE, H), nn.ReLU(), nn.Linear(H, H)) for _ in range(3)])
        s.up  = nn.ModuleList([nn.Linear(H, H) for _ in range(3)])
        s.ln  = nn.ModuleList([nn.LayerNorm(H) for _ in range(3)])
        s.eout = nn.Sequential(nn.Linear(2*H+FE, 64), nn.ReLU(), nn.Linear(64, 1))
        s.nout = nn.Sequential(nn.Linear(H, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, nf, ei, ef, nb):
        h = s.vin(nf)
        for L in range(3):
            hs = torch.cat([h[ei[0]], h[ei[1]], ef], 1)
            m = s.msg[L](hs)
            agg = torch.zeros_like(h).index_add_(0, ei[1], m)
            deg = torch.zeros(h.shape[0]).index_add_(0, ei[1], torch.ones(ei.shape[1])).clamp(min=1).unsqueeze(1)
            h = s.ln[L](h + torch.relu(s.up[L](agg/deg)))
        return s.eout(torch.cat([h[ei[0]], h[ei[1]], ef], 1)).squeeze(-1), s.nout(h).squeeze(-1)

def build_graph(smi):
    """Return node/edge tensors plus per-edge and per-X/H-node R1 priors."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    m = Chem.AddHs(m)
    Chem.Kekulize(m, clearAromaticFlags=True)
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() != 1]
    lmap = {g: k for k, g in enumerate(heavy)}
    nh = len(heavy)
    zlist = [m.GetAtomWithIdx(g).GetAtomicNum() for g in heavy]

    bo_mat = {}
    for bd in m.GetBonds():
        g1, g2 = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        z1, z2 = m.GetAtomWithIdx(g1).GetAtomicNum(), m.GetAtomWithIdx(g2).GetAtomicNum()
        if z1 != 1 and z2 != 1:
            a, b = lmap[g1], lmap[g2]
            bo_mat[(min(a, b), max(a, b))] = bo_type(bd.GetBondType())

    # rings (SSSR approximates networkx.minimum_cycle_basis for these small molecules)
    rsize = [0]*nh; arom = [0]*nh
    for ring in GetSymmSSSR(m):
        cyc = [lmap[g] for g in ring if m.GetAtomWithIdx(g).GetAtomicNum() != 1]
        rs = len(cyc)
        if rs < 3: continue
        nd = sum(1 for i in range(rs) for j in range(i+1, rs)
                 if bo_mat.get((min(cyc[i], cyc[j]), max(cyc[i], cyc[j])), 0) == 2)
        isa = 1 if (rs == 6 and nd >= 2) else 0
        for k in cyc:
            if rsize[k] == 0 or rs < rsize[k]: rsize[k] = rs
            if isa: arom[k] = 1

    deg = [0]*nh; bosum = [0]*nh; maxbo = [1]*nh; adjN = [0]*nh; adjO = [0]*nh
    hcnt = [0]*nh
    neigh = {k: [] for k in range(nh)}
    for (a, b), v in bo_mat.items():
        za, zb = zlist[a], zlist[b]
        deg[a] += 1; deg[b] += 1; bosum[a] += v; bosum[b] += v
        maxbo[a] = max(maxbo[a], v); maxbo[b] = max(maxbo[b], v)
        if zb == 7: adjN[a] += 1
        if za == 7: adjN[b] += 1
        if zb == 8: adjO[a] += 1
        if za == 8: adjO[b] += 1
        neigh[a].append((b, v)); neigh[b].append((a, v))
    for bd in m.GetBonds():
        g1, g2 = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        z1, z2 = m.GetAtomWithIdx(g1).GetAtomicNum(), m.GetAtomWithIdx(g2).GetAtomicNum()
        if (z1 == 1) != (z2 == 1):
            gk = g1 if z1 != 1 else g2
            hcnt[lmap[gk]] += 1

    nf = []
    for a in range(nh):
        z = zlist[a]
        hyb = 3 if maxbo[a] == 3 else (2 if maxbo[a] == 2 else 1)
        row = [1 if ZIDX.get(z, -1) == k else 0 for k in range(4)]
        row += [hcnt[a]/4, deg[a]/4, bosum[a]/4]
        row += [1 if hyb == k else 0 for k in (1, 2, 3)]
        row += [1 if rsize[a] > 0 else 0, arom[a]]
        row += [1 if rsize[a] == r else 0 for r in RING_LEVELS]
        row += [chi(z)/10, adjN[a]/3, adjO[a]/3]
        nf.append(row)
    nf = np.array(nf, dtype=np.float32)

    def mc(x, ex): return any(v >= 2 for (y, v) in neigh[x] if y != ex)
    ei, ef, edges = [], [], []
    for (a, b), bo in sorted(bo_mat.items()):
        za, zb = zlist[a], zlist[b]
        mu = redmass(za, zb); Rij = math.sqrt(R_V6[za]*R_V6[zb]); nup = r1_nu(mu, Rij, bo)
        conj = (1 if (mc(a, b) or mc(b, a)) else 0) if bo >= 2 else (1 if (mc(a, b) and mc(b, a)) else 0)
        row = [1 if bo == k else 0 for k in (1, 2, 3)]
        row += [math.log(mu), math.log(Rij), math.log(nup), conj,
                1 if rsize[a] > 0 and rsize[a] == rsize[b] else 0]
        row += [1 if rsize[a] == r and rsize[b] == r else 0 for r in RING_LEVELS]
        p = len(edges)
        ei.append([a, b]); ei.append([b, a]); ef.append(row); ef.append(row)
        edges.append(dict(p=p, a=a, b=b, key=(min(za, zb), max(za, zb), bo), nup=nup))

    hnup = [0.0]*nh
    for k in range(nh):
        zX = zlist[k]
        mu = redmass(1, zX); Rij = math.sqrt(R_V6[1]*R_V6[zX])
        hnup[k] = r1_nu(mu, Rij, 1.0)

    nft = torch.tensor(nf)
    eit = torch.tensor(np.array(ei).T, dtype=torch.long) if ei else torch.zeros(2, 0, dtype=torch.long)
    eft = torch.tensor(np.array(ef, dtype=np.float32)) if ef else torch.zeros(0, FE)
    return dict(zlist=zlist, nf=nft, ei=eit, ef=eft, edges=edges, hnup=hnup)

def load_data():
    rows = list(csv.DictReader(open(os.path.join(HERE, 'VIBFREQ1295_Data.csv'), encoding='latin-1')))
    allowed = {'C', 'H', 'O', 'N', 'F'}
    def elems(f): return set(re.findall(r'[A-Z][a-z]?', f))
    bymol = {}
    for r in rows:
        if elems(r['Molecular_Formula']) - allowed: continue
        if 'tretch' not in (r['Mode_Description_Exp'] or r['Mode_Description']).lower(): continue
        bymol.setdefault(r['Molecule'], []).append(r)
    return bymol

SYM2Z = {'H':1,'C':6,'N':7,'O':8,'F':9}
def parse_desc(d):
    d = d.strip()
    if 'Ring' in d or '+' in d: return None
    m = re.match(r'^([CNHOF])\s*(=|#|-)\s*([CNHOF])', d)
    if not m: return None
    a = SYM2Z[m.group(1)]; b = SYM2Z[m.group(3)]
    bo = 2 if m.group(2) == '=' else (3 if m.group(2) == '#' else 1)
    if b == 1: bo = 1
    return (a, b, bo)

def main():
    import time; t0 = time.time()
    net = MPNN()
    net.load_state_dict(torch.load(os.path.join(ROOT, 'weights', 'c2_mpnn_weights.pt'), map_location='cpu'))
    net.eval()
    print(f'weights loaded ({time.time()-t0:.1f}s)', flush=True)

    bymol = load_data()
    pred_cache = {}
    def mpnn_value(mol, key):
        if mol not in pred_cache:
            modes = bymol[mol]; g = build_graph(modes[0]['SMILES'])
            edge_val, node_val = {}, {}
            if g is not None:
                with torch.no_grad():
                    pe, pn = net(g['nf'], g['ei'], g['ef'], None)
                for e in g['edges']:
                    edge_val.setdefault(e['key'], []).append(e['nup']*math.exp(float(pe[2*e['p']])))
                for k, zX in enumerate(g['zlist']):
                    node_val.setdefault((1, zX, 1), []).append(g['hnup'][k]*math.exp(float(pn[k])))
            pred_cache[mol] = (edge_val, node_val)
        edge_val, node_val = pred_cache[mol]
        x, y = min(key[0], key[1]), max(key[0], key[1])
        vals = node_val.get((1, y, 1)) if x == 1 else edge_val.get((x, y, key[2]))
        return float(np.mean(vals)) if vals else None

    recs = []
    for mol, modes in bymol.items():
        for r in modes:
            desc = r['Mode_Description_Exp'] or r['Mode_Description']
            kb = parse_desc(desc)
            if kb is None: continue
            pv = mpnn_value(mol, kb)
            if pv is None: continue
            try: ccsd = float(r['CCSD(T)-F12c_Freqs']); exp = float(r['ExpFreq_New'])
            except Exception: continue
            recs.append(dict(mol=mol, desc=desc, key=kb, pred=pv, ccsd=ccsd, exp=exp))
    print(f'predictions done ({time.time()-t0:.1f}s): {len(recs)} modes', flush=True)

    # ---- self-check vs shipped nu_MPNN ground truth (greedy nearest per molecule) ----
    si = os.path.join(ROOT, 'examples', 'si_modes.csv')
    if os.path.exists(si):
        gt = {}
        for r in csv.DictReader(open(si)): gt.setdefault(r['Molecule'], []).append(float(r['nu_MPNN']))
        diffs = []
        bymol_rec = {}
        for x in recs: bymol_rec.setdefault(x['mol'], []).append(x['pred'])
        for mol, preds in bymol_rec.items():
            gtv = sorted(gt.get(mol, [])); pv = sorted(preds)
            if len(gtv) == len(pv):
                for a, b in zip(gtv, pv):
                    if a > 0: diffs.append(abs(a-b)/a)
        if diffs:
            print(f"[self-check] vs shipped nu_MPNN: mean|d|={np.mean(diffs)*100:.2f}%  max={np.max(diffs)*100:.1f}%  matched={len(diffs)}")

    from collections import defaultdict
    SYMN = {1:'H',6:'C',7:'N',8:'O',9:'F'}
    g = defaultdict(list)
    for x in recs:
        a, b, bo = x['key']
        nm = f"{SYMN[a]}{'=' if bo==2 else ('#' if bo==3 else '-')}{SYMN[b]}" if b != 1 else f"{SYMN[a]}-H"
        g[nm].append(x)
    print(f"\nR1+MPNN external validation   分子={len(bymol)}  可比较模式={len(recs)}")
    print(f"{'键型':>6} {'n':>4} {'vs CCSD(T) relMAE':>18} {'vs 实验 relMAE':>16}")
    tc, te = [], []
    for nm in sorted(g):
        xs = g[nm]
        rc = [abs(x['pred']/x['ccsd']-1) for x in xs]; re_ = [abs(x['pred']/x['exp']-1) for x in xs]
        tc += rc; te += re_
        print(f"{nm:>6} {len(xs):>4} {np.mean(rc)*100:>17.2f}% {np.mean(re_)*100:>15.2f}%")
    print(f"{'总体':>6} {len(tc):>4} {np.mean(tc)*100:>17.2f}% {np.mean(te)*100:>15.2f}%")

if __name__ == '__main__':
    main()
