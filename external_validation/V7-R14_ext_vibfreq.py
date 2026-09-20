#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V7-R14 外源验证: VIBFREQ1295 (独立实验/CCSD(T)-F12c benchmark) CHONF 子集.
从 SMILES 用 rdkit 建图, 跑 R1 伸缩定律, 与金标准谐振频率对比."""
import csv, re, math, os, json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))

R_V6={1:0.9166,6:1.119,7:1.073,8:1.034,9:1.002}
MASS={1:1.008,6:12.011,7:14.007,8:15.999,9:18.998}
SYM2Z={'H':1,'C':6,'N':7,'O':8,'F':9}
def r1_nu(mu,Rij,bo): return 2962.0*mu**(-0.527)*Rij**(-0.856)*bo**0.680
def redmass(a,b): return MASS[a]*MASS[b]/(MASS[a]+MASS[b])
def bo_type(t):
    return {Chem.BondType.SINGLE:1,Chem.BondType.DOUBLE:2,Chem.BondType.TRIPLE:3}.get(t,1)

def parse_desc(d):
    """'C=O stretch' -> (6,8,2); 'N-H asym stretch' -> (7,1,1). 返回 None=无法归键"""
    d=d.strip()
    if 'Ring' in d or '+' in d: return None
    m=re.match(r'^([CNHOF])\s*(=|#|-)\s*([CNHOF])',d)
    if not m: return None
    a=SYM2Z[m.group(1)]; b=SYM2Z[m.group(3)]
    sep=m.group(2); bo=2 if sep=='=' else (3 if sep=='#' else 1)
    if b==1: bo=1
    return (a,b,bo)

rows=list(csv.DictReader(open(os.path.join(HERE,'VIBFREQ1295_Data.csv'),encoding='latin-1')))
# B2 bond-type multiplicative factors, frozen from the QM9 training set and
# shipped as a small JSON so the external validation needs no large derived array.
_b2=json.load(open(os.path.join(HERE,'b2_bondtype_factors.json')))
B2={tuple(int(x) for x in k.split('-')):float(v) for k,v in _b2.items()}
print("B2 键型数:",len(B2))
allowed={'C','H','O','N','F'}
def elems(f): return set(re.findall(r'[A-Z][a-z]?',f))
# 按分子聚合 stretch 模式
bymol={}
for r in rows:
    if elems(r['Molecular_Formula']) - allowed: continue
    if 'tretch' not in (r['Mode_Description_Exp'] or r['Mode_Description']).lower(): continue
    bymol.setdefault(r['Molecule'],[]).append(r)

recs=[]; skipped=0
for mol,modes in bymol.items():
    smi=modes[0]['SMILES']
    m=Chem.MolFromSmiles(smi)
    if m is None: skipped+=1; continue
    m=Chem.AddHs(m)
    Chem.Kekulize(m,clearAromaticFlags=True)
    # 建键表
    bonds=[]
    for bd in m.GetBonds():
        z1=m.GetAtomWithIdx(bd.GetBeginAtomIdx()).GetAtomicNum()
        z2=m.GetAtomWithIdx(bd.GetEndAtomIdx()).GetAtomicNum()
        if z1 not in MASS or z2 not in MASS: continue
        if z1==1 and z2==1: continue
        bo=bo_type(bd.GetBondType())
        mu=redmass(z1,z2); Rij=math.sqrt(R_V6[z1]*R_V6[z2])
        bonds.append((min(z1,z2),max(z1,z2),bo,r1_nu(mu,Rij,bo)))
    def best_pred(a,b,bo):
        # 同键型最优匹配(元素对+BO)
        cand=[nu for (x,y,o,nu) in bonds if x==min(a,b) and y==max(a,b) and o==bo]
        return float(np.mean(cand)) if cand else None
    for r in modes:
        desc=r['Mode_Description_Exp'] or r['Mode_Description']
        kb=parse_desc(desc)
        if kb is None: skipped+=1; continue
        pred=best_pred(*kb)
        if pred is None: skipped+=1; continue
        # B2 factors are keyed by the undirected (min Z, max Z, BO) tuple; normalize
        # the parsed key so X-H (e.g. O-H -> (8,1,1)) and N#C -> (7,6,3) lookups hit.
        bkey=(min(kb[0],kb[1]),max(kb[0],kb[1]),kb[2])
        pred=pred*B2.get(bkey,1.0)
        try: ccsd=float(r['CCSD(T)-F12c_Freqs']); exp=float(r['ExpFreq_New'])
        except: continue
        recs.append(dict(mol=mol,desc=desc,key=kb,pred=pred,ccsd=ccsd,exp=exp))

print(f"分子={len(bymol)}  可比较模式={len(recs)}  跳过={skipped}")
# 按键型统计对 CCSD(T) 谐振
from collections import defaultdict
g=defaultdict(list)
for x in recs:
    a,b,bo=x['key']; sym={1:'H',6:'C',7:'N',8:'O',9:'F'}
    nm=f"{sym[a]}{'=' if bo==2 else ('#' if bo==3 else '-')}{sym[b]}" if b!=1 else f"{sym[a]}-H"
    g[nm].append(x)
print(f"\n{'键型':>6} {'n':>4} {'vs CCSD(T) relMAE':>18} {'vs 实验 relMAE':>16}")
tot_c=[];tot_e=[]
for nm in sorted(g):
    xs=g[nm]
    rc=[abs(x['pred']/x['ccsd']-1) for x in xs]
    re_=[abs(x['pred']/x['exp']-1) for x in xs]
    tot_c+=rc;tot_e+=re_
    print(f"{nm:>6} {len(xs):>4} {np.mean(rc)*100:>17.2f}% {np.mean(re_)*100:>15.2f}%")
print(f"{'总体':>6} {len(tot_c):>4} {np.mean(tot_c)*100:>17.2f}% {np.mean(tot_e)*100:>15.2f}%")
# 保存
json.dump(recs,open(os.path.join(HERE,'r14_ext_vibfreq.json'),'w'),default=float)
print("\nsaved r14_ext_vibfreq.json")
