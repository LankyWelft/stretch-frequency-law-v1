# -*- coding: utf-8 -*-
"""r14e 补充: 把外部 307 的低可学性分解为 (XHn 对称劈裂) vs (单模式键型环境信号);
并量化 (log mu,log R,log BO) 设计矩阵的病态条件数。"""
import os, numpy as np, pandas as pd
from collections import defaultdict
ROOT = '/home/user/Doubao/chats/38438738864945410'
BASE = os.path.join(ROOT, 'V7-R10')
def banner(t): print('\n' + '=' * 74 + f'\n{t}\n' + '=' * 74)

df = pd.read_csv(os.path.join(ROOT, 'stretch-frequency-law/examples/si_modes.csv'))
df['e'] = np.log(df['nu_R1'] / df['nu_CCSD(T)'])
def btype(m):
    return [t for t in m.replace(' stretch','').split()
            if t.lower() not in ('sym','asym','symmetric','antisymmetric')][0]
def sym(m):
    ml=m.lower()
    return 'asym' if 'asym' in ml else ('sym' if 'sym' in ml else '-')
df['bt']=df['Mode'].map(btype); df['sy']=df['Mode'].map(sym)

def frac(sub, minn=5, keycol='bt'):
    ee=sub['e'].values
    bucket=defaultdict(list)
    for k,ek in zip(sub[keycol].values,ee): bucket[k].append(ek)
    resid=np.concatenate([np.array(v)-np.mean(v) for v in bucket.values() if len(v)>=minn])
    used=sum(len(v) for v in bucket.values() if len(v)>=minn)
    return ee.std(), resid.std(), 1-(resid.std()/ee.std())**2, used, len(ee)

banner('A1. C-H 的 per-carbon 输出地板: R1 对 sym/asym 给同值, CCSD 劈裂')
ch=df[df['bt']=='C-H']
print(f"C-H n={len(ch)}: nu_R1 取值数={ch['nu_R1'].nunique()}  log(nu_R1) std={np.log(ch['nu_R1']).std():.4f}")
print(f"          log(nu_CCSD) std={np.log(ch['nu_CCSD(T)']).std():.4f}  "
      f"CCSD 频率范围 {ch['nu_CCSD(T)'].min():.0f}-{ch['nu_CCSD(T)'].max():.0f} cm-1")
for s in ('sym','asym'):
    c=ch[ch['sy']==s]
    print(f"  C-H {s:4s}: n={len(c):3d}  CCSD 均值 {c['nu_CCSD(T)'].mean():.1f}  R1 均值 {c['nu_R1'].mean():.1f}")
# 同分子 sym/asym 配对劈裂
piv=ch.pivot_table(index='Molecule', columns='sy', values='nu_CCSD(T)', aggfunc='first')
have=piv.dropna(subset=['sym','asym'])
split=(have['sym']-have['asym']).abs()
print(f"  同分子 sym-asym 劈裂: n={len(have)} 对, 均值 {split.mean():.1f} cm-1 "
      f"(相对 { (split/have.mean(axis=1)*100).mean():.1f}% )")

banner('A2. 排除 C-H 后 (单模式键型为主) 的可学性')
for label, sub in [
    ('全部 307', df),
    ('排除 C-H', df[df['bt']!='C-H']),
    ('仅单模式有监督键型 (O-H,N-H,C=O,C=C,C#C,C#N,C=N,N=O,N=N)',
     df[df['bt'].isin(['O-H','N-H','C=O','C=C','C#C','C#N','C=N','N=O','N=N'])]),
    ('仅重-重单键/卤素 (R1未监督: C-C,C-O,C-N,C-F,N-F,O-F,N-N,O-O,N-O)',
     df[df['bt'].isin(['C-C','C-O','C-N','C-F','N-F','O-F','N-N','O-O','N-O'])])]:
    tot,w,f,used,n=frac(sub,5)
    print(f"{label:62s} n={n:3d} 总std={tot:.3f} 组内std={w:.3f} 可缩减={f*100:5.1f}% (覆盖{used})")

banner('B1. 设计矩阵共线性/条件数 (QM9 训练集, 标准化后)')
d=np.load(os.path.join(BASE,'records_aug.npz')); tr=d['gid']%2==1
X=np.column_stack([np.log(d['mu'].astype(float)),np.log(d['Rij'].astype(float)),
                   np.log(d['bo'].astype(float).clip(1.0))])[tr]
Xs=(X-X.mean(0))/X.std(0)
C=np.corrcoef(Xs,rowvar=False)
ev=np.linalg.eigvalsh(C)
print('相关矩阵 [logmu, logR, logBO]:'); print(np.round(C,3))
print(f'特征值={np.round(ev,4)}  条件数={ev.max()/ev.min():.1f}  最小特征值={ev.min():.4f}')
# 11 个有效设计点上的相关 (质心层面才是真正拟合的)
zi,zj,bo,fam=d['zi'].astype(int),d['zj'].astype(int),d['bo'].astype(int),d['fam'].astype(int)
key=fam*1000000+zi*1000+zj*10+bo
import numpy as _np
cen=pd.DataFrame(np.column_stack([X,key[tr]]),columns=['mu','R','bo','k']).groupby('k').mean()
print(f'\n{len(cen)} 个有效设计点(质心)上 corr(logmu,logR)={np.corrcoef(cen.mu,cen.R)[0,1]:.3f}, '
      f'corr(logR,logBO)={np.corrcoef(cen.R,cen.bo)[0,1]:.3f}')
