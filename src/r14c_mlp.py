# -*- coding: utf-8 -*-
"""V7-R14 实验C1: 边 MLP 的 Δ-学习 (非线性, 不做消息传递)
特征同 B3, 3层 MLP 学 log(sigma), 分子级划分。"""
import os, numpy as np, math, time
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v7_qm9_lib import BASE, SYM, R_V6

torch.manual_seed(0); np.random.seed(0)
d=np.load(os.path.join(BASE,'records_aug.npz'))
gid=d['gid']; zi=d['zi'].astype(int); zj=d['zj'].astype(int); bo=d['bo'].astype(int)
fam=d['fam'].astype(int); nu0=d['nu0'].astype(float); nuref=d['nuref'].astype(float)
mu=d['mu'].astype(float); Rij=d['Rij'].astype(float)
nheavy=d['nheavy'].astype(int); nH=d['nH'].astype(int); nmult=d['nmult'].astype(int); nring=d['nring'].astype(int)
degi=d['degi'].astype(int); degj=d['degj'].astype(int); hi=d['hi'].astype(int); hj=d['hj'].astype(int)
bosumi=d['bosumi'].astype(int); bosumj=d['bosumj'].astype(int)
hybi=d['hybi'].astype(int); hybj=d['hybj'].astype(int)
inring=d['inring'].astype(int); ringsize=d['ringsize'].astype(int)
arom=d['arom'].astype(int); conj=d['conj'].astype(int)
adjNi=d['adjNi'].astype(int); adjOi=d['adjOi'].astype(int)
adjNj=d['adjNj'].astype(int); adjOj=d['adjOj'].astype(int)
ZINNER={1:0,6:2,7:2,8:2,9:2}
chi_v=np.vectorize(lambda z:(z-ZINNER[z]/math.pi)/R_V6[z])
chi_i=chi_v(zi); chi_j=chi_v(zj); abs_dchi=np.abs(chi_i-chi_j)
def blabel(k):
    if fam[k]==1: return f"{SYM[zi[k]]}-H"
    sym={2:'=',3:'#'}.get(bo[k],'-')
    return f"{SYM[zi[k]]}{sym}{SYM[zj[k]]}"
lab=np.array([blabel(k) for k in range(len(gid))])
y=np.log(nuref/nu0).astype(np.float32)

def onehot(arr,levels):
    return np.stack([(arr==lv).astype(np.float32) for lv in levels],1)
types=sorted(np.unique(lab).tolist())
num=np.column_stack([np.log(mu),np.log(Rij),degi,degj,hi,hj,bosumi,bosumj,hybi,hybj,
    inring,arom,conj,ringsize,adjNi+adjNj,adjOi+adjOj,nheavy,nH,nmult,nring,chi_i,chi_j,abs_dchi]).astype(np.float32)
X=np.hstack([num,onehot(lab,types),onehot(ringsize,[0,3,4,5,6])])
sc=StandardScaler(); Xs=sc.fit_transform(X).astype(np.float32)

tr=(gid%2==1); te=~tr
Xtr=torch.tensor(Xs[tr]); ytr=torch.tensor(y[tr])
Xte=torch.tensor(Xs[te]); yte=torch.tensor(y[te])
dim=Xs.shape[1]
net=nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
opt=torch.optim.Adam(net.parameters(),lr=2e-3)
lossf=nn.MSELoss()
t0=time.time(); bs=8192; n=len(Xtr)
for ep in range(25):
    perm=torch.randperm(n)
    net.train()
    for s in range(0,n,bs):
        idx=perm[s:s+bs]
        opt.zero_grad(); loss=lossf(net(Xtr[idx]).squeeze(-1),ytr[idx]); loss.backward(); opt.step()
    if ep%5==0 or ep==24:
        net.eval()
        with torch.no_grad():
            pte=net(Xte).squeeze(-1)
            rel=np.mean(np.abs(np.exp(pte.numpy())*np.exp(-yte.numpy())-1))*100
        print(f"  epoch {ep:>2}: test relMAE={rel:.2f}%  ({time.time()-t0:.0f}s)",flush=True)

net.eval()
with torch.no_grad():
    sig_te=np.exp(net(Xte).squeeze(-1).numpy())
    sig_tr=np.exp(net(Xtr).squeeze(-1).numpy())
sig=np.zeros(len(gid)); sig[tr]=sig_tr; sig[te]=sig_te

def rmae(mask,s): return np.mean(np.abs(nu0[mask]*s[mask]/nuref[mask]-1))*100
print("\n=== C1 MLP vs 基线 (测试集) ===")
print(f"  R1 原始:       {rmae(te,np.ones(len(gid))):.2f}%")
# 查表基线
sig_type=np.ones(len(gid))
for t in types:
    m=tr&(lab==t)
    if m.sum()>=20: sig_type[lab==t]=math.exp(np.median(y[m]))
print(f"  键型查表:       {rmae(te,sig_type):.2f}%")
print(f"  MLP Δ-学习:    {rmae(te,sig):.2f}%")
print("\n逐键型 (R1原始 / 查表 / MLP):")
for t in ['C-H','N-H','O-H','C=C','C#C','C=N','C#N','C=O','N=O']:
    m=te&(lab==t)
    if m.sum()<20: continue
    print(f"  {t:>6} n={m.sum():>7}: {rmae(m,np.ones(len(gid))):5.2f}% / {rmae(m,sig_type):5.2f}% / {rmae(m,sig):5.2f}%")
# 外推
trx=(nheavy<=6); tex=(nheavy>=7)
netx=nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
optx=torch.optim.Adam(netx.parameters(),lr=2e-3)
Xa=torch.tensor(Xs[trx]); ya=torch.tensor(y[trx]); Xb=torch.tensor(Xs[tex])
for ep in range(20):
    perm=torch.randperm(len(Xa)); netx.train()
    for s in range(0,len(Xa),bs):
        idx=perm[s:s+bs]; optx.zero_grad()
        l=lossf(netx(Xa[idx]).squeeze(-1),ya[idx]); l.backward(); optx.step()
netx.eval()
with torch.no_grad(): sx=np.exp(netx(Xb).squeeze(-1).numpy())
sigx=np.ones(len(gid)); sigx[tex]=sx
print(f"\n外推 (训练≤6重, 测试7-9重, n={tex.sum()}): R1 {rmae(tex,np.ones(len(gid))):.2f}% -> MLP {rmae(tex,sigx):.2f}%")
