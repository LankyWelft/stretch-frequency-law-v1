# -*- coding: utf-8 -*-
"""V7-R14 实验D: 主动学习飞轮模拟 (QM9 作为 DFT 预言机)
四条独立飞轮: 随机 / 不确定性(MC-dropout) / 分层(KMeans) / 混合
每轮"查询" batch 个分子的真值(查 QM9 表, 模拟 DFT), 对比学习曲线。
真实 DFT(B3LYP) 计算替换 acquire() 即可上生产。"""
import os, numpy as np, math, time
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v7_qm9_lib import BASE, SYM, R_V6

torch.manual_seed(0); np.random.seed(0)
d=np.load(os.path.join(BASE,'records_aug.npz'))
gid=d['gid']; zi=d['zi'].astype(int); zj=d['zj'].astype(int); bo=d['bo'].astype(int)
fam=d['fam'].astype(int); nu0=d['nu0'].astype(float); nuref=d['nuref'].astype(float)
mu=d['mu'].astype(float); Rij=d['Rij'].astype(float)
nheavy=d['nheavy'].astype(int); nH=d['nH'].astype(int); nmult=d['nmult'].astype(int); nring=d['nring'].astype(int)
degi=d['degi'].astype(int);degj=d['degj'].astype(int);hi=d['hi'].astype(int);hj=d['hj'].astype(int)
bosumi=d['bosumi'].astype(int);bosumj=d['bosumj'].astype(int);hybi=d['hybi'].astype(int);hybj=d['hybj'].astype(int)
inring=d['inring'].astype(int);ringsize=d['ringsize'].astype(int);arom=d['arom'].astype(int);conj=d['conj'].astype(int)
adjN=(d['adjNi']+d['adjNj']).astype(int);adjO=(d['adjOi']+d['adjOj']).astype(int)
ZINNER={1:0,6:2,7:2,8:2,9:2}
chi_v=np.vectorize(lambda z:(z-ZINNER[z]/math.pi)/R_V6[z]); chi_i=chi_v(zi);chi_j=chi_v(zj);adchi=np.abs(chi_i-chi_j)
def blabel(k):
    if fam[k]==1: return f"{SYM[zi[k]]}-H"
    sym={2:'=',3:'#'}.get(bo[k],'-')
    return f"{SYM[zi[k]]}{sym}{SYM[zj[k]]}"
lab=np.array([blabel(k) for k in range(len(gid))])
y=np.log(nuref/nu0).astype(np.float32)
def oh(a,lv): return np.stack([(a==x).astype(np.float32) for x in lv],1)
types=sorted(np.unique(lab).tolist())
num=np.column_stack([np.log(mu),np.log(Rij),degi,degj,hi,hj,bosumi,bosumj,hybi,hybj,
    inring,arom,conj,ringsize,adjN,adjO,nheavy,nH,nmult,nring,chi_i,chi_j,adchi]).astype(np.float32)
X=np.hstack([num,oh(lab,types),oh(ringsize,[0,3,4,5,6])]).astype(np.float32)

# 分子级划分
uniq=np.unique(gid)
rng=np.random.RandomState(42); rng.shuffle(uniq)
N0,NPOOL,NTEST=3000,45000,15000
init_g=set(uniq[:N0]); test_g=set(uniq[N0:N0+NTEST]); pool_g=uniq[N0+NTEST:N0+NTEST+NPOOL]
gindex={}
order=np.argsort(gid,kind='stable'); gs=gid[order]
_gmin=gs.min()
bnds=np.searchsorted(gs,np.arange(_gmin,gs.max()+2),side='left')
def rows_of(g):
    lo=bnds[g-_gmin]; hi=bnds[g-_gmin+1]; return order[lo:hi]
test_rows=np.sort(np.concatenate([rows_of(g) for g in test_g]))
pool_list=list(pool_g)

# 分子级描述子(用于分层采样): 组成+全局拓扑
comp=np.zeros((len(uniq),6))
gpos={g:k for k,g in enumerate(uniq)}
desc=np.zeros((len(uniq),10))
for g in uniq:
    r=rows_of(g); k=gpos[g]
    for z,c in [(6,0),(7,1),(8,2),(9,3),(1,4)]:
        desc[k,c]=np.mean((zi[r]==z)|(zj[r]==z)) if len(r) else 0
    desc[k,5]=nheavy[r][0]; desc[k,6]=nmult[r][0]; desc[k,7]=nring[r][0]
    desc[k,8]=np.mean(conj[r]); desc[k,9]=np.mean(inring[r])
sc_desc=StandardScaler(); desc_s=sc_desc.fit_transform(desc)
pool_k=np.array([gpos[g] for g in pool_list])
km=KMeans(n_clusters=40,random_state=0,n_init=5).fit(desc_s[pool_k])
pool_cluster=dict(zip(pool_list,km.labels_))

class Net(nn.Module):
    def __init__(s,dim):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(dim,48),nn.ReLU(),nn.Dropout(0.15),
                            nn.Linear(48,48),nn.ReLU(),nn.Dropout(0.15),nn.Linear(48,1))
    def forward(s,x): return s.net(x).squeeze(-1)

def train_and_score(train_g, pool_g_list):
    tr_rows=np.sort(np.concatenate([rows_of(g) for g in train_g]))
    sc=StandardScaler(); Xtr=torch.tensor(sc.fit_transform(X[tr_rows]).astype(np.float32)); ytr=torch.tensor(y[tr_rows])
    Xte=torch.tensor(sc.transform(X[test_rows]).astype(np.float32))
    net=Net(X.shape[1]); opt=torch.optim.Adam(net.parameters(),lr=2e-3); lf=nn.MSELoss()
    bs=8192; n=len(Xtr)
    for ep in range(15):
        net.train(); perm=torch.randperm(n)
        for s0 in range(0,n,bs):
            idx=perm[s0:s0+bs]; opt.zero_grad()
            l=lf(net(Xtr[idx]),ytr[idx]); l.backward(); opt.step()
    net.eval()
    with torch.no_grad(): pred_te=net(Xte).numpy()
    relmae=np.mean(np.abs(np.exp(pred_te-y[test_rows])-1))*100
    # 候选池向量化 MC-dropout
    pool_rows=np.sort(np.concatenate([rows_of(g) for g in pool_g_list]))
    Xp=torch.tensor(sc.transform(X[pool_rows]).astype(np.float32))
    pg=gid[pool_rows]
    net.train()
    with torch.no_grad():
        ps=np.stack([net(Xp).numpy() for _ in range(12)])  # [12, nrows]
    row_var=ps.var(0)
    # 按 gid 聚合均值方差
    uniq_p,inv=np.unique(pg,return_inverse=True)
    acc=np.zeros(len(uniq_p)); cnt=np.zeros(len(uniq_p))
    np.add.at(acc,inv,row_var); np.add.at(cnt,inv,1)
    mvar=acc/np.maximum(cnt,1)
    unc=dict(zip(uniq_p.tolist(),mvar.tolist()))
    return relmae, unc

def run_flywheel(strategy, rounds=6, batch=1500):
    train=set(init_g); pool=list(pool_list); curve=[]
    for rnd in range(rounds+1):
        relmae,unc=train_and_score(train,pool)
        curve.append((len(train),relmae))
        print(f"    [{strategy}] 标注分子={len(train):>6}  test relMAE={relmae:.2f}%",flush=True)
        if rnd==rounds: break
        if strategy=='random':
            pick=rng.choice(len(pool),batch,replace=False)
        elif strategy=='uncertainty':
            order_u=sorted(range(len(pool)),key=lambda k:-unc[pool[k]])
            pick=order_u[:batch]
        elif strategy=='stratified':
            # 每簇轮询均匀选
            from collections import defaultdict
            byc=defaultdict(list)
            for k,g in enumerate(pool): byc[pool_cluster[g]].append(k)
            pick=[]; cl=sorted(byc)
            while len(pick)<batch:
                for c in cl:
                    if byc[c]: pick.append(byc[c].pop(0))
                    if len(pick)>=batch: break
            pick=np.array(pick)
        else:  # mixed: 一半高不确定, 一半分层
            order_u=sorted(range(len(pool)),key=lambda k:-unc[pool[k]])
            half=batch//2; pu=order_u[:half]
            from collections import defaultdict
            byc=defaultdict(list)
            for k,g in enumerate(pool):
                if k not in set(pu): byc[pool_cluster[g]].append(k)
            pick=list(pu); cl=sorted(byc)
            while len(pick)<batch:
                for c in cl:
                    if byc[c]: pick.append(byc[c].pop(0))
                    if len(pick)>=batch: break
            pick=np.array(pick)
        chosen=[pool[k] for k in sorted(pick,reverse=True)]
        for g in chosen: train.add(g); pool.remove(g)
    return curve

t0=time.time()
results={}
for strat in ['random','uncertainty','stratified','mixed']:
    print(f"\n=== 飞轮策略: {strat} ===",flush=True)
    results[strat]=run_flywheel(strat)
print(f"\n总耗时 {time.time()-t0:.0f}s")
print("\n=== 学习曲线汇总 (标注分子数: relMAE%) ===")
for strat,cur in results.items():
    print(f"  {strat:>12}: "+"  ".join(f"{n}:{v:.2f}" for n,v in cur))
import json
json.dump({k:[(int(a),float(b)) for a,b in v] for k,v in results.items()},
          open(os.path.join(BASE,'r14d_active_curves.json'),'w'),indent=1)
print('saved r14d_active_curves.json')
