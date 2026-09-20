# -*- coding: utf-8 -*-
"""V7-R14 实验B: 线性 Δ-修正基线
学乘性修正因子 log(sigma)=log(nu_ref/nu_R1), 分子级划分。
对比 R1 原始 -> 全局常数 -> 键型查表 -> 完整环境线性模型。"""
import os, numpy as np, math
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v7_qm9_lib import BASE, SYM, R_V6

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
def chi(z): return (z-ZINNER[z]/math.pi)/R_V6[z]
chi_v=np.vectorize(chi)
chi_i=chi_v(zi); chi_j=chi_v(zj); abs_dchi=np.abs(chi_i-chi_j)

def blabel(k):
    if fam[k]==1: return f"{SYM[zi[k]]}-H"
    sym={2:'=',3:'#'}.get(bo[k],'-')
    return f"{SYM[zi[k]]}{sym}{SYM[zj[k]]}"
lab=np.array([blabel(k) for k in range(len(gid))])

y=np.log(nuref/nu0)  # 目标: 乘性修正
tr=(gid%2==1); te=~tr

def metrics(mask,sigma_corr,tag):
    pred=nu0[mask]*sigma_corr[mask]
    ref=nuref[mask]; r=pred/ref
    return dict(tag=tag,n=int(mask.sum()),rel_mae=np.mean(np.abs(r-1))*100,
                std=np.std(np.log(r)),w10=np.mean(np.abs(r-1)<=.10)*100,
                bias=np.mean(np.log(r)))

def report(res):
    print(f"\n{res['tag']:>34} n={res['n']:>7} relMAE={res['rel_mae']:5.2f}%  std(log)={res['std']:.4f}  ±10%={res['w10']:5.1f}%  bias={res['bias']:+.4f}")

print("="*90)
print("实验B: 线性 Δ-修正 (测试集=偶数gid, 训练=奇数gid, 严格分子级划分)")
print("="*90)

# 基线0: 无修正
report(metrics(te,np.ones(len(gid)),'B0: R1 原始(无修正)'))
# 基线1: 全局常数(训练集拟合)
sig_global=math.exp(np.mean(y[tr]))
report(metrics(te,np.full(len(gid),sig_global),'B1: 全局常数 σ'))
# 基线2: 键型查表(训练集中位)
sig_type=np.ones(len(gid))
for t in np.unique(lab):
    m=tr&(lab==t)
    if m.sum()>=20: sig_type[lab==t]=math.exp(np.median(y[m]))
report(metrics(te,sig_type,'B2: 键型查表(15类)'))

# === 完整环境线性模型 ===
def onehot(arr, levels, prefix):
    return np.stack([(arr==lv).astype(float) for lv in levels],axis=1)

# 数值特征
num=np.column_stack([
    np.log(mu), np.log(Rij),
    degi,degj,hi,hj,bosumi,bosumj,hybi,hybj,
    inring,arom,conj,ringsize,
    adjNi+adjNj, adjOi+adjOj,
    nheavy,nH,nmult,nring,
    chi_i,chi_j,abs_dchi
])
num_names=['logμ','logR','degi','degj','hi','hj','bosumi','bosumj','hybi','hybj',
           'inring','arom','conj','ringsize','adjN','adjO','nheavy','nH','nmult','nring',
           'χi','χj','|Δχ|']
# one-hot: 键型, 环大小
types=sorted(np.unique(lab).tolist())
oh_type=onehot(lab,types,'t')
oh_ring=onehot(ringsize,[0,3,4,5,6],'r')
X=np.hstack([num,oh_type,oh_ring])
sc=StandardScaler()
Xtr=sc.fit_transform(X[tr]); Xte=sc.transform(X[te])
alphas=np.logspace(-4,3,30)
model=RidgeCV(alphas=alphas)
model.fit(Xtr,y[tr])
sig_full=np.exp(np.concatenate([model.predict(Xtr),model.predict(Xte)]))
# 重组回原顺序
sig_full_all=np.zeros(len(gid)); sig_full_all[tr]=np.exp(model.predict(Xtr)); sig_full_all[te]=np.exp(model.predict(Xte))
report(metrics(te,sig_full_all,'B3: 完整环境线性模型'))

# 消融: 去掉键型 one-hot(只靠环境+物理量)
keep_num=X
Xnum=sc.fit_transform(num[tr]); Xnum_te=sc.transform(num[te])
m2=RidgeCV(alphas=alphas); m2.fit(Xnum,y[tr])
sig_num=np.zeros(len(gid)); sig_num[tr]=np.exp(m2.predict(Xnum)); sig_num[te]=np.exp(m2.predict(Xnum_te))
report(metrics(te,sig_num,'B4: 仅物理+环境(无键型onehot)'))

# 消融: 键型+共轭/环(最精简环境)
small=np.hstack([oh_type, onehot(conj,[0,1],'c'), onehot(arom,[0,1],'a'),
                 onehot(ringsize,[0,3,4,5,6],'r'),
                 (adjNi+adjNj)[:,None],(adjOi+adjOj)[:,None],
                 np.log(mu)[:,None],np.log(Rij)[:,None]])
sc2=StandardScaler(); Xs=sc2.fit_transform(small[tr]); Xs_te=sc2.transform(small[te])
m3=RidgeCV(alphas=alphas); m3.fit(Xs,y[tr])
sig_s=np.zeros(len(gid)); sig_s[tr]=np.exp(m3.predict(Xs)); sig_s[te]=np.exp(m3.predict(Xs_te))
report(metrics(te,sig_s,'B5: 键型+共轭/环/α杂原子'))

# 逐键型 (B3)
print("\n=== B3 逐键型测试集表现 ===")
print(f"{'键型':>8} {'n':>7} {'R1原始%':>9} {'查表%':>8} {'线性B3%':>9}")
for t in ['C-H','N-H','O-H','C=C','C#C','C=N','C#N','C=O','N=O']:
    m=te&(lab==t)
    if m.sum()<20: continue
    def rmae(sig):
        return np.mean(np.abs(nu0[m]*sig[m]/nuref[m]-1))*100
    print(f"{t:>8} {m.sum():>7} {rmae(np.ones(len(gid))):>9.2f} {rmae(sig_type):>8.2f} {rmae(sig_full_all):>9.2f}")

# 外推划分: 训练 <=6 重原子, 测试 7-9 重原子
print("\n=== 外推测试: 训练 nheavy<=6, 测试 nheavy>=7 ===")
trx=(nheavy<=6); tex=(nheavy>=7)
scx=StandardScaler(); Xa=scx.fit_transform(X[trx]); Xb=scx.transform(X[tex])
mx=RidgeCV(alphas=alphas); mx.fit(Xa,y[trx])
sigx=np.ones(len(gid)); sigx[tex]=np.exp(mx.predict(Xb))
mm=tex
r0=np.mean(np.abs(nu0[mm]/nuref[mm]-1))*100
r1=np.mean(np.abs(nu0[mm]*sigx[mm]/nuref[mm]-1))*100
print(f"  n_test={mm.sum()}  R1原始 relMAE={r0:.2f}%  线性外推 relMAE={r1:.2f}%")
