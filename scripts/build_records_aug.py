# -*- coding: utf-8 -*-
"""V7-R14 实验A: 增强逐键记录构建
在 R10 指派逻辑基础上, 为每条已指派键附加完整化学环境特征:
环/环大小/芳香、共轭、杂化、alpha-杂原子、分子大小、多重键数等。"""
import os, glob, math, time, json
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from v7_qm9_lib import BASE, QM9_DIR, BAD_IDS_FILE, PRIOR_FILE, SYM, MASS, R_V6, reduced_mass, r1_nu, parse_xyz, _fl

def _pair(a,b): return (min(a,b), max(a,b))
_THR = {
 _pair(6,6):(1.27,1.44,1.70), _pair(6,7):(1.22,1.38,1.70),
 _pair(6,8):(1.17,1.32,1.60), _pair(7,7):(1.18,1.35,1.70),
 _pair(7,8):(1.14,1.31,1.60), _pair(8,8):(1.10,1.28,1.60),
 _pair(6,9):(0.0,1.0,1.70), _pair(7,9):(0.0,1.0,1.70),
 _pair(8,9):(0.0,1.0,1.70), _pair(9,9):(0.0,1.0,2.0),
}
def bo_from_distance(za,zb,d):
    t=_THR.get(_pair(za,zb))
    if t is None: return 0
    t3,t2,t1=t
    if t3>0 and d<t3: return 3
    if d<t2: return 2
    if d<t1: return 1
    return 0

PRIOR = json.load(open(PRIOR_FILE)) if os.path.exists(PRIOR_FILE) else {}
def _mlab(za,zb,order):
    a,b=SYM[min(za,zb)],SYM[max(za,zb)]
    sym={2:'=',3:'#'}.get(order,'-')
    return f"{a}{sym}{b}"

# 输出列
COLS = ['gid','zi','zj','bo','mu','Rij','nu0','nuref','fam',
        'nheavy','nH','nmult','nring',
        'degi','degj','hi','hj','bosumi','bosumj','hybi','hybj',
        'inring','ringsize','arom','conj',
        'adjNi','adjOi','adjNj','adjOj']
R = {c:[] for c in COLS}

def add(gid,zi,zj,bo,mu,Rij,nu0,nuref,fam,env):
    vals=[gid,zi,zj,bo,mu,Rij,nu0,nuref,fam]+env
    for c,v in zip(COLS,vals): R[c].append(v)

files=sorted(glob.glob(os.path.join(QM9_DIR,'*.xyz')))
LIMIT = int(os.environ.get('R14_LIMIT','0'))
if LIMIT: files=files[:LIMIT]
bad=set()
with open(BAD_IDS_FILE) as fh:
    for ln in fh:
        if ln.strip().isdigit(): bad.add(int(ln))

t0=time.time(); nused=0
for idx,f in enumerate(files):
    gid,atoms,coords,freqs,smi=parse_xyz(f)
    if gid in bad: continue
    freqs=np.array(freqs)
    if np.any(freqs<=0): continue
    Zn=[{'H':1,'C':6,'N':7,'O':8,'F':9}[a] for a in atoms]
    n=len(Zn); heavy=[i for i,z in enumerate(Zn) if z!=1]
    nheavy=len(heavy); nH=n-nheavy

    # 重建重原子图
    G=nx.Graph(); G.add_nodes_from(heavy)
    hb=[]  # (i,j,zi,zj,bo,mu,Rij,nu)
    bo_mat={}
    for ai in range(len(heavy)):
        for aj in range(ai+1,len(heavy)):
            i,j=heavy[ai],heavy[aj]
            d=float(np.linalg.norm(coords[i]-coords[j]))
            bo=bo_from_distance(Zn[i],Zn[j],d)
            if bo>=1:
                G.add_edge(i,j,bo=bo)
                bo_mat[(min(i,j),max(i,j))]=bo
                mu=reduced_mass(Zn[i],Zn[j]); Rij=math.sqrt(R_V6[Zn[i]]*R_V6[Zn[j]])
                hb.append((i,j,Zn[i],Zn[j],bo,mu,Rij,r1_nu(mu,Rij,bo)))
    # H 连接
    h_neigh={i:0 for i in heavy}
    hc=coords[heavy]
    H_assign=[]
    for h in range(n):
        if Zn[h]==1:
            jj=heavy[int(np.argmin(np.linalg.norm(hc-coords[h],axis=1)))]
            h_neigh[jj]+=1
            H_assign.append((h,jj))

    # 环信息: 最小环基
    ring_size_of={i:0 for i in heavy}
    arom_of={i:0 for i in heavy}
    try:
        cycles=nx.minimum_cycle_basis(G)
    except Exception:
        cycles=[]
    for cyc in cycles:
        rs=len(cyc)
        # 芳香: 六元环且环内有>=2个双键
        ndbl=sum(1 for a in range(rs) for b in range(a+1,rs)
                 if bo_mat.get((min(cyc[a],cyc[b]),max(cyc[a],cyc[b])),0)==2)
        is_arom = 1 if (rs==6 and ndbl>=2) else 0
        for node in cyc:
            if ring_size_of[node]==0 or rs<ring_size_of[node]:
                ring_size_of[node]=rs
            if is_arom: arom_of[node]=1
    nring=len(cycles)
    nmult=sum(1 for b in hb if b[4]>=2)

    # 原子级环境
    def atom_env(i):
        neigh=[j for j in G.neighbors(i)]
        deg=len(neigh)
        bosum=sum(bo_mat[(min(i,j),max(i,j))] for j in neigh)
        maxbo=max([bo_mat[(min(i,j),max(i,j))] for j in neigh]+[1])
        hyb = 3 if maxbo==3 else (2 if maxbo==2 else 1)
        adjN=sum(1 for j in neigh if Zn[j]==7)
        adjO=sum(1 for j in neigh if Zn[j]==8)
        return deg,h_neigh.get(i,0),bosum,hyb,adjN,adjO,neigh

    aenv={i:atom_env(i) for i in heavy}

    def is_conjugated(i,j,bo):
        # 该键共轭: 多重键且一端通过单键连到另一个多重键中心; 或单键两端都参与多重键
        def mult_center(x,exclude):
            return any(bo_mat.get((min(x,y),max(x,y)),0)>=2 for y in G.neighbors(x) if y!=exclude)
        if bo>=2:
            return 1 if (mult_center(i,j) or mult_center(j,i)) else 0
        else:
            return 1 if (mult_center(i,j) and mult_center(j,i)) else 0

    def edge_feat(i,j,bo):
        di,hi_,bsi,hyi,ani,aoi,_=aenv[i]
        dj,hj_,bsj,hyj,anj,aoj,_=aenv[j]
        return [nheavy,nH,nmult,nring,
                di,dj,hi_,hj_,bsi,bsj,hyi,hyj,
                1 if ring_size_of[i]>0 and ring_size_of[j]>0 and ring_size_of[i]==ring_size_of[j] else 0,
                ring_size_of[i] if ring_size_of[i]==ring_size_of[j] else 0,
                max(arom_of[i],arom_of[j]),
                is_conjugated(i,j,bo),
                ani,aoi,anj,aoj]

    # ---- X-H 指派 (R10 逻辑) ----
    if nH>0:
        pred=[]
        for (h,jj) in H_assign:
            zX=Zn[jj]
            mu=reduced_mass(1,zX); Rij=math.sqrt(R_V6[1]*R_V6[zX])
            pred.append((jj,zX,r1_nu(mu,Rij,1.0)))
        q=np.sort(freqs)[-nH:]
        if len(pred)==nH and q.min()>=2500:
            ps=sorted(pred,key=lambda t:t[2]); qs=sorted(q)
            for (jj,zX,nup),nuq in zip(ps,qs):
                di,hi_,bsi,hyi,ani,aoi,_=aenv[jj]
                env=[nheavy,nH,nmult,nring,
                     di,0,hi_,0,bsi,0,hyi,0,
                     1 if ring_size_of[jj]>0 else 0, ring_size_of[jj],
                     arom_of[jj], 0, ani,aoi,0,0]
                add(gid,zX,1,1,reduced_mass(1,zX),math.sqrt(R_V6[1]*R_V6[zX]),
                    nup,float(nuq),1,env)

    # ---- 重-重多重键 Hungarian ----
    q=np.sort(freqs)
    for order,(LO,HI) in {3:(1850.,2500.),2:(1350.,1900.)}.items():
        mb=[b for b in hb if b[4]==order]
        if mb:
            cand=q[(q>=LO)&(q<=HI)]
            if len(cand)>=len(mb):
                Pmatch=np.array([bb[7]*PRIOR.get(_mlab(bb[2],bb[3],order),1.0) for bb in mb])
                cost=np.abs(np.log(Pmatch[:,None]/cand[None,:]))
                ri,ci=linear_sum_assignment(cost)
                for a,b in zip(ri,ci):
                    if cost[a,b]<math.log(1.25):
                        ii,jj,za,zb,bo2,mu,Rij,nup=mb[a]
                        env=edge_feat(ii,jj,bo2)
                        add(gid,min(za,zb),max(za,zb),bo2,mu,Rij,nup,float(cand[b]),2,env)
    nused+=1
    if (idx+1)%20000==0: print(f'  {idx+1} files, {time.time()-t0:.0f}s, recs={len(R["gid"])}',flush=True)

dt={k:np.array(v,dtype=(np.int32 if k in('gid',) else np.int16 if k in
    ('zi','zj','bo','fam','nheavy','nH','nmult','nring','degi','degj','hi','hj',
     'bosumi','bosumj','hybi','hybj','inring','ringsize','arom','conj',
     'adjNi','adjOi','adjNj','adjOj') else np.float32)) for k,v in R.items()}
out=os.path.join(BASE,'records_aug.npz')
np.savez(out,**dt)
print('molecules',nused,'records',dt['gid'].size,f'{time.time()-t0:.0f}s')
print('X-H',int((dt['fam']==1).sum()),'multi',int((dt['fam']==2).sum()))
print('saved',out)
