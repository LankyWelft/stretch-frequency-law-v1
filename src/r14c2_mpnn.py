# -*- coding: utf-8 -*-
"""V7-R14 实验C2: 消息传递 GNN 的 Δ-学习 (多跳化学环境)
重原子图, 3层边条件消息传递; 多重键边目标 + X-H 节点目标。
无 PyG, 手写大图拼接 + scatter。分子级奇偶划分。"""
import os, glob, math, time, json
import numpy as np, networkx as nx
import torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v7_qm9_lib import BASE, QM9_DIR, BAD_IDS_FILE, PRIOR_FILE, SYM, MASS, R_V6, reduced_mass, r1_nu, parse_xyz

torch.manual_seed(0); np.random.seed(0)
LIMIT=int(os.environ.get('R14_LIMIT','0'))
EPOCHS=int(os.environ.get('R14_EPOCHS','25'))

def _pair(a,b): return (min(a,b),max(a,b))
_THR={_pair(6,6):(1.27,1.44,1.70),_pair(6,7):(1.22,1.38,1.70),_pair(6,8):(1.17,1.32,1.60),
 _pair(7,7):(1.18,1.35,1.70),_pair(7,8):(1.14,1.31,1.60),_pair(8,8):(1.10,1.28,1.60),
 _pair(6,9):(0.,1.,1.7),_pair(7,9):(0.,1.,1.7),_pair(8,9):(0.,1.,1.7),_pair(9,9):(0.,1.,2.)}
def bo_dist(za,zb,d):
    t=_THR.get(_pair(za,zb))
    if not t: return 0
    t3,t2,t1=t
    if t3>0 and d<t3: return 3
    if d<t2: return 2
    if d<t1: return 1
    return 0
PRIOR=json.load(open(PRIOR_FILE)) if os.path.exists(PRIOR_FILE) else {}
def _mlab(za,zb,o):
    sym={2:'=',3:'#'}.get(o,'-')
    return f"{SYM[min(za,zb)]}{sym}{SYM[max(za,zb)]}"
ZINNER={1:0,6:2,7:2,8:2,9:2}
def chi(z): return (z-ZINNER[z]/math.pi)/R_V6[z]
ZIDX={6:0,7:1,8:2,9:3}
RING_LEVELS=[0,3,4,5,6]

# ---------- 预处理: 构图 + 指派 ----------
mols=[]  # 每个分子 dict
files=sorted(glob.glob(os.path.join(QM9_DIR,'*.xyz')))
if LIMIT: files=files[:LIMIT]
bad=set(int(l) for l in open(BAD_IDS_FILE) if l.strip().isdigit())
t0=time.time()
for f in files:
    gid,atoms,coords,freqs,smi=parse_xyz(f)
    if gid in bad: continue
    freqs=np.array(freqs)
    if np.any(freqs<=0): continue
    Zn=[{'H':1,'C':6,'N':7,'O':8,'F':9}[a] for a in atoms]
    heavy=[i for i,z in enumerate(Zn) if z!=1]; hmap={g:k for k,g in enumerate(heavy)}
    nh=len(heavy)
    G=nx.Graph(); G.add_nodes_from(range(nh))
    edges=[]; bo_mat={}
    for a in range(nh):
        for b in range(a+1,nh):
            i,j=heavy[a],heavy[b]; d=float(np.linalg.norm(coords[i]-coords[j]))
            bo=bo_dist(Zn[i],Zn[j],d)
            if bo>=1:
                G.add_edge(a,b,bo=bo); bo_mat[(a,b)]=bo
                mu=reduced_mass(Zn[i],Zn[j]); Rij=math.sqrt(R_V6[Zn[i]]*R_V6[Zn[j]])
                edges.append((a,b,Zn[i],Zn[j],bo,mu,Rij,r1_nu(mu,Rij,bo)))
    hcnt=[0]*nh; Hpair=[]
    hc=coords[heavy]
    for h in range(len(Zn)):
        if Zn[h]==1:
            k=int(np.argmin(np.linalg.norm(hc-coords[h],axis=1))); hcnt[k]+=1
            mu=reduced_mass(1,Zn[heavy[k]]); Rij=math.sqrt(R_V6[1]*R_V6[Zn[heavy[k]]])
            Hpair.append((k,Zn[heavy[k]],r1_nu(mu,Rij,1.0)))
    # 环
    rsize=[0]*nh; arom=[0]*nh
    try: cycles=nx.minimum_cycle_basis(G)
    except Exception: cycles=[]
    for cyc in cycles:
        rs=len(cyc); nd=sum(1 for a in range(rs) for b in range(a+1,rs) if bo_mat.get((min(cyc[a],cyc[b]),max(cyc[a],cyc[b])),0)==2)
        isa=1 if(rs==6 and nd>=2) else 0
        for nd_ in cyc:
            if rsize[nd_]==0 or rs<rsize[nd_]: rsize[nd_]=rs
            if isa: arom[nd_]=1
    # 节点特征
    deg=[0]*nh; bosum=[0]*nh; maxbo=[1]*nh; adjN=[0]*nh; adjO=[0]*nh
    for (a,b),v in bo_mat.items():
        deg[a]+=1;deg[b]+=1;bosum[a]+=v;bosum[b]+=v
        maxbo[a]=max(maxbo[a],v);maxbo[b]=max(maxbo[b],v)
        if Zn[heavy[b]]==7: adjN[a]+=1
        if Zn[heavy[a]]==7: adjN[b]+=1
        if Zn[heavy[b]]==8: adjO[a]+=1
        if Zn[heavy[a]]==8: adjO[b]+=1
    nf=[]
    for a in range(nh):
        z=Zn[heavy[a]]; hyb=3 if maxbo[a]==3 else(2 if maxbo[a]==2 else 1)
        row=[1 if ZIDX.get(z,-1)==k else 0 for k in range(4)]
        row+=[hcnt[a]/4,deg[a]/4,bosum[a]/4]+[1 if hyb==k else 0 for k in (1,2,3)]
        row+=[1 if rsize[a]>0 else 0,arom[a]]+[1 if rsize[a]==r else 0 for r in RING_LEVELS]
        row+=[chi(z)/10,adjN[a]/3,adjO[a]/3]
        nf.append(row)
    nf=np.array(nf,dtype=np.float32)
    # 边特征 + 多重键目标
    ei=[]; ef=[]; mult_idx=[]; mult_y=[]; mult_lab=[]; mult_ref=[]
    def conjugated(a,b,bo):
        def mc(x,ex): return any(bo_mat.get((min(x,y),max(x,y)),0)>=2 for y in G.neighbors(x) if y!=ex)
        if bo>=2: return 1 if(mc(a,b) or mc(b,a)) else 0
        return 1 if(mc(a,b) and mc(b,a)) else 0
    edge_pos={}
    for p,(a,b,za,zb,bo,mu,Rij,nup) in enumerate(edges):
        ei.append([a,b]);ei.append([b,a])
        row=[1 if bo==k else 0 for k in(1,2,3)]+[math.log(mu),math.log(Rij),math.log(nup),
             conjugated(a,b,bo),1 if rsize[a]>0 and rsize[a]==rsize[b] else 0]+[1 if rsize[a]==r and rsize[b]==r else 0 for r in RING_LEVELS]
        ef.append(row);ef.append(row); edge_pos[(a,b)]=p
    # 指派多重键
    q=np.sort(freqs)
    for order,(LO,HI) in {3:(1850.,2500.),2:(1350.,1900.)}.items():
        mb=[e for e in edges if e[4]==order]
        if mb:
            cand=q[(q>=LO)&(q<=HI)]
            if len(cand)>=len(mb):
                Pm=np.array([e[7]*PRIOR.get(_mlab(e[2],e[3],order),1.) for e in mb])
                cost=np.abs(np.log(Pm[:,None]/cand[None,:])); ri,ci=linear_sum_assignment(cost)
                for aa,bb in zip(ri,ci):
                    if cost[aa,bb]<math.log(1.25):
                        a,b,za,zb,bo2,mu,Rij,nup=mb[aa]
                        mult_idx.append(2*edge_pos[(a,b)]); mult_y.append(math.log(cand[bb]/nup))
                        mult_ref.append(float(cand[bb]))
                        sym={2:'=',3:'#'}.get(order,'-')
                        mult_lab.append(f"{SYM[min(za,zb)]}{sym}{SYM[max(za,zb)]}")
    # X-H 节点目标(同原子多次指派取均值)
    nH=len(Hpair)
    if nH>0:
        qh=np.sort(freqs)[-nH:]
        if len(Hpair)==nH and qh.min()>=2500:
            ps=sorted(Hpair,key=lambda t:t[2]); qs=sorted(qh)
            agg={}; aggref={}
            for (k,zX,nup),nuq in zip(ps,qs):
                agg.setdefault(k,[]).append(math.log(nuq/nup)); aggref.setdefault(k,[]).append(float(nuq))
            xh_node=list(agg.keys()); xh_y=[float(np.mean(v)) for v in agg.values()]
            xh_ref=[float(np.mean(aggref[k])) for k in xh_node]
            xh_elem=np.array([Zn[heavy[k]] for k in xh_node],dtype=np.int64)
        else: xh_node=[];xh_y=[];xh_ref=[];xh_elem=np.array([],dtype=np.int64)
    else: xh_node=[];xh_y=[];xh_ref=[];xh_elem=np.array([],dtype=np.int64)
    if mult_idx or xh_node:
        mols.append(dict(gid=gid,nf=nf,ei=np.array(ei).T if ei else np.zeros((2,0),int),
                         ef=np.array(ef,dtype=np.float32) if ef else np.zeros((0,14),np.float32),
                         mi=np.array(mult_idx,int),my=np.array(mult_y,np.float32),ml=mult_lab,
                         mr=np.array(mult_ref,np.float32),
                         xn=np.array(xh_node,int),xy=np.array(xh_y,np.float32),xr=np.array(xh_ref,np.float32),
                         xe=xh_elem,nh=len(heavy)))
    if len(mols)%20000==0 and mols: print(f'  graph {len(mols)} {time.time()-t0:.0f}s',flush=True)
print(f'graphs built: {len(mols)} {time.time()-t0:.0f}s')

# ---------- MPNN ----------
FN=20; FE=13; H=64
class MPNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.vin=nn.Linear(FN,H)
        s.msg=nn.ModuleList([nn.Sequential(nn.Linear(2*H+FE,H),nn.ReLU(),nn.Linear(H,H)) for _ in range(3)])
        s.up=nn.ModuleList([nn.Linear(H,H) for _ in range(3)])
        s.ln=nn.ModuleList([nn.LayerNorm(H) for _ in range(3)])
        s.eout=nn.Sequential(nn.Linear(2*H+FE,64),nn.ReLU(),nn.Linear(64,1))
        s.nout=nn.Sequential(nn.Linear(H,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,nf,ei,ef,nb):
        h=s.vin(nf)
        for L in range(3):
            hs=torch.cat([h[ei[0]],h[ei[1]],ef],1)
            m=s.msg[L](hs)
            agg=torch.zeros_like(h).index_add_(0,ei[1],m)
            deg=torch.zeros(h.shape[0]).index_add_(0,ei[1],torch.ones(ei.shape[1])).clamp(min=1).unsqueeze(1)
            h=s.ln[L](h+torch.relu(s.up[L](agg/deg)))
        return s.eout(torch.cat([h[ei[0]],h[ei[1]],ef],1)).squeeze(-1), s.nout(h).squeeze(-1)

def batch(mols_sub):
    nfoff=0; NF_=[];EI_=[];EF_=[];MI_=[];MY_=[];ML_=[];XN_=[];XY_=[];GB_=[];EB_=[]
    for k,m in enumerate(mols_sub):
        NF_.append(m['nf']); GB_.append(np.full(m['nf'].shape[0],k))
        if m['ei'].shape[1]:
            EI_.append(m['ei']+nfoff); EF_.append(m['ef']); EB_.append(np.full(m['ei'].shape[1],k))
            MI_.append(m['mi']+ (nfoff*0))  # edge index is local to edge array
        # edge offsets need tracking
        nfoff+=m['nf'].shape[0]
    return None

# 简单逐分子训练(分子小, 向量化节点/边拼接)
def make_tensors(mols_sub):
    NF=[];EI=[];EF=[];MI=[];MY=[];ML=[];XN=[];XY=[];XE=[]
    noff=0; eoff=0
    for m in mols_sub:
        NF.append(m['nf'])
        if m['ei'].shape[1]:
            EI.append(m['ei']+noff); EF.append(m['ef'])
            MI.append(m['mi']+eoff)
        XN.append(m['xn']+noff)
        noff+=m['nf'].shape[0]; eoff+= (m['ei'].shape[1] if m['ei'].shape[1] else 0)
        MY.append(m['my']); ML+=m['ml']; XY.append(m['xy']); XE.append(m['xe'])
    return (torch.tensor(np.vstack(NF),dtype=torch.float32),
            torch.tensor(np.hstack(EI),dtype=torch.long) if EI else torch.zeros(2,0,dtype=torch.long),
            torch.tensor(np.vstack(EF),dtype=torch.float32) if EF else torch.zeros(0,FE),
            torch.tensor(np.concatenate([x for x in MI if len(x)]),dtype=torch.long) if MI else torch.zeros(0,dtype=torch.long),
            torch.tensor(np.concatenate([x for x in MY if len(x)]),dtype=torch.float32) if MY else torch.zeros(0),
            ML,
            torch.tensor(np.concatenate([x for x in XN if len(x)]),dtype=torch.long) if XN else torch.zeros(0,dtype=torch.long),
            torch.tensor(np.concatenate([x for x in XY if len(x)]),dtype=torch.float32) if XY else torch.zeros(0),
            np.concatenate([x for x in XE if len(x)]).astype(int) if any(len(x) for x in XE) else np.array([],int))

SPLIT=os.environ.get('R14_SPLIT','indist')
if SPLIT=='extrap':
    tr_mols=[m for m in mols if m['nh']<=6]
    te_mols=[m for m in mols if m['nh']>=7]
    print(f'外推划分: train nheavy<=6 ={len(tr_mols)}, test nheavy>=7 ={len(te_mols)}')
else:
    tr_mols=[m for m in mols if m['gid']%2==1]
    te_mols=[m for m in mols if m['gid']%2==0]
net=MPNN(); opt=torch.optim.Adam(net.parameters(),lr=2e-3); lf=nn.MSELoss()
BS=256
def evaluate():
    net.eval()
    with torch.no_grad():
        NF,EI,EF,MI,MY,ML,XN,XY,XE=make_tensors(te_mols)
        pe,pn=net(NF,EI,EF,None)
        # 多重键
        em=pe[MI].numpy(); ey=MY.numpy(); em=np.exp(em-ey)
        # X-H
        xm=np.exp(pn[XN].numpy()-XY.numpy())
    mref=np.concatenate([m['mr'] for m in te_mols if len(m['mr'])]) if any(len(m['mr']) for m in te_mols) else np.array([])
    xref=np.concatenate([m['xr'] for m in te_mols if len(m['xr'])]) if any(len(m['xr']) for m in te_mols) else np.array([])
    return em,np.array(ML),xm,mref,xref,XE
t0=time.time()
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(tr_mols)
    tot=0
    for s in range(0,len(tr_mols),BS):
        NF,EI,EF,MI,MY,ML,XN,XY,_=make_tensors(tr_mols[s:s+BS])
        opt.zero_grad()
        pe,pn=net(NF,EI,EF,None)
        l=lf(pe[MI],MY)
        if len(XN): l=l+0.5*lf(pn[XN],XY)
        l.backward(); opt.step(); tot+=l.item()
    if ep%3==0 or ep==EPOCHS-1:
        em,lab,xm,_,_,_=evaluate()
        print(f'  ep{ep:>2} loss={tot/(len(tr_mols)//BS):.4f} 多重键relMAE={np.mean(np.abs(em-1))*100:.2f}% X-H={np.mean(np.abs(xm-1))*100:.2f}% ({time.time()-t0:.0f}s)',flush=True)

em,lab,xm,mref,xref,xelem=evaluate()
lab=np.array(lab)
torch.save(net.state_dict(),os.path.join(BASE,'c2_mpnn_weights_retrained.pt'))
print('saved',os.path.join(BASE,'c2_mpnn_weights_retrained.pt'))
# 保存 parity 数据
np.savez(os.path.join(BASE,'r14c2_pred.npz'),
    mm_ref=mref, mm_pred=em*mref, mm_lab=lab,
    xh_ref=xref, xh_pred=xm*xref)
print('saved r14c2_pred.npz')
print("\n=== C2 MPNN 测试集 ===")
print(f"多重键总体 relMAE={np.mean(np.abs(em-1))*100:.2f}%  n={len(em)}")
print(f"X-H   总体 relMAE={np.mean(np.abs(xm-1))*100:.2f}%  n={len(xm)}")
for t in ['C=C','C#C','C=N','C#N','C=O','N=O']:
    m=lab==t
    if m.sum()>20: print(f"  {t:>4} n={m.sum():>6}: {np.mean(np.abs(em[m]-1))*100:.2f}%")
# X-H 按重原子元素拆分
for z,nm in [(6,'C-H'),(7,'N-H'),(8,'O-H')]:
    m=xelem==z
    if m.sum()>20: print(f"  {nm:>4} n={m.sum():>6}: {np.mean(np.abs(xm[m]-1))*100:.2f}%")
print(f"  X-H总体 n={len(xm)}: {np.mean(np.abs(xm-1))*100:.2f}%")
