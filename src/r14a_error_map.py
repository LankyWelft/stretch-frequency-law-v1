# -*- coding: utf-8 -*-
"""V7-R14 实验A: 误差结构全景表征
对每条已指派键, 残差 e=log(nu_R1/nu_DFT), 按化学环境分组,
判断残差是系统信号(可学习)还是白噪声(指派/数值噪声)。"""
import os, numpy as np
from collections import defaultdict
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v7_qm9_lib import BASE, SYM

d=np.load(os.path.join(BASE,'records_aug.npz'))
gid=d['gid']; zi=d['zi'].astype(int); zj=d['zj'].astype(int); bo=d['bo'].astype(int)
fam=d['fam'].astype(int); nu0=d['nu0'].astype(float); nuref=d['nuref'].astype(float)
e=np.log(nu0/nuref)  # >0: R1 预测偏高
te=(gid%2==0)  # 分子级 hold-out (偶数 gid 测试)

def blabel(k):
    if fam[k]==1: return f"{SYM[zi[k]]}-H"
    sym={2:'=',3:'#'}.get(bo[k],'-')
    return f"{SYM[zi[k]]}{sym}{SYM[zj[k]]}"
lab=np.array([blabel(k) for k in range(len(gid))])

def grp_stats(mask,name,minn=30):
    m=mask&te
    if m.sum()<minn: return None
    ee=e[m]; r=np.exp(ee)
    return dict(name=name,n=int(m.sum()),bias=float(ee.mean()),std=float(ee.std()),
                med_ratio=float(np.median(r)),
                rel_mae=float(np.mean(np.abs(r-1))*100),
                w10=float(np.mean(np.abs(r-1)<=.10)*100))

def show(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'组':>22} {'n':>7} {'bias(log)':>10} {'std':>7} {'中位ratio':>9} {'relMAE%':>8} {'±10%':>6}")
    seen=0
    for rr in rows:
        if rr is None: continue
        print(f"{rr['name']:>22} {rr['n']:>7} {rr['bias']:>+10.4f} {rr['std']:>7.3f} {rr['med_ratio']:>9.3f} {rr['rel_mae']:>8.2f} {rr['w10']:>5.1f}%")
        seen+=1
    return seen

print(f"总记录 {len(gid)}, 测试集(偶数gid) {te.sum()}")
print(f"测试集总体: bias={e[te].mean():+.4f} std={e[te].std():.3f} relMAE={np.mean(np.abs(np.exp(e[te])-1))*100:.2f}%")

# 1. 逐键型
rows=[]
for t in ['C-H','N-H','O-H','C=C','C#C','C=N','C#N','C=O','N=O','N=N']:
    rows.append(grp_stats(lab==t,t))
show('1. 逐键型残差', rows)

# 2. 环/非环 (仅多重键, X-H 环标记意义弱)
inring=d['inring'].astype(int); rs=d['ringsize'].astype(int)
rows=[grp_stats((fam==2)&(inring==0),'多重键-非环'),
      grp_stats((fam==2)&(inring==1),'多重键-环内')]
for r_ in [3,4,5,6]:
    rows.append(grp_stats((fam==2)&(rs==r_),f'多重键-{r_}元环'))
show('2. 环效应(重-重多重键)', rows)

# 3. 共轭/芳香
conj=d['conj'].astype(int); arom=d['arom'].astype(int)
rows=[grp_stats((fam==2)&(conj==0),'多重键-非共轭'),
      grp_stats((fam==2)&(conj==1),'多重键-共轭'),
      grp_stats((fam==2)&(arom==1),'多重键-芳香环')]
show('3. 共轭/芳香效应', rows)

# C=O 单独看共轭(酰胺/羧酸/酯 vs 酮)
rows=[grp_stats((lab=='C=O')&(conj==0),'C=O 非共轭(酮/醛)'),
      grp_stats((lab=='C=O')&(conj==1),'C=O 共轭(酸/酯/酰胺)')]
show('3b. C=O 共轭分层', rows)

# 4. α-杂原子 (多重键两端邻接 N/O 计数)
adjN=d['adjNi'].astype(int)+d['adjNj'].astype(int)
adjO=d['adjOi'].astype(int)+d['adjOj'].astype(int)
rows=[]
for k in range(0,3):
    rows.append(grp_stats((fam==2)&(adjO==k),f'多重键 α-O数={k}'))
show('4a. α-O 邻接(多重键)', rows)
rows=[]
for k in range(0,3):
    rows.append(grp_stats((fam==2)&(adjN==k),f'多重键 α-N数={k}'))
show('4b. α-N 邻接(多重键)', rows)

# 5. 分子大小分层 (重原子数)
nh=d['nheavy'].astype(int)
rows=[]
for lo,hi,t in [(1,3,'1-2重原子'),(3,5,'3-4重原子'),(5,7,'5-6重原子'),(7,10,'7-9重原子')]:
    rows.append(grp_stats((nh>=lo)&(nh<hi),t))
show('5. 分子大小(全部键)', rows)
rows=[]
for lo,hi,t in [(1,4,'X-H 1-3重'),(4,7,'X-H 4-6重'),(7,10,'X-H 7-9重')]:
    rows.append(grp_stats((fam==1)&(nh>=lo)&(nh<hi),t))
show('5b. X-H 随分子大小', rows)

# 6. X-H 承载原子的杂化/度
hyb=d['hybi'].astype(int); deg=d['degi'].astype(int)
rows=[grp_stats((lab=='C-H')&(hyb==1),'C-H 承载sp3'),
      grp_stats((lab=='C-H')&(hyb==2),'C-H 承载sp2'),
      grp_stats((lab=='C-H')&(hyb==3),'C-H 承载sp')]
show('6a. C-H 承载原子杂化', rows)
rows=[]
for dg in [1,2,3,4]:
    rows.append(grp_stats((lab=='C-H')&(deg==dg),f'C-H 承载原子度={dg}'))
show('6b. C-H 承载原子重原子度', rows)

# 7. 多重键数(分子级共轭程度)
nm=d['nmult'].astype(int)
rows=[]
for k in [1,2,3]:
    rows.append(grp_stats((fam==2)&(nm==k),f'分子多重键数={k}'))
rows.append(grp_stats((fam==2)&(nm>=4),'分子多重键数>=4'))
show('7. 分子多重键总数', rows)

# 8. 可学习性判定: 组间系统bias的方差 vs 组内噪声
print("\n=== 8. 可学习性判定 ===")
# 键型层
type_bias=[]; type_n=[]
for t in ['C-H','N-H','O-H','C=C','C#C','C=N','C#N','C=O']:
    m=te&(lab==t)
    if m.sum()>30: type_bias.append(e[m].mean()); type_n.append(m.sum())
type_bias=np.array(type_bias)
print(f"键型间 bias 范围: {type_bias.min():+.3f} ~ {type_bias.max():+.3f}, 跨度 {type_bias.max()-type_bias.min():.3f} log单位 (~{(np.exp(type_bias.max()-type_bias.min())-1)*100:.0f}% 相对)")
# 环境分层后组内残差缩减的理论上限估计: 用 (键型×共轭×环) 粗分桶, 计算组内std
bucket=defaultdict(list)
for k in np.where(te)[0]:
    key=(lab[k],int(conj[k]),int(inring[k]),int(hyb[k] if fam[k]==1 else 0))
    bucket[key].append(e[k])
within=np.concatenate([np.array(v)-np.mean(v) for v in bucket.values() if len(v)>=20])
print(f"粗环境分桶后组内 std={within.std():.3f} (原始 {e[te].std():.3f})")
print(f"理论可缩减方差比例: {(1-(within.std()/e[te].std())**2)*100:.1f}%")
