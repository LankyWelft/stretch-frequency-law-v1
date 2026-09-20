# -*- coding: utf-8 -*-
"""V7-R14 审稿补充实验 (rebuttal)
实验A: 在 VIBFREQ1295 的 307 个人工指派模式上复现"可学性"分箱方差分解,
       与 QM9 的 84.1% 口径对照 (人工指派 -> 不含算法指派噪声)。
实验B: (log mu, log R, log BO) 空间的共线性与指数可辨识性:
       OLS / 岭回归正则路径 / 留一键型 (leave-one-bond-type-out) 稳定性,
       对照 R1 锁定的物理指数 (-0.527,-0.856,0.680)。
"""
import os, numpy as np, pandas as pd
from collections import defaultdict

ROOT = '/home/user/Doubao/chats/38438738864945410'
BASE = os.path.join(ROOT, 'V7-R10')

def banner(t): print('\n' + '=' * 74 + f'\n{t}\n' + '=' * 74)

# ============================================================
# 实验 A: 外部 307 人工指派模式的可学性分解
# ============================================================
banner('实验A  VIBFREQ1295 307 人工指派模式 (参考=CCSD(T)-F12c)')
df = pd.read_csv(os.path.join(ROOT, 'stretch-frequency-law/examples/si_modes.csv'))
df['e'] = np.log(df['nu_R1'] / df['nu_CCSD(T)'])          # >0: R1 偏高
def btype(m):
    toks = [t for t in m.replace(' stretch', '').split()
            if t.lower() not in ('sym', 'asym', 'symmetric', 'antisymmetric')]
    return toks[0]
def sym(m):
    ml = m.lower()
    if 'asym' in ml: return 'asym'
    if 'sym' in ml:  return 'sym'
    return '-'
df['bt'] = df['Mode'].map(btype)
df['sy'] = df['Mode'].map(sym)

e = df['e'].values
print(f'n={len(df)}  bias={e.mean():+.4f}  total std={e.std():.4f}  '
      f'relMAE={np.mean(np.abs(np.exp(e)-1))*100:.2f}%')

def within_std(keys, minn):
    bucket = defaultdict(list)
    for k, ek in zip(keys, e):
        bucket[k].append(ek)
    resid = np.concatenate([np.array(v) - np.mean(v) for v in bucket.values()
                            if len(v) >= minn])
    used = sum(len(v) for v in bucket.values() if len(v) >= minn)
    return resid.std(), used, len([v for v in bucket.values() if len(v) >= minn])

for minn in (5, 3):
    w, used, nb = within_std(df['bt'].values, minn)
    frac = 1 - (w / e.std()) ** 2
    print(f'[键型分桶 n>={minn}] 组内 std={w:.4f}  覆盖 {used}/{len(df)} 模式, '
          f'{nb} 桶  可缩减方差比例={frac*100:.1f}%')
# 更细: 键型 x 对称/反对称 (直接对应 per-carbon 输出无法分辨的劈裂)
for minn in (5, 3):
    keys = list(zip(df['bt'], df['sy']))
    w, used, nb = within_std(keys, minn)
    frac = 1 - (w / e.std()) ** 2
    print(f'[键型 x sym/asym n>={minn}] 组内 std={w:.4f}  覆盖 {used}/{len(df)}, '
          f'{nb} 桶  可缩减方差比例={frac*100:.1f}%')

# 仅"有监督"键型 (R1+MPNN 在 QM9 训练过的: X-H + 多重键), 排除重-重单键/卤素
sup = {'C-H', 'N-H', 'O-H', 'C=C', 'C#C', 'C=N', 'C#N', 'C=O', 'N=O', 'N=N'}
m = df['bt'].isin(sup).values
es = e[m]
print(f'\n仅 QM9 有监督键型: n={m.sum()}  bias={es.mean():+.4f}  std={es.std():.4f} '
      f'relMAE={np.mean(np.abs(np.exp(es)-1))*100:.2f}%')
for minn in (5, 3):
    bucket = defaultdict(list)
    for k, ek in zip(df['bt'].values[m], es):
        bucket[k].append(ek)
    resid = np.concatenate(
        [np.array(v) - np.mean(v) for v in bucket.values() if len(v) >= minn])
    frac = 1 - (resid.std() / es.std()) ** 2
    print(f'  [键型分桶 n>={minn}] 组内 std={resid.std():.4f}  可缩减比例={frac*100:.1f}%')

# ============================================================
# 实验 B: 共线性 + 指数可辨识性
# ============================================================
banner('实验B  (log mu, log R, log BO) 共线性与指数稳定性 (QM9 训练集=奇数gid)')
d = np.load(os.path.join(BASE, 'records_aug.npz'))
gid = d['gid']; tr = (gid % 2 == 1)
lmu = np.log(d['mu'].astype(float)); lR = np.log(d['Rij'].astype(float))
lbo = np.log(d['bo'].astype(float).clip(1.0)); y = np.log(d['nuref'].astype(float))
X = np.column_stack([lmu, lR, lbo])
names = ['log mu', 'log R', 'log BO']
Xtr, ytr = X[tr], y[tr]
print(f'训练记录 {tr.sum()}')

# 共线性: 相关系数 + 每个键型内 mu,R 是否常数 (有效自由度)
corr = np.corrcoef(lmu, lR)[0, 1]
print(f'corr(log mu, log R) = {corr:.4f}')
# 键型质心数 (有效设计点)
zi, zj, bo, fam = d['zi'].astype(int), d['zj'].astype(int), d['bo'].astype(int), d['fam'].astype(int)
key = fam * 1000000 + zi * 1000 + zj * 10 + bo
ncent = len(np.unique(key[tr]))
print(f'不同 (元素对,键级) 组合数 = {ncent}  (mu,R,BO 在组合内为常数 -> 有效设计点)')

def ols_fit(Xb, yb):
    A = np.column_stack([Xb, np.ones(len(Xb))])
    coef, *_ = np.linalg.lstsq(A, yb, rcond=None)
    pred = A @ coef
    r2 = 1 - ((yb - pred) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()
    return coef[:3], coef[3], r2

c, icpt, r2 = ols_fit(Xtr, ytr)
print(f'\n全训练集 OLS: 指数 mu={c[0]:+.3f}  R={c[1]:+.3f}  BO={c[2]:+.3f}  '
      f' intercept={icpt:.2f} (前因子={np.exp(icpt):.0f})  R2={r2:.4f}')
print('R1 锁定物理指数:              mu=-0.527  R=-0.856  BO=+0.680  前因子=2962')

# 岭回归正则路径 (标准化后拟合, 转回原始尺度指数)
mu_, sd = Xtr.mean(0), Xtr.std(0)
Xs = (Xtr - mu_) / sd
print('\n岭回归正则路径 (原始尺度指数):')
print(f'{"alpha":>10} {"mu":>8} {"R":>8} {"BO":>8}')
for alpha in [0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4]:
    if alpha == 0:
        bs, *_ = np.linalg.lstsq(np.column_stack([Xs, np.ones(len(Xs))]), ytr, rcond=None)
        bstd = bs[:3]
    else:
        M = Xs.T @ Xs + alpha * np.eye(3)
        bstd = np.linalg.solve(M, Xs.T @ ytr)
    braw = bstd / sd
    print(f'{alpha:>10.0e} {braw[0]:>8.3f} {braw[1]:>8.3f} {braw[2]:>8.3f}')

# 留一键型: 逐折剔除一个主要键型后 OLS, 看指数跨折波动
keys = pd.Series(key[tr])
counts = keys.value_counts()
big = counts[counts > 5000].index.tolist()
fold = []
for kk in big:
    keep = (keys != kk).values
    cf, _, _ = ols_fit(Xtr[keep], ytr[keep])
    fold.append(cf)
fold = np.array(fold)
print(f'\n留一键型 OLS ({len(big)} 折, 每折剔除一个 n>5000 的键型):')
for j, nm in enumerate(names):
    print(f'  指数 {nm:7s}: 均值 {fold[:,j].mean():+.3f}  折间 std {fold[:,j].std():.3f}  '
          f'范围 [{fold[:,j].min():+.3f},{fold[:,j].max():+.3f}]')
print('  对照 R1 锁定: mu -0.527, R -0.856, BO +0.680')
