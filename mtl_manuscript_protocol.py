"""Pilot re-implemented with the protocol described in manuscript Section 4.
NumPy implementation (manual backprop + Adam, torch-style L2 weight decay) because PyTorch
was unavailable in this environment. Gradients are verified by finite differences (gradcheck()).
"""
import json, re, sys, numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor

H, D, EPOCHS, LR, WD = 16, 24, 1000, 1e-3, 1e-4
NSPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 20

# ---------- data ----------
v = pd.read_csv('data/vaferi_migration_data.csv'); d = pd.read_csv('data/dechakupt_antimicrobial_data.csv')
c = pd.read_csv('data/park_cytotoxicity_data.csv')
def onehot(df, cols):
    out = []
    for k in cols:
        s = df[k].fillna('NA').astype(str)
        for cat in sorted(s.unique()): out.append((s == cat).values.astype(float)[:, None])
    return np.hstack(out)
def total_ratio(x):
    if pd.isna(x): return 0.0
    return float(sum(float(t) for t in re.findall(r'\d+\.?\d*', str(x))))
Xm = np.hstack([v[['etoh_eq', 'logKow', 'temp']].values.astype(float), onehot(v, ['material_code'])]); ym = v['target_logK'].values.astype(float)
gm = v['logKow'].astype(str).values
xc = [f'x{i}' for i in range(1, 22)]
Xa = d[xc].values.astype(float); ya = d['y'].values.astype(float); ga = d[xc].astype(str).agg('|'.join, axis=1).values
cat = ['Core', 'Doping', 'Shell', 'Coating and Modification(Ligand and Surface modification chemical)', 'Cell-identification',
       'Cell-anatomical-type', 'Cell-source-species', 'Cell-origin', 'Cell-tissue-organ-origin']
Xc = np.hstack([np.c_[pd.to_numeric(c['Diameter(nm)'], errors='coerce').values, c['Doping Ratio(%)'].map(total_ratio).values], onehot(c, cat)]).astype(float)
yc = c['pIC50'].values.astype(float)
gc = c[['Core', 'Doping', 'Doping Ratio(%)', 'Shell', 'Coating and Modification(Ligand and Surface modification chemical)', 'Diameter(nm)']].fillna('NA').astype(str).agg('|'.join, axis=1).values
TASKS = {'migration': (Xm, ym, gm), 'cytotoxicity': (Xc, yc, gc), 'antimicrobial': (Xa, ya, ga)}
ORDER = list(TASKS)
assert not np.isnan(Xc).any()
print({t: (X.shape, len(set(g))) for t, (X, y, g) in TASKS.items()})

def split(g, seed):
    rng = np.random.RandomState(seed); u = np.array(sorted(set(g))); u = u[rng.permutation(len(u))]
    n = len(u); nt = int(round(.15 * n)); nv = nt
    te, va = set(u[:nt]), set(u[nt:nt + nv])
    return (np.array([x not in te and x not in va for x in g]), np.array([x in va for x in g]), np.array([x in te for x in g]))

# ---------- tiny NN with manual backprop ----------
def lin_init(rng, i, o):
    b = 1 / np.sqrt(i); return rng.uniform(-b, b, (i, o)), rng.uniform(-b, b, o)
def init_task(rng, i):
    We, be = lin_init(rng, i, H); Wh, bh = lin_init(rng, D, 1); return dict(We=We, be=be, Wh=Wh, bh=bh)
def init_trunk(rng):
    W1, b1 = lin_init(rng, H, D); W2, b2 = lin_init(rng, D, D); return dict(W1=W1, b1=b1, W2=W2, b2=b2)
def fwd(tp, tr, X):
    h = X @ tp['We'] + tp['be']; z1 = h @ tr['W1'] + tr['b1']; a1 = np.maximum(z1, 0)
    z2 = a1 @ tr['W2'] + tr['b2']; a2 = np.maximum(z2, 0); o = (a2 @ tp['Wh'] + tp['bh'])[:, 0]
    return o, (X, h, z1, a1, z2, a2)
def bwd(tp, tr, cache, o, y):
    X, h, z1, a1, z2, a2 = cache; n = len(y); go = (2 / n) * (o - y)[:, None]
    gt = dict(Wh=a2.T @ go, bh=go.sum(0)); ga2 = go @ tp['Wh'].T; gz2 = ga2 * (z2 > 0)
    gtr = dict(W2=a1.T @ gz2, b2=gz2.sum(0)); ga1 = gz2 @ tr['W2'].T; gz1 = ga1 * (z1 > 0)
    gtr.update(W1=h.T @ gz1, b1=gz1.sum(0)); gh = gz1 @ tr['W1'].T
    gt.update(We=X.T @ gh, be=gh.sum(0)); return gt, gtr
class Adam:
    def __init__(s): s.m = {}; s.v = {}; s.t = 0
    def step(s, params, grads):  # params, grads: dict name->(array), updated in place
        s.t += 1
        for k, p in params.items():
            g = grads[k] + WD * p
            m = s.m.get(k, 0) * .9 + .1 * g; vv = s.v.get(k, 0) * .999 + .001 * g * g; s.m[k], s.v[k] = m, vv
            p -= LR * (m / (1 - .9 ** s.t)) / (np.sqrt(vv / (1 - .999 ** s.t)) + 1e-8)

def gradcheck():
    rng = np.random.RandomState(0); tp = init_task(rng, 5); tr = init_trunk(rng); X = rng.randn(7, 5); y = rng.randn(7)
    def L(): o, _ = fwd(tp, tr, X); return ((o - y) ** 2).mean()
    o, ca = fwd(tp, tr, X); gt, gtr = bwd(tp, tr, ca, o, y); worst = 0
    for grads, pars in ((gt, tp), (gtr, tr)):
        for k in grads:
            idx = np.ndindex(pars[k].shape)
            for ix in list(idx)[:6]:
                old = pars[k][ix]; pars[k][ix] = old + 1e-6; a = L(); pars[k][ix] = old - 1e-6; b = L(); pars[k][ix] = old
                worst = max(worst, abs((a - b) / 2e-6 - grads[k][ix]))
    return worst

def run_split(seed):
    idx = {t: split(TASKS[t][2], 1000 + seed) for t in ORDER}
    data = {}
    for t in ORDER:
        X, y, g = TASKS[t]; tr, va, te = idx[t]
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1; ym_, ys = y[tr].mean(), y[tr].std()
        Z = (X - mu) / sd; data[t] = dict(Xtr=Z[tr], ytr=(y[tr] - ym_) / ys, Xva=Z[va], yva=y[va], Xte=Z[te], yte=y[te], m=ym_, s=ys,
                                         n=(tr.sum(), va.sum(), te.sum()), ng=(len({*g[te]}),))
    def evaluate(t, tp, trk, rec):
        dd = data[t]
        ov, _ = fwd(tp, trk, dd['Xva']); ot, _ = fwd(tp, trk, dd['Xte'])
        rec.append((np.sqrt(((ov * dd['s'] + dd['m'] - dd['yva']) ** 2).mean()), np.sqrt(((ot * dd['s'] + dd['m'] - dd['yte']) ** 2).mean())))
    out = {'single': {}, 'multi': {}, 'gbr': {}, 'n_test_rows': {t: int(data[t]['n'][2]) for t in ORDER}, 'n_test_groups': {t: data[t]['ng'][0] for t in ORDER}, 'n_train_rows': {t: int(data[t]['n'][0]) for t in ORDER}, 'best_epoch': {'single': {}, 'multi': {}}}
    # single-task
    for k, t in enumerate(ORDER):
        rng = np.random.RandomState(seed * 10 + k); tp = init_task(rng, data[t]['Xtr'].shape[1]); trk = init_trunk(rng); opt = Adam(); rec = []
        for e in range(EPOCHS):
            o, ca = fwd(tp, trk, data[t]['Xtr']); gt, gtr = bwd(tp, trk, ca, o, data[t]['ytr'])
            params = {**{('t', k_): v_ for k_, v_ in tp.items()}, **{('s', k_): v_ for k_, v_ in trk.items()}}
            grads = {**{('t', k_): v_ for k_, v_ in gt.items()}, **{('s', k_): v_ for k_, v_ in gtr.items()}}
            opt.step(params, grads); evaluate(t, tp, trk, rec)
        r = np.array(rec); b = int(r[:, 0].argmin()); out['single'][t] = float(r[b, 1]); out['best_epoch']['single'][t] = b + 1
    # multi-task: one Adam step per epoch on the sum of the three task losses
    rng = np.random.RandomState(seed * 10 + 9); tps = {t: init_task(rng, data[t]['Xtr'].shape[1]) for t in ORDER}; trk = init_trunk(rng); opt = Adam(); rec = {t: [] for t in ORDER}
    for e in range(EPOCHS):
        gtr_sum = {k_: 0 for k_ in trk}; grads = {}; params = {}
        for t in ORDER:
            o, ca = fwd(tps[t], trk, data[t]['Xtr']); gt, gtr = bwd(tps[t], trk, ca, o, data[t]['ytr'])
            for k_ in gtr: gtr_sum[k_] = gtr_sum[k_] + gtr[k_]
            for k_ in gt: params[(t, k_)] = tps[t][k_]; grads[(t, k_)] = gt[k_]
        for k_ in trk: params[('s', k_)] = trk[k_]; grads[('s', k_)] = gtr_sum[k_]
        opt.step(params, grads)
        for t in ORDER: evaluate(t, tps[t], trk, rec[t])
    for t in ORDER:
        r = np.array(rec[t]); b = int(r[:, 0].argmin()); out['multi'][t] = float(r[b, 1]); out['best_epoch']['multi'][t] = b + 1
    for t in ORDER:
        dd = data[t]; m = GradientBoostingRegressor(random_state=seed).fit(dd['Xtr'], dd['ytr'] * dd['s'] + dd['m'])
        out['gbr'][t] = float(np.sqrt(((m.predict(dd['Xte']) - dd['yte']) ** 2).mean()))
    return out

if __name__ == '__main__':
    print('gradcheck max abs diff:', gradcheck())
    res = []
    for s in range(NSPLIT):
        res.append(run_split(s)); print(s, {t: (round(res[-1]['single'][t], 3), round(res[-1]['multi'][t], 3)) for t in ORDER}, flush=True)
    import os; os.makedirs('results', exist_ok=True)
    json.dump(res, open('results/results_manuscript_protocol.json', 'w'), indent=1)
