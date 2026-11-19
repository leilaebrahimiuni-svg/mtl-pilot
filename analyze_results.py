"""Statistics reported in the manuscript (Section 4.4, Table 2), computed from
results/results_manuscript_protocol.json."""
import json, numpy as np
from scipy import stats
R = json.load(open('results/results_manuscript_protocol.json')); k = len(R)
T = ['migration', 'cytotoxicity', 'antimicrobial']
fac = 1 / k + 0.15 / 0.70            # Nadeau-Bengio factor, nominal test/train group ratio
out, ps = {}, []
for t in T:
    s = np.array([r['single'][t] for r in R]); m = np.array([r['multi'][t] for r in R]); g = np.array([r['gbr'][t] for r in R])
    d = m - s; se = np.sqrt(fac * d.var(ddof=1)); p = 2 * stats.t.sf(abs(d.mean() / se), k - 1); h = stats.t.ppf(.975, k - 1) * se
    out[t] = dict(single_mean=s.mean(), single_sd=s.std(ddof=1), multi_mean=m.mean(), multi_sd=m.std(ddof=1), delta=d.mean(),
                  ci_low=d.mean() - h, ci_high=d.mean() + h, pct_low=100 * (d.mean() - h) / s.mean(), pct_high=100 * (d.mean() + h) / s.mean(),
                  p_nadeau_bengio=p, p_wilcoxon=stats.wilcoxon(m, s).pvalue, multi_worse=int((d > 0).sum()), n_splits=k,
                  gbr_mean=g.mean(), nn_vs_gbr_pct=100 * (s.mean() - g.mean()) / g.mean(),
                  mean_test_rows=float(np.mean([r['n_test_rows'][t] for r in R])), mean_test_groups=float(np.mean([r['n_test_groups'][t] for r in R])))
    ps.append(p)
order = np.argsort(ps); run = 0
for rank, j in enumerate(order):
    run = max(run, min(1, (len(T) - rank) * ps[j])); out[T[j]]['p_holm'] = run
json.dump(out, open('results/summary_statistics.json', 'w'), indent=1)
for t in T: print(t, {a: round(b, 3) for a, b in out[t].items()})
