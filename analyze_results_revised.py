"""Analysis for results_manuscript_protocol_revised.json.

Uses each split's observed test/train group ratio in the Nadeau-Bengio
corrected resampled t-test. Holm correction is reported separately for the
three NB tests and the three exploratory Wilcoxon tests.
"""
import json
import numpy as np
from scipy import stats

with open("results/results_manuscript_protocol_revised.json") as f:
    R=json.load(f)
tasks=["migration","cytotoxicity","antimicrobial"]
k=len(R)
if k<2: raise ValueError("At least two repeated splits are required.")

def holm(pvalues):
    p=np.asarray(pvalues,float); order=np.argsort(p); adjusted=np.empty_like(p)
    running=0.0; m=len(p)
    for rank,idx in enumerate(order):
        running=max(running,min(1.0,(m-rank)*p[idx]))
        adjusted[idx]=running
    return adjusted

out={}; p_nb=[]; p_w=[]
for task in tasks:
    single=np.asarray([r["single"][task] for r in R],float)
    multi=np.asarray([r["multi"][task] for r in R],float)
    delta=multi-single
    ratios=np.asarray([r["split_info"][task]["test_train_group_ratio"] for r in R],float)
    correction=1.0/k+float(np.mean(ratios))
    se=float(np.sqrt(correction*np.var(delta,ddof=1)))
    if se==0:
        p=1.0 if np.mean(delta)==0 else 0.0
        half=0.0
    else:
        statistic=float(np.mean(delta)/se)
        p=float(2*stats.t.sf(abs(statistic),df=k-1))
        half=float(stats.t.ppf(.975,df=k-1)*se)
    # Wilcoxon is retained as an exploratory paired test; repeated splits are dependent.
    try:
        pw=float(stats.wilcoxon(multi,single,alternative="two-sided").pvalue)
    except ValueError:
        pw=1.0
    gbr=np.asarray([r["gbr"][task] for r in R],float)
    out[task]={
        "single_mean":float(single.mean()),"single_sd":float(single.std(ddof=1)),
        "multi_mean":float(multi.mean()),"multi_sd":float(multi.std(ddof=1)),
        "delta_multi_minus_single":float(delta.mean()),
        "ci95_low":float(delta.mean()-half),"ci95_high":float(delta.mean()+half),
        "p_nadeau_bengio":p,"p_wilcoxon_exploratory":pw,
        "multi_worse_splits":int(np.sum(delta>0)),"n_splits":k,
        "mean_test_train_group_ratio":float(ratios.mean()),
        "gbr_mean_rmse":float(gbr.mean()),
        "single_vs_gbr_percent":float(100*(single.mean()-gbr.mean())/gbr.mean())
    }
    p_nb.append(p); p_w.append(pw)
for task,p in zip(tasks,holm(p_nb)): out[task]["p_holm_nadeau_bengio"]=float(p)
for task,p in zip(tasks,holm(p_w)): out[task]["p_holm_wilcoxon_exploratory"]=float(p)
with open("results/summary_statistics_revised.json","w") as f:
    json.dump(out,f,indent=2)
for task in tasks: print(task,out[task])
