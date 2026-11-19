"""Revised pilot protocol.

This is a side-by-side revision; it does not overwrite the original script.
Key changes:
- categorical vocabularies are fitted on each training split only;
- one shared MTL checkpoint is selected by mean standardized validation RMSE;
- test sets are evaluated only once, after checkpoint selection;
- Doping Ratio parsing is strict: ambiguous ranges/ratios raise an error;
- migration grouping uses compound_id when present, otherwise explicitly falls
  back to logKow and records that limitation;
- split group counts and test/train group ratios are saved for corrected tests.

Run from repository root:
    python mtl_manuscript_protocol_revised.py [number_of_splits]
"""
import json, re, sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

H, D, EPOCHS, LR, WD = 16, 24, 1000, 1e-3, 1e-4
NSPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEEDS_OFFSET = 1000

v = pd.read_csv("data/vaferi_migration_data.csv")
d = pd.read_csv("data/dechakupt_antimicrobial_data.csv")
c = pd.read_csv("data/park_cytotoxicity_data.csv")

def parse_ratio_scalar(value):
    """Accept one numeric percentage only; never silently sum numbers."""
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError("Missing Doping Ratio(%) value; resolve before modeling.")
    s = str(value).strip().replace(",", ".")
    m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?", s)
    if not m:
        raise ValueError(
            f"Ambiguous Doping Ratio value {value!r}. "
            "Ranges and ratios require source-specific interpretation."
        )
    return float(m.group(1))

def make_split(groups, seed):
    rng = np.random.RandomState(seed)
    unique = np.asarray(sorted(set(map(str, groups))))
    unique = unique[rng.permutation(len(unique))]
    n = len(unique)
    n_test = max(1, int(round(.15 * n)))
    n_val = max(1, int(round(.15 * n)))
    test_groups = set(unique[:n_test])
    val_groups = set(unique[n_test:n_test+n_val])
    train_mask = np.array([str(x) not in test_groups | val_groups for x in groups])
    val_mask = np.array([str(x) in val_groups for x in groups])
    test_mask = np.array([str(x) in test_groups for x in groups])
    return train_mask, val_mask, test_mask, {
        "n_groups_total": n, "n_groups_train": len(unique)-len(test_groups)-len(val_groups),
        "n_groups_validation": len(val_groups), "n_groups_test": len(test_groups),
        "test_train_group_ratio": len(test_groups) / max(1, len(unique)-len(test_groups)-len(val_groups))
    }

def fit_transform_features(frame, numeric_cols, categorical_cols, train_mask):
    """Fit numeric scaling and categorical vocabularies on training rows only."""
    blocks_train, blocks_all = [], []
    if numeric_cols:
        raw = frame[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(raw[train_mask]).all():
            raise ValueError("Non-finite numeric feature in training partition.")
        # Impute non-finite validation/test values with training medians.
        med = np.nanmedian(np.where(np.isfinite(raw[train_mask]), raw[train_mask], np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        raw = np.where(np.isfinite(raw), raw, med)
        mu = raw[train_mask].mean(axis=0)
        sd = raw[train_mask].std(axis=0)
        sd[sd == 0] = 1.0
        blocks_all.append((raw-mu)/sd)
    for col in categorical_cols:
        s = frame[col].fillna("NA").astype(str)
        vocab = sorted(s[train_mask].unique())
        # Unknown categories are represented by an all-zero block.
        mat = np.column_stack([(s == cat).to_numpy(float) for cat in vocab]) if vocab else np.zeros((len(frame), 0))
        blocks_all.append(mat)
    return np.hstack(blocks_all).astype(float)

def prepare_tasks(migration_group_col=None):
    # Migration: only use a true compound identifier if it is present and populated.
    if migration_group_col is None:
        if "compound_id" in v.columns and v["compound_id"].notna().all():
            migration_group_col = "compound_id"
        else:
            migration_group_col = "logKow"
            warnings.warn(
                "Migration grouping falls back to logKow, not compound identity. "
                "Interpret unseen-compound generalization cautiously.",
                RuntimeWarning
            )
    gm = v[migration_group_col].fillna("NA").astype(str).to_numpy()
    ym = pd.to_numeric(v["target_logK"], errors="raise").to_numpy(float)

    # Cytotoxicity: ambiguous/missing ratios deliberately stop execution.
    c_numeric = c.copy()
    c_numeric["__doping_ratio_numeric"] = c["Doping Ratio(%)"].map(parse_ratio_scalar)
    yc = pd.to_numeric(c["pIC50"], errors="raise").to_numpy(float)
    gc_cols = ["Core", "Doping", "Doping Ratio(%)", "Shell",
               "Coating and Modification(Ligand and Surface modification chemical)",
               "Diameter(nm)"]
    gc = c[gc_cols].fillna("NA").astype(str).agg("|".join, axis=1).to_numpy()

    xa_cols = [f"x{i}" for i in range(1, 22)]
    Xa = d[xa_cols].apply(pd.to_numeric, errors="raise").to_numpy(float)
    ya = pd.to_numeric(d["y"], errors="raise").to_numpy(float)
    ga = d[xa_cols].astype(str).agg("|".join, axis=1).to_numpy()

    return {
        "migration": (v, ["etoh_eq", "logKow", "temp"], ["material_code"], ym, gm),
        "cytotoxicity": (c_numeric, ["Diameter(nm)", "__doping_ratio_numeric"],
            ["Core", "Doping", "Shell", "Coating and Modification(Ligand and Surface modification chemical)",
             "Cell-identification", "Cell-anatomical-type", "Cell-source-species", "Cell-origin",
             "Cell-tissue-organ-origin"], yc, gc),
        "antimicrobial": (d, xa_cols, [], ya, ga)
    }

def lin_init(rng, i, o):
    b = 1 / np.sqrt(i)
    return rng.uniform(-b, b, (i,o)), rng.uniform(-b,b,o)
def init_task(rng, i):
    We,be=lin_init(rng,i,H); Wh,bh=lin_init(rng,D,1)
    return dict(We=We,be=be,Wh=Wh,bh=bh)
def init_trunk(rng):
    W1,b1=lin_init(rng,H,D); W2,b2=lin_init(rng,D,D)
    return dict(W1=W1,b1=b1,W2=W2,b2=b2)
def fwd(tp,tr,X):
    h=X@tp["We"]+tp["be"]; z1=h@tr["W1"]+tr["b1"]; a1=np.maximum(z1,0)
    z2=a1@tr["W2"]+tr["b2"]; a2=np.maximum(z2,0)
    out=(a2@tp["Wh"]+tp["bh"])[:,0]
    return out,(X,h,z1,a1,z2,a2)
def bwd(tp,tr,cache,out,y):
    X,h,z1,a1,z2,a2=cache; n=len(y); go=(2/n)*(out-y)[:,None]
    gt={"Wh":a2.T@go,"bh":go.sum(0)}
    gz2=(go@tp["Wh"].T)*(z2>0)
    gtr={"W2":a1.T@gz2,"b2":gz2.sum(0)}
    gz1=(gz2@tr["W2"].T)*(z1>0)
    gtr.update({"W1":h.T@gz1,"b1":gz1.sum(0)})
    gh=gz1@tr["W1"].T
    gt.update({"We":X.T@gh,"be":gh.sum(0)})
    return gt,gtr
class Adam:
    def __init__(self):
        self.m={}; self.v={}; self.t=0
    def step(self,params,grads):
        self.t+=1
        for k,p in params.items():
            g=grads[k]+WD*p
            m=self.m.get(k,0)*.9+.1*g
            vv=self.v.get(k,0)*.999+.001*g*g
            self.m[k],self.v[k]=m,vv
            p-=LR*(m/(1-.9**self.t))/(np.sqrt(vv/(1-.999**self.t))+1e-8)

def clone_params(obj):
    return {k:x.copy() for k,x in obj.items()}

def gradcheck_all():
    rng=np.random.RandomState(0); tp=init_task(rng,5); tr=init_trunk(rng)
    X=rng.randn(7,5); y=rng.randn(7)
    def loss():
        o,_=fwd(tp,tr,X); return np.mean((o-y)**2)
    o,cache=fwd(tp,tr,X); gt,gtr=bwd(tp,tr,cache,o,y); worst=0.0
    for grads,params in ((gt,tp),(gtr,tr)):
        for name,g in grads.items():
            for ix in np.ndindex(params[name].shape):
                old=params[name][ix]
                params[name][ix]=old+1e-6; plus=loss()
                params[name][ix]=old-1e-6; minus=loss()
                params[name][ix]=old
                worst=max(worst,abs((plus-minus)/2e-6-g[ix]))
    return worst

def run_split(seed, task_specs):
    split_info={}; prepared={}
    for name,(frame,num_cols,cat_cols,y,groups) in task_specs.items():
        tr,va,te,info=make_split(groups,SEEDS_OFFSET+seed)
        split_info[name]=info
        X=fit_transform_features(frame,num_cols,cat_cols,tr)
        ym=float(y[tr].mean()); ys=float(y[tr].std())
        if not np.isfinite(ys) or ys==0: raise ValueError(f"{name}: target SD is zero/non-finite")
        prepared[name]={"Xtr":X[tr],"Xva":X[va],"Xte":X[te],
            "ytr":(y[tr]-ym)/ys,"yva":(y[va]-ym)/ys,"yte":y[te],
            "ym":ym,"ys":ys,"groups":info}
    order=list(prepared)
    out={"single":{},"multi":{},"gbr":{},"best_epoch":{"single":{},"multi":{}},
         "split_info":split_info}

    def rmse_scaled(name,tp,trunk,part):
        dd=prepared[name]; pred,_=fwd(tp,trunk,dd["X"+part])
        target=dd["y"+part] if part!="te" else (dd["yte"]-dd["ym"])/dd["ys"]
        return float(np.sqrt(np.mean((pred-target)**2)))

    # Single-task models: validation checkpoint selection, one final test evaluation.
    for k,name in enumerate(order):
        dd=prepared[name]; rng=np.random.RandomState(seed*10+k)
        tp=init_task(rng,dd["Xtr"].shape[1]); trunk=init_trunk(rng); opt=Adam()
        best=float("inf"); best_epoch=0; best_tp=None; best_trunk=None
        for epoch in range(EPOCHS):
            pred,cache=fwd(tp,trunk,dd["Xtr"])
            gt,gtr=bwd(tp,trunk,cache,pred,dd["ytr"])
            params={("task",key):val for key,val in tp.items()}
            params.update({("shared",key):val for key,val in trunk.items()})
            grads={("task",key):val for key,val in gt.items()}
            grads.update({("shared",key):val for key,val in gtr.items()})
            opt.step(params,grads)
            val=rmse_scaled(name,tp,trunk,"va")
            if val<best:
                best=val; best_epoch=epoch+1
                best_tp=clone_params(tp); best_trunk=clone_params(trunk)
        dd=prepared[name]
        test_pred,_=fwd(best_tp,best_trunk,dd["Xte"])
        test_rmse=float(np.sqrt(np.mean((test_pred*dd["ys"]+dd["ym"]-dd["yte"])**2)))
        out["single"][name]=test_rmse; out["best_epoch"]["single"][name]=best_epoch

    # Multi-task model: one shared checkpoint selected by mean standardized val RMSE.
    rng=np.random.RandomState(seed*10+9)
    tps={name:init_task(rng,prepared[name]["Xtr"].shape[1]) for name in order}
    trunk=init_trunk(rng); opt=Adam()
    best_aggregate=float("inf"); best_epoch=0; best_tps=None; best_trunk=None
    for epoch in range(EPOCHS):
        shared_grads={key:np.zeros_like(val) for key,val in trunk.items()}
        params={}; grads={}
        for name in order:
            dd=prepared[name]; pred,cache=fwd(tps[name],trunk,dd["Xtr"])
            gt,gtr=bwd(tps[name],trunk,cache,pred,dd["ytr"])
            for key,val in gt.items():
                params[(name,key)]=tps[name][key]; grads[(name,key)]=val
            for key,val in gtr.items(): shared_grads[key]+=val
        for key,val in trunk.items():
            params[("shared",key)]=val; grads[("shared",key)]=shared_grads[key]
        opt.step(params,grads)
        val_scores=[rmse_scaled(name,tps[name],trunk,"va") for name in order]
        aggregate=float(np.mean(val_scores))
        if aggregate<best_aggregate:
            best_aggregate=aggregate; best_epoch=epoch+1
            best_tps={name:clone_params(tp) for name,tp in tps.items()}
            best_trunk=clone_params(trunk)
    out["best_epoch"]["multi"]["shared"]=best_epoch
    for name in order:
        dd=prepared[name]
        pred,_=fwd(best_tps[name],best_trunk,dd["Xte"])
        out["multi"][name]=float(np.sqrt(np.mean((pred*dd["ys"]+dd["ym"]-dd["yte"])**2)))
        model=GradientBoostingRegressor(random_state=seed).fit(
            dd["Xtr"],dd["ytr"]*dd["ys"]+dd["ym"])
        out["gbr"][name]=float(np.sqrt(np.mean((model.predict(dd["Xte"])-dd["yte"])**2)))
    return out

if __name__=="__main__":
    specs=prepare_tasks()
    print("gradient check max absolute difference:",gradcheck_all())
    results=[]
    for seed in range(NSPLIT):
        r=run_split(seed,specs); results.append(r)
        print(seed,{t:(round(r["single"][t],4),round(r["multi"][t],4)) for t in specs},flush=True)
    with open("results/results_manuscript_protocol_revised.json","w") as f:
        json.dump(results,f,indent=2)
