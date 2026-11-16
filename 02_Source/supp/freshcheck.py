"""FreshCheck: leakage-aware freshness classifier with a 'refer to staff' option.

Proposed system
  1. Three frozen feature extractors per image: colour/texture (HSV, LBP, Lab), MobileNetV3-Small, ResNet-50.
  2. One standardised L2 logistic regression per feature set; class probabilities averaged (late fusion).
  3. Safety rule: if the highest averaged probability is below a threshold tau, the image is referred to staff.
     tau is chosen on the TRAINING folds only (inner grouped 3-fold CV) as the smallest value in 0.50-0.99
     for which spoiled images accepted as fresh stay at or below 1% of spoiled training images.
Evaluation
  Outer 5-fold cross-validation grouped by capture session (blocks of 100 consecutive file numbers per class;
  fruit: near-copy families, pHash <= 30 bits), 5 seeds; image-level folds reported for reference.
  Baselines: each single feature set with logistic regression; early fusion (concatenated features).
  Cross-dataset: binary models trained on all red meat (half-fresh removed) and tested on chicken, and reverse.
Usage: python freshcheck.py --meat <Phteven/data/raw> --poultry "<.../dataset 200x200>" --rob <dir with
       features_meat.npz, features_poultry.npz> --d3 <dir with features_fruit_3class.npz and
       assignments_fruit_3class_T30.csv> --out out/
"""
import argparse, os, re, json
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ap = argparse.ArgumentParser()
for a in ('--meat', '--poultry', '--rob', '--d3', '--out'): ap.add_argument(a)
ap.add_argument('--seeds', type=int, default=5)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)

def inv(root, splits, pat):
    rows = []
    for sp in splits:
        for f in sorted(os.listdir(os.path.join(root, sp))):
            m = re.match(pat, f)
            if m: rows.append((m.group(1), int(m.group(2))))
    return rows

def load():
    D = {}
    m = inv(A.meat, ('train', 'test'), r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-')
    cmap = {'FRESH': 0, 'HALF-FRESH': 1, 'SPOILED': 2}
    F = dict(np.load(os.path.join(A.rob, 'features_meat.npz')))
    D['meat'] = dict(F=F, y=np.array([cmap[c] for c, _ in m]), g=np.array([f'{c}-{s // 100}' for c, s in m]), spoiled=2, fresh=0)
    p = inv(A.poultry, ('training', 'validation'), r'^(segar|busuk)_(\d+)\.jpe?g$')
    F = dict(np.load(os.path.join(A.rob, 'features_poultry.npz')))
    D['poultry'] = dict(F=F, y=np.array([0 if c == 'segar' else 1 for c, _ in p]), g=np.array([f'{c}-{s // 100}' for c, s in p]), spoiled=1, fresh=0)
    fr = pd.read_csv(os.path.join(A.d3, 'assignments_fruit_3class_T30.csv'))
    F = dict(np.load(os.path.join(A.d3, 'features_fruit_3class.npz')))
    D['fruit'] = dict(F=F, y=fr.cls.map({'FRESH': 0, 'NORMAL': 1, 'ROTTEN': 2}).values, g=fr.hash_family.values.astype(str), spoiled=2, fresh=0)
    for k, v in D.items():
        assert all(len(x) == len(v['y']) for x in v['F'].values()), k
    return D

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))

def fit_proba(F, keys, tr, te, early=False):
    if early:
        X = np.hstack([F[k] for k in keys]); return lr().fit(X[tr], y_[tr]).predict_proba(X[te])
    return np.mean([lr().fit(F[k][tr], y_[tr]).predict_proba(F[k][te]) for k in keys], axis=0)

def unsafe(P, y, tau, spoiled, fresh):
    acc = P.max(1) >= tau; pred = P.argmax(1)
    return np.sum(acc & (y == spoiled) & (pred == fresh)) / max(1, np.sum(y == spoiled))

def choose_tau(F, keys, tr, y, g, spoiled, fresh, seed, target=0.01):
    P = np.zeros((len(tr), len(np.unique(y))))
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=100 + seed)
    for a, b in inner.split(tr, y[tr], g[tr]):
        P[b] = fit_proba(F, keys, tr[a], tr[b])
    for tau in np.round(np.arange(0.50, 0.995, 0.01), 2):
        if unsafe(P, y[tr], tau, spoiled, fresh) <= target: return float(tau)
    return 0.99

def evaluate(name, d, grouped=True):
    global y_
    F, y, g, sp, fr = d['F'], d['y'], d['g'], d['spoiled'], d['fresh']; y_ = y
    keys_all = [k for k in ('colour', 'mobilenetv3', 'resnet50') if k in F]
    configs = {k: ([k], False) for k in keys_all}
    configs['early_fusion'] = (keys_all, True)
    configs['FreshCheck_fusion'] = (keys_all, False)
    R = {c: [] for c in configs}; R['FreshCheck_refer'] = []
    for s in range(A.seeds):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=s) if grouped else StratifiedKFold(5, shuffle=True, random_state=s)
        splits = list(cv.split(np.zeros(len(y)), y, g) if grouped else cv.split(np.zeros(len(y)), y))
        preds = {c: np.zeros(len(y), int) for c in configs}
        Pf = np.zeros((len(y), len(np.unique(y)))); refer = np.zeros(len(y), bool); taus = []
        for tr, te in splits:
            for c, (keys, early) in configs.items():
                P = fit_proba(F, keys, tr, te, early); preds[c][te] = P.argmax(1)
                if c == 'FreshCheck_fusion': Pf[te] = P
            if grouped:
                tau = choose_tau(F, keys_all, tr, y, g, sp, fr, s); taus.append(tau)
                refer[te] = Pf[te].max(1) < tau
        for c in configs:
            p = preds[c]; R[c].append(dict(acc=float((p == y).mean()), spoiled_as_fresh=float(np.mean(p[y == sp] == fr))))
        if grouped:
            p = Pf.argmax(1); acc_m = ~refer
            R['FreshCheck_refer'].append(dict(coverage=float(acc_m.mean()), acc_on_decided=float((p[acc_m] == y[acc_m]).mean()),
                                              spoiled_as_fresh=float(np.mean((p == fr) & acc_m & (y == sp)) / np.mean(y == sp)),
                                              spoiled_referred=float(np.mean(refer[y == sp])), fresh_referred=float(np.mean(refer[y == fr])),
                                              taus=taus))
        print(name, 'grouped' if grouped else 'image', s, {c: round(R[c][-1].get('acc', R[c][-1].get('coverage', 0)) * 100, 1) for c in R if R[c]}, flush=True)
    out = {}
    for c, lst in R.items():
        if not lst: continue
        out[c] = {k: [float(np.mean([x[k] for x in lst])), float(np.std([x[k] for x in lst]))] for k in lst[0] if k != 'taus'}
        if c == 'FreshCheck_refer': out[c]['taus'] = [t for x in lst for t in x['taus']]
    return out

D = load(); RES = {}
for name, d in D.items():
    RES[name] = {'session_grouped': evaluate(name, d, True), 'image_level': evaluate(name, d, False)}
    json.dump(RES, open(os.path.join(A.out, 'freshcheck_results.json'), 'w'), indent=1)

# cross-dataset (binary) with and without the refer rule
def binarise(d, drop_half):
    keep = np.ones(len(d['y']), bool)
    if drop_half: keep = d['y'] != 1
    y = (d['y'][keep] == d['spoiled']).astype(int)
    return {k: v[keep] for k, v in d['F'].items() if k in ('colour', 'mobilenetv3', 'resnet50')}, y, d['g'][keep]
M = binarise(D['meat'], True); P_ = binarise(D['poultry'], False)
CROSS = {}
for (src, (Fs, ys, gs)), (dst, (Ft, yt, _)) in ((('meat', M), ('poultry', P_)), (('poultry', P_), ('meat', M))):
    keys = ['colour', 'mobilenetv3', 'resnet50']; y_ = ys
    tr = np.arange(len(ys))
    tau = choose_tau(Fs, keys, tr, ys, gs, 1, 0, 0)
    P = np.mean([lr().fit(Fs[k], ys).predict_proba(Ft[k]) for k in keys], axis=0); p = P.argmax(1); dec = P.max(1) >= tau
    CROSS[f'{src}_to_{dst}'] = dict(tau=tau, acc_all=float((p == yt).mean()), spoiled_as_fresh_no_refer=float(np.mean(p[yt == 1] == 0)),
                                   coverage=float(dec.mean()), acc_on_decided=float((p[dec] == yt[dec]).mean()) if dec.any() else None,
                                   spoiled_as_fresh_with_refer=float(np.mean((p == 0) & dec & (yt == 1)) / np.mean(yt == 1)))
    print('cross', src, dst, CROSS[f'{src}_to_{dst}'], flush=True)
RES['cross_dataset'] = CROSS
json.dump(RES, open(os.path.join(A.out, 'freshcheck_results.json'), 'w'), indent=1)
