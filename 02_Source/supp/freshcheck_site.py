"""FreshCheck site calibration: how many local labelled images does a new kitchen need?

Source model: all red-meat images (half-fresh removed, binary fresh vs spoiled). Target: chicken breast.
k capture sessions per class (blocks of 20 consecutive files) from the target are added to training; the model is
tested on all remaining target sessions. k = 0, 1, 2, 5, 10 (0 to 400 local images); 20 random draws per k.
Same model as FreshCheck (late fusion of colour, MobileNetV3, ResNet-50 logistic regressions).
Also run in reverse (chicken source, red-meat target). Control: the same local images used alone, without the source dataset.
Usage: python freshcheck_site.py --meat ... --poultry ... --rob ... --out out/
"""
import argparse, os, re, json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
for a in ('--meat', '--poultry', '--rob', '--out'): ap.add_argument(a)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
def inv(root, splits, pat):
    rows = []
    for sp in splits:
        for f in sorted(os.listdir(os.path.join(root, sp))):
            m = re.match(pat, f)
            if m: rows.append((m.group(1), int(m.group(2))))
    return rows
KEYS = ('colour', 'mobilenetv3', 'resnet50')
m = inv(A.meat, ('train', 'test'), r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-'); Fm = dict(np.load(os.path.join(A.rob, 'features_meat.npz')))
keep = np.array([c != 'HALF-FRESH' for c, _ in m])
M = dict(F={k: Fm[k][keep] for k in KEYS}, y=np.array([int(c == 'SPOILED') for c, _ in m])[keep], g=np.array([f'{c}-{s // 20}' for c, s in m])[keep])
p = inv(A.poultry, ('training', 'validation'), r'^(segar|busuk)_(\d+)\.jpe?g$'); Fp = dict(np.load(os.path.join(A.rob, 'features_poultry.npz')))
P = dict(F={k: Fp[k] for k in KEYS}, y=np.array([int(c == 'busuk') for c, _ in p]), g=np.array([f'{c}-{s // 20}' for c, s in p]))
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
OUT = {}
for name, S, T in (('meat_to_chicken', M, P), ('chicken_to_meat', P, M)):
    res = {}
    for k in (0, 1, 2, 5, 10):
        accs, saf, nloc, la, ls = [], [], [], [], []
        for draw in range(20 if k else 1):
            rng = np.random.default_rng(draw); add = []
            for c in (0, 1):
                groups = sorted(set(T['g'][T['y'] == c])); pick = rng.choice(groups, size=k, replace=False) if k else []
                add += list(pick)
            mask = np.isin(T['g'], add)
            Pr = np.mean([lr().fit(np.vstack([S['F'][f], T['F'][f][mask]]), np.concatenate([S['y'], T['y'][mask]])).predict_proba(T['F'][f][~mask]) for f in KEYS], axis=0)
            pred = Pr.argmax(1); yt = T['y'][~mask]
            if k:
                Pl = np.mean([lr().fit(T['F'][f][mask], T['y'][mask]).predict_proba(T['F'][f][~mask]) for f in KEYS], axis=0).argmax(1)
                la.append(float((Pl == yt).mean())); ls.append(float(np.mean(Pl[yt == 1] == 0)))
            nloc.append(int(mask.sum())); accs.append(float((pred == yt).mean())); saf.append(float(np.mean(pred[yt == 1] == 0)))
        res[k] = dict(local_images=[float(np.mean(nloc)), float(np.std(nloc))], acc=[float(np.mean(accs)), float(np.std(accs))], spoiled_as_fresh=[float(np.mean(saf)), float(np.std(saf))],
                      local_only_acc=[float(np.mean(la)), float(np.std(la))] if k else None, local_only_spoiled_as_fresh=[float(np.mean(ls)), float(np.std(ls))] if k else None)
        print(name, k, res[k], flush=True)
    OUT[name] = res
json.dump(OUT, open(os.path.join(A.out, 'freshcheck_site_results.json'), 'w'), indent=1)
