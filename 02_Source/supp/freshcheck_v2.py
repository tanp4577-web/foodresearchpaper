"""FreshCheck v2: late fusion over multi-level, multi-view frozen ResNet-50 features (layers 2-4; global + local views),
MobileNetV3 and colour features, with the same refer rule as v1. Compared with v1 under session-grouped and image-level CV,
against a random-grouping null, and in cross-dataset transfer with local labelling.
Usage: python freshcheck_v2.py --meat .. --poultry .. --rob <dir> --d3 <dir> --out <dir>"""
import argparse, os, re, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ap = argparse.ArgumentParser(); [ap.add_argument(a) for a in ('--meat', '--poultry', '--rob', '--d3', '--out')]
ap.add_argument('--seeds', type=int, default=5); A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
ML = ['g_l2', 'g_l3', 'g_l4', 'p_l2', 'p_l3', 'p_l4']
def inv(root, splits, pat):
    return [(m.group(1), int(m.group(2))) for sp in splits for f in sorted(os.listdir(os.path.join(root, sp))) for m in [re.match(pat, f)] if m]
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
D = {}
m = inv(A.meat, ('train', 'test'), r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-')
F = dict(np.load(os.path.join(A.rob, 'features_meat.npz'))); F.update(np.load(os.path.join(A.rob, 'features_meat_v2.npz')))
D['meat'] = dict(F=F, y=np.array([{'FRESH': 0, 'HALF-FRESH': 1, 'SPOILED': 2}[c] for c, _ in m]), g=np.array([f'{c}-{s // 100}' for c, s in m]), num=np.array([s for _, s in m]), cls=np.array([c for c, _ in m]), spoiled=2)
p = inv(A.poultry, ('training', 'validation'), r'^(segar|busuk)_(\d+)\.jpe?g$')
F = dict(np.load(os.path.join(A.rob, 'features_poultry.npz'))); F.update(np.load(os.path.join(A.rob, 'features_poultry_v2.npz')))
D['poultry'] = dict(F=F, y=np.array([0 if c == 'segar' else 1 for c, _ in p]), g=np.array([f'{c}-{s // 100}' for c, s in p]), num=np.array([s for _, s in p]), cls=np.array([c for c, _ in p]), spoiled=1)
fr = pd.read_csv(os.path.join(A.d3, 'assignments_fruit_3class_T30.csv'))
F = dict(np.load(os.path.join(A.d3, 'features_fruit_3class.npz'))); F.update(np.load(os.path.join(A.d3, 'features_fruit_3class_v2.npz')))
D['fruit'] = dict(F=F, y=fr.cls.map({'FRESH': 0, 'NORMAL': 1, 'ROTTEN': 2}).values, g=fr.hash_family.values.astype(str), spoiled=2)
def V1(d): return [k for k in ('colour', 'mobilenetv3', 'resnet50') if k in d['F']]
def V2(d): return [k for k in ('colour', 'colour_fine', 'mobilenetv3') if k in d['F']] + ML

def probs(d, keys, tr, te): return {k: lr().fit(d['F'][k][tr], d['y'][tr]).predict_proba(d['F'][k][te]) for k in keys}
def choose_tau(d, keys, tr, seed):
    y, g = d['y'][tr], d['g'][tr]; P = np.zeros((len(tr), len(np.unique(d['y']))))
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=100 + seed).split(y, y, g):
        P[b] = np.mean(list(probs(d, keys, tr[a], tr[b]).values()), 0)
    sp = y == d['spoiled']
    for t in np.arange(0.50, 0.995, 0.01):
        if np.mean((P[sp].argmax(1) == 0) & (P[sp].max(1) >= t)) <= 0.01: return float(t)
    return 0.99
def summarise(d, P, tau=None):
    y = d['y']; pr = P.argmax(1); sp = y == d['spoiled']; o = dict(acc=float((pr == y).mean()), saf=float(np.mean(pr[sp] == 0)))
    if tau is not None:
        dec = P.max(1) >= tau
        o.update(coverage=float(dec.mean()), acc_decided=float((pr[dec] == y[dec]).mean()), saf_refer=float(np.mean((pr[sp] == 0) & dec[sp])),
                 spoiled_referred=float(np.mean(~dec[sp])), fresh_referred=float(np.mean(~dec[y == 0])))
    return o
def agg(L): return {k: [float(np.mean([x[k] for x in L])), float(np.std([x[k] for x in L]))] for k in L[0]}

R = {}
for name, d in D.items():
    y, g = d['y'], d['g']; keys = sorted(set(V1(d)) | set(V2(d))); out = {}
    for proto in ('session_grouped', 'image_level'):
        S = {}
        for seed in range(A.seeds):
            spl = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, y, g) if proto == 'session_grouped' else StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y)
            P = {k: np.zeros((len(y), len(np.unique(y)))) for k in keys}; tau = np.zeros(len(y)); tau1 = np.zeros(len(y))
            for tr, te in spl:
                for k, v in probs(d, keys, tr, te).items(): P[k][te] = v
                if proto == 'session_grouped': tau[te] = choose_tau(d, V2(d), tr, seed); tau1[te] = choose_tau(d, V1(d), tr, seed)
            for k in keys: S.setdefault(k, []).append(summarise(d, P[k]))
            P1 = np.mean([P[k] for k in V1(d)], 0); P2 = np.mean([P[k] for k in V2(d)], 0)
            S.setdefault('FreshCheck_v1', []).append(summarise(d, P1, tau1 if proto == 'session_grouped' else None))
            S.setdefault('FreshCheck_v2', []).append(summarise(d, P2, tau if proto == 'session_grouped' else None))
            S.setdefault('v2_resnet_multilevel_only', []).append(summarise(d, np.mean([P[k] for k in ML], 0)))
            S.setdefault('v2_global_views_only', []).append(summarise(d, np.mean([P[k] for k in V2(d) if not k.startswith('p_')], 0)))
            S.setdefault('v2_local_views_only', []).append(summarise(d, np.mean([P[k] for k in V2(d) if not k.startswith('g_')], 0)))
            if proto == 'session_grouped': S.setdefault('taus_v2', []).append({'min': float(tau.min()), 'max': float(tau.max()), 'median': float(np.median(tau))})
        out[proto] = {k: agg(v) for k, v in S.items()}
        print(name, proto, {k: round(v['acc'][0] * 100, 1) for k, v in out[proto].items() if 'acc' in v}, flush=True)
    if name != 'fruit':   # random-grouping null for v2 (blocks of 100, 20 groupings)
        null = []
        for r in range(20):
            rng = np.random.default_rng(1000 + r); gg = np.empty(len(y), object)
            for c in np.unique(d['cls']):
                idx = rng.permutation(np.where(d['cls'] == c)[0]); gg[idx] = [f'{c}-{i // 100}' for i in range(len(idx))]
            P2 = np.zeros((len(y), len(np.unique(y))))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(y, y, gg):
                P2[te] = np.mean(list(probs(d, V2(d), tr, te).values()), 0)
            null.append(float((P2.argmax(1) == y).mean()))
        best = max(out['session_grouped']['FreshCheck_v2']['acc'][0], 0)
        out['null_v2'] = dict(mean=float(np.mean(null)), min=float(min(null)), max=float(max(null)), p=float((1 + sum(n <= best for n in null)) / 21))
        print(name, 'null', out['null_v2'], flush=True)
    R[name] = out
    json.dump(R, open(os.path.join(A.out, 'freshcheck_v2_results.json'), 'w'), indent=1)

# ---- transfer + local labelling (binary fresh vs spoiled) with v1 and v2
def binary(d, name):
    keep = np.ones(len(d['y']), bool) if name == 'poultry' else d['y'] != 1
    return dict(F={k: v[keep] for k, v in d['F'].items()}, y=(d['y'][keep] > 0).astype(int), num=d['num'][keep], cls=d['cls'][keep])
B = {n: binary(D[n], n) for n in ('meat', 'poultry')}; T = {}
for src, tgt in (('meat', 'poultry'), ('poultry', 'meat')):
    S_, T_ = B[src], B[tgt]; sess = np.array([f'{c}-{s // 20}' for c, s in zip(T_['cls'], T_['num'])]); res = {}
    for k in (0, 1, 2, 5, 10):
        rec = {v: dict(acc=[], saf=[]) for v in ('v1', 'v2', 'v1_local_only', 'v2_local_only')}; nloc = []
        for draw in range(1 if k == 0 else 20):
            rng = np.random.default_rng(draw); mask = np.zeros(len(T_['y']), bool)
            for c in np.unique(T_['cls']):
                ss = np.unique(sess[T_['cls'] == c]); mask |= np.isin(sess, rng.choice(ss, k, replace=False)) if k else False
            nloc.append(int(mask.sum())); yt = T_['y'][~mask]
            for v, keys in (('v1', V1(D[src])), ('v2', V2(D[src]))):
                P = np.mean([lr().fit(np.vstack([S_['F'][f], T_['F'][f][mask]]), np.concatenate([S_['y'], T_['y'][mask]])).predict_proba(T_['F'][f][~mask]) for f in keys], 0).argmax(1)
                rec[v]['acc'].append(float((P == yt).mean())); rec[v]['saf'].append(float(np.mean(P[yt == 1] == 0)))
                if k:
                    P = np.mean([lr().fit(T_['F'][f][mask], T_['y'][mask]).predict_proba(T_['F'][f][~mask]) for f in keys], 0).argmax(1)
                    rec[v + '_local_only']['acc'].append(float((P == yt).mean())); rec[v + '_local_only']['saf'].append(float(np.mean(P[yt == 1] == 0)))
        res[k] = dict(local_images=float(np.mean(nloc)), **{v: {m: [float(np.mean(x)), float(np.std(x))] for m, x in r.items() if x} for v, r in rec.items()})
        print(src, '->', tgt, k, {v: round(r['acc'][0] * 100, 1) for v, r in res[k].items() if isinstance(r, dict) and 'acc' in r}, flush=True)
    T[f'{src}_to_{tgt}'] = res
R['transfer'] = T; json.dump(R, open(os.path.join(A.out, 'freshcheck_v2_results.json'), 'w'), indent=1); print('saved')
