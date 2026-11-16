"""Robustness checks for the leakage audit (Supplementary material).

1. Null distribution for capture-order grouping: 20 random within-class groupings of the same
   group size (20 and 100 images), one fold assignment each, against the five capture-order
   fold assignments; empirical p = (1 + #null <= best capture-order accuracy) / (1 + 20).
2. Second backbone: frozen ImageNet ResNet-50 (timm resnet50.a1_in1k, 2,048-d pooled features)
   under every protocol and in cross-dataset transfer.
Usage: python robustness_audit.py --meat Phteven/data/raw --poultry "<poultry dir>" --out out/
       RESNET_WEIGHTS=<resnet50_a1_0-14fe96d1.pth>  MNV3_WEIGHTS=<mnv3s_tf.pth>
"""
import argparse, os, re, json
import numpy as np, torch, timm, cv2
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import imagehash
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ap = argparse.ArgumentParser(); ap.add_argument('--meat'); ap.add_argument('--poultry'); ap.add_argument('--out')
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); torch.set_num_threads(os.cpu_count())

def inv(root, splits, pat, cmap):
    rows = []
    for sp in splits:
        for f in sorted(os.listdir(os.path.join(root, sp))):
            m = re.match(pat, f)
            if m: rows.append(dict(split=sp, path=os.path.join(root, sp, f), cls=cmap(m.group(1)), src=int(m.group(2))))
    return rows
DS = {
 'meat': (inv(A.meat, ('train', 'test'), r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-', lambda c: c), {'FRESH': 0, 'HALF-FRESH': 1, 'SPOILED': 2}, ('train', 'test')),
 'poultry': (inv(A.poultry, ('training', 'validation'), r'^(segar|busuk)_(\d+)\.jpe?g$', lambda c: {'segar': 'FRESH', 'busuk': 'SPOILED'}[c]), {'FRESH': 0, 'SPOILED': 1}, ('training', 'validation'))}

def colour(p):
    im = cv2.resize(cv2.imread(p), (224, 224)); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    h = [cv2.calcHist([hsv], [c], None, [b], [0, r]).ravel() for c, b, r in ((0, 18, 180), (1, 8, 256), (2, 8, 256))]
    lh = np.histogram(local_binary_pattern(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 8, 1, 'uniform'), bins=10, range=(0, 10))[0]
    return np.concatenate([x / x.sum() for x in h] + [lh / lh.sum(), lab.reshape(-1, 3).mean(0) / 255, lab.reshape(-1, 3).std(0) / 255])
def deep(name, wfile, paths):
    m = timm.create_model(name, pretrained=wfile is None, num_classes=0)
    if wfile:
        sd = torch.load(wfile, map_location='cpu'); sd = {k: v for k, v in sd.items() if not k.startswith(('classifier', 'fc.'))}
        print(name, m.load_state_dict(sd, strict=False))
    m.eval(); cfg = timm.data.resolve_data_config({}, model=m); tf = timm.data.create_transform(**cfg)
    with torch.no_grad():
        return np.concatenate([m(torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + 32]])).numpy() for i in range(0, len(paths), 32)])
FEATS = {}
for ds, (rows, CL, sp) in DS.items():
    cache = os.path.join(A.out, f'features_{ds}.npz')
    if os.path.exists(cache): FEATS[ds] = dict(np.load(cache)); continue
    paths = [r['path'] for r in rows]
    F = {'colour': np.array([colour(p) for p in paths]),
         'mobilenetv3': deep('tf_mobilenetv3_small_100', os.environ.get('MNV3_WEIGHTS'), paths),
         'resnet50': deep('resnet50', os.environ.get('RESNET_WEIGHTS'), paths)}
    np.savez(cache, **F); FEATS[ds] = F; print(ds, {k: v.shape for k, v in F.items()}, flush=True)

def clf(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
def acc_cv(X, y, groups, seed):
    sp = StratifiedKFold(5, shuffle=True, random_state=seed) if groups is None else StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    yp = np.zeros_like(y)
    for a, b in (sp.split(X, y) if groups is None else sp.split(X, y, groups)): yp[b] = clf().fit(X[a], y[a]).predict(X[b])
    return yp
def summ(y, yps, spoiled):
    accs = [float((y == p).mean()) for p in yps]; cms = [confusion_matrix(y, p, labels=list(range(spoiled + 1))) for p in yps]
    saf = [float(c[spoiled, 0] / c[spoiled].sum()) for c in cms]
    return dict(acc=float(np.mean(accs)), acc_sd=float(np.std(accs)), accs=accs, macroF1=float(np.mean([f1_score(y, p, average='macro') for p in yps])), spoiled_as_fresh=float(np.mean(saf)))
OUT = {}
for ds, (rows, CL, splits) in DS.items():
    y = np.array([CL[r['cls']] for r in rows]); cls = np.array([r['cls'] for r in rows]); src = np.array([r['src'] for r in rows]); split = np.array([r['split'] for r in rows])
    spoiled = CL['SPOILED']; n = len(y)
    B = np.stack([np.array([int(c, 16) >> i & 1 for c in str(imagehash.phash(Image.open(r['path']).convert('RGB'), hash_size=16)) for i in (3, 2, 1, 0)], np.uint8) for r in rows])
    D = (B[:, None, :] != B[None, :, :]).sum(2); _, fam = connected_components(csr_matrix(D <= 60), directed=False)
    def blocks(s): return np.array([f'{c}-{v // s}' for c, v in zip(cls, src)])
    def rgroups(s, seed):
        g = np.empty(n, dtype=object); r_ = np.random.default_rng(seed)
        for c in sorted(set(cls)):
            for j, i in enumerate(r_.permutation(np.where(cls == c)[0])): g[i] = f'{c}-{j // s}'
        return g
    tr, te = np.where(split == splits[0])[0], np.where(split == splits[1])[0]
    OUT[ds] = {}
    for fn, X in FEATS[ds].items():
        R = {}
        if fn == 'resnet50':
            p = clf().fit(X[tr], y[tr]).predict(X[te]); R['official_split'] = summ(y[te], [p], spoiled)
            R['image_level_5fold'] = summ(y, [acc_cv(X, y, None, s) for s in range(5)], spoiled)
            R['hash_family_5fold'] = summ(y, [acc_cv(X, y, fam, s) for s in range(5)], spoiled)
        for size in (20, 100):
            cap = summ(y, [acc_cv(X, y, blocks(size), s) for s in range(5)], spoiled)
            null = [float((acc_cv(X, y, rgroups(size, g), g) == y).mean()) for g in range(20)]
            p = (1 + sum(v <= max(cap['accs']) for v in null)) / (1 + len(null))
            R[f'capture_block_{size}'] = cap
            R[f'random_null_{size}'] = dict(accs=null, mean=float(np.mean(null)), min=float(min(null)), max=float(max(null)), p_value=p)
        OUT[ds][fn] = R
        print(ds, fn, {k: (round(v['acc'] * 100, 1) if 'acc' in v else (round(v['min'] * 100, 1), round(v['max'] * 100, 1), round(v['p_value'], 3))) for k, v in R.items()}, flush=True)
# cross-dataset with ResNet-50
cm, cp = np.array([r['cls'] for r in DS['meat'][0]]), np.array([r['cls'] for r in DS['poultry'][0]])
keep = cm != 'HALF-FRESH'; yA = (cm[keep] == 'SPOILED').astype(int); yB = (cp == 'SPOILED').astype(int)
XA, XB = FEATS['meat']['resnet50'][keep], FEATS['poultry']['resnet50']
cross = {}
for a, b, Xa, ya, Xb, yb in (('meat', 'poultry', XA, yA, XB, yB), ('poultry', 'meat', XB, yB, XA, yA)):
    m = clf().fit(Xa, ya); p = m.predict(Xb); pr = m.predict_proba(Xb)[:, 1]; c = confusion_matrix(yb, p, labels=[0, 1])
    cross[f'train_{a}_test_{b}'] = dict(acc=float((p == yb).mean()), auc=float(roc_auc_score(yb, pr)), spoiled_as_fresh=float(c[1, 0] / c[1].sum()), fresh_as_spoiled=float(c[0, 1] / c[0].sum()))
cross['meat_within_image5fold'] = summ(yA, [acc_cv(XA, yA, None, s) for s in range(5)], 1)['acc']
cross['poultry_within_image5fold'] = summ(yB, [acc_cv(XB, yB, None, s) for s in range(5)], 1)['acc']
OUT['cross_resnet50'] = cross; print('cross resnet50', cross, flush=True)
json.dump(OUT, open(os.path.join(A.out, 'robustness_results.json'), 'w'), indent=1); print('done', flush=True)
