"""Freshness-image leakage audit (Supplementary File S2).

Reproduces Sections 4.3-4.4, Tables 5-6 and Figure 3 of
"Image-Based Food Freshness Assessment: Reference Labels, Validation and Kitchen Readiness".

Datasets (public; read from local copies):
  A  Meat Freshness Image Dataset (Shanawad, Kaggle 2022), red meat, 3 classes; mirror:
     git clone https://github.com/TheLohia/Phteven            -> Phteven/data/raw/{train,test}
  B  Fresh and rotten poultry meat (Kaggle, calvinsama), chicken breast, 2 classes; 1,300-image subset in:
     git clone https://github.com/LinggarM/Poultry-Meat-Freshness-Classification-with-Transfer-Learning-using-ResNet
     -> <repo>/dataset_citra_dada_ayam/dataset 200x200/{training,validation}

Usage:
  pip install numpy scipy scikit-learn scikit-image pillow imagehash opencv-python-headless torch timm
  python freshness_leakage_audit.py --meat Phteven/data/raw --poultry "<repo>/dataset_citra_dada_ayam/dataset 200x200" --out out/
  (optional) MNV3_WEIGHTS=<file> loads local ImageNet weights for timm tf_mobilenetv3_small_100.

Audit steps for each dataset:
  1. 256-bit perceptual hashes; near-duplicate families (Hamming <= 60 bits; connected components).
  2. Capture-order evidence: share of images whose nearest perceptual neighbour is a same-class image
     within 5 file numbers, against the share expected by chance (random same-class image).
  3. Two feature sets (colour/texture; frozen MobileNetV3-Small) + L2 logistic regression.
  4. Protocols: official split; image-level stratified 5-fold; 5-fold grouped by hash family;
     5-fold grouped by blocks of consecutive file numbers (capture order); random groups of equal size.
  5. Cross-dataset transfer (binary fresh vs spoiled; half-fresh left out): train on A, test on B and reverse.
"""
import argparse, os, re, json, csv, collections
import numpy as np
from PIL import Image
import imagehash, cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.feature import local_binary_pattern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument('--meat', required=True); ap.add_argument('--poultry', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--reps', type=int, default=5)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
T_FAMILY = 60

# ---------------- inventories ----------------
def inv_meat(root):
    rows = []
    for split in ('train', 'test'):
        for f in sorted(os.listdir(os.path.join(root, split))):
            m = re.match(r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-', f)
            if m: rows.append(dict(split=split, path=os.path.join(root, split, f), file=f, cls=m.group(1), src=int(m.group(2))))
    return rows, {'FRESH': 0, 'HALF-FRESH': 1, 'SPOILED': 2}, ('train', 'test')
def inv_poultry(root):
    rows = []
    for split in ('training', 'validation'):
        for f in sorted(os.listdir(os.path.join(root, split))):
            m = re.match(r'^(segar|busuk)_(\d+)\.jpe?g$', f)
            if m: rows.append(dict(split=split, path=os.path.join(root, split, f), file=f, cls={'segar': 'FRESH', 'busuk': 'SPOILED'}[m.group(1)], src=int(m.group(2))))
    return rows, {'FRESH': 0, 'SPOILED': 1}, ('training', 'validation')

# ---------------- features ----------------
def colour_feats(path):
    """34 marginal HSV bins (H 18, S 8, V 8), 10 uniform-LBP bins, Lab mean and SD (OpenCV, 224 x 224)."""
    im = cv2.resize(cv2.imread(path), (224, 224))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    h = [cv2.calcHist([hsv], [c], None, [b], [0, r_]).ravel() for c, b, r_ in ((0, 18, 180), (1, 8, 256), (2, 8, 256))]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lh = np.histogram(local_binary_pattern(g, 8, 1, 'uniform'), bins=10, range=(0, 10))[0]
    return np.concatenate([x / x.sum() for x in h] + [lh / lh.sum(), lab.reshape(-1, 3).mean(0) / 255, lab.reshape(-1, 3).std(0) / 255])
_cnn = None
def cnn_feats(paths):
    global _cnn
    import torch, timm
    if _cnn is None:
        w = os.environ.get('MNV3_WEIGHTS')
        m = timm.create_model('tf_mobilenetv3_small_100', pretrained=w is None, num_classes=0)
        if w: m.load_state_dict({k: v for k, v in torch.load(w, map_location='cpu').items() if not k.startswith('classifier')})
        m.eval(); cfg = timm.data.resolve_data_config({}, model=m); _cnn = (m, timm.data.create_transform(**cfg))
    m, tf = _cnn
    with torch.no_grad():
        return np.concatenate([m(torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + 64]])).numpy() for i in range(0, len(paths), 64)])

# ---------------- evaluation helpers ----------------
def clf(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
def metrics(yt, yp, spoiled, fresh=0):
    labels = sorted(set(yt) | set(yp)); cm = confusion_matrix(yt, yp, labels=list(range(spoiled + 1)))
    return dict(acc=float((yt == yp).mean()), bal_acc=float(balanced_accuracy_score(yt, yp)), macroF1=float(f1_score(yt, yp, average='macro')),
                spoiled_as_fresh=float(cm[spoiled, fresh] / cm[spoiled].sum()), spoiled_missed=float(1 - cm[spoiled, spoiled] / cm[spoiled].sum()),
                fresh_as_spoiled=float(cm[fresh, spoiled] / cm[fresh].sum()), cm=cm.tolist())
def cv(X, y, spoiled, groups=None, reps=A.reps):
    res = []
    for s in range(reps):
        sp = StratifiedGroupKFold(5, shuffle=True, random_state=s) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=s)
        yp = np.zeros_like(y)
        for a, b in (sp.split(X, y, groups) if groups is not None else sp.split(X, y)):
            yp[b] = clf().fit(X[a], y[a]).predict(X[b])
        res.append(metrics(y, yp, spoiled))
    out = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k != 'cm'}
    out['acc_sd'] = float(np.std([r['acc'] for r in res])); out['cm_mean'] = np.mean([r['cm'] for r in res], 0).tolist()
    return out

def audit(name, rows, CLS, splits):
    n = len(rows); y = np.array([CLS[r['cls']] for r in rows]); spoiled = CLS['SPOILED']
    split = np.array([r['split'] for r in rows]); src = np.array([r['src'] for r in rows]); cls = np.array([r['cls'] for r in rows])
    for r in rows: r['ph'] = str(imagehash.phash(Image.open(r['path']).convert('RGB'), hash_size=16))
    B = np.stack([np.array([int(c, 16) >> i & 1 for c in r['ph'] for i in (3, 2, 1, 0)], np.uint8) for r in rows])
    D = (B[:, None, :] != B[None, :, :]).sum(2)
    _, fam = connected_components(csr_matrix(D <= T_FAMILY), directed=False)
    Dn = D.copy(); np.fill_diagonal(Dn, 999); nn = Dn.argmin(1)
    adj = float(((cls[nn] == cls) & (np.abs(src[nn] - src) <= 5)).mean())
    rng = np.random.default_rng(0)
    chance = float(np.mean([abs(src[rng.choice(np.where(cls == cls[i])[0])] - src[i]) <= 5 for i in range(n)]))
    tr, te = np.where(split == splits[0])[0], np.where(split == splits[1])[0]
    nn_test = D[np.ix_(te, tr)].min(1); cross = {f for f in set(fam) if len(set(split[fam == f])) > 1}
    ev = dict(n=n, classes=dict(collections.Counter(cls)), nn_same_class_within5=adj, chance_within5=chance,
              nn_same_class=float((cls[nn] == cls).mean()), n_test=int(len(te)),
              test_in_cross_split_family=int(np.isin(fam[te], list(cross)).sum()), test_within50_of_train=float((nn_test <= 50).mean()),
              random_pair_p0_1=float(np.percentile(D[np.triu_indices(n, 1)], 0.1)), families=int(len(set(fam))),
              nn_test_dist=nn_test.tolist())
    print(name, {k: v for k, v in ev.items() if k != 'nn_test_dist'})
    paths = [r['path'] for r in rows]
    feats = {'colour': np.array([colour_feats(p) for p in paths]), 'cnn': cnn_feats(paths)}
    def blocks(size): return np.array([f'{c}-{s // size}' for c, s in zip(cls, src)])
    def rgroups(size, seed=0):
        g = np.empty(n, dtype=object); r_ = np.random.default_rng(seed)
        for c in sorted(set(cls)):
            for j, i in enumerate(r_.permutation(np.where(cls == c)[0])): g[i] = f'{c}-{j // size}'
        return g
    res = {'evidence': ev}
    for fn, X in feats.items():
        R = {'official_split': metrics(y[te], clf().fit(X[tr], y[tr]).predict(X[te]), spoiled),
             'image_level_5fold': cv(X, y, spoiled), 'hash_family_5fold': cv(X, y, spoiled, fam)}
        for size in (5, 10, 20, 50, 100): R[f'capture_block_{size}'] = cv(X, y, spoiled, blocks(size))
        for size in (5, 20, 100): R[f'random_group_{size}'] = cv(X, y, spoiled, rgroups(size))
        res[fn] = R; print(name, fn, {k: round(v['acc'] * 100, 1) for k, v in R.items()})
    with open(os.path.join(A.out, f'assignments_{name}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['split', 'file', 'class', 'running_number', 'phash16', f'hash_family_T{T_FAMILY}', 'capture_block_20', 'capture_block_100'])
        b20, b100 = blocks(20), blocks(100)
        for i, r in enumerate(rows): w.writerow([r['split'], r['file'], r['cls'], r['src'], r['ph'], int(fam[i]), b20[i], b100[i]])
    return res, feats, y, cls

out = {}
rA, CA, sA = inv_meat(A.meat); out['meat'], FA, yA, cA = audit('meat', rA, CA, sA)
rB, CB, sB = inv_poultry(A.poultry); out['poultry'], FB, yB, cB = audit('poultry', rB, CB, sB)

# ---------------- cross-dataset transfer (binary fresh=0 vs spoiled=1) ----------------
keepA = cA != 'HALF-FRESH'; ybA = (cA[keepA] == 'SPOILED').astype(int); ybB = (cB == 'SPOILED').astype(int)
X_ = {fn: (FA[fn][keepA], FB[fn]) for fn in FA}
cross = {}
for fn, (XA, XB) in X_.items():
    d = {}
    d['meat_within_image5fold'] = cv(XA, ybA, 1)
    d['poultry_within_image5fold'] = cv(XB, ybB, 1)
    for a, b, Xa, ya, Xb, yb in (('meat', 'poultry', XA, ybA, XB, ybB), ('poultry', 'meat', XB, ybB, XA, ybA)):
        m = clf().fit(Xa, ya); p = m.predict(Xb); pr = m.predict_proba(Xb)[:, 1]
        r = metrics(yb, p, 1); r['auc'] = float(roc_auc_score(yb, pr)); r['pred_spoiled_share'] = float(p.mean())
        d[f'train_{a}_test_{b}'] = r
    cross[fn] = d; print('cross', fn, {k: (round(v['acc'] * 100, 1), round(v.get('auc', 0), 3)) for k, v in d.items()})
out['cross_dataset'] = cross
json.dump(out, open(os.path.join(A.out, 'results.json'), 'w'), indent=1)
print('done')
