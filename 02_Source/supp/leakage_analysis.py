"""Capture-order leakage analysis of the Meat Freshness Image Dataset (Supplementary File S2).

Reproduces Section 4.3 / Table 5 / Figure 3 of
"Image-Based Food Freshness Assessment: Reference Labels, Validation and Kitchen Readiness".

Usage:
    pip install numpy scipy scikit-learn scikit-image pillow imagehash opencv-python-headless torch timm
    git clone https://github.com/TheLohia/Phteven          # mirror of the Kaggle dataset
    python leakage_analysis.py Phteven/data/raw out/

The data folder must contain train/ and test/ with files named like
FRESH-1000-_JPG.rf.<hash>.jpg (class prefix, running number). Outputs:
    out/assignments.csv   per-file split, class, running number, hash family, capture blocks
    out/results.json      all protocol results (accuracy, macro-F1, spoiled-as-fresh, confusion)
"""
import sys, os, re, json, csv, collections
import numpy as np
from PIL import Image
import imagehash
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.feature import local_binary_pattern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, f1_score

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
CLS = {'FRESH': 0, 'HALF-FRESH': 1, 'SPOILED': 2}
T_FAMILY = 60          # Hamming threshold (bits of 256) for near-duplicate families
REPS = 5               # fold assignments per grouped protocol

# ---------------- 1. inventory and perceptual hashes ----------------
rows = []
for split in ('train', 'test'):
    for f in sorted(os.listdir(os.path.join(ROOT, split))):
        if not f.lower().endswith('.jpg'): continue
        m = re.match(r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-', f)
        im = Image.open(os.path.join(ROOT, split, f)).convert('RGB')
        rows.append(dict(split=split, file=f, cls=m.group(1), src=int(m.group(2)),
                         ph=str(imagehash.phash(im, hash_size=16))))
n = len(rows); print('images', n, collections.Counter(r['cls'] for r in rows))
y = np.array([CLS[r['cls']] for r in rows]); split = np.array([r['split'] for r in rows])
src = np.array([r['src'] for r in rows]); cls = np.array([r['cls'] for r in rows])
B = np.stack([np.array([int(c, 16) >> i & 1 for c in r['ph'] for i in (3, 2, 1, 0)], np.uint8) for r in rows])
D = (B[:, None, :] != B[None, :, :]).sum(2)
_, fam = connected_components(csr_matrix(D <= T_FAMILY), directed=False)

# capture-order evidence: nearest perceptual neighbour within 5 running numbers, vs chance
Dn = D.copy(); np.fill_diagonal(Dn, 999); nn = Dn.argmin(1)
adj = float(((cls[nn] == cls) & (np.abs(src[nn] - src) <= 5)).mean())
rng = np.random.default_rng(0)
chance = float(np.mean([abs(src[rng.choice(np.where(cls == cls[i])[0])] - src[i]) <= 5 for i in range(n)]))
tr, te = np.where(split == 'train')[0], np.where(split == 'test')[0]
nn_test = D[np.ix_(te, tr)].min(1)
cross = {f for f in set(fam) if len(set(split[fam == f])) > 1}
evidence = dict(nn_same_class_within5=adj, chance_within5=chance,
                test_in_cross_split_family=int(np.isin(fam[te], list(cross)).sum()),
                test_within50_of_train=float((nn_test <= 50).mean()),
                random_pair_p0_1=float(np.percentile(D[np.triu_indices(n, 1)], 0.1)))
print(evidence)

# ---------------- 2. features ----------------
def colour_feats(path):
    """34 marginal HSV histogram bins (H 18, S 8, V 8), 10 uniform-LBP bins, Lab mean and SD (OpenCV, 224 x 224)."""
    im = cv2.resize(cv2.imread(path), (224, 224))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    h = [cv2.calcHist([hsv], [c], None, [b], [0, r_]).ravel() for c, b, r_ in ((0, 18, 180), (1, 8, 256), (2, 8, 256))]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lh = np.histogram(local_binary_pattern(g, 8, 1, 'uniform'), bins=10, range=(0, 10))[0]
    return np.concatenate([x / x.sum() for x in h] + [lh / lh.sum(), lab.reshape(-1, 3).mean(0) / 255, lab.reshape(-1, 3).std(0) / 255])
images = [Image.open(os.path.join(ROOT, r['split'], r['file'])).convert('RGB') for r in rows]
FEATS = {'colour': np.array([colour_feats(os.path.join(ROOT, r['split'], r['file'])) for r in rows])}
try:
    import torch, timm
    w = os.environ.get('MNV3_WEIGHTS')   # optional local ImageNet weights file
    m = timm.create_model('tf_mobilenetv3_small_100', pretrained=w is None, num_classes=0)
    if w: m.load_state_dict({k: v for k, v in torch.load(w, map_location='cpu').items() if not k.startswith('classifier')})
    m.eval()
    cfg = timm.data.resolve_data_config({}, model=m); tf = timm.data.create_transform(**cfg)
    with torch.no_grad():
        FEATS['cnn'] = np.concatenate([m(torch.stack([tf(im) for im in images[i:i + 64]])).numpy() for i in range(0, n, 64)])
except Exception as e:
    print('CNN features skipped:', e)

# ---------------- 3. protocols ----------------
def clf(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
def metrics(yt, yp):
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    return dict(acc=float((yt == yp).mean()), macroF1=float(f1_score(yt, yp, average='macro')),
                spoiled_as_fresh=float(cm[2, 0] / cm[2].sum()), spoiled_missed=float(1 - cm[2, 2] / cm[2].sum()), cm=cm.tolist())
def cv(X, groups=None, reps=REPS):
    res = []
    for s in range(reps):
        sp = StratifiedGroupKFold(5, shuffle=True, random_state=s) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=s)
        yp = np.zeros_like(y)
        for a, b in (sp.split(X, y, groups) if groups is not None else sp.split(X, y)):
            yp[b] = clf().fit(X[a], y[a]).predict(X[b])
        res.append(metrics(y, yp))
    out = {k: float(np.mean([r[k] for r in res])) for k in ('acc', 'macroF1', 'spoiled_as_fresh', 'spoiled_missed')}
    out['acc_sd'] = float(np.std([r['acc'] for r in res])); out['cm_mean'] = np.mean([r['cm'] for r in res], 0).tolist()
    return out
def blocks(size): return np.array([f'{c}-{s // size}' for c, s in zip(cls, src)])
def random_groups(size, seed=0):
    g = np.empty(n, dtype=object); r = np.random.default_rng(seed)
    for c in set(cls):
        idx = r.permutation(np.where(cls == c)[0])
        for j, i in enumerate(idx): g[i] = f'{c}-{j // size}'
    return g
results = {'evidence': evidence}
for name, X in FEATS.items():
    R = {'official_split': metrics(y[te], clf().fit(X[tr], y[tr]).predict(X[te])),
         'image_level_5fold': cv(X), 'hash_family_5fold': cv(X, fam)}
    for size in (5, 10, 20, 50, 100):
        R[f'capture_block_{size}'] = cv(X, blocks(size))
    for size in (5, 20, 100):
        R[f'random_group_{size}'] = cv(X, random_groups(size))
    results[name] = R
    print(name, {k: round(v['acc'] * 100, 1) for k, v in R.items()})
json.dump(results, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
with open(os.path.join(OUT, 'assignments.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['split', 'file', 'class', 'running_number', 'phash16', f'hash_family_T{T_FAMILY}', 'capture_block_20', 'capture_block_100'])
    b20, b100 = blocks(20), blocks(100)
    for i, r in enumerate(rows): w.writerow([r['split'], r['file'], r['cls'], r['src'], r['ph'], int(fam[i]), b20[i], b100[i]])
print('done')
