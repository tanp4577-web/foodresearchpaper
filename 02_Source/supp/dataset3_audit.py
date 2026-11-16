"""Third-dataset audit and a reusable provenance-leakage check (Supplementary material).

Part A - provenance leakage check (PLC), applicable to any labelled image folder:
  A1 order test: share of images whose nearest same-class neighbour by 256-bit pHash lies within
     +/-5 positions in within-class file order; null = 999 within-class permutations of file order;
     p = (1 + #null >= observed) / 1000.
  A2 near-duplicate density: share of images with a same-class neighbour within T = 60 bits.
  Run on the red-meat and poultry sets (from their assignment files) and on the fruit set.
Part B - fruit freshness set (zijianchen98/Fruit-freshness-detection-dataset; apple, banana,
  orange; Pascal-VOC labels). Images whose objects all carry one label are kept; binary task
  fresh_* vs rotten_* (normal_* dropped); with --three, fresh/normal/rotten. Frozen MobileNetV3-Small and ResNet-50 features with
  logistic regression (C = 1) under image-level 5-fold CV, hash-family 5-fold CV (T = 60), and a
  null of 20 random within-class regroupings with the same family-size distribution.
Usage: python dataset3_audit.py --fruit <repo dir> --assign <dir with assignments_*.csv> --out out/
       MNV3_WEIGHTS=<mnv3s_tf.pth> RESNET_WEIGHTS=<resnet50_a1_0.pth>
"""
import argparse, os, re, json, glob, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, torch, timm, imagehash
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ap = argparse.ArgumentParser(); ap.add_argument('--fruit'); ap.add_argument('--assign'); ap.add_argument('--out'); ap.add_argument('--three', action='store_true'); ap.add_argument('--T', type=int, default=60)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); torch.set_num_threads(os.cpu_count())
RES = {}; SUF = ('_3class' if A.three else '') + ('' if A.T == 60 else f'_T{A.T}')

def bits(hexes):
    return np.array([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in hexes], dtype=np.uint8)
def hamming(B):
    B = B.astype(np.int32); return B.shape[1] - (B @ B.T + (1 - B) @ (1 - B).T)

def plc(cls, order, hexes, T=60, perms=999, seed=0):
    """cls: class per image; order: running number per image; hexes: pHash hex strings."""
    cls = np.asarray(cls); order = np.asarray(order); D = hamming(bits(hexes)).astype(float)
    same = cls[:, None] == cls[None, :]; D[~same] = np.inf; np.fill_diagonal(D, np.inf)
    nn = D.argmin(1); dmin = D.min(1)
    rank = np.empty(len(cls), int)
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]; rank[idx[np.argsort(order[idx])]] = np.arange(len(idx))
    obs = float(np.mean(np.abs(rank - rank[nn]) <= 5))
    rng = np.random.default_rng(seed); null = []
    for _ in range(perms):
        r2 = rank.copy()
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]; r2[idx] = rng.permutation(r2[idx])
        null.append(np.mean(np.abs(r2 - r2[nn]) <= 5))
    null = np.array(null)
    comp = connected_components(csr_matrix(D <= T), directed=False)[1]
    sizes = np.bincount(comp)
    return dict(n=int(len(cls)), order_stat=round(100 * obs, 1), order_null_mean=round(100 * null.mean(), 1),
                order_null_max=round(100 * null.max(), 1), order_p=round((1 + int((null >= obs).sum())) / (perms + 1), 3),
                neardup_share=round(100 * float(np.mean(dmin <= T)), 1), families=int(comp.max() + 1),
                largest_family=int(sizes.max()), images_in_multi_families=round(100 * float(np.mean(sizes[comp] > 1)), 1)), comp

# ---- Part A on the two meat datasets
for ds in ('meat', 'poultry'):
    df = pd.read_csv(os.path.join(A.assign, f'assignments_{ds}.csv'))
    RES[f'plc_{ds}'], _ = plc(df['class'], df['running_number'], df['phash16'], T=A.T); print(ds, RES[f'plc_{ds}'], flush=True)

# ---- Part B: fruit set
rows = []
for f in glob.glob(os.path.join(A.fruit, 'Annotations', '*.xml')):
    n = int(re.search(r'(\d+)\.xml$', f).group(1)); labs = {o.find('name').text for o in ET.parse(f).getroot().findall('object')}
    if len(labs) == 1:
        lab = labs.pop(); state = lab.split('_')[0]
        if state in (('fresh', 'normal', 'rotten') if A.three else ('fresh', 'rotten')):
            rows.append(dict(n=n, path=os.path.join(A.fruit, 'JPEGImages', f'{n}.jpg'), fruit=lab.split('_')[1], cls=state.upper()))
fr = pd.DataFrame(rows).sort_values('n').reset_index(drop=True)
fr = fr[fr.path.map(os.path.exists)].reset_index(drop=True)
fr['phash16'] = [str(imagehash.phash(Image.open(p).convert('RGB'), hash_size=16)) for p in fr.path]
RES[f'plc_fruit{SUF}'], comp = plc(fr['cls'], fr['n'], fr['phash16'], T=A.T); fr['hash_family'] = comp
RES['fruit_counts'] = fr.groupby(['fruit', 'cls']).size().unstack().to_dict()
print('fruit', RES[f'plc_fruit{SUF}'], RES['fruit_counts'], flush=True)

def deep(name, wfile, paths):
    m = timm.create_model(name, pretrained=False, num_classes=0)
    sd = torch.load(wfile, map_location='cpu'); sd = {k: v for k, v in sd.items() if not k.startswith(('classifier', 'fc.'))}
    m.load_state_dict(sd, strict=False); m.eval(); cfg = timm.data.resolve_data_config({}, model=m); tf = timm.data.create_transform(**cfg)
    with torch.no_grad():
        return np.concatenate([m(torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + 32]])).numpy() for i in range(0, len(paths), 32)])
cache = os.path.join(A.out, 'features_fruit' + ('_3class' if A.three else '') + '.npz')
if os.path.exists(cache): F = dict(np.load(cache))
else:
    F = {'mobilenetv3': deep('tf_mobilenetv3_small_100', os.environ['MNV3_WEIGHTS'], list(fr.path)),
         'resnet50': deep('resnet50', os.environ['RESNET_WEIGHTS'], list(fr.path))}
    np.savez(cache, **F)
y = fr.cls.map({'FRESH': 0, 'NORMAL': 1, 'ROTTEN': 2} if A.three else {'FRESH': 0, 'ROTTEN': 1}).values.astype(int); fam = fr.hash_family.values

def cv(X, splits):
    pred = np.empty_like(y)
    for tr, te in splits:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)).fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
    K = y.max(); cm = confusion_matrix(y, pred, labels=list(range(K + 1))); return 100 * np.trace(cm) / cm.sum(), 100 * cm[K, 0] / cm[K].sum()
def regroup(seed):
    """random within-class groups with the same family-size multiset as the hash families"""
    rng = np.random.default_rng(seed); g = np.empty(len(y), int); nxt = 0
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); sizes = rng.permutation(np.bincount(fam[y == c])[np.bincount(fam[y == c]) > 0])
        pos = 0
        for s in sizes: g[idx[pos:pos + s]] = nxt; nxt += 1; pos += s
    return g
for feat, X in F.items():
    out = {}
    img = [cv(X, StratifiedKFold(5, shuffle=True, random_state=s).split(X, y)) for s in range(5)]
    hf = [cv(X, StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, fam)) for s in range(5)]
    null = [cv(X, StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, regroup(100 + s)))[0] for s in range(20)]
    best = max(a for a, _ in hf)
    out = dict(image_level=[round(np.mean([a for a, _ in img]), 1), round(np.std([a for a, _ in img]), 1)],
               image_level_rotten_as_fresh=round(np.mean([b for _, b in img]), 1),
               hash_family=[round(np.mean([a for a, _ in hf]), 1), round(np.std([a for a, _ in hf]), 1)],
               hash_family_rotten_as_fresh=round(np.mean([b for _, b in hf]), 1),
               null_range=[round(min(null), 1), round(max(null), 1)], null_mean=round(float(np.mean(null)), 1), null_sd=round(float(np.std(null)), 2),
               p=round((1 + sum(v <= best for v in null)) / 21, 3))
    RES[f'fruit{SUF}_{feat}'] = out; print(feat, out, flush=True)
fr.drop(columns='path').to_csv(os.path.join(A.out, f'assignments_fruit{SUF}.csv'), index=False)
json.dump(RES, open(os.path.join(A.out, f'dataset3_results{SUF}.json'), 'w'), indent=1)
