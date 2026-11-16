"""Fine-tuned CNN check for the leakage audit (Supplementary File S2).

Full fine-tuning of ImageNet MobileNetV3-Small (timm tf_mobilenetv3_small_100), 160 px, 4 epochs,
AdamW (lr 5e-4, wd 1e-4, cosine), batch 32, random-resized-crop + horizontal flip.
Protocols on the red-meat dataset (3 classes), one fold assignment (seed 0), 5 folds each:
image-level, random groups of 100, capture-order blocks of 100.
Cross-dataset (binary fresh vs spoiled, half-fresh left out): train on one dataset, test on the other.
Usage: python finetune_audit.py --meat Phteven/data/raw --poultry "<repo>/dataset_citra_dada_ayam/dataset 200x200" --out out/
"""
import argparse, os, re, json, time
import numpy as np, torch, timm
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

ap = argparse.ArgumentParser(); ap.add_argument('--meat'); ap.add_argument('--poultry'); ap.add_argument('--out')
ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--protocols', default='image_level,random_group_100,capture_block_100'); ap.add_argument('--no_cross', action='store_true'); ap.add_argument('--size', type=int, default=160)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
torch.set_num_threads(os.cpu_count()); torch.manual_seed(A.seed); np.random.seed(A.seed)
RES = os.path.join(A.out, 'finetune_results.json')
res = json.load(open(RES)) if os.path.exists(RES) else {}
def save(): json.dump(res, open(RES, 'w'), indent=1)

def load(paths, size):
    s = int(size * 1.1)
    return torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert('RGB').resize((s, s)))).permute(2, 0, 1) for p in paths])
meat = []
for sp in ('train', 'test'):
    for f in sorted(os.listdir(os.path.join(A.meat, sp))):
        m = re.match(r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-', f)
        if m: meat.append((os.path.join(A.meat, sp, f), m.group(1), int(m.group(2))))
poul = []
for sp in ('training', 'validation'):
    for f in sorted(os.listdir(os.path.join(A.poultry, sp))):
        m = re.match(r'^(segar|busuk)_(\d+)\.jpe?g$', f)
        if m: poul.append((os.path.join(A.poultry, sp, f), {'segar': 'FRESH', 'busuk': 'SPOILED'}[m.group(1)], int(m.group(2))))
XM = load([p for p, _, _ in meat], A.size); XP = load([p for p, _, _ in poul], A.size)
cm_ = np.array([c for _, c, _ in meat]); sm = np.array([s for _, _, s in meat]); cp = np.array([c for _, c, _ in poul])
cfg = timm.data.resolve_data_config({}, model=timm.create_model('tf_mobilenetv3_small_100', pretrained=False))
MEAN = torch.tensor(cfg['mean']).view(1, 3, 1, 1); STD = torch.tensor(cfg['std']).view(1, 3, 1, 1)
W = os.environ.get('MNV3_WEIGHTS')

def model(nc):
    m = timm.create_model('tf_mobilenetv3_small_100', pretrained=W is None, num_classes=nc)
    if W:
        sd = {k: v for k, v in torch.load(W, map_location='cpu').items() if not k.startswith('classifier')}
        m.load_state_dict(sd, strict=False)
    return m
def augment(x):
    n, _, H, Wd = x.shape; out = torch.empty(n, 3, A.size, A.size)
    for i in range(n):
        s = np.random.uniform(0.7, 1.0); h = int(H * np.sqrt(s)); w = h
        t = np.random.randint(0, H - h + 1); l = np.random.randint(0, Wd - w + 1)
        c = x[i:i + 1, :, t:t + h, l:l + w].float() / 255
        c = F.interpolate(c, size=(A.size, A.size), mode='bilinear', align_corners=False)
        if np.random.rand() < 0.5: c = c.flip(3)
        out[i] = c[0]
    return (out - MEAN) / STD
def center(x):
    o = (x.shape[2] - A.size) // 2
    return (x[:, :, o:o + A.size, o:o + A.size].float() / 255 - MEAN) / STD
def train(X, y, nc):
    m = model(nc); opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-4)
    steps = A.epochs * int(np.ceil(len(y) / 32)); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    yt = torch.tensor(y)
    for ep in range(A.epochs):
        m.train(); perm = np.random.permutation(len(y))
        for i in range(0, len(y), 32):
            b = perm[i:i + 32]; opt.zero_grad()
            F.cross_entropy(m(augment(X[b])), yt[b]).backward(); opt.step(); sch.step()
    return m
@torch.no_grad()
def predict(m, X):
    m.eval(); return torch.cat([F.softmax(m(center(X[i:i + 64])), 1) for i in range(0, len(X), 64)]).numpy()
def metrics(yt, P, spoiled):
    yp = P.argmax(1); cm = confusion_matrix(yt, yp, labels=list(range(P.shape[1])))
    return dict(acc=float((yt == yp).mean()), macroF1=float(f1_score(yt, yp, average='macro')),
                spoiled_as_fresh=float(cm[spoiled, 0] / cm[spoiled].sum()), spoiled_missed=float(1 - cm[spoiled, spoiled] / cm[spoiled].sum()), cm=cm.tolist())

# ---- red meat, 3 classes ----
ym = np.array([{'FRESH': 0, 'HALF-FRESH': 1, 'SPOILED': 2}[c] for c in cm_])
def rgroups(size):
    g = np.empty(len(ym), dtype=object); r = np.random.default_rng(0)
    for c in sorted(set(cm_)):
        for j, i in enumerate(r.permutation(np.where(cm_ == c)[0])): g[i] = f'{c}-{j // size}'
    return g
protocols = {'image_level': None, 'random_group_100': rgroups(100), 'capture_block_100': np.array([f'{c}-{s // 100}' for c, s in zip(cm_, sm)])}
for name, g in [(k, v) for k, v in protocols.items() if k in A.protocols.split(',')]:
    key = f'meat_{name}'
    if key in res and res[key].get('done'): continue
    sp = StratifiedKFold(5, shuffle=True, random_state=A.seed) if g is None else StratifiedGroupKFold(5, shuffle=True, random_state=A.seed)
    P = np.zeros((len(ym), 3)); folds = res.get(key, {}).get('folds', [])
    for k, (a, b) in enumerate(sp.split(XM, ym) if g is None else sp.split(XM, ym, g)):
        t0 = time.time(); m = train(XM[a], ym[a], 3); P[b] = predict(m, XM[b])
        folds.append(dict(fold=k, acc=float((P[b].argmax(1) == ym[b]).mean()), sec=round(time.time() - t0)))
        res[key] = dict(folds=folds); save(); print(key, folds[-1], flush=True)
    res[key] = dict(metrics(ym, P, 2), folds=folds, done=True); save(); print(key, round(res[key]['acc'] * 100, 2), flush=True)

# ---- cross-dataset, binary ----
keep = cm_ != 'HALF-FRESH'; yb = (cm_[keep] == 'SPOILED').astype(int); yp_ = (cp == 'SPOILED').astype(int)
for a_, b_, Xa, ya, Xb, yb_ in ([] if A.no_cross else (('meat', 'poultry', XM[keep], yb, XP, yp_), ('poultry', 'meat', XP, yp_, XM[keep], yb))):
    key = f'cross_train_{a_}_test_{b_}'
    if key in res: continue
    m = train(Xa, ya, 2); P = predict(m, Xb); r = metrics(yb_, P, 1); r['auc'] = float(roc_auc_score(yb_, P[:, 1]))
    res[key] = r; save(); print(key, round(r['acc'] * 100, 1), round(r['auc'], 3), flush=True)
print('done', flush=True)
