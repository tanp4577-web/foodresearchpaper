"""Figure 2: what the leakage looks like (example images from the public datasets).
A: six consecutive red-meat files (one capture session) vs six random same-class files.
B: one fruit near-copy family; D: paneer-curry "good" class, distinct photographs with copy counts; E: six consecutive chicken files (no capture sessions) (pHash <= 30 bits) whose members are scattered in file order.
Usage: python fig_examples.py --meat <Phteven/data/raw> --fruit <fruit repo> --assign <dir> --d3 <dir> --out fig.png
"""
import argparse, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
ap = argparse.ArgumentParser()
for a in ('--meat', '--fruit', '--assign', '--d3', '--out', '--paneer', '--poultry'): ap.add_argument(a)
A = ap.parse_args()
def bits(h): return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))
def ham(a, b): return int((bits(a) != bits(b)).sum())

m = pd.read_csv(os.path.join(A.assign, 'assignments_meat.csv'))
f = m[m['class'] == 'FRESH'].sort_values('running_number').reset_index(drop=True)
best = None
for i in range(len(f) - 5):
    w = f.iloc[i:i + 6]
    if (w.running_number.diff().dropna() == 1).all():
        d = np.mean([ham(w.phash16.iloc[k], w.phash16.iloc[k + 1]) for k in range(5)])
        if best is None or d < best[0]: best = (d, i)
run = f.iloc[best[1]:best[1] + 6]
rnd = f.sample(6, random_state=3).sort_values('running_number')
fr = pd.read_csv(os.path.join(A.d3, 'assignments_fruit_3class_T30.csv'))
fam = fr.groupby('hash_family').filter(lambda g: len(g) >= 6 and g.fruit.nunique() == 1 and g.cls.nunique() == 1 and g.cls.iloc[0] != 'FRESH')
fid = fam.groupby('hash_family').n.apply(lambda s: s.max() - s.min()).idxmax()
g = fr[fr.hash_family == fid].sort_values('n')
g = g.iloc[np.linspace(0, len(g) - 1, 6).round().astype(int)]

def mpath(row): return os.path.join(A.meat, row.split, row.file)
import hashlib, glob
grp = {}
for p in sorted(glob.glob(os.path.join(A.paneer, '*.jp*g'))):
    grp.setdefault(hashlib.md5(open(p, 'rb').read()).hexdigest(), []).append(p)
pan = sorted(grp.values(), key=len, reverse=True)[:6]
pq = pd.read_csv(os.path.join(A.assign, 'assignments_poultry.csv'))
pf = pq[pq['class'] == 'FRESH'].sort_values('running_number').reset_index(drop=True)
pbest = None
for i in range(len(pf) - 5):
    w = pf.iloc[i:i + 6]
    if (w.running_number.diff().dropna() == 1).all():
        pbest = i; break
prun = pf.iloc[pbest:pbest + 6]
fig, ax = plt.subplots(5, 6, figsize=(12, 11))
def show(r, c, path, label):
    im = Image.open(path).convert('RGB'); im.thumbnail((300, 300)); ax[r, c].imshow(im); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[r, c].set_xlabel(label, fontsize=8)
for c, (_, row) in enumerate(run.iterrows()):
    lab = f'FRESH-{row.running_number}' + ('' if c == 0 else f'\n{ham(run.phash16.iloc[c - 1], row.phash16)} bits from previous')
    show(0, c, mpath(row), lab)
for c, (_, row) in enumerate(rnd.iterrows()): show(1, c, mpath(row), f'FRESH-{row.running_number}')
for c, (_, row) in enumerate(g.iterrows()): show(2, c, os.path.join(A.fruit, 'JPEGImages', f'{row.n}.jpg'), f'file {row.n} ({row.cls.lower()} {row.fruit})')
for c, fs in enumerate(pan): show(3, c, fs[0], f"{os.path.basename(fs[0])[:19]}\n{len(fs)} identical files")
ax[3, 0].set_ylabel('(d) Paneer curry,\n"good" class', fontsize=9)
for c, (_, row) in enumerate(prun.iterrows()):
    lab = f'segar_{row.running_number:04d}' + ('' if c == 0 else f'\n{ham(prun.phash16.iloc[c - 1], row.phash16)} bits from previous')
    show(4, c, os.path.join(A.poultry, row.split, row.file), lab)
ax[4, 0].set_ylabel('(e) Chicken breast:\nconsecutive files', fontsize=9)
ax[0, 0].set_ylabel('(a) Red meat:\nconsecutive files', fontsize=9)
ax[1, 0].set_ylabel('(b) Red meat:\nrandom same class', fontsize=9)
ax[2, 0].set_ylabel('(c) Fruit: one family\n(≤ 30 bits)', fontsize=9)
plt.tight_layout(); plt.savefig(A.out, dpi=300); plt.savefig(A.out.replace('.png', '.pdf'))
print('chicken run', list(prun.running_number), [ham(prun.phash16.iloc[k], prun.phash16.iloc[k+1]) for k in range(5)]); print('meat run', list(run.running_number), 'mean step bits', round(best[0], 1), 'fruit family', fid, 'size', int((fr.hash_family == fid).sum()), 'files', list(g.n))
