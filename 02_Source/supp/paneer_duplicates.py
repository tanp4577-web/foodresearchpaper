"""Exact-copy audit of the paneer-curry dataset (Basak & Sarkar, 2025; doi:10.17632/4898ctyf5n.1).
Writes per-file MD5/SHA-256 and copy-group sizes, and simulates random splits.
Usage: python paneer_duplicates.py <folder with the 'Good' images> <out.csv>
"""
import sys, os, glob, hashlib, csv, numpy as np
d, out = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(d, '*.jp*g'))):
    b = open(p, 'rb').read(); rows.append([os.path.basename(p), len(b), hashlib.md5(b).hexdigest(), hashlib.sha256(b).hexdigest()])
groups = {}
for r in rows: groups.setdefault(r[3], []).append(r)
gid = {h: i for i, h in enumerate(sorted(groups, key=lambda h: -len(groups[h])))}
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['file', 'bytes', 'md5', 'sha256', 'copy_group', 'group_size'])
    for r in rows: w.writerow(r + [gid[r[3]], len(groups[r[3]])])
lab = np.array([gid[r[3]] for r in rows]); n = len(lab); rng = np.random.default_rng(0)
sh = []
for _ in range(1000):
    p = rng.permutation(n); te, tr = p[:n // 5], set(lab[p[n // 5:]]); sh.append(np.mean([l in tr for l in lab[te]]))
print(f'files {n}, distinct {len(groups)}, group sizes {sorted((len(v) for v in groups.values()), reverse=True)}')
print(f'80/20 splits: share of test files with an identical copy in training, mean {np.mean(sh):.3f}, min {np.min(sh):.3f}')
