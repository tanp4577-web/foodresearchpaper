"""Figures 4 and 5 for FreshCheck (multi-level): results/transfer and held-out examples."""
"""Figures for the FreshCheck paper: (3) system pipeline, (4) held-out example decisions, (5) results + site calibration."""
import os, re, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import argparse
ap = argparse.ArgumentParser()
for a in ('--meat', '--poultry', '--rob', '--res', '--out'): ap.add_argument(a)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
KEYS = ('colour', 'colour_fine', 'mobilenetv3', 'g_l2', 'g_l3', 'g_l4', 'p_l2', 'p_l3', 'p_l4')
# ---------- Fig 4 held-out examples ----------
def inv(root, splits, pat):
    return [(os.path.join(root, sp, f), m.group(1), int(m.group(2))) for sp in splits for f in sorted(os.listdir(os.path.join(root, sp))) for m in [re.match(pat, f)] if m]
R = json.load(open(os.path.join(A.res, 'freshcheck_v2_results.json')))
def oof(rows, F, y, groups, tau):
    P = np.zeros((len(y), len(np.unique(y))))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups):
        P[te] = np.mean([make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(F[k][tr], y[tr]).predict_proba(F[k][te]) for k in KEYS], axis=0)
    return P
sets = []
m = inv(A.meat, ('train', 'test'), r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-'); Fm = dict(np.load(os.path.join(A.rob, 'features_meat.npz'))); Fm.update(np.load(os.path.join(A.rob, 'features_meat_v2.npz')))
ym = np.array([{'FRESH': 0, 'HALF-FRESH': 1, 'SPOILED': 2}[c] for _, c, _ in m]); gm = np.array([f'{c}-{s // 100}' for _, c, s in m])
sets.append(('Red meat', m, ym, oof(m, Fm, ym, gm, 0.5), 0.5, ['Fresh', 'Half-fresh', 'Spoiled']))
p = inv(A.poultry, ('training', 'validation'), r'^(segar|busuk)_(\d+)\.jpe?g$'); Fp = dict(np.load(os.path.join(A.rob, 'features_poultry.npz'))); Fp.update(np.load(os.path.join(A.rob, 'features_poultry_v2.npz')))
yp = np.array([0 if c == 'segar' else 1 for _, c, _ in p]); gp = np.array([f'{c}-{s // 100}' for _, c, s in p])
tp = round(R['poultry']['session_grouped']['taus_v2']['median'][0], 2)
sets.append(('Chicken breast', p, yp, oof(p, Fp, yp, gp, tp), tp, ['Fresh', 'Spoiled']))
fig, axes = plt.subplots(2, 6, figsize=(13, 5.4))
for r, (name, rows, y, P, tau, cl) in enumerate(sets):
    pred = P.argmax(1); conf = P.max(1); refer = conf < tau; rng = np.random.default_rng(1)
    picks = []
    for c in range(len(cl)):
        ok = np.where((y == c) & (pred == c) & ~refer)[0]; picks.append(('correct', rng.choice(ok)))
    ref = np.where(refer)[0]
    if len(ref): picks.append(('referred', ref[np.argmin(conf[ref])]))
    err = np.where((pred != y) & ~refer)[0]
    while len(picks) < 6 and len(err): i = rng.choice(err); err = err[err != i]; picks.append(('error', i))
    for c in range(6):
        a = axes[r, c]; a.set_xticks([]); a.set_yticks([])
        if c >= len(picks): a.axis('off'); continue
        kind, i = picks[c]; im = Image.open(rows[i][0]).convert('RGB'); im.thumbnail((300, 300)); a.imshow(im)
        dec = 'Refer to staff' if refer[i] else cl[pred[i]]
        col = {'correct': '#2e7d32', 'referred': '#b26a00', 'error': '#c62828'}[kind]
        for s in a.spines.values(): s.set_edgecolor(col); s.set_linewidth(3)
        a.set_xlabel(f'True: {cl[y[i]]}\nFreshCheck: {dec} ({conf[i]:.2f})', fontsize=8.5, color=col)
    axes[r, 0].set_ylabel(name, fontsize=10, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(A.out, 'fig5_examples_v2.png'), dpi=300); plt.savefig(os.path.join(A.out, 'fig5_examples_v2.pdf')); plt.close()
# ---------- Fig 5 results + site calibration ----------
S = R['transfer']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
names = ['colour', 'mobilenetv3', 'resnet50', 'g_l2', 'FreshCheck_v1', 'FreshCheck_v2']; lab = ['Colour +\ntexture', 'MobileNetV3', 'ResNet-50\nfinal layer', 'ResNet-50\nlayer 2', 'Final-layer\nfusion', 'FreshCheck\n(multi-level)']
x = np.arange(len(names)); w = 0.38
for j, (ds, c) in enumerate((('meat', '#1f3a5f'), ('poultry', '#7aa6c2'))):
    g = R[ds]['session_grouped']; v = [g[n]['acc'][0] * 100 for n in names]; e = [g[n]['acc'][1] * 100 for n in names]
    a1.bar(x + (j - 0.5) * w, v, w, yerr=e, color=c, label={'meat': 'Red meat (3 classes)', 'poultry': 'Chicken breast'}[ds], capsize=2)
    for xi, vi in zip(x + (j - 0.5) * w, v): a1.text(xi, vi + 1.15, f'{vi:.1f}', ha='center', fontsize=7.5)
a1.set_xticks(x); a1.set_xticklabels(lab, fontsize=8.5); a1.set_ylim(80, 101); a1.set_ylabel('Accuracy, session-grouped 5-fold CV (%)')
a1.set_title('(a) FreshCheck vs baselines', fontsize=10, loc='left'); a1.legend(fontsize=8, loc='lower right'); a1.spines[['top', 'right']].set_visible(False)
for key, c, l in (('meat_to_poultry', '#1f3a5f', 'Red meat → chicken'), ('poultry_to_meat', '#c0504d', 'Chicken → red meat')):
    ks = sorted(S[key], key=int); xs = [S[key][k]['local_images'] for k in ks]; ys = [S[key][k]['v2']['acc'][0] * 100 for k in ks]; es = [S[key][k]['v2']['acc'][1] * 100 for k in ks]
    a2.errorbar(xs, ys, yerr=es, marker='o', color=c, label=l + ' (other dataset + local)', capsize=3)
    k1 = [k for k in ks if S[key][k].get('v2_local_only')]
    a2.plot([S[key][k]['local_images'] for k in k1], [S[key][k]['v2_local_only']['acc'][0] * 100 for k in k1], ls='--', marker='s', ms=4, color=c, alpha=0.8, label=l.split(' → ')[1].capitalize() + ': local images only')
a2.axhline(50, color='#999', ls=':', lw=1); a2.text(330, 51, 'chance', fontsize=8, color='#777')
a2.set_xlabel('Labelled local images from the new dataset'); a2.set_ylabel('Accuracy on the new dataset (%)'); a2.set_ylim(40, 100)
a2.set_title('(b) Moving to a new dataset: effect of local labelled images', fontsize=10, loc='left'); a2.legend(fontsize=7.5, loc='center right', bbox_to_anchor=(1.0, 0.33)); a2.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(A.out, 'fig4_results_v2.png'), dpi=300); plt.savefig(os.path.join(A.out, 'fig4_results_v2.pdf')); plt.close()
print('figures done')
