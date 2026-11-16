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
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
# ---------- Fig 3 pipeline ----------
fig, ax = plt.subplots(figsize=(11, 3.3)); ax.set_xlim(-0.1, 11); ax.set_ylim(0, 3.3); ax.axis('off')
def box(x, y, w, h, t, fc='#ffffff', ec='#33475b', bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=9.5, fontweight='bold' if bold else 'normal')
def arr(x1, y1, x2, y2): ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='#33475b', lw=1.2))
box(0.0, 1.25, 1.55, 0.8, 'Food photo\n(phone\ncamera)', '#eef3f8', bold=True)
for i, (t, y) in enumerate((('Colour + texture\n(HSV, LBP, Lab)', 2.35), ('MobileNetV3\nfeatures', 1.25), ('ResNet-50\nfeatures', 0.15))):
    box(2.1, y, 1.8, 0.8, t); arr(1.5, 1.65, 2.1, y + 0.4)
    box(4.4, y, 1.7, 0.8, 'Logistic\nregression'); arr(3.9, y + 0.4, 4.4, y + 0.4); arr(6.1, y + 0.4, 6.7, 1.65)
box(6.7, 1.25, 1.6, 0.8, 'Average class\nprobabilities', '#eef3f8', bold=True)
box(8.8, 1.25, 1.1, 0.8, 'p ≥ τ ?', '#fff4d6', bold=True); arr(8.3, 1.65, 8.8, 1.65)
box(10.15, 2.2, 0.8, 0.7, 'Class\nlabel', '#e3f1e3'); box(10.15, 0.4, 0.8, 0.7, 'Refer to\nstaff', '#fde9e7')
arr(9.9, 1.8, 10.15, 2.5); arr(9.9, 1.5, 10.15, 0.8)
ax.text(9.7, 2.25, 'yes', fontsize=8.5, color='#555'); ax.text(9.7, 0.95, 'no', fontsize=8.5, color='#555')
ax.text(7.9, 0.35, 'τ chosen on training data so that\n≤ 1% of spoiled samples are accepted as fresh', fontsize=8, color='#555', ha='center')
plt.savefig(os.path.join(A.out, 'fig3_pipeline.png'), dpi=300, bbox_inches='tight'); plt.savefig(os.path.join(A.out, 'fig3_pipeline.pdf'), bbox_inches='tight'); plt.close()
# ---------- Fig 4 held-out examples ----------
def inv(root, splits, pat):
    return [(os.path.join(root, sp, f), m.group(1), int(m.group(2))) for sp in splits for f in sorted(os.listdir(os.path.join(root, sp))) for m in [re.match(pat, f)] if m]
R = json.load(open(os.path.join(A.res, 'freshcheck_results.json')))
def oof(rows, F, y, groups, tau):
    P = np.zeros((len(y), len(np.unique(y))))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups):
        P[te] = np.mean([make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(F[k][tr], y[tr]).predict_proba(F[k][te]) for k in ('colour', 'mobilenetv3', 'resnet50')], axis=0)
    return P
sets = []
m = inv(A.meat, ('train', 'test'), r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-'); Fm = dict(np.load(os.path.join(A.rob, 'features_meat.npz')))
ym = np.array([{'FRESH': 0, 'HALF-FRESH': 1, 'SPOILED': 2}[c] for _, c, _ in m]); gm = np.array([f'{c}-{s // 100}' for _, c, s in m])
sets.append(('Red meat', m, ym, oof(m, Fm, ym, gm, 0.5), 0.5, ['Fresh', 'Half-fresh', 'Spoiled']))
p = inv(A.poultry, ('training', 'validation'), r'^(segar|busuk)_(\d+)\.jpe?g$'); Fp = dict(np.load(os.path.join(A.rob, 'features_poultry.npz')))
yp = np.array([0 if c == 'segar' else 1 for _, c, _ in p]); gp = np.array([f'{c}-{s // 100}' for _, c, s in p])
tp = float(np.median(R['poultry']['session_grouped']['FreshCheck_refer']['taus']))
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
plt.tight_layout(); plt.savefig(os.path.join(A.out, 'fig4_examples.png'), dpi=300); plt.savefig(os.path.join(A.out, 'fig4_examples.pdf')); plt.close()
# ---------- Fig 5 results + site calibration ----------
S = json.load(open(os.path.join(A.res, 'freshcheck_site_results.json')))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
names = ['colour', 'mobilenetv3', 'resnet50', 'early_fusion', 'FreshCheck_fusion']; lab = ['Colour +\ntexture', 'MobileNetV3', 'ResNet-50', 'Early\nfusion', 'FreshCheck']
x = np.arange(len(names)); w = 0.38
for j, (ds, c) in enumerate((('meat', '#1f3a5f'), ('poultry', '#7aa6c2'))):
    g = R[ds]['session_grouped']; v = [g[n]['acc'][0] * 100 for n in names]; e = [g[n]['acc'][1] * 100 for n in names]
    a1.bar(x + (j - 0.5) * w, v, w, yerr=e, color=c, label={'meat': 'Red meat (3 classes)', 'poultry': 'Chicken breast'}[ds], capsize=2)
    for xi, vi in zip(x + (j - 0.5) * w, v): a1.text(xi, vi + 1.15, f'{vi:.1f}', ha='center', fontsize=7.5)
a1.set_xticks(x); a1.set_xticklabels(lab, fontsize=8.5); a1.set_ylim(80, 101); a1.set_ylabel('Accuracy, session-grouped 5-fold CV (%)')
a1.set_title('(a) FreshCheck vs single-model baselines', fontsize=10, loc='left'); a1.legend(fontsize=8, loc='lower right'); a1.spines[['top', 'right']].set_visible(False)
for key, c, l in (('meat_to_chicken', '#1f3a5f', 'Red meat → chicken'), ('chicken_to_meat', '#c0504d', 'Chicken → red meat')):
    ks = sorted(S[key], key=int); xs = [S[key][k]['local_images'][0] for k in ks]; ys = [S[key][k]['acc'][0] * 100 for k in ks]; es = [S[key][k]['acc'][1] * 100 for k in ks]
    a2.errorbar(xs, ys, yerr=es, marker='o', color=c, label=l + ' (other dataset + local)', capsize=3)
    k1 = [k for k in ks if S[key][k].get('local_only_acc')]
    a2.plot([S[key][k]['local_images'][0] for k in k1], [S[key][k]['local_only_acc'][0] * 100 for k in k1], ls='--', marker='s', ms=4, color=c, alpha=0.8, label=l.split(' → ')[1].capitalize() + ': local images only')
a2.axhline(50, color='#999', ls=':', lw=1); a2.text(330, 51, 'chance', fontsize=8, color='#777')
a2.set_xlabel('Labelled local images from the new dataset'); a2.set_ylabel('Accuracy on the new dataset (%)'); a2.set_ylim(40, 100)
a2.set_title('(b) Moving to a new dataset: effect of local labelled images', fontsize=10, loc='left'); a2.legend(fontsize=7.5, loc='center right', bbox_to_anchor=(1.0, 0.33)); a2.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(A.out, 'fig5_results.png'), dpi=300); plt.savefig(os.path.join(A.out, 'fig5_results.pdf')); plt.close()
print('figures done')
