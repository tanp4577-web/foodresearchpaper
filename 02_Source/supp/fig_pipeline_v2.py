"""Figure 1 (v2): FreshCheck pipeline drawn with a real example photo, its real colour/texture features and
real per-model probabilities from models trained WITHOUT that photo's capture session (red meat, 3 classes)."""
import argparse, os, re, numpy as np, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from skimage.feature import local_binary_pattern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ap = argparse.ArgumentParser(); [ap.add_argument(a) for a in ('--meat', '--rob', '--out', '--pick')]; A = ap.parse_args()
rows = []
for sp in ('train', 'test'):
    for f in sorted(os.listdir(os.path.join(A.meat, sp))):
        m = re.match(r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-', f)
        if m: rows.append((sp, f, m.group(1), int(m.group(2))))
F = dict(np.load(os.path.join(A.rob, 'features_meat.npz'))); KEYS = ('colour', 'mobilenetv3', 'resnet50')
y = np.array([{'FRESH': 0, 'HALF-FRESH': 1, 'SPOILED': 2}[r[2]] for r in rows]); g = np.array([f'{r[2]}-{r[3] // 100}' for r in rows])
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
def oof(i):
    tr = g != g[i]
    return {k: lr().fit(F[k][tr], y[tr]).predict_proba(F[k][i:i + 1])[0] for k in KEYS}
if A.pick is None:   # search spoiled/half-fresh test images where single models disagree but the average is right
    rng = np.random.default_rng(1); best = None
    for i in rng.choice(np.where(y > 0)[0], 60, replace=False):
        P = oof(i); avg = np.mean(list(P.values()), 0); preds = {k: P[k].argmax() for k in KEYS}
        if avg.argmax() == y[i] and len(set(preds.values())) > 1 and avg.max() > 0.6:
            best = i; break
    i = best if best is not None else int(rng.choice(np.where(y == 2)[0]))
else:
    i = [j for j, r in enumerate(rows) if r[1] == A.pick][0]
P = oof(i); avg = np.mean([P[k] for k in KEYS], 0); print(rows[i], {k: np.round(v, 2) for k, v in P.items()}, np.round(avg, 2))
img = cv2.imread(os.path.join(A.meat, rows[i][0], rows[i][1])); rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im = cv2.resize(img, (224, 224)); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
lbp = local_binary_pattern(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 8, 1, 'uniform')
CL = ['Fresh', 'Half-fresh', 'Spoiled']; COL = ['#4c9a52', '#d9a33b', '#c0504d']
INK = '#2b3a4a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig = plt.figure(figsize=(13, 5.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis('off')
def card(x, y0, w, h, fc='#ffffff', ec='#c9d3dd'):
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=1))
def arrow(x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=11, color='#7b8a99', lw=1.2, connectionstyle=f'arc3,rad={rad}'))
def step(x, y0, n, t):
    ax.add_patch(plt.Circle((x, y0), 0.16, color=INK)); ax.text(x, y0, str(n), color='white', ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax.text(x + 0.25, y0, t, va='center', fontsize=10, fontweight='bold', color=INK)
# 1 photo
step(0.35, 4.9, 1, 'Photo'); ia = fig.add_axes([0.02, 0.27, 0.15, 0.37]); ia.imshow(rgb); ia.axis('off')
ax.text(1.25, 1.15, 'Phone or tablet photo\nof the food', ha='center', fontsize=8.5, color='#555')
# 2 features
step(2.55, 4.9, 2, 'Three ways of describing it')
ys = [3.55, 2.1, 0.65]
for yy in ys: card(2.45, yy, 2.75, 1.2, '#f7f9fb')
a = fig.add_axes([2.55 / 13, 3.8 / 5.2, 1.25 / 13, 0.85 / 5.2])
hue = cv2.calcHist([hsv], [0], None, [18], [0, 180]).ravel(); hue /= hue.sum()
cols = [matplotlib.colors.hsv_to_rgb((k / 18 + 1 / 36, 0.75, 0.9)) for k in range(18)]
a.bar(range(18), hue, color=cols, width=0.9); a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
b = fig.add_axes([3.95 / 13, 3.8 / 5.2, 0.85 / 13, 0.85 / 5.2]); b.imshow(lbp, cmap='gray'); b.axis('off')
ax.text(3.83, 3.75, 'Colour + texture (50 values)', ha='center', fontsize=8.5, color=INK, va='top')
for yy, key, n, name in ((2.1, 'mobilenetv3', 1024, 'MobileNetV3: 1,024 values'), (0.65, 'resnet50', 2048, 'ResNet-50: 2,048 values')):
    c = fig.add_axes([2.6 / 13, (yy + 0.38) / 5.2, 2.45 / 13, 0.7 / 5.2])
    v = F[key][i][:256]; c.imshow(v.reshape(8, 32), aspect='auto', cmap='Blues'); c.axis('off')
    ax.text(3.83, yy + 0.32, name + ' (first 256 shown)', ha='center', fontsize=8.5, color=INK, va='top')
for yy in ys: arrow(2.32, 2.85, 2.43, yy + 0.6, rad=0.0)
# 3 per-model probabilities
step(5.65, 4.9, 3, 'Each model votes')
names = {'colour': 'Colour model', 'mobilenetv3': 'MobileNetV3 model', 'resnet50': 'ResNet-50 model'}
for yy, key in zip(ys, KEYS):
    card(5.55, yy, 2.3, 1.2); arrow(5.22, yy + 0.6, 5.53, yy + 0.6)
    pa = fig.add_axes([5.95 / 13, (yy + 0.12) / 5.2, 1.75 / 13, 0.75 / 5.2])
    pa.barh(range(3)[::-1], P[key], color=COL, height=0.7); pa.set_xlim(0, 1); pa.set_yticks(range(3)[::-1]); pa.set_yticklabels(CL, fontsize=7.5)
    pa.set_xticks([]); [s.set_visible(False) for s in pa.spines.values()]
    for k in range(3): pa.text(min(P[key][k] + 0.03, 0.8), 2 - k, f'{P[key][k]:.2f}', va='center', fontsize=7.5)
    ax.text(6.7, yy + 1.1, names[key], ha='center', va='top', fontsize=8.5, color=INK)
# 4 average
step(8.25, 4.9, 4, 'Average the votes')
card(8.15, 1.6, 2.2, 2.4, '#eef3f8'); [arrow(7.87, yy + 0.6, 8.13, 2.8, rad=0) for yy in ys]
va = fig.add_axes([8.55 / 13, 2.0 / 5.2, 1.6 / 13, 1.5 / 5.2])
va.bar(range(3), avg, color=COL, width=0.65); va.set_ylim(0, 1.05); va.set_xticks(range(3)); va.set_xticklabels(['Fresh', 'Half-\nfresh', 'Spoiled'], fontsize=7.5)
va.axhline(0.50, color=INK, ls='--', lw=1); va.text(2.45, 0.52, 'τ', fontsize=10, color=INK)
va.set_yticks([0, 0.5, 1]); va.tick_params(labelsize=7); [va.spines[s].set_visible(False) for s in ('top', 'right')]
for k in range(3): va.text(k, avg[k] + 0.03, f'{avg[k]:.2f}', ha='center', fontsize=7.5)
ax.text(9.25, 3.85, 'Averaged probability', ha='center', va='top', fontsize=8.5, color=INK)
# 5 decision
step(10.75, 4.9, 5, 'Decide')
card(10.7, 3.0, 2.1, 1.05, '#e6f2e6', '#9cc59c'); ax.text(11.75, 3.72, 'Top probability ≥ τ', ha='center', fontsize=8.5, fontweight='bold', color='#2e6b33')
ax.text(11.75, 3.3, 'show the class\n(here: ' + CL[int(avg.argmax())] + ')', ha='center', fontsize=8.5, color='#2e6b33', va='center')
card(10.7, 1.5, 2.1, 1.05, '#fbe9e7', '#e3a49c'); ax.text(11.75, 2.22, 'Top probability < τ', ha='center', fontsize=8.5, fontweight='bold', color='#9b3a30')
ax.text(11.75, 1.8, '"Refer to staff":\na person checks it', ha='center', fontsize=8.5, color='#9b3a30', va='center')
arrow(10.37, 2.9, 10.68, 3.5, rad=-0.15); arrow(10.37, 2.6, 10.68, 2.0, rad=0.15)
ax.text(11.75, 1.05, 'τ is set on training photos so that\nat most 1% of spoiled food is\naccepted as fresh (dashed line: τ = 0.50,\nthe red-meat model; 0.74 for chicken)', ha='center', va='top', fontsize=7.5, color='#555')
plt.savefig(os.path.join(A.out, 'fig1_pipeline_v2.png'), dpi=300, bbox_inches='tight', pad_inches=0.05); plt.savefig(os.path.join(A.out, 'fig1_pipeline_v2.pdf'), bbox_inches='tight', pad_inches=0.05); print('done')
