"""Quantum-kernel and neuro-inspired (V1 Gabor) extensions of FreshCheck, under session-grouped CV.
Quantum part: 8-qubit ZZ feature map (Havlicek et al., 2019 style, 2 repetitions) SIMULATED exactly as a statevector
on a classical computer; fidelity kernel |<psi(x)|psi(x')>|^2 used in an SVM. Inputs: PCA(8) of frozen ResNet-50 features.
Neuro part: bank of Gabor filters (model of V1 simple/complex cells) on luminance and colour-opponent channels.
Usage: python quantum_neuro.py --meat <Phteven/data/raw> --poultry <dataset 200x200> --rob <dir with features_*.npz> --out <dir>"""
import argparse, os, re, json, numpy as np, cv2
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
ap = argparse.ArgumentParser(); [ap.add_argument(a) for a in ('--meat', '--poultry', '--rob', '--out')]
ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--qubits', type=int, default=8)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)

def inv(root, splits, pat):
    rows = []
    for sp in splits:
        for f in sorted(os.listdir(os.path.join(root, sp))):
            m = re.match(pat, f)
            if m: rows.append((m.group(1), int(m.group(2)), os.path.join(root, sp, f)))
    return rows

# ---------- neuro-inspired V1 Gabor features ----------
KERNELS = [cv2.getGaborKernel((21, 21), sigma=s, theta=t, lambd=l, gamma=0.5, psi=0)
           for (s, l) in ((2.0, 4.0), (3.5, 8.0), (6.0, 16.0)) for t in np.arange(4) * np.pi / 4]
KERNELS_Q = [cv2.getGaborKernel((21, 21), sigma=s, theta=t, lambd=l, gamma=0.5, psi=np.pi / 2)
             for (s, l) in ((2.0, 4.0), (3.5, 8.0), (6.0, 16.0)) for t in np.arange(4) * np.pi / 4]
def gabor_features(path):
    im = cv2.resize(cv2.imread(path), (128, 128)).astype(np.float32) / 255
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    chans = [lab[..., 0] / 100, (lab[..., 1] + 128) / 255, (lab[..., 2] + 128) / 255]  # luminance + red-green + blue-yellow opponent
    out = []
    for c in chans:
        c = c - c.mean()
        for k, kq in zip(KERNELS, KERNELS_Q):
            e = np.sqrt(cv2.filter2D(c, -1, k) ** 2 + cv2.filter2D(c, -1, kq) ** 2)   # complex-cell energy
            out += [e.mean(), e.std()]
    return np.array(out, np.float32)   # 3 channels x 12 filters x 2 = 72 values

# ---------- quantum fidelity kernel (exact statevector simulation) ----------
def zz_states(X, reps=2):
    n = X.shape[1]; N = 2 ** n
    z = 1 - 2 * ((np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1)          # N x n, entries +-1
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    H = np.array([[1]])
    for _ in range(n): H = np.kron(H, np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    S = np.zeros((len(X), N), complex)
    for r, x in enumerate(X):
        phase = z @ x + sum((np.pi - x[i]) * (np.pi - x[j]) * z[:, i] * z[:, j] for i, j in pairs)
        D = np.exp(1j * phase)
        psi = np.zeros(N, complex); psi[0] = 1
        for _ in range(reps): psi = D * (H @ psi)
        S[r] = psi
    return S
def qkernel(SA, SB): return np.abs(SA.conj() @ SB.T) ** 2

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))

def run(name, rows, F, y, g, spoiled, fresh):
    G = np.stack([gabor_features(p) for _, _, p in rows]); F = dict(F, gabor=G)
    res = {}; BW = []
    def add(k, acc, saf): res.setdefault(k, {'acc': [], 'saf': []}); res[k]['acc'].append(acc); res[k]['saf'].append(saf)
    for seed in range(A.seeds):
        P = {k: np.zeros((len(y), len(np.unique(y)))) for k in ('colour', 'mobilenetv3', 'resnet50', 'gabor', 'qsvm', 'rbf_svm_pca8', 'lr_pca8')}
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(F['colour'], y, g):
            for k in ('colour', 'mobilenetv3', 'resnet50', 'gabor'):
                P[k][te] = lr().fit(F[k][tr], y[tr]).predict_proba(F[k][te])
            pca = make_pipeline(StandardScaler(), PCA(A.qubits, random_state=0)).fit(F['resnet50'][tr])
            Ztr, Zte = pca.transform(F['resnet50'][tr]), pca.transform(F['resnet50'][te])
            sd = Ztr.std(0)
            def qfit(Za, ya, bw):
                Sa = zz_states(bw * Za / sd); return Sa, SVC(kernel='precomputed', C=1.0, probability=True, random_state=seed).fit(qkernel(Sa, Sa), ya)
            # bandwidth chosen by inner grouped 3-fold CV on the training fold only (avoids exponential kernel concentration)
            best = (-1, None)
            for bw in (0.01, 0.02, 0.05, 0.1):
                acc = []
                for itr, ite in StratifiedGroupKFold(3, shuffle=True, random_state=100 + seed).split(Ztr, y[tr], g[tr]):
                    Sa, m_ = qfit(Ztr[itr], y[tr][itr], bw)
                    acc.append((m_.predict(qkernel(zz_states(bw * Ztr[ite] / sd), Sa)) == y[tr][ite]).mean())
                if np.mean(acc) > best[0]: best = (np.mean(acc), bw)
            BW.append(best[1]); Str, q = qfit(Ztr, y[tr], best[1])
            P['qsvm'][te] = q.predict_proba(qkernel(zz_states(best[1] * Zte / sd), Str))
            sc = lambda Z: Z / sd
            P['rbf_svm_pca8'][te] = SVC(C=1.0, gamma='scale', probability=True, random_state=seed).fit(sc(Ztr), y[tr]).predict_proba(sc(Zte))
            P['lr_pca8'][te] = lr().fit(Ztr, y[tr]).predict_proba(Zte)
        combos = {k: [k] for k in P}
        combos.update({'FreshCheck': ['colour', 'mobilenetv3', 'resnet50'],
                       'FreshCheck+Gabor': ['colour', 'mobilenetv3', 'resnet50', 'gabor'],
                       'FreshCheck+QSVM': ['colour', 'mobilenetv3', 'resnet50', 'qsvm'],
                       'FreshCheck+Gabor+QSVM': ['colour', 'mobilenetv3', 'resnet50', 'gabor', 'qsvm']})
        for k, ks in combos.items():
            pr = np.mean([P[m] for m in ks], 0).argmax(1)
            add(k, float((pr == y).mean()), float(np.mean(pr[y == spoiled] == fresh)))
        print(name, seed, {k: round(np.mean(v['acc']) * 100, 1) for k, v in res.items()}, flush=True)
    o = {k: {m: [float(np.mean(v[m])), float(np.std(v[m]))] for m in v} for k, v in res.items()}; o['qsvm_bandwidths'] = BW; return o

out = {}
m = inv(A.meat, ('train', 'test'), r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-')
cmap = {'FRESH': 0, 'HALF-FRESH': 1, 'SPOILED': 2}
out['meat'] = run('meat', m, dict(np.load(os.path.join(A.rob, 'features_meat.npz'))), np.array([cmap[c] for c, _, _ in m]),
                  np.array([f'{c}-{s // 100}' for c, s, _ in m]), 2, 0)
p = inv(A.poultry, ('training', 'validation'), r'^(segar|busuk)_(\d+)\.jpe?g$')
out['poultry'] = run('poultry', p, dict(np.load(os.path.join(A.rob, 'features_poultry.npz'))), np.array([0 if c == 'segar' else 1 for c, _, _ in p]),
                     np.array([f'{c}-{s // 100}' for c, s, _ in p]), 1, 0)
json.dump(out, open(os.path.join(A.out, 'quantum_neuro_results.json'), 'w'), indent=1); print('saved')
