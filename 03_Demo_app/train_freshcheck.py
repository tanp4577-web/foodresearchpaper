"""Train the final FreshCheck models on all images of one dataset and save them (joblib).
tau = median of the thresholds chosen on training folds in the cross-validated evaluation (freshcheck_v2_results.json).
Usage: python train_freshcheck.py --rob <features dir> --meat <...> --poultry <...> --results freshcheck_v2_results.json --out models/"""
import argparse, os, re, json, joblib, numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ap = argparse.ArgumentParser()
for a in ('--rob', '--meat', '--poultry', '--results', '--out'): ap.add_argument(a)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
def inv(root, splits, pat):
    return [m.group(1) for sp in splits for f in sorted(os.listdir(os.path.join(root, sp))) for m in [re.match(pat, f)] if m]
R = json.load(open(A.results))
for ds, root, splits, pat, cmap, classes in (
    ('meat', A.meat, ('train', 'test'), r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-', {'FRESH': 0, 'HALF-FRESH': 1, 'SPOILED': 2}, ['Fresh', 'Half-fresh', 'Spoiled']),
    ('poultry', A.poultry, ('training', 'validation'), r'^(segar|busuk)_(\d+)\.jpe?g$', {'segar': 0, 'busuk': 1}, ['Fresh', 'Spoiled'])):
    F = dict(np.load(os.path.join(A.rob, f'features_{ds}.npz'))); F.update(np.load(os.path.join(A.rob, f'features_{ds}_v2.npz'))); y = np.array([cmap[c] for c in inv(root, splits, pat)])
    models = {k: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(F[k], y) for k in ('colour', 'colour_fine', 'mobilenetv3', 'g_l2', 'g_l3', 'g_l4', 'p_l2', 'p_l3', 'p_l4')}
    tau = round(float(R[ds]['session_grouped']['taus_v2']['median'][0]), 2)
    joblib.dump(dict(models=models, classes=classes, tau=tau, food=ds), os.path.join(A.out, f'freshcheck_{ds}.joblib'))
    print(ds, 'saved; classes', classes, 'tau', tau, 'n', len(y))
