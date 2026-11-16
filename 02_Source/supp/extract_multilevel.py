"""Multi-level, multi-view frozen features (FreshCheck v2).
For each image: ResNet-50 (a1_in1k) global-average-pooled outputs of layer2 (512), layer3 (1024) and layer4 (2048),
averaged over (a) the full image at 224, its horizontal flip, and (b) a 2x2 grid of 224-px patches from the image resized to 448
(local views emphasise meat surface texture over scene layout). Also fine colour statistics (Lab percentiles + a*b* 2-D histogram).
Usage: python extract_multilevel.py (--meat <Phteven/data/raw> | --poultry <dataset 200x200> | --fruit <repo> --assign <assignments csv>) --weights <resnet50_a1_0.pth> --out <features_meat_v2.npz>"""
import argparse, os, re, numpy as np, torch, timm, cv2
ap = argparse.ArgumentParser(); [ap.add_argument(a) for a in ('--meat', '--poultry', '--fruit', '--assign', '--weights', '--out')]; A = ap.parse_args()
torch.set_num_threads(2)
m = timm.create_model('resnet50', pretrained=False, features_only=True, out_indices=(2, 3, 4)); 
sd = torch.load(A.weights, map_location='cpu'); m.load_state_dict({k: v for k, v in sd.items() if not k.startswith('fc.')}, strict=False); m.eval()
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
def prep(rgb): return torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))
def colour_fine(bgr):
    im = cv2.resize(bgr, (224, 224)); lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    pct = np.percentile(lab, [5, 25, 50, 75, 95], axis=0).ravel() / 255
    h = np.histogram2d(lab[:, 1], lab[:, 2], bins=8, range=[[96, 200], [96, 200]])[0].ravel(); h = h / h.sum()
    return np.concatenate([pct, h, lab.mean(0) / 255, lab.std(0) / 255])
rows = []
if A.meat:
    for sp in ('train', 'test'):
        for f in sorted(os.listdir(os.path.join(A.meat, sp))):
            if re.match(r'^(FRESH|HALF-FRESH|SPOILED)-(\d+)-', f): rows.append(os.path.join(A.meat, sp, f))
elif A.poultry:
    for sp in ('training', 'validation'):
        for f in sorted(os.listdir(os.path.join(A.poultry, sp))):
            if re.match(r'^(segar|busuk)_(\d+)\.jpe?g$', f): rows.append(os.path.join(A.poultry, sp, f))
else:   # fruit: same row order as assignments_fruit_3class_T30.csv (column n = image number)
    import csv
    rows = [os.path.join(A.fruit, 'JPEGImages', r['n'] + '.jpg') for r in csv.DictReader(open(A.assign))]
G = {k: [] for k in ('g_l2', 'g_l3', 'g_l4', 'p_l2', 'p_l3', 'p_l4', 'colour_fine')}
with torch.no_grad():
    for i, p in enumerate(rows):
        bgr = cv2.imread(p); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        a = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA); b = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_AREA)
        glob = torch.stack([prep(a), prep(a[:, ::-1].copy())])
        pat = torch.stack([prep(b[y:y + 224, x:x + 224]) for y in (0, 224) for x in (0, 224)])
        fo = m(torch.cat([glob, pat]))
        for name, f in zip(('l2', 'l3', 'l4'), fo):
            v = f.mean((2, 3)).numpy(); G['g_' + name].append(v[:2].mean(0)); G['p_' + name].append(v[2:].mean(0))
        G['colour_fine'].append(colour_fine(bgr))
        if i % 100 == 0: print(i, len(rows), flush=True)
np.savez_compressed(A.out, **{k: np.array(v, np.float32) for k, v in G.items()}); print('saved')
