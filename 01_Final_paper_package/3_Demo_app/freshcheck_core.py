"""FreshCheck (v2, multi-level) core: feature extraction + fused logistic-regression models + refer rule.
Used by the training script (train_freshcheck.py) and the demo app (freshcheck_app.py)."""
import os, numpy as np, cv2, torch, timm
from PIL import Image
from skimage.feature import local_binary_pattern

def colour_features(img_bgr):
    im = cv2.resize(img_bgr, (224, 224)); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    h = [cv2.calcHist([hsv], [c], None, [b], [0, r]).ravel() for c, b, r in ((0, 18, 180), (1, 8, 256), (2, 8, 256))]
    lh = np.histogram(local_binary_pattern(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 8, 1, 'uniform'), bins=10, range=(0, 10))[0]
    return np.concatenate([x / x.sum() for x in h] + [lh / lh.sum(), lab.reshape(-1, 3).mean(0) / 255, lab.reshape(-1, 3).std(0) / 255])

_NETS = {}
def _net(name, env):
    if name not in _NETS:
        w = os.environ.get(env)
        m = timm.create_model(name, pretrained=w is None, num_classes=0)
        if w:
            sd = torch.load(w, map_location='cpu'); m.load_state_dict({k: v for k, v in sd.items() if not k.startswith(('classifier', 'fc.'))}, strict=False)
        m.eval(); cfg = timm.data.resolve_data_config({}, model=m)
        _NETS[name] = (m, timm.data.create_transform(**cfg))
    return _NETS[name]

def deep_features(pil_imgs, name, env):
    m, tf = _net(name, env)
    with torch.no_grad():
        return np.concatenate([m(torch.stack([tf(im.convert('RGB')) for im in pil_imgs[i:i + 32]])).numpy() for i in range(0, len(pil_imgs), 32)])

def colour_fine(img_bgr):
    im = cv2.resize(img_bgr, (224, 224)); lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    pct = np.percentile(lab, [5, 25, 50, 75, 95], axis=0).ravel() / 255
    h = np.histogram2d(lab[:, 1], lab[:, 2], bins=8, range=[[96, 200], [96, 200]])[0].ravel(); h = h / h.sum()
    return np.concatenate([pct, h, lab.mean(0) / 255, lab.std(0) / 255])

_ML = {}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32); _STD = np.array([0.229, 0.224, 0.225], np.float32)
def multilevel_features(pil_imgs):
    """ResNet-50 layer2/3/4 pooled features; g_* = full image + flip at 224, p_* = 2x2 grid of 224-px patches from a 448 resize."""
    if 'm' not in _ML:
        w = os.environ.get('RESNET_WEIGHTS'); m = timm.create_model('resnet50.a1_in1k' if w is None else 'resnet50', pretrained=w is None, features_only=True, out_indices=(2, 3, 4))
        if w:
            sd = torch.load(w, map_location='cpu'); m.load_state_dict({k: v for k, v in sd.items() if not k.startswith('fc.')}, strict=False)
        _ML['m'] = m.eval()
    prep = lambda rgb: torch.from_numpy(((rgb.astype(np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1))
    out = {k: [] for k in ('g_l2', 'g_l3', 'g_l4', 'p_l2', 'p_l3', 'p_l4')}
    with torch.no_grad():
        for im in pil_imgs:
            rgb = np.array(im.convert('RGB')); a = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA); b = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_AREA)
            x = torch.stack([prep(a), prep(a[:, ::-1].copy())] + [prep(b[y:y + 224, x:x + 224]) for y in (0, 224) for x in (0, 224)])
            for name, f in zip(('l2', 'l3', 'l4'), _ML['m'](x)):
                v = f.mean((2, 3)).numpy(); out['g_' + name].append(v[:2].mean(0)); out['p_' + name].append(v[2:].mean(0))
    return {k: np.array(v, np.float32) for k, v in out.items()}

def all_features(paths_or_pils):
    pils = [p if isinstance(p, Image.Image) else Image.open(p) for p in paths_or_pils]
    col = np.array([colour_features(cv2.cvtColor(np.array(im.convert('RGB')), cv2.COLOR_RGB2BGR)) for im in pils])
    bgr = [cv2.cvtColor(np.array(im.convert('RGB')), cv2.COLOR_RGB2BGR) for im in pils]
    F = {'colour': col, 'colour_fine': np.array([colour_fine(b) for b in bgr]),
         'mobilenetv3': deep_features(pils, 'tf_mobilenetv3_small_100', 'MNV3_WEIGHTS')}
    F.update(multilevel_features(pils)); return F

def predict(bundle, feats):
    """bundle = dict(models={key: sklearn pipeline}, classes=[...], tau=float). Returns probs, labels, decisions."""
    P = np.mean([bundle['models'][k].predict_proba(feats[k]) for k in bundle['models']], axis=0)
    labels = [bundle['classes'][i] for i in P.argmax(1)]
    decisions = [lab if p >= bundle['tau'] else 'Refer to staff' for lab, p in zip(labels, P.max(1))]
    return P, labels, decisions
