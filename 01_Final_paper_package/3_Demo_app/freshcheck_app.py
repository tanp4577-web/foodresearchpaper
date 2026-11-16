"""FreshCheck demo app (Gradio). Upload or take a photo of red meat or chicken breast; the app returns
Fresh / Half-fresh / Spoiled, the class probabilities, or 'Refer to staff' when the model is not confident.
Run:  pip install gradio timm torch scikit-learn scikit-image opencv-python-headless joblib
      python freshcheck_app.py            (opens http://127.0.0.1:7860)
Command-line use without Gradio:  python freshcheck_app.py --food meat photo1.jpg photo2.jpg
Research prototype: not a food-safety device. A 'Fresh' answer means 'no visible problem found', not 'safe to eat'."""
import os, sys, argparse, joblib
from PIL import Image
from freshcheck_core import all_features, predict
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLES = {'meat': 'freshcheck_meat.joblib', 'chicken': 'freshcheck_poultry.joblib'}
def load(food): return joblib.load(os.path.join(HERE, 'models', BUNDLES[food]))
def run(food, images):
    b = load(food); P, labels, dec = predict(b, all_features(images))
    return [(d, {c: float(p) for c, p in zip(b['classes'], row)}) for d, row in zip(dec, P)]
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--food', choices=list(BUNDLES)); ap.add_argument('images', nargs='*')
    a = ap.parse_args()
    if a.images:
        for path, (d, probs) in zip(a.images, run(a.food or 'meat', [Image.open(p) for p in a.images])):
            print(f'{os.path.basename(path)}: {d}  ' + ', '.join(f'{c} {v:.2f}' for c, v in probs.items()))
        sys.exit()
    import gradio as gr
    def ui(img, food):
        d, probs = run('meat' if food == 'Red meat' else 'chicken', [img])[0]
        note = {'Refer to staff': 'Model not confident: ask a trained person to check.'}.get(d, 'Visual check only, not a safety certificate.')
        return d, probs, note
    gr.Interface(fn=ui, inputs=[gr.Image(type='pil', label='Food photo'), gr.Radio(['Red meat', 'Chicken breast'], value='Red meat', label='Food')],
                 outputs=[gr.Textbox(label='Decision'), gr.Label(label='Probabilities'), gr.Textbox(label='Note')],
                 title='FreshCheck — image-based freshness check (research prototype)').launch()
