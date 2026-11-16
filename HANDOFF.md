# FreshCheck project: handoff file (read this first in the new chat)

**Owner:** Yash Phoujdar (VIT, 24BCE2152) | **Supervisor / corresponding author:** Dr Vipin Kumar Singh (vipin.singh@vit.ac.in) | **Co-author:** Tanmay Pondhe
**Target journal:** Food Control (Elsevier, Q1). **Status date:** 16 Nov 2026

---

## 1. Where things stand

- Faculty said on 15 Nov 2026: *do not go for a review paper; implement something on the datasets.* So the paper is now an **implementation paper**.
- **Paper title:** "A leakage-aware image classifier for fresh and spoiled food with a refer-to-staff option: development and session-grouped validation on public datasets".
- **System (FreshCheck):**
  - Three feature sets: colour/texture (HSV 18/8/8 + uniform LBP 10 bins + Lab mean/std = 50 values), frozen MobileNetV3-Small (1024) and frozen ResNet-50 (2048).
  - Each set has its own StandardScaler + LogisticRegression(C=1). The three probability vectors are averaged (late fusion).
  - **Refer rule:** if the top probability is below τ, the output is "Refer to staff". τ is the smallest value in 0.50–0.99 such that ≤1% of spoiled images are accepted as fresh, chosen by inner grouped 3-fold CV on the training data.
- **Validation:** session-grouped 5-fold CV, 5 seeds. Groups are blocks of 100 consecutive file numbers per class (meat, chicken) and near-copy pHash families at 30 bits (fruit).
- **The manuscript is complete:**
  - abstract 249 words, ~4,600 words of text, 6 tables, 5 figures, 23 references;
  - highlights and cover letter done;
  - supplementary zip and Gradio demo app done.

### Key numbers (session-grouped)

| | Red meat (3 cl) | Chicken (2 cl) | Fruit (3 cl) |
|---|---|---|---|
| Colour/texture | 83.5 | 92.4 | – |
| MobileNetV3 | 86.9 | 95.0 | 96.8 |
| ResNet-50 | 87.4 | 96.1 | 96.7 |
| Early fusion | 89.8 | 97.1 | 97.1 |
| **FreshCheck** | **90.3** | **97.2** | **97.2** |
| + refer rule | 98.7% decided, 90.8% acc | 91.9% decided, **99.0%** acc, spoiled-as-fresh 3.2%→**1.0%** | 99.3% decided, 97.3% |

**Validation leakage**
- Red meat: image-level 98.4% vs capture-blocks of 100 86.9% (MobileNetV3), p = 0.048 (20 random groupings).
- Paneer: 491 of 500 "good" files are copies of 9 photos.
- Momo: 12–16 distinct checksums per ~250 listing entries.

**Transfer between datasets (binary)**
- With no local data: meat→chicken 49.8%, chicken→meat 54.8%.
- With local data, adding k sessions per class:
  - meat→chicken: 1 → 67.7, 2 → 80.0, 5 → 92.2, 10 → 95.7;
  - chicken→meat: 1 → 82.2, 2 → 90.9, 5 → 96.5, 10 → 98.2.
- Training on the local images alone is equal or better, so the other dataset adds nothing.

---

## 2. Datasets (all public; the shell in Claude's workspace can only reach GitHub + PyPI)

| Name in paper | Source (git clone) | Notes |
|---|---|---|
| Red meat (Shanawad 2022, Kaggle "Meat Freshness Image Dataset") | https://github.com/TheLohia/Phteven.git → `data/raw/{train,test}` | Files named FRESH-/HALF-FRESH-/SPOILED-<n>-; 2,266 images (853/789/624) |
| Chicken breast (Cendani et al.) | https://github.com/LinggarM/Poultry-Meat-Freshness-Classification-with-Transfer-Learning-using-ResNet → `dataset_citra_dada_ayam/dataset 200x200/{training,validation}` | segar_/busuk_ files; 650/650 |
| Fruit (zijianchen98) | https://github.com/zijianchen98/Fruit-freshness-detection-dataset | 3-class used: 1,894 images |
| Paneer curry (Basak & Sarkar 2025) | Mendeley Data, CC BY 4.0 (upload zip manually) | duplicate screening only |
| Momo x3 (Paul/Sarkar/Singh & Sarkar 2026) | Mendeley listings | duplicate screening only |

**New candidate datasets found on GitHub (not yet used):**
- **Fresh & Stale Images of Fruits and Vegetables** (Kaggle, R. Potdar). This is the "Food Aayush" project: phone photos captured daily, with frames extracted from video. The full mirror is at https://github.com/zohaib-khawaja/Capstone-3 under `data/` (~14.7k images, 12 classes: apple, banana, orange, **capsicum, tomato, bitter gourd**, each fresh/stale; ~1.5 GB). **Best next dataset**: it adds vegetables, and the video-frame capture means strong session leakage, so it is the ideal showcase for FreshCheck's validation.
- **Fruits fresh and rotten for classification** (Kaggle, S. R. Kalluri). Test split (2,698 images, 6 classes) mirrored at https://github.com/Renu-code123/Fruit-freshness-Detection-using-Machine-Learning under `data/fruit_dataset/test`. A widely used benchmark that is known to contain rotated/flipped copies.
- Fish eye datasets (FFE, MFED) are on Mendeley only; they need a manual upload.

> Verify the licence and citation of each dataset before adding it to the paper (the Kaggle pages can't be fetched from the workspace).

---

## 3. Plan agreed for the next chat: focus on 1–2 datasets and improve them with our own implementation

**Recommended focus:**
- **Dataset A: chicken breast**, which is already in the paper and clean (no session structure).
- **Dataset B: Fresh & Stale fruits + vegetables** (new; leakage-heavy, and it adds vegetables).

**Improvements to implement and test** (all under session-grouped validation; keep a result only if it helps):
1. **Background-suppressed input.** Crop or segment the food (GrabCut / saliency / simple colour threshold) before feature extraction, so the model sees food instead of the tray or table. Expected to reduce session shortcut learning.
2. **Fine-tuned FreshCheck.** Fine-tune MobileNetV3/ResNet-50 with group-aware early stopping, then fuse with the colour model.
3. **Probability calibration.** Temperature scaling or isotonic regression on the inner folds before applying τ, so the refer rule is better calibrated.
4. **Test-time augmentation.** Average over flips/crops.
5. **For dataset B:**
   - run the provenance check (file order vs pHash) and the session/near-copy grouping;
   - report image-level vs grouped accuracy;
   - test cross-dataset transfer to the zijianchen98 fruit set (shared apple/banana/orange classes), with local-labelling curves.

**Output:** update manuscript sections 2–4, Tables 5–6 and Fig. 4; add the new dataset to Table 1; rebuild the docx/pdf; repackage.

---

## 4. What is in this zip

```
01_Final_paper_package/     ready-to-submit files (manuscript docx/pdf, highlights, cover letter, Figures 1–5 tif/pdf, supplementary zip, demo app)
02_Source/
  fc/                       manuscript source: content.json (all text/tables), build.js (docx-js), extras.js (highlights + cover letter), figure PNGs
  scripts/impl_fc.py        script that generates fc/content.json (edit text here, then run it, then `node build.js`)
  supp/                     all analysis scripts (freshcheck.py, freshcheck_site.py, fig_*.py, robustness_audit.py, dataset3_audit.py, finetune_audit.py, paneer_duplicates.py, ...)
  supp/fc_sys/              results JSONs + generated figures
  features/                 cached features (features_meat.npz, features_poultry.npz, features_fruit_3class.npz + assignments): rerun FreshCheck without images
03_Demo_app/                freshcheck_app.py (Gradio + CLI), freshcheck_core.py, train_freshcheck.py, models/*.joblib
04_Faculty_docs/            Hinglish explainers and the faculty highlighted PDF
HANDOFF.md                  this file
```

### How to rebuild the manuscript

```
cd 02_Source && python3 scripts/impl_fc.py   # writes fc/content.json (paths inside assume /home/claude; adjust)
cd fc && npm i docx && node build.js && node extras.js
soffice --headless --convert-to pdf Manuscript_FoodControl.docx
```

### How to rerun FreshCheck

```
python freshcheck.py --meat <Phteven>/data/raw --poultry "<LinggarM>/dataset_citra_dada_ayam/dataset 200x200" --rob <dir with features_*.npz> --d3 <dir with features_fruit_3class.npz> --out out/
```

Weights download automatically through timm. The offline weights were `tf_mobilenetv3_small_100` and `resnet50.a1_in1k`; to use local copies, set `MNV3_WEIGHTS` / `RESNET_WEIGHTS`.

---

## 5. Still open (the user must do these)

- Confirm the author list with Sir (Yash is not yet listed; update CRediT if he is added).
- Check the Kaggle licences for the red-meat and chicken datasets.
- Get Sir's approval; run a plagiarism check.
- The AI-use declaration is in the manuscript and must stay (Elsevier policy).

## 6. Starter prompt for the new chat

> I'm continuing my FreshCheck food-freshness paper (Food Control). The attached zip has everything; read HANDOFF.md first. Goal now: take 1–2 datasets (chicken breast + the Fresh & Stale fruits/vegetables set from github.com/zohaib-khawaja/Capstone-3) and improve results with our own implementation (background-suppressed crops, fine-tuned FreshCheck, calibrated refer rule, TTA), all under session-grouped validation. Then update the manuscript and repackage. Reply in Hinglish, use tables and headings, verify everything.

---

## 7. Quantum + neuro experiment (DONE, 16 Nov 2026): `02_Source/supp/quantum_neuro.py`, results in `supp/fc_sys/quantum_neuro_results.json`

**What was tested** (session-grouped 5-fold, 3 seeds):
- **Neuro:** 72 V1-inspired Gabor features: complex-cell energy, 4 orientations x 3 scales, on the luminance and the red-green / blue-yellow opponent channels.
- **Quantum:** an 8-qubit ZZ-feature-map fidelity kernel SVM, simulated exactly as a statevector on a classical CPU. Input = PCA(8) of the ResNet-50 features. Bandwidth was chosen by inner CV; with no bandwidth the kernel concentrates and accuracy falls to 53–66%.
- **Fair classical twins:** an RBF-SVM and a logistic regression on the same 8 PCA values.

| Model | Red meat acc | Chicken acc | Chicken spoiled-as-fresh |
|---|---|---|---|
| Gabor (neuro) alone | 79.5 | 88.5 | 12.9 |
| Quantum kernel SVM (8 qubits, simulated) | 79.6 | 92.9 | 8.1 |
| RBF-SVM, same 8 PCA values | 80.8 | 93.3 | 7.2 |
| LogReg, same 8 PCA values | 80.5 | 94.2 | 6.6 |
| FreshCheck (3 models) | 89.6 | 97.1 | 3.2 |
| FreshCheck + Gabor | **90.0** | 96.8 | 3.5 |
| FreshCheck + Quantum | 89.1 | **97.2** | **2.9** |
| FreshCheck + Gabor + Quantum | 89.5 | 96.8 | 3.3 |

**Honest reading**
- The quantum kernel gives no advantage over classical models on the same 8 inputs (it is about 1 point lower).
- Inside the fusion the changes are ±0.5 points, which is within noise.
- Gabor helps red meat slightly (+0.4) and hurts chicken slightly.
- Recommendation: report this as a short supplementary or extra section ("we tested quantum-kernel and bio-inspired features; no reliable gain"), or as a separate short paper. Do not claim a quantum advantage.

---

## 8. UPDATE 16 Nov 2026 (evening): FreshCheck is now MULTI-LEVEL (v2). This supersedes the numbers in Sections 1 and 3

**Why:** the user wanted higher scores on a dataset that was not already ~98%. Red meat (90.3% session-grouped) was chosen.

**What changed (the paper, figures, app and supplementary are all updated to this):**
- FreshCheck now fuses **nine** logistic regressions: colour/texture (50), fine colour (85), MobileNetV3 (1024), and ResNet-50 pooled outputs of **stages 2, 3, 4** from **two views** (whole image + flip at 224; 2x2 grid of 224 patches from a 448 resize).
- Finding: mid-level features generalise to unseen sessions far better than the final layer (red meat: stage 2 alone 94.7% vs final layer 87.4%).
- New title: "A leakage-aware multi-level image classifier for fresh and spoiled food with a refer-to-staff option: development and session-grouped validation on public datasets".
- Scripts: `supp/extract_multilevel.py`, `supp/freshcheck_v2.py`, `supp/fig_v2.py`, `supp/fig_pipeline_v3.py`; results in `supp/fc_sys/freshcheck_v2_results.json`; features in `features/*_v2.npz`; manuscript post-processor `scripts/impl_v2.py` (run after `impl_fc.py`).

| Session-grouped accuracy | Red meat | Chicken | Fruit |
|---|---|---|---|
| ResNet-50 final layer | 87.4 | 96.1 | 96.7 |
| Final-layer fusion (old FreshCheck v1) | 90.3 | 97.2 | 97.2 |
| ResNet-50 stage 2 alone | 94.7 | 97.7 | 97.1 |
| **FreshCheck multi-level** | **95.2 ± 0.6** | **98.4 ± 0.1** | **97.4 ± 0.2** |
| Image-level (leaky) FreshCheck | 99.8 | 98.8 | 97.9 |

- **Refer rule, chicken:** 96.8% decided, 99.3% accurate, spoiled-as-fresh 1.8% → 0.8% (tau ≈ 0.71). Red meat: spoiled-as-fresh 0.0%.
- **Random-grouping null still above session-grouped:** red meat 99.6–99.8 vs 95.2; chicken 98.5–99.0 vs 98.4 (p = 0.048 each).
- **Transfer, no local data:** meat→chicken 49.0%, chicken→meat 66.5%. With 5 local sessions per class: 93.7% / 96.9%; local-only 94.3% / 99.6%.
- **Demo app:** ~0.35 s per photo on a 2-core CPU.
- **Manuscript:** abstract 250 words, ~5,050 words, 6 tables, 5 figures, 23 references.

**Ideas not yet done:** Grad-CAM heatmaps, background-suppressed crops, the Fresh & Stale fruit+veg dataset, a real kitchen field test.
