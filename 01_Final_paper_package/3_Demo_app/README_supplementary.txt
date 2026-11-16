Supplementary material for "A leakage-aware multi-level image classifier for fresh and spoiled food with a
refer-to-staff option: development and session-grouped validation on public datasets"

A_freshcheck/   The proposed classifier and its evaluation (Sections 2.3-2.6, 3.3-3.5; Tables 5-6; Figures 1, 4, 5)
  extract_multilevel.py       multi-level, multi-view frozen ResNet-50 features (stages 2-4; whole-image + local views)
                              and fine colour features -> features_<dataset>_v2.npz
  freshcheck_v2.py            FreshCheck (multi-level late fusion of nine logistic regressions + refer rule, tau chosen
                              on training folds) vs single feature sets, final-layer fusion ("FreshCheck_v1" in the
                              JSON) and ablations; session-grouped and image-level 5-fold CV (5 seeds); random-grouping
                              null; cross-dataset transfer and local labelling (20 draws). -> freshcheck_v2_results.json
  freshcheck.py, freshcheck_site.py   earlier evaluation of final-layer fusion and early fusion
                              (freshcheck_results.json, freshcheck_site_results.json); source of the early-fusion row.
  fig_pipeline_v3.py          draws Figure 1;  fig_v2.py draws Figures 4 and 5;  fig_freshcheck.py: earlier versions.
  quantum_neuro.py            additional feature extractors tested (Section 2.4): V1-style Gabor filter bank (72 values)
                              and a simulated 8-qubit ZZ-feature-map quantum-kernel SVM on PCA(8) of ResNet-50 features,
                              with classical controls on the same inputs. Session-grouped, 3 seeds, added to final-layer
                              fusion: red meat 89.6 -> 90.0 (Gabor), 89.1 (quantum); chicken 97.1 -> 96.8 (Gabor),
                              97.2 (quantum). Quantum kernel alone 79.6 / 92.9 vs RBF-SVM on the same inputs 80.8 / 93.3.
                              No reliable gain. -> quantum_neuro_results.json
  Feature caches: features_meat.npz / features_poultry.npz come from C/robustness_audit.py,
  features_fruit_3class.npz from C/dataset3_audit.py --three --T 30, and the *_v2.npz files from extract_multilevel.py.
  Run:
    python extract_multilevel.py --meat Phteven/data/raw --weights <resnet50 a1_in1k weights> --out feats/features_meat_v2.npz
    python freshcheck_v2.py --meat Phteven/data/raw --poultry "<repo>/dataset_citra_dada_ayam/dataset 200x200" \
        --rob <dir with meat/poultry features> --d3 <dir with fruit features> --out out/

B_demo_app/     Demonstration application (Sections 2.7, 3.6)
  freshcheck_app.py           Gradio web interface and command line:
                                python freshcheck_app.py                 (opens http://127.0.0.1:7860)
                                python freshcheck_app.py --food meat photo.jpg
  freshcheck_core.py          feature extraction and decision rule
  train_freshcheck.py         trains the released models on all images (tau = median of CV taus)
  models/                     freshcheck_meat.joblib (tau 0.50), freshcheck_poultry.joblib (tau 0.71)
  Requirements: python 3.10+, torch, timm, scikit-learn, scikit-image, opencv-python-headless, joblib, gradio.
  Optional local network weights: MNV3_WEIGHTS=<file>, RESNET_WEIGHTS=<file>; otherwise timm downloads them.
  Research prototype, not a food-safety device. The released models were trained on all images of the public
  datasets, so the app demonstrates the workflow; held-out performance is in Tables 5-6.

C_dataset_screening_and_validation/   Dataset screening, provenance check, validation-design analysis,
  fine-tuning and paneer/momo duplicate checks (Sections 2.2, 2.5, 3.1, 3.2). See README_screening.txt.

Data (all public): see the Data availability statement in the manuscript.
