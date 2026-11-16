import json
B=json.load(open('/tmp/fc_content_v16.json'))
T={i:b.get('text','') for i,b in enumerate(B)}
tbl={b['n']:b for b in B if b.get('k')=='table'}
figs=[b for b in B if b.get('k')=='fig']
def P(t): return {'k':'p','text':t}
def H1(t): return {'k':'h1','text':t}
def H2(t): return {'k':'h2','text':t}
def swap(t):  # old Table 3 (poultry/transfer) -> 4, old Table 4 (null) -> 3
    return t.replace('Table 3','@@T4').replace('Table 4','Table 3').replace('@@T4','Table 4')
def sub(t,a,b):
    assert a in t,(a[:60]); return t.replace(a,b)
front=B[0:5]
abstract=dict(B[5]); abstract['text']=("This study examines how reliably image-based classifiers separate fresh from spoiled food, and how their accuracy should be validated before such systems are used in food control. "
"Four public image datasets were analysed: red meat (2,266 images, three classes), chicken breast (1,300 images), fruit (1,894 images) and a prepared vegetarian dish, paneer curry. "
"Duplicate and near-duplicate images were identified by checksums and perceptual hashing, and classifiers based on colour and texture features, frozen MobileNetV3 and ResNet-50 features and a fine-tuned network were compared under image-level, near-duplicate-grouped and capture-session-grouped five-fold cross-validation, against size-matched random grouping. "
"In the red-meat set, consecutive files were frames of the same capture session, and accuracy fell from 98.4% under image-level cross-validation to 86.9% when sessions were kept together, whereas hash deduplication and random grouping left it unchanged (p = 0.048); ResNet-50 features and fine-tuning showed the same drop. "
"In the fruit set, grouping repeated photographs raised the three-class error rate by half (p = 0.048), and 491 of the 500 'good' paneer-curry files were copies of nine photographs of one plate. "
"A provenance check applied before training flagged the affected datasets (p = 0.001) and cleared the chicken-breast set. "
"Models scoring 96.0–99.9% within one meat dataset reached 42–64% on the other and passed up to 99.8% of spoiled samples as fresh. "
"Near-perfect published accuracies therefore largely reflect related images in training and test sets, and image-based freshness classifiers should be validated with item- or session-grouped splits and external test sets.")
kw=dict(B[6]); kw['text']="Food freshness; Spoilage detection; Image classification; Deep learning; Data leakage; Food safety"
intro1=("This study is concerned with image-based classification of food as fresh or spoiled, and in particular with how the accuracy of such classifiers is measured. "
"Visual assessment of freshness is a routine control point at receiving, in storage and at service for meat, fresh produce and prepared dishes, and camera-based classifiers trained on public image datasets promise to make this check faster and more consistent. "
"Reported performance is high: studies built on public meat image datasets report 98.5–99.8% accuracy for fresh, half-fresh and spoiled categories (Abd Elfattah et al., 2025; Büyükarıkan, 2024; Külcü & Balpetek Külcü, 2026), and frozen-network features have been compared across several public meat collections (Hidalgo et al., 2025). "
"Figures of this kind shape what food businesses and regulators expect a camera-based check to achieve.")
intro3=sub(T[10],"Public meat image datasets are particularly exposed, because they tend to contain many photographs of a few pieces of meat taken in a handful of sessions,","Public food image datasets are particularly exposed, because they often contain many photographs of a few items taken in a handful of sessions,")
intro3=sub(intro3,"As far as we know, no one has tested whether hash-based deduplication is enough for meat images,","As far as we know, no study has tested whether hash-based deduplication is enough for food freshness images,")
intro4=("The aim of this study was therefore to evaluate image-based fresh/spoiled classification on four public food image datasets, covering red meat, chicken breast, fruit and paneer curry, under validation designs that either allow or prevent related images from appearing on both sides of the split. "
"The specific objectives were to (i) characterise the duplicate and near-duplicate structure of each dataset and test whether file order encodes capture sessions; (ii) compare classification accuracy under image-level, near-duplicate-grouped and capture-session-grouped cross-validation against size-matched random grouping; (iii) test whether accuracy obtained within one dataset transfers to an independently collected one; and (iv) propose and evaluate a provenance check that can be applied before any model is trained. "
"Hand-crafted colour and texture features, frozen convolutional features from two networks and a fine-tuned network were compared, and the rate at which spoiled samples are passed as fresh was reported as the safety-relevant error. The analysis code is released for reuse.")
ds=sub(T[14],"We used three public datasets (Table 1).","Four public datasets covering red meat, chicken breast, fruit and a prepared vegetarian dish were used (Table 1).")
fruit_m=sub(T[15],"As a non-meat check we added a public fruit set","The fruit set is a public collection")
paneer_m=sub(T[16],"Finally, we checked a prepared Indian vegetarian dish for exact copies. The paneer-curry dataset","The fourth dataset represents a prepared Indian vegetarian dish. The paneer-curry dataset")
dup_m=T[19]+" For the paneer-curry set, exact copies were identified by comparing file checksums (MD5 and SHA-256)."
fruit_dup=("The fruit set had no usable capture-order structure because its file numbers are interleaved across classes, but the images themselves repeat: 68.2% had a same-class near-copy within 30 bits elsewhere in the set, and the largest family held 36 images. "
"Fig. 2c shows one such family, the same two rotten bananas photographed again and again, about every eleventh file. At the looser 60-bit threshold, families also joined different fruit photographed on similar plain backgrounds, which is why the stricter threshold was used for this set.")
paneer_r=sub(T[48],"The paneer-curry set showed the simplest form of the problem.","The paneer-curry set showed the simplest form of duplication.")
paneer_r=sub(paneer_r," A checksum comparison, the cheapest check there is, finds this in seconds, and the provenance check flags it at once."," A checksum comparison, the cheapest check available, finds this in seconds.")
fruit_cls=("In the fruit set, frozen MobileNetV3 features scored 97.8 ± 0.1% on the three-class task with image-level folds and 96.6 ± 0.4% with near-copy families kept together, so errors rose by about half (from 2.2% to 3.4%), while all 20 random regroupings with the same family sizes scored 97.5–98.1% (p = 0.048). "
"ResNet-50 features gave 97.4% and 96.7% (p = 0.048), and looser 60-bit families gave a larger drop (to 95.6%). "
"The binary fresh-versus-rotten task did not change (99.5% under both protocols): rotten fruit is easy to recognise, and leakage affected only the finer distinction involving the intermediate class.")
concl=("This study evaluated how image-based classifiers that separate fresh from spoiled food should be validated, using four public datasets of red meat, chicken breast, fruit and paneer curry. "
+sub(T[61],"On a public red-meat benchmark","On the red-meat benchmark"))
t2=dict(tbl[2]); t3=dict(tbl[4]); t3['n']=3; t4=dict(tbl[3]); t4['n']=4; t5=tbl[5]; t1=tbl[1]
body=[H1('1. Introduction'),P(intro1),P(T[9]),P(intro3),P(intro4),
H1('2. Materials and methods'),H2('2.1. Image datasets'),P(ds),P(fruit_m),P(paneer_m),t1,
H2('2.2. Duplicate and near-duplicate analysis'),P(dup_m),
H2('2.3. Provenance leakage check'),P(T[20]),
H2('2.4. Feature extraction and classification'),P(T[22]),
H2('2.5. Validation protocols'),P(T[24]),
H2('2.6. Cross-dataset transfer'),P(T[26]),
H2('2.7. Performance metrics and statistical analysis'),P(T[28]),
H1('3. Results'),H2('3.1. Duplicate and near-duplicate structure of the datasets'),P(T[31]),P(fruit_dup),P(paneer_r),figs[0],figs[1],
H2('3.2. Effect of validation design on red-meat classification'),P(swap(T[33])),P(swap(T[35])),t2,t3,
H2('3.3. Effect of validation design on the chicken-breast and fruit datasets'),P(swap(T[38])),P(fruit_cls),
H2('3.4. Cross-dataset transfer'),P(swap(T[40])),t4,
H2('3.5. Provenance leakage check'),P(T[45]),t5,
H1('4. Discussion'),H2('4.1. Why near-duplicate removal does not prevent leakage'),P(T[53]),
H2('4.2. Interpretation of published accuracies for food control'),P(T[55]),
H2('4.3. Recommendations for image-based freshness studies'),P(T[57]),
H2('4.4. Limitations'),P(T[59]),
H1('5. Conclusions'),P(concl)]
decls=[b for b in B if b.get('k')=='decl']; refs=[b for b in B if b.get('k')=='ref']
rest=[b for b in B[62:] if b.get('k') not in ('decl','ref')]
assert not rest, rest[:2]
NEW=front+[abstract,kw]+body+decls+refs
json.dump(NEW,open('/home/claude/fc/content.json','w'),ensure_ascii=False)
print('abstract words',len(abstract['text'].split()))
import re
s=' '.join(b.get('text','') for b in body)
first=[int(m) for m in re.findall(r'Table (\d)',s)]; seen=[]
[seen.append(x) for x in first if x not in seen]; print('table order',seen)
f=[m for m in re.findall(r'Fig\. (\d)',s)]; sf=[]; [sf.append(x) for x in f if x not in sf]; print('fig order',sf)
print('words body',len(s.split()))
