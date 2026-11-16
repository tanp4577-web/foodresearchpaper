const fs=require('fs'); const {Document,Packer,Paragraph,TextRun,AlignmentType,LevelFormat}=require('docx');
const F='Times New Roman', R=(t,o={})=>new TextRun({text:t,font:F,size:24,bold:o.b,italics:o.i});
const P=(c,o={})=>new Paragraph({children:Array.isArray(c)?c:[R(c)],spacing:{line:360,after:o.after??160},alignment:o.al||AlignmentType.JUSTIFIED});
const num={config:[{reference:'b',levels:[{level:0,format:LevelFormat.BULLET,text:'•',alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:540,hanging:300}}}}]}]};
const H=["FreshCheck labels food photos fresh or spoiled and refers uncertain ones to staff", "Mid-level network features beat final-layer features on unseen capture sessions", "Session-grouped accuracy rose from 90.3% to 95.2% on red meat (chicken: 98.4%)", "Refer rule cut spoiled chicken accepted as fresh to 0.8% while deciding 96.8%", "No transfer between meat datasets; about 80–170 local images restored 94–97%"];
const hl=new Document({numbering:num,sections:[{children:[P([R('Highlights',{b:true})]),...H.map(h=>new Paragraph({numbering:{reference:'b',level:0},children:[R(h)],spacing:{line:360,after:120}}))]}]});
Packer.toBuffer(hl).then(b=>fs.writeFileSync('Highlights.docx',b));
const L=[
P('16 November 2026',{al:AlignmentType.LEFT}),
P([R('The Editor-in-Chief',{})],{after:0,al:AlignmentType.LEFT}),P('Food Control',{al:AlignmentType.LEFT}),
P([R('Subject: ',{b:true}),R('Submission of a research article, "A leakage-aware multi-level image classifier for fresh and spoiled food with a refer-to-staff option: development and session-grouped validation on public datasets"')]),
P('Dear Editor,'),
P('We submit the above manuscript for consideration as an original research paper in Food Control.'),
P('The manuscript develops and validates FreshCheck, an image-based classifier that labels food as fresh or spoiled and refers uncertain photographs to staff instead of guessing. FreshCheck fuses colour and texture features with frozen network features taken at several depths and image scales, and its refer threshold is set on training data to a safety target: no more than 1% of spoiled samples accepted as fresh. Because public food image datasets often contain many photographs from a few capture sessions, we first screened every dataset for duplicates and session structure and validated FreshCheck with capture sessions kept apart. Image-level splits overstated red-meat accuracy by up to 11.5 points, and two public prepared-dish collections consisted largely of copies. Under session-grouped validation, mid-level features proved far more reliable than the final-layer features used in earlier studies: FreshCheck reached 95.2% on red meat, 98.4% on chicken breast and 97.4% on fruit, against 90.3%, 97.2% and 97.2% for fusion of final-layer features. On chicken its refer rule lowered spoiled samples accepted as fresh to 0.8% while deciding 96.8% of images. Models did not transfer between two meat datasets, but about five labelled local capture sessions per class restored 94–97% accuracy.'),
P('These results bear directly on food control: they give a classifier that can be set to a safety target, an honest estimate of what it achieves on unseen material, and a practical route to installing it at a new site. All analyses use public data; the code, trained models and a demonstration application are supplied, and the scripts reproduce every number in the paper.'),
P('We confirm that this manuscript has not been published previously and is not under consideration elsewhere. Both authors approved the submission, declare no competing interests, and disclose the use of generative AI tools in the manuscript as required by Elsevier policy.'),
P('Thank you for considering our submission.'),
P('Yours sincerely,',{after:60,al:AlignmentType.LEFT}),
P([R('Dr. Vipin Kumar Singh',{b:true})],{after:0,al:AlignmentType.LEFT}),
P('School of Hotel and Tourism Management, Vellore Institute of Technology, Vellore 632014, Tamil Nadu, India',{after:0,al:AlignmentType.LEFT}),
P('vipin.singh@vit.ac.in (corresponding author)',{al:AlignmentType.LEFT}),
P('On behalf of all authors (Vipin Kumar Singh and Tanmay Pondhe)',{al:AlignmentType.LEFT})];
const cl=new Document({styles:{default:{document:{run:{font:F,size:24}}}},sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:1440,bottom:1440,left:1440,right:1440}}},children:L}]});
Packer.toBuffer(cl).then(b=>{fs.writeFileSync('Cover_Letter.docx',b);console.log('ok')});
