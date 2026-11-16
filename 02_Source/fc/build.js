const fs=require('fs'); const d=require('docx');
const {Document,Packer,Paragraph,TextRun,ImageRun,Table,TableRow,TableCell,WidthType,AlignmentType,BorderStyle,Footer,PageNumber,LineNumberRestartFormat,HeadingLevel}=d;
const C=JSON.parse(fs.readFileSync('content.json','utf8'));
let declDone=false;
const F='Times New Roman', SZ=24, LINE=480; // 12 pt, double spacing (journal review copy)
const W=11906-2880; // A4 text width (1" margins)
const R=(t,o={})=>new TextRun({text:t,font:F,size:o.size||SZ,bold:o.b,italics:o.i,superScript:o.sup,subScript:o.sub});
const runsOf=(rs,size)=>rs.map(r=>R(r.t,{size,b:r.b,i:r.i,sup:r.sup,sub:r.sub}));
const sp={line:LINE,after:120};
const para=(t)=>new Paragraph({children:[R(t)],alignment:AlignmentType.JUSTIFIED,spacing:sp});
const NB={style:BorderStyle.NONE,size:0,color:'FFFFFF'}, RULE={style:BorderStyle.SINGLE,size:8,color:'000000'};
const WEIGHTS={6:[0.26,0.1,0.17,0.16,0.15,0.16],1:[0.2,0.12,0.2,0.1,0.09,0.12,0.17],3:[0.3,0.14,0.14,0.14,0.14,0.14],4:[0.3,0.12,0.14,0.12,0.14,0.18],2:[0.18,0.115,0.215,0.08,0.15,0.26],5:[0.24,0.13,0.13,0.13,0.13,0.12,0.12]};
const DIMS={'fig1_pipeline_v3.png':[3930,1590,620],'fig4_results_v2.png':[3600,1260,600],'fig5_examples_v2.png':[3900,1620,600],'fig1_pipeline_v2.png':[3930,1590,620],'fig3_pipeline.png':[2617,822,600],'fig4_examples.png':[3900,1620,600],'fig5_results.png':[3600,1260,600],'fig_examples.png':[3600,3300,600],'fig_leakage_v8.png':[2160,1680,560],'fig_leakage.png':[2160,780,600],'ff230910e6b0225f118eedebb5047e008dd3e8d0.png':[2100,1320,576],'8c0318c627c1c966f45e7add6d95c38631b9ff8d.png':[2100,1260,576],'7f7f9fbccddadbc7f46ffa837fdc32b7cbcb9132.png':[1675,1711,420],'57effdbdf69da93b89e10e2a470a747d5a535399.png':[1675,1572,420]};
const kids=[];
for(const o of C){
  if(o.k=='title'){
    kids.push(new Paragraph({children:[R(o.text,{b:true,size:28})],alignment:AlignmentType.CENTER,spacing:{line:LINE,after:240}}));
  } else if(o.k=='front'){
    const first=o.runs[0].t;
    const isAuth=first.startsWith('Vipin');
    kids.push(new Paragraph({children:runsOf(o.runs),alignment:isAuth?AlignmentType.CENTER:AlignmentType.LEFT,spacing:{line:LINE,after:isAuth?200:60}}));
  } else if(o.k=='labelled_block'){
    if(o.abstract){ kids.push(new Paragraph({children:[],spacing:{after:120}})); kids.push(new Paragraph({children:[R('Abstract',{b:true})],spacing:sp})); kids.push(para(o.text)); }
    else kids.push(new Paragraph({children:[R('Keywords: ',{b:true}),R(o.text)],spacing:{line:LINE,after:240}}));
  } else if(o.k=='h1') kids.push(new Paragraph({children:[R(o.text,{b:true})],spacing:{line:LINE,before:240,after:120},keepNext:true}));
  else if(o.k=='h2') kids.push(new Paragraph({children:[R(o.text,{b:true,i:true})],spacing:{line:LINE,before:160,after:80},keepNext:true}));
  else if(o.k=='p') kids.push(para(o.text));
  else if(o.k=='fig'){
    const [w,h,px]=DIMS[o.file];
    kids.push(new Paragraph({children:[new ImageRun({type:'png',data:fs.readFileSync(o.file),transformation:{width:px,height:Math.round(px*h/w)}})],alignment:AlignmentType.CENTER,spacing:{before:120,after:60},keepNext:true}));
    const m=o.caption.match(/^(Figure \d+\.)\s(.*)$/);
    kids.push(new Paragraph({children:[R(m[1]+' ',{b:true}),R(m[2])],alignment:AlignmentType.JUSTIFIED,spacing:{line:LINE,after:200}}));
  } else if(o.k=='table'){
    kids.push(new Paragraph({children:[R(`Table ${o.n}. `,{b:true}),R(o.title)],spacing:{line:LINE,before:200,after:80},keepNext:true}));
    const wt=WEIGHTS[o.n]; let cw=wt.map(x=>Math.floor(x*W)); cw[cw.length-1]+=W-cw.reduce((a,b)=>a+b,0);
    const last=o.rows.length-1;
    const rows=o.rows.map((r,ri)=>new TableRow({tableHeader:ri==0,cantSplit:true,children:r.map((c,ci)=>new TableCell({width:{size:cw[ci],type:WidthType.DXA},margins:{top:40,bottom:40,left:80,right:80},
      borders:{left:NB,right:NB,top:ri==0?RULE:NB,bottom:(ri==0||ri==last)?RULE:NB},
      children:[new Paragraph({children:[R(c,{size:22,b:ri==0})],spacing:{line:276,after:0}})]}))}));
    kids.push(new Table({width:{size:W,type:WidthType.DXA},columnWidths:cw,rows}));
    kids.push(new Paragraph({children:[],spacing:{after:200}}));
  } else if(o.k=='decl'){
    kids.push(new Paragraph({children:[R(o.label,{b:true})],spacing:{line:LINE,before:240,after:80},keepNext:true}));
    kids.push(new Paragraph({children:[R(o.text)],alignment:AlignmentType.JUSTIFIED,spacing:sp}));
  }
}
kids.push(new Paragraph({children:[R('References',{b:true})],spacing:{line:LINE,before:240,after:120}}));
for(const o of C.filter(x=>x.k=='ref')){
  kids.push(new Paragraph({children:runsOf(o.runs),indent:{left:540,hanging:540},alignment:AlignmentType.LEFT,spacing:{line:LINE,after:60}}));
}
const doc=new Document({styles:{default:{document:{run:{font:F,size:SZ}}}},
  sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:1440,bottom:1440,left:1440,right:1440}},lineNumbers:{countBy:1,restart:LineNumberRestartFormat.CONTINUOUS}},
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,suppressLineNumbers:true,children:[new TextRun({children:[PageNumber.CURRENT],font:F,size:SZ})]})]})},
    children:kids}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync('Manuscript_FoodControl.docx',b);console.log('ok')});
