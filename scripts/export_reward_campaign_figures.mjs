// Native editable evidence figures. Run with the bundled Node runtime after
// the three-arm seed42 pilot is complete; no model calls or illustrative art.
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const root = path.resolve(process.argv[2]);
const campaign = path.resolve(process.argv[3]);
const runtime = path.resolve(process.argv[4]);
process.env.RUNTIME_NODE_MODULES=runtime;
const skill = path.resolve(process.argv[5]);
const runtimePython = path.resolve(process.argv[6]);
const diagnosticsOnly=process.argv.includes("--diagnostics-only");
const build = path.join(root, "tmp", "reward_debug_20261104T054605Z", "pptx_build");
const output = path.join(campaign, "figures");
const evidence = path.join(campaign, "evidence");
const finalPath = diagnosticsOnly ? path.join(build,"diagnostic_qa.pptx") : path.join(output, "reward_debug_editable_figures.pptx");
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(
  path.join(runtime, "@oai", "artifact-tool", "dist", "artifact_tool.mjs")).href);
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } =
  await import(pathToFileURL(path.join(skill, "container_tools", "artifact_tool_utils.mjs")).href);
const font = resolvePresentationFont({ fontFamily: "Arial" });
const state = diagnosticsOnly ? {} : JSON.parse(await fs.readFile(path.join(campaign, "internal_pilot", "run_state.json"), "utf8"));
if (!diagnosticsOnly && state.status !== "COMPLETED") throw new Error("Final native training figure requires completed pilot");
const cal = JSON.parse(await fs.readFile(path.join(campaign, "lighting_v3_calibration.json"), "utf8"));
const diagnostic = JSON.parse(await fs.readFile(path.join(campaign, "diagnostics", "debug_results.json"), "utf8"));
await Promise.all([build, output, evidence].map(p => fs.mkdir(p, { recursive: true })));
const presentation = Presentation.create({ slideSize: { width: 1344, height: 1056 } });
const text = (slide, value, x, y, w, h, size=24, bold=false, alignment="left") => {
  const shape = slide.shapes.add({ geometry: "textbox", position: {left:x, top:y, width:w, height:h},
    fill: "none", line: {fill:"none", width:0, style:"solid"} });
  shape.text = value;
  shape.text.style = {typeface:font, fontSize:size, bold, color:"#172633", insets:0,
    alignment, verticalAlignment:"middle", wrap:"square", autoFit:"none"};
  return shape;
};
const axisStyle = {fontSize:20, fill:"#263744"};
// Editable Excel charts use a nine-decimal display snapshot. Full-precision
// empirical records remain in the accompanying JSON/CSV evidence.
const plotted=value=>Number(value.toFixed(9));
const embeddedPng=async file=>{
  const bytes=await fs.readFile(file);
  return bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);
};

// Slide1: literal native chart of the task controls, with an editable workbook.
{
  const slide = presentation.slides.add(); slide.background.fill="#FFFFFF";
  text(slide, "Lighting reward controls", 70, 35, 1204, 64, 38, true);
  text(slide, "Four prespecified TRAIN sources. Bars show means on the lighting-v3 scale.", 70, 103, 1204, 44, 23);
  const variants = ["identity", "exposure_plus3pct", "exposure_minus3pct", "channel_swap", "unrelated_image"];
  const values = variants.map(v => {
    const group=cal.controls.filter(r=>r.variant===v);return plotted(group.reduce((s,r)=>s+r.common_reward,0)/group.length);
  });
  const colors=["#777777","#397AA6","#B56950","#B56950","#B56950"];
  const chart=slide.charts.add("bar", {position:{left:85, top:175, width:1170, height:720},
    title:"Mean lighting-v3 reward", titleTextStyle:{fontSize:27, fill:"#172633"},
    categories:["Identity","+3% exposure","-3% exposure","Channel swap","Unrelated image"],
    series:[{name:"Mean of four source controls",values,fill:"#397AA6",
      points:colors.map((fill,idx)=>({idx,fill}))}],
    barOptions:{direction:"column",grouping:"clustered",gapWidth:90},hasLegend:false,
    xAxis:{position:"bottom",tickLabelPosition:"low",textStyle:axisStyle,line:{fill:"#55616B",width:1}},
    yAxis:{title:{text:"Task reward",textStyle:axisStyle},textStyle:axisStyle,
      majorGridlines:{fill:"#D9DFE4",width:1,style:"solid"},numberFormatCode:"0.0"},
    dataLabels:{showValue:true,position:"outEnd",numberFormatCode:"0.000",textStyle:{fontSize:20,fill:"#172633"}},
    chartFill:"#FFFFFF",plotAreaFill:"#FFFFFF",chartLine:{fill:"none",width:0},plotAreaLine:{fill:"none",width:0}});
  applyPresentationChartFont(chart,{fontFamily:font});
  text(slide,"Identity = 0. Positive exposure controls score above zero. Analytical controls do not establish trained-generator success.",70,920,1204,64,22);
  slide.speakerNotes.text = "Measured evidence: lighting_v3_calibration.json, controls. CLIP/LPIPS values reused from diagnostics/debug_results.json on the same saved control images. V3 mean-property arithmetic rescored without new model calls. Source files and SHA256 are in the campaign manifests. These are four source controls, not four independent training seeds. Old and revised reward scales differ.";
}

// Slide2: full fixed-seed validation trajectories, no invented seed intervals.
if(!diagnosticsOnly){
  const slide=presentation.slides.add();slide.background.fill="#FFFFFF";
  text(slide,"Lighting-v3 validation",70,35,1204,64,38,true);
  text(slide,"Seed42, 16 development-validation sources, 100 generator updates per trained arm.",70,103,1204,44,23);
  const definitions=[["frozen","Frozen = Fixed reference","#777777"],["ddpo_only","DDPO-only","#397AA6"],
    ["rule_rsi","Rule RSI","#3D9B77"],["rl_rsi","RL-RSI","#BC663F"]];
  const series=definitions.map(([key,name,color])=>{
    const rows=state.validation[key];if(rows.length!==4)throw new Error("Missing validation round for "+key);
    return {name,values:rows.map(r=>plotted(r.common_reward)),line:{fill:color,width:3,style:"solid"},
      marker:{symbol:key==="frozen"?"none":"circle",...(key==="frozen"?{}:{size:6}),fill:color}};
  });
  series.push({name:"Pixel identity",values:[0,0,0,0],line:{fill:"#111111",width:1.5,style:"solid"},marker:{symbol:"none"}});
  const chart=slide.charts.add("line",{position:{left:85,top:175,width:1170,height:720},
    title:"Common validation reward",titleTextStyle:{fontSize:27,fill:"#172633"},categories:["0","34","67","100"],series,
    hasLegend:true,legend:{position:"bottom",overlay:false,textStyle:{fontSize:20,fill:"#172633"}},
    xAxis:{title:{text:"Generator updates in trained arms",textStyle:axisStyle},position:"bottom",tickLabelPosition:"low",textStyle:axisStyle},
    yAxis:{title:{text:"Common lighting-v3 reward",textStyle:axisStyle},textStyle:axisStyle,
      majorGridlines:{fill:"#D9DFE4",width:1,style:"solid"},numberFormatCode:"0.00"},
    lineOptions:{smooth:false},chartFill:"#FFFFFF",plotAreaFill:"#FFFFFF",
    chartLine:{fill:"none",width:0},plotAreaLine:{fill:"none",width:0}});
  applyPresentationChartFont(chart,{fontFamily:font});
  text(slide,"Frozen and Fixed share one checkpoint and observations by user choice. Their flat reference repeats at training milestones. One seed provides no estimate of training-seed uncertainty.",70,920,1204,84,22);
  slide.speakerNotes.text="Measured source: internal_pilot/run_state.json validation, raw observations.jsonl and training.jsonl. Native chart connects four reported checkpoints for readability, with no imputed intermediate measurements, uncertainty band or selection by assessment. The zero line is the mathematically defined pixel-identity score. Frozen/Fixed receive no optimizer updates. Common evaluator fixes semantic/task weight2 and LPIPS weight1 for every arm. This is benign editing, not ASR or poisoned-victim DeltaClean.";
}

// Slide3: empirical photos remain individual picture objects, never a screenshot.
{
  const slide=presentation.slides.add();slide.background.fill="#FFFFFF";
  text(slide,"Frozen editor parameter audit",60,22,1224,56,34,true);
  const settings=["legacy10_g75_eta1","steps50_g75_eta1","steps50_g5_eta1","steps50_g5_eta0"];
  const labels=["Original","10 DDIM, g=7.5\neta=1","50 DDIM, g=7.5\neta=1","50 DDIM, g=5\neta=1","50 DDIM, g=5\neta=0"];
  const ids=diagnostic.quality.filter(r=>r.setting===settings[0]).map(r=>r.sample_id);
  const left=110,step=232,top=137,cell=210,gap=12;
  labels.forEach((label,j)=>text(slide,label,left+j*step,78,cell,50,20,true,"center"));
  for(let i=0;i<4;i++){
    const id=ids[i];text(slide,id,4,top+i*(cell+gap)+90,98,28,15,false,"center");
    const paths=[path.join(campaign,"diagnostics","quality_images",settings[0],id+"_source.png"),
      ...settings.map(s=>path.join(campaign,"diagnostics","quality_images",s,id+"_edited.png"))];
    for(let j=0;j<5;j++) slide.images.add({blob:await embeddedPng(paths[j]),contentType:"image/png",alt:`Source ${id}, ${labels[j].replaceAll("\n"," ")}`,
      fit:"contain",position:{left:left+j*step,top:top+i*(cell+gap),width:cell,height:cell}});
  }
  text(slide,"Prespecified first four validation sources. Full padded canvases shown. No image chosen by quality.",60,1021,1224,26,17);
  slide.speakerNotes.text="Actual saved source/generated image files in diagnostics/quality_images. Same source order and generation seeds for the four declared settings. All source and generated images retain their 512x512 canvases and proportions. Photos are raster evidence, each a separate native PPTX picture object. Captions and source labels are editable text. Quantitative quality metrics are in reports/EDITING_PARAMETER_AUDIT.csv. This grid is a frozen-sampler diagnostic, not final trained-arm evidence.";
}

// Slides4-7: all16 development-assessment sources in fixed cohort order.
if(!diagnosticsOnly)for(let block=0;block<4;block++){
  const slide=presentation.slides.add();slide.background.fill="#FFFFFF";
  text(slide,`Final editor assessment (${block+1}/4)`,60,22,1224,56,34,true);
  const keys=["frozen","ddpo_only","rule_rsi","rl_rsi"];
  const labels=["Original","Frozen = Fixed","DDPO-only","Rule RSI","RL-RSI"];
  const cases=state.assessment.frozen.slice(block*4,block*4+4);
  const left=110,step=232,top=137,cell=210,gap=12;
  labels.forEach((label,j)=>text(slide,label,left+j*step,78,cell,50,20,true,"center"));
  for(let i=0;i<4;i++){
    const record=cases[i];text(slide,record.sample_id,4,top+i*(cell+gap)+90,98,28,15,false,"center");
    const paths=[path.join(campaign,"internal_pilot",record.source_visual_path),...keys.map(key=>{
      const row=state.assessment[key][block*4+i];
      if(row.sample_id!==record.sample_id)throw new Error("Assessment source pairing changed");
      return path.join(campaign,"internal_pilot",row.image_path);
    })];
    for(let j=0;j<5;j++)slide.images.add({blob:await embeddedPng(paths[j]),contentType:"image/png",alt:`Assessment ${record.sample_id}, ${labels[j]}`,
      fit:"contain",position:{left:left+j*step,top:top+i*(cell+gap),width:cell,height:cell}});
  }
  text(slide,`Sources ${block*4+1}-${block*4+4} of 16, full padded canvases. Fixed cohort order; no selection by quality.`,60,1021,1224,26,17);
  slide.speakerNotes.text="Measured evidence: internal_pilot/run_state.json assessment and matching raw observations.jsonl. All16 development-assessment sources are shown in cohort order across four slides. All three trained arms receive100 generator updates. Only final checkpoints were assessed; no checkpoint was chosen by these images. The development assessment uses previously exposed source data and is not a sealed final test. Frozen=Fixed are the user-approved equivalence control. Pixel identity is shown as Original. These images do not establish ASR, victim clean utility or complete semantic preservation.";
}

const candidate=path.join(build,"candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidate);
// These are new measured-data charts, with no original Excel formulas/workbook.
// Use the bundled adapter to make honest literal-data workbook snapshots in
// run scratch, then validate the exact portable package. No manual OOXML edits.
const snapshotKey=`${diagnosticsOnly?"qa":"final"}_${Date.now()}`;
const portable=path.join(build,`portable_${snapshotKey}.pptx`);
const snapshotReceipt=path.join(build,`chart_data_${snapshotKey}.json`);
const snapshot=spawnSync(runtimePython,[path.join(skill,"container_tools","materialize_literal_chart_workbooks.py"),
  candidate,portable,"--workspace",build,"--receipt",snapshotReceipt],
  {encoding:"utf8",windowsHide:true,timeout:90000,env:{...process.env,PYTHONDONTWRITEBYTECODE:"1"}});
if(snapshot.error||snapshot.status!==0)throw new Error(`Literal measured-data workbook packaging failed: ${snapshot.stderr}`);
const result=await finalizePresentation({workspaceDir:root,candidatePath:portable,finalPath,
  explicitTotalSlideCount:diagnosticsOnly?2:7,requiredNativeChartOwnerSlides:diagnosticsOnly?[1]:[1,2],requiredNativeTableOwnerSlides:[],
  pythonExecutable:runtimePython,
  integrityValidatorPath:path.join(skill,"container_tools","inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(skill,"container_tools","inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu",`${1344*9525},${1056*9525}`,"--validate-bullet-geometry","--validate-heading-fit"],
  fontPolicy:{basis:"design",families:[font]},verifyArtifactToolImport:true,
  receiptPath:diagnosticsOnly?path.join(build,"..","diagnostic_qa_validation.json"):path.join(evidence,"editable_figures_validation.json")});
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(const [i,slide] of finalDeck.slides.items.entries()){
  const png=await finalDeck.export({slide,format:"png",scale:1});
  await fs.writeFile(path.join(build,`editable_figures_slide${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
if(!diagnosticsOnly)await fs.writeFile(path.join(evidence,"editable_figure_data.json"),JSON.stringify({calibration:cal.controls,validation:state.validation,
  chart_display_rounding:"nine decimal places, preserving full raw values here and in CSV; embedded workbooks are new literal-data snapshots"},null,2));
if(!diagnosticsOnly)await fs.copyFile(snapshotReceipt,path.join(evidence,"chart_literal_data_snapshot.json"));
console.log(JSON.stringify({finalPath,slides:diagnosticsOnly?2:7,requiredNativeCharts:diagnosticsOnly?1:2,embeddedWorkbookSnapshots:true,
  diagnosticsOnly,previewDirectory:build}));
