const {chromium}=require('C:/Users/jingb/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');
const path=require('path');
(async()=>{
 const out='H:/Project/Poisoning/reports/phase6_browser';fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--disable-gpu']});
 const results=[];
 for(const [lang,size] of [['zh',{width:1440,height:1000}],['en',{width:1440,height:1000}],['zh',{width:390,height:844}]]){
  const page=await browser.newPage({viewport:size});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const response=await page.goto('file:///H:/Project/Poisoning/dashboard/index_'+lang+'.html');
  const parsed=await page.locator('#dashboard-state').textContent();const state=JSON.parse(parsed);
  const info=await page.evaluate(()=>({title:document.title,overflow:document.documentElement.scrollWidth>innerWidth,text:document.body.innerText}));
  if((response && response.status()!==200)||errors.length||info.overflow)throw Error(JSON.stringify({status:response?response.status():200,errors,info}));
  if(lang==='en' && /advisor|audience|presentation purpose|internal instruction/i.test(info.text))throw Error('Forbidden English editorial language');
  const links=await page.locator('a').evaluateAll(as=>as.map(a=>a.href));
  for(const href of links){if(!fs.existsSync(require('url').fileURLToPath(href)))throw Error('Broken local link '+href);}
  await page.screenshot({path:path.join(out,lang+'_'+size.width+'.png')});
  results.push({language:lang,width:size.width,status:response?response.status():200,page_errors:errors,overflow:info.overflow,snapshot:state.updated_utc,links_checked:links.length});
  await page.close();
 }
 const state=JSON.parse(fs.readFileSync('H:/Project/Poisoning/dashboard/state.json','utf8'));
 const id=state.stage_receipts.S1_REVIEW_PACKAGE.experiment_id;
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 await page.goto('file:///H:/Project/Poisoning/experiments/'+id+'/index.html');
 const review=await page.evaluate(()=>({images:document.images.length,missing:[...document.images].filter(x=>!x.complete||!x.naturalWidth).length,filled:[...document.querySelectorAll('input')].filter(x=>x.value).length,forms:document.forms.length}));
 if(review.missing||review.filled||review.forms!==24)throw Error(JSON.stringify(review));
 await page.screenshot({path:path.join(out,'independent_review.png')});
 await browser.close();
 fs.writeFileSync('H:/Project/Poisoning/reports/phase6/browser_qa.json',JSON.stringify({browser:'installed Chrome, Playwright, headless',results,review},null,2));
 process.stdout.write(JSON.stringify({results,review}));
})().catch(e=>{process.stderr.write(String(e));process.exit(1)});
