// Headless validation of this project's existing HTML, not a user browser session.
const fs=require('fs'),path=require('path'),http=require('http'),assert=require('assert');
const {chromium}=require('C:/Users/jingb/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'..'),out=path.join(root,'reports/small_scale_browser_20261031');fs.mkdirSync(out,{recursive:true});
const scratch=path.join(root,'tmp','small-scale-qa-'+Date.now());fs.mkdirSync(scratch,{recursive:true});process.env.TEMP=scratch;process.env.TMP=scratch;
const server=http.createServer((req,res)=>{const p=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]));if(!p.startsWith(root+path.sep)){res.writeHead(403).end();return;}try{res.setHeader('Content-Type',p.endsWith('.json')?'application/json':p.endsWith('.html')?'text/html; charset=utf-8':'text/plain');res.end(fs.readFileSync(p));}catch{res.writeHead(404).end();}});
(async()=>{
 let browser;
 try{
  await new Promise(r=>server.listen(0,'127.0.0.1',r));const port=server.address().port;
  browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--disable-gpu']});
  const data=JSON.parse(fs.readFileSync(path.join(root,'dashboard/dashboard_data.json'),'utf8'));assert.deepStrictEqual(data,JSON.parse(fs.readFileSync(path.join(root,'dashboard/state.json'),'utf8')));
  const views=[];
  for(const [lang,width] of [['zh',1440],['en',1440],['zh',390]]){
   const page=await browser.newPage({viewport:{width,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
   await page.goto(`http://127.0.0.1:${port}/dashboard/rsi_poisoning_${lang}.html`);await page.locator('#small-scale-20261031').waitFor();
   const embedded=JSON.parse(await page.locator('#boot-data').textContent());assert.deepStrictEqual(embedded,data);
   const info=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth+1,rows:document.querySelectorAll('#small-scale-20261031 tbody tr').length,text:document.querySelector('#small-scale-20261031').innerText}));
   assert.equal(info.rows,6);assert.equal(info.overflow,false);assert.equal(errors.length,0);assert(info.text.includes('87921'));assert(info.text.includes('78.44%'));assert(info.text.includes('80.62%')||info.text.includes('80.63%'));
   for(const link of await page.locator('#small-scale-20261031 a').all()){
    const href=await link.getAttribute('href');const target=path.resolve(root,'dashboard',href);assert(fs.existsSync(target),target);
   }
   if(width===1440){await page.locator('#small-scale-20261031').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,lang+'_validation.png')});}
   views.push({language:lang,width,errors,overflow:info.overflow,result_rows:info.rows,embedded_matches_authority:true});await page.close();
  }
  fs.writeFileSync(path.join(out,'QA.json'),JSON.stringify({status:'PASS',views,run:data.small_scale_validation_20261031.run_id},null,2));console.log(JSON.stringify({status:'PASS',views}));
 }finally{
  if(browser)await browser.close();server.close();const resolved=fs.realpathSync(scratch);assert(path.dirname(resolved)===path.join(root,'tmp'));assert(path.basename(resolved).startsWith('small-scale-qa-'));fs.rmSync(resolved,{recursive:true,force:true});
 }
})().catch(e=>{console.error(e);process.exitCode=1});
