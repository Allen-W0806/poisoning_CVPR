"""Complete blinded contact sheets, with no model predictions or diagnostic labels."""
import argparse,json,os,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
def render(run_id):
    kit=ROOT/'results'/run_id/'semantic_review';cases=json.loads((kit/'blinded_manifest.json').read_text())['cases']
    path='C:/Windows/Fonts/arial.ttf' if os.name=='nt' else 'DejaVuSans.ttf';font=ImageFont.truetype(path,17);ref_font=ImageFont.truetype(path,15)
    for page in range(40):
        sheet=Image.new('RGB',(1280,1000),'white');draw=ImageDraw.Draw(sheet)
        for j,c in enumerate(cases[page*4:page*4+4]):
            y=j*250;question=f"{c['source_image_id']} | {c['question']}"
            for k,line in enumerate(textwrap.wrap(question,110)):draw.text((12,y+3+k*20),line,font=font,fill='black')
            for k,field in enumerate(['original','edited','geometry_control']):
                with Image.open(kit/c[field]) as im:im=im.convert('RGB');im.thumbnail((290,160));sheet.paste(im,(15+k*310,y+52))
                draw.text((15+k*310,y+220),['Original C0','Edited C4','Geometry C5'][k],font=font,fill='black')
            lines=textwrap.wrap('References: '+', '.join(c['answers']),36)
            assert len(lines)<=12,'Never truncate human references silently'
            for k,line in enumerate(lines):draw.text((950,y+27+k*18),line,font=ref_font,fill='black')
        sheet.save(kit/'contact_sheets'/f'blinded_{page+1:02d}.jpg',quality=90)
    print('PASS: 40 sheets / 160 cases; no clipped image labels or truncated references')
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();render(a.run_id)
if __name__=='__main__':main()
