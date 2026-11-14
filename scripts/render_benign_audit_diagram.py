"""Editable vector workflow only: no empirical improvement is implied."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'output/benign_ddpo_audit_20261111'


def main():
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    OUTPUT.mkdir(parents=True,exist_ok=True)
    width,height = 1200,640
    boxes = [
        (45,125,245,140,'LOCKED INPUTS',['Pinned IP2Pix base','14 source-disjoint images','Train 4 / dev 2 / test 4','Darkening transfer: 4'],'#edf3fa'),
        (340,125,245,140,'NUMERICAL GATE',['Native frozen inference','BF16 and FP32 repeats','Raw gradients / Adam / RNG','Checkpoint next-step check'],'#eaf4ef'),
        (635,125,245,140,'INNER CONTROL',['Shared zero-LoRA start','4 fresh on-policy updates','Terminal benign reward','Paired Frozen / Inner test'],'#edf3fa'),
        (930,125,225,140,'OUTER SEARCH',['Fixed / rule / UCB','3 proposals per method','Equal rollout / dev budget','Accept or retain incumbent'],'#edf3fa'),
        (340,365,540,120,'ASSESSMENT BOUNDARY',['Dev only selects; assessment and transfer do not select.','Lighting + edge reward are proxies, without edited ground truth.','Learning benefit and transferable RSI value require separate evidence.'],'#fff4dd')]
    title = 'Benign IP2Pix validation: implementation and evidence boundaries'
    subtitle = 'Conceptual workflow for the fixed smoke protocol. Numerical execution is distinct from editing efficacy.'
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="1200" height="640" fill="white"/>',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#60748c"/></marker></defs>']
    pdf = canvas.Canvas(str(OUTPUT/'benign_validation_framework.pdf'),pagesize=(width,height))
    pdf.setTitle(title)
    def text(x,y,value,size=16,bold=False,color='#24354b'):
        svg.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(value)}</text>')
        pdf.setFillColor(HexColor(color));pdf.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
        pdf.drawString(x,height-y,value)
    def line(x1,y1,x2,y2,arrow=True):
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#60748c" stroke-width="2"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
        pdf.setStrokeColor(HexColor('#60748c'));pdf.setLineWidth(2);pdf.line(x1,height-y1,x2,height-y2)
        if arrow:
            import math
            angle=math.atan2(y2-y1,x2-x1)
            path=pdf.beginPath();path.moveTo(x2,height-y2)
            for offset in (-.45,.45):
                path.lineTo(x2-10*math.cos(angle+offset),height-(y2-10*math.sin(angle+offset)))
            path.close();pdf.setFillColor(HexColor('#60748c'));pdf.drawPath(path,fill=1,stroke=0)
    text(45,47,title,25,True)
    text(45,78,subtitle,16)
    for x,y,w,h,heading,rows,color in boxes:
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{color}" stroke="#b8c5d3"/>')
        pdf.setFillColor(HexColor(color));pdf.setStrokeColor(HexColor('#b8c5d3'))
        pdf.roundRect(x,height-y-h,w,h,10,fill=1,stroke=1)
        text(x+16,y+29,heading,17,True)
        for index,row in enumerate(rows): text(x+16,y+57+23*index,row,14)
    for x1,x2 in ((290,340),(585,635),(880,930)):line(x1,195,x2,195)
    line(757,265,757,350)
    line(1042,265,1042,425,False);line(1042,425,890,425)
    text(45,560,'Implemented: stochastic DDIM + FP32 LoRA/Adam + strict checkpoint integrity + bounded recipe search.',16)
    text(45,592,'Not established by this smoke: broad BF16 repair, useful editing gains, trained Outer PPO, or transferable RSI.',16)
    text(45,620,'Source: configs/benign_ddpo_minimal.json. Actual observations remain in logs and checkpoints.',13,color='#60748c')
    svg.append('</svg>')
    (OUTPUT/'benign_validation_framework.svg').write_text('\n'.join(svg),encoding='utf-8')
    pdf.showPage();pdf.save()
    print(str(OUTPUT))


if __name__=='__main__':main()
