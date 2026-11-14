"""Fixed-dev preprocessing diagnosis only; no test queries or optimizer."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.p0_protocol import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets-root',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    assets,run=Path(a.assets_root),Path(a.run);parent=run.parent
    protocol=json.loads((run/'protocol.json').read_text())
    manifest=json.loads((parent/'manifest.json').read_text())
    if sha256(parent/'manifest.json')!=protocol['manifest_SHA256']:raise ValueError('Manifest changed')
    import numpy as np
    import torch
    from PIL import Image
    from skimage.metrics import structural_similarity
    from src.editing.native_preprocessing import native_aspect_image
    from src.models.pretrained_editor_loader import pretrained_config
    from src.editing.pretrained_editor import PretrainedEditor
    from src.evaluation.benign_editing_quality import EditingQualityMetrics
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise ValueError('Slurm CUDA required')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();state={'status':'RUNNING','SLURM_JOB_ID':os.getenv('SLURM_JOB_ID'),'optimizer_updates':0}
    try:
        editor=PretrainedEditor(pretrained_config(assets,precision='bf16',seed=42,image_resolution=512,
                           num_inference_steps=30,guidance_scale=7.5,image_guidance_scale=1.5),device='cuda')
        metric=EditingQualityMetrics(assets)
        rows=[]
        for setting in protocol['configs']:
            for index,record in enumerate(manifest['splits']['dev']):
                if time.perf_counter()-started>240:raise TimeoutError('P0 inference-only followup cap')
                with Image.open(assets/record['image_path']) as image:source=native_aspect_image(image)
                path=run/'images'/setting['name']/f"{record['sample_id']}.png";path.parent.mkdir(parents=True,exist_ok=True)
                source_path=run/'sources'/f"{record['sample_id']}.png";source_path.parent.mkdir(exist_ok=True);source.save(source_path)
                seed=42000000+index;g=torch.Generator(device='cuda').manual_seed(seed);t=time.perf_counter()
                with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    image=editor.pipe(prompt=protocol['instruction'],image=source,num_inference_steps=30,
                        guidance_scale=setting['text_guidance'],image_guidance_scale=setting['image_guidance'],generator=g).images[0]
                image.save(path)
                x=np.asarray(source,dtype=np.float32)/255.;y=np.asarray(image,dtype=np.float32)/255.
                if x.shape!=y.shape or not np.isfinite(y).all():raise ValueError('Native geometry/output invalid')
                def tensor(value):return torch.from_numpy(value.copy()).permute(2,0,1).unsqueeze(0).cuda()*2-1
                with torch.no_grad():lpips=float(metric.model(tensor(x),tensor(y)).mean())
                luma=np.array([.2126,.7152,.0722],dtype=np.float32)
                row={'config':setting['name'],'sample_id':record['sample_id'],'image_id':record['image_id'],
                     'source_path':source_path.relative_to(parent).as_posix(),'path':path.relative_to(parent).as_posix(),
                     'source_SHA256':sha256(source_path),'image_SHA256':sha256(path),'dimensions':list(source.size),
                     'instruction':protocol['instruction'],'seed':seed,'steps':30,'settings':setting,
                     'SSIM':float(structural_similarity(x,y,data_range=1.,channel_axis=2)),
                     'LPIPS':lpips,'luminance_delta':float((y@luma).mean()-(x@luma).mean()),
                     'seconds':time.perf_counter()-t,'optimizer_updates':0}
                rows.append(row)
                with (run/'observations.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        (run/'observations.json').write_text(json.dumps(rows,indent=2))
        state.update(status='COMPLETED_MEASURED',observations=len(rows),summary={setting['name']:{key:float(np.mean([r[key] for r in rows if r['config']==setting['name']])) for key in ('SSIM','LPIPS','luminance_delta')} for setting in protocol['configs']})
    except Exception:
        state.update(status='FAILED',trace=traceback.format_exc());(run/'failure.txt').write_text(state['trace']);raise
    finally:
        state['elapsed_seconds']=time.perf_counter()-started;(run/'status.json').write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
