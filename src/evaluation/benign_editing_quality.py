"""Frozen-edit quality diagnostics. Evaluation only; no training reward hooks."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from src.data.p0_protocol import sha256
from src.mini_pilot.benign_validation import content_box


class EditingQualityMetrics:
    def __init__(self,assets_root,device='cuda'):
        receipt=json.loads((Path(assets_root)/'results/execution_remediation_20261102_v1/lpips_dependency_verification.json').read_text())
        module=Path(receipt['module_root']);calibration=module/'lpips/weights/v0.1/alex.pth';trunk=Path(receipt['base_file'])
        if sha256(calibration)!=receipt['calibration_sha256'] or sha256(trunk)!=receipt['base_sha256']:
            raise ValueError('Calibrated LPIPS dependency checksum mismatch')
        os.environ['TORCH_HOME']=str(trunk.parent.parent.parent);sys.path.insert(0,str(module))
        import lpips
        self.model=lpips.LPIPS(net='alex',version='0.1',lpips=True,pretrained=True,pnet_rand=False,verbose=False).to(device).eval().requires_grad_(False)
        self.device=torch.device(device)
        self.receipt={'package':'lpips==0.1.4','AlexNet_SHA256':sha256(trunk),'calibration_SHA256':sha256(calibration),
                      'input_range':'[-1,1]','direction':'lower = source perceptual distance; not instruction success',
                      'scope':'diagnostic only; never used as generator/controller feedback'}

    @torch.no_grad()
    def measure(self,original,source,edited,resolution):
        from skimage.metrics import structural_similarity
        box=content_box(original,resolution)
        first=np.asarray(source.crop(box),dtype=np.float32)/255.
        second=np.asarray(edited.crop(box),dtype=np.float32)/255.
        if first.shape!=second.shape or not np.isfinite(second).all():raise ValueError('Invalid output pixels')
        def tensor(array):return torch.from_numpy(array.copy()).permute(2,0,1).unsqueeze(0).to(self.device)*2-1
        distance=float(self.model(tensor(first),tensor(second)).mean())
        y=np.array([.2126,.7152,.0722],dtype=np.float32)
        before,after=float((first@y).mean()),float((second@y).mean())
        return {'SSIM':float(structural_similarity(first,second,data_range=1.,channel_axis=2)),
                'LPIPS':distance,'source_luminance':before,'edited_luminance':after,'luminance_delta':after-before,
                'clipped_channel_fraction':float(np.mean((second<=0)|(second>=1))),
                'content_box':box,'exact_pixel_noop':bool(np.array_equal(first,second)),
                'metric_scope':'source-referenced structure/perception; not semantic or instruction ground truth'}
