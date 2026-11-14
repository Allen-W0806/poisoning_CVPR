"""Read-only audit of a withdrawn BF16 flag proposal. No GPU stage is admitted."""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mini_pilot.data import sha256


def preflight(c):
    import torch
    source=ROOT/'src/rl/diffusion_policy.py'
    definition=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DiffusionPolicy')
    constructor=next(n for n in definition.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assignments=[n for n in constructor.body if isinstance(n,ast.Assign) and
                 any(ast.unparse(t)=='torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction' for t in n.targets)]
    disabled=[n for n in assignments if isinstance(n.value,ast.Constant) and n.value.value is False]
    original=json.loads((c/'internal_pilot/protocol.json').read_text())['source_SHA256']
    return {'status':'WITHDRAWN_BEFORE_GPU_SUBMISSION','GPU_seconds':0,'torch':torch.__version__,
        'process_default_before_policy_construction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'actual_policy_constructor_already_disables_reduced_precision_reduction':bool(disabled),
        'unconditional_assignment_lines':[n.lineno for n in disabled],
        'protected_policy_source_SHA256':sha256(source),
        'matches_active_pilot_source':sha256(source)==original['src/rl/diffusion_policy.py'],
        'reason':'Process-default True was not the effective policy value; existing BF16 failure already occurred with the flag disabled. This is not a new repair candidate.',
        'proposal':'rejected_proposals/check_bf16_reduction_candidate_UNEXECUTED.py',
        'no_model_or_image_loaded':True,'BF16_repair_claim':False}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--campaign',required=True,type=Path)
    p.add_argument('--stage',choices=['preflight','diagnose'],required=True);args=p.parse_args()
    if args.stage=='diagnose':raise RuntimeError('Withdrawn proposal: no GPU diagnosis is admitted')
    c=args.campaign.resolve()
    if not c.is_relative_to(ROOT/'outputs'):raise ValueError('Expected canonical campaign')
    print(json.dumps(preflight(c),indent=2))


if __name__=='__main__':main()
