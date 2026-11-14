"""Read-only syntax/entrypoint/stub audit; never imports or runs the target."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re


def inspect(source_path,readme_path,slurm_path):
    source_path=Path(source_path);raw=source_path.read_bytes();source=raw.decode('utf-8-sig')
    tree=ast.parse(source,filename=str(source_path));compile(tree,str(source_path),'exec')
    functions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    stubs=[]
    for node in functions:
        raises=[n for n in ast.walk(node) if isinstance(n,ast.Raise) and isinstance(n.exc,ast.Call)
                and isinstance(n.exc.func,ast.Name) and n.exc.func.id=='NotImplementedError']
        if raises:stubs.append({'function':node.name,'definition_line':node.lineno,'raise_lines':[n.lineno for n in raises]})
    main_blocks=[n for n in tree.body if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test)]
    main_calls=[{'call':ast.unparse(n.func),'line':n.lineno} for block in main_blocks for n in ast.walk(block) if isinstance(n,ast.Call)]
    all_calls=[{'call':ast.unparse(n.func),'line':n.lineno} for n in ast.walk(tree) if isinstance(n,ast.Call)]
    persistence=[n for n in all_calls if n['call'] in {'torch.save','torch.load','json.dump','json.dumps','open','safetensors.torch.save_file'}
                 or n['call'].endswith(('.write_text','.write_bytes','.save_pretrained','.load_state_dict'))]
    slurm=Path(slurm_path).read_text(encoding='utf-8-sig');directives=[]
    for index,line in enumerate(slurm.splitlines(),1):
        if line.startswith('#SBATCH '):directives.append({'line':index,'directive':line[8:].strip()})
    time_limit=re.search(r'^#SBATCH\s+--time[= ]([^\s]+)',slurm,re.M)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source_path,Path(readme_path),Path(slurm_path)]}
    return {'status':'STATIC_SYNTAX_PASS_READINESS_NOT_ESTABLISHED','checked_UTC':datetime.now(timezone.utc).isoformat(),
       'scope':'Read-only structural inspection, not model/optimizer/reward execution or attack-loop integration.',
       'input_SHA256':hashes,'syntax_parse_compile':'PASS','target_imported':False,'target_executed':False,
       'GPU_jobs_submitted':0,'stub_count':len(stubs),'stubs':stubs,
       'functions':[{'name':n.name,'line':n.lineno,'end_line':n.end_lineno} for n in functions],
       'main_calls':main_calls,'persistence_calls_detected':persistence,'slurm_directives':directives,
       'slurm_time_limit':time_limit.group(1) if time_limit else None,
       'readme_claims_are_execution_evidence':False,
       'limitations':['Static syntax success does not demonstrate importability or end-to-end execution.',
                      'Symbol absence is a scoped source observation, not proof about every external dependency.',
                      'No sampler, gradients, optimizer, checkpoint restore, model quality, or method efficacy was dynamically tested.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--readme',required=True)
    parser.add_argument('--slurm',required=True);parser.add_argument('--output',required=True);a=parser.parse_args()
    value=inspect(a.source,a.readme,a.slurm);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    print(json.dumps(value,ensure_ascii=False))
