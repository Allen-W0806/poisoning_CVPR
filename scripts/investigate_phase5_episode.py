"""Acquire new CPU observations for one saved TRAINING episode; no model calls."""
import json
from pathlib import Path
import _bootstrap
from src.utils.logging import Experiment,write_json
from src.rsi.harness.trace_collector import TraceCollector
from src.rsi.harness.harness_executor import HarnessExecutor
from src.rsi.harness.failure_investigator import FailureInvestigator

def main():
    root=_bootstrap.ROOT;parent='EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL'
    m=json.loads((root/'experiments'/parent/'metrics.json').read_text());obs=m['updates'][0]['observations'][0]
    train=json.loads((root/'data/processed/vqav2_seed42_v1/train.json').read_text())
    record=next(r for r in train if r['sample_id']==obs['sample_id'])
    assert record['image_id']==obs['source_image_id'] and record['question']==obs['question'] and record['answers']==obs['answers']
    receipt=json.loads((root/'reports/phase4_evidence.json').read_text())
    index=next(r for r in receipt['phase4_runs'] if r['id']==parent)
    path=Path(obs['source_path']);rel=path.relative_to(root/'experiments'/parent).as_posix()
    expected=next(r['sha256'] for r in index['files'] if r['path']==rel)
    trajectory=Path(obs['output_path']).parents[1]/'checkpoints'/f"train1_trajectory_0.pt"
    case={'case_id':obs['sample_id'],'split':'train','image_path':str(path),'expected_size':[512,512],
        'expected_sha256':expected,'source_path':obs['source_path'],'edited_path':obs['output_path'],
        'trajectory_path':str(trajectory),'question':record['question'],'original_answers':record['answers'],
        'instruction':obs['instruction'],'generation_metadata':obs['settings'],'stored_quality':obs['quality'],
        'stored_semantic_proxy':obs['semantic'],'edited_factual_answer':'UNKNOWN'}
    with Experiment(root,'phase5_active_saved_episode',{'phase':5,'training_parent':parent,'case':obs['sample_id'],'new_model_calls':0}) as run:
        trace=TraceCollector(run.path/'trace.jsonl');trace.append('retrieved_training_context',case)
        investigator=FailureInvestigator(HarnessExecutor(root),trace,[case],max_queries=4)
        headers=investigator.observe(case,'image_headers');digest=investigator.observe(case,'content_hash')
        quality=investigator.observe(case,'pair_quality');trajectory_summary=investigator.observe(case,'trajectory_summary')
        matched=all(abs(quality['observation'][k]-obs['quality'][k])<1e-8 for k in ['mse','mae','ssim','distortion'])
        result={'parent':parent,'case':obs['sample_id'],'queries':investigator.queries,'resources':investigator.resources.__dict__,
            'source_manifest_matches':digest['observation']['actual_sha256']==expected,'stored_quality_reproduced':matched,
            'contradicted':['saved_vs_recomputed_numeric_quality_discrepancy'] if matched else [],
            'quality_scope':'Same implementation recomputation confirms saved arithmetic, not metric implementation or construct validity',
            'unresolved':['instruction_following_failure','question_answer_validity_error','semantic_verifier_disagreement'],
            'revision_proposed':False,'reason':'Source and numerical quality corroborated; independent semantic/task facts absent',
            'new_model_calls':0,'new_optimizer_updates':0}
        assert result['source_manifest_matches'] and matched and trajectory_summary['observation']['finite_states']
        run.metrics(result);run.finish(summary='Four newly acquired CPU observations on a saved training episode; semantic cause remains unresolved, no invented revision.')
        write_json(root/'reports/phase5_saved_episode_investigation.json',{'id':run.id,'path':str(run.path),'results':result})
    print(json.dumps({'id':run.id,'queries':4,'semantic_status':'UNRESOLVED','new_model_calls':0}))

if __name__=='__main__':main()
