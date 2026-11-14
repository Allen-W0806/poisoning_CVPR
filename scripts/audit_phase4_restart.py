"""CPU-derived state audit; keep the completed diagnostic unchanged."""
import json
from pathlib import Path
import torch
import _bootstrap
from _phase4 import phase4_run
from src.rl.optimizer_validation import state_equal,tensor_changes
from src.utils.logging import write_json

if __name__=='__main__':
    root=_bootstrap.ROOT
    source=sorted((root/'experiments').glob('*phase4_PH4_RESTART_DIAGNOSTIC'))[-1]
    metrics=json.loads((source/'metrics.json').read_text())
    origin=root/'experiments/EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke'
    original=torch.load(origin/'checkpoints/step_0003/training_state.pt',weights_only=False,map_location='cpu')
    results=[]
    for i in range(2):
        replay=torch.load(source/f'checkpoints/replay_{i}/training_state.pt',weights_only=False,map_location='cpu')
        results.append({'repeat':i,'historical_adapter_exact':state_equal(original['adapter'],replay['adapter']),
            'historical_optimizer_exact':state_equal(original['optimizer'],replay['optimizer']),
            'historical_scheduler_exact':state_equal(original['scheduler'],replay['scheduler']),
            'historical_step_exact':original['step']==replay['step'], 'historical_version_exact':original['version']==replay['version'],
            'historical_consumed_exact':state_equal(original['consumed'],replay['consumed']),
            'rng_scope':'Stored trajectory replay bypasses ten rollout seed draws; final RNG is not a continuation equivalence test',
            'tensor_difference':tensor_changes(original['adapter'],replay['adapter'])})
    forward_exact=all(r['update']['behavior_logp_replay_max_difference']==0 for r in metrics['replays'])
    gradient_exact=all(r['gradient_vs_original']['exact'] for r in metrics['replays'])
    audit={'source_run':source.name,'historical_states':results,
        'first_observed_difference':'backward gradients' if forward_exact and not gradient_exact else 'likelihood or later state',
        'classification':'restored state exact for checked fields; forward likelihood exact, backward numerical differences observed' if forward_exact and not gradient_exact else 'see state comparisons',
        'full_stochastic_continuation_accepted':False,'numeric_acceptance':'No relaxed threshold. Bitwise equality remains required for exact continuation. Descriptive numerical discrepancy only.',
        'original_metric_caveat':'historical_continuation_exact in source run measured adapters only and is false; source evidence retained verbatim'}
    with phase4_run('PH4_RESTART_STATE_AUDIT',{'seed':42,'source_run':source.name,'cpu_only':True},splits=()) as run:
        run.metrics(audit);run.finish(summary='Derived full-state comparison and corrected scope; does not overwrite completed diagnostic.')
        write_json(root/'reports/phase4_latest_restart_audit.json',{'id':run.id,'path':str(run.path)});print(json.dumps(audit))
