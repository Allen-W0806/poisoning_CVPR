"""Prepare frozen M0–M5 planning artifacts, never launch a campaign."""
import copy
import json
from pathlib import Path
import statistics
import yaml
import _bootstrap
from src.utils.logging import write_json

def main():
    root=_bootstrap.ROOT;target=root/'configs/phase5_v2';target.mkdir(exist_ok=False)
    config=yaml.safe_load((root/'configs/phase3_generator_rl.yaml').read_text())
    dataset=json.loads((root/'reports/phase5_dataset_v2.json').read_text())['provenance']
    readiness={'independent_editing_validity':False,'validated_training_reward':False,'content_disjoint_splits':True,
      'active_harness_tests':True,'conventional_scheduler_baseline':True,'pre_intervention_comparability':False,
      'private_gate_isolation':False,'frozen_resource_protocol':True}
    names=['frozen_generator','standard_generator_RL','conventional_scheduler',
           'trace_only_no_revision','diagnosis_guided_revision','active_independently_gated_revision']
    paths={}
    for i,name in enumerate(names):
        value={'schema_version':'phase5_comparison_v2','supersedes':'configs/phase5/'+f'M{i}.yaml','arm':f'M{i}','condition':name,'status':'PREPARED_NOT_RUN',
          'backend':{'policy':'src.rl.diffusion_policy_v2.ContentBoundDiffusionPolicy','trainer':'src.rl.diffusion_rl_trainer_v2.GuardedDiffusionRLTrainer',
                     'factory':'src.rl.phase5_backend.make_backend','parameter_target':'IP2Pix_UNet_attention_LoRA'},
          'seeds':[42,123,2026],'optimizer_updates':0 if i==0 else 8,'training_source_exposures':0 if i==0 else 80,
          'generator_config':copy.deepcopy(config),'dataset_version':'vqav2_content_v2','split_sha256':dataset['split_sha256'],
          'development_sources':64,'private_gate_sources':64,'development_looks':[2,4,6,8] if i else [0],
          'final_test':'operator-only, freeze selected harness before access; old v1 test already opened',
          'rsi_mode':name if i>=3 else 'none','max_candidate_attempts':2 if i>=4 else 0,
          'training_reward_config':'configs/phase5_reward.yaml','training_reward_ready':False,
          'launch_readiness':readiness,'launch_authorized':False,'strict_continuation_accepted':False,
          'pre_intervention_checks':['adapter','optimizer','scheduler','RNG','source_order','noise_seeds','contract','frozen_weights'],
          'statistical_unit':'source-image clusters within seed; seed-level estimates kept distinct',
          'selection_policy':'independent frozen validity, paired uncertainty and measured complete cost; no per-item Gate feedback'}
        if i==2:value['scheduler_baseline']={'kind':'predeclared_one_cooldown','after_update':4,'learning_rate_before':1e-5,'learning_rate_after':5e-6,'classification':'ordinary_scheduler'}
        p=target/f'M{i}.yaml'
        with p.open('x') as f:yaml.safe_dump(value,f,sort_keys=False)
        paths[f'M{i}']=str(p)
    # Prospective planning, scaled explicitly from historical realized cost.
    old=json.loads((root/'reports/phase4_results.json').read_text())
    per_run=[v for f in old['fairness'] for v in f['actual_gpu_seconds']]
    times=[]
    for name in ['EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL','EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL',
                 'EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI','EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI']:
        m=json.loads((root/'experiments'/name/'metrics.json').read_text())
        times.extend(o['generation_seconds']+o['reward_seconds'] for v in m['evaluations'] for o in v['observations'])
    seconds=statistics.median(times);base=statistics.median(per_run)
    trained_runs=5*3;extra_development=(64-10)*4
    trained=trained_runs*(base+extra_development*seconds)
    frozen=3*64*seconds
    gate=2*3*2*2*64*seconds # M4/M5,3seeds,2attempts,parent+candidate,64sources.
    estimate={'type':'planning_estimate_not_an_allocation','historical_gpu_seconds_per_8update_run':per_run,
       'historical_validation_seconds_per_observation_median':seconds,
       'scenario':{'seeds':3,'trained_arms':5,'updates_per_arm':8,'development_sources':64,'gate_sources':64,'attempts_per_evolution_arm':2},
       'trained_plus_development_gpu_hours':trained/3600,'frozen_generation_gpu_hours':frozen/3600,'paired_gate_gpu_hours':gate/3600,
       'subtotal_gpu_hours':(trained+frozen+gate)/3600,'with_20percent_planning_overhead_gpu_hours':1.2*(trained+frozen+gate)/3600,
       'limits':'Old automatic-verifier runtime only. Independent human review labor and future evaluator/model-call behavior are not measured. No clean-target fine-tuning cost estimate is asserted.',
       'next_authorized_engineering_gpu_hours':0}
    write_json(root/'reports/phase5_comparison_prepared_v2.json',{'config_paths':paths,'readiness':readiness,'overall_decision':'NO-GO','resource_estimate':estimate})
    print(json.dumps({'configs':6,'status':'PREPARED_NOT_RUN','estimated_gpu_hours':estimate['with_20percent_planning_overhead_gpu_hours']}))

if __name__=='__main__':main()
