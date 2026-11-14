# Smoke test report

Suite: `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu`

| Test | Status | Actual command | Evidence |
|---|---|---|---|
| A environment | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python scripts/check_environment.py --output /storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/environment_probe.json` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_A_environment.log` |
| C dataset | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python scripts/smoke_dataset.py` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_C_dataset.log` |
| D editor | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python scripts/smoke_editor.py` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_D_editor.log` |
| E rl | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python scripts/smoke_rl.py --device cuda` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_E_rl.log` |
| F rsi | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python scripts/smoke_rsi.py` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_F_rsi.log` |
| G evaluation_units | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python -m unittest tests.test_evaluation_reward -v` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_G_evaluation_units.log` |
| G_SECURITY security_fixture | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python scripts/run_evaluation.py --fixture tests/fixtures/security_synthetic.json --fixture-id synthetic-observations-v1 --synthetic` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_G_SECURITY_security_fixture.log` |
| G_ORACLE clean_fixture | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python scripts/run_clean_baseline.py --oracle-fixture` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_G_ORACLE_clean_fixture.log` |
| H logging | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python scripts/smoke_logging.py` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_H_logging.log` |
| UNIT all_units | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python -m unittest discover -v` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_UNIT_all_units.log` |
| B qwen | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python scripts/smoke_qwen.py --device cuda` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_B_qwen.log` |
| G_REAL clean_vqav2 | PASS | `/storage/home/ydk297/projects/poisoning/.venv/bin/python scripts/run_clean_baseline.py --device cuda --limit 10` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu/test_G_REAL_clean_vqav2.log` |
| I slurm | PASS | `python scripts/submit_smoke.py` | `/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T035054_8851028cc7e5_slurm_smoke_gpu` |

Synthetic fixtures validate engineering only. Real VQAv2 and model inference are separately named.
All earlier failures and fixes are retained in reports/FAILURES.md and immutable failed runs.
