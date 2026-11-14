# Fresh-agent documentation invocation

A fresh agent read README.md and the provider ledger, then executed the documented command without modifications.

Local: `python -m unittest discover -v`, working directory `H:\Project\Poisoning`, executable `H:\Python\python.exe`; exit 0, seven tests PASS, unittest duration 3.284s. No documentation divergence observed.

Remote: `ssh utc-login 'cd /storage/home/ydk297/projects/poisoning; source .venv/bin/activate; python -m unittest discover -v'`; activated executable `/storage/home/ydk297/projects/poisoning/.venv/bin/python`; exit 0, seven tests PASS, duration 2.974s. No documentation divergence observed.

These are CPU/source contract checks, not independent CUDA certification. The primary executor separately recorded actual allocated A100 kernel, Qwen and RL tests.
