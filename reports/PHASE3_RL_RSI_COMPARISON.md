# RL/RSI comparison

**PREPARED_NOT_RUN.** No Phase3 M0–M4 performance comparison or generator-RSI gain is claimed. `configs/phase3_comparison.yaml` distinguishes frozen reference, instruction-selection controller, supervised paired-target fitting, reward-based diffusion transitions and configuration adaptation. Prior Phase2 Euler runs are historical evidence, not scheduler-matched controls for new DDIM runs.

Fixed M1/M2/M3 arms plan six updates of batch10=60 training exposures and two validation looks. M4 plans three base plus three candidate updates from a common initial state, total60 exposures/two looks. M0 has no fictitious optimization exposures; its inference count/cost are matched and reported. All generation/judging/selection/rejected-trial/failure costs count toward separate actual GPU caps; a cap may prevent matching exposures, which must be reported.

The first smoke trains one fixed instruction. Three-task controller comparisons need a separately validated instruction/trajectory contract. Human semantic truth and meaningful independent performance evaluation are not established. Therefore the configurations are study specifications, not an automatic launch-ready sweep. RSI may change LR1e-5 to5e-6 only, using fixed reward/splits and independent validation; tiny single-batch gains cannot accept a revision. Accept/reject/reason/cost/rollback records are required when run. No Phase3 proposal/acceptance/rollback trial was executed; Phase2 rejected controller candidate remains unchanged.

Limited stability candidates are batch10/20, LR1e-5/5e-6, rank8/16 and seeds42/123/3407; `configs/phase3_stability.yaml` disables the full grid. Next meaningful single-change run would use seed123 after the remaining gates.

| Method | Implemented/executed evidence | Matched Phase3 performance comparison |
|---|---|---|
| M0 | Phase2 frozen native editor outputs | NOT_RUN |
| M1 | Phase2 external controller, frozen native editor | NOT_RUN |
| M2 | Phase2 three supervised LoRA updates | NOT_RUN |
| M3 | Phase3 implementation; full smoke not PASS | NOT_RUN |
| M4 | Phase2 controller-RSI rejection/rollback only; Phase3 generator-RSI NOT_RUN | NOT_RUN |
