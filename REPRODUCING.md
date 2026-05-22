# Reproducing the DDCL workshop experiments

This file documents the paper-facing reproduction path for the DDCL, FSQ, VQ,
and continuous-latent comparisons. The original DC-MPC commands in `README.md`
still work; the scripts below are the exact experiment families used for the
workshop analysis.

## Pre-flight checks

Run these checks before launching training or checkpoint re-evaluation:

```bash
python scripts/sanity_check_eval_metrics.py
python -m unittest tests/test_metric_invariants.py
python -m py_compile train.py eval.py scripts/sanity_check_eval_metrics.py \
  results/plotting/make_paper_plots.py
bash -n scripts/reproduce_toy_paper.sh scripts/reproduce_remote_paper_runs.sh \
  scripts/run_toy_ddcl_objective_completion.sh \
  scripts/reevaluate_toy_objective_checkpoints.sh
```

`scripts/sanity_check_eval_metrics.py` verifies success aggregation, codebook
usage aggregation, empirical entropy accounting, Hydra composition for the
paper toy config, and the DDCL nominal-capacity caveat.

## Toy experiments

Use the dispatcher for the exact toy run families:

```bash
STAGE=baselines scripts/reproduce_toy_paper.sh
STAGE=ddcl_ce_repair scripts/reproduce_toy_paper.sh
STAGE=ddcl_objectives scripts/reproduce_toy_paper.sh
STAGE=ddcl_objective_finish scripts/reproduce_toy_paper.sh
```

The paper-facing DDCL objective settings are:

- `ddcl_mse`: DDCL quantizer with MSE consistency loss.
- `ddcl_cosine`: DDCL quantizer with cosine consistency loss.

Both use `toy-precision-gate-final`, deterministic DDCL eval/targets,
`plan_unc_prop_mode=weighted-avg`, `unc_prop_mode=sample`,
`ddcl_scale=3.5`, `ddcl_delta=1.0`, and `ddcl_lambda=0.001`.

Toy quantizer Pareto sweep:

```bash
STAGE=pareto DEVICE=auto scripts/reproduce_toy_paper.sh
```

This varies FSQ/VQ maximum codebook size and DDCL scale/lambda. `device=auto`
uses CUDA when available and CPU otherwise. Apple MPS is intentionally not used
by `auto` because the current TorchRL/planning path can crash in Metal; use
explicit `device=mps` only for debugging.

Re-evaluate checkpoints with the corrected eval protocol:

```bash
STAGE=reeval_baselines scripts/reproduce_toy_paper.sh

SOURCE_ROOTS="output/toy_runs/ddcl_objective_sweep_20260514_073320/hydra \
output/toy_runs/ddcl_objective_completion_YYYYMMDD_HHMMSS/hydra" \
STAGE=reeval_objectives scripts/reproduce_toy_paper.sh
```

Known 2026-05-16 local artifacts:

- Corrected DDCL cosine/MSE objective checkpoint eval:
  `output/toy_runs/objective_checkpoint_reeval_20260516_172600`.
- Held-out DDCL cosine/MSE objective checkpoint eval:
  `output/toy_runs/objective_checkpoint_heldout_eval_20260516_173200`.
- Held-out main paper checkpoint eval:
  `output/toy_runs/paper_checkpoint_heldout_eval_20260516_174100`.
- Current CPU Pareto launch:
  `output/toy_runs/quantizer_pareto_20260516_175600`, W&B project
  `ddcl_mbrl_toy_pareto`, `MAX_PARALLEL=4`.

For long-running launches, record the run root and command context, then return
control to the user. Do not poll or monitor unless explicitly asked.

## Plotting

Generate toy plots from saved aggregate metrics:

```bash
python results/plotting/make_paper_plots.py \
  --input "private/Metrics/Toy/Corrected Reevaluation/wandb_toy_corrected_reeval_20260515.json" \
  --input "private/Metrics/Toy/Corrected Reevaluation/wandb_toy_objective_corrected_reeval_20260515.json" \
  --outdir results/paper_plots/toy
```

The same plotting script can read CSV exports or W&B summaries:

```bash
python results/plotting/make_paper_plots.py \
  --input path/to/metrics.csv \
  --outdir results/paper_plots/custom

python results/plotting/make_paper_plots.py \
  --wandb-project ENTITY/PROJECT \
  --wandb-filters '{"tags": "paper"}' \
  --outdir results/paper_plots/from_wandb
```

Paper comparisons should use empirical entropy bits per transition as the fair
rate axis. Codebook usage is a diagnostic unless DDCL/FSQ/VQ cardinalities are
explicitly matched.

## Remote cluster runs

Do not run Meta-World or DMControl locally for the workshop workflow. Use the
remote dispatcher:

```bash
STAGE=toy_pareto scripts/reproduce_remote_paper_runs.sh
STAGE=metaworld_baselines scripts/reproduce_remote_paper_runs.sh
STAGE=metaworld_ddcl_sensitivity scripts/reproduce_remote_paper_runs.sh
STAGE=dmcontrol_baselines scripts/reproduce_remote_paper_runs.sh
```

Useful overrides:

```bash
LAUNCHER=slurm DEVICE=cuda GRES=gpu:a100:1 \
WANDB_METAWORLD_PROJECT_NAME=ddcl_mbrl_metaworld \
WANDB_DMCONTROL_PROJECT_NAME=ddcl_mbrl_dmcontrol \
STAGE=dmcontrol_baselines scripts/reproduce_remote_paper_runs.sh
```

Keep `restore_best_checkpoint_at_end=true` and
`log_best_checkpoint_eval=true`. `train.py` saves the selected `checkpoint.pt`
locally, uploads it to W&B as a model artifact with aliases
`best`, `eval-checkpoint`, and `latest`, and writes W&B summary fields under
`eval_checkpoint/*`. The W&B run note marks that artifact as the checkpoint to
use for later `eval.py` runs.

The 2026-05-16 MPS Pareto attempts are not reproducible paper results:
`output/toy_runs/quantizer_pareto_20260516_164900` failed under Apple MPS and
`output/toy_runs/quantizer_pareto_20260516_165200` was stopped during device
debugging. Prefer `DEVICE=cpu` locally and `DEVICE=cuda` on clusters.

## Rerun criteria

Retrain only when the training algorithm or checkpoint-selection policy changes.
When eval metrics, plotting, or W&B reporting changes, re-run `eval.py` on the
saved checkpoints instead. If a required checkpoint is missing locally, use the
W&B `eval-checkpoint` artifact for that run.
