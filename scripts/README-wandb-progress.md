# W&B Progress Reports

Use `wandb_progress_report.py` when you want Codex to analyse experiment
progress without manually exporting CSV files or screenshots from W&B.

## Setup

Log in with W&B once, or export an API key:

```bash
wandb login
# or
export WANDB_API_KEY=...
```

## Basic Usage

From the `dcmpc` directory:

```bash
python scripts/wandb_progress_report.py ENTITY/DCWM \
  --max-runs 30 \
  --samples 1000 \
  --output reports/wandb_progress_report.md
```

Then ask Codex to read `reports/wandb_progress_report.md` and recommend the
next experiment.

## Useful Variants

Filter to a task/group:

```bash
python scripts/wandb_progress_report.py ENTITY/DCWM \
  --filters '{"group": "toy-precision-gate-default"}' \
  --output reports/toy_gate_progress.md
```

Analyse specific runs:

```bash
python scripts/wandb_progress_report.py ENTITY/DCWM \
  --run abc123 \
  --run def456 \
  --output reports/run_comparison.md
```

Add custom metric keys:

```bash
python scripts/wandb_progress_report.py ENTITY/DCWM \
  --metric eval/my_custom_metric \
  --metric train/my_custom_loss \
  --output reports/custom_metrics.md
```

The script analyses the project's default training signals:

- `eval/episodic_return`, `eval/episodic_success`
- `rollout/episodic_return`, `rollout/episodic_success`
- `train/enc_loss`, `train/tc_loss`, `train/reward_loss`
- `train/q_loss`, `train/actor_loss`
- DDCL rate/codebook metrics such as `eval/comms_bits`

