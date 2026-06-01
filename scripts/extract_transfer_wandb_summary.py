#!/usr/bin/env python3
"""Extract completed-only transfer/Reacher summaries from W&B.

The workshop paper uses this as the source of truth for transfer tables:
finished, non-archive runs are included; running/pending runs are listed
separately and should be marked WIP in paper captions.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb


def _cfg_agent(config: dict[str, Any]) -> dict[str, Any]:
    agent = config.get("agent")
    return agent if isinstance(agent, dict) else {}


def _get_config(config: dict[str, Any], agent: dict[str, Any], key: str) -> Any:
    return agent.get(key) if key in agent else config.get(f"agent.{key}")


def _infer_env(config: dict[str, Any], name: str) -> str:
    raw = str(config.get("env_name") or config.get("env") or config.get("task") or name).lower()
    name_l = name.lower()
    if "button" in raw or "button" in name_l:
        return "button"
    if "reacher" in raw or "reacher" in name_l:
        return "reacher-hard"
    if "walker" in raw or "walker" in name_l:
        return "walker-walk"
    if "dog" in raw or "dog" in name_l:
        return "dog-run"
    if "humanoid" in raw or "humanoid" in name_l:
        return "humanoid-walk"
    return raw


def _infer_seed(tags: list[str], name: str, config: dict[str, Any], agent: dict[str, Any]) -> int | None:
    seed = agent.get("seed", config.get("seed"))
    for text in [" ".join(tags), name]:
        m = re.search(r"seed=(\d+)", text) or re.search(r"-s(\d+)\b", text)
        if m:
            return int(m.group(1))
    return int(seed) if seed is not None else None


def _infer_protocol(tags: list[str], name: str, config: dict[str, Any]) -> str:
    raw = config.get("protocol")
    if isinstance(raw, str) and raw:
        return raw
    for tag in tags:
        if tag.startswith("protocol="):
            return tag.split("=", 1)[1]
    name_l = name.lower()
    if "-stable-" in name_l or name_l.endswith("-stable"):
        return "stable"
    if "-default-" in name_l or name_l.endswith("-default"):
        return "default"
    return "unspecified"


def _method_name(quantizer: str, loss: str | None, det_eval: Any, det_targets: Any) -> str:
    if quantizer == "fsq":
        return "FSQ-CE"
    if quantizer == "vq":
        return "VQ-CE"
    if quantizer == "ddcl":
        if loss == "cosine":
            return "DDCL-Cos"
        if loss == "mse":
            return "DDCL-MSE"
        if loss == "cross-entropy":
            eval_flag = "s" if det_eval is False else "d"
            target_flag = "s" if det_targets is False else "d"
            return f"DDCL-CE({eval_flag},{target_flag})"
        return "DDCL"
    return "Continuous/unknown" if not quantizer else quantizer


def _metric(summary: dict[str, Any], key: str) -> Any:
    flat = summary.get(f"eval/{key}")
    if flat is not None:
        return flat
    nested = summary.get("eval/")
    if nested is not None:
        try:
            nested_dict = dict(nested)
        except Exception:
            nested_dict = nested
        if isinstance(nested_dict, dict):
            return nested_dict.get(key)
    dotted = summary.get(f"eval.{key}")
    return dotted


def _correct_allocated_bits(summary: dict[str, Any], quantizer: str) -> float | None:
    raw = _metric(summary, "rate/allocated_bits_per_transition")
    if quantizer != "ddcl":
        return raw
    loss_pd = _metric(summary, "rate/ddcl_loss_bits_per_dim")
    signed_total = _metric(summary, "rate/ddcl_signed_bits_per_transition")
    signed_pd = _metric(summary, "rate/ddcl_signed_bits_per_dim")
    if loss_pd is not None and signed_total is not None and signed_pd not in (None, 0):
        return float(loss_pd) * round(float(signed_total) / float(signed_pd))
    return raw


def _t_ci(values: list[Any]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    if n == 0:
        return {"n": 0, "values": [], "mean": None, "std": None, "ci95": None}
    if n == 1:
        return {"n": 1, "values": vals, "mean": vals[0], "std": 0.0, "ci95": 0.0}
    tcrit = {
        1: 12.706204736432095,
        2: 4.302652729749464,
        3: 3.182446305284263,
        4: 2.7764451051977987,
        5: 2.570581835636305,
        6: 2.4469118511449692,
        7: 2.3646242510102993,
        8: 2.3060041350333704,
        9: 2.2621571628540993,
        10: 2.2281388519649385,
    }.get(n - 1, 1.959963984540054)
    std = statistics.stdev(vals)
    return {
        "n": n,
        "values": vals,
        "mean": statistics.mean(vals),
        "std": std,
        "ci95": tcrit * std / math.sqrt(n),
    }


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["env"], record["method"], str(record["lambda"]))].append(record)

    rows = []
    for (env, method, lam), records_for_group in sorted(groups.items()):
        perf_key = "episodic_success" if env == "button" else "normalized_return"
        rows.append({
            "env": env,
            "method": method,
            "lambda": None if lam == "None" else lam,
            "n_runs": len(records_for_group),
            "seeds": sorted([
                r["seed"] for r in records_for_group if r.get("seed") is not None
            ]),
            "performance_metric": perf_key,
            "performance": _t_ci([r.get(perf_key) for r in records_for_group]),
            "empirical_entropy": _t_ci([r.get("empirical_entropy") for r in records_for_group]),
            "codebook_usage": _t_ci([r.get("codebook_usage") for r in records_for_group]),
            "allocated_bits": _t_ci([r.get("allocated_bits") for r in records_for_group]),
            "run_ids": [r["id"] for r in records_for_group],
        })
    return rows


def _best_ddcl_cos(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = []
    for env in sorted({row["env"] for row in summary}):
        candidates = [
            row for row in summary
            if row["env"] == env
            and row["method"] == "DDCL-Cos"
            and row["performance"]["mean"] is not None
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda row: (
                row["performance"]["mean"],
                -float(row["empirical_entropy"]["mean"] or 1e18),
            ),
            reverse=True,
        )
        best.append(candidates[0])
    return best


PAPER_ENV_ORDER = ["walker-walk", "reacher-hard", "dog-run", "humanoid-walk", "button"]
PAPER_METHODS = ["DDCL-Cos", "DDCL-CE(s,d)", "FSQ-CE", "VQ-CE"]
PAPER_EXPECTED_SEEDS = [1, 2, 3, 4, 5]
PAPER_PROTOCOL_BY_METHOD = {
    "DDCL-Cos": "stable",
    "DDCL-CE(s,d)": "default",
    "FSQ-CE": "default",
    "VQ-CE": "default",
}


def _lambda_is(value: Any, target: float) -> bool:
    try:
        return math.isclose(float(value), target, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _protocol_matches(record: dict[str, Any], preferred: str) -> bool:
    protocol = record.get("protocol")
    if protocol == preferred:
        return True
    # Some baseline transfer runs predate the protocol tag but are part of the
    # same default fixed-protocol sweep.
    if record.get("quantizer") in {"fsq", "vq"} and protocol == "unspecified" and preferred == "default":
        return True
    return False


def _paper_summary(records: list[dict[str, Any]], fixed_lambda: float = 1e-3) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []

    for env in PAPER_ENV_ORDER:
        for method in PAPER_METHODS:
            preferred_protocol = PAPER_PROTOCOL_BY_METHOD[method]
            candidates = [
                r for r in records
                if r["env"] == env
                and r["method"] == method
                and _lambda_is(r.get("lambda"), fixed_lambda)
                and r.get("seed") in PAPER_EXPECTED_SEEDS
                and _protocol_matches(r, preferred_protocol)
            ]
            by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for candidate in candidates:
                by_seed[int(candidate["seed"])].append(candidate)

            missing = [seed for seed in PAPER_EXPECTED_SEEDS if seed not in by_seed]
            duplicates = {
                str(seed): [r["id"] for r in seed_records]
                for seed, seed_records in by_seed.items()
                if len(seed_records) > 1
            }

            for seed in PAPER_EXPECTED_SEEDS:
                seed_records = by_seed.get(seed, [])
                if not seed_records:
                    continue
                seed_records.sort(key=lambda r: r["id"])
                selected.append(seed_records[0])

            excluded_same_cell = [
                r for r in records
                if r["env"] == env
                and r["method"] == method
                and _lambda_is(r.get("lambda"), fixed_lambda)
                and r.get("seed") in PAPER_EXPECTED_SEEDS
                and not _protocol_matches(r, preferred_protocol)
            ]
            if missing or duplicates or excluded_same_cell:
                notes.append({
                    "env": env,
                    "method": method,
                    "lambda": fixed_lambda,
                    "preferred_protocol": preferred_protocol,
                    "missing_seeds": missing,
                    "duplicate_selected_protocol_seed_run_ids": duplicates,
                    "excluded_other_protocol_run_ids": [r["id"] for r in excluded_same_cell],
                })

    return {
        "selection_rule": (
            "fixed lambda=1e-3; planned seeds 1..5; DDCL-Cos uses protocol=stable; "
            "DDCL-CE(s,d)/FSQ/VQ use protocol=default, accepting unspecified protocol "
            "for older FSQ/VQ baseline runs; one run per seed"
        ),
        "rows": _summary_rows(selected),
        "selection_notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-project", default="f20210966/dcmpc")
    parser.add_argument(
        "--output",
        default="private/Metrics/transfer_wandb_completed_wip_20260530.json",
    )
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    completed: list[dict[str, Any]] = []
    wip: list[dict[str, Any]] = []

    for run in api.runs(args.entity_project):
        tags = list(run.tags or [])
        if "archive" in tags:
            continue
        config = dict(run.config or {})
        agent = _cfg_agent(config)
        name = run.name or ""
        env = _infer_env(config, name)
        if env not in {"button", "reacher-hard", "walker-walk", "dog-run", "humanoid-walk"}:
            continue

        quantizer = str(_get_config(config, agent, "quantizer") or config.get("quantizer") or "").lower()
        loss = _get_config(config, agent, "consistency_loss")
        det_eval = _get_config(config, agent, "ddcl_deterministic_eval")
        det_targets = _get_config(config, agent, "ddcl_deterministic_targets")
        lam = _get_config(config, agent, "ddcl_lambda")
        summary = dict(run.summary or {})
        ret = _metric(summary, "episodic_return")
        norm_ret = _metric(summary, "normalized_return")
        if norm_ret is None and ret is not None and env != "button":
            norm_ret = float(ret) / 1000.0

        record = {
            "id": run.id,
            "name": name,
            "state": run.state,
            "tags": tags,
            "env": env,
            "method": _method_name(quantizer, loss, det_eval, det_targets),
            "quantizer": quantizer,
            "protocol": _infer_protocol(tags, name, config),
            "loss": loss,
            "lambda": lam,
            "det_eval": det_eval,
            "det_targets": det_targets,
            "seed": _infer_seed(tags, name, config, agent),
            "normalized_return": norm_ret,
            "episodic_return": ret,
            "episodic_success": _metric(summary, "episodic_success"),
            "empirical_entropy": _metric(summary, "rate/empirical_entropy_bits_per_transition"),
            "codebook_usage": _metric(summary, "codebook/usage_percent"),
            "allocated_bits": _correct_allocated_bits(summary, quantizer),
        }

        if run.state == "finished":
            completed.append(record)
        elif run.state in {"running", "pending", "preempted"}:
            wip.append(record)

    summary = _summary_rows(completed)
    out = {
        "meta": {
            "source": f"W&B {args.entity_project}",
            "inclusion": "finished non-archive runs only; running/pending/preempted listed separately",
            "performance_rule": "DMControl normalized_return; fallback episodic_return/1000; Button episodic_success",
            "rate_rule": "eval/rate/empirical_entropy_bits_per_transition for transfer/Reacher; codebook usage is diagnostic",
            "lambda_selection": "best DDCL-Cos lambda per environment chooses highest completed mean performance; tie-breaker lower empirical entropy",
        },
        "completed_runs": completed,
        "wip_runs": wip,
        "summary": summary,
        "paper_summary_fixed_lambda_1e-3": _paper_summary(completed, fixed_lambda=1e-3),
        "best_ddcl_cos_lambda_by_env_completed_only": _best_ddcl_cos(summary),
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}")
    print(f"completed={len(completed)} wip={len(wip)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
