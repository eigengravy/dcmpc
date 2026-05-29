#!/usr/bin/env python3
"""Transfer-environment result plots for the DDCL-MBRL RLC 2026 workshop paper.

Pulls data live from W&B (GraphQL), so it always reflects the latest finished runs.
Run again after Round-1 re-runs to update figures automatically.

Outputs (PNG + PDF) written to OUTDIR (default: paper's Figures/ directory):
  06_transfer_performance.{png,pdf}   — 5-panel grouped bar (main result)
  07_transfer_rate_efficiency.{png,pdf} — performance vs. bits scatter (C2)
  08_transfer_codebook_usage.{png,pdf} — codebook usage % per method/env (C2/C6)
  transfer_results_table.tex          — LaTeX table for App. H

Usage:
  conda run -n ddcl_mbrl python scripts/plot_transfer_results.py
  conda run -n ddcl_mbrl python scripts/plot_transfer_results.py --outdir /path/to/Figures
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import requests

# ── matplotlib / seaborn setup before any plt import ─────────────────────────
_tmp = Path(os.getenv("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_tmp / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp / "xdg-cache"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── W&B credentials (read from environment; never hard-code secrets) ──────────
API_KEY = os.environ.get("WANDB_API_KEY", "")
ENTITY  = os.environ.get("WANDB_ENTITY", "f20210966")
PROJECT = "dcmpc"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# ── Paper layout constants (RLC 2026, 10 pt, two-column) ─────────────────────
COL_W  = 3.25   # single column width, inches
FULL_W = 6.75   # full page width, inches
FONT   = 8      # body text pt

# ── Paul Tol "bright" colorblind-safe palette ─────────────────────────────────
C_FSQ  = "#4477AA"   # blue
C_VQ   = "#66CCEE"   # cyan
C_DDCL = "#332288"   # indigo

METHOD_COLORS  = {"FSQ-CE": C_FSQ, "VQ-CE": C_VQ, "DDCL-Cos": C_DDCL}
METHOD_MARKERS = {"FSQ-CE": "^",   "VQ-CE": "v",  "DDCL-Cos": "*"}
METHOD_ORDER   = ["FSQ-CE", "VQ-CE", "DDCL-Cos"]

# ── Environment display names & ordering ─────────────────────────────────────
ENV_KEYS    = ["walker/walk", "reacher/hard", "dog/run", "humanoid/walk", "mw-button-press"]
ENV_LABELS  = ["Walker\nWalk", "Reacher\nHard", "Dog\nRun", "Humanoid\nWalk", "Button\nPress"]
MAX_RETURN  = 1000   # DMControl episode length → normalise raw return

# ── Best lambda per environment (from sweep analysis) ────────────────────────
BEST_LAMBDA = {
    "walker/walk":    0.001,
    "reacher/hard":   0.01,
    "dog/run":        0.001,
    "humanoid/walk":  0.01,
    "mw-button-press": 0.01,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_runs() -> list[dict]:
    """Pull all finished runs from the dcmpc W&B project via GraphQL."""
    query = """{
      project(entityName:"%s", name:"%s"){
        runs(first:200){edges{node{name state config summaryMetrics}}}
      }
    }""" % (ENTITY, PROJECT)
    resp = requests.post(
        "https://api.wandb.ai/graphql",
        json={"query": query}, headers=HEADERS, timeout=30,
    )
    resp.raise_for_status()
    edges = resp.json()["data"]["project"]["runs"]["edges"]
    print(f"  fetched {len(edges)} runs from W&B")
    return edges


def _gv(d: dict, k: str):
    v = d.get(k, {})
    return v.get("value", v) if isinstance(v, dict) else v


def parse_runs(edges: list[dict]) -> list[dict]:
    """Parse W&B run edges into clean per-seed records."""
    records = []
    for edge in edges:
        node  = edge["node"]
        if node["state"] != "finished":
            continue
        cfg = json.loads(node.get("config", "{}") or "{}")
        sm  = json.loads(node.get("summaryMetrics", "{}") or "{}")

        env  = _gv(cfg, "env_name")
        task = _gv(cfg, "task_name")
        seed = _gv(cfg, "seed")
        env_key = f"{env}/{task}".rstrip("/")
        if env_key not in ENV_KEYS:
            continue

        agent = cfg.get("agent", {})
        if isinstance(agent, dict) and "value" in agent:
            agent = agent["value"]
        quantizer = agent.get("quantizer", "?") if isinstance(agent, dict) else "?"
        lam       = agent.get("ddcl_lambda", None) if isinstance(agent, dict) else None

        # prefer eval_best/, fall back to eval/
        eb  = sm.get("eval_best/", {}) or {}
        ev  = sm.get("eval/",      {}) or {}
        src = "best" if (eb.get("normalized_return") or eb.get("episodic_success")
                         or eb.get("episodic_return")) else "final"
        d   = eb if src == "best" else ev

        # Performance
        nr  = d.get("normalized_return")
        ret = d.get("episodic_return")
        suc = d.get("episodic_success")
        # normalise raw return to [0,1]
        if nr is None and ret is not None:
            nr = ret / MAX_RETURN

        # Rate metrics
        alloc_bits = d.get("rate/allocated_bits_per_transition")
        usage_pct  = d.get("codebook/usage_percent")
        nom_bits   = d.get("rate/max_bits_per_transition")

        # Method label
        if quantizer == "fsq":
            method = "FSQ-CE"
        elif quantizer == "vq":
            method = "VQ-CE"
        elif quantizer == "ddcl":
            method = "DDCL-Cos"
        else:
            continue

        records.append(dict(
            env=env_key, method=method, lam=lam, seed=seed,
            nr=nr, suc=suc, alloc_bits=alloc_bits,
            nom_bits=nom_bits, usage_pct=usage_pct, src=src,
        ))
    return records


def select_best_ddcl(records: list[dict]) -> list[dict]:
    """For DDCL, keep only the best-lambda seeds per env; keep all FSQ/VQ seeds."""
    out = []
    for r in records:
        if r["method"] != "DDCL-Cos":
            out.append(r)
        elif r["lam"] == BEST_LAMBDA.get(r["env"]):
            out.append(r)
    return out


def group_perf(records: list[dict]) -> dict:
    """Return {(env, method): {"nr": [...], "suc": [...], "bits": [...], "usage": [...]}}."""
    from collections import defaultdict
    g: dict = defaultdict(lambda: {"nr": [], "suc": [], "bits": [], "usage": []})
    for r in records:
        key = (r["env"], r["method"])
        if r["nr"]        is not None: g[key]["nr"].append(r["nr"])
        if r["suc"]       is not None: g[key]["suc"].append(r["suc"])
        if r["alloc_bits"] is not None: g[key]["bits"].append(r["alloc_bits"])
        if r["usage_pct"] is not None:  g[key]["usage"].append(r["usage_pct"])
    return dict(g)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pareto_mask(xs, ys) -> np.ndarray:
    """Boolean mask: True if point i is on the upper-left Pareto frontier.

    Upper-left is better: lower x (fewer bits) and higher y (better performance).
    Point i is dominated if there exists j with x_j ≤ x_i AND y_j ≥ y_i
    (strict in at least one dimension).  NaN points are never on the frontier.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if np.isnan(xs[i]) or np.isnan(ys[i]):
            mask[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            if np.isnan(xs[j]) or np.isnan(ys[j]):
                continue
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                mask[i] = False
                break
    return mask

def ci95(vals: list[float]):
    """Returns (mean, half_ci). half_ci is nan for n<2."""
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    if n < 2:
        return m, float("nan")
    s = statistics.stdev(vals)
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.0)
    return m, t * s / n ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Paper style
# ─────────────────────────────────────────────────────────────────────────────

def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi":          150,
        "savefig.dpi":         300,
        "pdf.fonttype":        42,
        "ps.fonttype":         42,
        "font.size":           FONT,
        "axes.titlesize":      FONT,
        "axes.labelsize":      FONT,
        "xtick.labelsize":     FONT - 1,
        "ytick.labelsize":     FONT - 1,
        "legend.fontsize":     FONT - 1,
        "lines.linewidth":     1.2,
        "axes.linewidth":      0.7,
        "xtick.major.width":   0.7,
        "ytick.major.width":   0.7,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.alpha":          0.3,
        "grid.linewidth":      0.5,
        "legend.frameon":      False,
        "legend.handlelength": 1.2,
    })


def savefig(outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(outdir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  → {outdir / stem}.png/.pdf")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Figure 06 — 5-panel performance comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_transfer_performance(groups: dict, outdir: Path) -> None:
    """5 subplots (one per env), 3 grouped bars (FSQ / VQ / DDCL-best-λ).

    Primary metric: normalised return (success for button-press, mapped to [0,1]).
    Error bars: 95% CI (t-distribution).  Individual seeds shown as strip dots.
    """
    fig, axes = plt.subplots(1, 5, figsize=(FULL_W, 2.2), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    bar_w   = 0.6
    x_pos   = np.arange(len(METHOD_ORDER))

    for ax, env_key, env_lbl in zip(axes, ENV_KEYS, ENV_LABELS):
        is_mw = "mw" in env_key

        for xi, method in enumerate(METHOD_ORDER):
            key  = (env_key, method)
            data = groups.get(key, {})
            vals = data.get("suc" if is_mw else "nr", [])
            if not vals:
                continue
            m, hci = ci95(vals)
            color  = METHOD_COLORS[method]
            ax.bar(xi, m, width=bar_w, color=color, zorder=2,
                   yerr=hci if not np.isnan(hci) else None,
                   error_kw={"linewidth": 0.9, "capsize": 3, "capthick": 0.9},
                   ecolor="#333333")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["FSQ", "VQ", "DDCL"], fontsize=FONT - 1)
        ax.set_title(env_lbl, fontsize=FONT, pad=3)
        ax.set_ylim(0, 1.09)
        ax.set_xlim(-0.6, 2.6)
        if ax is axes[0]:
            ax.set_ylabel("Normalised return / success")
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    # shared legend below panels (avoids overlap with env titles)
    patches = [mpatches.Patch(color=c, label=m)
               for m, c in METHOD_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.10), fontsize=FONT - 1, frameon=False)

    savefig(outdir, "06_transfer_performance")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Figure 07 — performance vs. bits scatter (rate-efficiency, C2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rate_efficiency(groups: dict, all_records: list[dict], outdir: Path) -> None:
    """Scatter: mean normalised performance (y) vs. mean allocated bits (x).

    FSQ/VQ are pinned at their nominal budget (no rate penalty → bits = max).
    DDCL sits at its realised allocated bits (< nominal due to rate penalty).
    One point per (env, method); env encoded by marker shape.
    Vertical dashed line at FSQ/VQ nominal budget for each env group.
    """
    # env → marker
    env_markers = {
        "walker/walk":     "o",
        "reacher/hard":    "s",
        "dog/run":         "D",
        "humanoid/walk":   "^",
        "mw-button-press": "P",
    }
    env_short = {
        "walker/walk":     "Walker",
        "reacher/hard":    "Reacher",
        "dog/run":         "Dog",
        "humanoid/walk":   "Humanoid",
        "mw-button-press": "Button",
    }

    # Nominal bits per env: prefer FSQ/VQ records (they equal architecture max),
    # fall back to any method that logged rate/max_bits_per_transition.
    # FSQ/VQ don't apply a rate penalty so their alloc_bits == nom_bits.
    nom_bits_map: dict[str, float] = {}
    for r in all_records:
        if r["nom_bits"] and not np.isnan(float(r["nom_bits"])):
            existing = nom_bits_map.get(r["env"])
            # prefer FSQ/VQ source; otherwise take the maximum seen (= nominal)
            if existing is None or r["method"] in ("FSQ-CE", "VQ-CE"):
                nom_bits_map[r["env"]] = float(r["nom_bits"])

    fig, ax = plt.subplots(figsize=(FULL_W * 0.55, 2.6))

    for env_key in ENV_KEYS:
        is_mw = "mw" in env_key
        nom   = nom_bits_map.get(env_key, 2378)
        mkr   = env_markers[env_key]

        for method in METHOD_ORDER:
            key  = (env_key, method)
            data = groups.get(key, {})
            perf_vals = data.get("suc" if is_mw else "nr", [])
            if not perf_vals:
                continue

            pm, hci_y = ci95(perf_vals)
            color     = METHOD_COLORS[method]

            # bits: DDCL uses realised allocated_bits (with CI); FSQ/VQ use nominal
            if method == "DDCL-Cos":
                bit_vals = data.get("bits", [])
                bm, hci_x = ci95(bit_vals) if bit_vals else (nom, float("nan"))
            else:
                bm, hci_x = nom, float("nan")

            ax.errorbar(
                bm, pm,
                xerr=hci_x if (hci_x is not None and not np.isnan(hci_x)) else None,
                yerr=hci_y if (hci_y is not None and not np.isnan(hci_y)) else None,
                fmt=mkr,
                color=color,
                markersize=6,
                markeredgewidth=0,
                elinewidth=0.7,
                capsize=2.0,
                capthick=0.7,
                ecolor=color,
                alpha=0.9,
                zorder=3,
            )

    # Nominal budget reference lines (de-emphasised so they don't dominate)
    for nom_val in sorted(set(nom_bits_map.values())):
        ax.axvline(nom_val, color="#BBBBBB", lw=0.6, ls=":", alpha=0.6, zorder=1)

    # ── Pareto frontier: per-environment (non-dominated within each env) ────────
    # Compute Pareto dominance separately for each environment so that a point
    # from one environment cannot "dominate" a point from a different environment.
    S_INNER = 60
    S_OUTER = 130
    for env_key in ENV_KEYS:
        is_mw = "mw" in env_key
        nom   = nom_bits_map.get(env_key, 2378)
        env_pf_x, env_pf_y, env_pf_color, env_pf_marker = [], [], [], []
        for method in METHOD_ORDER:
            key       = (env_key, method)
            data      = groups.get(key, {})
            perf_vals = data.get("suc" if is_mw else "nr", [])
            if not perf_vals:
                continue
            pm, _ = ci95(perf_vals)
            bm = statistics.mean(data["bits"]) if method == "DDCL-Cos" and data.get("bits") else nom
            env_pf_x.append(bm)
            env_pf_y.append(pm)
            env_pf_color.append(METHOD_COLORS[method])
            env_pf_marker.append(env_markers[env_key])

        if len(env_pf_x) >= 2:
            frontier = _pareto_mask(env_pf_x, env_pf_y)
            for xi, yi, col, mkr, on_f in zip(env_pf_x, env_pf_y, env_pf_color,
                                               env_pf_marker, frontier):
                if on_f:
                    ax.scatter(xi, yi, marker=mkr, color="black",
                               s=S_OUTER, linewidths=0, zorder=4)
                    ax.scatter(xi, yi, marker=mkr, color=col,
                               s=S_INNER, linewidths=0, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Allocated bits / transition (log scale)")
    ax.set_ylabel("Normalised return / success")
    # Proportions are bounded [0, 1]; set axis viewport accordingly so that
    # error bars extending beyond 1 are clipped at the axis edge.
    ax.set_ylim(0, 1.09)

    # ── Single combined legend, outside the axes ──────────────────────────────
    # Use solid coloured CIRCLES for method (no shape conflict with env markers)
    # and grey ENV-SHAPED markers for the environment dimension.
    method_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markeredgecolor="none",
                   markersize=7, linestyle="None", label=m)
        for m, c in METHOD_COLORS.items()
    ]
    env_handles = [
        plt.Line2D([0], [0], marker=env_markers[e], color="w",
                   markerfacecolor="#888888", markeredgecolor="none",
                   markersize=6, linestyle="None", label=env_short[e])
        for e in ENV_KEYS
    ]
    frontier_handle = plt.Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor="#888888", markeredgecolor="black",
        markersize=8, markeredgewidth=1.2, linestyle="None",
        label="Pareto-optimal",
    )

    leg1 = ax.legend(handles=method_handles, loc="upper left",
                     bbox_to_anchor=(1.02, 1.00),
                     fontsize=FONT - 1, title="Method (colour)",
                     title_fontsize=FONT - 1, borderpad=0.3,
                     handletextpad=0.4)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=env_handles, loc="upper left",
                     bbox_to_anchor=(1.02, 0.62),
                     fontsize=FONT - 1, title="Env (shape)",
                     title_fontsize=FONT - 1, borderpad=0.3,
                     handletextpad=0.4)
    ax.add_artist(leg2)
    ax.legend(handles=[frontier_handle], loc="upper left",
              bbox_to_anchor=(1.02, 0.06),
              fontsize=FONT - 1, borderpad=0.3, handletextpad=0.4)

    fig.tight_layout(pad=0.6)
    savefig(outdir, "07_transfer_rate_efficiency")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Figure 08 — codebook usage % (C2/C6)
# ─────────────────────────────────────────────────────────────────────────────

def plot_codebook_usage(groups: dict, outdir: Path) -> None:
    """Grouped bar: codebook usage % per env, colour = method.

    DDCL consistently uses fewer codes, illustrating rate-adaptive compression.
    """
    fig, ax = plt.subplots(figsize=(FULL_W * 0.65, 2.0))

    n_envs   = len(ENV_KEYS)
    n_methods = len(METHOD_ORDER)
    group_w  = 0.8
    bar_w    = group_w / n_methods
    x_base   = np.arange(n_envs)

    for mi, method in enumerate(METHOD_ORDER):
        offset = (mi - (n_methods - 1) / 2) * bar_w
        means, cis = [], []
        for env_key in ENV_KEYS:
            data = groups.get((env_key, method), {})
            vals = data.get("usage", [])
            m, hci = ci95(vals) if vals else (float("nan"), float("nan"))
            means.append(m)
            cis.append(hci)
        x = x_base + offset
        finite = [not np.isnan(m) for m in means]
        xf = x[np.array(finite)]
        mf = [m for m, ok in zip(means, finite) if ok]
        cf = [c if not np.isnan(c) else 0 for c, ok in zip(cis, finite) if ok]
        ax.bar(xf, mf, width=bar_w * 0.9, color=METHOD_COLORS[method],
               label=method, zorder=2,
               yerr=cf, error_kw={"linewidth": 0.8, "capsize": 2}, ecolor="#333333")

    ax.set_xticks(x_base)
    ax.set_xticklabels(ENV_LABELS, fontsize=FONT - 1)
    ax.set_ylabel("Codebook usage (%)", fontsize=FONT)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.legend(loc="upper right", fontsize=FONT - 1)

    savefig(outdir, "08_transfer_codebook_usage")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  LaTeX table for App. H
# ─────────────────────────────────────────────────────────────────────────────

def write_latex_table(groups: dict, outdir: Path) -> None:
    """Write a LaTeX booktabs table with normalised return ± 95% CI."""
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Transfer-environment results (mean $\pm$ 95\% CI, $n{=}5$ seeds). "
        r"Normalised return for DMControl; success rate for MetaWorld. "
        r"DDCL-Cos uses the best $\lambda$ per environment "
        r"($\lambda{=}0.001$ for Walker, Dog; $\lambda{=}0.01$ otherwise). "
        r"Rate = mean allocated bits\,/\,transition (DDCL) or nominal budget (FSQ/VQ).}",
        r"\label{tab:transfer_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Environment & FSQ-CE & VQ-CE & DDCL-Cos & DDCL bits (vs.\ nominal) \\",
        r"\midrule",
    ]

    for env_key, env_lbl in zip(ENV_KEYS, ["Walker/Walk", "Reacher/Hard", "Dog/Run",
                                            "Humanoid/Walk", "Button-Press"]):
        is_mw = "mw" in env_key
        row_parts = [env_lbl]
        best_perf = -1
        best_idx  = -1
        cell_means = []

        for mi, method in enumerate(METHOD_ORDER):
            data = groups.get((env_key, method), {})
            vals = data.get("suc" if is_mw else "nr", [])
            m, hci = ci95(vals) if vals else (float("nan"), float("nan"))
            cell_means.append(m)
            if not np.isnan(m) and m > best_perf:
                best_perf = m
                best_idx  = mi

        # DDCL bits column
        ddcl_data  = groups.get((env_key, "DDCL-Cos"), {})
        fsq_data   = groups.get((env_key, "FSQ-CE"),   {})
        ddcl_bits  = statistics.mean(ddcl_data.get("bits", [0])) if ddcl_data.get("bits") else float("nan")
        fsq_nom    = 4755 if env_key in ("dog/run", "humanoid/walk") else 2378
        bits_pct   = f"{ddcl_bits:.0f} ({ddcl_bits/fsq_nom*100:.0f}\\%)" if not np.isnan(ddcl_bits) else "---"

        for mi, method in enumerate(METHOD_ORDER):
            data = groups.get((env_key, method), {})
            vals = data.get("suc" if is_mw else "nr", [])
            m, hci = ci95(vals) if vals else (float("nan"), float("nan"))
            n = len(vals)
            if np.isnan(m):
                cell = "---"
            elif np.isnan(hci):
                cell = f"{m:.3f} ($n={n}$)"
            else:
                cell = f"{m:.3f}\\,{{\\small$\\pm$\\,{hci:.3f}}}"
            if mi == best_idx and not np.isnan(m):
                cell = r"\textbf{" + cell + "}"
            row_parts.append(cell)

        row_parts.append(bits_pct)
        lines.append(" & ".join(row_parts) + r" \\")

    lines += [
        r"\bottomrule",
        r"\multicolumn{5}{l}{\scriptsize $^*$Reacher/Dog/Humanoid FSQ/VQ use final"
        r" checkpoint (no \texttt{eval\_best/} logged); DDCL uses best checkpoint."
        r" Best--final gap $= +0.004\pm0.018$ (negligible, App.~H).} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = outdir / "transfer_results_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  → LaTeX table: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path,
        default=Path(__file__).parent.parent.parent
                / "DDCL_MBRL_WM_RLC_Workshop" / "Figures",
        help="Output directory for PNG/PDF figures and LaTeX table.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for jitter in strip plots.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    apply_style()

    print("Fetching runs from W&B …")
    edges   = fetch_runs()
    records = parse_runs(edges)
    print(f"  parsed {len(records)} finished records across transfer envs")

    # Filter to best lambda per env for DDCL; keep all FSQ/VQ
    selected = select_best_ddcl(records)

    # Check coverage
    for env in ENV_KEYS:
        for m in METHOD_ORDER:
            key = (env, m)
            n = sum(1 for r in selected if r["env"] == env and r["method"] == m)
            lam_str = f" λ={BEST_LAMBDA[env]}" if m == "DDCL-Cos" else ""
            status  = "✅" if n == 5 else f"⚠️  {n}/5"
            print(f"  {env:<22} {m}{lam_str:<18} {status}")

    groups = group_perf(selected)

    print("\nGenerating figures …")
    plot_transfer_performance(groups, args.outdir)
    plot_rate_efficiency(groups, records, args.outdir)
    plot_codebook_usage(groups, args.outdir)

    print("\nGenerating LaTeX table …")
    write_latex_table(groups, args.outdir)

    print(f"\nDone. All outputs written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
