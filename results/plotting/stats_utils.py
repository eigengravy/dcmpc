"""Central statistics utilities for DDCL-MBRL paper reporting.

All paper-facing metrics use mean ± 95% CI (t-distribution, sample std ddof=1).
Import this module everywhere metrics are aggregated or plotted.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def compute_stats(values: list[float | None]) -> dict:
    """Compute mean, sample std, and 95% CI for a list of per-seed values.

    Args:
        values: Per-seed scalar values. None entries are silently dropped.

    Returns:
        dict with keys: mean, std (ddof=1), ci95, n, values (cleaned list).
        If n <= 1, ci95 is nan.
    """
    cleaned = [v for v in values if v is not None and not np.isnan(float(v))]
    arr = np.array(cleaned, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"),
                "n": 0, "values": []}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        ci95 = float(stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n))
    else:
        ci95 = float("nan")
    return {"mean": mean, "std": std, "ci95": ci95, "n": n, "values": cleaned}


def format_mean_ci(stats_dict: dict, decimals: int = 3) -> str:
    """Format a stats dict as 'mean ± ci95' string for reporting."""
    mean = stats_dict.get("mean", float("nan"))
    ci95 = stats_dict.get("ci95", float("nan"))
    if np.isnan(mean):
        return "N/A"
    if np.isnan(ci95):
        fmt = f"{{:.{decimals}f}}"
        return fmt.format(mean)
    fmt = f"{{:.{decimals}f}} ± {{:.{decimals}f}}"
    return fmt.format(mean, ci95)


# t-distribution critical values for quick reference
# t_{n-1, 0.975} for 95% CI
T_CRIT = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,   # n=5 seeds
    5: 2.571,
    9: 2.262,   # n=10 seeds
    19: 2.093,  # n=20 seeds
    29: 2.045,  # n=30 seeds
}
