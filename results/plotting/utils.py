import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 400
plt.rcParams["font.size"] = 28
plt.rcParams["legend.fontsize"] = 29
plt.rcParams["legend.loc"] = "lower right"
plt.rcParams["text.usetex"] = True

AGENTS = {
    "DCWM": "DC-MPC (ours)",
    "TD-MPC2": "TD-MPC2",
    "DreamerV3": "DreamerV3",
    "SAC": "SAC",
    "TD-MPC": "TD-MPC",
}
PALETTE = {
    "DCWM (ours)": "#984ea3",
    "DC-MPC (ours)": "#984ea3",
    "DCWM-ref (ours)": "black",
    "TD-MPC2": "#377eb8",
    "DreamerV3": "#e41a1c",
    "SAC": "#ff7f00",
    "TD-MPC": "green",
}
YLABELS = {
    "episode_reward": "Episode Return",
    "episode_success": "Success Rate ($\%$)",
    "active_percent": "Codebook Active Percent (\%)",
    "rank1": "Matrix Rank",
    "rank_percent_1": "Matrix Rank (\% of full)",
    "episodic_return": "Episode Return",
    "episodic_success": "Success Rate",
    "rate/allocated_bits_per_transition": "Allocated Bits / Transition",
    "rate/max_bits_per_transition": "Max Bits / Transition",
    "rate/compression_ratio_vs_max": "Allocated / Max Bits",
    "rate/empirical_entropy_bits_per_transition": "Empirical Entropy Bits / Transition",
    "codebook/per_group_entropy_mean": "Code Entropy (normalized)",
    "codebook/usage_percent": "Active Codes (\%)",
    "comms_bits": "DDCL Bits / Latent Dim",
    "control/state_norm_mean": "State Norm",
    "control/state_delta_norm_mean": "State Delta Norm",
    "control/action_norm_mean": "Action Norm",
    "control/action_abs_mean": "Mean Absolute Action",
    "control/action_saturation_frac": "Action Saturation Fraction",
    "control/reward_step_mean": "Mean Step Reward",
}
