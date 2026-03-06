"""
Unified hyperparameter config for chemotherapy offline RL
"""

ODE_CONFIG = {
    "dt": 0.3,
    "n_sub": 5,
    "max_steps": 300,
    "T_clear": 0.02,
    "C_tox": 8.0,
    "N_min": 0.1,
    "I_min": 0.05,
}

REWARD_CONFIG = {
    "w_tumor": 3.0,
    "w_progress": 2.0,
    "w_normal": 0.5,
    "w_immune": 2.0,
    "w_toxicity": 0.5,
    "R_clear": 50.0,
    "C_ref": 2.0,
    "collapse_penalty": -20.0,
    "milestone_5": 3.0,
    "milestone_10": 5.0,
}

DATA_CONFIG = {
    "num_trajectories": 500,
    "horizon": 300,
    "seed": 42,
    "expert_ratio": 0.60,
    "balanced_ratio": 0.20,
    "aggressive_ratio": 0.10,
    "conservative_ratio": 0.10,
}

BC_CONFIG = {
    "hidden_dim": 128,
    "lr": 3e-4,
    "batch_size": 128,
    "epochs": 300,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "patience": 30,
}

CQL_CONFIG = {
    "hidden_dim": 128,
    "lr": 3e-4,
    "gamma": 0.99,
    "alpha": 2.0,
    "batch_size": 256,
    "target_update": 200,
    "tau": 0.005,
    "total_steps": 100_000,
}
